# My own inference code.

import os
import torch
from typing import NamedTuple

from datasets.utils import VoxelizeCollate
from omegaconf import DictConfig
from pathlib import Path
import hydra
from plyfile import PlyData
import numpy as np
import open3d as o3d
import yaml
from scipy.spatial import KDTree
from datasets.scannet200.scannet200_constants import (
    CLASS_LABELS_200,
    VALID_CLASS_IDS_200,
)
from trainer.trainer import InstanceSegmentation, get_evenly_distributed_colors
from utils.point_cloud_utils import write_point_cloud_in_ply
from utils.utils import (
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)

DEFAULT_COLOR_MEAN_STD = (
    (0.47793125906962, 0.4303257521323044, 0.3749598901421883),
    (0.2834475483823543, 0.27566157565723015, 0.27018971370874995),
)

SCANNET200_LABELS_BY_ID = dict(zip(VALID_CLASS_IDS_200, CLASS_LABELS_200))


def remove_isolated_points(coords: np.ndarray, radius: float = 0.05, min_neighbors: int = 5) -> np.ndarray:
    """Return boolean mask keeping points that have >= min_neighbors within radius."""
    if len(coords) < min_neighbors + 1:
        return np.ones(len(coords), dtype=bool)
    tree = KDTree(coords)
    neighbor_counts = tree.query_ball_point(coords, r=radius, return_length=True)
    return neighbor_counts >= (min_neighbors + 1)  # +1 because point counts itself


class InferenceDatasetAdapter:
    def __init__(self, dataset_name: str, label_offset: int = 0, remap_fn=None):
        self.dataset_name = dataset_name
        self.label_offset = label_offset
        self.data = []
        self._remap_fn = remap_fn

    def _remap_model_output(self, output):
        if self._remap_fn is not None:
            return np.asarray(self._remap_fn(output))
        return np.asarray(output)


def build_inference_dataset_adapter(cfg: DictConfig) -> InferenceDatasetAdapter:
    dataset_cfg = cfg.data.test_dataset
    label_db_path = Path(dataset_cfg.label_db_filepath)

    if label_db_path.exists():
        with label_db_path.open() as file:
            labels = yaml.safe_load(file)

        number_of_validation_labels = sum(
            1 for value in labels.values() if value["validation"]
        )
        number_of_all_labels = len(labels)
        num_labels = int(cfg.data.num_labels)

        if num_labels == number_of_all_labels:
            label_info = labels
        elif num_labels == number_of_validation_labels:
            label_info = {
                key: value for key, value in labels.items() if value["validation"]
            }
        else:
            raise ValueError(
                "not available number labels, select from: "
                f"{number_of_validation_labels}, {number_of_all_labels}"
            )

        label_keys = np.asarray(list(label_info.keys()), dtype=np.int32)
    elif dataset_cfg.dataset_name == "scannet200":
        # Match the trainer remap convention without requiring local preprocessing artifacts.
        label_keys = np.asarray(VALID_CLASS_IDS_200, dtype=np.int32)
    else:
        raise FileNotFoundError(
            f"Label database not found: {label_db_path}"
        )

    def remap_model_output(output):
        output = np.asarray(output)
        remapped_output = output.copy()
        for index, key in enumerate(label_keys):
            remapped_output[output == index] = key
        return remapped_output

    return InferenceDatasetAdapter(
        dataset_name=dataset_cfg.dataset_name,
        label_offset=int(dataset_cfg.label_offset),
        remap_fn=remap_model_output,
    )


class PredictionOutput(NamedTuple):
    pred_masks: np.ndarray
    pred_scores: np.ndarray
    pred_classes: np.ndarray


class OutputPaths(NamedTuple):
    ply: Path
    segment_semantic: Path
    instance: Path


def build_output_paths(scene_path: Path) -> OutputPaths:
    base_name = scene_path.stem
    return OutputPaths(
        ply=Path(f"{base_name}_pred_instances.ply"),
        segment_semantic=Path(f"{base_name}_segment_semantic.npy"),
        instance=Path(f"{base_name}_instance.npy"),
    )


def save_prediction_outputs(
    scene,
    predictions: PredictionOutput,
    threshold: float,
    output_paths: OutputPaths,
):
    coords, raw_colors, _, _ = scene
    segment_ids = np.arange(coords.shape[0], dtype=np.int32)

    assigned_scores = np.full(coords.shape[0], -np.inf, dtype=np.float32)
    semantic_labels = np.full(coords.shape[0], 255, dtype=np.uint16)
    instance_ids = np.full(coords.shape[0], -1, dtype=np.int32)

    valid_instances = np.where(predictions.pred_scores >= threshold)[0]
    for instance_index in valid_instances:
        mask = predictions.pred_masks[:, instance_index].astype(bool)
        update = mask & (predictions.pred_scores[instance_index] > assigned_scores)
        assigned_scores[update] = predictions.pred_scores[instance_index]
        semantic_labels[update] = np.uint16(
            max(predictions.pred_classes[instance_index], 0)
        )
        instance_ids[update] = int(instance_index)

    output_colors = raw_colors.copy()
    if len(valid_instances) > 0:
        palette = np.asarray(
            get_evenly_distributed_colors(len(valid_instances) + 1), dtype=np.uint8
        )
        for palette_index, instance_index in enumerate(valid_instances, start=1):
            output_colors[instance_ids == instance_index] = palette[palette_index]

    write_point_cloud_in_ply(
        output_paths.ply,
        coords.astype(np.float32),
        feats=output_colors.astype(np.uint8),
        labels=semantic_labels,
    )

    seg_sem_array = np.column_stack(
        (segment_ids.astype(np.int32), semantic_labels.astype(np.int32))
    )
    np.save(output_paths.segment_semantic, seg_sem_array)
    np.save(output_paths.instance, instance_ids.astype(np.int32))

    print(f"Saved {output_paths.ply}")
    print(f"Saved {output_paths.segment_semantic}")
    print(f"Saved {output_paths.instance}")


def save_predictions_compressed(predictions, output_path: Path):
    packed = np.packbits(predictions.pred_masks.astype(bool), axis=0)
    np.savez_compressed(
        output_path,
        pred_masks_packed=packed,
        pred_masks_shape=np.array(predictions.pred_masks.shape),
        pred_scores=predictions.pred_scores,
        pred_classes=predictions.pred_classes,
    )
    print(f"Saved {output_path}.npz")


def preprocess_scene(scene_path: Path):
    def load_scene(scene_path):
        with scene_path.open("rb") as f:
            ply = PlyData.read(f)
        data = ply["vertex"].data
        field_names = set(data.dtype.names)
        coords = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(
            np.float32
        )
        colors = np.full((coords.shape[0], 3), 255, dtype=np.uint8)
        normals = None
        if {"red", "green", "blue"}.issubset(field_names):
            colors = np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(
                np.uint8
            )
        if {"nx", "ny", "nz"}.issubset(field_names):
            normals = np.stack([data["nx"], data["ny"], data["nz"]], axis=1).astype(
                np.float32
            )
        return coords, colors, normals

    def normalize_colors(colors):
        mean, std = DEFAULT_COLOR_MEAN_STD
        colors = (colors / 255.0 - mean) / std
        return colors

    def estimate_normals(coords, knn):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(coords.astype(np.float64))
        search_param = o3d.geometry.KDTreeSearchParamKNN(knn=knn)
        point_cloud.estimate_normals(search_param=search_param)
        return np.asarray(point_cloud.normals, dtype=np.float32)

    coords, colors, normals = load_scene(Path(scene_path))
    colors_normalized = normalize_colors(colors)
    if normals is None:
        normals = estimate_normals(coords, knn=30)
    return coords, colors, colors_normalized, normals


class Mask3D:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        model = InstanceSegmentation(self.cfg)
        self.cfg, self.model = load_checkpoint_with_missing_or_exsessive_keys(self.cfg, model)

    @staticmethod
    def _build_dataset_adapter(cfg: DictConfig) -> InferenceDatasetAdapter:
        dataset_cfg = cfg.data.test_dataset
        label_db_path = Path(dataset_cfg.label_db_filepath)

        if label_db_path.exists():
            with label_db_path.open() as file:
                labels = yaml.safe_load(file)

            number_of_validation_labels = sum(
                1 for value in labels.values() if value["validation"]
            )
            number_of_all_labels = len(labels)
            num_labels = int(cfg.data.num_labels)

            if num_labels == number_of_all_labels:
                label_info = labels
            elif num_labels == number_of_validation_labels:
                label_info = {
                    key: value for key, value in labels.items() if value["validation"]
                }
            else:
                raise ValueError(
                    "not available number labels, select from: "
                    f"{number_of_validation_labels}, {number_of_all_labels}"
                )

            label_keys = np.asarray(list(label_info.keys()), dtype=np.int32)
        elif dataset_cfg.dataset_name == "scannet200":
            # Match the trainer remap convention without requiring local preprocessing artifacts.
            label_keys = np.asarray(VALID_CLASS_IDS_200, dtype=np.int32)
        else:
            raise FileNotFoundError(f"Label database not found: {label_db_path}")

        def remap_model_output(output):
            output = np.asarray(output)
            remapped_output = output.copy()
            for index, key in enumerate(label_keys):
                remapped_output[output == index] = key
            return remapped_output

        return InferenceDatasetAdapter(
            dataset_name=dataset_cfg.dataset_name,
            label_offset=int(dataset_cfg.label_offset),
            remap_fn=remap_model_output,
        )

    def run_single_scene_inference(self, scene) -> PredictionOutput:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        adapter = self._build_dataset_adapter(self.cfg)
        self.model.validation_dataset = adapter
        self.model.test_dataset = adapter

        coords, raw_colors, colors_norm, normals = scene

        feats = colors_norm.astype(np.float32)
        if self.cfg.data.add_normals:
            feats = np.hstack((feats, normals.astype(np.float32)))
        if self.cfg.data.add_raw_coordinates:
            feats = np.hstack((feats, coords.astype(np.float32)))

        # Fallback segmentation: each point is its own segment.
        segment_ids = np.arange(coords.shape[0], dtype=np.int32)[:, None]

        sample = (
            coords.astype(np.float32),
            feats.astype(np.float32),
            segment_ids.astype(np.int32),
            "inference_scene",
            raw_colors.astype(np.uint8),
            normals.astype(np.float32),
            coords.astype(np.float32),
            0,
        )

        collate = VoxelizeCollate(
            ignore_label=self.cfg.data.test_collation.ignore_label,
            voxel_size=self.cfg.data.test_collation.voxel_size,
            mode=self.cfg.data.test_collation.mode,
            probing=self.cfg.data.test_collation.probing,
            task=self.cfg.data.test_collation.task,
            ignore_class_threshold=self.cfg.data.test_collation.ignore_class_threshold,
            filter_out_classes=list(self.cfg.data.test_collation.filter_out_classes),
            label_offset=0,
            num_queries=self.cfg.data.test_collation.num_queries,
        )

        batch = collate([sample])
        data, target, file_names = batch
        for item in target:
            if "point2segment" in item:
                item["point2segment"] = item["point2segment"].to(device)
        batch = (data, target, file_names)

        with torch.no_grad():
            self.model.eval_step(batch, 0)

        preds = self.model.preds[file_names[0]]
        return PredictionOutput(
            pred_masks=np.asarray(preds["pred_masks"]),
            pred_scores=np.asarray(preds["pred_scores"]),
            pred_classes=np.asarray(preds["pred_classes"]),
        )


def visualize_instance_masks(
    scene,
    predictions: PredictionOutput,
    threshold: float,
    output_dir: Path,
    outlier_radius: float = 0.05,
    outlier_min_neighbors: int = 5,
):
    try:
        import pyviz3d as viz
    except ImportError:
        print("pyviz3d not installed; skipping instance mask visualization")
        return

    coords, _, _, normals = scene
    valid_instances = np.where(predictions.pred_scores >= threshold)[0]
    if len(valid_instances) == 0:
        print("No instances above threshold; skipping instance mask visualization")
        return

    palette = np.asarray(
        get_evenly_distributed_colors(len(valid_instances) + 1), dtype=np.uint8
    )
    viewer = viz.Visualizer()
    label_texts = []
    label_positions = []
    label_colors = []

    for palette_index, instance_index in enumerate(valid_instances, start=1):
        mask = predictions.pred_masks[:, instance_index].astype(bool)
        if not np.any(mask):
            continue

        instance_coords = coords[mask].astype(np.float32)
        instance_normals = normals[mask].astype(np.float32)
        keep = remove_isolated_points(
            instance_coords,
            radius=outlier_radius,
            min_neighbors=outlier_min_neighbors,
        )
        instance_coords = instance_coords[keep]
        instance_normals = instance_normals[keep]
        color = palette[palette_index]
        instance_colors = np.tile(color, (instance_coords.shape[0], 1)).astype(
            np.uint8
        )

        viewer.add_points(
            name=f"{instance_index}_{predictions.pred_scores[instance_index]:.3f}",
            positions=instance_coords,
            colors=instance_colors,
            normals=instance_normals,
            visible=False,
        )

        class_id = int(predictions.pred_classes[instance_index])
        class_label = SCANNET200_LABELS_BY_ID.get(class_id, str(class_id))

        label_texts.append(class_label)
        label_positions.append(instance_coords.mean(axis=0).astype(np.float32))
        label_colors.append(color)

    if label_texts:
        viewer.add_labels(
            name="semantic_labels",
            labels=label_texts,
            positions=np.asarray(label_positions, dtype=np.float32),
            colors=np.asarray(label_colors, dtype=np.uint8),
            visible=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    viewer.save(str(output_dir))
    print(f"Saved {output_dir}")


@hydra.main(config_path="conf", config_name="config_scannet200_demo.yaml")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    mask3d = Mask3D(cfg)
    input_scene_path = Path('office.ply')
    scene = preprocess_scene(input_scene_path)
    predictions = mask3d.run_single_scene_inference(scene)

    # predictions.pred_masks: (num_points, num_instances)
    # predictions.pred_scores: (num_instances,)
    # predictions.pred_classes: (num_instances,)

    save_predictions_compressed(predictions, Path(f"{input_scene_path.stem}_predictions"))
    
    # output_paths = build_output_paths(input_scene_path)
    # save_prediction_outputs(
    #     scene=scene,
    #     predictions=predictions,
    #     threshold=cfg.general.export_threshold,
    #     output_paths=output_paths,
    # )

    # visualize_instance_masks(
    #     scene=scene,
    #     predictions=predictions,
    #     threshold=cfg.general.export_threshold,
    #     output_dir=Path(f"{input_scene_path.stem}_instance_masks_viz"),
    #     outlier_radius=cfg.general.outlier_removal_radius,
    #     outlier_min_neighbors=cfg.general.outlier_removal_min_neighbors,
    # )


if __name__ == "__main__":
    main()
