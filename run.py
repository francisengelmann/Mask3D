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


class InferenceDatasetAdapter:
    def __init__(self, dataset_name: str, label_offset: int = 0):
        self.dataset_name = dataset_name
        self.label_offset = label_offset
        self.data = []

    def _remap_model_output(self, output):
        return np.asarray(output)


class PredictionOutput(NamedTuple):
    pred_masks: np.ndarray
    pred_scores: np.ndarray
    pred_classes: np.ndarray


class OutputPaths(NamedTuple):
    ply: Path
    segment_semantic: Path
    instance: Path


def get_parameters(cfg: DictConfig):
    # logger = logging.getLogger(__name__)
    # load_dotenv(".env")

    # # parsing input parameters
    # seed_everything(cfg.general.seed)

    # # getting basic configuration
    # if cfg.general.get("gpus", None) is None:
    #     cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    # loggers = []

    # if not os.path.exists(cfg.general.save_dir):
    #     os.makedirs(cfg.general.save_dir)
    # else:
    #     print("EXPERIMENT ALREADY EXIST")
    #     cfg["trainer"][
    #         "resume_from_checkpoint"
    #     ] = f"{cfg.general.save_dir}/last-epoch.ckpt"

    # for log in cfg.logging:
    #     print(log)
    #     loggers.append(hydra.utils.instantiate(log))
    #     loggers[-1].log_hyperparams(
    #         flatten_dict(OmegaConf.to_container(cfg, resolve=True))
    #     )

    model = InstanceSegmentation(cfg)
    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(
            cfg, model
        )
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    return cfg, model


def visualize_scene(scene):
    try:
        import pyviz3d as viz
    except ImportError:
        print("pyviz3d not installed; skipping scene visualization")
        return
    coords, colors, colors_normalized, normals = scene
    v = viz.Visualizer()
    v.add_points(name="rgb", positions=coords, colors=colors, normals=normals)
    v.add_points(name="solid", positions=coords, normals=normals)
    v.save("scene")


def visualize_instance_masks(scene, predictions: PredictionOutput, threshold: float, output_dir: Path):
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

    for palette_index, instance_index in enumerate(valid_instances, start=1):
        mask = predictions.pred_masks[:, instance_index].astype(bool)
        if not np.any(mask):
            continue

        instance_coords = coords[mask].astype(np.float32)
        instance_normals = normals[mask].astype(np.float32)
        color = palette[palette_index]
        instance_colors = np.tile(color, (instance_coords.shape[0], 1)).astype(np.uint8)

        viewer.add_points(
            name=f"{instance_index}_{predictions.pred_scores[instance_index]:.3f}",
            positions=instance_coords,
            colors=instance_colors,
            normals=instance_normals,
            visible=False,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    viewer.save(str(output_dir))
    print(f"Saved {output_dir}")

def preprocess_scene(scene_path):
    
    def load_scene(scene_path):
        with scene_path.open("rb") as f:
            ply = PlyData.read(f)
        data = ply["vertex"].data
        field_names = set(data.dtype.names)
        coords = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
        colors = np.full((coords.shape[0], 3), 255, dtype=np.uint8)
        normals = None
        if {"red", "green", "blue"}.issubset(field_names):
            colors = np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(np.uint8)
        if {"nx", "ny", "nz"}.issubset(field_names):
            normals = np.stack([data["nx"], data["ny"], data["nz"]], axis=1).astype(np.float32)
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


def run_single_scene_inference(cfg: DictConfig, model, scene, file_name: str) -> PredictionOutput:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    adapter = InferenceDatasetAdapter(
        dataset_name=cfg.data.test_dataset.dataset_name,
        label_offset=0,
    )
    model.validation_dataset = adapter
    model.test_dataset = adapter

    coords, raw_colors, colors_norm, normals = scene

    feats = colors_norm.astype(np.float32)
    if cfg.data.add_normals:
        feats = np.hstack((feats, normals.astype(np.float32)))
    if cfg.data.add_raw_coordinates:
        feats = np.hstack((feats, coords.astype(np.float32)))

    # Fallback segmentation: each point is its own segment.
    segment_ids = np.arange(coords.shape[0], dtype=np.int32)[:, None]

    sample = (
        coords.astype(np.float32),
        feats.astype(np.float32),
        segment_ids.astype(np.int32),
        file_name,
        raw_colors.astype(np.uint8),
        normals.astype(np.float32),
        coords.astype(np.float32),
        0,
    )

    collate = VoxelizeCollate(
        ignore_label=cfg.data.test_collation.ignore_label,
        voxel_size=cfg.data.test_collation.voxel_size,
        mode=cfg.data.test_collation.mode,
        probing=cfg.data.test_collation.probing,
        task=cfg.data.test_collation.task,
        ignore_class_threshold=cfg.data.test_collation.ignore_class_threshold,
        filter_out_classes=list(cfg.data.test_collation.filter_out_classes),
        label_offset=0,
        num_queries=cfg.data.test_collation.num_queries,
    )

    batch = collate([sample])
    data, target, file_names = batch
    for item in target:
        if "point2segment" in item:
            item["point2segment"] = item["point2segment"].to(device)
    batch = (data, target, file_names)

    with torch.no_grad():
        model.eval_step(batch, 0)

    preds = model.preds[file_names[0]]
    return PredictionOutput(
        pred_masks=np.asarray(preds["pred_masks"]),
        pred_scores=np.asarray(preds["pred_scores"]),
        pred_classes=np.asarray(preds["pred_classes"]),
    )


def build_output_paths(scene_path: Path) -> OutputPaths:
    base_name = scene_path.stem
    return OutputPaths(
        ply=Path(f"{base_name}_pred_instances.ply"),
        segment_semantic=Path(f"{base_name}_segment_semantic.npy"),
        instance=Path(f"{base_name}_instance.npy"),
    )


def save_prediction_outputs(scene, predictions: PredictionOutput, threshold: float, output_paths: OutputPaths):
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
        semantic_labels[update] = np.uint16(max(predictions.pred_classes[instance_index], 0))
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


@hydra.main(
    config_path="conf", config_name="config_base_instance_segmentation.yaml"
)
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    
    # modify config
    cfg.general.checkpoint = 'checkpoints/scannet200/scannet200_benchmark.ckpt'
    cfg.general.num_targets=201
    cfg.data.num_labels=200
    cfg.general.eval_on_segments=True
    cfg.general.train_on_segments=True
    cfg.general.train_mode=False
    cfg.model.num_queries=150
    cfg.general.topk_per_image=300
    cfg.general.use_dbscan=True
    cfg.general.dbscan_eps=0.05
    cfg.general.dbscan_min_points=10
    cfg.general.export=False
    cfg.data.test_mode='test'
    cfg.general.export_threshold=0.001

    cfg, model = get_parameters(cfg)
    input_scene_path = Path('office.ply')
    scene = preprocess_scene(input_scene_path)

    predictions = run_single_scene_inference(cfg, model, scene, input_scene_path.stem)
    kept = int((predictions.pred_scores >= cfg.general.export_threshold).sum())
    output_paths = build_output_paths(input_scene_path)
    save_prediction_outputs(
        scene=scene,
        predictions=predictions,
        threshold=cfg.general.export_threshold,
        output_paths=output_paths,
    )

    # Visualization is intentionally run after inference/output generation.
    visualize_scene(scene)
    visualize_instance_masks(
        scene=scene,
        predictions=predictions,
        threshold=cfg.general.export_threshold,
        output_dir=Path(f"{input_scene_path.stem}_instance_masks_viz"),
    )

    print(f"Predicted masks shape: {predictions.pred_masks.shape}")
    print(f"Predicted scores shape: {predictions.pred_scores.shape}")
    print(f"Predicted classes shape: {predictions.pred_classes.shape}")
    print(
        f"Predicted instances above threshold ({cfg.general.export_threshold}): {kept}"
    )


if __name__ == "__main__":
    main()
