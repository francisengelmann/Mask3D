import argparse
import os
from pathlib import Path
from typing import NamedTuple

os.environ.setdefault("OMP_NUM_THREADS", "12")

import numpy as np
import open3d as o3d
import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from plyfile import PlyData
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from datasets.utils import VoxelizeCollate
from trainer.trainer import InstanceSegmentation, get_evenly_distributed_colors
from utils.point_cloud_utils import write_point_cloud_in_ply
from utils.utils import load_checkpoint_with_missing_or_exsessive_keys


ROOT = Path(__file__).resolve().parent
SEGMENT_FIELDS = ("segment_id", "segment", "instance", "objectId")
DEFAULT_COLOR_MEAN_STD = (
    (0.47793125906962, 0.4303257521323044, 0.3749598901421883),
    (0.2834475483823543, 0.27566157565723015, 0.27018971370874995),
)

SAMPLE_COORDS = 0
SAMPLE_LABELS = 2
SAMPLE_RAW_COLORS = 4


class OutputPaths(NamedTuple):
    ply: Path
    segment_semantic: Path
    instance: Path
    segment_viz: Path


class InferenceDatasetAdapter:
    def __init__(self, dataset_name: str, label_offset: int = 0):
        self.dataset_name = dataset_name
        self.label_offset = label_offset
        self.data = []

    def _remap_model_output(self, output):
        return np.asarray(output)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Mask3D inference on a single PLY scene and save output files.")
    parser.add_argument("--scene", type=Path, default=Path("office.ply"), help="Path to the input scene PLY file")
    parser.add_argument("--output", type=Path, default=Path("output.ply"), help="Path to the output PLY file",)
    parser.add_argument("--curr-query", type=int, default=300)
    parser.add_argument("--curr-topk", type=int, default=300)
    parser.add_argument("--curr-dbscan", type=float, default=0.95)
    parser.add_argument("--curr-t", type=float, default=0.6, help="Confidence threshold for predicted instances")
    parser.add_argument(
        "--auto-superpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate superpoint-like segment IDs with DBSCAN when none are present in the PLY",
    )
    parser.add_argument(
        "--superpoint-eps-factor",
        type=float,
        default=8.0,
        help="Multiplier on median nearest-neighbor distance for auto-superpoint DBSCAN eps",
    )
    parser.add_argument(
        "--superpoint-min-samples",
        type=int,
        default=20,
        help="DBSCAN min_samples for auto-superpoint generation",
    )
    parser.add_argument(
        "--superpoint-normal-weight",
        type=float,
        default=0.15,
        help="Weight for normals in auto-superpoint DBSCAN feature space",
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/scannet200/scannet200_benchmark.ckpt"),)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"],)
    return parser.parse_args()


def robust_eps(coords: np.ndarray, eps_factor: float) -> float:
    if coords.shape[0] < 2:
        return eps_factor

    neighbors = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(coords)
    distances, _ = neighbors.kneighbors(coords)
    base_distance = np.median(distances[:, 1])
    return float(max(base_distance * eps_factor, np.finfo(np.float32).eps))


def estimate_normals(coords: np.ndarray) -> np.ndarray:
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(coords.astype(np.float64))
    point_cloud.estimate_normals()
    return np.asarray(point_cloud.normals, dtype=np.float32)


def remap_dbscan_segments(labels: np.ndarray) -> np.ndarray:
    segments = labels.copy()
    noise_mask = segments == -1

    if np.any(noise_mask):
        max_non_noise = segments[~noise_mask].max() if np.any(~noise_mask) else -1
        segments[noise_mask] = np.arange(
            max_non_noise + 1,
            max_non_noise + 1 + noise_mask.sum(),
            dtype=np.int64,
        )

    _, segments = np.unique(segments, return_inverse=True)
    return segments.astype(np.int32)


def generate_superpoint_segments(
    coords: np.ndarray,
    normals: np.ndarray,
    eps_factor: float,
    min_samples: int,
    normal_weight: float,
) -> np.ndarray:
    eps = robust_eps(coords, eps_factor)
    features = np.hstack((coords, normals * normal_weight)).astype(np.float32)
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(features)
    return remap_dbscan_segments(labels)


def load_segments(
    vertex_data,
    field_names,
    coords: np.ndarray,
    normals: np.ndarray,
    args,
) -> np.ndarray:
    segment_field = next(
        (candidate for candidate in SEGMENT_FIELDS if candidate in field_names), None
    )

    if segment_field is not None:
        segments = np.asarray(vertex_data[segment_field], dtype=np.int32)
        _, segments = np.unique(segments, return_inverse=True)
        return segments.astype(np.int32)

    if args.auto_superpoints:
        segments = generate_superpoint_segments(
            coords=coords,
            normals=normals,
            eps_factor=args.superpoint_eps_factor,
            min_samples=args.superpoint_min_samples,
            normal_weight=args.superpoint_normal_weight,
        )
        print(
            f"Generated {int(segments.max()) + 1} auto-superpoints from input without segment labels"
        )
        return segments

    return np.arange(coords.shape[0], dtype=np.int32)


def load_scene(scene_path: Path, need_normals: bool, args):
    with scene_path.open("rb") as handle:
        ply = PlyData.read(handle)
    data = ply["vertex"].data
    field_names = set(data.dtype.names)

    coords = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)

    if {"red", "green", "blue"}.issubset(field_names):
        colors = np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(np.uint8)
    else:
        colors = np.full((coords.shape[0], 3), 255, dtype=np.uint8)

    normals = None
    if {"nx", "ny", "nz"}.issubset(field_names):
        normals = np.stack([data["nx"], data["ny"], data["nz"]], axis=1).astype(np.float32)
    elif need_normals or args.auto_superpoints:
        normals = estimate_normals(coords)

    if normals is None:
        normals = np.zeros((coords.shape[0], 3), dtype=np.float32)

    segments = load_segments(data, field_names, coords, normals, args)

    return coords, colors, normals, segments


def load_color_normalization(path_or_stats):
    if isinstance(path_or_stats, str):
        stats_path = ROOT / path_or_stats
        if stats_path.exists():
            with stats_path.open() as handle:
                stats = yaml.safe_load(handle)
            mean = np.asarray(stats["mean"], dtype=np.float32)
            std = np.asarray(stats["std"], dtype=np.float32)
            return mean, std

        path_or_stats = DEFAULT_COLOR_MEAN_STD

    mean = np.asarray(path_or_stats[0], dtype=np.float32)
    std = np.asarray(path_or_stats[1], dtype=np.float32)
    return mean, std


def build_sample(cfg, scene_path: Path, args):
    """Loads a scene and prepares the input features and labels for inference."""

    coords, raw_colors, raw_normals, segments = load_scene(
        scene_path, need_normals=cfg.data.add_normals, args=args
    )
    coords_f32 = coords.astype(np.float32)
    raw_normals_f32 = raw_normals.astype(np.float32)
    raw_colors_u8 = raw_colors.astype(np.uint8)

    colors = raw_colors_u8.astype(np.float32)

    if cfg.data.add_colors:
        mean, std = load_color_normalization(cfg.data.test_dataset.color_mean_std)
        colors = (colors / 255.0 - mean) / std
    else:
        colors = np.ones((coords.shape[0], 3), dtype=np.float32)

    features = colors
    if cfg.data.add_normals:
        features = np.hstack((features, raw_normals_f32))
    if cfg.data.add_raw_coordinates:
        features = np.hstack((features, coords_f32))

    labels = segments[:, None].astype(np.int32)
    return (
        coords_f32,
        features.astype(np.float32),
        labels,
        scene_path.stem,
        raw_colors_u8,
        raw_normals_f32,
        coords_f32,
        0,
    )


def resolve_output_paths(args) -> OutputPaths:
    return OutputPaths(
        ply=args.output,
        segment_semantic=args.output_seg_sem
        or args.output.with_name(f"{args.output.stem}_segment_semantic.npy"),
        instance=args.output_instance
        or args.output.with_name(f"{args.output.stem}_instance.npy"),
        segment_viz=args.viz_segments_dir
        or args.output.with_name(f"{args.output.stem}_segments_viz"),
    )


def ensure_output_directories(output_paths: OutputPaths, save_segment_viz: bool):
    output_paths.ply.parent.mkdir(parents=True, exist_ok=True)
    output_paths.segment_semantic.parent.mkdir(parents=True, exist_ok=True)
    output_paths.instance.parent.mkdir(parents=True, exist_ok=True)
    if save_segment_viz:
        output_paths.segment_viz.parent.mkdir(parents=True, exist_ok=True)


def compose_cfg(args):
    overrides = [
        f'general.experiment_name=inference_query_{args.curr_query}_topk_{args.curr_topk}_dbscan_{args.curr_dbscan}_export_{args.curr_t}',
        "general.project_name=scannet200_eval",
        f"general.checkpoint={args.checkpoint.as_posix()}",
        "data/datasets=scannet200",
        "general.num_targets=201",
        "data.num_labels=200",
        "general.eval_on_segments=true",
        "general.train_on_segments=true",
        "general.train_mode=false",
        f"model.num_queries={args.curr_query}",
        f"general.topk_per_image={args.curr_topk}",
        "general.use_dbscan=true",
        f"general.dbscan_eps={args.curr_dbscan}",
        "general.export=false",
        "data.test_mode=test",
        f"general.export_threshold={args.curr_t}",
    ]

    with initialize_config_dir(version_base="1.1", config_dir=str(ROOT / "conf")):
        cfg = compose(
            config_name="config_base_instance_segmentation.yaml",
            overrides=overrides,
        )

    OmegaConf.update(cfg, "logging", [], merge=False)

    return cfg


def build_module(cfg, device: torch.device):
    module = InstanceSegmentation(cfg)
    _, module = load_checkpoint_with_missing_or_exsessive_keys(cfg, module)
    dataset_adapter = InferenceDatasetAdapter(
        dataset_name=cfg.data.test_dataset.dataset_name,
        label_offset=0,
    )
    module.validation_dataset = dataset_adapter
    module.test_dataset = dataset_adapter
    module.to(device)
    module.eval()
    return module


def build_collate(cfg):
    return VoxelizeCollate(
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


def run_single_scene(module, collate, sample):
    batch = collate([sample])
    device = next(module.parameters()).device
    data, target, file_names = batch
    for item in target:
        if "point2segment" in item:
            item["point2segment"] = item["point2segment"].to(device)
    batch = data, target, file_names

    with torch.no_grad():
        module.eval_step(batch, 0)

    file_name = batch[2][0]
    return module.preds[file_name]


def build_output_arrays(sample, preds, score_threshold):
    coords = sample[SAMPLE_COORDS]
    segment_ids = sample[SAMPLE_LABELS][:, 0].astype(np.int32)
    original_colors = sample[SAMPLE_RAW_COLORS].copy()
    pred_masks = preds["pred_masks"]
    pred_scores = np.asarray(preds["pred_scores"])
    pred_classes = np.asarray(preds["pred_classes"])

    assigned_scores = np.full(coords.shape[0], -np.inf, dtype=np.float32)
    semantic_labels = np.full(coords.shape[0], 255, dtype=np.uint16)
    instance_ids = np.full(coords.shape[0], -1, dtype=np.int32)

    valid_instances = np.where(pred_scores >= score_threshold)[0]
    for instance_index in valid_instances:
        mask = pred_masks[:, instance_index].astype(bool)
        update = mask & (pred_scores[instance_index] > assigned_scores)
        assigned_scores[update] = pred_scores[instance_index]
        semantic_labels[update] = np.uint16(max(pred_classes[instance_index], 0))
        instance_ids[update] = instance_index

    output_colors = original_colors.copy()
    if len(valid_instances) == 0:
        return coords, segment_ids, output_colors, semantic_labels, instance_ids

    palette = np.asarray(
        get_evenly_distributed_colors(len(valid_instances) + 1), dtype=np.uint8
    )
    for palette_index, instance_index in enumerate(valid_instances, start=1):
        output_colors[instance_ids == instance_index] = palette[palette_index]

    return coords, segment_ids, output_colors, semantic_labels, instance_ids


def save_segment_visualization(
    coords: np.ndarray,
    segment_ids: np.ndarray,
    output_dir: Path,
    point_size: int,
):
    try:
        import pyviz3d.visualizer as viz
    except ImportError as import_error:
        raise RuntimeError(
            "pyviz3d is required for --viz-segments-dir but is not installed"
        ) from import_error

    unique_segments, inverse = np.unique(segment_ids, return_inverse=True)
    palette = np.asarray(
        get_evenly_distributed_colors(len(unique_segments) + 1), dtype=np.uint8
    )
    colors = palette[inverse + 1]

    v = viz.Visualizer()
    v.add_points(
        name="segments",
        positions=coords.astype(np.float32),
        colors=colors,
        point_size=point_size,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    v.save(str(output_dir), verbose=True)


def main():
    args = parse_args()
    cfg = compose_cfg(args)
    device = torch.device(args.device)
    score_threshold = args.curr_t
    output_paths = resolve_output_paths(args)   

    sample = build_sample(cfg, args.scene, args)
    module = build_module(cfg, device)
    collate = build_collate(cfg)

    preds = run_single_scene(module, collate, sample)

    coords, segment_ids, output_colors, semantic_labels, instance_ids = build_output_arrays(
        sample, preds, score_threshold
    )

    ensure_output_directories(
        output_paths=output_paths,
        save_segment_viz=args.viz_segments_dir is not None,
    )

    write_point_cloud_in_ply(
        output_paths.ply,
        coords,
        feats=output_colors,
        labels=semantic_labels,
    )

    seg_sem_array = np.column_stack(
        (segment_ids.astype(np.int32), semantic_labels.astype(np.int32))
    )
    np.save(output_paths.segment_semantic, seg_sem_array)
    np.save(output_paths.instance, instance_ids.astype(np.int32))

    if args.viz_segments_dir is not None:
        save_segment_visualization(
            coords=coords,
            segment_ids=segment_ids,
            output_dir=output_paths.segment_viz,
            point_size=args.viz_point_size,
        )

    kept_instances = int((np.asarray(preds["pred_scores"]) >= score_threshold).sum())
    print(f"Saved {output_paths.ply}")
    print(f"Saved {output_paths.segment_semantic}")
    print(f"Saved {output_paths.instance}")
    if args.viz_segments_dir is not None:
        print(f"Saved {output_paths.segment_viz}")
    print(f"Predicted instances above threshold: {kept_instances}")


if __name__ == "__main__":
    main()