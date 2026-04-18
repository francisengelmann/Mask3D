"""Visualize compressed Mask3D predictions.

Usage:
    python visualize_predictions.py <scene.ply> <predictions.npz> [--threshold 0.5] [--output-dir viz_out]

Requires: numpy, plyfile, open3d, pyviz3d
    pip install numpy plyfile open3d pyviz3d
"""

import argparse
import colorsys
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData


SCANNET200_LABELS = {
    1: 'wall', 2: 'chair', 3: 'floor', 4: 'table', 5: 'door', 6: 'couch',
    7: 'cabinet', 8: 'shelf', 9: 'desk', 10: 'office chair', 11: 'bed',
    13: 'pillow', 14: 'sink', 15: 'picture', 16: 'window', 17: 'toilet',
    18: 'bookshelf', 19: 'monitor', 21: 'curtain', 22: 'book',
    23: 'armchair', 24: 'coffee table', 26: 'box', 27: 'refrigerator',
    28: 'lamp', 29: 'kitchen cabinet', 31: 'towel', 32: 'clothes',
    33: 'tv', 34: 'nightstand', 35: 'counter', 36: 'dresser',
    38: 'stool', 39: 'cushion', 40: 'plant', 41: 'ceiling',
    42: 'bathtub', 44: 'end table', 45: 'dining table', 46: 'keyboard',
    47: 'bag', 48: 'backpack', 49: 'toilet paper', 50: 'printer',
    51: 'tv stand', 52: 'whiteboard', 54: 'blanket', 55: 'shower curtain',
    56: 'trash can', 57: 'closet', 58: 'stairs', 59: 'microwave',
    62: 'stove', 63: 'shoe', 64: 'computer tower', 65: 'bottle',
    66: 'bin', 67: 'ottoman', 68: 'bench', 69: 'board', 70: 'washing machine',
    71: 'mirror', 72: 'copier', 73: 'basket', 74: 'sofa chair',
    75: 'file cabinet', 76: 'fan', 77: 'laptop', 78: 'shower',
    79: 'paper', 80: 'person', 82: 'paper towel dispenser', 84: 'oven',
    86: 'blinds', 87: 'rack', 88: 'plate', 89: 'blackboard',
    90: 'piano', 93: 'suitcase', 95: 'rail', 96: 'radiator',
    97: 'recycling bin', 98: 'container', 99: 'wardrobe',
    100: 'soap dispenser', 101: 'telephone', 102: 'bucket', 103: 'clock',
    104: 'stand', 105: 'light', 106: 'laundry basket', 107: 'pipe',
    110: 'clothes dryer', 112: 'guitar', 115: 'toilet paper holder',
    116: 'seat', 118: 'speaker', 120: 'column', 121: 'bicycle',
    122: 'ladder', 125: 'bathroom stall', 128: 'shower wall',
    130: 'cup', 131: 'jacket', 132: 'storage bin', 134: 'coffee maker',
    136: 'dishwasher', 138: 'paper towel roll', 139: 'machine',
    140: 'mat', 141: 'windowsill', 145: 'bar', 148: 'toaster',
    154: 'bulletin board', 155: 'ironing board', 156: 'fireplace',
    157: 'soap dish', 159: 'kitchen counter', 161: 'doorframe',
    163: 'toilet paper dispenser', 165: 'mini fridge', 166: 'fire extinguisher',
    168: 'ball', 169: 'hat', 170: 'shower curtain rod', 177: 'water cooler',
    180: 'paper cutter', 185: 'tray', 188: 'shower door', 191: 'pillar',
    193: 'ledge', 195: 'toaster oven', 202: 'mouse',
    208: 'toilet seat cover dispenser', 213: 'furniture', 214: 'cart',
    221: 'storage container', 229: 'scale', 230: 'tissue box',
    232: 'light switch', 233: 'crate', 242: 'power outlet',
    250: 'decoration', 261: 'sign', 264: 'projector', 276: 'closet door',
    283: 'vacuum cleaner', 286: 'candle', 300: 'plunger',
    304: 'stuffed animal', 312: 'headphones', 323: 'dish rack',
    325: 'broom', 331: 'guitar case', 342: 'range hood', 356: 'dustpan',
    370: 'hair dryer', 392: 'water bottle', 395: 'handicap bar',
    399: 'purse', 408: 'vent', 417: 'shower floor', 488: 'water pitcher',
    540: 'mailbox', 562: 'bowl', 570: 'paper bag', 572: 'alarm clock',
    581: 'music stand', 609: 'projector screen', 748: 'divider',
    776: 'laundry detergent', 1156: 'bathroom counter', 1163: 'object',
    1164: 'bathroom vanity', 1165: 'closet wall', 1166: 'laundry hamper',
    1167: 'bathroom stall door', 1168: 'ceiling light', 1169: 'trash bin',
    1170: 'dumbbell', 1171: 'stair rail', 1172: 'tube',
    1173: 'bathroom cabinet', 1174: 'cd case', 1175: 'closet rod',
    1176: 'coffee kettle', 1178: 'structure', 1179: 'shower head',
    1180: 'keyboard piano', 1181: 'case of water bottles',
    1182: 'coat rack', 1183: 'storage organizer', 1184: 'folded chair',
    1185: 'fire alarm', 1186: 'power strip', 1187: 'calendar',
    1188: 'poster', 1189: 'potted plant', 1190: 'luggage', 1191: 'mattress',
}


def random_colors(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hues = rng.uniform(0, 1, size=n)
    return np.array(
        [np.array(colorsys.hsv_to_rgb(h, 1.0, 1.0)) * 255 for h in hues],
        dtype=np.uint8,
    )


def estimate_normals(coords: np.ndarray, knn: int = 30) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    return np.asarray(pcd.normals, dtype=np.float32)


def load_ply(path: Path):
    with path.open("rb") as f:
        ply = PlyData.read(f)
    data = ply["vertex"].data
    field_names = set(data.dtype.names)
    coords = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    normals = None
    if {"nx", "ny", "nz"}.issubset(field_names):
        normals = np.stack([data["nx"], data["ny"], data["nz"]], axis=1).astype(np.float32)
    return coords, normals


def load_predictions(path: Path):
    data = np.load(path)
    shape = tuple(data["pred_masks_shape"])
    pred_masks = np.unpackbits(data["pred_masks_packed"], axis=0)[:shape[0], :shape[1]].astype(bool)
    return pred_masks, data["pred_scores"], data["pred_classes"]


def main():
    parser = argparse.ArgumentParser(description="Visualize Mask3D compressed predictions.")
    parser.add_argument("scene", type=Path, help="Input point cloud (.ply)")
    parser.add_argument("predictions", type=Path, help="Compressed predictions (.npz)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold (default: 0.5)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for visualization")
    args = parser.parse_args()

    try:
        import pyviz3d as viz
    except ImportError:
        raise SystemExit("pyviz3d is required: pip install pyviz3d")

    output_dir = args.output_dir or Path(args.scene.stem + "_viz")

    coords, normals = load_ply(args.scene)
    if normals is None:
        print("No normals in PLY — estimating via KNN (knn=30)...")
        normals = estimate_normals(coords)
    pred_masks, pred_scores, pred_classes = load_predictions(args.predictions)

    valid = np.where(pred_scores >= args.threshold)[0]
    if len(valid) == 0:
        print(f"No instances above threshold {args.threshold}")
        return

    print(f"Visualizing {len(valid)} instances above threshold {args.threshold}")

    palette = random_colors(len(valid))
    viewer = viz.Visualizer()
    label_texts, label_positions, label_colors = [], [], []

    for palette_index, instance_index in enumerate(valid):
        mask = pred_masks[:, instance_index]
        if not np.any(mask):
            continue

        instance_coords = coords[mask]
        color = palette[palette_index]
        instance_colors = np.tile(color, (instance_coords.shape[0], 1)).astype(np.uint8)

        kwargs = dict(
            name=f"{instance_index}_{pred_scores[instance_index]:.3f}",
            positions=instance_coords,
            colors=instance_colors,
            normals=normals[mask],
            visible=True,
        )

        viewer.add_points(**kwargs)

        class_id = int(pred_classes[instance_index])
        class_label = SCANNET200_LABELS.get(class_id, str(class_id))
        label_texts.append(class_label)
        label_positions.append(instance_coords.mean(axis=0).astype(np.float32))
        label_colors.append(color)

    if label_texts:
        viewer.add_labels(
            name="semantic_labels",
            labels=label_texts,
            positions=np.array(label_positions, dtype=np.float32),
            colors=np.array(label_colors, dtype=np.uint8),
            visible=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    viewer.save(str(output_dir))
    print(f"Saved visualization to {output_dir}/")
    print(f"Open {output_dir}/index.html in a browser to view.")


if __name__ == "__main__":
    main()
