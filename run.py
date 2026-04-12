# My own inference code.

import click
from pathlib import Path
from plyfile import PlyData
import numpy as np
import open3d as o3d

DEFAULT_COLOR_MEAN_STD = (
    (0.47793125906962, 0.4303257521323044, 0.3749598901421883),
    (0.2834475483823543, 0.27566157565723015, 0.27018971370874995),
)


def visualize_scene(scene):
    import pyviz3d as viz
    coords, colors, colors_normalized, normals = scene
    v = viz.Visualizer()
    v.add_points(name="rgb", positions=coords, colors=colors, normals=normals)
    v.add_points(name="solid", positions=coords, normals=normals)
    v.save("scene")


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



@click.command()
@click.option('--input_scene_path', type=Path, default=Path('office.ply'), help='Input scene.')
def main(input_scene_path):
    scene = preprocess_scene(input_scene_path)
    visualize_scene(scene)


if __name__ == "__main__":
    main()
