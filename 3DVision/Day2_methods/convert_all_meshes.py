from pathlib import Path

import numpy as np
import trimesh
from tqdm import tqdm


def voxelize_to_fixed_grid(mesh, grid_size=32):
    # Get mesh bounds
    bounds = mesh.bounds
    min_bounds, max_bounds = bounds

    # Set pitch based on object dimensions
    object_size = (max_bounds - min_bounds).max()
    pitch = (
        1.2 * object_size / grid_size
    )  # This ensures the object is visible in the grid

    # Voxelize
    voxelized = mesh.voxelized(pitch=pitch)
    dense = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)

    indices = voxelized.sparse_indices
    for i in indices:
        if np.all(i < grid_size):
            dense[tuple(i)] = 1

    return dense, voxelized


def process_modelnet10(
    modelnet_dir, output_voxels_dir, output_pointcloud_dir, grid_size=32, n_points=1024
):
    mesh_paths = list(Path(modelnet_dir).rglob("*.off"))

    output_voxels_dir = Path(output_voxels_dir)
    output_voxels_dir.mkdir(parents=True, exist_ok=True)

    output_pointcloud_dir = Path(output_pointcloud_dir)
    output_pointcloud_dir.mkdir(parents=True, exist_ok=True)

    for path in tqdm(mesh_paths, desc="Processing meshes"):
        try:
            mesh = trimesh.load_mesh(path, process=False)

            # Generate voxel representation
            dense, voxelized = voxelize_to_fixed_grid(mesh, grid_size)

            # Save voxel data as NPY (primary format)
            rel_path = path.relative_to(modelnet_dir).with_suffix(".npy")
            save_path = output_voxels_dir / rel_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, dense)

            # Generate and save pointcloud with 1024 points as PLY
            points = mesh.sample(n_points)
            pc_save_path = output_pointcloud_dir / path.relative_to(
                modelnet_dir
            ).with_suffix(".ply")
            pc_save_path.parent.mkdir(parents=True, exist_ok=True)

            # Create a point cloud mesh and save as PLY
            pc_mesh = trimesh.PointCloud(points)
            pc_mesh.export(pc_save_path)

        except Exception as e:
            print(f"Failed to process {path}: {e}")


# Example usage:
# Replace with your actual ModelNet10 path
modelnet10_dir = "Day2_methods/data/ModelNet10"
output_voxels_dir = "Day2_methods/data/ModelNet10_voxels32"
output_pointcloud_dir = "Day2_methods/data/ModelNet10_pointcloud1024"
process_modelnet10(modelnet10_dir, output_voxels_dir, output_pointcloud_dir)
