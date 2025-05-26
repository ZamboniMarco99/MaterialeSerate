

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import math
    import trimesh
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import torch
    return Poly3DCollection, np, plt, trimesh


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Original Mesh""")
    return


@app.cell
def _(trimesh):
    example_mesh = "Day2_methods/data/ModelNet10/chair/train/chair_0001.off"
    mesh = trimesh.load(example_mesh)
    mesh
    return (mesh,)


@app.cell
def _(mesh):
    vertices = mesh.vertices
    faces = mesh.faces
    return faces, vertices


@app.cell
def _(Poly3DCollection, faces, plt, vertices):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each face as a polygon
    for face in faces:
        poly3d = [vertices[face]]
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=0.1, edgecolors='r', alpha=1))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Mesh')

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Point cloud""")
    return


@app.cell
def _(mesh):
    num_points = 1024
    point_cloud = mesh.sample(num_points)
    point_cloud
    return (point_cloud,)


@app.cell
def _(plt, point_cloud):
    def _():
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return fig


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Voxels""")
    return


@app.cell
def _(np):
    def voxelize_to_fixed_grid(mesh, grid_size=32):
        # Get mesh bounds
        bounds = mesh.bounds
        min_bounds, max_bounds = bounds

        # Set pitch based on object dimensions
        object_size = (max_bounds - min_bounds).max()
        pitch = (
            1.1 * object_size / grid_size
        )  # This ensures the object is visible in the grid

        # Voxelize
        voxelized = mesh.voxelized(pitch=pitch)
        dense = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)

        indices = voxelized.sparse_indices
        for i in indices:
            if np.all(i < grid_size):
                dense[tuple(i)] = 1

        return dense, voxelized
    return (voxelize_to_fixed_grid,)


@app.cell
def _(mesh, voxelize_to_fixed_grid):
    voxel_grid, voxelized = voxelize_to_fixed_grid(mesh)
    voxel_grid.shape, voxelized.shape
    return (voxel_grid,)


@app.cell
def _():
    # voxel_grid = mesh.voxelized(pitch=mesh.scale / 32).matrix.astype(np.uint8)
    return


@app.cell
def _(voxel_grid):
    print(voxel_grid.shape)
    return


@app.cell
def _(plt, voxel_grid):
    # # Get the coordinates of occupied voxels
    occupied = voxel_grid.nonzero()
    x_occupied, y_occupied, z_occupied = occupied[0], occupied[1], occupied[2]

    # # Get the RGB values corresponding to these voxels
    colors_occupied = voxel_grid[x_occupied, y_occupied, z_occupied]

    fig_occupied = plt.figure(figsize=(8, 8))
    ax_occupied = fig_occupied.add_subplot(111, projection="3d")

    # Plot the occupied voxels with RGB colors
    ax_occupied.scatter(x_occupied, y_occupied, z_occupied, zdir='z', c=colors_occupied, marker='s')

    ax_occupied.set_xlabel('X')
    ax_occupied.set_ylabel('Y')
    ax_occupied.set_zlabel('Z')
    ax_occupied.set_xlim(0, 32)
    ax_occupied.set_ylim(0, 32)
    ax_occupied.set_zlim(0, 32)
    ax_occupied.set_title("Voxel Chair")

    fig_occupied

    return


@app.cell
def _():
    converted_voxel_path = "Day2_methods/data/ModelNet10_voxels32/monitor/train/monitor_0023.npy"
    return (converted_voxel_path,)


@app.cell
def _(converted_voxel_path, np):
    converted_voxel = np.load(converted_voxel_path)
    return (converted_voxel,)


@app.cell
def _(converted_voxel):
    converted_voxel.shape
    return


@app.cell
def _(converted_voxel):
    converted_voxel.sum()
    return


@app.cell
def _(converted_voxel, plt):
    def _():
        # # Get the coordinates of occupied voxels
        occupied = converted_voxel.nonzero()
        x_occupied, y_occupied, z_occupied = occupied[0], occupied[1], occupied[2]

        # # Get the RGB values corresponding to these voxels
        colors_occupied = converted_voxel[x_occupied, y_occupied, z_occupied]

        fig_occupied = plt.figure(figsize=(8, 8))
        ax_occupied = fig_occupied.add_subplot(111, projection="3d")

        # Plot the occupied voxels with RGB colors
        ax_occupied.scatter(x_occupied, y_occupied, z_occupied, zdir='z', c=colors_occupied, marker='s')

        ax_occupied.set_xlabel('X')
        ax_occupied.set_ylabel('Y')
        ax_occupied.set_zlabel('Z')
        ax_occupied.set_xlim(0, 32)
        ax_occupied.set_ylim(0, 32)
        ax_occupied.set_zlim(0, 32)
        ax_occupied.set_title("Voxel Chair")
        return fig_occupied


    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
