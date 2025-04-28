

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    import base64
    with open("Day1_intro/resources/minecraft.png", "rb") as f:
        encoded_minecraft_image = base64.b64encode(f.read()).decode()


    mo.md(f"""
        # Voxels
        <img src="data:image/png;base64,{encoded_minecraft_image}" width="800px" />

        Equivalent to the pixel
    """)
    return (base64,)


@app.cell
def _():
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    return plt, torch


@app.cell
def _(torch):
    # ## Creating an RGB Voxel Cube
    # We will now create a voxel grid where each voxel has an RGB value.
    # The color will be based on the position of the voxel in the grid.

    # Create a 32x32x32 voxel grid
    grid_size = 32
    voxel_grid = torch.zeros((grid_size, grid_size, grid_size, 3))

    # Fill a cube in the center with RGB values
    # Assign different colors to different faces of the cube
    voxel_grid[10:22, 10:22, 10:22, 0] = 0.1  # Red face
    voxel_grid[10:22, 10:22, 10:22, 1] = 0.1  # Green face
    voxel_grid[10:22, 10:22, 10:22, 2] = 1  # Blue face

    print("RGB Voxel grid shape:", voxel_grid.shape)

    return grid_size, voxel_grid


@app.cell
def _(grid_size, plt, voxel_grid):
    # # Get the coordinates of occupied voxels
    occupied = voxel_grid.nonzero()
    x_occupied, y_occupied, z_occupied = occupied[:, 0], occupied[:, 1], occupied[:, 2]

    # # Get the RGB values corresponding to these voxels
    colors_occupied = voxel_grid[x_occupied, y_occupied, z_occupied]

    fig_occupied = plt.figure(figsize=(8, 8))
    ax_occupied = fig_occupied.add_subplot(111, projection="3d")

    # Plot the occupied voxels with RGB colors
    ax_occupied.scatter(x_occupied, y_occupied, z_occupied, zdir='z', c=colors_occupied.numpy(), marker='s')

    ax_occupied.set_xlabel('X')
    ax_occupied.set_ylabel('Y')
    ax_occupied.set_zlabel('Z')
    ax_occupied.set_xlim(0, grid_size)
    ax_occupied.set_ylim(0, grid_size)
    ax_occupied.set_zlim(0, grid_size)
    ax_occupied.set_title("Voxel Cube")

    fig_occupied

    return


@app.cell
def _(grid_size, plt, torch, voxel_grid):
    # Create a 3D grid of all voxel coordinates
    x, y, z = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), torch.arange(grid_size))

    # Reshape to make the coordinates easier to work with for plotting
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)

    # Get the RGB values corresponding to every voxel
    colors = voxel_grid[x, y, z]

    # Visualize every voxel (including black ones)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot all voxels, even the empty ones (colored black)
    ax.scatter(x, y, z, c=colors.numpy(), marker='s', s=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("All Voxels")

    fig

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Pros
        - Dense, uniform and simple representation
        - Discrete representation
        - Compatible with 3D convolutions
        ## Cons
        - High memory usage
        - Limited resolution
        - Sparsity: in most 3D scenes there is a lot of empty space
        """
    )
    return


@app.cell(hide_code=True)
def _(base64, mo):
    def _():
        with open("Day1_intro/resources/ct_scan.png", "rb") as f:
            encoded_brain_image = base64.b64encode(f.read()).decode()
        with open("Day1_intro/resources/geo_tomography.png", "rb") as f:
            encoded_geo_image = base64.b64encode(f.read()).decode()
        return mo.md(f"""
            ## Tomography
            <img src="data:image/png;base64,{encoded_brain_image}" width="800px" />
            <img src="data:image/png;base64,{encoded_geo_image}" width="800px" />
        """)


    _()
    return


if __name__ == "__main__":
    app.run()
