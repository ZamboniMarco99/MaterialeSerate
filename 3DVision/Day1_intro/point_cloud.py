

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(f"""
        # Point Cloud
    """)
    return


@app.cell
def _():
    import torch
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    return plt, torch


@app.cell
def _(torch):
    num_points = 1000
    point_cloud = torch.randn(num_points, 3)
    point_cloud.shape
    return (point_cloud,)


@app.cell
def _(plt, point_cloud):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    return


@app.cell
def _(plt, torch):
    def _():
        size = 1.0
        center = torch.tensor([0.0, 0.0, 0.0])

        # Generate points uniformly distributed within the cube
        num_points = 1000
        random_points = torch.rand(num_points, 3) * size - (size / 2) + center

        # Create the figure for 3D plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot of the points inside the cube
        ax.scatter(random_points[:, 0], random_points[:, 1], random_points[:, 2], marker='s')

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Display the plot
        return plt.show()


    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Pros
        - Lightweight and simple
        - Can capture fine details where is necessary
        - Can be acquired easily with LiDAR or similar
        # Cons
        - No order and no structure
        - Points are disconnected, no information about surfaces
        - Often sampling is not uniform
        """
    )
    return


if __name__ == "__main__":
    app.run()
