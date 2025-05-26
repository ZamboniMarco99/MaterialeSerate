

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""# Mesh""")
    return


@app.cell
def _():
    import torch
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np
    return Poly3DCollection, plt, torch


@app.cell
def _(torch):
    vertices = torch.tensor([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
        [0.0, 0.0, 1.0],  # 4
        [1.0, 0.0, 1.0],  # 5
        [1.0, 1.0, 1.0],  # 6
        [0.0, 1.0, 1.0],  # 7
    ], dtype=torch.float32)

    # Define faces (each face is a triangle, listed by vertex indices)
    faces_triangular = torch.tensor([
        [0, 1, 2], [0, 2, 3],  # Bottom
        [4, 5, 6], [4, 6, 7],  # Top
        [0, 1, 5], [0, 5, 4],  # Front
        [1, 2, 6], [1, 6, 5],  # Right
        [2, 3, 7], [2, 7, 6],  # Back
        [3, 0, 4], [3, 4, 7],  # Left
    ], dtype=torch.int32)
    faces = torch.tensor([
        [0, 1, 2, 3],  # Bottom
        [4, 5, 6, 7],  # Top
        [0, 1, 5, 4],  # Front
        [1, 2, 6, 5],  # Right
        [2, 3, 7, 6],  # Back
        [3, 0, 4, 7]  # Left
    ])
    # faces = faces_triangular

    print(vertices)
    print(faces)
    return faces, vertices


@app.cell
def _(mo):
    mo.md(r"""Meshes are graphs""")
    return


@app.cell
def _(faces, torch):
    edges_list = []
    for i in range(faces.shape[1]):
        edge = torch.stack([faces[:, i], faces[:, (i+1) % faces.shape[1]]], dim=1)
        edges_list.append(edge)
    edges = torch.cat(edges_list, dim=0)
    edges = torch.sort(edges, dim=1)[0]
    edges = torch.unique(edges, dim=0)
    print(edges)
    return


@app.cell
def _(Poly3DCollection, faces, plt, vertices):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each face as a polygon
    for face in faces:
        poly3d = [vertices[face].numpy()]
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Mesh')

    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Pros
        - Structured
        - Efficient
        # Cons
        - Complicated structure
        - Only model the surface
        """
    )
    return


if __name__ == "__main__":
    app.run()
