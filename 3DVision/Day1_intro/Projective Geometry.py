

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
    with open("Day1_intro/resources/projective.png", "rb") as f:
        encoded_projective_image = base64.b64encode(f.read()).decode()
    with open("Day1_intro/resources/pinhole_camera.png", "rb") as f:
        encoded_pinhole_image = base64.b64encode(f.read()).decode()

    from_euclidean_2d = r"""
        \[
        \begin{pmatrix}
        x \\
        y \\
        \end{pmatrix}
        \rightarrow
        w
        \begin{pmatrix}
        x \\
        y \\
        1
        \end{pmatrix},
        \;
        w \neq 0
        \]
        """
    to_euclidean_2d = r"""
        \[
        \begin{pmatrix}
        x \\
        y \\
        z
        \end{pmatrix}
        \rightarrow
        \begin{pmatrix}
        x / z \\
        y / z \\
        \end{pmatrix},
        \;
        \]
        """

    from_euclidean_3d = r"""
        \[
        \begin{pmatrix}
        x \\
        y \\
        z \\
        \end{pmatrix}
        \rightarrow
        w
        \begin{pmatrix}
        x \\
        y \\
        z \\
        1
        \end{pmatrix},
        \;
        w \neq 0
        \]
    """

    equivalence_class_vectors = r"""
        \[
        \begin{pmatrix}
        3 \\
        -2 \\
        1
        \end{pmatrix}
        =
        \begin{pmatrix}
        6 \\
        -4 \\
        2 \\
        \end{pmatrix}
        =
        \begin{pmatrix}
        -9 \\
        6 \\
        -3 \\
        \end{pmatrix}
        \]
    """

    projective_space = r"""
        \[
        \mathcal{P}^2 = \mathcal{R}^3 \setminus \{(0, 0, 0)\}
        \]
    """

    transform_euclidean = r"""
        \[
        \begin{pmatrix}
        x' \\
        y' \\
        \end{pmatrix}
        =
        \begin{bmatrix}
        cos(\alpha) & -sin(\alpha) \\
        sin(\alpha) & cos(\alpha) \\
        \end{bmatrix}
        \begin{pmatrix}
        x \\
        y \\
        \end{pmatrix}
        \]
    
        \[
        \begin{pmatrix}
        x'' \\
        y''' \\
        \end{pmatrix}
        =
        \begin{pmatrix}
        x' \\
        y' \\
        \end{pmatrix}
        +
        \begin{pmatrix}
        t_x \\
        t_y \\
        \end{pmatrix}
        \]
    """

    transform_projective = r"""
        \[
        \begin{pmatrix}
        x'' \\
        y'' \\
        1
        \end{pmatrix}
        =
        \begin{bmatrix}
        cos(\alpha) & -sin(\alpha) & t_x \\
        sin(\alpha) & cos(\alpha) & t_y \\
        0 & 0 & 1
        \end{bmatrix}
        \begin{pmatrix}
        x \\
        y \\
        1
        \end{pmatrix}
        \]
    """

    intrinsics_matrix = r"""
        \[
        \mathbb{K} =
        \begin{bmatrix}
        f & s & p_x \\
        0 & \alpha f & p_y \\
        0 & 0 & 1
        \end{bmatrix}
        \]
    """

    intrinsics_matrix_typical = r"""
        \[
        \mathbb{K} =
        \begin{bmatrix}
        f & 0 & w / 2 \\
        0 & f & h / 2 \\
        0 & 0 & 1
        \end{bmatrix}
        \]
    """

    projection_3d_2d = r"""
        \[
        \begin{pmatrix}
        x' \\
        y' \\
        z' \\
        \end{pmatrix}
        =
        \mathbb{K}
        \begin{pmatrix}
        X \\
        Y \\
        Z \\
        \end{pmatrix}
        \]
    """

    extrinsics_matrix = r"""
        \[
        \begin{bmatrix}
        cos(\alpha) & -sin(\alpha) & t_x \\
        sin(\alpha) & cos(\alpha) & t_y \\
        0 & 0 & 1
        \end{bmatrix}
        \]
    """

    mo.md(
        f"""
        # Projective Geometry
        ## Homogeneous coordinates
        From euclidean coordinates to Homogeneous coordinates in 2D:

        {from_euclidean_2d}

        Equivalence class of vectors:

        {equivalence_class_vectors}

        From to Homogeneous coordinates to euclidean coordinates in 2D:

        {to_euclidean_2d}

        2D Projective space:

        {projective_space}

        <img src="data:image/png;base64,{encoded_projective_image}" width="250px" />

        From euclidean coordinates in 3D:

        {from_euclidean_3d}

        ## Transformations
        Let's say we want to rotate the point $(x, y)$ of $\\alpha$ degrees and translate it of $(t_x, t_y)$
    
        In Euclidean geometry:

        {transform_euclidean}

        In projective geometry:

        {transform_projective}

        * It let us stack multiple transformations just by multiplying the corresponding transformation matrices
        * Specific properties on the transformation matrix translate to invariants of the transformation (i.e. maintain lengths or ratios or parallelism, ...)

        ## Other properties
        * Point-line duality
        * Line at infinity
        * ...

        # Camera model - From 3D scenes to 2D images
        <img src="data:image/png;base64,{encoded_pinhole_image}" width="400px" />

        ## Camera intrinsics

        - Focal length $f$: distance between the camera and the image plane
        - Principal point $(p_x, p_y)$: where the optical axis intersects the image plane
        - Skew $s$: Accounts for non-orthogonal axis

        {intrinsics_matrix}

        Ideally:

        {intrinsics_matrix_typical}

        For an image of width $w$ and height $h$

        ## Projecting to the image
        Let's take a 3D point in euclidean coordinates $(X, Y, Z)$ we can obtain the projected point using the intrinsics matrix:

        {projection_3d_2d}

        The obtained $(x', y', z')$ point is actually a 2D point in homogeneous coordinates!

        The 2D point in euclidean coordinates will be $(x' / z', y' / z')$

        ## Camera Extrinsics
        The above assumes that the camera is positioned at the origin and has it's axis aligned with the 3D coordinate system. What do we do if that's not true and the camera is rotated of $\\alpha$ degrees and translated by $(t_x, t_y)$? Let's define the extrinsics as a transformation matrix:

        {extrinsics_matrix}
    
        """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import torch

    return go, plt, torch


@app.cell(hide_code=True)
def _(mo):
    x = mo.ui.slider(-50, 50, value=0, step=1, label="x")
    y = mo.ui.slider(-50, 50, value=0, step=1, label="y")
    z = mo.ui.slider(0, 100, value=50, step=1, label="z")
    mo.md(
        f"""
        Let's define the coordinate of a point

        {x}
        {y}
        {z}
        """
    )
    return x, y, z


@app.cell
def _(mo, x, y, z):
    point = [x.value, y.value, z.value]
    mo.md(f"{point=}")
    return


@app.cell
def _(mo, torch, x, y, z):
    points = torch.tensor(
        [
            [x.value + shift_x, y.value + shift_y, z.value]
            for shift_x, shift_y in ((0, 0), (-5, -5), (-5, 5), (5, -5), (5, 5))
        ]
    )
    mo.md(
        f"""
        We use that point to a torch tensor with a set of points defining a square (the set contains the vertices of the square and the center)

        {points.shape=}
    
        {points=}
        """
    )
    return (points,)


@app.cell
def _(fig, points):
    fig.update_traces(
        x=points.T[0],
        y=points.T[1],
        z=points.T[2],
    )
    fig
    return


@app.cell
def _(go):
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=[0], y=[0], z=[0], mode="markers", marker=dict(size=5, color="red")
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-50, 50]),
            yaxis=dict(range=[-50, 50]),
            zaxis=dict(range=[0, 100]),
        )
    )
    pass
    return (fig,)


@app.cell
def _(mo):
    mo.md(
        f"""
        Let's define the intrinsics for our camera, $f=50$ and $p=(50, 50)$
        """
    )
    return


@app.cell
def _(torch):
    intrinsics = torch.tensor([[50, 0, 50], [0, 50, 50], [0, 0, 1]])
    intrinsics
    return (intrinsics,)


@app.cell
def _(mo):
    mo.md(
        f"""
        Multiplying the intrinsics matrix to our 3D points we get the 2D points in homogeneous coordinates
        """
    )
    return


@app.cell
def _(intrinsics, points):
    points_2d_hom = intrinsics @ points.T
    points_2d_hom.T
    return (points_2d_hom,)


@app.cell
def _(mo):
    mo.md(
        f"""
        From which we obtain the euclidean coordinates
        """
    )
    return


@app.cell
def _(points_2d_hom):
    points_2d = (points_2d_hom[:2] / points_2d_hom[2]).T
    points_2d
    return (points_2d,)


@app.cell
def _(plt, points_2d):
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.plot(points_2d.T[0], points_2d.T[1], "ro")
    return


@app.cell(hide_code=True)
def _(mo, x, y, z):
    mo.md(
        f"""
        {x}
        {y}
        {z}
        """
    )
    return


@app.cell
def _(fig, points):
    fig.update_traces(
        x=points.T[0],
        y=points.T[1],
        z=points.T[2],
    )
    fig
    return


@app.cell
def _(mo):
    mo.md(
        f"""
        Let's try together adding extrinsics
        """
    )
    return


if __name__ == "__main__":
    app.run()
