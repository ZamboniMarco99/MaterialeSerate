

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.collections import PolyCollection
    import numpy as np
    from manopth.manolayer import ManoLayer
    import torch
    return ManoLayer, Poly3DCollection, PolyCollection, np, plt, torch


@app.cell
def _(mo):
    import base64
    with open("Day1_intro/resources/MANO.png", "rb") as f:
        encoded_mano_image = base64.b64encode(f.read()).decode()

    mo.md(
        f"""
        # Body models - MANO

        <img src="data:image/png;base64,{encoded_mano_image}" width="400px" />
        """
    )
    return


@app.cell
def _():
    with open("Day1_intro/resources/example/hand_pose_mano/000560.txt") as file:
        data = list(map(float, file.read().split(" ")))

    return (data,)


@app.cell
def _(data):
    left_hand = data[1:62]
    right_hand = data[63:]
    len(left_hand), len(right_hand)
    return left_hand, right_hand


@app.cell
def _(mo):
    choice = mo.ui.dropdown(["Left hand", "Right hand"])
    choice
    return (choice,)


@app.cell
def _(choice, left_hand, right_hand):
    if choice.value == "Left hand":
        hand_trans = left_hand[:3]
        hand_pose = left_hand[3:51]
        hand_shape = left_hand[51:]
    else:
        hand_trans = right_hand[:3]
        hand_pose = right_hand[3:51]
        hand_shape = right_hand[51:]
    return hand_pose, hand_shape, hand_trans


@app.cell
def _(hand_pose, hand_shape, hand_trans, torch):
    hand_trans_t = torch.unsqueeze(torch.tensor(hand_trans), 0)
    hand_pose_t = torch.unsqueeze(torch.tensor(hand_pose), 0)
    hand_shape_t = torch.unsqueeze(torch.tensor(hand_shape), 0)
    return hand_pose_t, hand_shape_t, hand_trans_t


@app.cell
def _(ManoLayer, choice):
    mano_model = ManoLayer(
        mano_root="Day1_intro/resources/example/mano",
        ncomps=45,
        use_pca=False,
        flat_hand_mean=True,
        side="left" if choice.value == "Left hand" else "right"
    )
    return (mano_model,)


@app.cell
def _(hand_pose_t, hand_shape_t, hand_trans_t, mano_model):
    vertices, joints = mano_model(
        hand_pose_t,
        hand_shape_t,
        hand_trans_t,
    )
    return (vertices,)


@app.cell
def _(mano_model):
    faces = mano_model.th_faces
    return (faces,)


@app.cell
def _(Poly3DCollection, faces, plt, vertices):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each face as a polygon
    for face in faces:
        poly3d = [vertices.squeeze()[face].numpy()]
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=0.05, edgecolors='r', alpha=1))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Mesh')

    plt.show()
    return


@app.cell(hide_code=True)
def _(choice):
    choice
    return


@app.cell(hide_code=True)
def _(plt):
    with open("Day1_intro/resources/example/rgb/000560.png", "rb") as image_file:
        image = plt.imread(image_file)
    plt.axis('off')
    plt.imshow(image)
    return (image,)


@app.cell
def _(np, torch):
    intrinsics_path = "Day1_intro/resources/example/cam_intrinsics.txt"
    [fx, fy, cx, cy, w, h] = np.loadtxt(intrinsics_path)
    intrinsics = torch.tensor(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )
    intrinsics.shape
    return (intrinsics,)


@app.cell
def _(vertices):
    vertices.shape
    return


@app.cell
def _(intrinsics, torch, vertices):
    vertices_2d = torch.matmul(intrinsics, vertices.squeeze().T)
    vertices_2d = (vertices_2d[:2] / vertices_2d[2]).T
    vertices_2d.shape
    return (vertices_2d,)


@app.cell
def _(PolyCollection, faces, image, plt, vertices_2d):
    def _():
        triangles = vertices_2d[faces]
        fig, ax = plt.subplots()
        ax.imshow(image)

        # Add mesh
        mesh = PolyCollection(triangles, facecolors='cyan', edgecolors='r', linewidths=0.05)
        ax.add_collection(mesh)

        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        ax.set_aspect('equal')
        # plt.title("3D Mesh Projected on Image")
        plt.axis('off')
        return plt.show()


    _()
    return


@app.cell
def _(ManoLayer):
    left_mano_model = ManoLayer(
        mano_root="Day1_intro/resources/example/mano",
        ncomps=45,
        use_pca=False,
        flat_hand_mean=True,
        side="left"
    )
    right_mano_model = ManoLayer(
        mano_root="Day1_intro/resources/example/mano",
        ncomps=45,
        use_pca=False,
        flat_hand_mean=True,
        side="right"
    )
    return left_mano_model, right_mano_model


@app.cell(hide_code=True)
def _(
    PolyCollection,
    intrinsics,
    left_mano_model,
    plt,
    right_mano_model,
    torch,
):
    def generate_all_frames():
        for num in range(894):
            with open(f"Day1_intro/resources/example/rgb/{num:06d}.png", "rb") as image_file:
                image = plt.imread(image_file)
            # plt.imshow(image)
            with open(f"Day1_intro/resources/example/hand_pose_mano/{num:06d}.txt") as file:
                data = list(map(float, file.read().split(" ")))
            left_hand = data[1:62]
            right_hand = data[63:]
    
            left_hand_trans = left_hand[:3]
            left_hand_pose = left_hand[3:51]
            left_hand_shape = left_hand[51:]
        
            right_hand_trans = right_hand[:3]
            right_hand_pose = right_hand[3:51]
            right_hand_shape = right_hand[51:]
    
            left_hand_trans_t = torch.unsqueeze(torch.tensor(left_hand_trans), 0)
            left_hand_pose_t = torch.unsqueeze(torch.tensor(left_hand_pose), 0)
            left_hand_shape_t = torch.unsqueeze(torch.tensor(left_hand_shape), 0)
    
            right_hand_trans_t = torch.unsqueeze(torch.tensor(right_hand_trans), 0)
            right_hand_pose_t = torch.unsqueeze(torch.tensor(right_hand_pose), 0)
            right_hand_shape_t = torch.unsqueeze(torch.tensor(right_hand_shape), 0)
    
            left_vertices, left_joints = left_mano_model(
                left_hand_pose_t,
                left_hand_shape_t,
                left_hand_trans_t,
            )
            left_faces = left_mano_model.th_faces
            right_vertices, right_joints = right_mano_model(
                right_hand_pose_t,
                right_hand_shape_t,
                right_hand_trans_t,
            )
            right_faces = right_mano_model.th_faces
    
            left_vertices_2d = torch.matmul(intrinsics, left_vertices.squeeze().T)
            left_vertices_2d = (left_vertices_2d[:2] / left_vertices_2d[2]).T
        
            right_vertices_2d = torch.matmul(intrinsics, right_vertices.squeeze().T)
            right_vertices_2d = (right_vertices_2d[:2] / right_vertices_2d[2]).T
    
            left_triangles = left_vertices_2d[left_faces]
            right_triangles = right_vertices_2d[right_faces]
            fig, ax = plt.subplots()
            ax.imshow(image)
    
            # Add mesh
            left_mesh = PolyCollection(left_triangles, facecolors='cyan', edgecolors='r', linewidths=0.05)
            right_mesh = PolyCollection(right_triangles, facecolors='red', edgecolors='c', linewidths=0.05)
            ax.add_collection(left_mesh)
            ax.add_collection(right_mesh)
    
            ax.set_xlim(0, image.shape[1])
            ax.set_ylim(image.shape[0], 0)
            ax.set_aspect('equal')
            # plt.title("3D Mesh Projected on Image")
            plt.axis('off')
            fig.savefig(f"Day1_intro/resources/example/mesh/{num:06d}.png")
            plt.close()
    return (generate_all_frames,)


@app.cell
def _(generate_all_frames):
    generate_all_frames()
    return


if __name__ == "__main__":
    app.run()
