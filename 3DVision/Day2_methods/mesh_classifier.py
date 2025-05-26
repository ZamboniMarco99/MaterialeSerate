

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

    with open("Day2_methods/resources/gcn_layer.png", "rb") as f:
        gcn_layer_image = base64.b64encode(f.read()).decode()

    with open("Day2_methods/resources/example_graph.png", "rb") as f:
        example_graph_image = base64.b64encode(f.read()).decode()

    with open("Day2_methods/resources/convolution.jpg", "rb") as f:
        image_convolution_image = base64.b64encode(f.read()).decode()

    with open("Day2_methods/resources/neural_network.png", "rb") as f:
        nn_image = base64.b64encode(f.read()).decode()

    mesh_graph = r"""
        \[
        Mesh := G(V, E) \\
        V \in R^3 \\
        E \in V^2
        \]
    """

    function_in_primitives = r"""
        \[
        f(x, y) = x * (y + 2) \\
        x := x \\
        y := y \\
        z := 2 \\
        a := y + z \\
        b := x * a \\
        f := b
        \]
    """

    mo.md(
        f"""
        # Mesh classification

        Goal: Design a neural network able to ingest meshes and exploit their structural information
    
        ## Mesh <-> Graph
    
        * A mesh is a collection of adjacent polygons (often triangles); each vertex might be part of multiple polygons
        * Equivalent: a mesh is a collection of vertices and faces
        * The faces explicitly define the edges (between the vertices) of the polygons
        * We can view and represent the mesh as an undirect graph

        {mesh_graph}


        Goal: Design a neural network able to ingest graphs and exploit their structural information

        ## Getting inspired by images
        Images can be thought as a special subclass of graphs:
    
        * The pixels are the vertices
        * The neighborhood of a pixel defines the vertices connected to that pixel

        We know a way to use images in Neural Networks and exploit their structural informations: Convolutions

        What is a convolution actually doing? For every pixel it just collect information from its neighborhood (defined by the kernel) and outputs (to the next NN layer) a weighted average

        <img src="data:image/jpg;base64,{image_convolution_image}" width="500px" />

        Question: Can we generalize this concept to make it work on generic graphs?

        ## Revisiting Neural Networks

        <img src="data:image/png;base64,{nn_image}" width="500px" />
    
        A neural network is nothing more than a way to represent a generic parametric function $f_w(x)$. The function just needs to be differentiable in order to run gradient descent.

        In particular the NN is representing the function as a computational **graph** where each node is a primitive function that we know how to compute and how to differentiate

        {function_in_primitives}

        ...Wait, if a NN is a graph and we are trying to process graphs, can we exploit this duality?


        ## Message passing
        We can use the input graph as the computational graph of the NN!

        * The nodes that form a layer are the nodes of the input graph
        * The nodes of the input graph are duplicated and used multiple times, once for each layer
        * In the computational graph there exist an edge between node $X$ of layer $l$ and node $Y$ of layer $l+1$ iff there was an edge between node $X$ and node $Y$ in the input graph
        * We are free to choose the primitive function, but what we are doing is aggregating information from the neighborhood of each node, just like in convolution -> Graph Convolutional Networks, introduced in 2017 by Kipf and Welling

        Graph
        <img src="data:image/png;base64,{example_graph_image}" width="200px" />

        Graph convolutional layer
        <img src="data:image/png;base64,{gcn_layer_image}" width="250px" />
        """
    )
    return


@app.cell
def _():
    from utils.mesh_dataset import ModelNet10MeshDataset
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import MLFlowLogger
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.loader import DataLoader
    from torchmetrics.classification import Accuracy
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    return (
        Accuracy,
        DataLoader,
        F,
        GCNConv,
        MLFlowLogger,
        ModelNet10MeshDataset,
        Poly3DCollection,
        global_mean_pool,
        pl,
        plt,
        torch,
    )


@app.cell
def _():
    data_root_dir = "Day2_methods/data/ModelNet10"
    return (data_root_dir,)


@app.cell
def _(ModelNet10MeshDataset, data_root_dir):
    train_dataset = ModelNet10MeshDataset(root_dir=data_root_dir)
    test_dataset = ModelNet10MeshDataset(root_dir=data_root_dir, split="test")
    len(train_dataset), len(test_dataset)
    return test_dataset, train_dataset


@app.cell
def _(DataLoader, test_dataset, train_dataset):
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=7)
    val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=7)
    return train_loader, val_loader


@app.cell
def _(train_dataset):
    train_dataset.classes
    return


@app.cell
def _(train_dataset):
    example_mesh = train_dataset[1014]
    example_label = example_mesh.y
    example_vertices = example_mesh.pos
    example_faces = example_mesh.face.T
    print(example_vertices.shape, example_faces.shape, example_label)
    return example_faces, example_mesh, example_vertices


@app.cell
def _(example_mesh):
    example_mesh
    return


@app.cell
def _(Poly3DCollection, example_faces, example_vertices, plt):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each face as a polygon
    for face in example_faces:
        poly3d = [example_vertices[face]]
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=0.1, edgecolors='r', alpha=1))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Mesh')

    plt.show()
    return


@app.cell
def _(Accuracy, F, GCNConv, global_mean_pool, pl, torch):
    class MeshClassifier(pl.LightningModule):
        def __init__(self, num_classes=10):
            super(MeshClassifier, self).__init__()
            self.conv1 = GCNConv(3, 32)
            self.conv2 = GCNConv(32, 32)
            self.lin = torch.nn.Linear(32, num_classes)

            self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
            self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        def forward(self, x, edge_index, batch):
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = global_mean_pool(x, batch)
            x = self.lin(x)
            return x

        def training_step(self, batch, batch_idx):
            outputs = self.forward(batch.x, batch.edge_index, batch.batch)

            loss = F.cross_entropy(outputs, batch.y)
            self.train_accuracy(outputs, batch.y)
            self.log('train/loss', loss, on_step=True, on_epoch=True, batch_size=32)
            self.log('train/accuracy', self.train_accuracy, on_step=True, on_epoch=True, batch_size=32)
            return loss

        def validation_step(self, batch, batch_idx):
            outputs = self.forward(batch.x, batch.edge_index, batch.batch)

            loss = F.cross_entropy(outputs, batch.y)
            self.val_accuracy(outputs, batch.y)

            self.log('val/loss', loss, on_step=True, on_epoch=True, batch_size=32)
            self.log('val/accuracy', self.val_accuracy, on_step=True, on_epoch=True, batch_size=32)

            return {'val_loss': loss, 'val_accuracy': self.val_accuracy}

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.001)
    return (MeshClassifier,)


@app.cell
def _(MLFlowLogger, MeshClassifier, pl):
    mesh_model = MeshClassifier()

    logger = MLFlowLogger(experiment_name="meshes")

    trainer = pl.Trainer(logger=logger, max_epochs=20)  # Adjust for your system
    return mesh_model, trainer


@app.cell
def _(mesh_model, train_loader, trainer, val_loader):
    def train():
        trainer.fit(mesh_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return


@app.cell
def _():
    # train()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
