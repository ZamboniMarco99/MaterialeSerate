

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from utils.pointcloud_dataset import ModelNet10PointCloudDataset
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import torch.nn.functional as F
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import MLFlowLogger
    from torchmetrics.classification import Accuracy
    import numpy as np
    import matplotlib.pyplot as plt
    return (
        Accuracy,
        DataLoader,
        F,
        MLFlowLogger,
        ModelNet10PointCloudDataset,
        nn,
        pl,
        plt,
        torch,
    )


@app.cell
def _():
    data_root_dir = "Day2_methods/data/ModelNet10_pointcloud1024"
    return (data_root_dir,)


@app.cell
def _(ModelNet10PointCloudDataset, data_root_dir):
    train_dataset = ModelNet10PointCloudDataset(root_dir=data_root_dir)
    test_dataset = ModelNet10PointCloudDataset(root_dir=data_root_dir, split="test")
    len(train_dataset), len(test_dataset)
    return test_dataset, train_dataset


@app.cell
def _(DataLoader, test_dataset, train_dataset):
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, val_loader


@app.cell
def _(train_dataset):
    train_dataset.classes
    return


@app.cell
def _(train_dataset):
    example_points, example_label = train_dataset[0]
    example_points = example_points.transpose(1, 0)
    example_points.shape, example_label
    return (example_points,)


@app.cell
def _(example_points, plt):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(example_points[:, 0], example_points[:, 1], example_points[:, 2], s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig
    return


@app.cell
def _(nn):
    class PointNet(nn.Module):
        def __init__(self, num_classes=10, input_dim=3):
            super(PointNet, self).__init__()

            # PointNet Shared MLP layers
            self.mlp1 = nn.Sequential(
                nn.Conv1d(input_dim, 64, 1),  # Input: (B, 3, N), Output: (B, 64, N)
                nn.ReLU(),
                nn.Conv1d(64, 128, 1),  # Output: (B, 128, N)
                nn.ReLU(),
                nn.Conv1d(128, 1024, 1),  # Output: (B, 1024, N)
                nn.ReLU(),
            )

            # Global feature max pooling
            self.max_pool = nn.AdaptiveMaxPool1d(1)  # Output: (B, 1024, 1)

            self.fc = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):

            # Step 2: Shared MLP for per-point feature extraction
            x = self.mlp1(x)

            # Step 3: Global max pooling (to get the global feature)
            x = self.max_pool(x)

            x = x.view(x.size(0), -1)  # Flatten (B, 1024)

            # Step 4: Classification head (fully connected layers)
            out = self.fc(x)
            return out
    return (PointNet,)


@app.cell
def _(Accuracy, PointNet, nn, pl, torch):
    class PointNetLightning(pl.LightningModule):
        def __init__(self, num_classes=10):
            super(PointNetLightning, self).__init__()
            self.model = PointNet(num_classes=num_classes)
            self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
            self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            points, labels = batch
            outputs = self(points)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            self.train_accuracy(outputs, labels)

            self.log('train/loss', loss, on_step=True, on_epoch=True)
            self.log('train/accuracy', self.train_accuracy, on_step=True, on_epoch=True)
            return loss

        def validation_step(self, batch, batch_idx):
            points, labels = batch
            outputs = self(points)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            self.val_accuracy(outputs, labels)

            self.log('val/loss', loss, on_step=True, on_epoch=True)
            self.log('val/accuracy', self.val_accuracy, on_step=True, on_epoch=True)

            return {'val_loss': loss, 'val_accuracy': self.val_accuracy}

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            return optimizer
    return (PointNetLightning,)


@app.cell
def _(MLFlowLogger, PointNetLightning, pl):
    pointnet_model = PointNetLightning()

    logger = MLFlowLogger(experiment_name="pointnet")

    trainer = pl.Trainer(logger=logger, max_epochs=20)  # Adjust for your system
    return pointnet_model, trainer


@app.cell
def _(pointnet_model, train_loader, trainer, val_loader):
    def train():
        trainer.fit(pointnet_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return


@app.cell
def _():
    # train()
    return


@app.cell
def _(mo):
    mo.md(r"""# Predict a Transformation matrix""")
    return


@app.cell
def _(F, nn):
    class TNet(nn.Module):
        def __init__(self, input_dim=3):
            super(TNet, self).__init__()
            # Define the layers of the T-Net
            self.mlp = nn.Sequential(
                nn.Conv1d(input_dim, 64, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(128, 1024, kernel_size=1),
                nn.ReLU()
            )

            self.fc = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 9)  # Output 3x3 transformation matrix
            )

        def forward(self, x):
            # Apply MLP layers
            x = self.mlp(x)
            # Global feature extraction via max pooling
            x = F.max_pool1d(x, x.size(2))  # [batch_size, 1024, 1]
            x = x.view(-1, 1024)  # Flatten to [batch_size, 1024]

            # Apply the fully connected layers
            transformation_matrix = self.fc(x)  # [batch_size, 9]

            # Reshape to [batch_size, 3, 3]
            transformation_matrix = transformation_matrix.view(-1, 3, 3)
            return transformation_matrix
    return (TNet,)


@app.cell
def _(Accuracy, PointNet, TNet, nn, pl, torch):
    class PointNetTransformedLightning(pl.LightningModule):
        def __init__(self, num_classes=10):
            super(PointNetTransformedLightning, self).__init__()
            self.t_net = TNet()
            self.model = PointNet(num_classes=num_classes)
            self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
            self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        def forward(self, x):
            transformation_matrix = self.t_net(x)
            x = torch.bmm(transformation_matrix, x)
            return self.model(x)

        def training_step(self, batch, batch_idx):
            points, labels = batch
            outputs = self(points)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            self.train_accuracy(outputs, labels)

            self.log('train/loss', loss, on_step=True, on_epoch=True)
            self.log('train/accuracy', self.train_accuracy, on_step=True, on_epoch=True)
            return loss

        def validation_step(self, batch, batch_idx):
            points, labels = batch
            outputs = self(points)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            self.val_accuracy(outputs, labels)

            self.log('val/loss', loss, on_step=True, on_epoch=True)
            self.log('val/accuracy', self.val_accuracy, on_step=True, on_epoch=True)

            return {'val_loss': loss, 'val_accuracy': self.val_accuracy}

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            return optimizer
    return (PointNetTransformedLightning,)


@app.cell
def _(MLFlowLogger, PointNetTransformedLightning, pl):
    pointnet_transformed_model = PointNetTransformedLightning()

    logger_transformed = MLFlowLogger(experiment_name="pointnet_transformed")

    trainer_transformed = pl.Trainer(logger=logger_transformed, max_epochs=20)  # Adjust for your system
    return pointnet_transformed_model, trainer_transformed


@app.cell
def _(
    pointnet_transformed_model,
    train_loader,
    trainer_transformed,
    val_loader,
):
    def train_transformed():
        trainer_transformed.fit(pointnet_transformed_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return


@app.cell
def _():
    # train_transformed()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
