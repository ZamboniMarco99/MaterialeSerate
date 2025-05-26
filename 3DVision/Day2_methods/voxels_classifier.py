

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    from utils.voxel_dataset import ModelNet10VoxelDataset
    import torch
    from torch.utils.data import DataLoader
    import torch.nn as nn
    import torch.nn.functional as F
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import MLFlowLogger
    from torchmetrics.classification import Accuracy
    import numpy as np
    import matplotlib.pyplot as plt
    return (
        Accuracy,
        DataLoader,
        MLFlowLogger,
        ModelNet10VoxelDataset,
        nn,
        pl,
        plt,
        torch,
    )


@app.cell
def _():
    data_root_dir = "Day2_methods/data/ModelNet10_voxels32"
    return (data_root_dir,)


@app.cell
def _(ModelNet10VoxelDataset, data_root_dir):
    train_dataset = ModelNet10VoxelDataset(root_dir=data_root_dir)
    test_dataset = ModelNet10VoxelDataset(root_dir=data_root_dir, split="test")
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
    example_voxel, example_label = train_dataset[0]
    example_voxel = example_voxel[0]
    example_voxel.shape, example_label
    return (example_voxel,)


@app.cell
def _(example_voxel, plt):
    # Get the coordinates of occupied voxels
    occupied = example_voxel.nonzero()
    x_occupied, y_occupied, z_occupied = occupied[:, 0], occupied[:, 1], occupied[:, 2]

    # # Get the RGB values corresponding to these voxels
    colors_occupied = example_voxel[x_occupied, y_occupied, z_occupied]

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
    ax_occupied.set_title("Voxel Cube")

    fig_occupied

    return


@app.cell
def _(Accuracy, nn, pl, torch):
    class VoxelClassifier3D(pl.LightningModule):
        def __init__(self, num_classes=10, lr=1e-3):
            super().__init__()
            self.save_hyperparameters()

            self.convs = nn.Sequential(
                nn.Conv3d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm3d(16),
                nn.ReLU(),
                nn.Conv3d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.Conv3d(32, 128, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.max_pool = nn.AdaptiveMaxPool3d(1)  # Output: (B, 128, 1)

            self.fc = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes),
            )

            self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
            self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)


            self.loss_fn = nn.CrossEntropyLoss()

        def forward(self, x):
            x = self.convs(x)
            x = self.max_pool(x).view(x.size(0), -1)  # Flatten to (B, 512)
            # print(x.shape)
            return self.fc(x)

        def training_step(self, batch, batch_idx):
            voxels, labels = batch  # x: (B, 1, 32, 32, 32), y: (B, 32, 32, 32)
            outputs = self(voxels)
            loss = self.loss_fn(outputs, labels)

            self.train_accuracy(outputs, labels)
            self.log('train/loss', loss, on_step=True, on_epoch=True)
            self.log('train/accuracy', self.train_accuracy, on_step=True, on_epoch=True)
            return loss

        def validation_step(self, batch, batch_idx):
            voxels, labels = batch
            outputs = self(voxels)
            loss = self.loss_fn(outputs, labels)
        
            self.val_accuracy(outputs, labels)
            self.log('val/loss', loss, on_step=True, on_epoch=True)
            self.log('val/accuracy', self.val_accuracy, on_step=True, on_epoch=True)

            return {'val_loss': loss, 'val_accuracy': self.val_accuracy}

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    return (VoxelClassifier3D,)


@app.cell
def _(MLFlowLogger, VoxelClassifier3D, pl):
    voxel_model = VoxelClassifier3D()

    logger = MLFlowLogger(experiment_name="voxel")

    trainer = pl.Trainer(logger=logger, max_epochs=20)  # Adjust for your system
    return trainer, voxel_model


@app.cell
def _(train_loader, trainer, val_loader, voxel_model):
    def train():
        trainer.fit(voxel_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return


@app.cell
def _(example_voxel):
    example_voxel.shape
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
