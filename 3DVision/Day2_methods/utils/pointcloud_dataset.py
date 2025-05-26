import os

import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset


class ModelNet10PointCloudDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (string): Directory with the ModelNet10_pointcloud1024 dataset
            split (string): 'train' or 'test'
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Get all class directories
        self.classes = sorted(
            [
                d
                for d in os.listdir(self.root_dir)
                if os.path.isdir(os.path.join(self.root_dir, d))
            ]
        )
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Load all file paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name, split)
            if not os.path.exists(class_dir):
                continue

            for file_name in os.listdir(class_dir):
                if file_name.endswith(".ply"):
                    self.samples.append(
                        {
                            "path": os.path.join(class_dir, file_name),
                            "label": self.class_to_idx[class_name],
                        }
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_path = self.samples[idx]["path"]
        label = self.samples[idx]["label"]

        # Load point cloud data using trimesh
        mesh = trimesh.load(sample_path)
        points = np.array(mesh.vertices)

        # Convert to torch tensor
        points = torch.from_numpy(points).float()
        points = points.transpose(1, 0)

        if self.transform:
            points = self.transform(points)

        return points, label
