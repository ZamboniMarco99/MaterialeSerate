import os

import torch
import trimesh
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms.face_to_edge import FaceToEdge


class ModelNet10MeshDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (string): Directory with the ModelNet10 dataset
            split (string): 'train' or 'test'
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        super().__init__(root_dir, transform)

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
                if file_name.endswith(".off"):
                    self.samples.append(
                        {
                            "path": os.path.join(class_dir, file_name),
                            "label": self.class_to_idx[class_name],
                        }
                    )

    def len(self):
        return len(self.samples)

    def get(self, idx):
        sample_path = self.samples[idx]["path"]
        label = self.samples[idx]["label"]

        # Load mesh data using trimesh
        mesh = trimesh.load(sample_path)

        # Get vertices and faces
        vertices = torch.tensor(mesh.vertices, dtype=torch.float)
        faces = torch.tensor(
            mesh.faces, dtype=torch.long
        ).t()  # Transpose faces to match PyG format

        face_to_edge = FaceToEdge(remove_faces=False)
        data = Data(
            pos=vertices,
            x=vertices,  # Adding x as the 3D coordinates
            face=faces,
            y=torch.tensor(label, dtype=torch.long),
        )

        data = face_to_edge(data)

        if self.transform is not None:
            data = self.transform(data)

        return data
