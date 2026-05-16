import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class FaceDataset(Dataset):
    """Custom dataset for face images and attractiveness scores."""

    def __init__(self, df, root, transform, img_size):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = row["filename"]
        y = torch.tensor(float(row["score"]), dtype=torch.float32)
        path = os.path.join(self.root, name)

        try:
            img = Image.open(path).convert("RGB")
        except:
            img = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))

        if self.transform:
            img = self.transform(img)
        return img, y
