# utils/preprocessing.py
from torchvision import transforms

class Preprocessor:
    """
    Utility class to create image preprocessing pipelines
    for training, validation, and testing.
    """

    def __init__(self, image_size=300):
        """
        Initialize the Preprocessor with a specific image size.
        Args:
            image_size (int): Target size for image transformations.
        """
        self.image_size = image_size

    def get_train_transforms(self):
        """Return transformation pipeline for training data."""
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomResizedCrop(self.image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def get_val_transforms(self):
        """Return transformation pipeline for validation or test data."""
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def get_all(self):
        """Return both train and validation transforms."""
        return self.get_train_transforms(), self.get_val_transforms()
