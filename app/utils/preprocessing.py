# app/utils/preprocessing.py
from torchvision import transforms

def get_transforms(image_size: int):
    """Return training and validation transforms."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform, transform
