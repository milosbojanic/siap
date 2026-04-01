import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int) -> A.Compose:
    """
    Augmentation pipeline for training.
    Includes geometric + color augmentations to improve generalization.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.1, rotate_limit=15,
            border_mode=0, p=0.5,
        ),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int) -> A.Compose:
    """Validation/test transforms: resize + normalize only."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
