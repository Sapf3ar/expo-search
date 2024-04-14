from abc import ABC, abstractmethod
import albumentations as A


def train_augmentations() -> A.Compose:
    return A.Compose(
        [
            A.RandomRotate90(),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            A.GaussNoise(p=0.6),
            A.ShiftScaleRotate(p=0.5),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1)
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.RandomBrightnessContrast(p=0.8),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
            A.Resize(512, 512, p=1.0),
            A.Normalize((0.485, 0.456, 0.406), p=1.0)
        ]
    )

def val_augmentations() -> A.Compose:
    return A.Compose(
        [
            A.Resize(512, 512, p=1.0),
            A.Normalize((0.485, 0.456, 0.406), p=1.0)
        ]
    )

def test_augmentations() -> A.Compose:
    return A.Compose(
        [
            A.Resize(512, 512, p=1.0),
            A.Normalize((0.485, 0.456, 0.406), p=1.0)
        ]
    )
