# datasets.py
# -*- coding: utf-8 -*-
import os
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class CustomTransforms:
    @staticmethod
    def RandomHorizontalFlip(sample: Dict) -> Dict:
        if torch.rand(1) > 0.5:
            sample["image"] = sample["image"].transpose(Image.FLIP_LEFT_RIGHT)
            for key in ["label", "label_a", "label_b"]:
                if key in sample and sample[key] is not None:
                    sample[key] = sample[key].transpose(Image.FLIP_LEFT_RIGHT)
        return sample

    @staticmethod
    def RandomGaussianBlur(sample: Dict) -> Dict:
        return sample

    @staticmethod
    def Normalize(sample: Dict) -> Dict:
        sample["image"] = transforms.ToTensor()(sample["image"])
        return sample

    @staticmethod
    def RGBToClassIndex(mask: Image.Image) -> np.ndarray:
        """Convert RGB mask to class index map."""
        palette = {
            (255, 255, 255): 0,  # Background
            (255, 0, 0): 1,      # TUM
            (0, 255, 0): 2,      # STR
            (0, 0, 255): 3,      # LYM
            (255, 0, 255): 4     # NEC
        }
        mask_np = np.array(mask)
        class_mask = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.int64)
        for rgb, class_idx in palette.items():
            class_mask[np.all(mask_np == rgb, axis=-1)] = class_idx
        return class_mask

    @staticmethod
    def ToTensor(sample: Dict) -> Dict:
        sample["image"] = transforms.ToTensor()(sample["image"])
        for key in ["label", "label_a", "label_b"]:
            if key in sample and sample[key] is not None:
                class_mask = CustomTransforms.RGBToClassIndex(sample[key])
                sample[key] = torch.tensor(class_mask, dtype=torch.long)
        return sample

class BaseImageDataset(Dataset):
    def __init__(self, data_path: str, transform: Optional[Any] = None):
        self.data_path = Path(data_path)
        self.transform = transform
        self.items = self._load_items()
        if not self.items:
            raise ValueError(f"No valid items found in {data_path}")

    def _load_items(self) -> List[Any]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict:
        raise NotImplementedError

class Stage1InferenceDataset(BaseImageDataset):
    def _load_items(self) -> List[Path]:
        image_extensions = (".png", ".jpg", ".jpeg")
        return [
            p for p in self.data_path.rglob("*")
            if p.is_file() and p.suffix.lower() in image_extensions
        ]

    def __getitem__(self, index: int) -> Dict:
        image_path = self.items[index]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")

        sample = {"image_id": image_path.stem, "image": image}
        if self.transform:
            sample = self.transform(sample)
        return sample

class Stage1TrainDataset(BaseImageDataset):
    def __init__(self, data_path: str, dataset: str, transform: Optional[Any] = None):
        self.dataset = dataset
        if dataset not in ["luad", "bcss"]:
            raise ValueError(f"Unsupported dataset: {dataset}")
        super().__init__(data_path, transform)

    def _load_items(self) -> List[Tuple[Path, torch.Tensor]]:
        items = []
        image_extensions = (".png", ".jpg", ".jpeg")
        for path in self.data_path.rglob("*"):
            if path.is_file() and path.suffix.lower() in image_extensions:
                label = self._extract_label(path.stem)
                if label is not None:
                    items.append((path, label))
        return items

    def _extract_label(self, fname: str) -> Optional[torch.Tensor]:
        try:
            label_str = fname.split("]")[0].split("[")[-1]
            if self.dataset == "luad":
                # Space-separated: [a b c d]
                label = torch.tensor([int(x) for x in label_str.split()])
            elif self.dataset == "bcss":
                # Example: [abcd] -> [int(a), int(b), int(c), int(d)]
                label = torch.tensor([int(label_str[0]), int(label_str[1]), 
                                    int(label_str[2]), int(label_str[3])])
            return label
        except (IndexError, ValueError):
            print(f"Warning: Invalid label format in {fname}")
            return None

    def __getitem__(self, index: int) -> Dict:
        image_path, label = self.items[index]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")

        sample = {"image_id": image_path.stem, "image": image, "label": label}
        if self.transform:
            sample = self.transform(sample)
        return sample

class Stage2Dataset(BaseImageDataset):
    def __init__(self, base_dir: str, split: str, transform: Optional[Any] = None):
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}")
        self.split = split
        self.dirs = self._set_directories(base_dir, split)
        super().__init__(self.dirs["image_dir"], transform)
        print(f"Number of images in {split}: {len(self.items)}")

    def _set_directories(self, base_dir: str, split: str) -> Dict[str, Path]:
        base_dir = Path(base_dir)
        if split == "train":
            return {
                "image_dir": base_dir / "train",
                "mask_dir": base_dir / "train_PM" / "PM_bn7",
                "mask_dir_a": base_dir / "train_PM" / "PM_b5_2",
                "mask_dir_b": base_dir / "train_PM" / "PM_b4_5",
            }
        elif split == "val":
            return {
                "image_dir": base_dir / "val" / "img",
                "mask_dir": base_dir / "val" / "mask",
            }
        else:  # test
            return {
                "image_dir": base_dir / "test" / "img",
                "mask_dir": base_dir / "test" / "mask",
            }

    def _load_items(self) -> List[Dict]:
        items = []
        image_extensions = (".png", ".jpg", ".jpeg")
        for fname in os.listdir(self.dirs["image_dir"]):
            if fname.startswith(".") or not fname.lower().endswith(image_extensions):
                continue
            image_path = self.dirs["image_dir"] / fname
            mask_path = self.dirs["mask_dir"] / fname
            item = {"image_path": image_path, "mask_path": mask_path}
            if self.split == "train":
                item["mask_path_a"] = self.dirs["mask_dir_a"] / fname
                item["mask_path_b"] = self.dirs["mask_dir_b"] / fname
            items.append(item)
        return items

    def __getitem__(self, index: int) -> Dict:
        item = self.items[index]
        try:
            image = Image.open(item["image_path"]).convert("RGB")
            mask = Image.open(item["mask_path"]) if item["mask_path"].exists() else None
            sample = {"image": image, "label": mask}
            if self.split == "train":
                sample["label_a"] = Image.open(item["mask_path_a"]) if item["mask_path_a"].exists() else None
                sample["label_b"] = Image.open(item["mask_path_b"]) if item["mask_path_b"].exists() else None
            elif self.split in ["val", "test"]:
                sample["image_path"] = str(item["image_path"])
        except Exception as e:
            raise RuntimeError(f"Failed to load item {item}: {e}")

        if self.transform:
            sample = self.transform(sample)
        return sample

def get_transform(split: str) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            CustomTransforms.RandomHorizontalFlip,
            CustomTransforms.RandomGaussianBlur,
            CustomTransforms.Normalize,
            CustomTransforms.ToTensor,
        ])
    else:  # val or test
        return transforms.Compose([
            CustomTransforms.Normalize,
            CustomTransforms.ToTensor,
        ])

def create_dataloaders(
    dataroot: str,
    dataset: str,
    batch_size: int,
    num_workers: int = 4,
    stage: str = "stage2"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if stage == "stage1":
        train_dataset = Stage1TrainDataset(
            data_path=os.path.join(dataroot, "train"),
            dataset=dataset,
            transform=get_transform("train")
        )
        val_dataset = Stage1InferenceDataset(
            data_path=os.path.join(dataroot, "val"),
            transform=get_transform("val")
        )
        test_dataset = Stage1InferenceDataset(
            data_path=os.path.join(dataroot, "test"),
            transform=get_transform("val")
        )
    else:  # stage2
        train_dataset = Stage2Dataset(
            base_dir=dataroot,
            split="train",
            transform=get_transform("train")
        )
        val_dataset = Stage2Dataset(
            base_dir=dataroot,
            split="val",
            transform=get_transform("val")
        )
        test_dataset = Stage2Dataset(
            base_dir=dataroot,
            split="test",
            transform=get_transform("val")
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader