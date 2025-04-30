# datasets.py
# -*- coding: utf-8 -*-
import os
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import torch
import re 
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random 

#=================Start: Transforms=================
class CustomTransforms:
    class RGBToClassIndex:
        """Convert an RGB mask to a class index tensor for BCSS dataset."""
        def __init__(self):
            # Define RGB to class index mapping for BCSS dataset
            self.rgb_to_class = {
                (0, 0, 0): 0,      # Background
                (255, 0, 0): 1,    # Tumor (TUM)
                (0, 255, 0): 2,    # Stroma (STR)
                (0, 0, 255): 3,    # Lymphocytes (LYM)
                (255, 255, 0): 4,  # Necrosis (NEC)
            }
            # Default class for unmapped colors (e.g., "Others")
            self.default_class = 0

        def __call__(self, mask: Image.Image) -> torch.Tensor:
            # Convert PIL Image to NumPy array
            mask_np = np.array(mask, dtype=np.uint8)  # Shape: [H, W, 3]
            
            # Initialize class index array
            class_indices = np.full(mask_np.shape[:2], self.default_class, dtype=np.int64)  # Shape: [H, W]
            
            # Map RGB values to class indices
            for rgb, class_idx in self.rgb_to_class.items():
                # Create a mask where the RGB values match
                rgb_match = np.all(mask_np == rgb, axis=-1)
                class_indices[rgb_match] = class_idx
            
            # Convert to PyTorch tensor
            class_tensor = torch.from_numpy(class_indices).long()  # Shape: [H, W], dtype=torch.long
            return class_tensor

    @staticmethod
    def ToTensor(sample):
        """Convert masks to class index tensors and optionally handle images."""
        if "image" in sample and sample["image"] is not None:
            sample["image"] = transforms.ToTensor()(sample["image"])
        
        # Convert masks to class index tensors using RGBToClassIndex
        rgb_to_class = CustomTransforms.RGBToClassIndex()
        if "label" in sample and sample["label"] is not None:
            sample["label"] = rgb_to_class(sample["label"])  # Convert RGB mask to class indices
        if "label_a" in sample and sample["label_a"] is not None:
            sample["label_a"] = rgb_to_class(sample["label_a"])
        if "label_b" in sample and sample["label_b"] is not None:
            sample["label_b"] = rgb_to_class(sample["label_b"])
        
        return sample

    @staticmethod
    def Normalize(sample):
        if "image" in sample and sample["image"] is not None:
            sample["image"] = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )(sample["image"])
        return sample

    @staticmethod
    def RandomHorizontalFlip(sample):
        if random.random() > 0.5:
            if "image" in sample and sample["image"] is not None:
                sample["image"] = transforms.RandomHorizontalFlip(p=1.0)(sample["image"])
            if "label" in sample and sample["label"] is not None:
                sample["label"] = transforms.RandomHorizontalFlip(p=1.0)(sample["label"])
            if "label_a" in sample and sample["label_a"] is not None:
                sample["label_a"] = transforms.RandomHorizontalFlip(p=1.0)(sample["label_a"])
            if "label_b" in sample and sample["label_b"] is not None:
                sample["label_b"] = transforms.RandomHorizontalFlip(p=1.0)(sample["label_b"])
        return sample

    @staticmethod
    def RandomGaussianBlur(sample):
        if random.random() > 0.5:
            if "image" in sample and sample["image"] is not None:
                sample["image"] = transforms.GaussianBlur(kernel_size=3)(sample["image"])
        return sample

def get_transform(split: str) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            CustomTransforms.RandomHorizontalFlip,
            CustomTransforms.RandomGaussianBlur,
            CustomTransforms.ToTensor,
            CustomTransforms.Normalize,
        ])
    else:  # val or test
        return transforms.Compose([
            CustomTransforms.ToTensor,
            CustomTransforms.Normalize,
        ])

def get_label_transform(split: str) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            CustomTransforms.RandomHorizontalFlip,
            CustomTransforms.ToTensor,  # Only transforms masks
        ])
    else:  # val or test
        return transforms.Compose([
            CustomTransforms.ToTensor,  # Only transforms masks
        ])
#=================End: Transforms=================

# ... (rest of the code unchanged, including Stage2IndiceDataset) 

# class CustomTransforms:
#     @staticmethod
#     def RandomHorizontalFlip(sample: Dict) -> Dict:
#         if torch.rand(1) > 0.5:
#             sample["image"] = sample["image"].transpose(Image.FLIP_LEFT_RIGHT)
#             for key in ["label", "label_a", "label_b"]:
#                 if key in sample and sample[key] is not None:
#                     sample[key] = sample[key].transpose(Image.FLIP_LEFT_RIGHT)
#         return sample

#     @staticmethod
#     def RandomGaussianBlur(sample: Dict) -> Dict:
#         if torch.rand(1) > 0.5:
#             blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
#             sample["image"] = blur(sample["image"])
#         return sample

#     @staticmethod
#     def RGBToClassIndex(mask: Image.Image) -> np.ndarray:
#         """Convert mask to class index map (handles both RGB and indexed masks)."""
#         mask_np = np.array(mask)

#         # Check if mask is single-channel (indexed)
#         if mask_np.ndim == 2:
#             # Assume values are already class indices (0 to 4)
#             class_mask = mask_np.astype(np.int64)
#             # Validate class indices
#             valid_classes = set(range(5))  # Classes 0 to 4
#             unique_classes = set(np.unique(class_mask))
#             if not unique_classes.issubset(valid_classes):
#                 raise ValueError(f"Invalid class indices in mask: {unique_classes}")
#             return class_mask

#         # Otherwise, assume RGB mask
#         palette = {
#             (255, 255, 255): 0,  # Background
#             (255, 0, 0): 1,      # TUM
#             (0, 255, 0): 2,      # STR
#             (0, 0, 255): 3,      # LYM
#             (255, 0, 255): 4     # NEC
#         }
#         if mask_np.shape[-1] != 3:
#             raise ValueError(f"Expected RGB mask with shape (H, W, 3), got shape {mask_np.shape}")
#         class_mask = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.int64)
#         for rgb, class_idx in palette.items():
#             class_mask[np.all(mask_np == rgb, axis=-1)] = class_idx
#         return class_mask

#     @staticmethod
#     def ToTensor(sample: Dict) -> Dict:
#         """Convert image and labels to tensors."""
#         sample["image"] = transforms.ToTensor()(sample["image"])
#         for key in ["label", "label_a", "label_b"]:
#             if key in sample and sample[key] is not None:
#                 class_mask = CustomTransforms.RGBToClassIndex(sample[key])
#                 sample[key] = torch.tensor(class_mask, dtype=torch.long)
#         return sample

#     @staticmethod
#     def Normalize(sample: Dict) -> Dict:
#         """Apply normalization to the image tensor."""
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                                        std=[0.229, 0.224, 0.225])
#         sample["image"] = normalize(sample["image"])
#         return sample

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
                label = torch.tensor([int(x) for x in label_str.split()])
            elif self.dataset == "bcss":
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
    def __init__(self, base_dir: str, split: str, dataset: str, transform: Optional[Any] = None):
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}")
        if dataset not in ["luad", "bcss"]:
            raise ValueError(f"Unsupported dataset: {dataset}")
        self.split = split
        self.dataset = dataset  # Add dataset parameter
        self.dirs = self._set_directories(base_dir, split)
        super().__init__(self.dirs["image_dir"], transform)
        print(f"Number of images in {split}: {len(self.items)}")

    def _set_directories(self, base_dir: str, split: str) -> Dict[str, Path]:
        base_dir = Path(base_dir)
        # 👇 new line: ensure masks always come from the "_organized" version
        organized_dir = Path(str(base_dir).replace("_indice", "_organized"))

        if split == "train":
            return {
                "image_dir": base_dir / "train",
                "mask_dir": organized_dir / "train_PM" / "PM_bn7",
                "mask_dir_a": organized_dir / "train_PM" / "PM_b5_2",
                "mask_dir_b": organized_dir / "train_PM" / "PM_b4_5",
            }
        elif split == "val":
            return {
                "image_dir": base_dir / "val" / "img",
                "mask_dir": organized_dir / "val" / "mask",
            }
        else:  # test
            return {
                "image_dir": base_dir / "test" / "img",
                "mask_dir": organized_dir / "test" / "mask",
            } 
            
    def _extract_label(self, fname: str) -> Optional[torch.Tensor]:
        """Extract image-level label from filename for BCSS or LUAD dataset."""
        try:
            label_str = fname.split("]")[0].split("[")[-1]  # e.g., "0100" for BCSS, "0 1 0 1" for LUAD
            if self.dataset == "luad":
                # Space-separated: [a b c d]
                label = torch.tensor([int(x) for x in label_str.split()])
            elif self.dataset == "bcss":
                # No spaces: [abcd]
                label = torch.tensor([int(label_str[0]), int(label_str[1]), 
                                    int(label_str[2]), int(label_str[3])])
            return label
        except (IndexError, ValueError):
            print(f"Warning: Invalid label format in {fname}")
            return None

    def _load_items(self) -> List[Dict]:
        items = []
        image_extensions = (".png", ".jpg", ".jpeg")
        for fname in os.listdir(self.dirs["image_dir"]):
            if fname.startswith(".") or not fname.lower().endswith(image_extensions):
                continue
            image_path = self.dirs["image_dir"] / fname
            mask_path = self.dirs["mask_dir"] / fname
            if not mask_path.exists():
                print(f"Warning: Mask not found for {image_path}, skipping...")
                print(f"Mask lookup: {mask_path}") 
                continue
            item = {"image_path": image_path, "mask_path": mask_path}
            if self.split == "train":
                mask_path_a = self.dirs["mask_dir_a"] / fname
                mask_path_b = self.dirs["mask_dir_b"] / fname
                if not mask_path_a.exists() or not mask_path_b.exists():
                    print(f"Warning: Additional masks not found for {image_path}, skipping...")
                    continue
                item["mask_path_a"] = mask_path_a
                item["mask_path_b"] = mask_path_b
                # Extract classification label for train split
                label = self._extract_label(fname)
                if label is not None:
                    item["class_label"] = label
                else:
                    continue  # Skip items with invalid labels
            items.append(item)
        return items
    def __getitem__(self, index: int) -> Dict:
        item = self.items[index]
        try:
            image = Image.open(item["image_path"]).convert("RGB")
            mask = Image.open(item["mask_path"]) if item["mask_path"].exists() else None
            sample = {
                "image": image,
                "label": mask,
                "image_path": str(item["image_path"])  # ✅ always include this
            }
            if self.split == "train":
                sample["label_a"] = Image.open(item["mask_path_a"]) if item["mask_path_a"].exists() else None
                sample["label_b"] = Image.open(item["mask_path_b"]) if item["mask_path_b"].exists() else None
                sample["class_label"] = item["class_label"]
        except Exception as e:
            raise RuntimeError(f"Failed to load item {item}: {e}")

        if self.transform:
            sample = self.transform(sample)
        return sample 
    
    # def __getitem__(self, index: int) -> Dict:
    #     item = self.items[index]
    #     try:
    #         image = Image.open(item["image_path"]).convert("RGB")
    #         mask = Image.open(item["mask_path"]) if item["mask_path"].exists() else None
    #         sample = {"image": image, "label": mask}
    #         if self.split == "train":
    #             sample["label_a"] = Image.open(item["mask_path_a"]) if item["mask_path_a"].exists() else None
    #             sample["label_b"] = Image.open(item["mask_path_b"]) if item["mask_path_b"].exists() else None
    #             sample["class_label"] = item["class_label"]  # Add classification label
    #         elif self.split in ["val", "test"]:
    #             sample["image_path"] = str(item["image_path"])
    #     except Exception as e:
    #         raise RuntimeError(f"Failed to load item {item}: {e}")

    #     if self.transform:
    #         sample = self.transform(sample)
    #     return sample 
    
def get_transform(split: str) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            CustomTransforms.RandomHorizontalFlip,
            CustomTransforms.RandomGaussianBlur,
            CustomTransforms.ToTensor,
            CustomTransforms.Normalize,
        ])
    else:  # val or test
        return transforms.Compose([
            CustomTransforms.ToTensor,
            CustomTransforms.Normalize,
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
            dataset=dataset,  # Pass dataset argument
            transform=get_transform("train")
        )
        val_dataset = Stage2Dataset(
            base_dir=dataroot,
            split="val",
            dataset=dataset,  # Pass dataset argument
            transform=get_transform("val")
        )
        test_dataset = Stage2Dataset(
            base_dir=dataroot,
            split="test",
            dataset=dataset,  # Pass dataset argument
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

def get_label_transform(split: str) -> transforms.Compose:
    # Transform for labels only (used by Stage2IndiceDataset)
    if split == "train":
        return transforms.Compose([
            CustomTransforms.ToTensor,  # Converts labels to tensors
        ])
    else:  # val or test
        return transforms.Compose([
            CustomTransforms.ToTensor,
        ])

import re

#=============Indice Dataloaders=================
class Stage2IndiceDataset(Dataset):
    def __init__(self, indices_base: str, masks_base: str, split: str, dataset: str):
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}")
        if dataset not in ["luad", "bcss"]:
            raise ValueError(f"Unsupported dataset: {dataset}")
        self.split = split
        self.dataset = dataset
        self.indices_base = indices_base  # Store for image_path derivation
        self.masks_base = masks_base      # Store for image_path derivation
        self.dirs = self._set_directories(indices_base, masks_base, split)
        self.label_transform = get_label_transform(split)  # Transform for labels
        self.items = self._load_items()
        print(f"Number of items in {split}: {len(self.items)}")

    def _set_directories(self, indices_base: str, masks_base: str, split: str) -> Dict[str, Path]:
        indices_base = Path(indices_base)
        masks_base = Path(masks_base)

        if split == "train":
            return {
                "indices_dir": indices_base / "train" / "indices",
                "mask_dir": masks_base / "train_PM" / "PM_bn7",
                "mask_dir_a": masks_base / "train_PM" / "PM_b5_2",
                "mask_dir_b": masks_base / "train_PM" / "PM_b4_5",
            }
        elif split == "val":
            return {
                "indices_dir": indices_base / "val" / "indices",
                "mask_dir": masks_base / "val" / "mask",
            }
        else:  # test
            return {
                "indices_dir": indices_base / "test" / "indices",
                "mask_dir": masks_base / "test" / "masks",  # Updated to test/masks
            }

    def _extract_label(self, fname: str) -> Optional[torch.Tensor]:
        # Extract label from filename, e.g., "[0101].pt"
        try:
            label_str = fname.split("]")[0].split("[")[-1]
            if self.dataset == "luad":
                label = torch.tensor([int(x) for x in label_str.split()])
            elif self.dataset == "bcss":
                label = torch.tensor([int(label_str[0]), int(label_str[1]), 
                                    int(label_str[2]), int(label_str[3])])
            return label
        except (IndexError, ValueError):
            print(f"Warning: Invalid label format in {fname}")
            return None

    def _load_items(self) -> List[Dict]:
        items = []
        for fname in sorted(os.listdir(self.dirs["indices_dir"])):
            if not fname.endswith(".pt"):
                continue

            indices_path = self.dirs["indices_dir"] / fname
            item = {"indices_path": indices_path}

            if self.split in ["val", "test"]:
                # Simply remove .pt and add .png, no embedded label in these filenames
                base_name = fname.replace(".pt", ".png")
                mask_path = self.dirs["mask_dir"] / base_name

                print(f"PT: {fname}")
                print(f"Mask lookup: {mask_path}")

                if not mask_path.exists():
                    print(f"Warning: Mask not found for {indices_path}, skipping...")
                    continue

                item["mask_path"] = mask_path

            elif self.split == "train":
                # Expect format like ...[0101].pt → extract classification label
                match = re.search(r"\[(\d{4})\]", fname)
                if not match:
                    print(f"Warning: Could not extract label from {fname}, skipping...")
                    continue

                label_str = match.group(1)
                try:
                    label = torch.tensor([int(c) for c in label_str], dtype=torch.long)
                    item["class_label"] = label
                except Exception:
                    print(f"Warning: Invalid label format in {fname}, skipping...")
                    continue

                # Masks for train split
                base_name = fname.replace(".pt", ".png")
                mask_path = self.dirs["mask_dir"] / base_name
                mask_path_a = self.dirs["mask_dir_a"] / base_name
                mask_path_b = self.dirs["mask_dir_b"] / base_name

                if not mask_path.exists():
                    print(f"Warning: Mask not found for {indices_path}, skipping...")
                    continue
                if not mask_path_a.exists() or not mask_path_b.exists():
                    print(f"Warning: Additional masks not found for {indices_path}, skipping...")
                    continue

                item["mask_path"] = mask_path
                item["mask_path_a"] = mask_path_a
                item["mask_path_b"] = mask_path_b

            items.append(item)

        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict:
        item = self.items[index]
        try:
            indices = torch.load(item["indices_path"])  # [28, 28]
            sample = {"indices": indices}
            mask = Image.open(item["mask_path"]) if item["mask_path"].exists() else None
            sample["label"] = mask
            if self.split == "train":
                sample["label_a"] = Image.open(item["mask_path_a"]) if item["mask_path_a"].exists() else None
                sample["label_b"] = Image.open(item["mask_path_b"]) if item["mask_path_b"].exists() else None
                sample["class_label"] = item["class_label"]
            elif self.split in ["val", "test"]:
                # Derive image_path by mapping indices path to the corresponding image in masks_base
                indices_path_str = str(item["indices_path"])
                image_path = indices_path_str.replace(self.indices_base, self.masks_base)
                image_path = image_path.replace(f"{self.split}/indices", f"{self.split}/img")
                image_path = image_path.replace(".pt", ".png")
                sample["image_path"] = image_path

            # Apply label transform (convert masks to tensors)
            if self.label_transform:
                sample = self.label_transform(sample)

        except Exception as e:
            raise RuntimeError(f"Failed to load item {item}: {e}")

        return sample

def create_indice_dataloaders(
    indice_root: str,
    mask_root: str,
    dataset: str,
    batch_size: int,
    num_workers: int = 4,
    stage: str = "stage2"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if stage != "stage2":
        raise ValueError("Stage2IndiceDataset is only for stage2")

    train_dataset = Stage2IndiceDataset(
        indices_base=indice_root,
        masks_base=mask_root,
        split="train",
        dataset=dataset,
    )
    val_dataset = Stage2IndiceDataset(
        indices_base=indice_root,
        masks_base=mask_root,
        split="val",
        dataset=dataset,
    )
    test_dataset = Stage2IndiceDataset(
        indices_base=indice_root,
        masks_base=mask_root,
        split="test",
        dataset=dataset,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
#=============End: Indice Dataloaders=================