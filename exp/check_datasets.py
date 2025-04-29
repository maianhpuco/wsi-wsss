# test_dataloaders.py
import os
import sys
import torch
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT) 

# Assuming the above code is in a file named 'datasets.py'
from src.datasets import create_dataloaders

def test_dataloaders():
    dataroot = "/project/hnguyen2/mvu9/datasets/processing_datasets/BCSS-WSSS_organized"
    
    # Test stage1
    print("Testing Stage 1 Dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        dataroot=dataroot,
        dataset="bcss",
        batch_size=16,
        num_workers=4,
        stage="stage1"
    )
    
    # Train loader
    for batch in train_loader:
        image_ids = batch["image_id"]
        images = batch["image"]
        labels = batch["label"]
        print(f"Stage 1 Train - Image IDs: {image_ids}")
        print(f"Stage 1 Train - Images shape: {images.shape}")
        print(f"Stage 1 Train - Labels shape: {labels.shape}")
        print(f"Stage 1 Train - Labels: {labels}")
        break
    
    # Val loader
    for batch in val_loader:
        image_ids = batch["image_id"]
        images = batch["image"]
        print(f"Stage 1 Val - Image IDs: {image_ids}")
        print(f"Stage 1 Val - Images shape: {images.shape}")
        break
    
    # Test stage2
    print("\nTesting Stage 2 Dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        dataroot=dataroot,
        dataset="bcss",
        batch_size=16,
        num_workers=4,
        stage="stage2"
    )
    
    # Train loader
    for batch in train_loader:
        images = batch["image"]
        labels = batch["label"]
        labels_a = batch["label_a"]
        labels_b = batch["label_b"]
        print(f"Stage 2 Train - Images shape: {images.shape}")
        print(f"Stage 2 Train - Labels shape: {labels.shape}")
        print(f"Stage 2 Train - Labels_a shape: {labels_a.shape}")
        print(f"Stage 2 Train - Labels_b shape: {labels_b.shape}")
        print(f"Stage 2 Train - Unique classes in labels: {torch.unique(labels)}")
        break
    
    # Val loader
    for batch in val_loader:
        images = batch["image"]
        labels = batch["label"]
        print(f"Stage 2 Val - Images shape: {images.shape}")
        print(f"Stage 2 Val - Labels shape: {labels.shape}")
        print(f"Stage 2 Val - Unique classes in labels: {torch.unique(labels)}")
        break

if __name__ == "__main__":
    test_dataloaders()