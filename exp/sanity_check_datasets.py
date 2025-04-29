# test_dataloaders.py
import os
import sys
import torch
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT) 

from src.datasets import create_dataloaders

def test_dataloaders(dataroot, dataset_name):

    print("\nTesting Stage 2 Dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        dataroot=dataroot,
        dataset=dataset_name,
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
        class_labels = batch["class_label"]
        print(f"Stage 2 Train - Images shape: {images.shape}")
        print(f"Stage 2 Train - Labels shape: {labels.shape}")
        print(f"Stage 2 Train - Labels_a shape: {labels_a.shape}")
        print(f"Stage 2 Train - Labels_b shape: {labels_b.shape}")
        print(f"Stage 2 Train - Unique classes in labels: {torch.unique(labels)}")
        print(f"Stage 2 Train - Classification labels: {class_labels}")
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
    dataset_name = 'luad'
    
    if dataset_name == 'luad': 
        dataroot = "/project/hnguyen2/mvu9/datasets/processing_datasets/LUAD-HistoSeg_organized"   
    
    elif dataset_name == 'bcss':
        dataroot = "/project/hnguyen2/mvu9/datasets/processing_datasets/BCSS-WSSS_organized"     
    
    test_dataloaders(dataroot, dataset_name)
    
'''    
dataroot = "/project/hnguyen2/mvu9/datasets/processing_datasets/BCSS-WSSS_organized"
Number of images in train: 23422
Number of images in val: 3418
Number of images in test: 4986

Stage 2 Train - Images shape: torch.Size([16, 3, 224, 224])
Stage 2 Train - Labels shape: torch.Size([16, 224, 224])
Stage 2 Train - Labels_a shape: torch.Size([16, 224, 224])
Stage 2 Train - Labels_b shape: torch.Size([16, 224, 224])
Stage 2 Train - Unique classes in labels: tensor([0])

Stage 2 Val - Images shape: torch.Size([16, 3, 224, 224])
Stage 2 Val - Labels shape: torch.Size([16, 224, 224])
Stage 2 Val - Unique classes in labels: tensor([0, 1, 2, 4]) 
''' 