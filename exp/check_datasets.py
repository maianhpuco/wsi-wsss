# import os
# import sys 
# from glob import glob
# import yaml
# import torch
# import argparse
# from tqdm import tqdm 

# from PIL import Image
# import numpy as np
# import torch.nn.functional as F
# import torchvision.transforms as T
# import torchvision.transforms.functional as TF
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from typing import Union, Any
# from omegaconf import OmegaConf 


import os 
import sys 

import argparse 

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 
sys.path.append(PROJECT_ROOT)
 
from src.datasets import create_dataloaders 




def main(args):
    # Create dataloaders for stage1
    # train_loader, val_loader, test_loader = create_dataloaders(
    #     dataroot=args.dataroot,
    #     dataset=args.dataset,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     stage="stage1"
    # )

    # Or for stage2
    train_loader, val_loader, test_loader = create_dataloaders(
        dataroot=args.dataroot,
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stage="stage2"
    )
        # Use in training loop
    for batch in train_loader:
        image_ids = batch.get("image_id", [])
        images = batch["image"]
        labels = batch.get("label")
        # Process batch...  
        print(f"Image IDs: {image_ids}")
        print(f"Images shape: {images.shape}")
        if labels is not None:
            print(f"Labels shape: {labels.shape}")
        else:
            print("No labels in this batch.") 


    
if __name__ == "__main__":
    # Example configuration
    args = argparse.Namespace(
        dataroot="/project/hnguyen2/mvu9/datasets/LUAD-HistoSeg",
        dataset="luad",  # or "bcss" for stage1
        batch_size=16,
        num_workers=4
    ) 
    main(args)
