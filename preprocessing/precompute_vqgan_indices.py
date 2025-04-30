# precompute_indices.py

import os
import sys 
import shutil 
import torch
from tqdm import tqdm
 

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f"Project root added to sys.path: {PROJECT_ROOT}")  
sys.path.append(PROJECT_ROOT) 
from src.datasets import create_dataloaders
from src.models import VQGANProcessor 
from utils import load_config 

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
    from taming.models.vqgan import VQModel, GumbelVQ 
    model_cls = GumbelVQ if is_gumbel else VQModel
    model = model_cls(**config.model.params)
    
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(sd, strict=False)
    return model.eval()


def precompute_indices(data_dir, dataset_name, vqgan_logs_dir, output_dir_root=None, is_gumbel=True, device="cuda"):
    DEVICE = torch.device(device)
    
    # Load VQGAN config + checkpoint
    config32x32 = load_config(
        os.path.join(vqgan_logs_dir, "vqgan_gumbel_f8/configs/model.yaml"), display=False)
    
    vqgan_model = load_vqgan(
        config=config32x32, 
        ckpt_path=os.path.join(vqgan_logs_dir, "vqgan_gumbel_f8/checkpoints/last.ckpt"), 
        is_gumbel=is_gumbel
    ).to(DEVICE)

    vqgan_processor = VQGANProcessor(vqgan_model).to(DEVICE)

    # Set default output root
    if output_dir_root is None:
        output_dir_root = data_dir.replace("_organized", "_indice")
    
    if os.path.exists(output_dir_root):
        shutil.rmtree(output_dir_root)  # Remove existing directory and all contents
    os.makedirs(output_dir_root)        # Create a fresh directory 
    
    for split in ["train", "val", "test"]:
        print(f"\n Processing split: {split}")
        
        # Create output directory
        output_dir = os.path.join(output_dir_root, split, "indices")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)  # Remove existing directory and all contents
        os.makedirs(output_dir)        # Create a fresh directory 
        
         
        # Load appropriate dataloader
        train_loader, val_loader, _ = create_dataloaders(
            dataroot=data_dir,
            dataset=dataset_name,
            batch_size=16,
            num_workers=4,
            stage="stage2"
        )
        loader = train_loader if split == "train" else val_loader
        
        for batch in tqdm(loader, desc=f"Encoding {split}", leave=False):
        # for batch in loader:
            images = batch["image"].to(DEVICE)
            indices = vqgan_processor(images)  # Shape: [B, H, W] or [B, T]

            for i, idx in enumerate(indices):
                try:
                    image_path = batch["image_path"][i]  # Should be string
                    # print(i, image_path) 
                    image_name = os.path.splitext(os.path.basename(image_path))[0]
                    # print(f"Image name: {image_name}")
                    save_path = os.path.join(output_dir, f"{image_name}.pt")
                    torch.save(idx.cpu(), save_path)
                    # print(f"Saved index: {save_path}")
                except Exception as e:
                    print(f"Failed to save index for item {i}: {e}")

if __name__ == "__main__":
    # dataset_name = 'bcss'
    dataset_name = 'luad'  
    if dataset_name == 'luad': 
        dataroot = "/project/hnguyen2/mvu9/datasets/processing_datasets/LUAD-HistoSeg_organized"   
    
    elif dataset_name == 'bcss':
        dataroot = "/project/hnguyen2/mvu9/datasets/processing_datasets/BCSS-WSSS_organized"     
    
    precompute_indices(
        data_dir=dataroot
        dataset_name=dataset_name,
        vqgan_logs_dir="/project/hnguyen2/mvu9/folder_04_ma/wsi_efficient_seg/resources/vqgan/logs",
        is_gumbel=True
    )
