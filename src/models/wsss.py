import os
import sys 
from glob import glob
import yaml
import torch
import argparse
from tqdm import tqdm 

from PIL import Image
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Union, Any
from omegaconf import OmegaConf 


# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
print(f"Project root added to sys.path: {PROJECT_ROOT}") 

sys.path.append(os.path.join(PROJECT_ROOT, "src", "includes", "taming-transformers"))

from taming.models.vqgan import VQModel, GumbelVQ
sys.path.append(os.path.join(PROJECT_ROOT, "src", "includes", "efficientvit")) 
from src.datasets.kpis.vqgan_indexed_dataset import  VQGANIndexedDataset
# from src.models.segmentors.efficientvit_model import Index1DToSegmentation 

#---------------- this part will be moved to a seperate file ---------------- 
import torch.nn as nn 
from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b0
from efficientvit.models.efficientvit.seg import EfficientViTSeg, SegHead
from efficientvit.models.utils import build_kwargs_from_config


class Index1DToSegmentation(nn.Module):
    def __init__(self, num_codes: int, num_classes: int, embed_dim: int = 256, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(num_codes, embed_dim)  # [B, 64] → [B, 64, 256]

        # Backbone expecting input shape: [B, 256, 64, 64]
        backbone = efficientvit_backbone_b0(in_channels=embed_dim, **kwargs)

        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[128, 64, 32],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=64,
            head_depth=2,
            expand_ratio=4,
            middle_op="mbconv",
            final_expand=4,
            n_classes=num_classes,
            **build_kwargs_from_config(kwargs, SegHead),
        )

        self.seg_model = EfficientViTSeg(backbone, head)
        self.upsample = nn.Upsample(scale_factor=32, mode="bilinear", align_corners=False)  # 64 → 2048

    def forward(self, x):  # x: [B, 64]
        x = self.embedding(x)                     # [B, 64, 256]
        x = x.view(x.size(0), 8, 8, -1)           # [B, 8, 8, 256]
        x = x.permute(0, 3, 1, 2).contiguous()    # [B, 256, 8, 8]
        x = nn.functional.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)  # [B, 256, 64, 64]
        x = self.seg_model(x)                     # [B, num_classes, 64, 64]
        return self.upsample(x)                   # [B, num_classes, 64, 64] s 

#------------------------------------------------------------------------------
# Device setup
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"DEVICE: {DEVICE}")

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config 


def load_vqgan(config, ckpt_path=None, is_gumbel=False):
    if is_gumbel:
        model = GumbelVQ(**config.model.params)
    else:
        model = VQModel(**config.model.params)
        
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()


def main(args):
    # Validate train_test_val
    if args.train_test_val not in ["train", "test", "val"]:
        raise ValueError("train_test_val must be one of 'train', 'test', or 'val'")
        # Define transformations
    img_transform = transforms.Compose([
        # transforms.Resize((512, 512)),  # Adjust size as needed 
        transforms.ToTensor(),  
    ])
    mask_transform = transforms.Compose([
        # transforms.Resize((512, 512), interpolation=Image.NEAREST),  # Preserve label values
        # No ToTensor
    ]) 
    # torch.Size([8192, 256]) 
    # Dataset and DataLoader
    patch_dir = config[f"{args.train_test_val}_wsi_procWritingessed_patch_save_dir"]
    dataset = VQGANIndexedDataset(
        patch_dir, 
        target_size=512, # target_size=2048, 
        img_transform=img_transform, 
        mask_transform=None
        )
    
    #--------------------------config VQGAN model-------------------------- 
        
    config32x32 = load_config(
        f"{args.vqgan_logs_dir}/vqgan_gumbel_f8/configs/model.yaml", display=False)
    vqgan_model = load_vqgan(
        config32x32, 
        ckpt_path=f"{args.vqgan_logs_dir}/vqgan_gumbel_f8/checkpoints/last.ckpt", 
        is_gumbel=args.is_gumbel).to(DEVICE)  
    # if we are usign VQModel, then 
    
    if args.is_gumbel: 
        codebook_weights = vqgan_model.quantize.embed.weight # shape: [n_embed, embed_dim]

    print("codebook weights:", codebook_weights.shape)
    
    #--------------------------load dataset--------------------------  
    dataset = VQGANIndexedDataset(  
        patch_dir, 
        vqgan_model=vqgan_model, 
        target_size=2048, 
        patch_size=256, 
        stride=256, 
        img_transform=None, 
        mask_transform=None
    )
    
    #---- get model--- 
    for vq_patches in dataset:
        print(len(vq_patches))
        for img_index_vector, mask_index_vector in vq_patches:
            # print(np.unique(img_index_vector))
            # print(np.unique(mask_index_vector))
            print("img_index_vector shape: ", img_index_vector.shape)
            print("mask_index_vector shape: ", mask_index_vector.shape)         
    
            model = Index1DToSegmentation( # this is just untrained model, can be replace by trained model 
                num_codes=1024,
                num_classes=2,
                embed_dim=256
            )
            model.train()
            model = model.to(DEVICE)  
            output_quantized_mask = model(img_index_vector) 
            print("output_quantized_mask shape: ", output_quantized_mask.shape) 
            
            # adding decoder here 
            break
        break 

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Process WSI patches")
    parser.add_argument("--config", type=str, default="configs/main_vqgandataset_2024.yaml", help="Path to YAML config file")
    parser.add_argument("--data_config", type=str, default="configs/kpis.yaml", help="Path to data YAML config file")
    parser.add_argument("--train_test_val", type=str, default="train", help="Specify train/test/val")
    
    args = parser.parse_args()

    # Load config from YAML
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    with open(args.data_config, 'r') as f:
        config.update(yaml.safe_load(f))
    
    # Validate config keys
    required_keys = ["data_dir", f"{args.train_test_val}_wsi_processed_patch_save_dir"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Missing required config keys: {missing_keys}")
 
    args.vqgan_logs_dir = config.get('vqgan_logs_dir') 
    args.is_gumbel = True  
    
    main(args)
