import os
import sys 
import torch
from torch.utils.data import DataLoader
import argparse

# Assuming the VQ-GAN loading functions are available
# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f"Project root added to sys.path: {PROJECT_ROOT}")  
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src", "includes", "taming-transformers")) 

from utils import load_config  # Placeholder for your VQ-GAN loading code
from src.datasets import create_dataloaders
from src.models import VQGANViTClassifier
from utils.train import train 

from taming.models.vqgan import VQModel, GumbelVQ
#sys.path.append(os.path.join(PROJECT_ROOT, "src", "includes", "efficientvit")) 


def load_vqgan(config, ckpt_path=None, is_gumbel=False):
    if is_gumbel:
        model = GumbelVQ(**config.model.params)
    else:
        model = VQModel(**config.model.params)
        
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()
  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default='bcss', type=str)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    from configs import exp_02 as config_file 
       
    args.vqgan_logs_dir = config_file.vqgan_logs_dir
    args.dataset_name = config_file.dataset_name 
    args.data_dir = config_file.data_dir 
    args.batch_size = config_file.batch_size 
    args.num_epochs = config_file.num_epochs 
    args.learning_rate = config_file.learning_rate 
    args.is_gumbel = config_file.is_gumbel 
     
     
    DEVICE = torch.device(args.device)

    #=================Start: Load VQ-GAN model=================
    config32x32 = load_config(
        f"{args.vqgan_logs_dir}/vqgan_gumbel_f8/configs/model.yaml", display=False)
    
    vqgan_model = load_vqgan(
        config32x32, 
        ckpt_path=f"{args.vqgan_logs_dir}/vqgan_gumbel_f8/checkpoints/last.ckpt", 
        is_gumbel=args.is_gumbel).to(DEVICE)    
    if args.is_gumbel:
        codebook_weights = vqgan_model.quantize.embed.weight  # [n_embed, embed_dim]
    else:
        raise NotImplementedError("Only Gumbel VQ-GAN is supported in this example")
    print("Codebook weights:", codebook_weights.shape)
    #=================End: Load VQ-GAN model================= 

    #=================Start: Training=================

    train_loader, val_loader, _ = create_dataloaders(
        dataroot=args.data_dir,
        dataset=args.dataset_name,
        batch_size=args.batch_size,
        num_workers=4,
        stage="stage2"
    )
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

    model = VQGANViTClassifier(
        vqgan_model=vqgan_model,
        codebook_weights=codebook_weights,
        num_classes=4,  # Multi-label classification for classes 1-4 (TUM, STR, LYM, NEC)
        patch_size=14  # 224/16 ≈ 14 patches per dimension for timm's ViT
    ).to(DEVICE)

    # Optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train the model
    # train(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     optimizer=optimizer,
    #     num_epochs=args.num_epochs,
    #     device=DEVICE
    # )

if __name__ == "__main__":
    main()