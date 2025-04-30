import os
import torch
from torch.utils.data import DataLoader
import argparse

# Assuming the VQ-GAN loading functions are available
from vqgan_utils import load_config, load_vqgan  # Placeholder for your VQ-GAN loading code
from src.datasets import create_dataloaders, create_indice_dataloaders
from model import VQGANProcessor, ViTClassifier
from train_utils import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqgan_logs_dir", type=str, required=True, help="Directory containing VQ-GAN model configs and checkpoints")
    parser.add_argument("--is_gumbel", action="store_true", help="Whether to use Gumbel VQ-GAN")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--dataset_name", type=str, required=True, choices=["bcss", "luad"], help="Dataset name (bcss or luad)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on")
    parser.add_argument("--use_indices", action="store_true", help="Use precomputed indices")
    args = parser.parse_args()

    DEVICE = torch.device(args.device)

    # Load VQ-GAN model (only if not using precomputed indices)
    vqgan_processor = None
    config32x32 = load_config(
        f"{args.vqgan_logs_dir}/vqgan_gumbel_f8/configs/model.yaml", display=False)
    vqgan_model = load_vqgan(
        config32x32, 
        ckpt_path=f"{args.vqgan_logs_dir}/vqgan_gumbel_f8/checkpoints/last.ckpt", 
        is_gumbel=args.is_gumbel).to(DEVICE)
    if not args.use_indices:
        vqgan_processor = VQGANProcessor(vqgan_model).to(DEVICE)

    if args.is_gumbel:
        codebook_weights = vqgan_model.quantize.embed.weight  # [n_embed, embed_dim]
    else:
        raise NotImplementedError("Only Gumbel VQ-GAN is supported in this example")
    print("Codebook weights:", codebook_weights.shape)

    # Load dataset
    if args.use_indices:
        train_loader, val_loader, _ = create_indice_dataloaders(
            dataroot=args.data_dir,
            dataset=args.dataset_name,
            batch_size=args.batch_size,
            num_workers=4,
            stage="stage2"
        )
    else:
        train_loader, val_loader, _ = create_dataloaders(
            dataroot=args.data_dir,
            dataset=args.dataset_name,
            batch_size=args.batch_size,
            num_workers=4,
            stage="stage2"
        )

    # Initialize ViT classifier
    vit_classifier = ViTClassifier(
        codebook_weights=codebook_weights,
        num_classes=4,  # Multi-label classification for classes 1-4 (TUM, STR, LYM, NEC)
    ).to(DEVICE)

    # Optimizer
    optimizer = torch.optim.Adam(vit_classifier.parameters(), lr=args.learning_rate)

    # Define the forward pass based on whether indices are precomputed
    if args.use_indices:
        forward_fn = lambda batch: vit_classifier(batch["indices"])
    else:
        forward_fn = lambda batch: vit_classifier(vqgan_processor(batch["image"]))

    # Train the model
    train(
        model=forward_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=DEVICE
    )

if __name__ == "__main__":
    main()