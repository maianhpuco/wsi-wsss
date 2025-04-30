# test_dataloaders.py
import os
import sys
import torch
import argparse
import yaml 
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT) 

from src.datasets import create_dataloaders

def main(dataroot, dataset_name):
    dataset_name = 'bcss'
    
    if dataset_name == 'luad': 
        dataroot = "/project/hnguyen2/mvu9/datasets/processing_datasets/LUAD-HistoSeg_organized"   
    
    elif dataset_name == 'bcss':
        dataroot = "/project/hnguyen2/mvu9/datasets/processing_datasets/BCSS-WSSS_organized"      
        
    
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
    

if __name__ == "__main__":
        # Argument parser
    default_dataset = 'bcss'
    default_experiment = os.path.basename(__file__)
    
    parser = argparse.ArgumentParser(description="Process WSI patches")
    parser.add_argument("--config", type=str, default=f"configs/{default_experiment}.yaml", help="Path to YAML config file")
    parser.add_argument("--data_config", type=str, default=f"configs/{default_dataset}.yaml", help="Path to data YAML config file")

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
    
