import os
import shutil
from pathlib import Path

def reorganize_luad_histoseg(source_dir: str, target_dir: str):
    """
    Reorganize the LUAD-HistoSeg dataset to match the expected structure.
    
    Args:
        source_dir: Path to the original LUAD-HistoSeg dataset.
        target_dir: Path to the new dataset directory (e.g., datasets/LUAD-HistoSeg).
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # Create target directories
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "train").mkdir(exist_ok=True)
    (target_dir / "val" / "img").mkdir(parents=True, exist_ok=True)
    (target_dir / "val" / "mask").mkdir(parents=True, exist_ok=True)
    (target_dir / "test" / "img").mkdir(parents=True, exist_ok=True)
    (target_dir / "test" / "mask").mkdir(parents=True, exist_ok=True)
    (target_dir / "docs").mkdir(exist_ok=True)
    
    # Copy train images (rename training/ to train/)
    train_source = source_dir / "training"
    train_target = target_dir / "train"
    for img_path in train_source.glob("*.png"):
        shutil.copy(img_path, train_target / img_path.name)
        print(f"Copied {img_path} to {train_target / img_path.name}")
    
    # Copy val images and masks
    val_img_source = source_dir / "val" / "img"
    val_mask_source = source_dir / "val" / "mask"
    val_img_target = target_dir / "val" / "img"
    val_mask_target = target_dir / "val" / "mask"
    for img_path in val_img_source.glob("*.png"):
        shutil.copy(img_path, val_img_target / img_path.name)
        print(f"Copied {img_path} to {val_img_target / img_path.name}")
    for mask_path in val_mask_source.glob("*.png"):
        shutil.copy(mask_path, val_mask_target / mask_path.name)
        print(f"Copied {mask_path} to {val_mask_target / mask_path.name}")
    
    # Copy test images and masks
    test_img_source = source_dir / "test" / "img"
    test_mask_source = source_dir / "test" / "mask"
    test_img_target = target_dir / "test" / "img"
    test_mask_target = target_dir / "test" / "mask"
    for img_path in test_img_source.glob("*.png"):
        shutil.copy(img_path, test_img_target / img_path.name)
        print(f"Copied {img_path} to {test_img_target / img_path.name}")
    for mask_path in test_mask_source.glob("*.png"):
        shutil.copy(mask_path, test_mask_target / mask_path.name)
        print(f"Copied {mask_path} to {test_mask_target / mask_path.name}")
    
    # Copy documentation files
    for doc_file in ["Readme.txt", "EXAMPLE_for_using_image-level_label.py"]:
        doc_path = source_dir / doc_file
        if doc_path.exists():
            shutil.copy(doc_path, target_dir / "docs" / doc_file)
            print(f"Copied {doc_path} to {target_dir / 'docs' / doc_file}")

    # Placeholder for train_PM directories (temporary workaround)
    # Since train_PM is missing, copy train images as placeholder masks
    (target_dir / "train_PM" / "PM_bn7").mkdir(parents=True, exist_ok=True)
    (target_dir / "train_PM" / "PM_b5_2").mkdir(parents=True, exist_ok=True)
    (target_dir / "train_PM" / "PM_b4_5").mkdir(parents=True, exist_ok=True)
    for img_path in train_source.glob("*.png"):
        for subdir in ["PM_bn7", "PM_b5_2", "PM_b4_5"]:
            shutil.copy(img_path, target_dir / "train_PM" / subdir / img_path.name)
            print(f"Copied {img_path} to {target_dir / 'train_PM' / subdir / img_path.name}")

if __name__ == "__main__":
    source_dir = "/project/hnguyen2/mvu9/datasets/LUAD-HistoSeg"
    target_dir = "/project/hnguyen2/mvu9/datasets/processing_datasets/LUAD-HistoSeg_organized"
    # create the target directory if it doesn't exist
    if not os.path.exists(target_dir):
        print("Target directory does not exist. Creating it...")
        os.makedirs(target_dir, exist_ok=True)
    else:
        # remove the target directory if it already exists
        print("Target directory already exists. Removing it...") 
        shutil.rmtree(target_dir)
        os.makedirs(target_dir, exist_ok=True) 
         
    reorganize_luad_histoseg(source_dir, target_dir)