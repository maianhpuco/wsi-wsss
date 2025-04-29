import os
import shutil
from pathlib import Path

def reorganize_bcss_wsss(source_dir: str, target_dir: str):
    """
    Reorganize the BCSS-WSSS dataset to match the expected structure.
    
    Args:
        source_dir: Path to the original BCSS-WSSS dataset.
        target_dir: Path to the new dataset directory (e.g., datasets/BCSS-WSSS).
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # Validate source directory
    if not source_dir.exists() or not source_dir.is_dir():
        raise ValueError(f"Source directory {source_dir} does not exist or is not a directory.")
    
    # Create target directories
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "train").mkdir(exist_ok=True)
    (target_dir / "val" / "img").mkdir(parents=True, exist_ok=True)
    (target_dir / "val" / "mask").mkdir(exist_ok=True)
    (target_dir / "test" / "img").mkdir(parents=True, exist_ok=True)
    (target_dir / "test" / "mask").mkdir(exist_ok=True)
    (target_dir / "docs").mkdir(exist_ok=True)
    
    # Copy train images
    train_source = source_dir / "train"
    train_target = target_dir / "train"
    train_count = 0
    # Collect all .png files (case-insensitive) in a list to reuse
    train_images = [img_path for img_path in train_source.iterdir() if img_path.is_file() and img_path.suffix.lower() == ".png"]
    print(f"Found {len(train_images)} .png images in {train_source}")
    
    for img_path in train_images:
        shutil.copy(img_path, train_target / img_path.name)
        train_count += 1
        print(f"Copied {img_path} to {train_target / img_path.name}")
    print(f"Total train images copied: {train_count}")
    
    # Copy val images and masks
    val_img_source = source_dir / "val" / "img"
    val_mask_source = source_dir / "val" / "mask"
    val_img_target = target_dir / "val" / "img"
    val_mask_target = target_dir / "val" / "mask"
    val_img_count = 0
    val_mask_count = 0
    for img_path in val_img_source.glob("*.png"):
        shutil.copy(img_path, val_img_target / img_path.name)
        val_img_count += 1
        print(f"Copied {img_path} to {val_img_target / img_path.name}")
    for mask_path in val_mask_source.glob("*.png"):
        shutil.copy(mask_path, val_mask_target / mask_path.name)
        val_mask_count += 1
        print(f"Copied {mask_path} to {val_mask_target / mask_path.name}")
    print(f"Total val images copied: {val_img_count}")
    print(f"Total val masks copied: {val_mask_count}")
    
    # Copy test images and masks
    test_img_source = source_dir / "test" / "img"
    test_mask_source = source_dir / "test" / "mask"
    test_img_target = target_dir / "test" / "img"
    test_mask_target = target_dir / "test" / "mask"
    test_img_count = 0
    test_mask_count = 0
    for img_path in test_img_source.glob("*.png"):
        shutil.copy(img_path, test_img_target / img_path.name)
        test_img_count += 1
        print(f"Copied {img_path} to {test_img_target / img_path.name}")
    for mask_path in test_mask_source.glob("*.png"):
        shutil.copy(mask_path, test_mask_target / mask_path.name)
        test_mask_count += 1
        print(f"Copied {mask_path} to {test_mask_target / mask_path.name}")
    print(f"Total test images copied: {test_img_count}")
    print(f"Total test masks copied: {test_mask_count}")
    
    # Copy documentation files
    doc_count = 0
    for doc_file in ["Readme.txt", "EXAMPLE_for_using_image-level_label.py"]:
        doc_path = source_dir / doc_file
        if doc_path.exists():
            shutil.copy(doc_path, target_dir / "docs" / doc_file)
            doc_count += 1
            print(f"Copied {doc_path} to {target_dir / 'docs' / doc_file}")
    print(f"Total documentation files copied: {doc_count}")

    # Placeholder for train_PM directories (temporary workaround)
    # Use the same list of train images to copy to train_PM subdirectories
    (target_dir / "train_PM" / "PM_bn7").mkdir(parents=True, exist_ok=True)
    (target_dir / "train_PM" / "PM_b5_2").mkdir(parents=True, exist_ok=True)
    (target_dir / "train_PM" / "PM_b4_5").mkdir(parents=True, exist_ok=True)
    train_pm_counts = {"PM_bn7": 0, "PM_b5_2": 0, "PM_b4_5": 0}
    for img_path in train_images:  # Reuse the same list of train images
        for subdir in ["PM_bn7", "PM_b5_2", "PM_b4_5"]:
            shutil.copy(img_path, target_dir / "train_PM" / subdir / img_path.name)
            train_pm_counts[subdir] += 1
            print(f"Copied {img_path} to {target_dir / 'train_PM' / subdir / img_path.name}")
    for subdir, count in train_pm_counts.items():
        print(f"Total images copied to train_PM/{subdir}: {count}")

if __name__ == "__main__":
    source_dir = "/project/hnguyen2/mvu9/datasets/BCSS-WSSS"
    target_dir = "/project/hnguyen2/mvu9/datasets/processing_datasets/BCSS-WSSS_organized"
    # Create or recreate the target directory
    if not os.path.exists(target_dir):
        print("Target directory does not exist. Creating it...")
        os.makedirs(target_dir, exist_ok=True)
    else:
        print("Target directory already exists. Removing it...")
        shutil.rmtree(target_dir)
        os.makedirs(target_dir, exist_ok=True)
    
    reorganize_bcss_wsss(source_dir, target_dir)