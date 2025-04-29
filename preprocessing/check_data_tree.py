import os
from pathlib import Path
from collections import defaultdict

def count_image_files(directory: Path) -> dict:
    """
    Count image files in a directory by their extensions.
    
    Args:
        directory: Path object representing the directory.
    
    Returns:
        A dictionary with extensions as keys and file counts as values.
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}  # Add more extensions if needed
    extension_counts = defaultdict(int)
    
    for item in directory.iterdir():
        if item.is_file() and item.suffix.lower() in image_extensions:
            extension_counts[item.suffix.lower()] += 1
    
    return dict(extension_counts)

def print_directory_tree(directory: Path, indent: str = "", prefix: str = ""):
    """
    Recursively print the directory tree. Summarize image files by count.
    
    Args:
        directory: Path object representing the directory.
        indent: String for indentation (used for recursive calls).
        prefix: Prefix for the current line (e.g., "| " or "└─").
    """
    print(f"{indent}{prefix}{directory.name}")
    
    # Get all items in the directory
    items = sorted(directory.iterdir(), key=lambda x: (x.is_file(), x.name))
    total_items = len(items)
    
    for index, item in enumerate(items):
        is_last = index == total_items - 1
        new_indent = indent + ("    " if is_last else "|   ")
        new_prefix = "└─" if is_last else "├─"
        
        if item.is_dir():
            # Recursively print subdirectories
            print_directory_tree(item, new_indent, new_prefix)
        else:
            # Check if the parent directory has image files to summarize
            image_counts = count_image_files(directory)
            if image_counts:
                # Print summary of image files and break (only print once per directory)
                summary = " ".join(f"{count} {ext} images" for ext, count in image_counts.items())
                print(f"{new_indent}{new_prefix}{summary}")
                break
            else:
                # Print non-image file
                print(f"{new_indent}{new_prefix}{item.name}")

def main():
    # Example usage
    folder_path = "/project/hnguyen2/mvu9/datasets/processing_datasets/LUAD-HistoSeg_organized"  # Replace with your folder path
    directory = Path(folder_path)
    
    if not directory.exists() or not directory.is_dir():
        print(f"Error: {folder_path} does not exist or is not a directory.")
        return
    
    print(f"Directory tree for {folder_path}:")
    print_directory_tree(directory)

if __name__ == "__main__":
    main()