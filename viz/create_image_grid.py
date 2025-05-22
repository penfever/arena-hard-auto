#!/usr/bin/env python3
"""
Creates a grid visualization of images from a specified directory,
with each image's filename displayed as a legend.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
from pathlib import Path

# Ensure compatibility with numpy 2.0+
if not hasattr(np, 'NAN'):
    np.NAN = np.nan

def create_image_grid(input_dir, output_file=None, columns=3, figsize=(16, 12), 
                     dpi=100, title=None, extensions=('.png', '.jpg', '.jpeg')):
    """
    Creates a grid of images from the specified directory.
    
    Args:
        input_dir (str): Path to directory containing images
        output_file (str, optional): Path where to save the output grid image
        columns (int, optional): Number of columns in the grid
        figsize (tuple, optional): Size of the figure in inches (width, height)
        dpi (int, optional): Resolution of the figure
        title (str, optional): Title for the entire figure
        extensions (tuple, optional): File extensions to include
    
    Returns:
        matplotlib.figure.Figure: The figure object containing the grid
    """
    # Create input directory path
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"Input directory {input_dir} does not exist or is not a directory")
    
    # Get all image files in the directory
    image_files = sorted([f for f in os.listdir(input_dir) 
                   if f.lower().endswith(extensions)])
    
    if not image_files:
        raise ValueError(f"No images with extensions {extensions} found in {input_dir}")
    
    # Calculate grid dimensions
    num_images = len(image_files)
    rows = math.ceil(num_images / columns)
    
    # Create figure
    fig, axes = plt.subplots(rows, columns, figsize=figsize, 
                            constrained_layout=True, squeeze=False)
    
    # Add figure title if provided
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)
    
    # Add each image to the grid
    for i, img_file in enumerate(image_files):
        row = i // columns
        col = i % columns
        
        # Open and display the image
        img_path = os.path.join(input_dir, img_file)
        try:
            img = Image.open(img_path)
            axes[row, col].imshow(np.array(img))
            
            # Get image filename without extension for the title
            img_name = os.path.splitext(img_file)[0]
            axes[row, col].set_title(img_name, fontsize=10)
            
            # Remove axis ticks
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            
        except Exception as e:
            print(f"Error loading {img_file}: {e}")
            axes[row, col].text(0.5, 0.5, f"Error loading\n{img_file}", 
                              ha='center', va='center', color='red')
            axes[row, col].set_facecolor('#f0f0f0')
    
    # Hide unused subplots
    for i in range(num_images, rows * columns):
        row = i // columns
        col = i % columns
        axes[row, col].axis('off')
    
    # Save the figure if an output file is specified
    if output_file:
        output_path = Path(output_file)
        # Create the output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Grid saved to {output_file}")
    
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a grid of images from a directory")
    parser.add_argument("--input-dir", type=str, 
                      default="/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/oumi/llm-judge-oumi/scratch/factor_loadings_old_bench/",
                      help="Directory containing the images")
    parser.add_argument("--output-file", type=str, default="image_grid.png",
                      help="Output file path")
    parser.add_argument("--columns", type=int, default=3,
                      help="Number of columns in the grid")
    parser.add_argument("--figsize-width", type=float, default=16,
                      help="Width of the figure in inches")
    parser.add_argument("--figsize-height", type=float, default=None,
                      help="Height of the figure in inches (calculated automatically if not provided)")
    parser.add_argument("--dpi", type=int, default=100,
                      help="Resolution of the output image")
    parser.add_argument("--title", type=str, default=None,
                      help="Title for the entire figure")
    parser.add_argument("--extensions", type=str, default=".png,.jpg,.jpeg",
                      help="Comma-separated list of file extensions to include")
    
    args = parser.parse_args()
    
    # Parse extensions
    extensions = tuple(args.extensions.split(','))
    
    # Calculate figure height if not provided
    if args.figsize_height is None:
        # Count number of images
        image_files = [f for f in os.listdir(args.input_dir) 
                       if f.lower().endswith(extensions)]
        num_images = len(image_files)
        rows = math.ceil(num_images / args.columns)
        # Estimate appropriate height
        args.figsize_height = args.figsize_width * (rows / args.columns) * 0.75
    
    figsize = (args.figsize_width, args.figsize_height)
    
    # Create the grid
    create_image_grid(args.input_dir, args.output_file, args.columns, 
                    figsize, args.dpi, args.title, extensions)
    
    print(f"Created grid with {args.columns} columns from {args.input_dir}")
    print(f"Output saved to {args.output_file}")