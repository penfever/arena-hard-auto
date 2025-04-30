# Factor Analysis Visualization Tools

This directory contains tools for visualizing factor analysis results and other images from the LLM Judge evaluation pipeline.

## Tools

### `create_image_grid.py`

A Python script that creates a grid of images from a specified directory, with each image's filename displayed as a legend.

Usage:
```bash
python create_image_grid.py [OPTIONS]
```

Options:
- `--input-dir DIR`: Directory containing the images (default: specified directory)
- `--output-file FILE`: Output file path (default: "image_grid.png")
- `--columns NUM`: Number of columns in the grid (default: 3)
- `--figsize-width FLOAT`: Width of the figure in inches (default: 16)
- `--figsize-height FLOAT`: Height of the figure in inches (calculated automatically if not provided)
- `--dpi INT`: Resolution of the output image (default: 100)
- `--title TEXT`: Title for the entire figure (default: None)
- `--extensions LIST`: Comma-separated list of file extensions to include (default: ".png,.jpg,.jpeg")

### `visualize_factors.sh`

A convenient shell script wrapper for `create_image_grid.py` that makes it easy to visualize factor analysis images.

Usage:
```bash
./visualize_factors.sh [OPTIONS]
```

Options:
- `-i, --input-dir DIR`: Directory containing the images
- `-o, --output-file FILE`: Output file path (default: "factor_grid.png")
- `-c, --columns NUM`: Number of columns in the grid (default: 2)
- `-t, --title TEXT`: Title for the figure (default: "Factor Analysis Visualizations")
- `-h, --help`: Show help message

## Examples

### Basic Usage

To create a grid of all images in the default directory:
```bash
./visualize_factors.sh
```

### Customized Usage

To create a grid with 3 columns, a custom title, and specific input/output locations:
```bash
./visualize_factors.sh \
  --input-dir "/path/to/factor_loadings/" \
  --output-file "my_factor_grid.png" \
  --columns 3 \
  --title "Factor Loadings Comparison Across Judges"
```

## Dependencies

The scripts require the following Python libraries:
- NumPy
- Matplotlib
- Pillow (PIL)