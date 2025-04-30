#!/bin/bash
# Wrapper script for creating image grids from factor analysis visualizations

# Default values
DEFAULT_INPUT_DIR="/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/oumi/llm-judge-oumi/scratch/factor_loadings_old_bench/"
DEFAULT_OUTPUT_FILE="factor_grid.png"
DEFAULT_COLUMNS=2
DEFAULT_TITLE="Factor Analysis Visualizations"

# Help function
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Create a grid visualization of factor analysis images"
    echo ""
    echo "Options:"
    echo "  -i, --input-dir DIR     Directory containing the images (default: $DEFAULT_INPUT_DIR)"
    echo "  -o, --output-file FILE  Output file path (default: $DEFAULT_OUTPUT_FILE)"
    echo "  -c, --columns NUM       Number of columns in the grid (default: $DEFAULT_COLUMNS)"
    echo "  -t, --title TEXT        Title for the figure (default: $DEFAULT_TITLE)"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --input-dir '/path/to/images' --output-file 'my_grid.png' --columns 3"
}

# Parse command line arguments
INPUT_DIR="$DEFAULT_INPUT_DIR"
OUTPUT_FILE="$DEFAULT_OUTPUT_FILE"
COLUMNS="$DEFAULT_COLUMNS"
TITLE="$DEFAULT_TITLE"

while [ "$1" != "" ]; do
    case $1 in
        -i | --input-dir )     shift
                               INPUT_DIR="$1"
                               ;;
        -o | --output-file )   shift
                               OUTPUT_FILE="$1"
                               ;;
        -c | --columns )       shift
                               COLUMNS="$1"
                               ;;
        -t | --title )         shift
                               TITLE="$1"
                               ;;
        -h | --help )          show_help
                               exit
                               ;;
        * )                    echo "Invalid option: $1"
                               show_help
                               exit 1
    esac
    shift
done

# Verify input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist."
    exit 1
fi

# Count images in the directory
IMAGE_COUNT=$(ls -1 "$INPUT_DIR"/*.{png,jpg,jpeg} 2>/dev/null | wc -l)
if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "Error: No PNG or JPG images found in '$INPUT_DIR'."
    exit 1
fi

echo "Creating grid visualization with $COLUMNS columns from $IMAGE_COUNT images in '$INPUT_DIR'"

# Run the Python script with the provided arguments
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
python "$SCRIPT_DIR/create_image_grid.py" \
    --input-dir "$INPUT_DIR" \
    --output-file "$OUTPUT_FILE" \
    --columns "$COLUMNS" \
    --title "$TITLE"

echo "Visualization saved to: $OUTPUT_FILE"