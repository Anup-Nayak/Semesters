#!/bin/bash

DB_NAME="a1_db"
USER="postgres"
BASE_DIR="$(pwd)/2022CS51827"  # Adjust BASE_DIR to the correct subdirectory
OUTPUT_DIR="$(pwd)/Avnikebacche_outputs"  # Output folder inside the base directory

# Use the password file for authentication
export PGPASSFILE=~/.pgpass

# Verify that the directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory $BASE_DIR not found!"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Find all SQL files in subdirectories and execute them
find "$BASE_DIR" -type f -name "*.sql" | while read -r file; do
    # Preserve subdirectory structure
    relative_path="${file#$BASE_DIR/}"
    output_subdir="$OUTPUT_DIR/$(dirname "$relative_path")"
    mkdir -p "$output_subdir"

    # Define output file name
    output_file="$output_subdir/$(basename "$file" .sql).txt"

    # Execute SQL file and save output
    psql -U "$USER" -d "$DB_NAME" -f "$file" > "$output_file"

    echo "Executed $file -> Output saved to $output_file"
done