#!/bin/bash

# Check if required tools are installed
for tool in pdftk gs; do
    if ! command -v $tool &> /dev/null; then
        echo "$tool could not be found. Please install it first."
        exit 1
    fi
done

# Check if at least one PDF file is provided
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 file1.pdf file2.pdf ..."
    exit 1
fi

# Temporary directory for processing
temp_dir=$(mktemp -d)

# Create a blank PDF page using ghostscript
gs -q -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER \
   -dDEVICEWIDTHPOINTS=612 -dDEVICEHEIGHTPOINTS=792 \
   -sOutputFile="$temp_dir/blank.pdf" -c "showpage"

# Counter for naming temporary files
counter=1

# Process each input PDF
for pdf in "$@"; do
    # Get the number of pages in the PDF
    pages=$(pdftk "$pdf" dump_data | grep NumberOfPages | awk '{print $2}')
    
    # Copy the PDF to the temp directory
    cp "$pdf" "$temp_dir/$counter.pdf"
    
    # If the number of pages is odd, add a blank page
    if [ $((pages % 2)) -eq 1 ]; then
        pdftk A="$temp_dir/$counter.pdf" B="$temp_dir/blank.pdf" cat A B output "$temp_dir/$counter-even.pdf"
        mv "$temp_dir/$counter-even.pdf" "$temp_dir/$counter.pdf"
    fi
    
    counter=$((counter + 1))
done

# Concatenate all PDFs
pdftk "$temp_dir"/*.pdf cat output concatenated.pdf

# Clean up
rm -rf "$temp_dir"

echo "All PDFs have been concatenated into concatenated.pdf"
