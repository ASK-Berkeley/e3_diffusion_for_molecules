#!/bin/bash

# The directory containing the files
dir="outputs/qm9_mc/flexible_mols/diffusion/$1/$2"

# The directory where the results will be stored temporarily
temp_dir="temp_results"

# The file where the final results will be stored
outfile="outputs/qm9_mc/flexible_mols/diffusion/$1/$2/energies.txt"

# Create the temporary directory
mkdir -p $temp_dir

# Remove the output file if it already exists
if [ -f "$outfile" ]; then
    rm "$outfile"
fi

# Define a shell function to process one file
process_file() {
    filename=$(basename "$1")
    python mc_get_energy.py "$1" | tail -n 1 > "$2/$filename"
}

# Export the shell function so it can be used by xargs
export -f process_file

# Process the files using xargs to control parallelism
ls $dir/step_*.xyz | xargs -P 48 -I{} bash -c "process_file {} $temp_dir"

# Concatenate and sort the output files
for filename in $(ls $temp_dir/step_*.xyz | xargs -n 1 basename | sort -V); do
    cat "$temp_dir/$filename" >> "$outfile"
done

# Delete the temporary directory
rm $temp_dir/*.xyz
rmdir $temp_dir
