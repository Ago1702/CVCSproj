#!/bin/bash
#SBATCH --job-name=transform_test       # Job name
#SBATCH --output=output.txt         # Output file
#SBATCH --error=error.txt           # Error file
#SBATCH --time=01:00:00             # Time limit (HH:MM:SS)
#SBATCH --partition=all_serial      # Partition
#SBATCH --gres=gpu:1                # Request one GPU
#SBATCH --account=cvcs2024          # Account name

# Run the Python script
cd ~/CVCSproj/slurm
python ~/CVCSproj/utils/transform_test.py
