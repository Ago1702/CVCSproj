#!/bin/bash
#SBATCH --job-name=VisionWise       # Job name
#SBATCH --output=output.txt         # Output file
#SBATCH --error=error.txt           # Error file
#SBATCH --time=24:00:00             # Time limit (HH:MM:SS)
#SBATCH --partition=all_serial      # Partition
#SBATCH --gres=gpu:1                # Request one GPU
#SBATCH --account=cvcs2024          # Account name

# Run the Python script
# istruzioni: 
# - clonare la git nella home directory
# - cambiare il file in "python ~/CVCSproj/percorso_al_file"
cd ~/CVCSproj/slurm
python ~/CVCSproj/model/iter_dataset.py
