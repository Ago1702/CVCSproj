#!/bin/bash
#SBATCH --job-name=two_block    # Job name
#SBATCH --output=output.txt         # Output file
#SBATCH --error=error.txt           # Error file
#SBATCH --time=24:00:00             # Time limit (HH:MM:SS)
#SBATCH --partition=all_usr_prod      # Partition
#SBATCH --gres=gpu:0                # Request one GPU
#SBATCH --account=cvcs2024          # Account name
#SBATCH --mem=10G                   # Request 10 GB of RAM
#SBATCH --cpus-per-task=4
# Run the Python script
# istruzioni: 
# - clonare la git nella home directory
# - cambiare il file in "python ~/CVCSproj/percorso_al_file"


cd ~/CVCSproj/dataset_generation

source ~/CVCSproj/.venv/bin/activate
python3 renamer.py
