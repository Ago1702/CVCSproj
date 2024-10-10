#!/bin/bash
#SBATCH --job-name=two_block    # Job name
#SBATCH --output=output.txt         # Output file
#SBATCH --error=error.txt           # Error file
#SBATCH --time=4:00:00             # Time limit (HH:MM:SS)
#SBATCH --partition=all_serial      # Partition
#SBATCH --gres=gpu:1                # Request one GPU
#SBATCH --account=cvcs2024          # Account name
#SBATCH --mem=20G                   # Request 20 GB of RAM
# Run the Python script
# istruzioni: 
# - clonare la git nella home directory
# - cambiare il file in "python ~/CVCSproj/percorso_al_file"


cd ~/CVCSproj/model

source ~/CVCSproj/.venv/bin/activate
bash ./sono_folle.bash 666
