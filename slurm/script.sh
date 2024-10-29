#!/bin/bash
#SBATCH --job-name=VisionWise                           # Job name
#SBATCH --output=outputs/output.txt                     # Output file
#SBATCH --error=outputs/error.txt                       # Error file
#SBATCH --time=10:00:00                                 # Time limit (HH:MM:SS)
#SBATCH --partition=all_usr_prod                        # all_serial all_usr_prod boost_usr_prod
#SBATCH --gres=gpu:3                                    # Request one GPU
#SBATCH --account=cvcs2024                              # Account name
#SBATCH --mem=20G                                       # Request 20 GB of RAM
#SBATCH --cpus-per-task=8

# Run the Python script
# istruzioni: 
# - clonare la git nella home directory
# - cambiare il file in "python ~/CVCSproj/percorso_al_file"


cd ~/CVCSproj/slurm

export PYTHONPATH=~/CVCSproj:$PYTHONPATH
source ~/CVCSproj/.venv/bin/activate
python3  ~/CVCSproj/models/resnet_contrastive.py
