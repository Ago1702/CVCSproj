#!/bin/bash
#SBATCH --job-name=50_cbam_contr_class                         
#SBATCH --output=outputs/output_50.txt                    
#SBATCH --error=outputs/error_50.txt                      
#SBATCH --time=24:00:00                                 
#SBATCH --partition=all_usr_prod                        # all_serial all_usr_prod boost_usr_prod
#SBATCH --gres=gpu:2                                   
#SBATCH --account=cvcs2024                             
#SBATCH --mem=60G                                       
#SBATCH --cpus-per-task=8

# Run the Python script
# istruzioni: 
# - clonare la git nella home directory
# - cambiare il file in "python ~/CVCSproj/percorso_al_file"

cd ~/CVCSproj/slurm

export PYTHONPATH=~/CVCSproj:$PYTHONPATH
source ~/CVCSproj/.venv/bin/activate
python3  ~/CVCSproj/train/contrastive_classifier_50.py