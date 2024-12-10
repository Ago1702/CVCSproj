#!/bin/bash
#SBATCH --job-name=crazy_ensemble                          
#SBATCH --output=outputs/output_ens.txt                    
#SBATCH --error=outputs/error_ens.txt                      
#SBATCH --time=24:00:00                                 
#SBATCH --partition=all_usr_prod                        # all_serial all_usr_prod boost_usr_prod
#SBATCH --gres=gpu:4                                   
#SBATCH --account=cvcs2024                             
#SBATCH --mem=80g                                  
#SBATCH --cpus-per-task=8

# Run the Python script
# istruzioni: 
# - clonare la git nella home directory
# - cambiare il file in "python ~/CVCSproj/percorso_al_file"

cd ~/CVCSproj/slurm

export PYTHONPATH=~/CVCSproj:$PYTHONPATH
source ~/CVCSproj/.venv/bin/activate
python3  ~/CVCSproj/train/3_ensemble.py