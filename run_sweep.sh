#!/bin/bash
#SBATCH --job-name=mindset_mllabiome
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=sweep_log_%j.txt
#SBATCH --error=sweep_err_%j.txt

module load anaconda3
source  activate "$HOME/.conda/envs/mllabiome-py311"

cd /project/3033006.01/Eefke/mscProject/mllabiome
export MPLBACKEND=Agg

which python
python --version
mllabiome examples/mindset_configs_sweep.py --stage report
