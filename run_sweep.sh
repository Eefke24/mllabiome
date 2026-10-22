#!/bin/bash
#SBATCH --job-name=mindset_mllabiome
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --output=sweep_log_%j.txt
#SBATCH --error=sweep_err_%j.txt

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line $LINENO: $BASH_COMMAND (exit $rc)" >&2' ERR

module load anaconda3
source /opt/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate "$HOME/.conda/envs/mllabiome-py311"

cd /project/3033006.01/Eefke/mscProject/mllabiome
export MPLBACKEND=Agg

command -v python
python --version
command -v mllabiome

mllabiome examples/mindset_configs_sweep.py
