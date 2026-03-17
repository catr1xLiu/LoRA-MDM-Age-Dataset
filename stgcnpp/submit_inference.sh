#!/bin/bash
#SBATCH --job-name=vc_inference
#SBATCH --partition=TrixieMain
#SBATCH --account=jpn-302
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --output=/gpfs/projects/AIP/jpn-302/LoRA-MDM-Age-Dataset/logs/vc_inference_%j.out
#SBATCH --error=/gpfs/projects/AIP/jpn-302/LoRA-MDM-Age-Dataset/logs/vc_inference_%j.err

echo "Node: $(hostname)"

module load conda/3-24.9.0
source activate stgcnpp

cd /gpfs/projects/AIP/jpn-302/LoRA-MDM-Age-Dataset/stgcnpp

echo "=== VAL SPLIT ==="
python batch_inference.py \
    --data data/vc_ntu25.pkl \
    --checkpoint checkpoints/stgcnpp_ntu120_3dkp_joint.pth \
    --modality joint \
    --task age \
    --split val

echo "=== TRAIN SPLIT ==="
python batch_inference.py \
    --data data/vc_ntu25.pkl \
    --checkpoint checkpoints/stgcnpp_ntu120_3dkp_joint.pth \
    --modality joint \
    --task age \
    --split train
