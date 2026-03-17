#!/bin/bash
#SBATCH --job-name=vc_train
#SBATCH --partition=TrixieMain
#SBATCH --account=jpn-302
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=/gpfs/projects/AIP/jpn-302/LoRA-MDM-Age-Dataset/logs/vc_train_%j.out
#SBATCH --error=/gpfs/projects/AIP/jpn-302/LoRA-MDM-Age-Dataset/logs/vc_train_%j.err

echo "Node: $(hostname)"
echo "GPU : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"

module load conda/3-24.9.0
source activate stgcnpp

cd /gpfs/projects/AIP/jpn-302/LoRA-MDM-Age-Dataset/stgcnpp

echo ""
echo "=== STEP 1: Fine-tune age classifier (50 epochs) ==="
python train_vc.py \
    --checkpoint    checkpoints/stgcnpp_ntu120_3dkp_joint.pth \
    --data          data/vc_ntu25.pkl \
    --epochs        50 \
    --batch-size    16 \
    --num-workers   8 \
    --unfreeze-blocks 1 \
    --output        checkpoints/vc_age_1block.pth

echo ""
echo "=== STEP 2: Extract z_age embeddings ==="
python extract_z_age.py \
    --checkpoint checkpoints/vc_age_1block.pth \
    --data       data/vc_ntu25.pkl \
    --output     data/z_age_embeddings_1block.npz \
    --batch-size 16 \
    --num-workers 8

echo ""
echo "=== Done: $(date) ==="
