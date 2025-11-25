#!/bin/bash

# ============================================================
# CSC2503 Project - Complete Experiment Script
# ============================================================
# 24 total experiments: 8 per model (ResNet18, MobileNet, ViT)
# Each model: 1 baseline + 3 feature + 3 logit + 1 combined
# ============================================================

set -e  # Exit on error

echo "============================================================"
echo "Starting All Experiments (24 total, 8 per model)"
echo "============================================================"
echo ""

# ------------------------------------------------------------
# RESNET18 EXPERIMENTS (8 total)
# ------------------------------------------------------------

echo "------------------------------------------------------------"
echo "RESNET18 - BASELINE"
echo "------------------------------------------------------------"

echo "[1/24] ResNet18 - Baseline (no distillation)..."
python train.py \
    --backbone resnet18 \
    --batch-size 8 \
    --epochs 1 \
    --lr 1e-3 \
    --run-name resnet18_baseline \
    --output-dir checkpoints/resnet18_baseline \
    --eval-final-only

echo "------------------------------------------------------------"
echo "RESNET18 - FEATURE DISTILLATION (DINO)"
echo "------------------------------------------------------------"

echo "[2/24] ResNet18 - Feature Only β=0.5..."
python train.py \
    --backbone resnet18 \
    --distill \
    --alpha 1.0 --beta 0.5 --gamma 0.0 \
    --batch-size 8 \
    --epochs 1 \
    --lr 1e-3 \
    --run-name resnet18_feat_b0.5 \
    --output-dir checkpoints/resnet18_feat_b0.5 \
    --eval-final-only

echo "[3/24] ResNet18 - Feature Only β=1.0..."
python train.py \
    --backbone resnet18 \
    --distill \
    --alpha 1.0 --beta 1.0 --gamma 0.0 \
    --batch-size 8 \
    --epochs 1 \
    --lr 1e-3 \
    --run-name resnet18_feat_b1.0 \
    --output-dir checkpoints/resnet18_feat_b1.0 \
    --eval-final-only

echo "[4/24] ResNet18 - Feature Only β=2.0..."
python train.py \
    --backbone resnet18 \
    --distill \
    --alpha 1.0 --beta 2.0 --gamma 0.0 \
    --batch-size 8 \
    --epochs 1 \
    --lr 1e-3 \
    --run-name resnet18_feat_b2.0 \
    --output-dir checkpoints/resnet18_feat_b2.0 \
    --eval-final-only

echo "------------------------------------------------------------"
echo "RESNET18 - LOGIT DISTILLATION (CLIP)"
echo "------------------------------------------------------------"

echo "[5/24] ResNet18 - Logit Only γ=0.5..."
python train.py \
    --backbone resnet18 \
    --distill \
    --alpha 1.0 --beta 0.0 --gamma 0.5 \
    --batch-size 8 \
    --epochs 1 \
    --lr 1e-3 \
    --run-name resnet18_logit_g0.5 \
    --output-dir checkpoints/resnet18_logit_g0.5 \
    --eval-final-only

echo "[6/24] ResNet18 - Logit Only γ=1.0..."
python train.py \
    --backbone resnet18 \
    --distill \
    --alpha 1.0 --beta 0.0 --gamma 1.0 \
    --batch-size 8 \
    --epochs 1 \
    --lr 1e-3 \
    --run-name resnet18_logit_g1.0 \
    --output-dir checkpoints/resnet18_logit_g1.0 \
    --eval-final-only

echo "[7/24] ResNet18 - Logit Only γ=2.0..."
python train.py \
    --backbone resnet18 \
    --distill \
    --alpha 1.0 --beta 0.0 --gamma 2.0 \
    --batch-size 8 \
    --epochs 1 \
    --lr 1e-3 \
    --run-name resnet18_logit_g2.0 \
    --output-dir checkpoints/resnet18_logit_g2.0 \
    --eval-final-only

echo "------------------------------------------------------------"
echo "RESNET18 - COMBINED DISTILLATION"
echo "------------------------------------------------------------"

echo "[8/24] ResNet18 - Combined β=1.0, γ=1.0..."
python train.py \
    --backbone resnet18 \
    --distill \
    --alpha 1.0 --beta 1.0 --gamma 1.0 \
    --batch-size 8 \
    --epochs 1 \
    --lr 1e-3 \
    --run-name resnet18_combined \
    --output-dir checkpoints/resnet18_combined \
    --eval-final-only

# ------------------------------------------------------------
# MOBILENET EXPERIMENTS (8 total)
# ------------------------------------------------------------

echo "------------------------------------------------------------"
echo "MOBILENET - BASELINE"
echo "------------------------------------------------------------"

echo "[9/24] MobileNet - Baseline (no distillation)..."
python train.py \
    --backbone mobilenet_v3_small \
    --batch-size 8 \
    --epochs 1 \
    --lr 1e-3 \
    --run-name mobilenet_baseline \
    --output-dir checkpoints/mobilenet_baseline \
    --eval-final-only

echo "------------------------------------------------------------"
echo "MOBILENET - FEATURE DISTILLATION (DINO)"
echo "------------------------------------------------------------"

echo "[10/24] MobileNet - Feature Only β=0.5..."
python train.py \
    --backbone mobilenet_v3_small \
    --distill \
    --alpha 1.0 --beta 0.5 --gamma 0.0 \
    --batch-size 8 \
    --epochs 1 \
    --lr 1e-3 \
    --run-name mobilenet_feat_b0.5 \
    --output-dir checkpoints/mobilenet_feat_b0.5 \
    --eval-final-only

echo "[11/24] MobileNet - Feature Only β=1.0..."
python train.py \
    --backbone mobilenet_v3_small \
    --distill \
    --alpha 1.0 --beta 1.0 --gamma 0.0 \
    --batch-size 8 \
    --epochs 1 \
    --lr 1e-3 \
    --run-name mobilenet_feat_b1.0 \
    --output-dir checkpoints/mobilenet_feat_b1.0 \
    --eval-final-only

echo "[12/24] MobileNet - Feature Only β=2.0..."
python train.py \
    --backbone mobilenet_v3_small \
    --distill \
    --alpha 1.0 --beta 2.0 --gamma 0.0 \
    --batch-size 8 \
    --epochs 1 \
    --lr 1e-3 \
    --run-name mobilenet_feat_b2.0 \
    --output-dir checkpoints/mobilenet_feat_b2.0 \
    --eval-final-only

echo "------------------------------------------------------------"
echo "MOBILENET - LOGIT DISTILLATION (CLIP)"
echo "------------------------------------------------------------"

echo "[13/24] MobileNet - Logit Only γ=0.5..."
python train.py \
    --backbone mobilenet_v3_small \
    --distill \
    --alpha 1.0 --beta 0.0 --gamma 0.5 \
    --batch-size 8 \
    --epochs 1 \
    --lr 1e-3 \
    --run-name mobilenet_logit_g0.5 \
    --output-dir checkpoints/mobilenet_logit_g0.5 \
    --eval-final-only

echo "[14/24] MobileNet - Logit Only γ=1.0..."
python train.py \
    --backbone mobilenet_v3_small \
    --distill \
    --alpha 1.0 --beta 0.0 --gamma 1.0 \
    --batch-size 8 \
    --epochs 1 \
    --lr 1e-3 \
    --run-name mobilenet_logit_g1.0 \
    --output-dir checkpoints/mobilenet_logit_g1.0 \
    --eval-final-only

echo "[15/24] MobileNet - Logit Only γ=2.0..."
python train.py \
    --backbone mobilenet_v3_small \
    --distill \
    --alpha 1.0 --beta 0.0 --gamma 2.0 \
    --batch-size 8 \
    --epochs 1 \
    --lr 1e-3 \
    --run-name mobilenet_logit_g2.0 \
    --output-dir checkpoints/mobilenet_logit_g2.0 \
    --eval-final-only

echo "------------------------------------------------------------"
echo "MOBILENET - COMBINED DISTILLATION"
echo "------------------------------------------------------------"

echo "[16/24] MobileNet - Combined β=1.0, γ=1.0..."
python train.py \
    --backbone mobilenet_v3_small \
    --distill \
    --alpha 1.0 --beta 1.0 --gamma 1.0 \
    --batch-size 8 \
    --epochs 1 \
    --lr 1e-3 \
    --run-name mobilenet_combined \
    --output-dir checkpoints/mobilenet_combined \
    --eval-final-only

# ------------------------------------------------------------
# VIT EXPERIMENTS (8 total)
# ------------------------------------------------------------

echo "------------------------------------------------------------"
echo "VIT - BASELINE"
echo "------------------------------------------------------------"

echo "[17/24] ViT - Baseline (no distillation)..."
python train.py \
    --backbone vit_tiny \
    --batch-size 4 \
    --epochs 1 \
    --lr 5e-4 \
    --run-name vit_baseline \
    --output-dir checkpoints/vit_baseline \
    --eval-final-only

echo "------------------------------------------------------------"
echo "VIT - FEATURE DISTILLATION (DINO)"
echo "------------------------------------------------------------"

echo "[18/24] ViT - Feature Only β=0.5..."
python train.py \
    --backbone vit_tiny \
    --distill \
    --alpha 1.0 --beta 0.5 --gamma 0.0 \
    --batch-size 4 \
    --epochs 1 \
    --lr 5e-4 \
    --run-name vit_feat_b0.5 \
    --output-dir checkpoints/vit_feat_b0.5 \
    --eval-final-only

echo "[19/24] ViT - Feature Only β=1.0..."
python train.py \
    --backbone vit_tiny \
    --distill \
    --alpha 1.0 --beta 1.0 --gamma 0.0 \
    --batch-size 4 \
    --epochs 1 \
    --lr 5e-4 \
    --run-name vit_feat_b1.0 \
    --output-dir checkpoints/vit_feat_b1.0 \
    --eval-final-only

echo "[20/24] ViT - Feature Only β=2.0..."
python train.py \
    --backbone vit_tiny \
    --distill \
    --alpha 1.0 --beta 2.0 --gamma 0.0 \
    --batch-size 4 \
    --epochs 1 \
    --lr 5e-4 \
    --run-name vit_feat_b2.0 \
    --output-dir checkpoints/vit_feat_b2.0 \
    --eval-final-only

echo "------------------------------------------------------------"
echo "VIT - LOGIT DISTILLATION (CLIP)"
echo "------------------------------------------------------------"

echo "[21/24] ViT - Logit Only γ=0.5..."
python train.py \
    --backbone vit_tiny \
    --distill \
    --alpha 1.0 --beta 0.0 --gamma 0.5 \
    --batch-size 4 \
    --epochs 1 \
    --lr 5e-4 \
    --run-name vit_logit_g0.5 \
    --output-dir checkpoints/vit_logit_g0.5 \
    --eval-final-only

echo "[22/24] ViT - Logit Only γ=1.0..."
python train.py \
    --backbone vit_tiny \
    --distill \
    --alpha 1.0 --beta 0.0 --gamma 1.0 \
    --batch-size 4 \
    --epochs 1 \
    --lr 5e-4 \
    --run-name vit_logit_g1.0 \
    --output-dir checkpoints/vit_logit_g1.0 \
    --eval-final-only

echo "[23/24] ViT - Logit Only γ=2.0..."
python train.py \
    --backbone vit_tiny \
    --distill \
    --alpha 1.0 --beta 0.0 --gamma 2.0 \
    --batch-size 4 \
    --epochs 1 \
    --lr 5e-4 \
    --run-name vit_logit_g2.0 \
    --output-dir checkpoints/vit_logit_g2.0 \
    --eval-final-only

echo "------------------------------------------------------------"
echo "VIT - COMBINED DISTILLATION"
echo "------------------------------------------------------------"

echo "[24/24] ViT - Combined β=1.0, γ=1.0..."
python train.py \
    --backbone vit_tiny \
    --distill \
    --alpha 1.0 --beta 1.0 --gamma 1.0 \
    --batch-size 4 \
    --epochs 1 \
    --lr 5e-4 \
    --run-name vit_combined \
    --output-dir checkpoints/vit_combined \
    --eval-final-only

echo ""
echo "============================================================"
echo "All 24 Training Experiments Complete! Starting Evaluation..."
echo "============================================================"
echo ""

# ============================================================
# EVALUATION
# ============================================================

# Create results directory
mkdir -p results

echo "Evaluating all 24 configurations..."
echo ""

# ------------------------------------------------------------
# RESNET18 EVALUATION (8 experiments)
# ------------------------------------------------------------

echo "------------------------------------------------------------"
echo "RESNET18 EVALUATION"
echo "------------------------------------------------------------"

echo "Evaluating ResNet18 - Baseline..."
python eval_performance.py --checkpoint checkpoints/resnet18_baseline/student_latest.pth --backbone resnet18 --batch-size 8 | tee results/resnet18_baseline.txt

echo "Evaluating ResNet18 - Feature β=0.5..."
python eval_performance.py --checkpoint checkpoints/resnet18_feat_b0.5/student_latest.pth --backbone resnet18 --batch-size 8 | tee results/resnet18_feat_b0.5.txt

echo "Evaluating ResNet18 - Feature β=1.0..."
python eval_performance.py --checkpoint checkpoints/resnet18_feat_b1.0/student_latest.pth --backbone resnet18 --batch-size 8 | tee results/resnet18_feat_b1.0.txt

echo "Evaluating ResNet18 - Feature β=2.0..."
python eval_performance.py --checkpoint checkpoints/resnet18_feat_b2.0/student_latest.pth --backbone resnet18 --batch-size 8 | tee results/resnet18_feat_b2.0.txt

echo "Evaluating ResNet18 - Logit γ=0.5..."
python eval_performance.py --checkpoint checkpoints/resnet18_logit_g0.5/student_latest.pth --backbone resnet18 --batch-size 8 | tee results/resnet18_logit_g0.5.txt

echo "Evaluating ResNet18 - Logit γ=1.0..."
python eval_performance.py --checkpoint checkpoints/resnet18_logit_g1.0/student_latest.pth --backbone resnet18 --batch-size 8 | tee results/resnet18_logit_g1.0.txt

echo "Evaluating ResNet18 - Logit γ=2.0..."
python eval_performance.py --checkpoint checkpoints/resnet18_logit_g2.0/student_latest.pth --backbone resnet18 --batch-size 8 | tee results/resnet18_logit_g2.0.txt

echo "Evaluating ResNet18 - Combined..."
python eval_performance.py --checkpoint checkpoints/resnet18_combined/student_latest.pth --backbone resnet18 --batch-size 8 | tee results/resnet18_combined.txt

# ------------------------------------------------------------
# MOBILENET EVALUATION (8 experiments)
# ------------------------------------------------------------

echo "------------------------------------------------------------"
echo "MOBILENET EVALUATION"
echo "------------------------------------------------------------"

echo "Evaluating MobileNet - Baseline..."
python eval_performance.py --checkpoint checkpoints/mobilenet_baseline/student_latest.pth --backbone mobilenet_v3_small --batch-size 8 | tee results/mobilenet_baseline.txt

echo "Evaluating MobileNet - Feature β=0.5..."
python eval_performance.py --checkpoint checkpoints/mobilenet_feat_b0.5/student_latest.pth --backbone mobilenet_v3_small --batch-size 8 | tee results/mobilenet_feat_b0.5.txt

echo "Evaluating MobileNet - Feature β=1.0..."
python eval_performance.py --checkpoint checkpoints/mobilenet_feat_b1.0/student_latest.pth --backbone mobilenet_v3_small --batch-size 8 | tee results/mobilenet_feat_b1.0.txt

echo "Evaluating MobileNet - Feature β=2.0..."
python eval_performance.py --checkpoint checkpoints/mobilenet_feat_b2.0/student_latest.pth --backbone mobilenet_v3_small --batch-size 8 | tee results/mobilenet_feat_b2.0.txt

echo "Evaluating MobileNet - Logit γ=0.5..."
python eval_performance.py --checkpoint checkpoints/mobilenet_logit_g0.5/student_latest.pth --backbone mobilenet_v3_small --batch-size 8 | tee results/mobilenet_logit_g0.5.txt

echo "Evaluating MobileNet - Logit γ=1.0..."
python eval_performance.py --checkpoint checkpoints/mobilenet_logit_g1.0/student_latest.pth --backbone mobilenet_v3_small --batch-size 8 | tee results/mobilenet_logit_g1.0.txt

echo "Evaluating MobileNet - Logit γ=2.0..."
python eval_performance.py --checkpoint checkpoints/mobilenet_logit_g2.0/student_latest.pth --backbone mobilenet_v3_small --batch-size 8 | tee results/mobilenet_logit_g2.0.txt

echo "Evaluating MobileNet - Combined..."
python eval_performance.py --checkpoint checkpoints/mobilenet_combined/student_latest.pth --backbone mobilenet_v3_small --batch-size 8 | tee results/mobilenet_combined.txt

# ------------------------------------------------------------
# VIT EVALUATION (8 experiments)
# ------------------------------------------------------------

echo "------------------------------------------------------------"
echo "VIT EVALUATION"
echo "------------------------------------------------------------"

echo "Evaluating ViT - Baseline..."
python eval_performance.py --checkpoint checkpoints/vit_baseline/student_latest.pth --backbone vit_tiny --batch-size 4 | tee results/vit_baseline.txt

echo "Evaluating ViT - Feature β=0.5..."
python eval_performance.py --checkpoint checkpoints/vit_feat_b0.5/student_latest.pth --backbone vit_tiny --batch-size 4 | tee results/vit_feat_b0.5.txt

echo "Evaluating ViT - Feature β=1.0..."
python eval_performance.py --checkpoint checkpoints/vit_feat_b1.0/student_latest.pth --backbone vit_tiny --batch-size 4 | tee results/vit_feat_b1.0.txt

echo "Evaluating ViT - Feature β=2.0..."
python eval_performance.py --checkpoint checkpoints/vit_feat_b2.0/student_latest.pth --backbone vit_tiny --batch-size 4 | tee results/vit_feat_b2.0.txt

echo "Evaluating ViT - Logit γ=0.5..."
python eval_performance.py --checkpoint checkpoints/vit_logit_g0.5/student_latest.pth --backbone vit_tiny --batch-size 4 | tee results/vit_logit_g0.5.txt

echo "Evaluating ViT - Logit γ=1.0..."
python eval_performance.py --checkpoint checkpoints/vit_logit_g1.0/student_latest.pth --backbone vit_tiny --batch-size 4 | tee results/vit_logit_g1.0.txt

echo "Evaluating ViT - Logit γ=2.0..."
python eval_performance.py --checkpoint checkpoints/vit_logit_g2.0/student_latest.pth --backbone vit_tiny --batch-size 4 | tee results/vit_logit_g2.0.txt

echo "Evaluating ViT - Combined..."
python eval_performance.py --checkpoint checkpoints/vit_combined/student_latest.pth --backbone vit_tiny --batch-size 4 | tee results/vit_combined.txt

echo ""
echo "============================================================"
echo "ALL 24 EXPERIMENTS COMPLETE!"
echo "============================================================"
echo ""
echo "Results saved to results/*.txt files"
echo ""
echo "Summary of experiments (24 total, 8 per model):"
echo ""
echo "RESNET18 (8 experiments):"
echo "  1. Baseline (no distillation)"
echo "  2-4. Feature Distillation: β=0.5, 1.0, 2.0 (γ=0.0)"
echo "  5-7. Logit Distillation: γ=0.5, 1.0, 2.0 (β=0.0)"
echo "  8. Combined: β=1.0, γ=1.0"
echo ""
echo "MOBILENET (8 experiments):"
echo "  1. Baseline (no distillation)"
echo "  2-4. Feature Distillation: β=0.5, 1.0, 2.0 (γ=0.0)"
echo "  5-7. Logit Distillation: γ=0.5, 1.0, 2.0 (β=0.0)"
echo "  8. Combined: β=1.0, γ=1.0"
echo ""
echo "VIT (8 experiments):"
echo "  1. Baseline (no distillation)"
echo "  2-4. Feature Distillation: β=0.5, 1.0, 2.0 (γ=0.0)"
echo "  5-7. Logit Distillation: γ=0.5, 1.0, 2.0 (β=0.0)"
echo "  8. Combined: β=1.0, γ=1.0"
echo ""
echo "To view TensorBoard logs, run:"
echo "  tensorboard --logdir runs/"
echo ""
