#!/bin/bash
#SBATCH --job-name=TIP.UAC
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-23:00:00
#SBATCH --output=xxxxxx.log
#SBATCH --nodelist=node004

# =============================================================================
# 3RTPR-UAC: Uncertainty-Aware Consensus Training
# Dataset: RSTPReid, CUHK-PEDES, ICFG-PEDES
# Architecture: ViT-B/16 + Fused Representation + UACS-2
# =============================================================================

# ============================================================
# Experiment 1: Baseline (original CCD + dynamic margin)
# ============================================================
for seed in 123 33035 1910 888; do
  python run.py --cfg config_model.yml --seed $seed \
    --d-names RSTPReid -n hihi --l-name sdm --bs 128 --saug-text --erpi 0.5 --erpt 0.2 \
    --ccd --ldynamic --ldynamic-t 6 --ldynamic-m min \
    --lossweight-sdm 1 --sratio 0.4 --fusedim 5120

  python run.py --cfg config_model.yml --seed $seed \
    --d-names CUHK-PEDES -n hihi --l-name sdm --bs 128 --saug-text --erpi 0.5 --erpt 0.2 \
    --ccd --ldynamic --ldynamic-t 6 --ldynamic-m min \
    --lossweight-sdm 1 --sratio 0.4 --fusedim 5120

  python run.py --cfg config_model.yml --seed $seed \
    --d-names ICFG-PEDES -n hihi --l-name sdm --bs 128 --saug-text --erpi 0.5 --erpt 0.2 \
    --ccd --ldynamic --ldynamic-t 6 --ldynamic-m min \
    --lossweight-sdm 1 --sratio 0.4 --fusedim 5120
done

# ============================================================
# Experiment 2: UACS (Multi-Signal GMM without Epistemic)
# ============================================================
for seed in 123 33035 1910 888; do
  python run.py --cfg config_model.yml --seed $seed \
    --d-names RSTPReid -n hihi --l-name sdm --bs 128 --saug-text --erpi 0.5 --erpt 0.2 \
    --ccd --ua --ua-wsim 0.15 --ua-wagree 0.1 --ua-clean 0.8 --ua-minw 0.1 \
    --ldynamic --ldynamic-t 6 --ldynamic-m min \
    --lossweight-sdm 1 --sratio 0.4 --fusedim 5120

  python run.py --cfg config_model.yml --seed $seed \
    --d-names CUHK-PEDES -n hihi --l-name sdm --bs 128 --saug-text --erpi 0.5 --erpt 0.2 \
    --ccd --ua --ua-wsim 0.15 --ua-wagree 0.1 --ua-clean 0.8 --ua-minw 0.1 \
    --ldynamic --ldynamic-t 6 --ldynamic-m min \
    --lossweight-sdm 1 --sratio 0.4 --fusedim 5120

  python run.py --cfg config_model.yml --seed $seed \
    --d-names ICFG-PEDES -n hihi --l-name sdm --bs 128 --saug-text --erpi 0.5 --erpt 0.2 \
    --ccd --ua --ua-wsim 0.15 --ua-wagree 0.1 --ua-clean 0.8 --ua-minw 0.1 \
    --ldynamic --ldynamic-t 6 --ldynamic-m min \
    --lossweight-sdm 1 --sratio 0.4 --fusedim 5120
done

# ============================================================
# Experiment 3: UACS-2 (Multi-Signal GMM + Epistemic via MC Dropout)
# Vary epistemic_alpha: 0.2, 0.3, 0.4
# ============================================================
for seed in 123 33035 1910 888; do
  for alpha in 0.2 0.3 0.4; do
    python run.py --cfg config_model.yml --seed $seed \
      --d-names RSTPReid -n hihi --l-name sdm --bs 128 --saug-text --erpi 0.5 --erpt 0.2 \
      --ccd --ua --ua-wsim 0.15 --ua-wagree 0.1 --ua-clean 0.8 --ua-minw 0.1 \
      --use-epistemic --epistemic-alpha $alpha --mc-samples 5 --mc-dropout-rate 0.1 \
      --ldynamic --ldynamic-t 6 --ldynamic-m min \
      --lossweight-sdm 1 --sratio 0.4 --fusedim 5120

    python run.py --cfg config_model.yml --seed $seed \
      --d-names CUHK-PEDES -n hihi --l-name sdm --bs 128 --saug-text --erpi 0.5 --erpt 0.2 \
      --ccd --ua --ua-wsim 0.15 --ua-wagree 0.1 --ua-clean 0.8 --ua-minw 0.1 \
      --use-epistemic --epistemic-alpha $alpha --mc-samples 5 --mc-dropout-rate 0.1 \
      --ldynamic --ldynamic-t 6 --ldynamic-m min \
      --lossweight-sdm 1 --sratio 0.4 --fusedim 5120

    python run.py --cfg config_model.yml --seed $seed \
      --d-names ICFG-PEDES -n hihi --l-name sdm --bs 128 --saug-text --erpi 0.5 --erpt 0.2 \
      --ccd --ua --ua-wsim 0.15 --ua-wagree 0.1 --ua-clean 0.8 --ua-minw 0.1 \
      --use-epistemic --epistemic-alpha $alpha --mc-samples 5 --mc-dropout-rate 0.1 \
      --ldynamic --ldynamic-t 6 --ldynamic-m min \
      --lossweight-sdm 1 --sratio 0.4 --fusedim 5120
  done
done

# ============================================================
# Experiment 4: UACS-2 with varying MC Dropout samples (3, 5, 7)
# ============================================================
for seed in 123 33035; do
  for mc in 3 5 7; do
    python run.py --cfg config_model.yml --seed $seed \
      --d-names RSTPReid -n hihi --l-name sdm --bs 128 --saug-text --erpi 0.5 --erpt 0.2 \
      --ccd --ua --ua-wsim 0.15 --ua-wagree 0.1 --ua-clean 0.8 --ua-minw 0.1 \
      --use-epistemic --epistemic-alpha 0.3 --mc-samples $mc --mc-dropout-rate 0.1 \
      --ldynamic --ldynamic-t 6 --ldynamic-m min \
      --lossweight-sdm 1 --sratio 0.4 --fusedim 5120
  done
done
