#!/bin/bash
# =============================================================================
# Test 3RTPR-UAC models
# =============================================================================

# Baseline models
for p in /path/to/baseline-exp-*/RSTPReid; do
  python run.py --cfg config_model.yml \
    --d-names RSTPReid --output_dir $p \
    --bs 128 --erpt 0.2 --test \
    --sratio 0.4 --fusedim 5120
done

# UACS models
for p in /path/to/uacs-exp-*/RSTPReid; do
  python run.py --cfg config_model.yml \
    --d-names RSTPReid --output_dir $p \
    --bs 128 --erpt 0.2 --test \
    --ccd --ua --sratio 0.4 --fuseddim 5120
done

# UACS-2 models
for p in /path/to/uacs2-exp-*/RSTPReid; do
  python run.py --cfg config_model.yml \
    --d-names RSTPReid --output_dir $p \
    --bs 128 --erpt 0.2 --test \
    --ccd --ua --use-epistemic --sratio 0.4 --fuseddim 5120
done
