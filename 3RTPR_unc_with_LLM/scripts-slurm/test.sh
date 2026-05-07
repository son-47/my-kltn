#!/bin/bash
#SBATCH --job-name=TIP.Rev1 # define job name
#SBATCH --nodes=1             # define node
#SBATCH --gpus-per-node=1     # define gpu limmit in 1 node
#SBATCH --ntasks=1            # define number tasks
#SBATCH --cpus-per-task=8    # There are 24 CPU cores
#SBATCH --time=7-23:00:00     # Max running time = 10 minutes
#SBATCH --output=xxxxxx.log
#SBATCH --nodelist=node004
#SBATCH --chdir=/data2/cmdir/home/giangnl1/work_son/3RTPR_unc_with_LLM
#SBATCH --export=ALL
# Load module
# Some module avail:
# source env/bin/activate

# ## pytorch-extra-py39-cuda11.2-gcc9
# module load cuda11.2/toolkit/11.2.2
# module load pytorch-py39-cuda11.2-gcc9/1.9.1
# module load pytorch-extra-py39-cuda11.2-gcc9
# module load opencv4-py39-cuda11.2-gcc9/4.5.4

# noisy_rate=0 #0.0 0.2 0.5 0.8
# noisy_file=./noiseindex/${DATASET_NAME}_${noisy_rate}.npy
# python run.py  --cfg config_model.yml \
#   --d-names CUHK-PEDES --output_dir /data2/cmdir/home/giangnl1/work_son/3RTPR_unc_with_LLM/datasets/saves/CUHK-PEDES/20260429_073452_hihi_9097 \
#   --bs 16 --erpt 0.2 --test \
#   --ldynamic  --ldynamic-t 2 --lossweight-sdm 0  --sratio 0.4 --fusedim 5120  \
#   --noisy_file $noisy_file  --noisy_rate 0 --lrx 0.1   

# python run.py --cfg config_model.yml \
#   --d-names CUHK-PEDES --test --output_dir /path/to/output \
#   --annotation_source raw
python run.py --cfg config_model.yml --d-names CUHK-PEDES --annotation_source raw --bs 16 --test --sratio 0.4 --fusedim 5120  --output_dir /data2/cmdir/home/giangnl1/work_son/3RTPR_unc_with_LLM/datasets/saves/CUHK-PEDES/20260505_150305_hihi_8759