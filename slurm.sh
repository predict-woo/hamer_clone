#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=v100:1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --mail-type=BEGIN,END

python demo.py \
    --img_folder example_data --out_folder demo_out \
    --batch_size=48 --side_view --save_mesh --full_frame