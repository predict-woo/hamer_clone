#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=v100:1
#SBATCH --time=30:00
#SBATCH --mem-per-cpu=4096
#SBATCH --mail-type=BEGIN,END

python demo.py \
    --img_folder example_data --out_folder demo_out \
    --batch_size=1 --side_view --save_mesh --full_frame --body_detector regnety


python process_images.py \
    --images  \
    --out_folder demo_out \
    --batch_size=1 --side_view --save_mesh --full_frame --body_detector regnety


#  cp /cluster/project/cvg/data/H2O/subject4/h2/3/cam2/rgb/000022.png example_data