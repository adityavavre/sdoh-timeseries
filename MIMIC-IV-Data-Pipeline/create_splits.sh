#! /bin/bash
python create_pretrain_finetune_splits_primenet.py \
    --data_dir "/home/av38898/projects/sdoh/MIMIC-IV-Data-Pipeline/data/" \
    --mimic_data_dir "/home/av38898/projects/sdoh/MIMIC-IV-Data-Pipeline/mimiciv/1.0" \
    --pretrain_ratio $1 \
    --output_dir "/data/av38898/sdoh/data_multiple_splits/"
