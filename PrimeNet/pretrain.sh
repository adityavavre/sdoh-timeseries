CUDA_VISIBLE_DEVICES=6 python3 pretrain.py --niters 10 --lr 0.0001 --batch-size 64 --rec-hidden 128 --n 8000 --quantization 0.016 \
--save 1 --classif --num-heads 1 --learn-emb --dataset physionet --seed 0 --add_pos --transformer \
--pooling bert --pretrain_tasks full2 --segment_num 16 --mask_ratio_per_seg 0.5 --dev 0 --path "/data/av38898/sdoh/data_multiple_splits/split_0.2/pretrain/" \
--output_dir "/data/av38898/sdoh/models/split_0.2/"
