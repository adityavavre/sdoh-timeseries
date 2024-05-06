CUDA_VISIBLE_DEVICES=4 python3 finetune.py --niters 40 --lr 0.0001 --batch-size 128 --rec-hidden 128 --n 8000 --quantization 0.016 \
--save 1 --classif --num-heads 1 --learn-emb --dataset physionet --seed 0 --task classification \
--pretrain_model "/data/av38898/sdoh/models/34930.h5" --pooling ave --dev 0 --oversample_pos --path "/data/av38898/sdoh/data/finetune/" \
--output_dir "/data/av38898/sdoh/models/split_0.2/"