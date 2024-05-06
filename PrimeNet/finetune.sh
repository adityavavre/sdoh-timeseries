CUDA_VISIBLE_DEVICES=$3 python3 finetune.py --niters 40 --lr 0.0001 --batch-size 128 --rec-hidden 128 --n 8000 --quantization 0.016 \
--save 1 --classif --num-heads 1 --learn-emb --dataset physionet --seed 0 --task classification \
--pretrain_model "/data/av38898/sdoh/models/split_$1/$2.h5" --pooling ave --dev 0 --path "/data/av38898/sdoh/data_multiple_splits/split_$1/finetune/" \
--output_dir "/data/av38898/sdoh/models/split_$1/"