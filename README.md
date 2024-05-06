# Time Series Representation Learning on MIMIC-IV

1. Run the MIMIC-IV-Data-Pipeline notebook using the instructions given in the notebook to extract MIMIC-IV time series data.
2. Run `create_splits.sh` script in the MIMIC-IV-Data-Pipeline folder to create pretrain and finetune splits for training the PrimeNet model.
3. Finally pretrain a model using the `pretrain.sh` script and finetune a model using the `finetune.sh` script in the PrimeNet folder.
