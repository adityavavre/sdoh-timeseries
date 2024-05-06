import pandas as pd
import pickle
import os
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import argparse
from ast import literal_eval

def extract_data(data, patients, feat_to_id, time_window, num_features):
    # extract pretrain data according to the PrimeNet Format for a given set of stay_ids. 
    # Each data point is an array of length 2T+1 where T is the time length. 
    # First T value denote the features, next T values denote the mask and the final value denotes the actual time.
    print("Extracting Data!")
    extracted_data = []
    labels = []
    for patient in tqdm(patients):
        patient_data = []
        labels.append(data[patient]['label'])
        for time in range(time_window):
            data_point = [0]*(2*num_features)
            data_point.append(time+1)
            ## check chart signals
            if 'signal' in data[patient]['Chart'] and 'val' in data[patient]['Chart']:
                for chart_feat in data[patient]['Chart']['signal'].keys():
                    feat_id = feat_to_id['chart_'+str(chart_feat)]
                    if data[patient]['Chart']['signal'][chart_feat][time]:
                        ## set value
                        data_point[feat_id] = data[patient]['Chart']['val'][chart_feat][time]
                        ## set mask
                        data_point[NUM_FEATURES+feat_id] = 1
            
            ## check med signals
            if 'signal' in data[patient]['Med'] and 'amount' in data[patient]['Med']:
                for med_feat in data[patient]['Med']['signal'].keys():
                    feat_id = feat_to_id['med_'+str(med_feat)]
                    if data[patient]['Med']['signal'][med_feat][time]:
                        ## set value
                        data_point[feat_id] = data[patient]['Med']['amount'][med_feat][time]
                        ## set mask
                        data_point[NUM_FEATURES+feat_id] = 1

            ## check proc signals
            for proc_feat in data[patient]['Proc'].keys():
                feat_id = feat_to_id['proc_'+str(proc_feat)]
                if data[patient]['Proc'][proc_feat][time]:
                    ## set value
                    data_point[feat_id] = 1
                    ## set mask
                    data_point[NUM_FEATURES+feat_id] = 1

            ## check out signals
            for out_feat in data[patient]['Out'].keys():
                feat_id = feat_to_id['out_'+str(out_feat)]
                if data[patient]['Out'][out_feat][time]:
                    ## set value
                    data_point[feat_id] = 1
                    ## set mask
                    data_point[NUM_FEATURES+feat_id] = 1
            
            patient_data.append(data_point)
        extracted_data.append(patient_data)
    return extracted_data, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to Pipeline Extracted Data')
    parser.add_argument('--mimic_data_dir', type=str, required=True, help='Path to Mimic Data eg: ./mimiciv/1.0/')
    parser.add_argument('--pretrain_ratio', type=str, default=0.8, help='Ratio of data to use for creating the pretraining data. Can either be a single ratio eg. 0.8. Or a list of ratios as a str eg. \"[0.2,0.4]\"')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to store the data in')
    args = parser.parse_args()

    # data_dir = "/home/av38898/projects/sdoh/MIMIC-IV-Data-Pipeline/data/"
    # mimic_data_dir = "/home/av38898/projects/sdoh/MIMIC-IV-Data-Pipeline/mimiciv/1.0"
    # output_dir = "/data/av38898/sdoh/data_multiple_splits/"
    # pretrain_ratio = 0.8

    if os.path.exists(f"{args.data_dir}/cohort/patients_visit.csv.gz"):
        print(f"Reading from {args.data_dir}/cohort/patients_visit.csv.gz")
        patients_visit_df = pd.read_csv(f"{args.data_dir}/cohort/patients_visit.csv.gz", compression='gzip', header=0, index_col=None)
    else:
        patients_df = pd.read_csv(f"{args.mimic_data_dir}/core/patients.csv.gz", compression='gzip', header=0, index_col=None)
        visit_df = pd.read_csv(f"{args.mimic_data_dir}/icu/icustays.csv.gz", compression='gzip', header=0, index_col=None)

        patients_df['anchor_year_group_mid'] = (patients_df['anchor_year_group'].str.slice(start=-4).astype(int) + patients_df['anchor_year_group'].str.slice(start=0, stop=4).astype(int))//2
        patients_visit_df = patients_df.merge(visit_df, how='inner', left_on='subject_id', right_on='subject_id')[['stay_id', 'anchor_year_group_mid']]
        
        print(f"Saving to {args.data_dir}/cohort/patients_visit.csv.gz for subsequent experiments")
        patients_visit_df.to_csv(f"{args.data_dir}/cohort/patients_visit.csv.gz", compression="gzip", index=None)

    f = open(f"{args.data_dir}/dict/dataDic", 'rb')
    data = pickle.load(f)
    f.close()

    data_patients = set(data.keys())

    ## Note that paitents actually refer to stay ids
    patients_2016 = set(patients_visit_df.loc[patients_visit_df['anchor_year_group_mid']<=2016]['stay_id'].unique())
    patients_after_2016 = set(patients_visit_df.loc[patients_visit_df['anchor_year_group_mid']>2016]['stay_id'].unique())

    patients_2016 = patients_2016.intersection(data_patients)
    patients_after_2016 = patients_after_2016.intersection(data_patients)
    print("# Records before 2016:", len(patients_2016))
    print("# Records after 2016:", len(patients_after_2016))
    print("Records after 2016 belong to the TEST Split!")

    ## Collect all features to convert them into ids
    all_chart_features = set()
    all_med_features = set()
    all_proc_features = set()
    all_out_features = set()
    for patient in list(patients_2016.union(patients_after_2016)):
        try:
            all_chart_features.update(data[patient]['Chart']['signal'].keys())
        except Exception as e:
            continue
        
        try:
            all_med_features.update(data[patient]['Med']['signal'].keys())
        except Exception as e:
            continue

        try:
            all_proc_features.update(data[patient]['Proc'].keys())
        except Exception as e:
            continue

        try:
            all_out_features.update(data[patient]['Out'].keys())
        except Exception as e:
            continue
    
    print(f"#Chart Features: {len(all_chart_features)}\n#Med Features: {len(all_med_features)}\n#Proc Features: {len(all_proc_features)}\n#Out Features: {len(all_out_features)}")
    all_chart_features, all_med_features, all_proc_features, all_out_features = sorted(list(all_chart_features)), sorted(list(all_med_features)), sorted(list(all_proc_features)), sorted(list(all_out_features))
    # chart_to_id = {v:k for k,v in enumerate(all_chart_features)}
    # med_to_id = {v:k+len(chart_to_id) for k,v in enumerate(all_med_features)}
    # proc_to_id = {v:k+len(chart_to_id)+len(med_to_id) for k,v in enumerate(all_proc_features)}
    # out_to_id = {v:k+len(chart_to_id)+len(med_to_id)+len(proc_to_id) for k,v in enumerate(all_out_features)}
    feat_to_id = {'chart_'+str(v):k for k,v in enumerate(all_chart_features)}
    feat_to_id.update({'med_'+str(v):k+len(all_chart_features) for k,v in enumerate(all_med_features)})
    feat_to_id.update({'proc_'+str(v):k+len(all_chart_features)+len(all_med_features) for k,v in enumerate(all_proc_features)})
    feat_to_id.update({'out_'+str(v):k+len(all_chart_features)+len(all_med_features)+len(all_proc_features) for k,v in enumerate(all_out_features)})

    # total number of features
    # NUM_FEATURES = len(chart_to_id)+len(med_to_id)+len(proc_to_id)+len(out_to_id)
    NUM_FEATURES = len(all_chart_features)+len(all_med_features)+len(all_proc_features)+len(all_out_features)
    print(f"#Total Features: {NUM_FEATURES}")
    # Time Window extracted from the Data Pipeline
    TIME_WINDOW = 72
    # get extracted data
    data_pretrain, data_pretrain_labels = extract_data(data, patients_2016, feat_to_id, TIME_WINDOW, NUM_FEATURES)
    data_test, data_test_labels = extract_data(data, patients_after_2016, feat_to_id, TIME_WINDOW, NUM_FEATURES)
    # convert to torch tensor
    print("Converting data to torch tensor. Might take some time!")
    data_pretrain_pt = torch.tensor(data_pretrain)
    data_pretrain_labels_pt = torch.tensor(data_pretrain_labels)
    test_X = torch.tensor(data_test)
    test_y = torch.tensor(data_test_labels)
    print(f"Test Data Shape: {test_X.shape} {test_y.shape}")
    pretrain_ratios = literal_eval(args.pretrain_ratio)
    if type(pretrain_ratios) == float:
        pretrain_ratios = [pretrain_ratios]
    print(f"Using pretrain ratios: {pretrain_ratios}")
    os.makedirs(args.output_dir, exist_ok=True)
    for pretrain_ratio in pretrain_ratios:
        print(f"Pretrain Ratio: {pretrain_ratio}")
        os.makedirs(f"{args.output_dir}/split_{pretrain_ratio}", exist_ok=True)
        os.makedirs(f"{args.output_dir}/split_{pretrain_ratio}/pretrain/", exist_ok=True)
        os.makedirs(f"{args.output_dir}/split_{pretrain_ratio}/finetune/", exist_ok=True)

        # create pretrain-finetune split according to the specified ratio
        pretrain_X = data_pretrain_pt[:int(data_pretrain_pt.shape[0]*pretrain_ratio)]
        pretrain_y =  data_pretrain_pt[:int(data_pretrain_pt.shape[0]*pretrain_ratio)] # not used because it is pretraining data (self-supervised)
        finetune_X = data_pretrain_pt[int(data_pretrain_pt.shape[0]*pretrain_ratio):]
        finetune_y =  data_pretrain_pt[int(data_pretrain_pt.shape[0]*pretrain_ratio):]

        # get train val split for pretrain and finetune data. Test split is already generated and fixed.
        pretrain_train_X, pretrain_val_X, pretrain_train_y, pretrain_val_y = train_test_split(pretrain_X, pretrain_y, test_size=0.1, random_state=42)
        finetune_train_X, finetune_val_X, finetune_train_y, finetune_val_y = train_test_split(finetune_X, finetune_y, test_size=0.1, random_state=42)
        print(f'#Pretrain examples- Train: {pretrain_train_X.shape[0]} Val: {pretrain_val_X.shape[0]}')
        print(f'#Finetune examples- Train: {finetune_train_X.shape[0]} Val: {finetune_val_X.shape[0]} Test: {test_X.shape[0]}')

        # save the data in the directory structure needed for PrimeNet
        print(f"Saving data to {args.output_dir}/split_{pretrain_ratio}/")
        # pretrain data
        # labels for pretrain data are not needed and hence not saved
        torch.save(pretrain_train_X, f'{args.output_dir}/split_{pretrain_ratio}/pretrain/X_train.pt')
        torch.save(pretrain_val_X, f'{args.output_dir}/split_{pretrain_ratio}/pretrain/X_val.pt')
        # finetune data
        torch.save(finetune_train_X, f'{args.output_dir}/split_{pretrain_ratio}/finetune/X_train.pt')
        torch.save(finetune_train_y, f'{args.output_dir}/split_{pretrain_ratio}/finetune/y_train.pt')
        torch.save(finetune_val_X, f'{args.output_dir}/split_{pretrain_ratio}/finetune/X_val.pt')
        torch.save(finetune_val_y, f'{args.output_dir}/split_{pretrain_ratio}/finetune/y_val.pt')
        torch.save(test_X, f'{args.output_dir}/split_{pretrain_ratio}/finetune/X_test.pt')
        torch.save(test_y, f'{args.output_dir}/split_{pretrain_ratio}/finetune/y_test.pt')
        print("Data Saved!")

    # save feat_to_id dict for future use
    print(f"Saving feat_to_id.json to {args.output_dir}")
    with open(f"{args.output_dir}/feat_to_id.json", "w") as f:
        json.dump(feat_to_id, f)
    print("Done!")

# Pretrain Ratio: 0.2
# #Pretrain examples- Train: 3087 Val: 344
# #Finetune examples- Train: 12354 Val: 1373 Test: 4422
# Saving data to /data/av38898/sdoh/data_multiple_splits//split_0.2/
# Data Saved!
# Pretrain Ratio: 0.4
# #Pretrain examples- Train: 6176 Val: 687
# #Finetune examples- Train: 9265 Val: 1030 Test: 4422
# Saving data to /data/av38898/sdoh/data_multiple_splits//split_0.4/
# Data Saved!
# Pretrain Ratio: 0.6
# #Pretrain examples- Train: 9264 Val: 1030
# #Finetune examples- Train: 6177 Val: 687 Test: 4422
# Saving data to /data/av38898/sdoh/data_multiple_splits//split_0.6/
# Data Saved!
# Pretrain Ratio: 0.8
# #Pretrain examples- Train: 12353 Val: 1373
# #Finetune examples- Train: 3088 Val: 344 Test: 4422
# Saving data to /data/av38898/sdoh/data_multiple_splits//split_0.8/
