from sklearn.preprocessing import StandardScaler
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

pwd = os.getcwd()

def load_csv_data(phase_path, file_name):
    file_path = os.path.join(phase_path, file_name)
    if not os.path.exists(file_path):
        return None
    data = pd.read_csv(file_path)
    return convert_time_to_datetime(data)

def load_excel_data(folder_path, file_name):
    file_path = f"{folder_path}/{file_name}"
    return pd.read_excel(file_path)

# convert the 'time' column to a datetime object and set it as the index
def convert_time_to_datetime(data):
    if 'time' not in data.columns:
        return data
    if data['time'].dtype == 'datetime64[ns]':
        return data.set_index('time')
    return data.set_index(pd.to_datetime(data['time']))

# resample the data at a common frequency
def resample_data(data, frequency='S'):
    if data is None:
        return None
    return data.resample(frequency).mean().interpolate(method='linear')

def combine_data(data_path):
    combined_data = []
    
    for day_folder in sorted(os.listdir(data_path)):
        day_path = os.path.join(data_path, day_folder)
        if not os.path.isdir(day_path):
            continue
            
        if day_folder == "D1_3":
            for d1_3_subfolder in sorted(os.listdir(day_path)):
                d1_3_subfolder_path = os.path.join(day_path, d1_3_subfolder)
                if not os.path.isdir(d1_3_subfolder_path):
                    continue
                
                for id_folder in sorted(os.listdir(d1_3_subfolder_path)):
                    process_id_folder(d1_3_subfolder_path, id_folder, combined_data)
        else:
            for id_folder in sorted(os.listdir(day_path)):
                process_id_folder(day_path, id_folder, combined_data)
                
    return combined_data

def process_id_folder(day_path, id_folder, combined_data):
    id_path = os.path.join(day_path, id_folder)
    if not os.path.isdir(id_path):
        return

    for round_folder in sorted(os.listdir(id_path)):
        round_path = os.path.join(id_path, round_folder)
        if not os.path.isdir(round_path):
            continue

        for phase_folder in sorted(os.listdir(round_path)):
            phase_path = os.path.join(round_path, phase_folder)
            if not os.path.isdir(phase_path):
                continue

            bvp_data = resample_data(load_csv_data(phase_path, "BVP.csv"))
            eda_data = resample_data(load_csv_data(phase_path, "EDA.csv"))
            hr_data = resample_data(load_csv_data(phase_path, "HR.csv"))
            temp_data = resample_data(load_csv_data(phase_path, "TEMP.csv"))
            response_data = load_csv_data(phase_path, "response.csv")

            combined_data.append({
                'day': os.path.split(day_path)[-1],
                'id': os.path.split(id_path)[-1],
                'round': os.path.split(round_path)[-1],
                'phase': os.path.split(phase_path)[-1],
                'bvp': bvp_data,
                'eda': eda_data,
                'hr': hr_data,
                'temp': temp_data,
                'response': response_data
            })

data_path = os.path.join(pwd, 'dataset')
combined_data = combine_data(data_path)

combined_dataframe = pd.DataFrame.from_records(combined_data)


### Data Preprocessing
def combine_biodata(bvp, eda, hr, temp):
    return pd.concat([bvp, eda, hr, temp], axis=1)

combined_dataframe['combined_biodata'] = combined_dataframe.apply(lambda row: combine_biodata(row['bvp'], row['eda'], row['hr'], row['temp']), axis=1)

def extract_features(combined_biodata):
    features = {
        'bvp_mean': combined_biodata['BVP'].mean(),
        'bvp_std': combined_biodata['BVP'].std(),
        'bvp_min': combined_biodata['BVP'].min(),
        'bvp_max': combined_biodata['BVP'].max(),
        'bvp_median': combined_biodata['BVP'].median(),
        'eda_mean': combined_biodata['EDA'].mean(),
        'eda_std': combined_biodata['EDA'].std(),
        'eda_min': combined_biodata['EDA'].min(),
        'eda_max': combined_biodata['EDA'].max(),
        'eda_median': combined_biodata['EDA'].median(),
        'hr_mean': combined_biodata['HR'].mean(),
        'hr_std': combined_biodata['HR'].std(),
        'hr_min': combined_biodata['HR'].min(),
        'hr_max': combined_biodata['HR'].max(),
        'hr_median': combined_biodata['HR'].median(),
        'temp_mean': combined_biodata['TEMP'].mean(),
        'temp_std': combined_biodata['TEMP'].std(),
        'temp_min': combined_biodata['TEMP'].min(),
        'temp_max': combined_biodata['TEMP'].max(),
        'temp_median': combined_biodata['TEMP'].median()
    }
    return features

combined_dataframe['features'] = combined_dataframe['combined_biodata'].apply(extract_features)

features_dataframe = combined_dataframe[['day', 'id', 'round', 'phase', 'features']].copy()
features_dataframe = pd.concat([features_dataframe.drop(['features'], axis=1), features_dataframe['features'].apply(pd.Series)], axis=1)

scaler = StandardScaler()
numeric_columns = ['bvp_mean', 'bvp_std', 'bvp_min', 'bvp_max', 'bvp_median',
                   'eda_mean', 'eda_std', 'eda_min', 'eda_max', 'eda_median',
                   'hr_mean', 'hr_std', 'hr_min', 'hr_max', 'hr_median',
                   'temp_mean', 'temp_std', 'temp_min', 'temp_max', 'temp_median']

features_dataframe[numeric_columns] = scaler.fit_transform(features_dataframe[numeric_columns])
