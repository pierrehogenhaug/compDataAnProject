from sklearn.preprocessing import StandardScaler
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

pwd = os.getcwd()

def load_csv_data(folder_path, file_name):
    file_path = f"{folder_path}/{file_name}"
    return pd.read_csv(file_path)

def load_excel_data(folder_path, file_name):
    file_path = f"{folder_path}/{file_name}"
    return pd.read_excel(file_path)

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

            #print(f"Processing: {phase_path}")

            bvp_data = load_csv_data(phase_path, "BVP.csv")
            eda_data = load_csv_data(phase_path, "EDA.csv")
            hr_data = load_csv_data(phase_path, "HR.csv")
            temp_data = load_csv_data(phase_path, "TEMP.csv")
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

biosignals = ['EDA', 'HR', 'TEMP', 'BVP']
num_biosignals = len(biosignals)
fig, axs = plt.subplots(num_biosignals, 3, figsize=(18, 6*num_biosignals), sharey='row')

for b_idx, biosignal in enumerate(biosignals):
    for p_idx, phase in enumerate(['phase1', 'phase2', 'phase3']):
        phase_data = combined_dataframe[combined_dataframe['phase'] == phase]
        signal_data = [row[biosignal] for row in phase_data[biosignal.lower()]]
        sns.histplot(signal_data, kde=True, ax=axs[b_idx, p_idx])
        axs[b_idx, p_idx].set_title(f"{biosignal} Distribution for {phase}")
        axs[b_idx, p_idx].set_xlabel(biosignal)
        axs[b_idx, p_idx].set_ylabel("Frequency")

plt.tight_layout()
plt.show()

measures = ['bvp', 'eda', 'hr', 'temp']
mean_correlations = {}

for phase in ['phase1', 'phase2', 'phase3']:
    phase_data = combined_dataframe[combined_dataframe['phase'] == phase]
    
    for measure1 in measures:
        for measure2 in measures:
            if measure1 == measure2:
                continue
            
            pair_key = f"{measure1}-{measure2}"
            if pair_key in mean_correlations:
                continue
            
            reverse_pair_key = f"{measure2}-{measure1}"
            correlations = []
            
            for index, row in phase_data.iterrows():
                data1 = row[measure1][measure1.upper()]
                data2 = row[measure2][measure2.upper()]

                # Resample the data1 to match the data2 length
                data1_resampled = np.interp(np.linspace(0, len(data1), len(data2)), np.arange(len(data1)), data1)

                # Calculate the correlation coefficient
                corr = np.corrcoef(data1_resampled, data2)[0, 1]
                correlations.append(corr)
                
            mean_correlations[pair_key] = np.mean(correlations)
            print(f"Mean correlation between {measure1.upper()} and {measure2.upper()} for {phase}: {mean_correlations[pair_key]}")

            biosignals = ['EDA', 'HR', 'TEMP', 'BVP']
phases = ['phase1', 'phase2', 'phase3']
summary_stats = []

for biosignal in biosignals:
    for phase in phases:
        phase_data = combined_dataframe[combined_dataframe['phase'] == phase]
        signal_data = [row[biosignal] for row in phase_data[biosignal.lower()]]
        combined_signal_data = pd.concat(signal_data)
        
        mean = combined_signal_data.mean()
        median = combined_signal_data.median()
        std_dev = combined_signal_data.std()
        
        summary_stats.append({
            'biosignal': biosignal,
            'phase': phase,
            'mean': mean,
            'median': median,
            'std_dev': std_dev
        })

summary_stats_df = pd.DataFrame(summary_stats)
print(summary_stats_df)