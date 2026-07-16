import pandas as pd
import os

def get_prepared_data():
    try:
        # Nayi Medical File
        features_path = '/content/drive/MyDrive/Alzheimer_Project/medical_features_tv.csv'
            
        features = pd.read_csv(features_path)
        participants = pd.read_csv('/content/drive/MyDrive/Alzheimer_Project/participants.tsv', delimiter='\t')
        
        features['participant_id'] = features['participant_id'].str.replace('_task-eyesclosed_eeg.set', '', regex=False)
        
        features.set_index('participant_id', inplace=True)
        participants.set_index('participant_id', inplace=True)
        
        data = features.merge(participants, left_index=True, right_index=True)
        data['Group_color'] = data['Group'].map({'A': 'Dementia', 'F': 'Dementia', 'C': 'Healthy'})
        
        # Naye Asli Brain Wave Features
        feature_cols = ['delta_power', 'theta_power', 'alpha_power', 'beta_power', 'theta_alpha_ratio']
        
        X = data[feature_cols]
        y = data['Group_color']
        
        return X, y
    except Exception as e:
        print(f"Data loading error: {e}")
        return None, None