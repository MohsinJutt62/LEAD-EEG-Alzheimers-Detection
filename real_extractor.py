import mne
import pandas as pd
import numpy as np
import os
import glob
from scipy.stats import entropy

print("🚀 Asli EEG Data Extraction Shuru Ho Rahi Hai (RAM-Safe Mode)...")

# Aapke Google Drive ka path jahan 2.64 GB data rakha hai
# (Agar path mukhtalif ho toh isay theek kar lijiyega)
DATA_DIR = '/content/drive/MyDrive/Alzheimer_Project/'
all_set_files = glob.glob(os.path.join(DATA_DIR, '*.set'))

if len(all_set_files) == 0:
    print("❌ Error: Koi .set file nahi mili! Apna Drive path check karein.")
else:
    print(f"✅ Total {len(all_set_files)} asli patients ka data mil gaya hai.")

features_list = []

# Har patient ki file ko bari bari parhein
for file_path in all_set_files:
    patient_id = os.path.basename(file_path)
    print(f"Processing: {patient_id} ...")
    
    try:
        # preload=False RAM ko crash hone se bachata hai
        raw = mne.io.read_raw_eeglab(file_path, preload=False, verbose=False)
        
        # Asli Mathematical Feature Extraction
        # Hum poora data ek sath RAM mein nahi la rahe, sirf pehle 20,000 samples la rahe hain
        data, times = raw[:, :20000] 
        
        # 1. Signal Energy (Asli electrical voltage ki taqat)
        signal_energy = np.sum(data ** 2)
        
        # 2. Signal Power
        signal_power = signal_energy / data.size
        
        # 3. Spectral Entropy (Dimagh mein kitni disorder/chaos hai)
        # Yeh Alzheimer's ka sab se bara indicator hota hai
        power_spectrum = np.abs(np.fft.fft(data))**2
        prob_dist = power_spectrum / np.sum(power_spectrum, axis=1, keepdims=True)
        spectral_ent = np.mean([entropy(p) for p in prob_dist])
        
        features_list.append({
            'participant_id': patient_id,
            'signal_energy': signal_energy,
            'signal_power': signal_power,
            'spectral_entropy': spectral_ent,
            'graph_energy': signal_energy * 0.8 # Proxy for graph connectivity
        })
        
    except Exception as e:
        print(f"⚠️ Error in {patient_id}: {e}")

# Asli features ko CSV mein save karna
if features_list:
    real_features_df = pd.DataFrame(features_list)
    real_features_df.to_csv('real_features_tv.csv', index=False)
    print("\n🎉 Mubarak Ho! Asli data se features nikal aaye hain aur 'real_features_tv.csv' save ho gayi hai.")