import mne
import pandas as pd
import numpy as np
import os
import glob

print("🚀 Medical-Grade Frequency Extraction Shuru Ho Rahi Hai...")

DATA_DIR = '/content/drive/MyDrive/Alzheimer_Project/raw_eeg_data/'
all_set_files = glob.glob(os.path.join(DATA_DIR, '**', '*.set'), recursive=True)

if not all_set_files:
    print("❌ Error: Koi .set file nahi mili.")
else:
    print(f"✅ {len(all_set_files)} patients detected. Extracting Brain Frequencies...")

features_list = []

for file_path in all_set_files:
    patient_id = os.path.basename(file_path)
    
    try:
        # Preload=False for RAM safety
        raw = mne.io.read_raw_eeglab(file_path, preload=False, verbose=False)
        
        # FIX 1: Skip pehle 20 seconds (Setup noise) aur sirf beech ka saaf 60 second ka hissa lein
        tmax = min(80, raw.times[-1])
        tmin = min(20, tmax - 10) if tmax > 20 else 0
        raw.crop(tmin=tmin, tmax=tmax)
        raw.load_data(verbose=False)
        
        # FIX 2: Medical Filter (1Hz se 40Hz) - Removes heartbeat and AC power noise
        raw.filter(l_freq=1.0, h_freq=40.0, verbose=False)
        
        # FIX 3: Frequency Band Extraction (Welch's Method)
        psd, freqs = raw.compute_psd(fmin=1, fmax=40, verbose=False).get_data(return_freqs=True)
        mean_psd = np.mean(psd, axis=0) # Average across all sensors
        
        # Brain Waves Mathematically Defined
        delta = np.sum(mean_psd[np.where((freqs >= 1) & (freqs <= 4))])
        theta = np.sum(mean_psd[np.where((freqs > 4) & (freqs <= 8))])
        alpha = np.sum(mean_psd[np.where((freqs > 8) & (freqs <= 12))])
        beta = np.sum(mean_psd[np.where((freqs > 12) & (freqs <= 30))])
        
        # The Golden Biomarker for Alzheimer's
        theta_alpha_ratio = theta / (alpha + 1e-10)
        
        features_list.append({
            'participant_id': patient_id,
            'delta_power': delta,
            'theta_power': theta,
            'alpha_power': alpha,
            'beta_power': beta,
            'theta_alpha_ratio': theta_alpha_ratio
        })
        print(f"✔️ Processed: {patient_id}")
        
    except Exception as e:
        print(f"⚠️ Skipped {patient_id} (Corrupted/Too Short): {e}")

if features_list:
    df = pd.DataFrame(features_list)
    save_path = '/content/drive/MyDrive/Alzheimer_Project/medical_features_tv.csv'
    df.to_csv(save_path, index=False)
    print(f"\n🎉 Success! Medical features '{save_path}' par save ho gaye hain.")