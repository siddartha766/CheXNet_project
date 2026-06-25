import os
from glob import glob
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# ==========================================
# PHASE 2: DATA PREP & PATIENT SPLIT
# ==========================================
print("1. Scanning the vault for image locations (Building GPS Map)...")

# 1. THE GPS MAP
# Hardcoded to your exact MobaXterm mount path
base_dir = '/media/MyPassport/siddarth_data'

# We tell the GPS to look inside all the scattered 'images_00X' folders
all_image_paths = {os.path.basename(x): x for x in glob(f'{base_dir}/images_*/**/*.png', recursive=True)}

print(f"   --> GPS Map created! Found {len(all_image_paths)} X-rays safely on the disk.")

# 2. LOAD THE MASTER RECORD
csv_path = os.path.join(base_dir, 'Data_Entry_2017.csv')
df = pd.read_csv(csv_path)

# 3. ATTACH THE GPS MAP TO THE PATIENT RECORD
# This creates a new column with the exact hard drive path for every image
df['Full_Path'] = df['Image Index'].map(all_image_paths)

# Safety Check: Drop any rows where the image file wasn't found on the hard drive
missing_images = df['Full_Path'].isna().sum()
if missing_images > 0:
    print(f"   --> WARNING: {missing_images} images from the CSV were not found on disk. Dropping them to prevent PyTorch crashes.")
    df = df.dropna(subset=['Full_Path'])

# 4. CREATE THE CLINICAL TARGET (Pneumonia vs. Everything Else)
df['Target'] = df['Finding Labels'].apply(lambda x: 1 if 'Pneumonia' in x else 0)

# 5. THE STANFORD-GRADE PATIENT SPLIT
print("\n2. Securing patient data to prevent AI cheating (No Patient Overlap)...")

# Split 1: Isolate the final 1.26% test patients (The Blind Exam)
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.0126, random_state=42)
train_val_idx, test_idx = next(gss_test.split(df, groups=df['Patient ID']))

train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

# Split 2: Isolate the 5.49% validation patients (The Practice Quizzes)
gss_val = GroupShuffleSplit(n_splits=1, test_size=0.0549, random_state=42)
train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df['Patient ID']))

train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

print(f"   --> Data successfully secured without patient leakage!")
print(f"   --> Training Set: {len(train_df)} images")
print(f"   --> Validation Set: {len(val_df)} images")
print(f"   --> Testing Set (Locked in Vault): {len(test_df)} images")
