import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# 🌟 THE CORRECTED BRIDGE: Pointing to the exact file that holds the dataframes
from cell_3 import train_df, val_df

# (And keep your transforms import pointing to wherever those are saved)
from cell_4 import train_transform_pipeline, val_transform_pipeline

print("4. Constructing the PyTorch Conveyor Belts...")

class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # 1. Look at the exact row
        row = self.dataframe.iloc[idx]
        
        # 2. Grab the GPS path and open the image
        img_path = row['Full_Path']
        image = Image.open(img_path).convert('RGB')
        
        # 3. Apply the Librarian's rules
        if self.transform:
            image = self.transform(image)
            
        # 4. Grab the math target (1.0 for Pneumonia, 0.0 for Not)
        label = float(row['Target'])
        
        return image, torch.tensor([label])

# Build the Datasets using the dataframes we generated in Cell 2
train_dataset = ChestXrayDataset(train_df, transform=train_transform_pipeline)
val_dataset = ChestXrayDataset(val_df, transform=val_transform_pipeline)

# Build the DataLoaders (Batch size strictly locked to 16 per the base paper)
BATCH_SIZE = 16 

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"   --> Conveyor belts active! Processing {BATCH_SIZE} X-rays per batch.")
