import torch
import torch.nn as nn
import torch.optim as optim

# 🌟 THE MASTER BRIDGES
from cell_1 import device
from cell_7 import model

# ==========================================
# PHASE 4: THE RED PEN & THE ENGINE
# ==========================================
print("7. Equipping the Heavy Red Pen and starting the Engine...")

# 1. The Heavy Red Pen (Weighted Loss Function)
# We apply a weight of 86.43 to force the AI to care about the minority Pneumonia class
clinical_weight = torch.tensor([86.43]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=clinical_weight)

# 2. The Engine (Optimizer)
# Adam is the standard, state-of-the-art engine. 
# lr=0.00001 (Learning Rate) is a "slow crawl" so we don't accidentally erase the pre-trained memory
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

print("   --> Red Pen and Engine active! The architecture is complete.")
