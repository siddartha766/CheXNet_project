# 1. THE MATH & TABLE TOOLS
import numpy as np
import pandas as pd

# 2. THE AI BRAIN TOOLS (PyTorch)
import torch
import torch.nn as nn
import torch.optim as optim

# 3.DATASET AND DATALOADER(creates the master index of where every image lives on your hard drive.)
from torch.utils.data import Dataset, DataLoader #DataLoader is the physical conveyor belt that grabs batches of 32 images at a time and feeds them to the GPU.

# 4. THE IMAGE & LIBRARIAN TOOLS
from torchvision import transforms
from PIL import Image

# 5. THE CLINICAL SPLIT TOOL
from sklearn.model_selection import GroupShuffleSplit

import time
import os

print("Unpacking surgical tools... Complete.")

# ==========================================
# THE POWER GRID: LOCKING IN THE GPU
# ==========================================
# PyTorch naturally wants to use your computer's CPU (Central Processing Unit).
# The CPU is smart, but it reads data one line at a time. It would take months to train this.
# We are forcing PyTorch to look for a "cuda" device (an NVIDIA Graphics Card/GPU).
# A GPU can read thousands of images simultaneously. 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f"Hardware locked. Computation will run on: {device}")

# 🚀 Increasing the GPU speed
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    print("   --> cuDNN Benchmark enabled: GPU will auto-tune for maximum speed.")
