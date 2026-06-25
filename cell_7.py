import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

# 🌟 THE BRIDGE: Reaching back to Module 1 to grab the Power Grid
from cell_1 import device

# ==========================================
# PHASE 3.5: THE ARCHITECTURE (THE BRAIN)
# ==========================================
print("6. Downloading the DenseNet Brain and performing surgery...")

class PneumoniaBrain(nn.Module):
    def __init__(self):
        super(PneumoniaBrain, self).__init__()
        
        # 1. Download the pre-trained "Genius" brain (ImageNet weights)
        self.densenet = densenet121(weights=DenseNet121_Weights.DEFAULT)
        
        # 2. Find where the old 1000-category ImageNet plug connects
        num_ftrs = self.densenet.classifier.in_features
        
        # 3. Cut off the old plug and attach our custom 1-category Pneumonia plug
        self.densenet.classifier = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.densenet(x)

# 4. Spawn the physical brain
model = PneumoniaBrain()


# 6. Move the brain to the secure GPU grid
model = model.to(device)

print("   --> Surgery successful. Brain locked onto the GPU grid and ready for training.")
