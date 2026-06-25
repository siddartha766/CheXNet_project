import matplotlib.pyplot as plt
import torch

# 🌟 THE BRIDGE: Reaching back to Cell 5 to grab the conveyor belt
from cell_5 import train_dataset

# ==========================================
# PHASE 3.4: QUALITY ASSURANCE (THE MICROSCOPE)
# ==========================================
print("5. Looking through the Microscope (Sanity Checking the Conveyor Belt)...")

# 1. Pull the very first X-ray off the factory line
sample_image, sample_label = train_dataset[0]

print(f"   --> The AI sees an image of size (Math Format): {sample_image.shape}")
print(f"   --> The Diagnosis is (1.0 = Sick, 0.0 = Healthy): {sample_label.item()}")

# 2. Translate "Robot Math" back into "Human Pixels"
human_image = sample_image.permute(1, 2, 0)

# 3. Undo the Librarian's normalization math to restore original lighting
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
human_image = human_image * std + mean
human_image = torch.clamp(human_image, 0, 1)

# 4. Save the X-ray(output) to the hard drive (because no display of image happening in terminal)
plt.imshow(human_image)
plt.title(f"Clinical Target (Label): {sample_label.item()}")
plt.axis('off')

# 🌟 THE HEADLESS FIX: Save it as a PNG file
plt.savefig('sanity_check_image.png', bbox_inches='tight')
print("   --> ✅ SUCCESS! Image saved to your folder as 'sanity_check_image.png'.")
