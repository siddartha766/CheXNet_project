import torch
from torchvision import transforms

print("3. Hiring the Phase 3 Librarian (Setting strict augmentation rules)...")

# 1. The Training Rules (Forces the AI to actually learn lungs, not stickers)
train_transform_pipeline = transforms.Compose([
    transforms.RandomHorizontalFlip(), 
    transforms.RandomRotation(degrees=10),
    transforms.RandomResizedCrop(size=224, scale=(0.85, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. The Exam Rules (Clean, unrotated images for the validation phase)
val_transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
print("   --> Librarian hired. Augmentation pipelines locked.")
