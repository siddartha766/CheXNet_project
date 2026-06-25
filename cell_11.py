import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# 🌟 THE PROFESSIONAL AI EXPLAINER TOOLS
from pytorch_grad_cam import GradCAM, GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from cell_1 import device
from cell_7 import model

print("10. Initializing the Clinical Explainability Dashboard for comparision.")

# 1. Load the Phase 1 "Base model"
model.load_state_dict(torch.load('chexnet_pneumonia_best.pth', map_location=device, weights_only=True))
model.eval()

# 2. Define the exact deep layer
target_layers = [model.densenet.features[-1]]

# 3. choosing one particular image  form hard drive
# 🛑 ACTION REQUIRED: Paste the exact image path here again!
image_path ="/media/MyPassport/siddarth_data/images_001/images/00000001_000.png"




# Prep image background
rgb_img = cv2.imread(image_path, 1)[:, :, ::-1] 
rgb_img = cv2.resize(rgb_img, (224, 224))
rgb_img = np.float32(rgb_img) / 255

# Prep tensor for the AI
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = transform(rgb_img).unsqueeze(0).to(device)

# Force the AI to explain Node 0 (The binary output)
targets = [ClassifierOutputTarget(0)] 

# ==========================================
# GENERATING THE 3 VISUALS
# ==========================================
print("   --> Generating  Grad-CAM...")
cam = GradCAM(model=model, target_layers=target_layers)
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

print("   --> Generating Guided Backpropagation...")
guided_model = GuidedBackpropReLUModel(model=model, device=device)
guided_grads = guided_model(input_tensor, target_category=0)
guided_bprop_image = deprocess_image(guided_grads)

print("   --> Generating Guided Grad-CAM...")
guided_gradcam = guided_grads * grayscale_cam[:, :, np.newaxis]
guided_gradcam_image = deprocess_image(guided_gradcam)

# ==========================================
# PREPARING THE DISPLAY TO SHOW ALL THOSE IMAGES
# ==========================================
print("   --> preparing  a Clinical Comparison Chart...")
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: Standard Grad-CAM
axs[0].imshow(cam_image)
axs[0].set_title('1.  Grad-CAM\n(Broad Heatmap Region)', fontsize=14, fontweight='bold')
axs[0].axis('off')

# Panel 2: Guided Backpropagation
axs[1].imshow(guided_bprop_image)
axs[1].set_title('2. Guided Backpropagation\n(All High-Res Edges Detected)', fontsize=14, fontweight='bold')
axs[1].axis('off')

# Panel 3: Guided Grad-CAM
axs[2].imshow(guided_gradcam_image)
axs[2].set_title('3. Guided Grad-CAM\n(The Final Filtered Logic)', fontsize=14, fontweight='bold')
axs[2].axis('off')

# Save the master image
plt.tight_layout()
plt.savefig('clinical_comparison_present.png', dpi=300, bbox_inches='tight')

print("   --> ✅ SUCCESS! 'clinical_comparison_present.png' has been saved.")
