import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# 🌟 THE MASTER BRIDGES
from cell_1 import device
from cell_7 import model
from cell_5 import val_loader as test_loader 

# ==========================================
# PHASE 5: THE BLIND EXAM & CLINICAL METRICS
# ==========================================
print("9. Locking the Brain and Starting the Blind Exam...")

# 1. Load your masterpiece model
model.load_state_dict(torch.load('chexnet_pneumonia_best.pth', map_location=device, weights_only=True))

# 2. Lock the brain (This disables dropout and learning so it can't cheat)
model.eval() 

true_labels = []
predictions = []
probabilities = [] # 🌟 Saves the raw confidence percentages for AUROC

# 3. Administer the Exam
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # The AI makes its raw mathematical guess
        outputs = model(images)
        
        # Convert raw math into percentages and strict diagnoses
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        
        # Save the answers for grading
        true_labels.extend(labels.cpu().numpy())
        predictions.extend(preds.cpu().numpy())
        probabilities.extend(probs.cpu().numpy())

print("   --> Exam finished. Grading the papers...")

# 4. Print the Clinical Report (This contains your F1-Score!)
print("\n=== CLINICAL PERFORMANCE REPORT ===")
print(classification_report(true_labels, predictions, target_names=['Healthy', 'Pneumonia']))

# 5. Calculate and print the AUROC Score
auroc_score = roc_auc_score(true_labels, probabilities)
print(f"\n=== THE GOLD STANDARD METRIC ===")
print(f"AUROC Score: {auroc_score:.4f} (1.0 is perfect, 0.5 is guessing)\n")

# 6. Draw and Save the Visual Confusion Matrix
cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Healthy', 'Pneumonia'], 
            yticklabels=['Healthy', 'Pneumonia'])
plt.ylabel('True Patient Status (Reality)')
plt.xlabel('AI Diagnosis (Prediction)')
plt.title('Clinical Confusion Matrix')

plt.savefig('confusion_matrix.png', bbox_inches='tight')
print("   --> ✅ SUCCESS! Matrix saved to your folder as 'confusion_matrix.png'.")
