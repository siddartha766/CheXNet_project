import os
import torch
import torch.optim as optim
import time # 🌟 THE STOPWATCH ADDED HERE

# 🌟 THE MASTER BRIDGES
from cell_1 import device
from cell_5 import train_loader, val_loader
from cell_7 import model
# 🌟 UNLOCKING THE BRAIN
# This ensures the model learns!
for param in model.parameters():
    param.requires_grad = True

from cell_8 import criterion


# ==========================================
# 1. SETUP THE TEACHER & THE SCHEDULER
# =======================================

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',        
    factor=0.1,        
    patience=2         
)

EPOCHS = 5
checkpoint_path = 'chexnet_training_checkpoint.pth'
best_model_path = 'chexnet_pneumonia_best.pth'

# ==========================================
# 2. THE RESUME PROTOCOL
# ==========================================
start_epoch = 0
best_val_loss = float('inf')
counter = 0
patience_limit = 3

if os.path.exists(checkpoint_path):
    print(f"🔄 Checkpoint found! Loading classroom state from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    counter = checkpoint['counter']
    
    print(f"✅ Successfully resumed! Restarting marathon at Epoch {start_epoch + 1}.")
else:
    print("8. Igniting the Engine! Starting a fresh Training Marathon...")

# ==========================================
# 3. THE MASTER TRAINING LOOP
# ==========================================
for epoch in range(start_epoch, EPOCHS):
    # 🌟 START THE STOPWATCH
    epoch_start_time = time.time()
    
    print(f"\n--- EPOCH {epoch+1}/{EPOCHS} ---")
    
    # ---------------------------------------
    # PHASE A: THE CLASSROOM (Training)
    # ---------------------------------------
    model.train()
    running_train_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item() * images.size(0)
        
        # ---> THE HEARTBEAT MONITOR <---
        if (i + 1) % 1000 == 0:
            print(f"   [Batch {i+1}/{len(train_loader)}] Processing...")
        
    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    
    # ---------------------------------------
    # PHASE B: THE EXAM (Validation)
    # ---------------------------------------
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_val_loss += loss.item() * images.size(0)
            
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    
    print(f"--> Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
    
    # ---------------------------------------
    # PHASE C: SCHEDULER & TRACKING
    # ---------------------------------------
    scheduler.step(epoch_val_loss)
    
    for param_group in optimizer.param_groups:
        print(f"   Current LR: {param_group['lr']}")
        
    # ---------------------------------------
    # PHASE D: THE CHECKPOINT SAVING PROTOCOL
    # ---------------------------------------
    checkpoint_state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'counter': counter
    }
    torch.save(checkpoint_state, checkpoint_path)
    print("   ⏱️ Classroom state saved.")
    
    if epoch_val_loss < best_val_loss:
        print(f"   🌟 Improvement! Saving masterpiece model...")
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), best_model_path)
        counter = 0
    else:
        counter += 1
        print(f"   ⚠️ No improvement. Patience counter: {counter}/{patience_limit}")
        if counter >= patience_limit:
            print("   🛑 Early stopping triggered. Medical School is over.")
            break

    # 🌟 STOP THE STOPWATCH AND PRINT THE TIME
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    minutes = int(epoch_duration // 60)
    seconds = int(epoch_duration % 60)
    print(f"   ⏳ Epoch completed in: {minutes}m {seconds}s")
