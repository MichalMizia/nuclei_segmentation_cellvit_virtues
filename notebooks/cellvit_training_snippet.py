# Full Training Run (300 Epochs)
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from src.models.cellvit_decoder import CellViTDecoder
from src.utils.cellvit_utils import CombinedLoss, calculate_dice_score

print("--- Starting Full Training Run (300 Epochs) ---")

# 1. Setup
num_epochs = 300
learning_rate = 1e-3
crop_size = 256

# Re-initialize model to ensure fresh weights
decoder = CellViTDecoder(
    num_nuclei_classes=num_classes,
    embed_dim=512, 
    drop_rate=0.1
).to(device)

optimizer = optim.AdamW(decoder.parameters(), lr=learning_rate, weight_decay=1e-2)
criterion = CombinedLoss(num_classes=num_classes, alpha=0.5)

# Tracking
train_losses = []
val_losses = []
val_dices = []

best_val_loss = float('inf')
save_path = "best_cellvit_model.pth"

# 2. Training Loop
for epoch in range(num_epochs):
    # --- Training ---
    decoder.train()
    running_loss = 0.0
    steps = 0
    
    # Use the feeder to iterate through training tissues
    for i, data_dict in enumerate(feeder.iterate_image_orion(train_tids, crop_size=crop_size)):
        pss = data_dict['pss'].to(device)
        mask = data_dict['mask'].to(device).long()
        
        optimizer.zero_grad()
        outputs = decoder(pss)
        pred_logits = outputs['nuclei_type_map']
        
        loss = criterion(pred_logits, mask)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        steps += 1
        
        # Cleanup to save memory
        del pss, mask, outputs, pred_logits, loss
    
    avg_train_loss = running_loss / steps if steps > 0 else 0
    train_losses.append(avg_train_loss)
    
    # --- Validation ---
    decoder.eval()
    val_running_loss = 0.0
    val_running_dice = 0.0
    val_steps = 0
    
    with torch.no_grad():
        for i, data_dict in enumerate(feeder.iterate_image_orion(val_tids, crop_size=crop_size)):
            pss = data_dict['pss'].to(device)
            mask = data_dict['mask'].to(device).long()
            
            outputs = decoder(pss)
            pred_logits = outputs['nuclei_type_map']
            
            loss = criterion(pred_logits, mask)
            val_running_loss += loss.item()
            
            pred_mask = torch.argmax(pred_logits, dim=1)
            dice = calculate_dice_score(pred_mask, mask, num_classes)
            val_running_dice += dice
            
            val_steps += 1
            
            del pss, mask, outputs, pred_logits, loss, pred_mask
    
    avg_val_loss = val_running_loss / val_steps if val_steps > 0 else 0
    avg_val_dice = val_running_dice / val_steps if val_steps > 0 else 0
    
    val_losses.append(avg_val_loss)
    val_dices.append(avg_val_dice)
    
    # Print progress every epoch
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f}")
    
    # Save Best Model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(decoder.state_dict(), save_path)
        print(f"  --> New best model saved (Val Loss: {best_val_loss:.4f})")

print("Training Complete.")

# 3. Plotting
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(val_dices, label='Val Dice Score', color='orange')
plt.title('Validation Dice Score')
plt.xlabel('Epoch')
plt.ylabel('Dice')
plt.legend()
plt.grid(True)

plt.show()