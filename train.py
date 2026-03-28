"""
Kaggle Ultimate Training Script
Architecture: EfficientNet-B3 + U-Net++
Loss Function: Focal Loss + Dice Loss (Anti-Lazy Setup)
Optimizations: AMP, Albumentations, Cosine Scheduler
"""
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from PIL import Image
import cv2
import os
from tqdm import tqdm

plt.switch_backend('Agg')

# ============================================================================
# 1. Mask Conversion & Setup
# ============================================================================
value_map = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}
n_classes = len(value_map)

MAX_VAL = max(value_map.keys()) + 1
lookup_table = np.zeros(MAX_VAL, dtype=np.uint8)
for k, v in value_map.items():
    lookup_table[k] = v

def convert_mask(mask):
    arr = np.array(mask)
    return Image.fromarray(lookup_table[arr])

# ============================================================================
# 2. Dataset Definition
# ============================================================================
class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.data_ids = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(convert_mask(Image.open(mask_path)))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].long()

        return image, mask

# ============================================================================
# 3. Metrics
# ============================================================================
def compute_iou(pred, target, num_classes=10):
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    iou_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().numpy())
    return np.nanmean(iou_per_class)

def evaluate_metrics(model, data_loader, device, num_classes=10):
    iou_scores = []
    model.eval()
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
            labels = labels.squeeze(dim=1).long()
            iou_scores.append(compute_iou(outputs, labels, num_classes=num_classes))
            
        if not iou_scores: return 0.0
    model.train()
    return np.mean(iou_scores)

# ============================================================================
# 4. Main Training Loop
# ============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- HYPERPARAMETERS ---
    batch_size = 4  # Drop to 2 if Kaggle throws an Out of Memory (OOM) error!
    lr = 3e-4       # Slightly higher starting LR for EfficientNet
    n_epochs = 60

    w = int(((960 / 2) // 32) * 32)
    h = int(((540 / 2) // 32) * 32)

    output_dir = '/kaggle/working/train_stats'
    os.makedirs(output_dir, exist_ok=True)

    # --- TRANSFORMS ---
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Resize(h, w),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # --- PATHS (Matches your Kaggle setup) ---
    base_data_path = '/kaggle/input/datasets/marabboni/offroad-segmentation-training-dataset/Offroad_Segmentation_Training_Dataset'
    data_dir = os.path.join(base_data_path, 'train')
    val_dir = os.path.join(base_data_path, 'val')

    trainset = MaskDataset(data_dir=data_dir, transform=train_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    valset = MaskDataset(data_dir=val_dir, transform=val_transform)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # --- THE NEW ARCHITECTURE ---
    print("\nLoading U-Net++ with EfficientNet-B3 backbone...")
    classifier = smp.UnetPlusPlus(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=n_classes
    ).to(device)

    # --- THE ANTI-LAZY LOSS FUNCTIONS ---
    print("Initializing Focal + Dice Hybrid Loss...")
    dice_loss_fn = smp.losses.DiceLoss(mode='multiclass')
    focal_loss_fn = smp.losses.FocalLoss(mode='multiclass', alpha=0.5, gamma=2.0)

    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)
    
    # Cosine Scheduler: Smoothly lowers learning rate over 60 epochs for a perfect finish
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    scaler = torch.amp.GradScaler('cuda')
    best_iou = 0.0

    print(f"\nStarting U-Net++ Training for {n_epochs} Epochs...")
    print("=" * 80)
    
    epoch_pbar = tqdm(range(n_epochs), desc="Training", unit="epoch")
    for epoch in epoch_pbar:
        # --- TRAINING PHASE ---
        classifier.train()
        train_losses = []

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", leave=False, unit="batch")
        for imgs, labels in train_pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            labels = labels.squeeze(dim=1).long()
            
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                logits = classifier(imgs)
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
                
                # HYBRID LOSS MATH: Force the model to care about borders and rare classes
                loss = dice_loss_fn(outputs, labels) + focal_loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Step the scheduler at the end of the epoch
        scheduler.step()

        # --- VALIDATION PHASE ---
        val_iou = evaluate_metrics(classifier, val_loader, device, num_classes=n_classes)
        epoch_train_loss = np.mean(train_losses)
        current_lr = scheduler.get_last_lr()[0]

        epoch_pbar.set_postfix(
            loss=f"{epoch_train_loss:.3f}",
            val_iou=f"{val_iou:.3f}",
            lr=f"{current_lr:.5f}"
        )

        # Save Best Model
        if val_iou > best_iou:
            best_iou = val_iou
            model_save_path = os.path.join(output_dir, "best_unetpp_model.pth")
            torch.save(classifier.state_dict(), model_save_path)
            # Tqdm write prevents the progress bar from breaking
            tqdm.write(f"🌟 Epoch {epoch+1}: New Best IoU -> {val_iou:.4f} (Model Saved!)")

    print("\nTraining complete! Best model saved to /kaggle/working/train_stats/")

if __name__ == "__main__":
    main()