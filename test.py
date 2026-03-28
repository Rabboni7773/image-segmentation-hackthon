"""
Hackathon Final Test Script (U-Net++ & TTA)
Architecture: EfficientNet-B3 + U-Net++
Outputs: 16-bit Raw masks (Leaderboard ready), Color masks, Comparisons, Mean IoU, and mAP50
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import cv2
import os
import segmentation_models_pytorch as smp
from tqdm import tqdm

plt.switch_backend('Agg')

# ============================================================================
# 1. Configuration & Kaggle Paths
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

w = int(((960 / 2) // 32) * 32) # 480
h = int(((540 / 2) // 32) * 32) # 256

# --- KAGGLE PATHS ---
test_data_dir = '/kaggle/input/datasets/marabboni/offroad-segmentation-testimages/Offroad_Segmentation_testImages'
model_weights_path = '/kaggle/input/models/marabboni/segmentation-model-final/pytorch/default/1/best_unetpp_model.pth' 

output_dir = '/kaggle/working/FINAL_TEST_RESULTS'
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# 2. Mask Conversion, Color Palette, & REVERSE LOOKUP
# ============================================================================
value_map = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]
n_classes = len(value_map)

color_palette = np.array([
    [0, 0, 0], [34, 139, 34], [0, 255, 0], [210, 180, 140], [139, 90, 43], 
    [128, 128, 0], [139, 69, 19], [128, 128, 128], [160, 82, 45], [135, 206, 235]
], dtype=np.uint8)

# --- CRITICAL: REVERSE LOOKUP ARRAY FOR LEADERBOARD ---
reverse_lookup = np.zeros(10, dtype=np.uint16)
reverse_map_dict = {v: k for k, v in value_map.items()}
for class_id, original_val in reverse_map_dict.items():
    reverse_lookup[class_id] = original_val

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)

def mask_to_color(mask):
    h_m, w_m = mask.shape
    color_mask = np.zeros((h_m, w_m, 3), dtype=np.uint8)
    for class_id in range(n_classes):
        color_mask[mask == class_id] = color_palette[class_id]
    return color_mask

# ============================================================================
# 3. Dataset & Albumentations
# ============================================================================
class TestDataset(Dataset):
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

        return image, mask, data_id

test_transform = A.Compose([
    A.Resize(h, w),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# ============================================================================
# 4. Metrics & Visuals
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
            
    return np.nanmean(iou_per_class), iou_per_class

def calculate_map50(all_class_ious):
    ap_per_class = []
    ious_array = np.array(all_class_ious)
    
    for class_id in range(n_classes):
        class_ious = ious_array[:, class_id]
        valid_ious = class_ious[~np.isnan(class_ious)]
        
        if len(valid_ious) == 0:
            ap_per_class.append(float('nan'))
            continue
            
        true_positives = np.sum(valid_ious >= 0.5)
        total_predictions = len(valid_ious)
        
        average_precision = true_positives / total_predictions
        ap_per_class.append(average_precision)
        
    mAP50 = np.nanmean(ap_per_class)
    return mAP50, ap_per_class

def save_prediction_comparison(img_tensor, gt_mask, pred_mask, output_path, data_id):
    img = img_tensor.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = img * std + mean
    img = np.clip(img, 0, 1)

    gt_color = mask_to_color(gt_mask.cpu().numpy().astype(np.uint8))
    pred_color = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img); axes[0].set_title('Input Image'); axes[0].axis('off')
    axes[1].imshow(gt_color); axes[1].set_title('Ground Truth'); axes[1].axis('off')
    axes[2].imshow(pred_color); axes[2].set_title('U-Net++ Prediction'); axes[2].axis('off')

    plt.suptitle(f'Sample: {data_id}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_metrics_summary(results, out_dir):
    filepath = os.path.join(out_dir, 'evaluation_metrics.txt')
    with open(filepath, 'w') as f:
        f.write("HACKATHON FINAL TEST RESULTS (U-NET++)\n" + "=" * 50 + "\n")
        f.write(f"mAP50 Score:       {results['map50']:.4f}  <-- OFFICIAL LEADERBOARD METRIC\n")
        f.write(f"Mean IoU:          {results['mean_iou']:.4f}\n" + "=" * 50 + "\n\n")
        
        f.write("Per-Class Breakdown:\n")
        f.write(f"{'Class Name':<20} | {'IoU':<10} | {'AP50':<10}\n")
        f.write("-" * 50 + "\n")
        
        for i, name in enumerate(class_names):
            iou_val = results['class_iou'][i]
            ap_val = results['class_ap50'][i]
            
            iou_str = f"{iou_val:.4f}" if not np.isnan(iou_val) else "N/A"
            ap_str = f"{ap_val:.4f}" if not np.isnan(ap_val) else "N/A"
            
            f.write(f"{name:<20} | {iou_str:<10} | {ap_str:<10}\n")

    fig, ax = plt.subplots(figsize=(10, 6))
    valid_iou = [iou if not np.isnan(iou) else 0 for iou in results['class_iou']]
    ax.bar(range(n_classes), valid_iou, color=[color_palette[i] / 255 for i in range(n_classes)], edgecolor='black')
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('IoU Score')
    ax.set_title(f'Test Set Per-Class IoU (mAP50: {results["map50"]:.4f})')
    ax.set_ylim(0, 1)
    ax.axhline(y=results['mean_iou'], color='red', linestyle='--', label='Mean IoU')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'per_class_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# 5. Main Execution Loop
# ============================================================================
def main():
    testset = TestDataset(data_dir=test_data_dir, transform=test_transform)
    test_loader = DataLoader(testset, batch_size=2, shuffle=False)

    print("Loading U-Net++ with EfficientNet-B3 backbone...")
    model = smp.UnetPlusPlus(encoder_name="efficientnet-b3", encoder_weights=None, in_channels=3, classes=10).to(device)
    
    if os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        print("SUCCESS: U-Net++ weights loaded!")
    else:
        print(f"ERROR: Weights not found at {model_weights_path}")
        return
        
    model.eval()

    masks_dir = os.path.join(output_dir, 'masks')
    masks_color_dir = os.path.join(output_dir, 'masks_color')
    comparisons_dir = os.path.join(output_dir, 'comparisons')
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(masks_color_dir, exist_ok=True)
    os.makedirs(comparisons_dir, exist_ok=True)

    iou_scores, all_class_iou = [], []
    sample_count = 0
    max_visuals = 15 

    print(f"\nEvaluating Final Test Set ({len(testset)} images) using TTA...")
    with torch.no_grad():
        for imgs, labels, data_ids in tqdm(test_loader, desc="Processing"):
            imgs, labels = imgs.to(device), labels.to(device)
            orig_h, orig_w = labels.shape[1], labels.shape[2]

            with torch.amp.autocast('cuda'):
                logits_normal = model(imgs)
                img_flipped = torch.flip(imgs, dims=[3])
                logits_flipped = model(img_flipped)
                logits_unflipped = torch.flip(logits_flipped, dims=[3])
                logits_combined = (logits_normal + logits_unflipped) / 2.0
                
                outputs = F.interpolate(logits_combined, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

            predicted_masks = torch.argmax(outputs, dim=1)
            
            iou, class_iou = compute_iou(outputs, labels, num_classes=n_classes)
            iou_scores.append(iou)
            all_class_iou.append(class_iou)

            for i in range(imgs.shape[0]):
                data_id = data_ids[i]
                base_name = os.path.splitext(data_id)[0]
                
                # Raw 0-9 tensor
                pred_mask_0_to_9 = predicted_masks[i].cpu().numpy().astype(np.uint8)

                # 1. Raw Masks for Submission (Mapped to 10000 and saved as 16-bit)
                final_hackathon_mask = reverse_lookup[pred_mask_0_to_9]
                cv2.imwrite(os.path.join(masks_dir, f'{base_name}_pred.png'), final_hackathon_mask)

                # 2. Color Masks
                pred_color = mask_to_color(pred_mask_0_to_9)
                cv2.imwrite(os.path.join(masks_color_dir, f'{base_name}_pred_color.png'), cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

                # 3. Comparisons
                if sample_count < max_visuals:
                    save_prediction_comparison(
                        imgs[i], labels[i], predicted_masks[i],
                        os.path.join(comparisons_dir, f'{base_name}_compare.png'), data_id
                    )
                sample_count += 1

    mean_iou = np.nanmean(iou_scores)
    avg_class_iou = np.nanmean(all_class_iou, axis=0)
    map50, ap_per_class = calculate_map50(all_class_iou)

    print("\n" + "=" * 50)
    print(f"🏆 FINAL mAP50 SCORE: {map50:.4f} 🏆")
    print(f"FINAL MEAN IOU:       {mean_iou:.4f}")
    print("=" * 50)

    results = {
        'mean_iou': mean_iou, 
        'class_iou': avg_class_iou,
        'map50': map50,
        'class_ap50': ap_per_class
    }
    save_metrics_summary(results, output_dir)
    print(f"Complete results and submission masks saved to: {output_dir}")

if __name__ == "__main__":
    main()