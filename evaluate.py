import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from dataset import BraTSDataset
from transforms import ToTensorMultiChannel
from model import UNet


def evaluate_model(model, data_dir, image_size=128):
    dataset = BraTSDataset(data_dir=data_dir, image_size=image_size, transform=ToTensorMultiChannel())
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()
    total_accuracy = 0.0
    total_dice = 0.0
    total_samples = len(dataloader.dataset)

    with torch.no_grad():
        for X, y in dataloader:
            outputs = model(X)
            pred = outputs.argmax(dim=1).cpu().numpy()  # Predicted segmentation mask
            true = y.cpu().numpy()                      # Ground truth segmentation mask

            # Calculate accuracy
            correct_pixels = np.sum(pred == true)
            total_pixels = np.prod(pred.shape)
            accuracy = correct_pixels / total_pixels
            total_accuracy += accuracy

            # Calculate Dice coefficient for each class
            dice_scores = []
            for class_id in range(4):  # Assuming 4 classes
                intersection = np.sum((pred == class_id) & (true == class_id))
                dice = (2. * intersection) / (np.sum(pred == class_id) + np.sum(true == class_id))
                dice_scores.append(dice)

            # Average Dice coefficient across all classes
            average_dice = np.mean(dice_scores)
            total_dice += average_dice

            # Plot the predicted and ground truth segmentation masks
            plot_segmentation_masks(X, pred, true)

    avg_accuracy = total_accuracy / total_samples
    avg_dice = total_dice / total_samples

    return avg_accuracy, avg_dice

def plot_segmentation_masks(X, pred, true):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ['Input Image', 'Predicted Mask', 'Ground Truth Mask']

    # Convert tensors to numpy arrays for visualization
    X_np = X.squeeze().cpu().numpy().transpose((1, 2, 0))
    pred_np = pred.squeeze()
    true_np = true.squeeze()

    axes[0].imshow(X_np[..., :3])  # Show only the first three channels (input image)
    axes[0].set_title(titles[0])
    axes[1].imshow(pred_np, cmap='jet', vmin=0, vmax=3)  # Assuming 4 classes (0 to 3)
    axes[1].set_title(titles[1])
    axes[2].imshow(true_np, cmap='jet', vmin=0, vmax=3)
    axes[2].set_title(titles[2])

    plt.tight_layout()
    plt.show()

