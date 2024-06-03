#Import Library

import pandas as pd
import numpy as np
import cv2 as cv
import torch
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn as nn
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader
from utils.data import getData
from sklearn.metrics import average_precision_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from collections import Counter
from tqdm import tqdm

def getModel(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1)
    return model

def collate_fn(batch):
    return batch

def plot_metrics(epochs, train_losses, accuracy_values, map_values, precision, recall, f1_score):
    # Pisahkan nilai mAP menjadi dua daftar terpisah
    map_values_50 = [values[0] for values in map_values]
    map_values_75 = [values[1] for values in map_values]

    plt.figure(figsize=(18, 10))

    # Plot Training Loss
    plt.subplot(2, 3, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(epochs, accuracy_values, label='Eval Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Eval Accuracies')
    plt.title('Eval Accuracies')
    plt.legend()

    # Plot mAP values
    plt.subplot(2, 3, 3)
    plt.plot(epochs, map_values_50, label='mAP@0.50', color='orange')
    plt.plot(epochs, map_values_75, label='mAP@0.75', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('mAP per Epoch')
    plt.legend()

    # Plot Precision
    plt.subplot(2, 3, 4)
    plt.plot(epochs, precision, label='Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Precision')
    plt.legend()

    # Plot Recall
    plt.subplot(2, 3, 5)
    plt.plot(epochs, recall, label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Recall')
    plt.legend()

    # Plot F1 Score
    plt.subplot(2, 3, 6)
    plt.plot(epochs, f1_score, label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score')
    plt.legend()



    plt.tight_layout()
    plt.savefig('metrics_plot.png')
    plt.show()

def save_checkpoint(model, optimizer, epoch, best_map, path='checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_map': best_map
    }, path)

def main():
    BATCH_SIZE = 8
    EPOCH = 30
    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    train_loader = DataLoader(getData(f='./train.csv'), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(getData(f='./test.csv'), batch_size=BATCH_SIZE, shuffle=False)

    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

    model = getModel(8).to(DEVICE)

    # Inisialisasi optimizer dan metrics
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    map_metric = MeanAveragePrecision(iou_thresholds=[0.5, 0.75])

    patience = 5
    best_map = 0
    epochs_no_improve = 0
    early_stop = False

    train_losses = []
    map_values = []
    precision_values = []
    recall_values = []
    f1_values = []
    accuracy_values = []
    epochs = []

    for epoch in range(EPOCH):
        model.train()
        print('Epoch: ', epoch + 1)
        epoch_loss = 0

        for batch, (src, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCH}")):
            src = src.to(DEVICE)
            box = [torch.tensor([[target[0][j][i] for j in range(4)]], dtype=torch.float32).to(DEVICE) for i in range(len(target[0][0]))]
            lab = [torch.tensor([target[1][0][i]], dtype=torch.int64).to(DEVICE) for i in range(len(target[0][0]))]

            targets = [{'boxes': b, 'labels': l} for b, l in zip(box, lab)]

            loss_dict = model(src, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += losses.item()

        epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch: {epoch + 1}, Loss: {epoch_loss}")

        model.eval()
        map_metric.reset()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch, (src, target) in enumerate(tqdm(test_loader, desc="Test")):
                src = src.to(DEVICE)
                box = [torch.tensor([[target[0][j][i] for j in range(4)]], dtype=torch.float32).to(DEVICE) for i in range(len(target[0][0]))]
                lab = [torch.tensor([target[1][0][i]], dtype=torch.int64).to(DEVICE) for i in range(len(target[0][0]))]

                targets = [{'boxes': b, 'labels': l} for b, l in zip(box, lab)]

                outputs = model(src)
                map_metric.update(outputs, targets)

                # Convert outputs to labels and confidence scores
                for output in outputs:
                    pred_labels = output['labels']
                    pred_scores = output['scores']
                    max_score_index = np.argmax(pred_scores.cpu().numpy())
                    all_preds.append(pred_labels[max_score_index].cpu().numpy())

                for t in targets:
                    all_targets.append(t['labels'][0].cpu().numpy())

        mAP_result = map_metric.compute()
        mean_ap_50 = mAP_result['map_50'].item()  # Get the mAP at IoU=0.5
        mean_ap_75 = mAP_result['map_75'].item()  # Get the mAP at IoU=0.75

        eval_accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

        print(f"Epoch: {epoch + 1}, mAP@0.5: {mean_ap_50}, mAP@0.75: {mean_ap_75}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}, Eval Accuracy: {eval_accuracy}")

        train_losses.append(epoch_loss)
        map_values.append((mean_ap_50, mean_ap_75))
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        accuracy_values.append(eval_accuracy)
        epochs.append(epoch)


        if mean_ap_50 > best_map:
            best_map = mean_ap_50
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch, best_map, 'rupiah_banknotes_checkpoint.pth')
            torch.save(model.state_dict(), 'rupiah_banknotes_model.pth')  # Save the best model
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            early_stop = True
            print("Early stopping due to no improvement in mAP")
            break

    if not early_stop:
        torch.save(model.state_dict(), 'rupiah_banknotes_model_final.pth')

    target_names = [f"Class {i}" for i in range(8)]  # Gantilah sesuai dengan nama kelas yang sesuai

    print("\nClassification Report per Class:")
    print(classification_report(all_targets, all_preds, target_names=target_names, zero_division=0))

    plot_metrics(epochs, train_losses, accuracy_values, map_values, precision_values, recall_values, f1_values)
if __name__ == "__main__":
    main()