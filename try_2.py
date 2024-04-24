import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
# from sklearn.metrics import accuracy_score
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils.data_2 import getData
import torch.nn.functional as F


def getModel(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1)
    return model


def collate_fn(batch):
    return batch

def main():
    BATCH_SIZE = 4
    EPOCH = 100
    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005

    train_loader = DataLoader(getData(f='./train.csv'), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(getData(f='./test.csv'), batch_size=BATCH_SIZE, shuffle=False)

    model = getModel(8)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    for epoch in range(EPOCH):
        model.train()
        print("EPOCH = ", epoch)
        for batch, (src, target) in enumerate(train_loader):

            src = torch.permute(src, (0, 3, 1, 2))
            src = list(img for img in src)

            box, lab, targets = [], [], []
            for i in range(BATCH_SIZE):
                b = []
                for j in range(4):
                    b.append(target[0][j][i])
                box.append([b])
                lab.append([target[1][0][i]])

            box = torch.tensor(box, dtype=torch.float32)
            lab = torch.tensor(lab, dtype=torch.int64)

            for i in range(len(src)):
                d = {}
                d['boxes'] = box[i]
                d['labels'] = lab[i]
                targets.append(d)

            loss_dict = model(src, targets)
            # print(pred)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            optimizer.zero_grad()

            print('train loss :', loss_dict)


def test():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 1, 4)

    boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]
    labels = torch.randint(1, 91, (4, 1))

    images = list(image for image in images)
    targets = []
    for i in range(len(images)):
        d = {}
        d['boxes'] = boxes[i]
        d['labels'] = labels[i]
        targets.append(d)

    print(len(images))
    print(len(targets))

    print(images[0].shape)

    output = model(images, targets)
    print(output)
    # >> >  # For inference
    # model.eval()
    # x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    # predictions = model(x)
    # print(predictions)
    # optionally, if you want to export the model to ONNX:
    # torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version=11)


if __name__ == "__main__":
    main()
    # test()
