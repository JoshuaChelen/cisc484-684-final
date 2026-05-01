import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models
from torch.utils.tensorboard import SummaryWriter

from pytorch_dataloader.py import build_loaders


#settings
BATCH_SIZE=32
NUM_WORKERS=0
EPOCHS=0
LEARNING_RATE=1e-4

MODEL_OUT = "efficientnet_b0_finetuned.pth"
CSV_LOG = "training_log.csv"
TENSORBOARD_DIR = "runs/efficientnet_b0_finetune"

#setup device
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#load data
train_loader, val_loader, test_loader=build_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
print("DataLoaders created.")
print("Train batches:", len(train_loader))
print("Val batches:", len(val_loader))
print("Test batches:", len(test_loader))

# Confirm augmentation setup
print("\nConfirming transforms:")
print("Train transform:", train_loader.dataset.transform)
print("Val transform:", val_loader.dataset.transform)
print("Test transform:", test_loader.dataset.transform)

#load efficientNet-B0
model=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

#replace with binarty classifier
num_features=model.classifier[1].in_features
model.classfier[1]=nn.Linear(num_features,2)

#freeze layers
for p in model.paramters():
    p.requires_grad=False

# Unfreeze last 2 EfficientNet feature blocks
for block in list(model.features.children())[-2:]:
    for p in block.parameters():
        p.requires_grad=True

#unfreeze classifier
for p in model.classifier.parameters():
        p.requires_grad=True

model=model.to(device)
print("\nTrainable parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)


#loss, optimizer, scheduler
criteria=nn.CrossEntropyLoss()
optimizer=optim.Adam(
     filter(lambda p: p.requires_grad, model.paramters()),
     lr=LEARNING_RATE
)
scheduler=optim.lr_scheduler.ReduceLROnPlateau(
     optimizer,
     mode="min",
     factor=0.5,
     patience=2
)

#helper functions
def train_epoch(model, loader, criteria, optimizer, device):
    model.train()
    running_loss=0
    correct=0
    total=0
    for images, labels in loader:
        images=images.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()

        outputs=model(images)
        loss=criteria(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss+=loss.item()*images.size(0)

        preds=outputs.argmax(dim=1)
        correct+=(preds==labels).sum().item()
        total+=labels.size(0)
    epoch_loss=running_loss/total
    epoch_acc=correct/total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criteria, device):
    model.eval()
    running_loss=0
    correct=0
    total=0
    with torch.no_grad():
        for images, labels in loader:
            images=images.to(device)
            labels.labels.to(device)
            outputs=model(images)
            loss=criteria(outputs, labels)
            running_loss+=loss.item*images.size(0)
            preds=outputs.argmax(dim=1)
            correct+=(preds==labels).sum().item()
            total+=labels.size(0)
    epoch_loss=running_loss/total
    epoch_acc=correct/total
    return epoch_loss, epoch_acc

#logging setup
    #training loop

#final test evaluation

#save model

    
