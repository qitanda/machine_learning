# -*- coding: utf-8 -*-
"""Image Classification

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15hMu9YiYjE_6HY99UXon2vKGk2KwugWu

# Get Data
Notes: if the links are dead, you can download the data directly from Kaggle and upload it to the workspace, or you can use the Kaggle API to directly download the data into colab.
"""

#! wget https://www.dropbox.com/s/6l2vcvxl54b0b6w/food11.zip
#! wget -O food11.zip "https://github.com/virginiakm1988/ML2022-Spring/blob/main/HW03/food11.zip?raw=true"
# 
#! unzip food11.zip

_exp_name = "sample"
OS = "Linux"

# Import necessary packages.
import numpy as np
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
import torchvision.models as models
import category

# This is for the progress bar.
from tqdm.auto import tqdm
import random

myseed = 6666  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

"""## **Transforms**
Torchvision provides lots of useful utilities for image preprocessing, data wrapping as well as data augmentation.

Please refer to PyTorch official website for details about different transforms.
"""

# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
# test_tfm = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
# ])

test_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
# train_tfm = transforms.Compose([
#     # Resize the image into a fixed shape (height = width = 128)
#     transforms.Resize((128, 128)),
#     # You may add some transforms here.
#     # ToTensor() should be the last one of the transforms.
#     transforms.ToTensor(),
# ])
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    # transforms.Resize((128, 128)),
    transforms.RandomResizedCrop(224),
    # Random horizontal flip with a probability of 0.5
    transforms.RandomHorizontalFlip(),
    # Randomly rotate the image by up to 15 degrees
    # transforms.RandomRotation(degrees=15),
    # Randomly adjust the brightness, contrast, and saturation
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # Random affine transformation with translation and scale
    # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    # Convert the image to tensor
    transforms.ToTensor(),
    # Normalize the image with mean and std (using ImageNet values as an example)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

"""## **Datasets**
The data is labelled by the name, so we load images and label while calling '__getitem__'
"""

class FoodDataset(Dataset):

    def __init__(self,path,tfm=test_tfm,files = None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
        print(f"One {path} sample",self.files[0])
        self.transform = tfm
  
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        #im = self.data[idx]
        try:
            if OS == "Windows":
                label = int(fname.split("\\")[-1].split("_")[0])
            else:
                label = int(fname.split("/")[-1].split("_")[0])
            # print("\n", label)
        except:
            label = -1 # test has no label
        return im,label

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

        #     nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

        #     nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

        #     nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            
        #     nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
        # )
        self.cnn = models.resnet50(pretrained=False)
        # 替换最后的全连接层，输入是ResNet的输出512维度，输出是11类
        self.cnn.fc = nn.Sequential(
            nn.Linear(2048, 1024),  # ResNet50的最后一层输出是2048维
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)  # 分类11种食物
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(512*4*4, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 11)
        # )

    # def forward(self, x):
    #     out = self.cnn(x)
    #     out = out.view(out.size()[0], -1)
    #     return self.fc(out)
    def forward(self, x):
        return self.cnn(x)

batch_size = 64
_dataset_dir = os.path.join(os.getcwd(), "food11")
# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

def Training_Demo():
    """# Training"""
    # Construct datasets.
    # The argument "loader" tells how torchvision reads the data.
    print(os.path.join(_dataset_dir,"training"))
    # train_set = FoodDataset(os.path.join(_dataset_dir,"training"), tfm=train_tfm)
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    # valid_set = FoodDataset(os.path.join(_dataset_dir,"validation"), tfm=test_tfm)
    # valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    # The number of training epochs and patience.
    n_epochs = 15
    patience = 300 # If no improvement in 'patience' epochs, early stop

    # Initialize a model, and put it on the device specified.
    model = Classifier()
    model.load_state_dict(torch.load('sample_best.ckpt'))
    model.to(device)

    # For the classification task, we use cross-entropy as the measurement of performance.
    criterion = nn.CrossEntropyLoss()

    # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5) 

    # Initialize trackers, these are not parameters and should not be changed
    stale = 0
    best_acc = 0

    for epoch in range(n_epochs):

        # ---------- Training ----------
        combined_dir = os.path.join(_dataset_dir,"combined")
        new_train_dir = os.path.join(_dataset_dir,"training")
        new_val_dir = os.path.join(_dataset_dir,"validation")
        # category.split_dataset(combined_dir, new_train_dir, new_val_dir, split_ratio=0.9)
        train_set = FoodDataset(new_train_dir, tfm=train_tfm)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        valid_set = FoodDataset(new_val_dir, tfm=test_tfm)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        # Make sure the model is in train mode before training.
        model.train()

        # These are used to record information in training.
        train_loss = []
        train_accs = []

        for batch in tqdm(train_loader):

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            #imgs = imgs.half()
            #print(imgs.shape,labels.shape)

            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs.to(device))

            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion(logits, labels.to(device))

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)
            
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()

        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []

        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader):

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            #imgs = imgs.half()

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))

            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels.to(device))

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)
            #break

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


        # update logs
        if valid_acc > best_acc:
            with open(f"./{_exp_name}_log.txt","a"):
                print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
        else:
            with open(f"./{_exp_name}_log.txt","a"):
                print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


        # save models
        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch}, saving model")
            torch.save(model.state_dict(), f"{_exp_name}_best.ckpt") # only save best to prevent output memory exceed error
            best_acc = valid_acc
            stale = 0
        else:
            stale += 1
            if stale > patience:
                print(f"No improvment {patience} consecutive epochs, early stopping")
                break

def Testing_Demo():
    valid_set = FoodDataset(os.path.join(_dataset_dir,"validation"), tfm=test_tfm)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    model_test = Classifier().to(device)
    model_test.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
    model_test.eval()
    valid_loss = []
    valid_accs = []
    ids = []
    predictions = []

    path = os.path.join(_dataset_dir,"validation")
    files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
    for fname in files:
        if OS == "Windows":
            ids.append( fname.split("\\")[-1].split(".")[0])
        else:
            ids.append( fname.split("/")[-1].split(".")[0])

    with torch.no_grad():
        for data, labels in tqdm(valid_loader):
            test_pred = model_test(data.to(device))
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            predictions += test_label.squeeze().tolist()
            acc = (test_pred.argmax(dim=-1) == labels.to(device)).float().mean()
            valid_accs.append(acc)

    valid_acc = sum(valid_accs) / len(valid_accs)
    print(f"acc = {valid_acc:.5f}")

    if OS == "Windows":
        with open(os.getcwd() + "\\result.txt", "w") as f:
            f.write(f"ID Category Accuracy: {valid_acc}\n")
            for i in range(len(ids)):
                f.write(ids[i])
                f.write(" ")
                f.write(str(predictions[i]))
                f.write("\n")
    else:
        with open("result.txt", "w") as f:
            f.write(f"ID Category Accuracy: {valid_acc}\n")
            for i in range(len(ids)):
                f.write(ids[i])
                f.write(" ")
                f.write(str(predictions[i]))
                f.write("\n")
        

def Predict_demo():
    """# Testing and generate prediction CSV"""
    import pandas as pd
    test_set = FoodDataset(os.path.join(_dataset_dir,"test"), tfm=test_tfm)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    model_best = Classifier().to(device)
    model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
    model_best.eval()
    prediction = []
    with torch.no_grad():
        for data,_ in test_loader:
            test_pred = model_best(data.to(device))
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            prediction += test_label.squeeze().tolist()

    #create test csv
    def pad4(i):
        return "0"*(4-len(str(i)))+str(i)
    df = pd.DataFrame()
    df["Id"] = [pad4(i) for i in range(1,len(test_set)+1)]
    df["Category"] = prediction
    df.to_csv("submission.csv",index = False)

if __name__ == "__main__":
    # Training_Demo()
    Testing_Demo()
    #Predict_demo()
