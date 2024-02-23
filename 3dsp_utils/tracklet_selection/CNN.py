import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional,Callable


class CNN(nn.Module):
    def __init__(self, input_channels, hidden_size, num_layers,num_linear_layers, kernel_size,num_image,drop_out):
        super(CNN, self).__init__()

        self.num_image = num_image
        self.num_linear_layers = num_linear_layers
        
        # Convolutional layer (ref ResNet)
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2d = nn.Dropout2d(p=0.2)

        
        # LSTM layer
        # self.lstm = nn.LSTM(441, hidden_size, num_layers, batch_first=True)
        # self.lstm = nn.GRU(4608, hidden_size, num_layers, batch_first=True,dropout=drop_out)

        self.linear=nn.Linear(4608*6, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(p=0.2)
        # self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        #remove the first dimension of x
        # x = x.squeeze(0)
        #convert x from (batch_size, images, channels, height, width) to (batch_size*images,  channels, height, width)
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        # Apply convolutional layer
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.batchnorm1(x)
        x = self.dropout2d(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.batchnorm2(x)
        x = self.dropout2d(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.batchnorm3(x)
        x = self.dropout2d(x)
        
        #convert x from (batch_size*images,  channels, height, width) to (batch_size, images, channels, height, width)
        x = x.view(-1, self.num_image, x.shape[1], x.shape[2], x.shape[3])
        #convert x from (batch_size, images, channels, height, width) to (batch_size, images, channels*height*width)
        # x = x.view(x.shape[0], x.shape[1], -1)
        #convert x from (batch_size, images, channels*height*width) to (batch_size, images*channels*height*width)
        x = x.reshape(x.shape[0], -1)

        # # Apply LSTM layer
        # output, _ = self.lstm(x)
        # #get the last output
        # output = output[:,-1,:]
        # pdb.set_trace()
        output=self.linear(x)
        output = self.relu(output)
        output = self.dropout(output)

        
        for num in range(self.num_linear_layers):
            output = self.linear1(output)
            output = self.relu(output)
            output = self.dropout(output)

        output = self.linear2(output)
        # output = self.softmax(output)
        output=output.squeeze(1)
        return output

import torch.optim as optim
import numpy as np
import pdb

def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, patience, save_dir, device,weight_for_loss=20):
    best_valid_loss = float('inf')
    epochs_no_improve = 0
    best_model_state_dict = None
    
    for epoch in range(num_epochs):
        save_path=os.path.join(save_dir, 'epoch_'+str(epoch+1)+'.pth')
        model.train()
        train_loss = 0.0
        
        # Training loop
        for inputs, labels in train_loader:
            #create weight with the same shape as labels
            weight = np.where(labels==0, 1, weight_for_loss)
            # labels = labels.squeeze(0)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss=weighted_mse_loss(outputs, labels, torch.tensor(weight).float().to(device))
            # loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation loop
        model.eval()
        valid_loss = 0.0
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        flag_output=True
        with torch.no_grad():
            for inputs, labels in valid_loader:
                #create weight with the same shape as labels
                weight = np.where(labels==0, 1, weight_for_loss)
                # labels = labels.squeeze(0)
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                # print(outputs, labels)
                if flag_output:
                    print(outputs, labels)
                    flag_output=False
                loss=weighted_mse_loss(outputs, labels, torch.tensor(weight).float().to(device))
                #claculate the tp, tn, fp, fn
                tp += ((outputs > 0.5) & (labels == 1)).sum().item()
                tn += ((outputs <= 0.5) & (labels == 0)).sum().item()
                fp += ((outputs > 0.5) & (labels == 0)).sum().item()
                fn += ((outputs <= 0.5) & (labels == 1)).sum().item()
                # loss = criterion(outputs, labels)
                valid_loss += loss.item()
        
        # Calculate average losses
        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)

        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')

        print(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')
        
        # Check for early stopping
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_state_dict = model.state_dict()
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping after {epoch+1} epochs without improvement.')
                break
    
    # Load the best model state
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
    
    return model

'''
trained_model = train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=50, patience=5, save_path='best_model.pth')
'''
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2
import random
import os

class CustomDataset(Dataset):
    def __init__(self, root_dir, train=True):
        self.train = train
        if train:
            self.root_dir = os.path.join(root_dir, 'train')
        else:
            self.root_dir = os.path.join(root_dir, 'valid')
        self.folders=os.listdir(self.root_dir)
        self.transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        #get the str after the last _
        label = int(self.folders[idx].split('_')[-1])
        
        image_folder = []
        # for j in os.listdir(os.path.join(self.root_dir,self.folders[idx])):
        for j in ["10.jpg","11.jpg","12.jpg","13.jpg","14.jpg","15.jpg"]:
            image = cv2.imread(os.path.join(self.root_dir,self.folders[idx],j))
            #change the channel to the first dimension
            image = image.transpose((2, 0, 1))
            image = image/255
            image_folder.append(image)

        image_folder = np.array(image_folder)
        image=torch.tensor(image_folder).float()
        label=torch.tensor(label).float()

        return image, label

def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)

import torch.nn.functional as F

def weighted_bce_loss(input, target, weight):
    # Apply sigmoid to the input tensor
    input_sigmoid = torch.sigmoid(input)
    # Compute BCE loss
    bce_loss = F.binary_cross_entropy(input_sigmoid, target, reduction='none')
    # Apply the weights
    weighted_bce_loss = torch.mean(weight * bce_loss)
    return weighted_bce_loss

def validation(valid_loader, model):
    model.eval()
    valid_loss = 0.0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    flag_output=True
    with torch.no_grad():
        for inputs, labels in valid_loader:
            #create weight with the same shape as labels
            # weight = np.where(labels==0, 1, weight_for_loss)
            # labels = labels.squeeze(0)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            #apply sigmoid to the output
            outputs = torch.sigmoid(outputs)
            print(outputs, labels)
            if flag_output:
                print(outputs, labels)
                flag_output=False
            # loss=weighted_bce_loss(outputs, labels, torch.tensor(weight).float().to(device))
            #claculate the tp, tn, fp, fn
            tp += ((outputs > 0.5) & (labels == 1)).sum().item()
            tn += ((outputs <= 0.5) & (labels == 0)).sum().item()
            fp += ((outputs > 0.5) & (labels == 0)).sum().item()
            fn += ((outputs <= 0.5) & (labels == 1)).sum().item()
            # loss = criterion(outputs, labels)
            # valid_loss += loss.item()
        
    # Calculate average losses
    # valid_loss /= len(valid_loader)

    
    print(f'Validation(BCE),  Valid Loss: {valid_loss:.4f}')

    print(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')

if __name__ == "__main__":
    import argparse
    import time
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path",  help="Path to the input image", default= os.getcwd()+"/dataset")
    ap.add_argument("--pt_out_path",  help="Path to the output image", default= os.getcwd()+"/params_cnn_f10-f15")
    ap.add_argument("--hidden_size",  help="Hidden size of LSTM (fintune) [128,256,512]", default=256)
    ap.add_argument("--num_layers",  help="Number of LSTM layers (fintune) [1,2,3]", default=2)
    ap.add_argument("--num_linear_layers",  help="Number of linear layers (fintune) [1,2,3]", default=3)
    args = ap.parse_args()
    
    args.pt_out_path = os.path.join(args.pt_out_path, time.strftime("%y%m%d-%H%M%S"))
    os.makedirs(args.pt_out_path, exist_ok=True)

    input_channels = 3                              # Adjust based on your input image channels
    hidden_size = int(args.hidden_size)                # Hidden size of LSTM (fintune) [128,256,512] 256
    num_layers = int(args.num_layers)                   # Number of LSTM layers (fintune) [1,2,3] 2
    num_linear_layers = int(args.num_linear_layers)     # Number of linear layers (fintune) [1,2,3] 3
    drop_out = 0.2                                  # Dropout rate
    kernel_size = 3                                 # Size of the convolutional kernel
    num_image = 6                                   # Number of images in the sequence
    weight_for_loss=10                              # Weight for the loss function (fintune)

    #print all the hyperparameters
    print("input_channels: ", input_channels)
    print("hidden_size: ", hidden_size)
    print("num_layers: ", num_layers)
    print("num_linear_layers: ", num_linear_layers)
    print("drop_out: ", drop_out)
    print("kernel_size: ", kernel_size)
    print("num_image: ", num_image)
    print("weight_for_loss: ", weight_for_loss)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=CNN(input_channels, hidden_size, num_layers,num_linear_layers, kernel_size,num_image,drop_out).to(device)

    train_dataset = CustomDataset(root_dir=args.dataset_path, train=True)
    valid_dataset = CustomDataset(root_dir=args.dataset_path, train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=24, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=12, shuffle=False)

    #binary cross entropy loss
    # criterion = nn.BCELoss()
    # mean squared error loss
    criterion = nn.MSELoss()

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    patience = 5
    save_path = args.pt_out_path

    model=train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, patience, save_path, device,weight_for_loss)
    # model.load_state_dict(torch.load("/home/c_yeung/workspace6/python/shot_posture/tracklet_classification/params/240221-033541-finetune-bce-11/epoch_7.pth"))
    
    # validation(valid_loader, model)
    # pdb.set_trace()
    valid_folders=valid_dataset.folders
    #get the unique id before the first _
    valid_clips=[int(i.split('_')[0]) for i in valid_folders]
    #get the unique clip id
    valid_clips=set(valid_clips)
    valid_clips=sorted(valid_clips)
    correct_pred=0
    incorrect_pred=0
    total_pred=0
    for clips in valid_clips:
        traklet_label=[]
        traklet_prob=[]
        #get the valid folder with the same clip id
        tracklet_list=[i for i in valid_folders if int(i.split('_')[0])==clips]
        for tracklet in tracklet_list:
            image_folder = []
            for j in ["10.jpg","11.jpg","12.jpg","13.jpg","14.jpg","15.jpg"]:
                image = cv2.imread(os.path.join(args.dataset_path,'valid',tracklet,j))
                #change the channel to the first dimension
                image = image.transpose((2, 0, 1))
                image = image/255
                image_folder.append(image)

            image_folder = np.array(image_folder)
            image=torch.tensor(image_folder).float()
            label=int(tracklet.split('_')[-1])
            #add the batch dim
            image=image.unsqueeze(0).to(device)
            model.eval()
            with torch.no_grad():
                output=model(image)
                output=output.cpu().numpy()

                traklet_label.append(label)
                traklet_prob.append(output[0])
            
        #get the max prob index
        max_index = traklet_prob.index(max(traklet_prob))
        if traklet_label[max_index]==1:
            correct_pred+=1
        else:
            # pdb.set_trace()
            incorrect_pred+=1
        total_pred+=1
    print("Clips accuracy, correct_pred: ", correct_pred, "incorrect_pred: ", incorrect_pred, "total_pred: ", total_pred)
        
