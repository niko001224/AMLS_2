#%%
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, ToPILImage, Normalize, Resize, CenterCrop, RandomHorizontalFlip, RandomRotation, RandomAffine
import os
from torch import nn

class EarlyStopping:
    def __init__(self, patience=7, delta=0, path='checkpoint.pt', verbose=False):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} to {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class LabelSmoothingCrossEntropy(nn.Module):
    
    def __init__(self, smoothing=0.1):
        
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        
        log_prob = torch.nn.functional.log_softmax(input, dim=-1) 
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1)  
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))  
        loss = (-weight * log_prob).sum(dim=-1).mean() 
        return loss 
#print(os.getcwd())
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # 加载标注文件S
        img_labels = pd.read_csv(annotations_file)
        
        # 检查图片文件是否存在，并仅保留存在的图片记录
        img_labels['img_path'] = img_labels.iloc[:, 0].apply(lambda x: os.path.join(img_dir, x))
        self.img_labels = img_labels[img_labels['img_path'].apply(os.path.exists)]
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx]['img_path']
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]  # 假设标签在第二列
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch

# 修改这部分来匹配你的类别数量
num_classes = 5  

model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
model.heads[0] = torch.nn.Linear(model.heads[0].in_features, num_classes)

# 移动模型到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
model.to(device)

from torch import nn, optim
import math
import torch.optim.lr_scheduler as lr_scheduler

pg = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(pg, lr=0.001, momentum=0.9, weight_decay=5E-5)
lf = lambda x: ((1 + math.cos(x * math.pi / 40)) / 2) * (1 - 0.01) + 0.01  # cosine
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

# 定义损失函数和优化器
loss_fn = LabelSmoothingCrossEntropy()
#optimizer = optim.SGD(model.parameters(), lr=0.001)

def train(dataloader, model, loss_fn, optimizer):
    model.train()  # 设置模型为训练模式
    total_loss, total_correct, total = 0, 0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        total += y.size(0)
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total
    return avg_loss, avg_acc

def validate(dataloader, model, loss_fn):
    model.eval()  # 设置模型为评估模式
    total_loss, total_correct, total = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            total_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total += y.size(0)
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total
    return avg_loss, avg_acc


train_losses, validate_losses, train_accuracies, validate_accuracies = [], [], [], []
transform_train = Compose([Resize(384),
                     CenterCrop(224),
                     RandomHorizontalFlip(),
                     RandomRotation(30), 
                     ToPILImage(),  
                     ToTensor(),
                     Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

transform_val = Compose([Resize(384),
                         CenterCrop(224),
                         ToPILImage(),  
                         ToTensor(),
                         Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

train_dataset = CustomImageDataset(
    annotations_file='/home/uceexuq/AMLS2-2/cassava-leaf-disease-classification/train.csv', 
    img_dir='/home/uceexuq/AMLS2-2/cassava-leaf-disease-classification/train_images', 
    transform=transform_train)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

validate_dataset = CustomImageDataset(
    annotations_file='/home/uceexuq/AMLS2-2/cassava-leaf-disease-classification/train.csv', 
    img_dir='/home/uceexuq/AMLS2-2/cassava-leaf-disease-classification/val_images', 
    transform=transform_val)
validate_dataloader = DataLoader(validate_dataset, batch_size=64, shuffle=True)

early_stopping = EarlyStopping(patience=10, verbose=True)
    
epochs = 40  # 设置训练的总轮次
for epoch in range(epochs):
    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
    validate_loss, validate_acc = validate(validate_dataloader, model, loss_fn)
    train_losses.append(train_loss)
    validate_losses.append(validate_loss)
    train_accuracies.append(train_acc)
    validate_accuracies.append(validate_acc)
    early_stopping(validate_loss, model)

    if early_stopping.early_stop:
        print("Early stopping")
        break

    print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Validate Loss: {validate_loss}, Train Accuracy: {train_acc}, Validate Accuracy: {validate_acc}")



print(train(train_dataloader, model, loss_fn, optimizer))
torch.save(model, './VIT_b_16_initial')


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(validate_losses, label='Validate Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(validate_accuracies, label='Validate Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# %%
