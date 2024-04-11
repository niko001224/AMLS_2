#%%
import os
import math
import argparse
import pandas as pd
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch import nn
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

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
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        img_labels = pd.read_csv(annotations_file)
        img_labels['img_path'] = img_labels.iloc[:, 0].apply(lambda x: os.path.join(img_dir, x))
        self.img_labels = img_labels[img_labels['img_path'].apply(os.path.exists)]
            
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx]['img_path']
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]  
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
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
      
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    #tb_writer = SummaryWriter()

    data_transform = {
        "train": transforms.Compose([transforms.Resize(384),
                                     transforms.CenterCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(30),  
                                     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
                                     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10), 
                                     ToPILImage(),  
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(384),
                                   transforms.CenterCrop(224),
                                   ToPILImage(),  
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    
    train_dataset = CustomImageDataset(
        annotations_file='/home/uceexuq/AMLS2-2/cassava-leaf-disease-classification/train.csv', 
        img_dir='/home/uceexuq/AMLS2-2/cassava-leaf-disease-classification/train_images', 
        transform=data_transform['train'])

    val_dataset = CustomImageDataset(
        annotations_file='/home/uceexuq/AMLS2-2/cassava-leaf-disease-classification/train.csv', 
        img_dir='/home/uceexuq/AMLS2-2/cassava-leaf-disease-classification/val_images', 
        transform=data_transform['val'])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]) 
    print('Using {} dataloader workers every process'.format(nw))
    
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=nw,
                              )

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=nw,
                            )

    num_classes = 5  
    model = model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    model.heads[0] = torch.nn.Linear(model.heads[0].in_features, num_classes)
    model.to(device)


    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    loss_fn = LabelSmoothingCrossEntropy()
    
    def train(dataloader, model, loss_fn, optimizer):
        model.train()  
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
        model.eval() 
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

    early_stopping = EarlyStopping(patience=10, verbose=True, path='./weights/early_stopping_model.pt')
    
    train_losses, val_losses, train_accuricies, val_accuricies = [], [], [], []
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train(train_loader, model, loss_fn, optimizer)
        train_losses.append(train_loss)
        train_accuricies.append(train_acc)
        val_loss, val_acc = validate(val_loader, model, loss_fn)
        val_losses.append(val_loss)
        val_accuricies.append(val_acc)
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        scheduler.step()
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Validate Loss: {val_loss}, Train Accuracy: {train_acc}, Validate Accuracy: {val_acc}")

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
        
    torch.save(model, './VIT_b_16')

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuricies, label='Train Accuracy')
    plt.plot(val_accuricies, label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--model-name', default='')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0')
    opt = parser.parse_args()
    

    main(opt)
# %%
