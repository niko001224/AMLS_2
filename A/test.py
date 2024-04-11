import torch
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, ToPILImage, Normalize, Resize, CenterCrop
import pandas as pd
import os

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #model_path = '/home/uceexuq/AMLS2-2/A/VIT_b_16_initial.pt' 
    model_path = 'VIT_b_16_initial.pt' 
    model = torch.load(model_path, map_location=device)  
    model.eval()  

    model = torch.load(model_path)
    model.eval()  
    transform = transforms.Compose([Resize(384),
                            CenterCrop(224),
                            ToPILImage(),  
                            ToTensor(),
                            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    tta_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        
    ])

    class CustomTTADataset(Dataset):
        def __init__(self, annotations_file, img_dir, transform=None, tta_transforms=None, tta_times=5):
            self.img_dir = img_dir
            self.transform = transform
            self.tta_transforms = tta_transforms
            self.tta_times = tta_times

            img_labels = pd.read_csv(annotations_file)
            img_labels['exists'] = img_labels.iloc[:, 0].apply(lambda x: os.path.exists(os.path.join(img_dir, x)))
            self.img_labels = img_labels[img_labels['exists']]

        def __len__(self):
            return len(self.img_labels)

        def __getitem__(self, idx):
            img_name = self.img_labels.iloc[idx, 0]  
            img_path = os.path.join(self.img_dir, img_name)
            image = read_image(img_path)
            label = self.img_labels.iloc[idx, 1]  

            if self.transform:
                image = self.transform(image)

            if self.tta_transforms:
                tta_images = [self.tta_transforms(image) for _ in range(self.tta_times)]
                image = torch.stack(tta_images)
            else:
                image = image.unsqueeze(0)  

            return image, label, img_name

    dataset = CustomTTADataset ( #annotations_file='/home/uceexuq/AMLS2-2/cassava-leaf-disease-classification/train.csv',
                                annotations_file='./cassava-leaf-disease-classification/train.csv',
                                img_dir='./cassava-leaf-disease-classification/test_images',
                                transform=transform,
                                tta_transforms=tta_transforms,
                                tta_times=5)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    correct = 0
    total = 0
    predictions = []

    for images, labels, image_names in dataloader:
        images = images.to(device).squeeze(0)  
        labels = labels.to(device)  
        outputs = torch.mean(torch.stack([model(img.unsqueeze(0)) for img in images]), 0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predictions.append((image_names[0], predicted.item()))

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy} %')
    df = pd.DataFrame(predictions, columns=['ImageName', 'Label'])
    df.to_csv('predictions.csv', index=False)

    return accuracy