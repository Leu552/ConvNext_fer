"""
Adapted from https://github.com/usef-kh/fer/tree/master/data
"""
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from my_dataset import MyDataSet

img_size = 48

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None, augment=False):
        self.images = images
        self.labels = labels
        self.transform = transform

        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = np.array(self.images[idx])

        img = Image.fromarray(img)
        # print('=========================')
        # print(type(img))
        # print(len(img.split()))
        # exit()
        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx]).type(torch.long)
        sample = (img, label)

        return sample


def load_data(path='data/fer2013/fer2013.csv'):
    fer2013 = pd.read_csv(path)
    emotion_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

    return fer2013, emotion_mapping


def prepare_data(data):
    """ Prepare data for modeling
        input: data frame with labels und pixel data
        output: image and label array """

    image_array = np.zeros(shape=(len(data), img_size, img_size))
    image_label = np.array(list(map(int, data['emotion'])))

    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (img_size, img_size))
        image_array[i] = image

    return image_array, image_label


def get_dataloaders(path='data/fer2013/fer2013.csv', bs=64, augment=True):
    """ Prepare train, val, & test dataloaders
        Augment training data using:
            - cropping 裁剪
            - shifting (vertical/horizental)
            - horizental flipping 水平翻转
            - rotation 旋转
        input: path to fer2013 csv file
        output: (Dataloader, Dataloader, Dataloader) """

    fer2013, emotion_mapping = load_data(path)

    xtrain, ytrain = prepare_data(fer2013[fer2013['Usage'] == 'Training'])
    xval, yval = prepare_data(fer2013[fer2013['Usage'] == 'PrivateTest'])
    xtest, ytest = prepare_data(fer2013[fer2013['Usage'] == 'PublicTest'])

    mu, st = 0, 255

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        # transforms.TenCrop(40),
        transforms.ToTensor(),
        # transforms.Lambda(lambda crops: torch.stack(
        #     [transforms.ToTensor()(crop) for crop in crops])),
        # transforms.Lambda(lambda tensors: torch.stack(
        #     [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            # 随机裁剪
            transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
            # 颜色抖动
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
            transforms.RandomApply(
                # 仿射变换
                [transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
                # 水平翻转
            transforms.RandomHorizontalFlip(),
            # 随机旋转
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            # transforms.FiveCrop(40),
            transforms.ToTensor(),
            # transforms.Lambda(lambda crops: torch.stack(
            #     [transforms.ToTensor()(crop) for crop in crops])),
            # transforms.Lambda(lambda tensors: torch.stack(
            #     [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
            # transforms.Lambda(lambda tensors: torch.stack(
            #     # 随机图像擦除
            #     [transforms.RandomErasing()(t) for t in tensors])),

        ])
    else:
        train_transform = test_transform

    # X = np.vstack((xtrain, xval))
    # Y = np.hstack((ytrain, yval))

    train = CustomDataset(xtrain, ytrain, train_transform)
    # print('++++++++++++++++++++++++')
    # print(type(train[0][0]))
    # print(train[0][0].shape)
    # exit()
    val = CustomDataset(xval, yval, test_transform)
    test = CustomDataset(xtest, ytest, test_transform)

    trainloader = DataLoader(train, batch_size=bs, shuffle=True, num_workers=2)
                                        
    valloader = DataLoader(val, batch_size=32, shuffle=True, num_workers=2)
    testloader = DataLoader(test, batch_size=32, shuffle=True, num_workers=2)

    return trainloader, valloader, testloader
