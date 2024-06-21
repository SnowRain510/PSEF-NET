import os
import sys
import pickle

from skimage import io
import matplotlib.pyplot as plt
import numpy
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import json


data_transform = {
         "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
         "val": transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

data_root = os.path.abspath('/Data20T/data20t/data20t/wrl')  # get data root path
image_path = data_root + "/CNN/resnet/IgG_3/"  # flower data set path

def CIFAR100Train(dataset):

     train_dataset = datasets.ImageFolder(root=image_path+"train",
                                          transform=data_transform["train"])
     train_num = len(train_dataset)

     # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
     flower_list = train_dataset.class_to_idx
     cla_dict = dict((val, key) for key, val in flower_list.items())
     # write dict into json file
     json_str = json.dumps(cla_dict, indent=4)
     with open('class_indices.json', 'w') as json_file:
         json_file.write(json_str)

     return train_dataset

def CIFAR100Test(dataset):

     test_dataset = datasets.ImageFolder(root=image_path + "test",
                                             transform=data_transform["val"])
     val_num = len(test_dataset)

     return test_dataset