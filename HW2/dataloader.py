import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image


class DrinksDataset(torch.utils.data.Dataset):
    def __init__(self, dictionary, transforms=None):
        self.dictionary = dictionary
        self.transforms = transforms

    def __len__(self):
        return len(self.dictionary)

    def __getitem__(self, idx):
        key = list(self.dictionary.keys())[idx]

        boxes = []
        labels = []
        arr = self.dictionary[key]
        arr_count = len(arr)
        # for i in range(arr_count):
        #     boxes.append([objs[i][0],objs[i][2],objs[i][1],objs[i][3]])
        #     labels.append(int(objs[i][-1]))
        for i in range(arr_count):    
            boxes.append([arr[i][0],arr[i][2], arr[i][1],arr[i][3]])
            labels.append(int(arr[i][-1]))

        # boxes = np.array(boxes)
        # labels = np.array(labels)

        boxes = torch.as_tensor(boxes,dtype=torch.float32)
        labels = torch.as_tensor(labels,dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((arr_count,), dtype=torch.int64)
        img = Image.open(key).convert("RGB")

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
