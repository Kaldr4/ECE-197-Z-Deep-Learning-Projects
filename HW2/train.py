import torch
import label_utils
import torchvision
import datetime
from dataloader import DrinksDataset
from torchvision import transforms 
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import gdown
import tarfile
import os

def get_model_instance(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main():

    if not os.path.isdir('drinks'):
        url = 'https://drive.google.com/file/d/1AdMbVK110IKLG7wJKhga2N2fitV1bVPA/view'
        output = 'drinks.tar.gz'
        gdown.download(url = url, output = output, quiet = False, fuzzy = True)
        tar = tarfile.open(output)
        tar.extractall()
        tar.close()

    torch.cuda.empty_cache()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 4
    train_dict, train_classes = label_utils.build_label_dictionary("drinks/labels_train.csv")
    test_dict, test_classes = label_utils.build_label_dictionary("drinks/labels_test.csv")
    train_split = DrinksDataset(train_dict, get_transform(train = True))
    test_split = DrinksDataset(test_dict, get_transform(train = False))
    train_loader = DataLoader(train_split, batch_size=4,shuffle=True,num_workers=4,pin_memory=True,collate_fn=utils.collate_fn)
    test_loader = DataLoader(test_split,
                            batch_size=4,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            collate_fn=utils.collate_fn)

    model = get_model_instance(num_classes)

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)

    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, test_loader, device=device)

    torch.save(model, 'model.pkl')

if __name__ == "__main__":
    main()
