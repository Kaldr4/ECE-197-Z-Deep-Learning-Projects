import torch
from engine import evaluate
import label_utils
from dataloader import DrinksDataset
from train import get_transform
import utils
from torch.utils.data import DataLoader
import gdown
import tarfile
import os

def main():
    url = 'https://drive.google.com/file/d/1AdMbVK110IKLG7wJKhga2N2fitV1bVPA/view'
    output = 'drinks.tar.gz'
    gdown.download(url = url, output = output, quiet = False, fuzzy = True)
    tar = tarfile.open(output)
    tar.extractall()
    tar.close()

    if not os.path.exists('model.pkl'):
        url = 'https://github.com/Kaldr4/EEE-197-Assignment-2/releases/download/Model/model.pkl'
        output = 'model.pkl'
        gdown.download(url = url, output = output, quiet = False, fuzzy = True)
        
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    test_dict, test_classes = label_utils.build_label_dictionary("drinks/labels_test.csv")
    test_split = DrinksDataset(test_dict, get_transform(train = False))
    test_loader = DataLoader(test_split,
                            batch_size=4,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            collate_fn=utils.collate_fn)

    model = torch.load("model.pkl")
    model.to(device)
    evaluate(model, test_loader, device=device)

if __name__ == "__main__":
    main()