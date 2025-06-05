
import argparse
import os
import torch
import os
from torchvision import datasets, transforms
import models_YaTC
from engine import evaluate
# import timm
# assert timm.__version__ == "0.3.2"  # version check

def build_dataset(is_train, path):
    mean = [0.5]
    std = [0.5]

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    root = os.path.join(path, 'train' if is_train else 'test')
    dataset = datasets.ImageFolder(root, transform=transform)

    return dataset

def load_test_dl(path, bs =  256):
    ds = build_dataset(is_train=False, path=path)
    return torch.utils.data.DataLoader(ds, bs, shuffle=False, num_workers=4, pin_memory=True)

def load_model(path, num_classes, device):
    model = models_YaTC.__dict__['TraFormer_YaTC'](
        num_classes=num_classes,
        drop_path_rate=0.1,
    )
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    return model

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dl = load_test_dl(args.data_path)
    num_classes = len(dl.dataset.classes)
    model = load_model(args.model_path, num_classes, device)
    evaluate(dl, model, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the model")
    parser.add_argument("data_path", help="Path to the dataset (root, not train or test)")
    args = parser.parse_args()
    print("Tesing model: {}".format(args.model_path))
    print("On dataset: {}".format(args.data_path))
    main(args)



