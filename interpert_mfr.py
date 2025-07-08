import matplotlib.patches as patches
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import os
import itertools

def get_data_loaders(data_dir, image_size, batch_size):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, len(train_dataset.classes)

class Marker:
    def __init__(self, ax):
        self.ax = ax
        
        # This is from ChatGPT, I checked some of the values
        self.PROPERTIES = {
            # IP header (assuming no options)
            "ver_ihl": [(0, 1)],
            "tos": [(1, 2)],
            "len": [(2, 4)],
            "id": [(4, 6)],
            "flags_frag": [(6, 8)],
            "ttl": [(8, 9)],
            "proto": [(9, 10)],
            "chksum": [(10, 12)],
            "ip_addr": [(12, 16), (16, 20)],   # [src_ip, dst_ip]

            # TCP header
            "port": [(20, 22), (22, 24)],      # [src_port, dst_port]
            "seq": [(24, 28)],
            "ack": [(28, 32)],
            "offset_flags": [(32, 34)],
            "tcp_win": [(34, 36)],
            "tcp_chksum": [(36, 38)],
            "urg_ptr": [(38, 40)],
        }

    

    def _mark_mfr_range(self, ranges: List[Tuple[int, int]], color, label=None) -> None:
        for packet_id in range(5):
            base_row = packet_id * 8 
            for i, (start, end) in enumerate(ranges):
                row = base_row + start // 40
                start = start % 40
                height = 1
                width = end - start
                curr_label = label if label and packet_id == 0 and i == 0 else None
                rect = patches.Rectangle(
                    (start - 0.5, row - 0.5),  # (x, y): left-most column - 0.5, row - 0.5
                    width, height,                  # width = 5 columns (cols 5–9), height = 1 row
                    linewidth=2,
                    edgecolor=color,
                    facecolor='none',
                    label = curr_label
                )
                self.ax.add_patch(rect)

                if label:
                    self.ax.legend()



    def _mark_mfr_property(self, property: str, color):
        ranges = self.PROPERTIES.get(property, None)
        if ranges:
            self._mark_mfr_range(ranges, color, label=property)
        else:
            raise ValueError(f"Unknown property: {property}")
        
    def _mark_multi_properties_mfr(self, properties: List[str]):
        colors = plt.cm.get_cmap('tab20', len(properties))
        for i, property in enumerate(properties):
            color = colors(i)
            self._mark_mfr_property(property, color)

    def _mark_all_properties_mfr(self):
        self._mark_multi_properties_mfr(list(self.PROPERTIES.keys()))

    def mark(self, properties: Union[List[str], None] = None):
        """
        Main marking function. can get a list of str properties,
        or leave blank to mark all
        """
        if properties:
            self._mark_multi_properties_mfr(properties)
        else:
            self._mark_all_properties_mfr()


def permute_packets(tensor, packet_size=8):
    """
    tensor: (C, H, W)
    returns: list of (C, H, W)
    That was hand tested by me.
    """
    C, H, W = tensor.shape
    num_packets = H // packet_size

    # Split into (num_packets, C, packet_size, W)
    packets = tensor.unfold(1, packet_size, packet_size).permute(1, 0, 3, 2)

    permuted_tensors = []
    for p in itertools.permutations(range(num_packets)):
        # Correct: concatenate along packet_size axis
        permuted = torch.cat([packets[i] for i in p], dim=1)
        # Shape: (C, total_H, W)
        permuted_tensors.append(permuted)

    return permuted_tensors


def load_mfr_resnet(model_path: str, num_classes: int, device):
    resnet = resnet18()
    # adapt for YATC MFR
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    resnet.maxpool = nn.Identity()
    resnet.fc = nn.Linear(512, num_classes)

    resnet.load_state_dict(torch.load(model_path))
    resnet.eval()
    resnet.to(device)

    return resnet


def plot_gradcam(resnet: nn.Module, images, labels, device, max=16, marker=True, path=None):
    assert len(images) == len(labels)
    assert max % 8 == 0

    fig, axes = plt.subplots(int(max / 8), 8, figsize=(30, 12))
    axes = axes.flatten()  # Flatten to easily index with i

    for i, (img, label) in enumerate(zip(images[:max], labels[:max])):
        ax = axes[i]
        if marker:
            marker = Marker(ax)
            marker.mark(['ip_addr', 'tcp_win'])
        img_input = img.unsqueeze(0).to(device)
        output = resnet(img_input)
        pred = output.argmax().item()

        # GradCAM
        resnet_target_layers = [resnet.layer4[-1]]
        with GradCAM(model=resnet, target_layers=resnet_target_layers) as cam:
            targets = [ClassifierOutputTarget(pred)]
            grayscale_cam = cam(input_tensor=img_input, targets=targets, aug_smooth=True, eigen_smooth=True)[0]

        # Convert image to RGB
        img_np = img[0].cpu().numpy()  # Assuming single-channel
        img_rgb = np.stack([img_np] * 3, axis=-1)
        visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)

        ax.imshow(visualization)
        ax.set_title(label)
        ax.axis("off")

    plt.tight_layout()
    if not path:
        plt.show()
    else:
        plt.savefig(path)



if __name__ == '__main__':
    # this is our permutations experiment
    parser = ArgumentParser()
    parser.add_argument("model_path", help="Path to the model")
    parser.add_argument("data_path", help="Path to the data")
    parser.add_argument('--save_images', default=True, help='Save images to folder')
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    

    model_path = args.model_path
    data_path = args.data_path

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dl_train, dl_test, num_classes = get_data_loaders(data_path, 40, 16)
    resnet = load_mfr_resnet(model_path, num_classes, device)

    batch_imgs = None
    batch_labels = None
    for x, y in dl_train:
        batch_imgs = x
        batch_labels = y
        break
    
    path = None
    if args.save_images:
        path = Path('gradcam_results')
        path.mkdir(exist_ok=True)

    imgs = batch_imgs[:5]
    labels = [dl_train.dataset.classes[label] for label in batch_labels[:5]]
    for i, (image, name) in enumerate(zip(imgs, labels)):
        print(f'{i}: Plot permutations for {name}')
        perms = permute_packets(image)
        for j in range(0,120,40):
            b = perms[j:j+40]
            l = [f'{name}-{k}' for k in range(j, j+40)]
            plot_gradcam(resnet,b,l,device, max=40,marker=False, path=path / f'{name}_part_{(j//40) + 1}.jpeg')







