import torch
import torch.nn.functional as F
from torchvision import  transforms, datasets
import matplotlib.pyplot as plt
import models_YaTC
from pathlib import Path
import torch.nn as nn
from torch.utils.data import DataLoader
from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
from typing import List, Tuple, Union
from torchvision.datasets import ImageFolder



class PacketYaTC(nn.Module):
    """
    A wrapper for the YaTC model that only encodes packets.
    """
    def __init__(self, classifier_model):
        super().__init__()
        self.model = classifier_model

    def encode_single_packet(self, x, i):
        """
        Based on forward_packet_features, but only returns the CLS token.
        X: (B, 1, 8, 40) (1/5 of the MFR)
        I: Packet index

        Returns:
        cls: (B, 1, 192)
        """
        B = x.shape[0]
        x = self.model.patch_embed(x)
        cls_tokens = self.model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        cls_pos = self.model.pos_embed[:, :1, :]
        packet_pos = self.model.pos_embed[:, i*80+1:i*80+81, :]
        pos_all = torch.cat((cls_pos, packet_pos), dim=1)
        x = x + pos_all
        x = self.model.pos_drop(x)

        for j, blk in enumerate(self.model.blocks):
            x = blk(x)

        cls = x[:, :1, :]
        cls = cls.squeeze(1)

        return cls

    def forward_features(self, x):
        """
        input: (B, 1, 40, 40)
        output: (B*5, 192)
        """
        packet_embeddings = []
        B, C, H, W = x.shape
        x = x.reshape(B, C, 5, -1)
        for i in range(5):
            packet_x = x[:, :, i, :]
            packet_x = packet_x.reshape(B, C, -1, 40)
            packet_cls = self.encode_single_packet(packet_x, i)
            packet_embeddings.append(packet_cls)

        packet_embeddings = torch.stack(packet_embeddings).reshape(B*5,192)
            
        return packet_embeddings

    def forward(self, x):
        return self.forward_features(x)
    

def load_yatc_model(model_path, num_classes, device, from_pretrained=False):
    """
    Loads a YaTC model from a checkpoint in the given path.
    If it is from the pretrained model, it will also interpolate the position embedding and
    eliminates irelevant weights.
    """
    model = models_YaTC.__dict__['TraFormer_YaTC'](
        num_classes=num_classes,
        drop_path_rate=0.1,
    )
    checkpoint_model = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint_model['model']
    
    if from_pretrained:
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)
        
    else:
        model.load_state_dict(state_dict)

    model.to(device)
    return model

def collect_for_supcon(batch):
    X = torch.stack([item[0] for item in batch])
    batch_size = X.size(0)
    y = torch.arange(batch_size).repeat(5)
    return X, y

def build_supcon_dl(path: Union[str, Path], batch_size=128, is_train=True):
    if isinstance(path, str):
        path = Path(path)

    mean = [0.5]
    std = [0.5]

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    root = path / 'train' if is_train else path / 'test'
    dataset = datasets.ImageFolder(root, transform=transform)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collect_for_supcon)

    return dl