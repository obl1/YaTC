import argparse
import numpy as np
import os
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from tqdm import tqdm
import os
from torchvision import datasets, transforms
import models_YaTC
from engine import evaluate
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv

import timm
assert timm.__version__ == "0.3.2"  # version check

class TrafficTransformerEmbedder(nn.Module):
    def __init__(self, classifier_model):
        super().__init__()
        self.model = classifier_model

    def forward_packet_features(self, x, i):
        B = x.shape[0]
        x = self.model.patch_embed(x)

        cls_tokens = self.model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        cls_pos = self.model.pos_embed[:, :1, :]
        packet_pos = self.model.pos_embed[:, i * 80 + 1: i * 80 + 81, :]
        pos_all = torch.cat((cls_pos, packet_pos), dim=1)
        x = x + pos_all

        x = self.model.pos_drop(x)
        for blk in self.model.blocks:
            x = blk(x)

        cls = x[:, :1, :]
        x = x[:, 1:, :].reshape(B, 4, 20, -1).mean(axis=1)
        x = torch.cat((cls, x), dim=1)

        return self.model.fc_norm(x), cls
    
    def embed_packets(self,x):
        packet_embeddings = []
        B, C, H, W = x.shape
        x = x.reshape(B, C, 5, -1)
        for i in range(5):
            packet_x = x[:, :, i, :].reshape(B, C, -1, 40)
            packet_features, cls = self.forward_packet_features(packet_x, i)
            packet_embeddings.append(cls)
            if i == 0:
                all_packets = packet_features
            else:
                all_packets = torch.cat((all_packets, packet_features), dim=1)

        return all_packets, torch.cat(packet_embeddings)

    def embed(self, x):
        B, C, H, W = x.shape
        x, _ = self.embed_packets(x)
        
        for blk in self.model.blocks:
            x = blk(x)

        x = x.reshape(B, 5, 21, -1)[:, :, 0, :]  # Get CLS tokens
        x = x.mean(dim=1)

        return self.model.fc_norm(x)  # Final 192-dim embedding

    def forward(self, x):
        return self.embed(x)
    

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

def load_dl(path, bs =  256):
    ds = build_dataset(True, path)
    return torch.utils.data.DataLoader(ds, bs, shuffle=False, num_workers=4, pin_memory=True)

def load_embedding_model(path, num_classes, device):
    model = models_YaTC.__dict__['TraFormer_YaTC'](
        num_classes=num_classes,
        drop_path_rate=0.1,
    )
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    
    model.to(device)
    embedder = TrafficTransformerEmbedder(model)
    embedder.to(device)
    return embedder


def claculate_cosine_sim_matrix(embeddings, labels):
    """
    calculates the average cosine similarity between embeddings of the same class
    to embeddings of each different class.
    embeddings: [N, D]
    labels: [N]
    """
    num_classes = len(np.unique(labels.cpu()))
    device = embeddings.device
    embeddings = F.normalize(embeddings, p=2, dim=1)
    class_embeddings = []
    for cls in np.unique(labels.cpu()):
        cls_emb = embeddings[labels == cls]
        class_embeddings.append(cls_emb)

    

    similarity_matrix = torch.empty((num_classes, num_classes), device=device)
    for i in range(num_classes):
        ei = class_embeddings[i]  # [Ni, D]
        # no samples of this class
        if ei.size(0) == 0: 
            similarity_matrix[i, :] = float('nan')
            continue
        # no samples of this class
        for j in range(num_classes):
            ej = class_embeddings[j]  # [Nj, D]
            if ej.size(0) == 0:
                similarity_matrix[i, j] = float('nan')
                continue

            # actual cosine similarity. vectors are normalized
            sim_ij = torch.mm(ei, ej.T)  # [Ni, Nj]
            avg_sim = sim_ij.mean()
            similarity_matrix[i, j] = avg_sim

    return similarity_matrix

def plot_sim_matrix(sim_matrix, label_names, path=None):
    """
    plots the similarity matrix as outputed by claculate_cosine_sim_matrix
    """
    num_classes = len(label_names)
    sim_matrix_np = sim_matrix.detach().cpu().numpy()

    plt.figure(figsize=(10, 8))
    im = plt.imshow(sim_matrix_np, cmap='viridis', aspect='auto')
    plt.title("Average Cosine Similarity Between Classes")
    plt.xlabel("Class")
    plt.ylabel("Class")
    plt.xticks(ticks=np.arange(num_classes), labels=label_names, rotation=90)
    plt.yticks(ticks=np.arange(num_classes), labels=label_names)
    cbar = plt.colorbar(im)
    cbar.set_label('Cosine Similarity')

    plt.tight_layout()
    # plt.show()

    if path:
        plt.savefig(path)




def save_embeddings(emb, y, name):
    torch.save((emb, y), name)

def load_embeddings(path):
    return torch.load(path)


def generate_patch_embeddings(embedder, dl, device):
    with torch.no_grad():
        embeddings = []
        emb_labels = []
        for x, y in tqdm(dl, total=len(dl)): 
            x = x.to(device)
            y = y.to(device)
            embeddings.append(embedder.embed(x))
            emb_labels.append(y)
        embeddings = torch.cat(embeddings)
        emb_labels = torch.cat(emb_labels)

        return embeddings, emb_labels

def generate_packet_embeddings(embedder, dl, device):
    embeddings = []
    emb_labels = []
    for x, y in tqdm(dl, total=len(dl)): 
        x = x.to(device)
        y = y.to(device)
        _, image_pkt_embeddings = embedder.embed_packets(x)
        embeddings.append(image_pkt_embeddings)
        for _ in range(5):
            emb_labels.append(y)
    embeddings = torch.cat(embeddings)
    emb_labels = torch.cat(emb_labels)

    return embeddings, emb_labels

def print_notable_similarities(sim_matrix, classes, sim_threshold=0.3, recon_threshold=0.5):
    # notably similar classes
    
    for i, name in enumerate(classes):
        sims = sim_matrix[i]
        most_similar = torch.topk(sims, 5)
        problem_sims = [x for x in zip(most_similar.indices, most_similar.values) if x[0] != i]
        recon = problem_sims[0][1] / sims[i]
        if recon > recon_threshold:
            print(f"Class={name}")
            print(f"\tSelf similarity: {sims[i]}")
            print(f"\tRatio between smost similar and self: {recon}")
            print(f"\tMost similar:")
            for j, sim in problem_sims:
                if sim > sim_threshold:
                    print(f"\t\t{classes[j]}: {sim}")


def get_category_labels(path, labels, train_labels):
    with open('domain_classifications.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        data = [row for row in reader]

    data = [{
        'domain': row[0],
        'category': row[1],
        'description': row[2]
    } for row in data]

    for item in data:
        item['domain_idx'] = labels.index(item['domain'])
        
    cat_unique = list(set([item['category'] for item in data]))
    cat_map = zip(cat_unique, range(len(cat_unique)))
    cat_map = dict(cat_map)

    for item in data:
        item['category_idx'] = cat_map[item['category']]

    domain_to_category = {item['domain_idx']: item['category_idx'] for item in data}

    category_labels = [domain_to_category[i] for i in [x.item() for x in train_labels]]
    category_labels = torch.tensor(category_labels)
    category_names = [f'{cat_unique[i]} ({len([x for x in domain_to_category.values() if x == i])})' for i in range(len(cat_unique))]
    return category_labels, category_names



    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='data')
    parser.add_argument('--model_path', default=None, type=str, help='model')
    parser.add_argument('--embeddings_path', default='embeddings.pt', type=str, help='embeddings') 
    parser.add_argument('--packet_level', default=False, help='packet level')  
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size')
    parser.add_argument('--seed', default=42, type=int, help='seed')
    parser.add_argument('--categories', default=False, type=bool, help='Plot by domains or by categories')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # just load the data
    dl = load_dl(args.data_path, bs=args.batch_size)
    

    embeddings, emb_labels = None, None
    if os.path.exists(args.embeddings_path):
        print("Loading embeddings...")
        embeddings, emb_labels = load_embeddings(args.embeddings_path)
    else:
        print("Generating embeddings...")
        embedder = load_embedding_model(args.model_path, len(dl.dataset.classes), device)
        if not args.packet_level:
            embeddings, emb_labels = generate_patch_embeddings(embedder, dl, device)
        else:
            embeddings, emb_labels = generate_packet_embeddings(embedder, dl, device)

        save_embeddings(embeddings, emb_labels, args.embeddings_path)

    labels = emb_labels
    names = dl.dataset.classes

    if args.categories:
        labels, names = get_category_labels('domain_classifications.csv', dl.dataset.classes, emb_labels)    

    sim_matrix = claculate_cosine_sim_matrix(embeddings, labels)
    plot_sim_matrix(sim_matrix, names, 'plot.png')
    print_notable_similarities(sim_matrix, names)
    
