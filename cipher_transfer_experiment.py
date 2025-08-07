
import os
import numpy as np
import torch
from torchvision import datasets, transforms
import models_YaTC
from engine import evaluate
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from study_latent_space_yatc import build_dataset, load_embedding_model, generate_patch_embeddings


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

def get_config(args, device):
    configs  = [
        {'name': 'AES128',
         'model_path': args.aes128_model,
         'data_path': args.aes128_data},
        {'name': 'AES256',
         'model_path': args.aes256_model,
         'data_path': args.aes256_data},
        {'name': 'CHACHA20',
         'model_path': args.chacha_model,
         'data_path': args.chacha_data},
        {'name': 'MIX',
         'model_path': args.mix_model,
         'data_path': args.mix_data}
    ]

    for config in configs:
        config['data'] = load_test_dl(config['data_path'])
        num_classes = len(config['data'].dataset.classes)
        config['model'] = load_model(config['model_path'], num_classes=num_classes, device=device)
        config['embedder'] = load_embedding_model(config['model_path'], num_classes=num_classes, device=device)

    return configs

def calculate_transferability_matrix(configs, device):
    n = len(configs)
    results = np.zeros((n,n))


    # evaluate all models
    for i, config_a in enumerate(configs):
        for j, config_b in enumerate(configs):
            print(f'Testing {config_a["name"]} model on {config_b["name"]} data')
            res = evaluate(config_b['data'], config_a['model'], device)
            acc = res['acc1']
            results[i][j] = acc
                       

    # Print the results matrix in a readable way
    print("\nAccuracy Matrix (rows: models, cols: data):")
    header = "        " + "".join([f"{c['name']:>10}" for c in configs])
    print(header)
    for i, row in enumerate(results):
        row_str = f"{configs[i]['name']:<8} " + "".join([f"{val:10.2f}" for val in row])
        print(row_str)

    return results


def plot_transferability_matrix(results, configs):
    labels = [c['name'] for c in configs]

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(results, cmap='YlGnBu')

    # Set axis ticks and labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Rotate the tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate each cell with the accuracy value
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f"{results[i, j]:.2f}", ha="center", va="center", color="black")

    # Title and axis labels
    ax.set_title("Model Accuracy Heatmap")
    ax.set_xlabel("Data Source")
    ax.set_ylabel("Model")

    # Add color bar
    fig.colorbar(cax)

    plt.tight_layout()
    plt.show()


def calculate_variance(configs, device):
    vars = []
    for config in configs:
        print(f'Calculating variance for {config["name"]} model...')
        embeddings, _ = generate_patch_embeddings(config['embedder'], config['data'], device)
        var = embeddings.var(unbiased=False).item()
        print(f'Variance for {config["name"]} model: {var}')
        vars.append(var)
    return vars

def polt_variance(vars, configs):
    labels = [c['name'] for c in configs]
    plt.bar(labels, vars)
    plt.xlabel('Model')
    plt.ylabel('Variance')
    plt.title('Variance of Embeddings')
    plt.show()
    



if __name__ == "__main__":
    parser =   ArgumentParser()
    parser.add_argument("--aes128_model", help="Path to the model")
    parser.add_argument("--aes256_model", help="Path to the model")
    parser.add_argument("--chacha_model", help="Path to the model")
    parser.add_argument("--mix_model", help="Path to the model")
    parser.add_argument("--aes128_data", help="Path to the dataset (root, not train or test)")
    parser.add_argument("--aes256_data", help="Path to the dataset (root, not train or test)")
    parser.add_argument("--chacha_data", help="Path to the dataset (root, not train or test)")
    parser.add_argument("--mix_data", help="Path to the dataset (root, not train or test)")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # load data and models
    configs = get_config(args, device)


    # transferability matrix
    results = calculate_transferability_matrix(configs, device)
    plot_transferability_matrix(results, configs)   

    # now compare variance
    vars = calculate_variance(configs, device)
    polt_variance(vars, configs)
        


    
    


