import os
import glob
import binascii
from PIL import Image
import scapy.all as scapy
from tqdm import tqdm
import numpy as np
from pathlib import Path
import shutil
import random
from urllib.parse import quote
from argparse import ArgumentParser



def makedir(path):
    try:
        os.mkdir(path)
    except Exception as E:
        pass


def read_MFR_bytes(pcap_dir):
    packets = scapy.rdpcap(pcap_dir)
    data = []
    for packet in packets:
        header = (binascii.hexlify(bytes(packet['IP']))).decode()
        try:
            payload = (binascii.hexlify(bytes(packet['Raw']))).decode()
            header = header.replace(payload, '')
        except:
            payload = ''
        if len(header) > 160:
            header = header[:160]
        elif len(header) < 160:
            header += '0' * (160 - len(header))
        if len(payload) > 480:
            payload = payload[:480]
        elif len(payload) < 480:
            payload += '0' * (480 - len(payload))
        data.append((header, payload))
        if len(data) >= 5:
            break
    if len(data) < 5:
        for i in range(5-len(data)):
            data.append(('0'*160, '0'*480))
    final_data = ''
    for h, p in data:
        final_data += h
        final_data += p
    return final_data

def MFR_generator(flows_pcap_path, output_path):
    flows = glob.glob(flows_pcap_path + "/*/*/*.pcap")
    makedir(output_path)
    makedir(output_path + "/train")
    makedir(output_path + "/test")
    classes = glob.glob(flows_pcap_path + "/*/*")
    for cla in tqdm(classes):
        makedir(cla.replace(flows_pcap_path, output_path))
    for flow in tqdm(flows):
        content = read_MFR_bytes(flow)
        content = np.array([int(content[i:i + 2], 16) for i in range(0, len(content), 2)])
        fh = np.reshape(content, (40, 40))
        fh = np.uint8(fh)
        im = Image.fromarray(fh)
        im.save(flow.replace('.pcap', '.png').replace(flows_pcap_path, output_path))


def yatc_split(base_path, split_ratio=0.8, random_seed=42):
    """
    Added by OBL to split the dataset into consistent train and test sets
    """
    # Configuration
    base_dir = Path(base_path)
    train_dir = base_dir / "train"
    test_dir = base_dir / "test"

    # mix dir is the other dir - child of base that isn't train or test
    mix_dir = [x for x in base_dir.iterdir() if x.name not in ["train", "test"]][0]

    # Ensure reproducibility
    random.seed(random_seed)

    # Ensure train and test directories exist
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # For each class folder in mix/
    for class_folder in tqdm.tqdm([x for x in mix_dir.iterdir()]):
        if class_folder.is_dir():
            images = list(class_folder.glob("*"))
            random.shuffle(images)
            
            split_idx = int(len(images) * split_ratio)
            train_images = images[:split_idx]
            test_images = images[split_idx:]

            # Create class subfolders in train/ and test/
            sanitized_name = quote(class_folder.name, safe="")
            train_class_dir = train_dir / sanitized_name
            test_class_dir = test_dir / sanitized_name

            # Ensure it's not a file already
            if train_class_dir.exists() and not train_class_dir.is_dir():
                raise Exception(f"{train_class_dir} exists and is not a directory!")

            if test_class_dir.exists() and not test_class_dir.is_dir():
                raise Exception(f"{test_class_dir} exists and is not a directory!")

            train_class_dir.mkdir(parents=True, exist_ok=True)
            test_class_dir.mkdir(parents=True, exist_ok=True)
            # Move or copy images
            for img in train_images:
                shutil.copy(img, train_dir / class_folder.name / img.name)
            for img in test_images:
                shutil.copy(img, test_dir / class_folder.name / img.name)

    print("Dataset split complete with seed =", random_seed)

def main(args):
    MFR_generator(args.src_path, args.dst_path)
    yatc_split(args.dst_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("src_path", help="Path to the source directory")
    parser.add_argument("dst_path", help="Path to the destination directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_ratio", type=float, default=0.8)
    args = parser.parse_args()
    main(args)