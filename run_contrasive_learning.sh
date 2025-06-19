#!/bin/bash

# Parse command line arguments
dataset_name="$1"
google_url="$2"
epochs="${3:-200}"  # default to 200 if not provided

if [ -z "$dataset_name" ]; then
  echo "Usage: $0 <dataset_name> [<google_drive_url>] [epochs (default=200)]"
  exit 1
fi

ENV_NAME="yatc"
PRETRAINED_MODEL_NAME="./output_dir/pretrained-model.pth"
DATASET_ZIP_NAME="data.zip"
DATASET_DIR="datasets/${dataset_name}"

# Check if conda is available
if ! command -v conda &> /dev/null; then
  echo "Conda is not installed. Please install Miniconda or Anaconda first."
  exit 1
fi

# Check if conda environment exists
if conda info --envs | grep -q "^${ENV_NAME}[[:space:]]"; then
  echo "Conda environment '$ENV_NAME' already exists. Activating it..."
else
  echo "Creating new conda environment '$ENV_NAME' with Python 3.8..."
  conda create -y -n "$ENV_NAME" python=3.8 || exit 1
fi

# Activate the conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME" || exit 1

# Upgrade pip and install packages
pip install --upgrade pip || exit 1
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html || exit 1
pip install gdown timm==0.3.2 numpy==1.19.5 scikit-learn==0.24.2 tensorboard scikit-image matplotlib || exit 1

mkdir -p output_dir

# Download pretrained model if not present
if [ ! -f "$PRETRAINED_MODEL_NAME" ]; then
  echo "Downloading pretrained model..."
  gdown https://drive.google.com/uc?id=1wWmZN87NgwujSd2-o5nm3HaQUIzWlv16 -O "$PRETRAINED_MODEL_NAME" || exit 1
fi

# Download fine-tuning dataset if not present
if [ ! -d "$DATASET_DIR" ]; then
  echo "Dataset ${dataset_name} not found in datasets/, downloading it..."

  mkdir -p datasets

  if [ -z "$google_url" ]; then
    echo "No Google Drive URL provided."
    exit 1
  fi

  gdown "$google_url" -O "datasets/$DATASET_ZIP_NAME" || exit 1
  unzip "datasets/$DATASET_ZIP_NAME" -d "datasets" || exit 1
  rm "datasets/$DATASET_ZIP_NAME"
fi

# Set PYTHONPATH and run training
export PYTHONPATH=.
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=12355
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

echo "Starting training with:"
echo "  Dataset: $DATASET_DIR"
echo "  Pretrained model: $PRETRAINED_MODEL_NAME"
echo "  Epochs: $epochs"

python3 train_supcon_packets.py \
  --base_model_path "$PRETRAINED_MODEL_NAME" \
  --data_folder "$DATASET_DIR" \
  --epochs "$epochs"
