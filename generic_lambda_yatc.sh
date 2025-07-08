#!/bin/bash

# Parse command line arguments
dataset_name="$1"
google_url="$2"
epochs="${4:-200}"  # default to 200 if not provided

if [[ -z "$dataset_name" ]]; then
  echo "Usage: $0 <dataset_name> [<google_drive_url>] [epochs (default=200)]"
  exit 1
fi

ENV_NAME="yatc"
PRETRAINED_MODEL_NAME=./output_dir/pretrained-model.pth
FINE_TUNING_DATASET_ZIP_NAME=data.zip
FINE_TUNING_DATASET_DIR="datasets/${dataset_name}"

# Check if conda is available
if ! command -v conda &> /dev/null; then
  echo "Conda is not installed. Please install Miniconda or Anaconda first."
  exit 1
fi

# Check if conda environment exists
if conda info --envs | grep -q "^$ENV_NAME\s"; then
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
  gdown https://drive.google.com/uc?id=1wWmZN87NgwujSd2-o5nm3HaQUIzWlv16 -O "$PRETRAINED_MODEL_NAME" || exit 1
fi

# Download fine-tuning dataset if not present
if [ ! -d "$FINE_TUNING_DATASET_DIR" ]; then
  echo "Dataset ${dataset_name} not found in datasets/, downloading it..."

  # Make sure the datasets directory exists
  mkdir -p datasets

  # Check if google drive url is provided
  if [ -z $google_url ]; then
    echo "No google drive url provided"
    exit 1
  fi

  # Download and unzip directly into datasets
  gdown "$google_url" -O "datasets/$FINE_TUNING_DATASET_ZIP_NAME" || exit 1
  unzip "datasets/$FINE_TUNING_DATASET_ZIP_NAME" -d datasets || exit 1
  rm "datasets/$FINE_TUNING_DATASET_ZIP_NAME"
fi

# Set PYTHONPATH and run training
export PYTHONPATH=.
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=12355
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

pwd

python3 fine-tune.py \
  --blr 2e-3 \
  --epochs "$epochs" \
  --data_path "./$FINE_TUNING_DATASET_DIR" \
  --num_workers 2 \
  --output_dir "${dataset_name}_out"
