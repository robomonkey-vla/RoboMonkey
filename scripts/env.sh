#!/bin/bash

# Exit on error
set -e

echo "Starting setup process..."

# Function to check command status
check_status() {
    if [ $? -eq 0 ]; then
        echo "✓ $1 completed successfully"
    else
        echo "✗ Error during $1"
        exit 1
    fi
}

# Basic setup
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod +x Miniconda3-latest-Linux-x86_64.sh
    ./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
    $HOME/miniconda3/bin/conda init bash
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    rm ./Miniconda3-latest-Linux-x86_64.sh
    check_status "Miniconda installation"
else
    echo "Conda is already installed. Skipping Miniconda installation."
fi


# Install required packages
echo "Installing required packages..."
sudo apt update
sudo apt install -y unzip libgl1-mesa-glx libosmesa6 ffmpeg libsm6 xvfb libxext6
check_status "Package installation"

# Git LFS setup
echo "Setting up Git and Git LFS..."
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install -y git-lfs
git lfs install
check_status "Git setup"

echo "Initializing conda..."
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc
