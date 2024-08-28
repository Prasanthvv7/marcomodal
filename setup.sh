#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Define variables
INDICTRANS2_REPO="https://github.com/AI4Bharat/IndicTrans2.git"
INDICTRANS_TOKENIZER_REPO="https://github.com/VarunGumma/IndicTransTokenizer"
TRAINER_REPO="https://github.com/gokulkarthik/Trainer"
TTS_REPO="https://github.com/gokulkarthik/TTS"
INDIC_TTS_REPO="https://github.com/AI4Bharat/Indic-TTS.git"
GUJARATI_CHECKPOINT_URL="https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/gu.zip"
ENGLISH_CHECKPOINT_URL="https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/en.zip"

# Function to clone a repository if it doesn't exist
clone_repo() {
    local repo_url=$1
    local repo_dir=$(basename "$repo_url" .git)
    if [ ! -d "$repo_dir" ]; then
        git clone "$repo_url"
    else
        echo "$repo_dir already exists, skipping clone."
    fi
}

# Function to install Python packages
install_python_packages() {
    python3 -m pip install "$@"
}

# Function to download and unzip language files
download_and_unzip() {
    local url=$1
    local zip_file=$(basename "$url")
    wget -q "$url" -O "$zip_file"
    unzip -o "$zip_file"
}

# Function to update config.json with the correct speakers_file path
update_config_json() {
    local config_file=$1
    local speakers_file_path=$2
    jq --arg sp "$speakers_file_path" '.speakers_file = $sp' "$config_file" > tmp.$$.json && mv tmp.$$.json "$config_file"
}

# Check if jq is installed, if not, install it
if ! command -v jq &> /dev/null; then
    echo "jq could not be found, installing..."
    sudo apt-get update
    sudo apt-get install -y jq
fi

# Clone the IndicTrans2 repository
clone_repo "$INDICTRANS2_REPO"

# Navigate to the huggingface_interface directory and install required Python packages
cd IndicTrans2/huggingface_interface
install_python_packages nltk sacremoses pandas regex mock mosestokenizer
python3 -c "import nltk; nltk.download('punkt')"
install_python_packages bitsandbytes scipy accelerate datasets sentencepiece
cd -

# Clone the IndicTransTokenizer repository and install the package
clone_repo "$INDICTRANS_TOKENIZER_REPO"
cd IndicTransTokenizer
install_python_packages --editable ./
cd -

# Download and unzip the Indic-TTS checkpoints
download_and_unzip "$GUJARATI_CHECKPOINT_URL"
download_and_unzip "$ENGLISH_CHECKPOINT_URL"

# Update config.json files with the correct speakers_file paths
update_config_json "gu/fastpitch/config.json" "gu/fastpitch/speaker.pth"
update_config_json "en/fastpitch/config.json" "en/fastpitch/speaker.pth"

# Install system dependencies
apt-get update
apt-get install -y libsndfile1-dev ffmpeg enchant libenchant1c2a libenchant-dev

# Install PyTorch
install_python_packages -U torch torchvision torchaudio
install_python_packages wtpsplit
install_python_packages bleach jq
install_python_packages flask
# Clone and setup Trainer repository
clone_repo "$TRAINER_REPO"
cd Trainer
install_python_packages -e .
cd -

# Clone and setup TTS repository
clone_repo "$TTS_REPO"
cd TTS
install_python_packages -e .
cd -

# Clone the Indic-TTS repository
clone_repo "$INDIC_TTS_REPO"