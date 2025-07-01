#!/usr/bin/env bash

# git clone https://github.com/charles-cai/MLX8-W3-Transfomers.git
# cd MLX8-W3-Transformers
# . .charles/gpu-setup.sh

apt update
apt install -y vim rsync git git-lfs nvtop htop tmux curl btop build-essential
git lfs install

# to run `ip a``, `ping 192.168.50.243`, `telnet 192.168.50.243 2230` 
apt install -y iproute2 iputils-ping install telnet

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# starship
curl -sS https://starship.rs/install.sh | sh -s -- --yes
echo 'eval "$(starship init bash)"' >> ~/.bashrc

mkdir -p "~/.config"
cat > ~/.config/startship.toml <<EOF
[directory]
truncation_length = 3
truncate_to_repo = false
fish_style_pwd_dir_length = 1
home_symbol = "~"
EOF

# duckdb
curl https://install.duckdb.org | sh
echo "export PATH='/root/.duckdb/cli/latest':\$PATH" >> ~/.bashrc

# redis
apt install sudo -y
sudo apt install -y lsb-release curl gpg
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
sudo chmod 644 /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
sudo apt update
sudo apt -y install redis

# fix UTF-8 locale issues
apt install locales -y

# Docker inside Docker? 
# To start (in GPU Docker, there's no systemd), and to ping
# /usr/bin/redis-server 
# redis-cli ping 

source ~/.bashrc

uv sync
# source .venv/bin/activate 
# we use uv run xxx.py instead of python xxx.py

echo "Setup complete - virtual environment activated. You can now run Python scripts directly."
echo "Run 'git lfs pull' to download large files."

which python
which uv

echo "!!Please finish the following tasks!!"
echo "Manully install Python extension in VS Code (remotely on SSH host for debugging)"
echo "cp .env.example to .env, and edit with your API_Keys!!"
echo "git config --global user.name 'Your Name'"
echo "git config --global user.email 'Your Email'"
echo "Setup complete!"

nvcc --version
nvidia-smi
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count()); print('CUDA device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')" || echo "PyTorch not installed"
