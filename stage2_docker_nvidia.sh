#!/usr/bin/env bash
set -euo pipefail

echo "[0/8] Sanity checks..."
if ! mountpoint -q /data; then
  echo "[ERROR] /data is not mounted. Run stage1 (disk) and reboot first."
  exit 1
fi

echo "[1/8] Install Docker..."
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker

echo "[2/8] Add current user to docker group..."
sudo usermod -aG docker "$USER"
echo "[INFO] Group change needs a new login shell to take effect."
echo "[INFO] After script finishes, you can run: newgrp docker  (or log out/in)."

echo "[3/8] Prepare Docker data-root at /data/docker..."
sudo mkdir -p /data/docker
sudo chown -R root:root /data/docker

echo "[4/8] Write /etc/docker/daemon.json (data-root + nvidia runtime) once..."
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json > /dev/null <<'EOF'
{
  "data-root": "/data/docker",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
EOF

sudo systemctl daemon-reload
sudo systemctl restart docker

echo "[5/8] Verify Docker Root Dir..."
sudo docker info | grep -E "Docker Root Dir|Runtimes|Default Runtime" || true

echo "[6/8] Install NVIDIA Container Toolkit..."
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /etc/apt/keyrings/nvidia-container-toolkit.gpg

curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

echo "[7/8] Configure docker runtime via nvidia-ctk (then restore daemon.json to desired final state)..."
# Backup current daemon.json because nvidia-ctk may overwrite/modify it.
sudo cp -a /etc/docker/daemon.json /etc/docker/daemon.json.pre-nvidia-ctk

# This may inject default-runtime or other fields; we allow it to run.
sudo nvidia-ctk runtime configure --runtime=docker

# Now force the final desired daemon.json (your canonical version).
sudo tee /etc/docker/daemon.json > /dev/null <<'EOF'
{
  "data-root": "/data/docker",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
EOF

sudo systemctl restart docker

echo "[8/8] Final verification..."
echo "---- docker info (root dir + runtimes) ----"
sudo docker info | grep -E "Docker Root Dir|Runtimes|Default Runtime" || true

echo "---- GPU test inside container ----"
sudo docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

echo "[DONE] If you want to run docker without sudo:"
echo "  1) run: newgrp docker"
echo "  or 2) log out and log back in"
