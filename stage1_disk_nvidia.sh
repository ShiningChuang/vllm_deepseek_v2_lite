#!/usr/bin/env bash
set -euo pipefail

DISK="/dev/sda"
PART="${DISK}4"
MOUNTPOINT="/data"

echo "[1/6] Sanity checks..."
if [[ $EUID -eq 0 ]]; then
  echo "Please run as a normal user (script uses sudo internally)." >&2
  exit 1
fi

if ! command -v sudo >/dev/null 2>&1; then
  echo "sudo not found." >&2
  exit 1
fi

if [[ ! -b "$DISK" ]]; then
  echo "Disk $DISK not found. Check with: lsblk" >&2
  exit 1
fi

echo "[2/6] Creating partition $PART with fdisk (non-interactive)..."
# This feeds: n -> 4 -> Enter -> Enter -> w
# It creates partition number 4 using default first/last sectors.
# NOTE: If partition 4 already exists, fdisk may fail; that's intended to stop the script.
printf "n\n4\n\n\nw\n" | sudo fdisk "$DISK"

echo "[3/6] Informing kernel of partition table changes..."
sudo partprobe "$DISK"

echo "[4/6] Formatting $PART as ext4..."
# -F forces creation even if it looks like it contains a filesystem (be careful)
sudo mkfs.ext4 -F "$PART"

echo "[5/6] Mounting $PART to $MOUNTPOINT and persisting in /etc/fstab (UUID)..."
sudo mkdir -p "$MOUNTPOINT"
sudo mount "$PART" "$MOUNTPOINT"

UUID="$(sudo blkid -s UUID -o value "$PART")"
if [[ -z "$UUID" ]]; then
  echo "Failed to read UUID for $PART" >&2
  exit 1
fi

# Avoid duplicate fstab entries
if sudo grep -qE "UUID=${UUID}[[:space:]]+${MOUNTPOINT}[[:space:]]" /etc/fstab; then
  echo "fstab already contains an entry for UUID=$UUID mounted at $MOUNTPOINT"
else
  echo "UUID=$UUID  $MOUNTPOINT  ext4  defaults  0 2" | sudo tee -a /etc/fstab > /dev/null
fi

echo "[Verify] df -h | grep $MOUNTPOINT"
df -h | grep "$MOUNTPOINT" || true

echo "[6/6] Installing NVIDIA driver (ubuntu-drivers) then rebooting..."
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall

echo "Rebooting now..."
sudo reboot
