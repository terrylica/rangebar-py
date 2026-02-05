#!/usr/bin/env bash
# Install Pueue on Linux (BigBlack)
#
# This script downloads and installs Pueue from GitHub releases.
# No sudo required - installs to ~/.local/bin
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/terrylica/rangebar-py/main/scripts/setup-pueue-linux.sh | bash
#   # OR
#   ./scripts/setup-pueue-linux.sh
#
# After installation:
#   pueued -d              # Start daemon
#   pueue status           # Check status

set -euo pipefail

PUEUE_VERSION="v4.0.2"
INSTALL_DIR="$HOME/.local/bin"
ARCH="x86_64"

echo "=== Pueue Installation for Linux ==="
echo "Version: $PUEUE_VERSION"
echo "Install directory: $INSTALL_DIR"
echo ""

# Create install directory if needed
mkdir -p "$INSTALL_DIR"

# Detect architecture
case "$(uname -m)" in
    x86_64)
        ARCH="x86_64"
        ;;
    aarch64)
        ARCH="aarch64"
        ;;
    armv7l)
        ARCH="armv7"
        ;;
    *)
        echo "❌ Unsupported architecture: $(uname -m)"
        exit 1
        ;;
esac

echo "Detected architecture: $ARCH"

# Download URLs
PUEUE_URL="https://github.com/Nukesor/pueue/releases/download/${PUEUE_VERSION}/pueue-${ARCH}-unknown-linux-musl"
PUEUED_URL="https://github.com/Nukesor/pueue/releases/download/${PUEUE_VERSION}/pueued-${ARCH}-unknown-linux-musl"

echo ""
echo "Downloading pueue..."
curl -sSL "$PUEUE_URL" -o "$INSTALL_DIR/pueue"
chmod +x "$INSTALL_DIR/pueue"
echo "✅ Downloaded pueue"

echo "Downloading pueued..."
curl -sSL "$PUEUED_URL" -o "$INSTALL_DIR/pueued"
chmod +x "$INSTALL_DIR/pueued"
echo "✅ Downloaded pueued"

# Verify installation
echo ""
echo "Verifying installation..."
if "$INSTALL_DIR/pueue" --version; then
    echo "✅ pueue installed successfully"
else
    echo "❌ pueue installation failed"
    exit 1
fi

# Check if ~/.local/bin is in PATH
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    echo ""
    echo "⚠️  $INSTALL_DIR is not in your PATH"
    echo ""
    echo "Add this to your ~/.bashrc or ~/.zshrc:"
    echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo ""
    echo "Then run: source ~/.bashrc"
fi

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Next steps:"
echo "  1. Start the daemon:     pueued -d"
echo "  2. Check status:         pueue status"
echo "  3. Add a test job:       pueue add -- echo 'Hello from Pueue'"
echo "  4. View job output:      pueue log 0"
echo ""
echo "For systemd auto-start (optional):"
echo "  mkdir -p ~/.config/systemd/user"
echo "  cat > ~/.config/systemd/user/pueued.service << 'EOF'"
echo "[Unit]"
echo "Description=Pueue Daemon"
echo "After=network.target"
echo ""
echo "[Service]"
echo "ExecStart=%h/.local/bin/pueued -v"
echo "Restart=on-failure"
echo ""
echo "[Install]"
echo "WantedBy=default.target"
echo "EOF"
echo "  systemctl --user daemon-reload"
echo "  systemctl --user enable --now pueued"
echo ""
