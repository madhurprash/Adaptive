#!/bin/bash

# Self-Healing Agent Installation Script
#
# This script installs the self-healing-agent CLI tool.
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/yourusername/self-healing-agent/main/scripts/install.sh | bash
#   or
#   bash scripts/install.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/yourusername/self-healing-agent"
INSTALL_DIR="$HOME/.self-healing-agent"
BIN_DIR="$HOME/.local/bin"
VERSION="latest"

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ${NC}  $1"
}

print_success() {
    echo -e "${GREEN}✓${NC}  $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC}  $1"
}

print_error() {
    echo -e "${RED}✗${NC}  $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
detect_os() {
    case "$(uname -s)" in
        Darwin*)
            echo "macos"
            ;;
        Linux*)
            echo "linux"
            ;;
        MINGW*|MSYS*|CYGWIN*)
            echo "windows"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# Function to check Python version
check_python() {
    if ! command_exists python3; then
        print_error "Python 3 is not installed. Please install Python 3.12 or later."
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 12 ]); then
        print_warning "Python 3.12+ is recommended. You have Python $PYTHON_VERSION"
    else
        print_success "Python $PYTHON_VERSION detected"
    fi
}

# Function to check/install uv
check_uv() {
    if command_exists uv; then
        print_success "uv is already installed"
        return 0
    fi

    print_info "Installing uv package manager..."

    if command_exists curl; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command_exists wget; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        print_error "Neither curl nor wget is available. Please install one of them."
        exit 1
    fi

    # Source the shell config to get uv in PATH
    export PATH="$HOME/.cargo/bin:$PATH"

    if command_exists uv; then
        print_success "uv installed successfully"
    else
        print_error "Failed to install uv. Please install it manually from https://docs.astral.sh/uv/"
        exit 1
    fi
}

# Function to install the package
install_package() {
    print_info "Installing self-healing-agent..."

    # Create installation directory
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$BIN_DIR"

    # Clone or update the repository
    if [ -d "$INSTALL_DIR/.git" ]; then
        print_info "Updating existing installation..."
        cd "$INSTALL_DIR"
        git pull
    else
        print_info "Cloning repository..."
        git clone "$REPO_URL" "$INSTALL_DIR"
        cd "$INSTALL_DIR"
    fi

    # Install using uv
    print_info "Installing dependencies..."
    uv pip install -e "$INSTALL_DIR"

    # Create symlinks in bin directory
    print_info "Creating command symlinks..."

    # Find the installed commands
    PYTHON_BIN_DIR=$(python3 -c "import site; print(site.USER_BASE + '/bin')")

    if [ -f "$PYTHON_BIN_DIR/self-healing-agent" ]; then
        ln -sf "$PYTHON_BIN_DIR/self-healing-agent" "$BIN_DIR/self-healing-agent"
        ln -sf "$PYTHON_BIN_DIR/self-healing-agent" "$BIN_DIR/evolve"
        print_success "Command symlinks created"
    else
        print_warning "Could not find installed commands. You may need to add Python's bin directory to your PATH."
    fi
}

# Function to update shell configuration
update_shell_config() {
    local shell_config=""
    local current_shell=$(basename "$SHELL")

    case "$current_shell" in
        bash)
            shell_config="$HOME/.bashrc"
            ;;
        zsh)
            shell_config="$HOME/.zshrc"
            ;;
        fish)
            shell_config="$HOME/.config/fish/config.fish"
            ;;
        *)
            print_warning "Unknown shell: $current_shell"
            return
            ;;
    esac

    if [ -f "$shell_config" ]; then
        if ! grep -q "$BIN_DIR" "$shell_config"; then
            print_info "Adding $BIN_DIR to PATH in $shell_config"
            echo "" >> "$shell_config"
            echo "# Self-Healing Agent" >> "$shell_config"
            echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> "$shell_config"
            print_success "Updated $shell_config"
        else
            print_success "$BIN_DIR already in PATH"
        fi
    fi
}

# Function to verify installation
verify_installation() {
    export PATH="$BIN_DIR:$PATH"

    if command_exists evolve; then
        print_success "Installation successful!"
        echo ""
        print_info "Try running: evolve --help"
        echo ""
        print_info "If the command is not found, you may need to:"
        echo "  1. Open a new terminal, or"
        echo "  2. Run: source ~/.bashrc (or ~/.zshrc)"
    else
        print_warning "Installation completed but 'evolve' command not found in PATH"
        print_info "Please add $BIN_DIR to your PATH and restart your terminal"
    fi
}

# Main installation flow
main() {
    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║     Self-Healing Agent Installation Script         ║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo ""

    OS=$(detect_os)
    print_info "Detected OS: $OS"

    # Check prerequisites
    check_python
    check_uv

    # Install the package
    install_package

    # Update shell configuration
    update_shell_config

    # Verify installation
    verify_installation

    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║            Installation Complete! 🎉                ║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo ""
    print_info "Documentation: $REPO_URL#readme"
    print_info "Issues: $REPO_URL/issues"
    echo ""
}

# Run the installation
main "$@"
