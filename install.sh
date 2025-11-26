s#!/bin/bash
# Adaptive Installation Script
# Inspired by Claude Code's installation approach

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="$HOME/.adaptive"
BIN_DIR="$HOME/.local/bin"
REPO_URL="https://github.com/madhurprash/adaptive.git"

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                        ║${NC}"
echo -e "${BLUE}║         ADAPTIVE INSTALLER             ║${NC}"
echo -e "${BLUE}║                                        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Check if Python 3.12+ is available
check_python() {
    echo -e "${BLUE}→${NC} Checking Python version..."

    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | awk '{print $2}')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

        if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 12 ]; then
            echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION found"
            PYTHON_CMD="python3"
            return 0
        fi
    fi

    echo -e "${RED}✗${NC} Python 3.12+ is required but not found"
    echo -e "${YELLOW}  Please install Python 3.12 or later:${NC}"
    echo -e "    macOS: brew install python@3.12"
    echo -e "    Ubuntu: sudo apt install python3.12"
    exit 1
}

# Check if git is installed
check_git() {
    echo -e "${BLUE}→${NC} Checking git..."

    if ! command -v git &> /dev/null; then
        echo -e "${RED}✗${NC} git is required but not found"
        echo -e "${YELLOW}  Please install git:${NC}"
        echo -e "    macOS: brew install git"
        echo -e "    Ubuntu: sudo apt install git"
        exit 1
    fi

    echo -e "${GREEN}✓${NC} git found"
}

# Check if uv is installed
check_uv() {
    echo -e "${BLUE}→${NC} Checking uv..."

    if ! command -v uv &> /dev/null; then
        echo -e "${YELLOW}⚠${NC} uv not found, installing..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
    fi

    echo -e "${GREEN}✓${NC} uv found"
}

# Clone or update repository
install_adaptive() {
    echo -e "${BLUE}→${NC} Installing Adaptive..."

    if [ -d "$INSTALL_DIR" ]; then
        echo -e "${YELLOW}⚠${NC} Existing installation found at $INSTALL_DIR"
        read -p "  Update existing installation? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}→${NC} Updating Adaptive..."
            cd "$INSTALL_DIR"
            git pull origin main
        else
            echo -e "${YELLOW}⚠${NC} Skipping installation"
            return 0
        fi
    else
        echo -e "${BLUE}→${NC} Cloning Adaptive repository..."
        git clone "$REPO_URL" "$INSTALL_DIR"
        cd "$INSTALL_DIR"
    fi

    echo -e "${GREEN}✓${NC} Adaptive installed to $INSTALL_DIR"
}

# Install Python dependencies
install_dependencies() {
    echo -e "${BLUE}→${NC} Installing Python dependencies..."

    cd "$INSTALL_DIR"

    # Install using uv
    uv pip install -e . --system

    echo -e "${GREEN}✓${NC} Dependencies installed"
}

# Create symlink in bin directory
create_symlink() {
    echo -e "${BLUE}→${NC} Creating symlink..."

    # Ensure bin directory exists
    mkdir -p "$BIN_DIR"

    # Get the installed adaptive script location
    ADAPTIVE_PATH=$(which adaptive || echo "")

    if [ -z "$ADAPTIVE_PATH" ]; then
        echo -e "${YELLOW}⚠${NC} adaptive command not found in PATH"
        echo -e "${YELLOW}  You may need to restart your terminal or run:${NC}"
        echo -e "    export PATH=\"\$HOME/.local/bin:\$PATH\""
    else
        echo -e "${GREEN}✓${NC} adaptive command is available at: $ADAPTIVE_PATH"
    fi
}

# Update shell configuration
update_shell_config() {
    echo -e "${BLUE}→${NC} Updating shell configuration..."

    SHELL_CONFIG=""
    if [ -n "$BASH_VERSION" ]; then
        SHELL_CONFIG="$HOME/.bashrc"
    elif [ -n "$ZSH_VERSION" ]; then
        SHELL_CONFIG="$HOME/.zshrc"
    fi

    if [ -n "$SHELL_CONFIG" ]; then
        # Check if PATH is already configured
        if ! grep -q "\.local/bin" "$SHELL_CONFIG" 2>/dev/null; then
            echo "" >> "$SHELL_CONFIG"
            echo "# Added by Adaptive installer" >> "$SHELL_CONFIG"
            echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> "$SHELL_CONFIG"
            echo -e "${GREEN}✓${NC} Updated $SHELL_CONFIG"
        else
            echo -e "${GREEN}✓${NC} Shell configuration already up to date"
        fi
    fi
}

# Main installation flow
main() {
    echo ""
    check_python
    check_git
    check_uv
    echo ""

    install_adaptive
    install_dependencies
    create_symlink
    update_shell_config

    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                        ║${NC}"
    echo -e "${GREEN}║   ✓ INSTALLATION COMPLETE!            ║${NC}"
    echo -e "${GREEN}║                                        ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo ""
    echo -e "  1. ${YELLOW}Restart your terminal${NC} or run:"
    echo -e "     ${BLUE}source ~/.bashrc${NC}  # or ~/.zshrc"
    echo ""
    echo -e "  2. ${YELLOW}Configure Adaptive:${NC}"
    echo -e "     ${BLUE}adaptive config set model us.anthropic.claude-sonnet-4-20250514-v1:0${NC}"
    echo -e "     ${BLUE}adaptive config set platform langsmith${NC}"
    echo -e "     ${BLUE}adaptive config set-key langsmith${NC}"
    echo ""
    echo -e "  3. ${YELLOW}Run Adaptive:${NC}"
    echo -e "     ${BLUE}adaptive run${NC}"
    echo ""
    echo -e "For help: ${BLUE}adaptive --help${NC}"
    echo -e "Documentation: ${BLUE}https://github.com/madhurprash/adaptive${NC}"
    echo ""
}

# Run installation
main
