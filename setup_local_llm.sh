#!/bin/bash

# Setup script for local LLM deployment with healthcare scraper
# Supports Ollama and other local LLM servers

set -e

echo "🚀 Setting up Enhanced Healthcare Facility Scraper with Local LLM Support"

# Configuration
OLLAMA_MODELS=("llama3.2:3b" "phi3:mini" "llama3.1:8b")
SETUP_TYPE=${1:-"ollama"}  # ollama, webui, or docker

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check available memory
    MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$MEMORY_GB" -lt 4 ]; then
        print_warning "Only ${MEMORY_GB}GB RAM available. Recommend 8GB+ for local LLMs"
    else
        print_success "Memory check passed: ${MEMORY_GB}GB RAM available"
    fi
    
    # Check disk space
    DISK_GB=$(df -BG . | awk 'NR==2{print $4}' | sed 's/G//')
    if [ "$DISK_GB" -lt 10 ]; then
        print_warning "Only ${DISK_GB}GB disk space available. Models require 5-20GB"
    else
        print_success "Disk space check passed: ${DISK_GB}GB available"
    fi
    
    # Check for GPU (optional)
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected - will enable GPU acceleration"
        export USE_GPU=true
    else
        print_status "No GPU detected - using CPU mode"
        export USE_GPU=false
    fi
}

# Install Ollama
install_ollama() {
    print_status "Installing Ollama..."
    
    if command -v ollama &> /dev/null; then
        print_success "Ollama already installed"
    else
        curl -fsSL https://ollama.ai/install.sh | sh
        print_success "Ollama installed successfully"
    fi
    
    # Start Ollama service
    print_status "Starting Ollama service..."
    if systemctl is-active --quiet ollama; then
        print_success "Ollama service already running"
    else
        sudo systemctl start ollama
        sudo systemctl enable ollama
        print_success "Ollama service started"
    fi
    
    # Wait for Ollama to be ready
    print_status "Waiting for Ollama to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags > /dev/null; then
            print_success "Ollama is ready"
            break
        fi
        sleep 2
    done
}

# Download Ollama models
download_ollama_models() {
    print_status "Downloading Ollama models..."
    
    for model in "${OLLAMA_MODELS[@]}"; do
        print_status "Downloading model: $model"
        if ollama list | grep -q "$model"; then
            print_success "Model $model already available"
        else
            ollama pull "$model"
            print_success "Downloaded model: $model"
        fi
    done
}

# Setup Docker environment
setup_docker() {
    print_status "Setting up Docker environment..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_status "Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        print_success "Docker installed"
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        print_status "Installing Docker Compose..."
        sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
        print_success "Docker Compose installed"
    fi
}

# Setup Python environment
setup_python() {
    print_status "Setting up Python environment..."
    
    # Install enhanced requirements
    if [ -f "requirements_enhanced.txt" ]; then
        pip install -r requirements_enhanced.txt
        print_success "Python dependencies installed"
    else
        print_error "requirements_enhanced.txt not found"
        exit 1
    fi
}

# Test LLM providers
test_providers() {
    print_status "Testing LLM providers..."
    
    python main_enhanced.py --status
}

# Create sample configuration
create_config() {
    print_status "Creating sample configuration..."
    
    # Create sample sites file
    cat > input/sites.txt << EOF
# Healthcare facility websites to scrape
# One URL per line, comments start with #

https://lcca.com
https://genesishcc.com
https://brookdaleliving.com
https://sunriseseniorliving.com
EOF
    
    # Create environment file template
    cat > .env.template << EOF
# OpenRouter API key (for cloud models)
OPENROUTER_API_KEY=your_openrouter_key_here

# Ollama configuration (for local models)
OLLAMA_BASE_URL=http://localhost:11434

# Local API configuration
LOCAL_API_BASE_URL=http://localhost:8000
EOF
    
    print_success "Sample configuration created"
    print_status "Edit input/sites.txt to add your target websites"
    print_status "Copy .env.template to .env and add your API keys"
}

# Main setup function
main() {
    echo "🏥 Enhanced Healthcare Facility Scraper Setup"
    echo "Setup type: $SETUP_TYPE"
    echo ""
    
    check_requirements
    
    case $SETUP_TYPE in
        "ollama")
            print_status "Setting up with Ollama (local LLM)"
            install_ollama
            download_ollama_models
            setup_python
            ;;
        "docker")
            print_status "Setting up with Docker"
            setup_docker
            print_status "Use: docker-compose -f docker-compose.enhanced.yml --profile full up -d"
            ;;
        "webui")
            print_status "Setting up with Text Generation WebUI"
            setup_docker
            print_status "Use: docker-compose -f docker-compose.enhanced.yml --profile webui up -d"
            ;;
        *)
            print_error "Unknown setup type: $SETUP_TYPE"
            echo "Usage: $0 [ollama|docker|webui]"
            exit 1
            ;;
    esac
    
    create_config
    
    if [ "$SETUP_TYPE" = "ollama" ]; then
        test_providers
    fi
    
    echo ""
    print_success "Setup completed successfully!"
    echo ""
    echo "🚀 Quick Start Commands:"
    echo ""
    echo "# Check provider status"
    echo "python main_enhanced.py --status"
    echo ""
    echo "# Auto-detect and use best LLM provider"
    echo "python main_enhanced.py --url https://lcca.com --output ./results --llm"
    echo ""
    echo "# Use specific Ollama model"
    echo "python main_enhanced.py --url https://lcca.com --output ./results --provider ollama --model llama3.2:3b"
    echo ""
    echo "# Use OpenRouter (requires API key)"
    echo "export OPENROUTER_API_KEY='your-key'"
    echo "python main_enhanced.py --url https://lcca.com --output ./results --provider openrouter"
    echo ""
    echo "# Batch processing"
    echo "python main_enhanced.py --urls-file input/sites.txt --output ./results --llm"
    echo ""
}

# Run main function
main "$@"

