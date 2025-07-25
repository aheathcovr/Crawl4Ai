#!/bin/bash

# Optimized deployment script for 2 vCPU / 4GB RAM Digital Ocean Droplet
# Automatically configures settings based on available resources

set -e

# Configuration
PROJECT_NAME="healthcare-scraper-optimized"
ENVIRONMENT=${1:-production}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "🚀 Deploying Optimized Healthcare Facility Scraper"
echo "Environment: $ENVIRONMENT"
echo "Target: 2 vCPU / 4GB RAM Digital Ocean Droplet"

# Check system resources
check_system_resources() {
    print_status "Checking system resources..."
    
    # Get memory info
    TOTAL_RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    TOTAL_RAM_GB=$((TOTAL_RAM_KB / 1024 / 1024))
    AVAILABLE_RAM_KB=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
    AVAILABLE_RAM_GB=$((AVAILABLE_RAM_KB / 1024 / 1024))
    
    # Get CPU info
    CPU_COUNT=$(nproc)
    
    # Get disk info
    DISK_AVAILABLE_GB=$(df -BG . | awk 'NR==2{print $4}' | sed 's/G//')
    
    print_success "System Resources:"
    print_success "  RAM: ${TOTAL_RAM_GB}GB total, ${AVAILABLE_RAM_GB}GB available"
    print_success "  CPU: ${CPU_COUNT} cores"
    print_success "  Disk: ${DISK_AVAILABLE_GB}GB available"
    
    # Validate resources
    if [ "$TOTAL_RAM_GB" -lt 3 ]; then
        print_error "Insufficient RAM: ${TOTAL_RAM_GB}GB (minimum 4GB recommended)"
        print_warning "Will use ultra-conservative settings"
        export MEMORY_MODE="ultra_low"
    elif [ "$TOTAL_RAM_GB" -lt 4 ]; then
        print_warning "Low RAM: ${TOTAL_RAM_GB}GB (4GB recommended)"
        export MEMORY_MODE="low"
    else
        print_success "RAM check passed: ${TOTAL_RAM_GB}GB"
        export MEMORY_MODE="normal"
    fi
    
    if [ "$DISK_AVAILABLE_GB" -lt 5 ]; then
        print_error "Insufficient disk space: ${DISK_AVAILABLE_GB}GB"
        exit 1
    fi
}

# Install optimized dependencies
install_dependencies() {
    print_status "Installing optimized dependencies..."
    
    # Update system
    apt-get update -qq
    
    # Install essential packages only
    apt-get install -y -qq \
        python3-pip \
        curl \
        htop \
        psmisc
    
    # Install Python dependencies with memory optimization
    if [ "$MEMORY_MODE" = "ultra_low" ]; then
        # Minimal installation
        pip3 install --no-cache-dir requests beautifulsoup4 pandas
        print_warning "Ultra-low memory mode: LLM features disabled"
    else
        # Install enhanced requirements but with memory limits
        pip3 install --no-cache-dir -r requirements_enhanced.txt
        print_success "Enhanced dependencies installed"
    fi
}

# Setup Docker with resource limits
setup_docker() {
    print_status "Setting up Docker with resource optimization..."
    
    # Install Docker if not present
    if ! command -v docker &> /dev/null; then
        curl -fsSL https://get.docker.com -o get-docker.sh
        sh get-docker.sh
        usermod -aG docker $USER
    fi
    
    # Configure Docker daemon for low memory
    cat > /etc/docker/daemon.json << EOF
{
    "default-runtime": "runc",
    "storage-driver": "overlay2",
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    },
    "default-ulimits": {
        "memlock": {
            "Hard": -1,
            "Name": "memlock",
            "Soft": -1
        },
        "nofile": {
            "Hard": 1024,
            "Name": "nofile", 
            "Soft": 1024
        }
    }
}
EOF
    
    systemctl restart docker
    print_success "Docker configured for 4GB RAM"
}

# Create optimized directories
create_directories() {
    print_status "Creating optimized directory structure..."
    
    mkdir -p output input logs
    chmod 755 output input logs
    
    # Create sample input with memory-appropriate sites
    if [ ! -f "input/sites.txt" ]; then
        cat > input/sites.txt << EOF
# Optimized for 4GB RAM - process these sequentially
# Start with smaller sites first

https://lcca.com
https://genesishcc.com
EOF
        print_success "Created sample sites.txt with memory-optimized selection"
    fi
}

# Configure system for optimal performance
optimize_system() {
    print_status "Optimizing system for 4GB RAM..."
    
    # Increase swap if needed (but warn about performance)
    SWAP_SIZE=$(free -m | awk '/^Swap:/ {print $2}')
    if [ "$SWAP_SIZE" -lt 1024 ]; then
        print_warning "Low swap space detected (${SWAP_SIZE}MB)"
        print_status "Creating 1GB swap file for emergency use..."
        
        fallocate -l 1G /swapfile
        chmod 600 /swapfile
        mkswap /swapfile
        swapon /swapfile
        echo '/swapfile none swap sw 0 0' >> /etc/fstab
        
        print_success "Swap file created (will impact performance)"
    fi
    
    # Optimize kernel parameters for low memory
    cat >> /etc/sysctl.conf << EOF

# Optimizations for 4GB RAM system
vm.swappiness=10
vm.vfs_cache_pressure=50
vm.dirty_ratio=15
vm.dirty_background_ratio=5
EOF
    
    sysctl -p
    print_success "Kernel parameters optimized"
}

# Run deployment based on environment
run_deployment() {
    print_status "Running optimized deployment..."
    
    case $ENVIRONMENT in
        "production")
            if [ "$MEMORY_MODE" = "ultra_low" ]; then
                print_warning "Ultra-low memory: using traditional scraping only"
                python3 main_optimized.py --url https://lcca.com --output ./output
            else
                print_status "Running with LLM optimization for 4GB RAM"
                python3 main_optimized.py --url https://lcca.com --output ./output --llm --monitor
            fi
            ;;
        "docker")
            print_status "Starting optimized Docker services..."
            docker-compose -f docker-compose.optimized.yml --profile optimized up -d
            
            # Download appropriate models
            if [ "$MEMORY_MODE" != "ultra_low" ]; then
                print_status "Downloading lightweight models..."
                docker-compose -f docker-compose.optimized.yml --profile setup-light run model-downloader-light
            fi
            ;;
        "test")
            print_status "Running system resource test..."
            python3 main_optimized.py --status
            ;;
        *)
            print_error "Unknown environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
}

# Monitor resources during deployment
monitor_resources() {
    print_status "Monitoring resources during deployment..."
    
    # Background resource monitor
    (
        while true; do
            MEMORY_PERCENT=$(free | awk '/^Mem:/ {printf "%.1f", $3/$2 * 100.0}')
            CPU_LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
            
            if (( $(echo "$MEMORY_PERCENT > 90" | bc -l) )); then
                print_error "Memory usage critical: ${MEMORY_PERCENT}%"
            elif (( $(echo "$MEMORY_PERCENT > 80" | bc -l) )); then
                print_warning "Memory usage high: ${MEMORY_PERCENT}%"
            fi
            
            sleep 30
        done
    ) &
    
    MONITOR_PID=$!
    
    # Cleanup function
    cleanup() {
        print_status "Cleaning up..."
        kill $MONITOR_PID 2>/dev/null || true
    }
    
    trap cleanup EXIT
}

# Main deployment function
main() {
    check_system_resources
    
    if [ "$ENVIRONMENT" != "test" ]; then
        install_dependencies
        
        if [ "$ENVIRONMENT" = "docker" ]; then
            setup_docker
        fi
        
        create_directories
        optimize_system
        monitor_resources
    fi
    
    run_deployment
    
    print_success "Optimized deployment completed!"
    
    echo ""
    echo "🎯 Optimization Summary:"
    echo "   Memory Mode: $MEMORY_MODE"
    echo "   RAM Available: ${AVAILABLE_RAM_GB}GB"
    echo "   CPU Cores: $CPU_COUNT"
    echo ""
    echo "📋 Recommended Commands for Your 4GB System:"
    echo ""
    
    if [ "$MEMORY_MODE" = "ultra_low" ]; then
        echo "   # Traditional scraping (no LLM)"
        echo "   python3 main_optimized.py --url https://site.com --output ./output"
    else
        echo "   # Auto-optimized LLM scraping"
        echo "   python3 main_optimized.py --url https://site.com --output ./output --llm"
        echo ""
        echo "   # Memory-efficient OpenRouter"
        echo "   export OPENROUTER_API_KEY='your-key'"
        echo "   python3 main_optimized.py --url https://site.com --output ./output --provider openrouter"
        echo ""
        echo "   # Local Ollama with small model"
        echo "   python3 main_optimized.py --url https://site.com --output ./output --provider ollama --model phi3:mini"
    fi
    
    echo ""
    echo "   # Monitor system resources"
    echo "   python3 main_optimized.py --status"
    echo ""
    echo "   # Sequential processing (saves memory)"
    echo "   python3 main_optimized.py --urls-file input/sites.txt --output ./output --llm --sequential"
}

# Run main function
main "$@"

