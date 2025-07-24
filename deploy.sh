#!/bin/bash

# Healthcare Facility Scraper - Digital Ocean Deployment Script
# Usage: ./deploy.sh [environment]

set -e

# Configuration
PROJECT_NAME="healthcare-scraper"
DOCKER_IMAGE="healthcare-scraper:latest"
ENVIRONMENT=${1:-production}

echo "🚀 Deploying Healthcare Facility Scraper to Digital Ocean"
echo "Environment: $ENVIRONMENT"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    echo "✅ Docker installed. Please log out and back in, then run this script again."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Installing..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "✅ Docker Compose installed."
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p output logs input

# Create sample input file if it doesn't exist
if [ ! -f "input/sites.txt" ]; then
    echo "📝 Creating sample sites.txt file..."
    cat > input/sites.txt << EOF
# Healthcare facility websites to scrape
# One URL per line, comments start with #

https://lcca.com
https://genesishcc.com
https://brookdaleliving.com
https://sunriseseniorliving.com
EOF
fi

# Build the Docker image
echo "🔨 Building Docker image..."
docker build -t $DOCKER_IMAGE .

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose down || true

# Start the services
echo "🚀 Starting services..."
if [ "$ENVIRONMENT" = "development" ]; then
    # Development mode - interactive shell
    docker-compose run --rm healthcare-scraper bash
elif [ "$ENVIRONMENT" = "web" ]; then
    # Web mode - with nginx server
    docker-compose --profile web up -d
    echo "✅ Services started with web interface at http://localhost:8080"
else
    # Production mode - run scraper
    echo "🏥 Running healthcare facility scraper..."
    
    # Check if sites.txt exists
    if [ -f "input/sites.txt" ]; then
        echo "📋 Using sites from input/sites.txt"
        docker-compose run --rm healthcare-scraper python main.py \
            --urls-file /app/input/sites.txt \
            --output /app/output \
            --formats json csv excel \
            --log-level INFO
    else
        echo "📋 Running with default LCCA site"
        docker-compose run --rm healthcare-scraper python main.py \
            --url https://lcca.com \
            --output /app/output \
            --formats json csv excel \
            --log-level INFO
    fi
fi

# Show results
if [ "$ENVIRONMENT" != "development" ] && [ "$ENVIRONMENT" != "web" ]; then
    echo ""
    echo "✅ Scraping completed!"
    echo "📊 Results available in:"
    echo "   - JSON files: ./output/*.json"
    echo "   - CSV files: ./output/*.csv"
    echo "   - Excel files: ./output/*.xlsx"
    echo "   - Logs: ./logs/"
    
    # Show summary of results
    if ls output/*.json 1> /dev/null 2>&1; then
        echo ""
        echo "📈 Results Summary:"
        for file in output/*.json; do
            if [ -f "$file" ]; then
                count=$(jq length "$file" 2>/dev/null || echo "0")
                echo "   - $(basename "$file"): $count facilities"
            fi
        done
    fi
fi

echo ""
echo "🎉 Deployment complete!"

# Cleanup function
cleanup() {
    echo "🧹 Cleaning up..."
    docker-compose down
}

# Set trap for cleanup on script exit
trap cleanup EXIT

