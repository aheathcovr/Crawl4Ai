# Healthcare Facility Scraper - Docker Configuration
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Install system dependencies required for Playwright
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    ca-certificates \
    procps \
    xvfb \
    # Dependencies for Playwright browsers
    libnss3 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxkbcommon0 \
    libgtk-3-0 \
    libgbm1 \
    libasound2 \
    # Additional dependencies
    libgstreamer1.0-0 \
    libgtk-4-1 \
    libgraphene-1.0-0 \
    libatomic1 \
    libxslt1.1 \
    libwoff2dec1.0.2 \
    libevent-2.1-7 \
    libenchant-2-2 \
    libsecret-1-0 \
    libhyphen0 \
    libharfbuzz-icu0 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers
RUN playwright install chromium
RUN playwright install-deps chromium

# Copy application code
COPY . .

# Create output directory
RUN mkdir -p /app/output

# Make main script executable
RUN chmod +x main.py

# Create non-root user for security
RUN useradd -m -u 1000 scraper && chown -R scraper:scraper /app
USER scraper

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "main.py", "--help"]

