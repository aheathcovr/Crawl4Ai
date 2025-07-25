"""
Optimized configuration for Digital Ocean 2 vCPU / 4GB RAM droplet
Balances performance with resource constraints
"""

import os
import psutil
from typing import Dict, Any

# System resource detection
AVAILABLE_MEMORY_GB = psutil.virtual_memory().total / (1024**3)
CPU_COUNT = psutil.cpu_count()

# Optimized settings for 2 vCPU / 4GB RAM droplet
OPTIMIZED_CONFIG = {
    # Crawler settings optimized for 4GB RAM
    'crawler': {
        'max_concurrent_pages': 3,  # Reduced from 5 to prevent memory issues
        'page_timeout': 45,         # Slightly longer timeout for stability
        'max_retries': 2,           # Reduced retries to save resources
        'delay_between_requests': 2, # Longer delay to be respectful
        'browser_args': [
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--disable-gpu',
            '--disable-extensions',
            '--disable-plugins',
            '--disable-images',      # Save bandwidth and memory
            '--disable-javascript',  # For basic content extraction
            '--memory-pressure-off', # Prevent memory pressure issues
        ]
    },
    
    # LLM provider settings optimized for 4GB RAM
    'llm': {
        'max_tokens': 2000,         # Reduced from 4000 to save memory
        'temperature': 0.1,         # Low temperature for consistent extraction
        'timeout': 60,              # Reasonable timeout
        'chunk_size': 4000,         # Smaller chunks for processing
        'max_concurrent_requests': 2, # Limit concurrent LLM calls
    },
    
    # Model recommendations for 4GB RAM
    'recommended_models': {
        'openrouter_free': 'meta-llama/llama-3.1-8b-instruct:free',
        'openrouter_fast': 'google/gemma-2-9b-it:free',
        'ollama_fast': 'llama3.2:3b',      # 2GB RAM usage
        'ollama_balanced': 'phi3:mini',     # 1GB RAM usage
        'ollama_quality': 'llama3.1:8b',   # 4GB RAM usage (max for your system)
    },
    
    # Memory management
    'memory': {
        'max_memory_usage_percent': 80,  # Leave 20% free for system
        'garbage_collection_interval': 10, # Clean up every 10 pages
        'clear_browser_cache': True,     # Clear cache between sessions
    },
    
    # Disk usage optimization
    'storage': {
        'max_log_size_mb': 100,         # Limit log file size
        'compress_results': True,        # Compress large result files
        'cleanup_temp_files': True,      # Clean up temporary files
        'max_cache_size_mb': 500,       # Limit browser cache
    },
    
    # Network optimization for 4TB transfer limit
    'network': {
        'max_page_size_mb': 10,         # Skip very large pages
        'connection_pool_size': 5,       # Limit connection pool
        'user_agent': 'Mozilla/5.0 (compatible; HealthcareScraper/1.0)',
        'respect_robots_txt': True,      # Be respectful
    }
}

# Dynamic configuration based on available resources
def get_dynamic_config() -> Dict[str, Any]:
    """Get configuration dynamically based on available resources"""
    
    config = OPTIMIZED_CONFIG.copy()
    
    # Adjust based on available memory
    if AVAILABLE_MEMORY_GB < 3:
        # Very conservative settings for low memory
        config['crawler']['max_concurrent_pages'] = 2
        config['llm']['max_tokens'] = 1500
        config['llm']['max_concurrent_requests'] = 1
        config['recommended_models']['ollama_quality'] = 'phi3:mini'
    
    elif AVAILABLE_MEMORY_GB >= 6:
        # More aggressive settings if more memory available
        config['crawler']['max_concurrent_pages'] = 4
        config['llm']['max_tokens'] = 3000
        config['llm']['max_concurrent_requests'] = 3
    
    # Adjust based on CPU count
    if CPU_COUNT == 1:
        config['crawler']['max_concurrent_pages'] = 2
        config['llm']['max_concurrent_requests'] = 1
    elif CPU_COUNT >= 4:
        config['crawler']['max_concurrent_pages'] = min(5, CPU_COUNT)
        config['llm']['max_concurrent_requests'] = min(4, CPU_COUNT)
    
    return config

# Ollama model configurations for different resource levels
OLLAMA_MODELS_BY_MEMORY = {
    # Models that work well with 4GB RAM
    'low_memory': {
        'model': 'phi3:mini',
        'ram_usage_gb': 1.0,
        'speed': 'very_fast',
        'quality': 'good',
        'description': 'Fastest option, minimal memory usage'
    },
    'balanced': {
        'model': 'llama3.2:3b',
        'ram_usage_gb': 2.0,
        'speed': 'fast',
        'quality': 'very_good',
        'description': 'Best balance for 4GB systems'
    },
    'high_quality': {
        'model': 'llama3.1:8b',
        'ram_usage_gb': 4.0,
        'speed': 'medium',
        'quality': 'excellent',
        'description': 'Maximum quality for 4GB systems'
    },
    # Models that require more than 4GB (not recommended for your droplet)
    'premium': {
        'model': 'llama3.1:70b',
        'ram_usage_gb': 40.0,
        'speed': 'slow',
        'quality': 'exceptional',
        'description': 'Requires 40GB+ RAM - not suitable for your droplet'
    }
}

def get_recommended_ollama_model() -> str:
    """Get the best Ollama model for current system"""
    
    if AVAILABLE_MEMORY_GB < 2:
        return OLLAMA_MODELS_BY_MEMORY['low_memory']['model']
    elif AVAILABLE_MEMORY_GB < 3:
        return OLLAMA_MODELS_BY_MEMORY['balanced']['model']
    elif AVAILABLE_MEMORY_GB >= 4:
        return OLLAMA_MODELS_BY_MEMORY['high_quality']['model']
    else:
        return OLLAMA_MODELS_BY_MEMORY['balanced']['model']

# Docker resource limits for your droplet
DOCKER_RESOURCE_LIMITS = {
    'memory': '3.5g',      # Leave 0.5GB for system
    'cpus': '1.8',         # Leave 0.2 CPU for system
    'shm_size': '512m',    # Shared memory for browser
    'ulimits': {
        'nofile': {'soft': 1024, 'hard': 2048}
    }
}

# Monitoring thresholds
MONITORING_THRESHOLDS = {
    'memory_warning_percent': 75,
    'memory_critical_percent': 90,
    'cpu_warning_percent': 80,
    'cpu_critical_percent': 95,
    'disk_warning_percent': 80,
    'disk_critical_percent': 90
}

def check_system_resources() -> Dict[str, Any]:
    """Check current system resource usage"""
    
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)
    disk = psutil.disk_usage('/')
    
    return {
        'memory': {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_percent': memory.percent,
            'status': 'critical' if memory.percent > MONITORING_THRESHOLDS['memory_critical_percent'] 
                     else 'warning' if memory.percent > MONITORING_THRESHOLDS['memory_warning_percent']
                     else 'ok'
        },
        'cpu': {
            'count': CPU_COUNT,
            'used_percent': cpu_percent,
            'status': 'critical' if cpu_percent > MONITORING_THRESHOLDS['cpu_critical_percent']
                     else 'warning' if cpu_percent > MONITORING_THRESHOLDS['cpu_warning_percent'] 
                     else 'ok'
        },
        'disk': {
            'total_gb': disk.total / (1024**3),
            'free_gb': disk.free / (1024**3),
            'used_percent': (disk.used / disk.total) * 100,
            'status': 'critical' if (disk.used / disk.total) * 100 > MONITORING_THRESHOLDS['disk_critical_percent']
                     else 'warning' if (disk.used / disk.total) * 100 > MONITORING_THRESHOLDS['disk_warning_percent']
                     else 'ok'
        }
    }

def print_system_status():
    """Print current system status"""
    
    status = check_system_resources()
    
    print("🖥️  System Resource Status:")
    print(f"   Memory: {status['memory']['used_percent']:.1f}% used ({status['memory']['available_gb']:.1f}GB free) - {status['memory']['status'].upper()}")
    print(f"   CPU: {status['cpu']['used_percent']:.1f}% used ({status['cpu']['count']} cores) - {status['cpu']['status'].upper()}")
    print(f"   Disk: {status['disk']['used_percent']:.1f}% used ({status['disk']['free_gb']:.1f}GB free) - {status['disk']['status'].upper()}")
    
    # Recommendations
    if status['memory']['status'] != 'ok':
        print("⚠️  Memory usage high - consider using smaller models or reducing concurrency")
    
    if status['cpu']['status'] != 'ok':
        print("⚠️  CPU usage high - consider reducing concurrent operations")
    
    if status['disk']['status'] != 'ok':
        print("⚠️  Disk usage high - consider cleaning up old results or logs")

# Performance optimization tips
OPTIMIZATION_TIPS = {
    'memory': [
        "Use phi3:mini or llama3.2:3b models for Ollama",
        "Reduce max_concurrent_pages to 2-3",
        "Enable browser cache clearing",
        "Use OpenRouter free models instead of local models"
    ],
    'cpu': [
        "Reduce concurrent LLM requests to 1-2",
        "Increase delay between requests",
        "Use faster models (phi3:mini)",
        "Process sites sequentially instead of in parallel"
    ],
    'disk': [
        "Enable result compression",
        "Clean up old log files regularly",
        "Limit browser cache size",
        "Use streaming for large result sets"
    ],
    'network': [
        "Your 4TB transfer limit is very generous",
        "No network optimizations needed",
        "Can scrape hundreds of sites per month"
    ]
}

def get_optimization_recommendations() -> Dict[str, str]:
    """Get optimization recommendations based on current system"""
    
    status = check_system_resources()
    recommendations = {}
    
    if status['memory']['used_percent'] > 70:
        recommendations['memory'] = "Consider using lighter models or reducing concurrency"
    
    if status['cpu']['used_percent'] > 70:
        recommendations['cpu'] = "Reduce concurrent operations for better stability"
    
    if status['disk']['used_percent'] > 70:
        recommendations['disk'] = "Clean up old files or enable compression"
    
    return recommendations

