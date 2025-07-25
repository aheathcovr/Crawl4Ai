"""
Centralized configuration for Crawl4AI and scraping settings
All magic constants and tunable parameters in one place
"""

from typing import Dict, Any

# Crawl4AI Browser Configuration
CRAWLER_CONFIG = {
    "headless": True,
    "verbose": False,
    "cache_dir": ".crawl_cache",
    "max_cache_age_hours": 24,
    "requests_per_minute": 30,
    "concurrent_tabs": 3,
    "session_cache": True,  # Reuse browser instances
    "viewport": {
        "width": 1920,
        "height": 1080
    }
}

# LLM Configuration Defaults
LLM_CONFIG = {
    "temperature": 0,  # Deterministic for extraction
    "top_p": 0.8,
    "max_tokens": 4000,
    "timeout": 30
}

# Extraction Configuration
EXTRACTION_CONFIG = {
    "max_targets_corporate": 5,  # Max pages to extract for corporate chains
    "max_targets_single": 2,     # Max pages for single facilities
    "chunk_token_threshold": 600,
    "chunk_overlap_rate": 0.1,
    "html_chunk_tags": ["div", "section", "article", "p", "li", "tr"],
    "regex_chunk_patterns": [r'\n\n+', r'\. (?=[A-Z])', r'</div>\s*<div']
}

# Scraping Delays and Limits
SCRAPING_LIMITS = {
    "delay_between_requests": 0,  # Let Crawl4AI handle rate limiting
    "batch_size": 5,
    "max_facilities_per_page": 100,
    "max_pages_per_site": 50,
    "request_timeout": 30
}

# Validation Configuration
VALIDATION_CONFIG = {
    "enable_deduplication": True,
    "dedup_threshold": 0.85,  # Similarity threshold for duplicates
    "cache_validation_results": True,
    "max_geocoding_requests": 100,  # Per session limit
    "validation_batch_size": 20
}

# URL Scoring Weights (for smart crawling)
URL_SCORING = {
    "keyword_weights": {
        "locations": 10,
        "facilities": 10,
        "communities": 8,
        "directory": 8,
        "find": 5,
        "search": 5,
        "our": 3,
        "contact": -5,  # Negative weight for non-listing pages
        "about": -5,
        "careers": -10,
        "news": -10
    },
    "depth_penalty": 2,  # Penalty per URL depth level
    "max_depth": 3
}

# Schema Cache Configuration
SCHEMA_CACHE_CONFIG = {
    "enabled": True,
    "ttl_hours": 168,  # 1 week
    "max_schemas": 100
}

# Memory Management
MEMORY_CONFIG = {
    "gc_threshold": 100,  # Run GC every N facilities
    "clear_cookies_interval": 50,  # Clear browser cookies every N pages
    "max_facilities_in_memory": 1000,  # Stream to disk after this
    "use_streaming": True
}