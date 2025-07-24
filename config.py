"""
Configuration settings for Healthcare Facility Scraper
"""

import os
from typing import List, Dict, Set

# Facility Types
FACILITY_TYPES = {
    'skilled_nursing': [
        'skilled nursing', 'nursing home', 'snf', 'skilled nursing facility',
        'rehabilitation center', 'rehab center', 'post-acute care'
    ],
    'assisted_living': [
        'assisted living', 'assisted care', 'senior living', 'independent living',
        'residential care', 'adult care home', 'personal care'
    ],
    'memory_care': [
        'memory care', 'alzheimer', 'dementia care', 'cognitive care',
        'memory support', 'alzheimer\'s care', 'dementia support'
    ],
    'continuing_care': [
        'ccrc', 'continuing care retirement community', 'life plan community',
        'retirement community', 'senior community'
    ]
}

# Common URL patterns for facility listings
FACILITY_URL_PATTERNS = [
    'locations', 'facilities', 'centers', 'communities', 'properties',
    'find-a-location', 'our-locations', 'care-centers', 'nursing-homes',
    'assisted-living', 'memory-care', 'senior-living', 'communities',
    'directory', 'find-care', 'location-finder', 'facility-locator'
]

# Keywords that indicate facility information pages
FACILITY_KEYWORDS = [
    'address', 'phone', 'contact', 'location', 'directions', 'services',
    'amenities', 'staff', 'administrator', 'director', 'beds', 'capacity',
    'license', 'certification', 'medicare', 'medicaid', 'insurance'
]

# Data fields to extract for each facility
FACILITY_DATA_FIELDS = {
    'basic_info': [
        'name', 'facility_type', 'address', 'city', 'state', 'zip_code',
        'phone', 'fax', 'email', 'website'
    ],
    'operational': [
        'administrator', 'director', 'medical_director', 'beds', 'capacity',
        'license_number', 'medicare_provider_id', 'medicaid_certified'
    ],
    'services': [
        'services_offered', 'specialties', 'amenities', 'care_levels'
    ],
    'additional': [
        'description', 'hours', 'visiting_hours', 'parking_info'
    ]
}

# Crawling configuration
CRAWL_CONFIG = {
    'max_depth': 4,
    'max_concurrent': 8,
    'delay_between_requests': 1.0,
    'timeout': 30,
    'max_pages_per_site': 500,
    'respect_robots_txt': True,
    'user_agent': 'HealthcareFacilityScraper/1.0 (+https://example.com/bot)'
}

# Browser configuration
BROWSER_CONFIG = {
    'headless': True,
    'viewport': {'width': 1920, 'height': 1080},
    'wait_for_load_state': 'networkidle',
    'javascript_enabled': True
}

# Output configuration
OUTPUT_CONFIG = {
    'formats': ['json', 'csv', 'excel'],
    'include_raw_html': False,
    'include_screenshots': False,
    'output_directory': './output'
}

# Patterns to exclude from crawling
EXCLUDE_PATTERNS = [
    'careers', 'jobs', 'employment', 'news', 'blog', 'events',
    'calendar', 'resources', 'forms', 'documents', 'pdf',
    'privacy', 'terms', 'legal', 'sitemap', 'search'
]

# Common selectors for facility information
CSS_SELECTORS = {
    'facility_cards': [
        '.facility-card', '.location-card', '.community-card',
        '.center-card', '.property-card', '[class*="facility"]',
        '[class*="location"]', '[class*="community"]'
    ],
    'facility_links': [
        'a[href*="location"]', 'a[href*="facility"]', 'a[href*="community"]',
        'a[href*="center"]', 'a[href*="property"]'
    ],
    'contact_info': [
        '.contact', '.address', '.phone', '.email',
        '[class*="contact"]', '[class*="address"]', '[class*="phone"]'
    ],
    'facility_details': [
        '.facility-info', '.location-details', '.community-info',
        '.property-details', '[class*="details"]', '[class*="info"]'
    ]
}

# Regular expressions for data extraction
REGEX_PATTERNS = {
    'phone': r'(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})',
    'address': r'(\d+\s+[A-Za-z0-9\s,.-]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Circle|Cir|Court|Ct|Place|Pl))',
    'zip_code': r'(\d{5}(?:-\d{4})?)',
    'email': r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
    'beds': r'(\d+)\s*(?:beds?|bed)',
    'license': r'(?:license|lic\.?)\s*#?\s*([A-Z0-9-]+)',
    'medicare_id': r'(?:medicare|provider)\s*(?:id|number|#)\s*:?\s*([A-Z0-9-]+)'
}

# State abbreviations mapping
STATE_ABBREVIATIONS = {
    'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR',
    'california': 'CA', 'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE',
    'florida': 'FL', 'georgia': 'GA', 'hawaii': 'HI', 'idaho': 'ID',
    'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA', 'kansas': 'KS',
    'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD',
    'massachusetts': 'MA', 'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS',
    'missouri': 'MO', 'montana': 'MT', 'nebraska': 'NE', 'nevada': 'NV',
    'new hampshire': 'NH', 'new jersey': 'NJ', 'new mexico': 'NM', 'new york': 'NY',
    'north carolina': 'NC', 'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK',
    'oregon': 'OR', 'pennsylvania': 'PA', 'rhode island': 'RI', 'south carolina': 'SC',
    'south dakota': 'SD', 'tennessee': 'TN', 'texas': 'TX', 'utah': 'UT',
    'vermont': 'VT', 'virginia': 'VA', 'washington': 'WA', 'west virginia': 'WV',
    'wisconsin': 'WI', 'wyoming': 'WY'
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'healthcare_scraper.log'
}

