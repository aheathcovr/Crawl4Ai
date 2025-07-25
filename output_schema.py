"""
Comprehensive Output Schema for Healthcare Facility Scraper
Defines the structure and format of all output data
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json


@dataclass
class FacilityOutputSchema:
    """Complete schema for individual facility output"""
    
    # Basic Information
    name: str                           # Official facility name
    facility_type: str                  # Type (Skilled Nursing, Assisted Living, etc.)
    source_url: str                     # URL where data was found
    source_corporation: str             # Parent corporation name
    
    # Location Information
    address: str                        # Street address
    city: str                          # City name
    state: str                         # State abbreviation (CA, NY, etc.)
    zip_code: str                      # ZIP/postal code
    county: str                        # County name (if available)
    region: str                        # Geographic region (if available)
    
    # Contact Information
    phone: str                         # Primary phone number
    fax: str                           # Fax number (if available)
    email: str                         # Contact email
    website: str                       # Facility website URL
    
    # Administrative Information
    administrator: str                  # Administrator/director name
    administrator_email: str           # Administrator email
    administrator_phone: str           # Administrator direct phone
    medical_director: str              # Medical director name
    
    # Capacity and Licensing
    beds: str                          # Number of beds/capacity
    license_number: str                # State license number
    medicare_provider_id: str          # Medicare provider ID
    medicaid_provider_id: str          # Medicaid provider ID
    npi_number: str                    # National Provider Identifier
    
    # Services and Specialties
    services_offered: List[str]        # List of services offered
    specialties: List[str]             # Medical specialties
    amenities: List[str]               # Facility amenities
    care_levels: List[str]             # Levels of care provided
    
    # Quality and Accreditation
    accreditation: str                 # Accreditation information
    quality_rating: str                # Quality rating (if available)
    cms_rating: str                    # CMS star rating
    certifications: List[str]          # Professional certifications
    
    # Operational Information
    visiting_hours: str                # Visiting hours
    admission_requirements: str        # Admission requirements
    payment_options: List[str]         # Accepted payment methods
    insurance_accepted: List[str]      # Insurance plans accepted
    
    # Additional Details
    description: str                   # Facility description
    year_established: str              # Year established
    ownership_type: str                # Ownership type (private, public, non-profit)
    parent_organization: str           # Parent organization
    
    # Metadata
    scraping_date: str                 # When data was scraped
    scraping_method: str               # Method used (traditional, LLM)
    data_quality_score: float          # Quality score (0-1)
    last_verified: str                 # Last verification date
    
    # Corporate Chain Information
    corporation_priority: int          # Priority level of parent corporation
    corporation_facility_types: List[str]  # Types operated by corporation
    estimated_chain_size: str          # Estimated size of parent chain


@dataclass
class ScrapingResultSchema:
    """Schema for scraping operation results"""
    
    # Operation Information
    corporation_name: str              # Corporation that was scraped
    url_scraped: str                   # Primary URL scraped
    secondary_urls: List[str]          # Additional URLs scraped
    
    # Results Summary
    facilities_found: int              # Number of facilities found
    scraping_success: bool             # Whether scraping succeeded
    scraping_date: str                 # Date/time of scraping
    processing_time_minutes: float     # Time taken to process
    
    # Technical Details
    scraping_method: str               # Method used (traditional/LLM)
    llm_provider: str                  # LLM provider used (if applicable)
    llm_model: str                     # LLM model used (if applicable)
    
    # Quality Metrics
    data_quality_score: float          # Overall data quality (0-1)
    duplicate_facilities_removed: int  # Number of duplicates removed
    validation_errors: int             # Number of validation errors
    
    # Output Files
    output_files: List[str]            # List of generated output files
    consolidated_file: str             # Main consolidated output file
    
    # Error Information
    error_message: str                 # Error message (if failed)
    warnings: List[str]                # List of warnings
    
    # Resource Usage
    memory_used_mb: float              # Memory usage during scraping
    cpu_time_seconds: float            # CPU time used
    network_requests: int              # Number of network requests made


@dataclass
class BatchProcessingSchema:
    """Schema for batch processing results"""
    
    # Batch Information
    batch_id: str                      # Unique batch identifier
    total_chains_processed: int        # Number of chains processed
    total_batches: int                 # Number of batches
    batch_size: int                    # Size of each batch
    
    # Results Summary
    total_facilities_found: int        # Total facilities across all chains
    successful_chains: int             # Number of successful chains
    failed_chains: int                 # Number of failed chains
    
    # Processing Details
    processing_start_time: str         # Batch processing start time
    processing_end_time: str           # Batch processing end time
    total_processing_time_hours: float # Total time taken
    
    # Quality Metrics
    average_data_quality: float        # Average data quality score
    total_duplicates_removed: int      # Total duplicates removed
    
    # Output Information
    consolidated_json_file: str        # Main JSON output file
    consolidated_csv_file: str         # Main CSV output file
    summary_report_file: str           # Summary report file
    individual_chain_files: List[str]  # Individual chain output files
    
    # Resource Usage
    peak_memory_usage_gb: float        # Peak memory usage
    total_network_requests: int        # Total network requests
    
    # Chain-Level Results
    chain_results: List[ScrapingResultSchema]  # Individual chain results


# Output File Formats and Structures

OUTPUT_FORMATS = {
    "json": {
        "description": "JSON format with full schema",
        "file_extension": ".json",
        "structure": "Array of facility objects with complete schema",
        "use_case": "API integration, data processing, full feature access"
    },
    "csv": {
        "description": "CSV format for spreadsheet analysis",
        "file_extension": ".csv", 
        "structure": "Flattened rows with all fields as columns",
        "use_case": "Excel analysis, database import, reporting"
    },
    "excel": {
        "description": "Excel workbook with multiple sheets",
        "file_extension": ".xlsx",
        "structure": "Multiple sheets: Facilities, Summary, Corporations, Quality",
        "use_case": "Business reporting, detailed analysis"
    },
    "parquet": {
        "description": "Parquet format for big data processing",
        "file_extension": ".parquet",
        "structure": "Columnar format optimized for analytics",
        "use_case": "Data science, large-scale analytics"
    }
}

# File Naming Conventions

FILE_NAMING_PATTERNS = {
    "individual_facility": "{corporation_name}_{timestamp}_facilities.{format}",
    "consolidated": "all_facilities_{timestamp}.{format}",
    "batch_summary": "batch_processing_summary_{batch_id}.{format}",
    "corporation_summary": "corporation_summary_{timestamp}.{format}",
    "quality_report": "data_quality_report_{timestamp}.{format}"
}

# Data Quality Scoring

DATA_QUALITY_CRITERIA = {
    "completeness": {
        "weight": 0.4,
        "required_fields": ["name", "facility_type", "address", "city", "state"],
        "optional_fields": ["phone", "website", "services_offered"],
        "scoring": "Percentage of non-empty required and optional fields"
    },
    "accuracy": {
        "weight": 0.3,
        "validation_rules": [
            "Phone number format validation",
            "Email format validation", 
            "URL format validation",
            "State abbreviation validation",
            "ZIP code format validation"
        ],
        "scoring": "Percentage of fields passing validation"
    },
    "consistency": {
        "weight": 0.2,
        "checks": [
            "Facility type standardization",
            "Address format consistency",
            "Service name standardization"
        ],
        "scoring": "Consistency score across standardized fields"
    },
    "freshness": {
        "weight": 0.1,
        "criteria": "How recently the data was scraped",
        "scoring": "Time-based decay from scraping date"
    }
}

# Expected Output Sizes

EXPECTED_OUTPUT_SIZES = {
    "small_corporation": {
        "facilities": "10-50",
        "json_size_mb": "0.5-2",
        "csv_size_mb": "0.2-1",
        "processing_time_minutes": "2-10"
    },
    "medium_corporation": {
        "facilities": "50-200", 
        "json_size_mb": "2-10",
        "csv_size_mb": "1-5",
        "processing_time_minutes": "10-30"
    },
    "large_corporation": {
        "facilities": "200-1000",
        "json_size_mb": "10-50",
        "csv_size_mb": "5-25", 
        "processing_time_minutes": "30-120"
    },
    "batch_processing": {
        "facilities": "1000-10000+",
        "json_size_mb": "50-500+",
        "csv_size_mb": "25-250+",
        "processing_time_hours": "2-24+"
    }
}


def generate_schema_documentation() -> str:
    """Generate comprehensive schema documentation"""
    
    doc = """
# Healthcare Facility Scraper - Output Schema Documentation

## Overview
This document describes the complete output schema for the Healthcare Facility Scraper, including data structures, file formats, and quality metrics.

## Individual Facility Schema

Each facility record contains the following fields:

### Basic Information
- **name**: Official facility name
- **facility_type**: Type of healthcare facility
- **source_url**: URL where data was found
- **source_corporation**: Parent corporation name

### Location Information  
- **address**: Complete street address
- **city**: City name
- **state**: State abbreviation
- **zip_code**: ZIP/postal code
- **county**: County name (when available)
- **region**: Geographic region

### Contact Information
- **phone**: Primary phone number
- **fax**: Fax number
- **email**: Contact email address
- **website**: Facility website URL

### Administrative Information
- **administrator**: Administrator/director name
- **administrator_email**: Administrator email
- **administrator_phone**: Administrator direct phone
- **medical_director**: Medical director name

### Capacity and Licensing
- **beds**: Number of beds/capacity
- **license_number**: State license number
- **medicare_provider_id**: Medicare provider ID
- **medicaid_provider_id**: Medicaid provider ID
- **npi_number**: National Provider Identifier

### Services and Care
- **services_offered**: Array of services offered
- **specialties**: Array of medical specialties
- **amenities**: Array of facility amenities
- **care_levels**: Array of care levels provided

### Quality Information
- **accreditation**: Accreditation details
- **quality_rating**: Quality rating
- **cms_rating**: CMS star rating
- **certifications**: Array of certifications

### Operational Details
- **visiting_hours**: Visiting hours information
- **admission_requirements**: Admission requirements
- **payment_options**: Array of payment methods
- **insurance_accepted**: Array of insurance plans

### Additional Information
- **description**: Facility description
- **year_established**: Year established
- **ownership_type**: Ownership type
- **parent_organization**: Parent organization

### Metadata
- **scraping_date**: When data was scraped
- **scraping_method**: Method used (traditional/LLM)
- **data_quality_score**: Quality score (0-1)
- **last_verified**: Last verification date

## Output File Formats

### JSON Format
- Complete schema with all fields
- Nested arrays for multi-value fields
- Ideal for API integration and data processing

### CSV Format  
- Flattened structure with pipe-separated arrays
- Compatible with Excel and database imports
- Optimized for spreadsheet analysis

### Excel Format
- Multiple worksheets:
  - Facilities: Main facility data
  - Summary: Corporation-level summary
  - Quality: Data quality metrics
  - Metadata: Scraping operation details

## File Naming Conventions

- Individual corporation: `{corporation_name}_{timestamp}_facilities.{format}`
- Consolidated results: `all_facilities_{timestamp}.{format}`
- Batch summary: `batch_processing_summary_{batch_id}.{format}`

## Data Quality Scoring

Quality scores are calculated based on:
- **Completeness** (40%): Required and optional field coverage
- **Accuracy** (30%): Format validation and data correctness
- **Consistency** (20%): Standardization across records
- **Freshness** (10%): How recently data was scraped

## Expected Output Sizes

- **Small corporation** (10-50 facilities): 0.5-2MB JSON, 2-10 minutes
- **Medium corporation** (50-200 facilities): 2-10MB JSON, 10-30 minutes  
- **Large corporation** (200-1000 facilities): 10-50MB JSON, 30-120 minutes
- **Batch processing** (1000+ facilities): 50MB+ JSON, 2-24+ hours

## Download and Access Methods

Results can be accessed via:
1. Direct file download from output directory
2. SCP/SFTP transfer from server
3. Automated sync to cloud storage
4. API endpoints (if web interface deployed)
5. Email delivery for completed batches
"""
    
    return doc


def create_sample_output() -> Dict[str, Any]:
    """Create a sample output showing the complete schema"""
    
    sample_facility = {
        "name": "Sunrise Senior Living of Springfield",
        "facility_type": "Assisted Living",
        "source_url": "https://sunriseseniorliving.com/communities/springfield",
        "source_corporation": "Sunrise Senior Living",
        
        "address": "123 Main Street",
        "city": "Springfield",
        "state": "IL",
        "zip_code": "62701",
        "county": "Sangamon County",
        "region": "Central Illinois",
        
        "phone": "(217) 555-0123",
        "fax": "(217) 555-0124",
        "email": "info@springfield.sunriseseniorliving.com",
        "website": "https://sunriseseniorliving.com/communities/springfield",
        
        "administrator": "Jane Smith",
        "administrator_email": "jane.smith@sunriseseniorliving.com",
        "administrator_phone": "(217) 555-0125",
        "medical_director": "Dr. Robert Johnson",
        
        "beds": "120",
        "license_number": "IL-AL-2024-001",
        "medicare_provider_id": "145678",
        "medicaid_provider_id": "IL-MC-789",
        "npi_number": "1234567890",
        
        "services_offered": [
            "Assisted Living",
            "Memory Care",
            "Medication Management",
            "Physical Therapy",
            "Occupational Therapy",
            "Social Activities",
            "Transportation Services"
        ],
        "specialties": [
            "Alzheimer's Care",
            "Dementia Care",
            "Post-Surgical Recovery"
        ],
        "amenities": [
            "Private Dining Room",
            "Fitness Center",
            "Library",
            "Garden Courtyard",
            "Beauty Salon",
            "Chapel"
        ],
        "care_levels": [
            "Independent Living",
            "Assisted Living",
            "Memory Care"
        ],
        
        "accreditation": "CARF Accredited",
        "quality_rating": "4.5/5 stars",
        "cms_rating": "4 stars",
        "certifications": [
            "CARF Accreditation",
            "Alzheimer's Association Certified"
        ],
        
        "visiting_hours": "Daily 8:00 AM - 8:00 PM",
        "admission_requirements": "Assessment required, minimum age 55",
        "payment_options": [
            "Private Pay",
            "Long-term Care Insurance",
            "Veterans Benefits"
        ],
        "insurance_accepted": [
            "Medicare",
            "Medicaid",
            "Private Insurance"
        ],
        
        "description": "Sunrise Senior Living of Springfield offers exceptional assisted living and memory care services in a warm, homelike environment.",
        "year_established": "2018",
        "ownership_type": "Private",
        "parent_organization": "Sunrise Senior Living Inc.",
        
        "scraping_date": "2024-01-15T14:30:00Z",
        "scraping_method": "LLM Enhanced",
        "data_quality_score": 0.92,
        "last_verified": "2024-01-15",
        
        "corporation_priority": 1,
        "corporation_facility_types": ["Assisted Living", "Memory Care", "Independent Living"],
        "estimated_chain_size": "320+ facilities"
    }
    
    return sample_facility

