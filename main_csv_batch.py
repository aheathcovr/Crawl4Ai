#!/usr/bin/env python3
"""
CSV-Driven Batch Processing for Healthcare Facility Scraper
Processes corporate chains from CSV file with priority management
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

from csv_processor import CorporateChainManager, BatchProcessor
from config_optimized import get_dynamic_config, print_system_status
from download_manager import DownloadManager
from output_schema import generate_schema_documentation


def setup_logging(log_level: str = 'INFO', log_file: str = None):
    """Setup logging for batch processing"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(log_file, maxBytes=100*1024*1024, backupCount=3)
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


async def process_csv_batch(csv_file: str, 
                           output_dir: str,
                           use_llm: bool = True,
                           priority_filter: int = 3,
                           batch_size: int = 5,
                           delay_between_batches: int = 60) -> Dict[str, Any]:
    """Process all chains from CSV file in batches"""
    
    logger = logging.getLogger(__name__)
    
    # Initialize CSV manager
    chain_manager = CorporateChainManager(csv_file)
    chains = chain_manager.load_chains()
    
    if not chains:
        logger.error(f"No chains loaded from {csv_file}")
        return {'success': False, 'error': 'No chains loaded'}
    
    # Print statistics
    stats = chain_manager.get_statistics()
    logger.info(f"📊 CSV Statistics:")
    logger.info(f"   Total chains: {stats['total_chains']}")
    logger.info(f"   File size: {stats['file_size_mb']}MB")
    logger.info(f"   Priority distribution: {stats['priority_distribution']}")
    logger.info(f"   Estimated total facilities: {stats['estimated_total_facilities']}")
    
    # Initialize batch processor
    batch_processor = BatchProcessor(chain_manager, output_dir)
    
    # Process all chains
    logger.info(f"🚀 Starting batch processing with priority filter <= {priority_filter}")
    
    result = await batch_processor.process_all_chains(
        use_llm=use_llm,
        batch_size=batch_size,
        priority_filter=priority_filter,
        delay_between_batches=delay_between_batches
    )
    
    return result


def print_csv_requirements():
    """Print CSV file requirements and recommendations"""
    
    print("""
📋 CSV File Requirements and Recommendations

## Required Columns:
- corporation_name: Official name of the healthcare corporation
- primary_url: Main website URL for the corporation

## Optional Columns:
- secondary_urls: Additional URLs separated by | (pipe)
- facility_types: Types operated (Skilled Nursing|Assisted Living|Memory Care)
- priority: Processing priority (1=highest, 5=lowest)
- notes: Additional notes about the corporation
- estimated_facilities: Estimated number of facilities (e.g., "200+", "100-150")
- last_updated: Last update date (YYYY-MM-DD format)

## Size Limits:
- Recommended: Up to 1,000 corporations
- Maximum: 10,000 corporations
- File size: Up to 50MB
- Processing time: 2-24+ hours for large batches

## Priority Levels:
- 1: Major national chains (highest priority)
- 2: Large regional chains
- 3: Medium regional chains
- 4: Small chains and REITs
- 5: Low priority or test entries

## Sample CSV Format:
```
corporation_name,primary_url,facility_types,priority,estimated_facilities
Life Care Centers,https://lcca.com,Skilled Nursing,1,200+
Sunrise Senior Living,https://sunriseseniorliving.com,Assisted Living|Memory Care,1,320+
Genesis HealthCare,https://genesishcc.com,Skilled Nursing|Rehabilitation,1,300+
```

## Processing Recommendations:
- Start with priority 1-2 chains for testing
- Use batch sizes of 5-10 for 4GB RAM systems
- Allow 60+ seconds between batches
- Monitor system resources during processing
""")


def print_output_information():
    """Print information about output schema and download methods"""
    
    print("""
📁 Output Schema and Download Information

## Output Files Generated:

### Individual Corporation Files:
- {corporation_name}_{timestamp}_facilities.json
- {corporation_name}_{timestamp}_facilities.csv
- Located in: ./batch_output/individual_chains/

### Consolidated Files:
- all_facilities_{timestamp}.json (master JSON file)
- all_facilities_{timestamp}.csv (master CSV file)
- Located in: ./batch_output/consolidated/

### Summary Files:
- batch_processing_summary.json (processing details)
- batch_processing_summary.csv (spreadsheet summary)
- Located in: ./batch_output/

## Individual Facility Schema:
Each facility record contains 30+ fields including:
- Basic info: name, facility_type, address, city, state, zip
- Contact: phone, email, website, administrator
- Services: services_offered[], specialties[], amenities[]
- Quality: accreditation, cms_rating, certifications[]
- Metadata: scraping_date, data_quality_score, source_corporation

## Download Methods:

### 1. Direct SCP Download:
```bash
# Download all results
scp -r root@YOUR_DROPLET_IP:/root/healthcare-scraper/batch_output ./local_results/

# Download specific files
scp root@YOUR_DROPLET_IP:/root/healthcare-scraper/batch_output/consolidated/*.json ./
```

### 2. Automated Download Manager:
```bash
# List available files
python download_manager.py --droplet-ip YOUR_IP list

# Download everything
python download_manager.py --droplet-ip YOUR_IP download-all --extract

# Sync results directory
python download_manager.py --droplet-ip YOUR_IP sync
```

### 3. Compressed Package Download:
```bash
# Create and download compressed package
python download_manager.py --droplet-ip YOUR_IP download-all
```

## Expected Output Sizes:
- Small batch (50 corporations): 5-25MB
- Medium batch (200 corporations): 25-100MB  
- Large batch (1000+ corporations): 100MB-1GB+
- Processing time: 2-24+ hours depending on size

## File Formats Available:
- JSON: Complete schema, ideal for APIs and data processing
- CSV: Flattened format, ideal for Excel and database import
- Excel: Multi-sheet workbook with summary and quality metrics
""")


def main():
    """Main CLI interface for CSV-driven batch processing"""
    
    parser = argparse.ArgumentParser(
        description='Healthcare Facility Scraper - CSV Batch Processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Process all priority 1-2 chains from CSV
  python main_csv_batch.py --csv corporate_chains.csv --output ./batch_results --priority 2

  # Process with traditional scraping (no LLM)
  python main_csv_batch.py --csv corporate_chains.csv --output ./batch_results --no-llm

  # Small batches for 4GB RAM system
  python main_csv_batch.py --csv corporate_chains.csv --output ./batch_results --batch-size 3 --delay 120

  # Process specific facility types only
  python main_csv_batch.py --csv corporate_chains.csv --output ./batch_results --facility-type "Assisted Living"

  # Show CSV requirements and output schema
  python main_csv_batch.py --help-csv
  python main_csv_batch.py --help-output

  # Download results from droplet
  python main_csv_batch.py --download --droplet-ip YOUR_IP
        """
    )
    
    # Input options
    parser.add_argument('--csv', help='CSV file with corporate chains')
    parser.add_argument('--output', '-o', default='./batch_output', help='Output directory')
    
    # Processing options
    parser.add_argument('--priority', type=int, default=3, help='Maximum priority level to process (1-5)')
    parser.add_argument('--batch-size', type=int, default=5, help='Number of chains per batch')
    parser.add_argument('--delay', type=int, default=60, help='Delay between batches (seconds)')
    parser.add_argument('--facility-type', help='Filter by facility type')
    parser.add_argument('--update-only', action='store_true', help='Only process chains due for update')
    
    # LLM options
    parser.add_argument('--llm', action='store_true', default=True, help='Use LLM extraction (default)')
    parser.add_argument('--no-llm', action='store_true', help='Use traditional extraction only')
    
    # System options
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
    parser.add_argument('--log-file', help='Log file path')
    parser.add_argument('--monitor', action='store_true', help='Monitor system resources')
    
    # Information options
    parser.add_argument('--help-csv', action='store_true', help='Show CSV requirements and format')
    parser.add_argument('--help-output', action='store_true', help='Show output schema and download info')
    parser.add_argument('--status', action='store_true', help='Show system status and CSV statistics')
    
    # Download options
    parser.add_argument('--download', action='store_true', help='Download results from droplet')
    parser.add_argument('--droplet-ip', help='Digital Ocean droplet IP for download')
    parser.add_argument('--download-dir', default='./downloads', help='Local download directory')
    
    args = parser.parse_args()
    
    # Handle help options
    if args.help_csv:
        print_csv_requirements()
        return
    
    if args.help_output:
        print_output_information()
        return
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    # Handle download option
    if args.download:
        if not args.droplet_ip:
            logger.error("--droplet-ip required for download")
            sys.exit(1)
        
        dm = DownloadManager(args.droplet_ip, local_download_dir=args.download_dir)
        
        print("📥 Downloading results from droplet...")
        package_path = dm.create_download_package()
        if package_path:
            local_path = dm.download_package(package_path)
            if local_path:
                extract_dir = dm.extract_package(local_path)
                print(f"✅ Results downloaded and extracted to: {extract_dir}")
            else:
                print("❌ Failed to download package")
        else:
            print("❌ Failed to create download package")
        return
    
    # Handle status option
    if args.status:
        print_system_status()
        
        if args.csv and os.path.exists(args.csv):
            chain_manager = CorporateChainManager(args.csv)
            chains = chain_manager.load_chains()
            stats = chain_manager.get_statistics()
            
            print(f"\n📊 CSV File Statistics ({args.csv}):")
            print(f"   Total chains: {stats['total_chains']}")
            print(f"   File size: {stats['file_size_mb']}MB")
            print(f"   Priority distribution: {stats['priority_distribution']}")
            print(f"   Facility types: {stats['facility_type_distribution']}")
            print(f"   Estimated facilities: {stats['estimated_total_facilities']}")
            print(f"   Chains due for update: {stats['chains_due_for_update']}")
        
        return
    
    # Require CSV file for processing
    if not args.csv:
        parser.error("--csv is required for processing")
    
    if not os.path.exists(args.csv):
        logger.error(f"CSV file not found: {args.csv}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Determine LLM usage
    use_llm = args.llm and not args.no_llm
    
    logger.info(f"🚀 Starting CSV-driven batch processing")
    logger.info(f"📁 CSV file: {args.csv}")
    logger.info(f"📁 Output directory: {args.output}")
    logger.info(f"🎯 Priority filter: <= {args.priority}")
    logger.info(f"📦 Batch size: {args.batch_size}")
    logger.info(f"⏱️  Delay between batches: {args.delay}s")
    logger.info(f"🧠 LLM extraction: {'enabled' if use_llm else 'disabled'}")
    
    # Show system status if monitoring
    if args.monitor:
        print_system_status()
    
    try:
        # Run batch processing
        result = asyncio.run(process_csv_batch(
            csv_file=args.csv,
            output_dir=args.output,
            use_llm=use_llm,
            priority_filter=args.priority,
            batch_size=args.batch_size,
            delay_between_batches=args.delay
        ))
        
        if result.get('success', False):
            print(f"\n🎉 Batch processing completed successfully!")
            print(f"📊 Results:")
            print(f"   Total chains processed: {result.get('total_chains_processed', 0)}")
            print(f"   Total facilities found: {result.get('total_facilities_found', 0)}")
            print(f"   Successful chains: {result.get('successful_chains', 0)}")
            print(f"   Failed chains: {result.get('failed_chains', 0)}")
            print(f"   Processing time: {result.get('total_processing_time_hours', 0):.1f} hours")
            print(f"📁 Consolidated file: {result.get('consolidated_file', 'N/A')}")
            
            # Show download instructions
            print(f"\n📥 To download results:")
            print(f"   scp -r root@YOUR_DROPLET_IP:{args.output} ./local_results/")
            print(f"   # OR use download manager:")
            print(f"   python main_csv_batch.py --download --droplet-ip YOUR_IP")
        
        else:
            print(f"\n❌ Batch processing failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("\n⏹️  Processing interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

