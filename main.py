#!/usr/bin/env python3
"""
Healthcare Facility Scraper - Main CLI Interface
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import List

from healthcare_scraper import HealthcareFacilityScraper
from navigation import SiteNavigator
from extractors import FacilityDataExtractor


def setup_logging(log_level: str = 'INFO', log_file: str = None):
    """Setup logging configuration"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


async def scrape_single_site(url: str, output_dir: str, formats: List[str]):
    """Scrape a single healthcare facility website"""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting scrape of {url}")
    
    try:
        scraper = HealthcareFacilityScraper(url, output_dir)
        facilities = await scraper.scrape_facilities()
        
        if facilities:
            logger.info(f"Successfully scraped {len(facilities)} facilities")
            
            # Save in requested formats
            saved_files = []
            for format_type in formats:
                filename = scraper.save_results(format_type)
                saved_files.append(filename)
                logger.info(f"Results saved to {filename}")
            
            return {
                'success': True,
                'facilities_count': len(facilities),
                'files': saved_files,
                'facilities': facilities
            }
        else:
            logger.warning(f"No facilities found for {url}")
            return {
                'success': False,
                'error': 'No facilities found',
                'facilities_count': 0
            }
            
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return {
            'success': False,
            'error': str(e),
            'facilities_count': 0
        }


async def scrape_multiple_sites(urls: List[str], output_dir: str, formats: List[str]):
    """Scrape multiple healthcare facility websites"""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting batch scrape of {len(urls)} websites")
    
    results = []
    
    for url in urls:
        logger.info(f"Processing {url}")
        result = await scrape_single_site(url, output_dir, formats)
        result['url'] = url
        results.append(result)
        
        # Add delay between sites to be respectful
        await asyncio.sleep(2)
    
    # Generate summary report
    successful_scrapes = [r for r in results if r['success']]
    total_facilities = sum(r['facilities_count'] for r in successful_scrapes)
    
    summary = {
        'total_sites': len(urls),
        'successful_sites': len(successful_scrapes),
        'failed_sites': len(urls) - len(successful_scrapes),
        'total_facilities': total_facilities,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save summary report
    summary_file = os.path.join(output_dir, f"scrape_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Batch scrape completed. Summary saved to {summary_file}")
    logger.info(f"Successfully scraped {len(successful_scrapes)}/{len(urls)} sites")
    logger.info(f"Total facilities found: {total_facilities}")
    
    return summary


def load_urls_from_file(file_path: str) -> List[str]:
    """Load URLs from a text file (one URL per line)"""
    urls = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                url = line.strip()
                if url and not url.startswith('#'):  # Skip comments
                    if not url.startswith('http'):
                        url = 'https://' + url
                    urls.append(url)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        sys.exit(1)
    
    return urls


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description='Healthcare Facility Scraper - Extract facility information from healthcare websites',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape a single website
  python main.py --url https://example.com --output ./results

  # Scrape multiple websites from a file
  python main.py --urls-file urls.txt --output ./results --formats json csv

  # Scrape with debug logging
  python main.py --url https://example.com --log-level DEBUG --log-file scraper.log

URL file format (one URL per line):
  https://lcca.com
  https://example-healthcare.com
  # This is a comment
  another-site.com  # https:// will be added automatically
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--url', '-u',
        help='Single URL to scrape'
    )
    input_group.add_argument(
        '--urls-file', '-f',
        help='File containing URLs to scrape (one per line)'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        default='./output',
        help='Output directory for results (default: ./output)'
    )
    parser.add_argument(
        '--formats',
        nargs='+',
        choices=['json', 'csv', 'excel'],
        default=['json'],
        help='Output formats (default: json)'
    )
    
    # Scraping options
    parser.add_argument(
        '--max-depth',
        type=int,
        default=4,
        help='Maximum crawl depth (default: 4)'
    )
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=8,
        help='Maximum concurrent requests (default: 8)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between requests in seconds (default: 1.0)'
    )
    
    # Logging options
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    parser.add_argument(
        '--log-file',
        help='Log file path (default: console only)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Update configuration with CLI arguments
    from config import CRAWL_CONFIG
    CRAWL_CONFIG['max_depth'] = args.max_depth
    CRAWL_CONFIG['max_concurrent'] = args.max_concurrent
    CRAWL_CONFIG['delay_between_requests'] = args.delay
    
    # Determine URLs to scrape
    if args.url:
        urls = [args.url]
    else:
        urls = load_urls_from_file(args.urls_file)
        if not urls:
            logger.error("No valid URLs found in file")
            sys.exit(1)
    
    logger.info(f"Starting healthcare facility scraper")
    logger.info(f"URLs to scrape: {len(urls)}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Output formats: {args.formats}")
    
    # Run the scraper
    try:
        if len(urls) == 1:
            result = asyncio.run(scrape_single_site(urls[0], args.output, args.formats))
            if result['success']:
                print(f"\n✅ Successfully scraped {result['facilities_count']} facilities")
                print(f"📁 Results saved to: {', '.join(result['files'])}")
            else:
                print(f"\n❌ Scraping failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)
        else:
            summary = asyncio.run(scrape_multiple_sites(urls, args.output, args.formats))
            print(f"\n📊 Batch Scraping Summary:")
            print(f"   Total sites: {summary['total_sites']}")
            print(f"   Successful: {summary['successful_sites']}")
            print(f"   Failed: {summary['failed_sites']}")
            print(f"   Total facilities: {summary['total_facilities']}")
            
            if summary['failed_sites'] > 0:
                print(f"\n❌ Some sites failed to scrape. Check the summary file for details.")
                sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
        print("\n⏹️  Scraping interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

