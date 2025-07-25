#!/usr/bin/env python3
"""
Enhanced Healthcare Facility Scraper with LLM Integration
Uses OpenAI GPT models for surgical precision in data extraction
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import List

from llm_extractors import EnhancedHealthcareScraper, LLMFacilityExtractor
from healthcare_scraper import HealthcareFacilityScraper


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


async def scrape_with_llm(url: str, output_dir: str, model: str = "gpt-4o-mini") -> dict:
    """Scrape using LLM-enhanced extraction"""
    logger = logging.getLogger(__name__)
    logger.info(f"🧠 Starting LLM-enhanced scraping of {url}")
    logger.info(f"🤖 Using model: {model}")
    
    try:
        # Check for OpenAI API key
        if not os.getenv('OPENAI_API_KEY'):
            logger.error("❌ OPENAI_API_KEY environment variable not set")
            logger.info("💡 Set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
            return {
                'success': False,
                'error': 'Missing OpenAI API key',
                'facilities_count': 0
            }
        
        # Initialize enhanced scraper
        scraper = EnhancedHealthcareScraper(url, output_dir, use_llm=True)
        
        # Run LLM-enhanced scraping
        facilities = await scraper.scrape_with_llm_enhancement()
        
        if facilities:
            logger.info(f"✅ LLM extraction successful: {len(facilities)} facilities found")
            
            # Save results using traditional scraper's save method
            traditional_scraper = HealthcareFacilityScraper(url, output_dir)
            traditional_scraper.facilities = facilities
            
            saved_files = []
            for format_type in ['json', 'csv']:
                filename = traditional_scraper.save_results(format_type)
                saved_files.append(filename)
                logger.info(f"💾 Results saved to {filename}")
            
            # Show sample of extracted data
            sample_facility = facilities[0]
            logger.info(f"📋 Sample facility extracted:")
            logger.info(f"   Name: {sample_facility.name}")
            logger.info(f"   Type: {sample_facility.facility_type}")
            logger.info(f"   Address: {sample_facility.address}, {sample_facility.city}, {sample_facility.state}")
            logger.info(f"   Phone: {sample_facility.phone}")
            logger.info(f"   Services: {len(sample_facility.services_offered)} services")
            
            return {
                'success': True,
                'facilities_count': len(facilities),
                'files': saved_files,
                'facilities': facilities
            }
        else:
            logger.warning(f"⚠️  No facilities found for {url}")
            return {
                'success': False,
                'error': 'No facilities found',
                'facilities_count': 0
            }
            
    except Exception as e:
        logger.error(f"💥 Error in LLM scraping: {e}")
        return {
            'success': False,
            'error': str(e),
            'facilities_count': 0
        }


async def compare_extraction_methods(url: str, output_dir: str) -> dict:
    """Compare traditional vs LLM extraction methods"""
    logger = logging.getLogger(__name__)
    logger.info(f"🔬 Comparing extraction methods for {url}")
    
    results = {
        'traditional': {'facilities_count': 0, 'time_taken': 0},
        'llm': {'facilities_count': 0, 'time_taken': 0},
        'comparison': {}
    }
    
    # Traditional extraction
    logger.info("🔧 Running traditional extraction...")
    start_time = datetime.now()
    try:
        traditional_scraper = HealthcareFacilityScraper(url, f"{output_dir}/traditional")
        traditional_facilities = await traditional_scraper.scrape_facilities()
        results['traditional']['facilities_count'] = len(traditional_facilities)
        results['traditional']['time_taken'] = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ Traditional: {len(traditional_facilities)} facilities in {results['traditional']['time_taken']:.1f}s")
    except Exception as e:
        logger.error(f"❌ Traditional extraction failed: {e}")
        results['traditional']['error'] = str(e)
    
    # LLM extraction
    logger.info("🧠 Running LLM extraction...")
    start_time = datetime.now()
    try:
        llm_result = await scrape_with_llm(url, f"{output_dir}/llm")
        results['llm']['facilities_count'] = llm_result['facilities_count']
        results['llm']['time_taken'] = (datetime.now() - start_time).total_seconds()
        results['llm']['success'] = llm_result['success']
        logger.info(f"✅ LLM: {llm_result['facilities_count']} facilities in {results['llm']['time_taken']:.1f}s")
    except Exception as e:
        logger.error(f"❌ LLM extraction failed: {e}")
        results['llm']['error'] = str(e)
    
    # Comparison analysis
    if results['traditional']['facilities_count'] > 0 and results['llm']['facilities_count'] > 0:
        improvement = ((results['llm']['facilities_count'] - results['traditional']['facilities_count']) / 
                      results['traditional']['facilities_count']) * 100
        results['comparison']['improvement_percentage'] = improvement
        results['comparison']['time_difference'] = results['llm']['time_taken'] - results['traditional']['time_taken']
        
        logger.info(f"📊 Comparison Results:")
        logger.info(f"   Traditional: {results['traditional']['facilities_count']} facilities")
        logger.info(f"   LLM: {results['llm']['facilities_count']} facilities")
        logger.info(f"   Improvement: {improvement:+.1f}%")
        logger.info(f"   Time difference: {results['comparison']['time_difference']:+.1f}s")
    
    # Save comparison report
    comparison_file = os.path.join(output_dir, f"extraction_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"📋 Comparison report saved to {comparison_file}")
    
    return results


def main():
    """Enhanced CLI interface with LLM options"""
    parser = argparse.ArgumentParser(
        description='Enhanced Healthcare Facility Scraper with LLM Integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # LLM-enhanced scraping (recommended)
  python main_llm.py --url https://lcca.com --output ./results --llm

  # Compare traditional vs LLM extraction
  python main_llm.py --url https://lcca.com --output ./results --compare

  # Use specific OpenAI model
  python main_llm.py --url https://lcca.com --output ./results --llm --model gpt-4

  # Traditional scraping (fallback)
  python main_llm.py --url https://lcca.com --output ./results

Environment Variables:
  OPENAI_API_KEY    Your OpenAI API key (required for LLM features)
  OPENAI_API_BASE   Custom OpenAI API base URL (optional)
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
    
    # LLM options
    parser.add_argument(
        '--llm',
        action='store_true',
        help='Use LLM-enhanced extraction (requires OPENAI_API_KEY)'
    )
    parser.add_argument(
        '--model',
        default='gpt-4o-mini',
        choices=['gpt-4o-mini', 'gpt-4o', 'gpt-4', 'gpt-3.5-turbo'],
        help='OpenAI model to use (default: gpt-4o-mini)'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare traditional vs LLM extraction methods'
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
    
    # Check for OpenAI API key if LLM features requested
    if (args.llm or args.compare) and not os.getenv('OPENAI_API_KEY'):
        logger.error("❌ OPENAI_API_KEY environment variable required for LLM features")
        logger.info("💡 Get your API key from: https://platform.openai.com/api-keys")
        logger.info("💡 Set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Determine URLs to scrape
    if args.url:
        urls = [args.url]
    else:
        try:
            with open(args.urls_file, 'r') as f:
                urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        except FileNotFoundError:
            logger.error(f"❌ File not found: {args.urls_file}")
            sys.exit(1)
    
    logger.info(f"🚀 Starting enhanced healthcare facility scraper")
    logger.info(f"🎯 URLs to process: {len(urls)}")
    logger.info(f"📁 Output directory: {args.output}")
    if args.llm or args.compare:
        logger.info(f"🤖 LLM model: {args.model}")
    
    # Run the scraper
    try:
        if args.compare:
            # Comparison mode
            for url in urls:
                logger.info(f"\n🔬 Comparing extraction methods for {url}")
                result = asyncio.run(compare_extraction_methods(url, args.output))
        
        elif args.llm:
            # LLM-enhanced mode
            for url in urls:
                logger.info(f"\n🧠 LLM-enhanced scraping: {url}")
                result = asyncio.run(scrape_with_llm(url, args.output, args.model))
                
                if result['success']:
                    print(f"\n✅ Successfully scraped {result['facilities_count']} facilities")
                    print(f"📁 Results saved to: {', '.join(result['files'])}")
                else:
                    print(f"\n❌ Scraping failed: {result.get('error', 'Unknown error')}")
        
        else:
            # Traditional mode
            for url in urls:
                logger.info(f"\n🔧 Traditional scraping: {url}")
                scraper = HealthcareFacilityScraper(url, args.output)
                facilities = asyncio.run(scraper.scrape_facilities())
                
                if facilities:
                    json_file = scraper.save_results('json')
                    csv_file = scraper.save_results('csv')
                    print(f"\n✅ Successfully scraped {len(facilities)} facilities")
                    print(f"📁 Results saved to: {json_file}, {csv_file}")
                else:
                    print(f"\n❌ No facilities found for {url}")
    
    except KeyboardInterrupt:
        logger.info("\n⏹️  Scraping interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

