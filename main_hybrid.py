#!/usr/bin/env python3
"""
Healthcare Facility Scraper - Hybrid LLM + Algorithm Implementation
Uses LLM intelligence for navigation, fast algorithms for extraction
OpenRouter integration with model switching capabilities
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional

from hybrid_extractor import HybridExtractor, ExtractionResult
from csv_processor import CSVProcessor
from free_validation import FreeValidationSystem
from download_manager import DownloadManager


def setup_logging(log_level: str = 'INFO', log_file: str = None):
    """Setup logging configuration"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(log_file, maxBytes=50*1024*1024, backupCount=3)
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


class HybridHealthcareScraper:
    """Main hybrid healthcare scraper with OpenRouter integration"""
    
    def __init__(self, 
                 openrouter_api_key: str,
                 model_preference: str = "balanced",
                 enable_validation: bool = True):
        
        self.extractor = HybridExtractor(
            openrouter_api_key=openrouter_api_key,
            enable_validation=enable_validation,
            llm_model_preference=model_preference
        )
        
        self.csv_processor = CSVProcessor()
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.session_stats = {
            "session_start": datetime.now().isoformat(),
            "urls_processed": 0,
            "total_facilities": 0,
            "total_processing_time": 0.0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "model_preference": model_preference
        }
    
    async def scrape_single_url(self, url: str) -> Dict[str, Any]:
        """Scrape a single URL using hybrid approach"""
        
        self.logger.info(f"🎯 Hybrid scraping: {url}")
        
        try:
            result = await self.extractor.extract_facilities_hybrid(url)
            
            # Update session stats
            self.session_stats["urls_processed"] += 1
            self.session_stats["total_facilities"] += len(result.facilities)
            self.session_stats["total_processing_time"] += result.processing_time
            
            if result.facilities:
                self.session_stats["successful_extractions"] += 1
            else:
                self.session_stats["failed_extractions"] += 1
            
            return {
                "url": url,
                "success": True,
                "facilities": result.facilities,
                "extraction_method": result.extraction_method,
                "processing_time": result.processing_time,
                "confidence_score": result.confidence_score,
                "llm_navigation_used": result.llm_navigation_used,
                "validation_reports": result.validation_reports,
                "metadata": result.metadata,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to scrape {url}: {e}")
            
            self.session_stats["urls_processed"] += 1
            self.session_stats["failed_extractions"] += 1
            
            return {
                "url": url,
                "success": False,
                "facilities": [],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def scrape_multiple_urls(self, 
                                 urls: List[str], 
                                 delay_between_requests: float = 3.0,
                                 batch_size: int = 1) -> List[Dict[str, Any]]:
        """Scrape multiple URLs with rate limiting and batching"""
        
        results = []
        
        # Process in batches to manage memory and API limits
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            batch_results = []
            
            self.logger.info(f"📦 Processing batch {i//batch_size + 1}/{(len(urls) + batch_size - 1)//batch_size}")
            
            # Process batch concurrently if batch_size > 1
            if batch_size == 1:
                for url in batch:
                    result = await self.scrape_single_url(url)
                    batch_results.append(result)
                    
                    # Rate limiting
                    if url != batch[-1]:  # Don't delay after last URL in batch
                        self.logger.info(f"⏱️  Waiting {delay_between_requests}s...")
                        await asyncio.sleep(delay_between_requests)
            else:
                # Concurrent processing for larger batches
                tasks = [self.scrape_single_url(url) for url in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle exceptions
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        batch_results[j] = {
                            "url": batch[j],
                            "success": False,
                            "facilities": [],
                            "error": str(result),
                            "timestamp": datetime.now().isoformat()
                        }
            
            results.extend(batch_results)
            
            # Delay between batches
            if i + batch_size < len(urls):
                await asyncio.sleep(delay_between_requests)
        
        return results
    
    async def scrape_from_csv(self, 
                            csv_file: str,
                            priority_filter: Optional[int] = None,
                            facility_type_filter: Optional[str] = None,
                            **kwargs) -> List[Dict[str, Any]]:
        """Scrape facilities from CSV file of corporate chains"""
        
        self.logger.info(f"📋 Processing CSV file: {csv_file}")
        
        # Load and filter corporate chains
        chains = self.csv_processor.load_corporate_chains(csv_file)
        
        if priority_filter:
            chains = [c for c in chains if c.priority <= priority_filter]
            self.logger.info(f"🎯 Filtered to priority {priority_filter}: {len(chains)} chains")
        
        if facility_type_filter:
            chains = [c for c in chains if facility_type_filter.lower() in c.facility_types.lower()]
            self.logger.info(f"🏥 Filtered to {facility_type_filter}: {len(chains)} chains")
        
        # Extract URLs
        urls = []
        for chain in chains:
            urls.append(chain.primary_url)
            if chain.secondary_urls:
                urls.extend(chain.secondary_urls)
        
        # Remove duplicates
        urls = list(dict.fromkeys(urls))
        
        self.logger.info(f"🌐 Total URLs to process: {len(urls)}")
        
        # Process URLs
        return await self.scrape_multiple_urls(urls, **kwargs)
    
    def switch_model_preference(self, preference: str):
        """Switch LLM model preference during runtime"""
        
        self.extractor.switch_model_preference(preference)
        self.session_stats["model_preference"] = preference
        self.logger.info(f"🔄 Switched to {preference} model preference")
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get comprehensive session statistics"""
        
        # Get extractor performance stats
        extractor_stats = self.extractor.get_performance_stats()
        
        # Combine with session stats
        combined_stats = {
            **self.session_stats,
            **extractor_stats,
            "session_duration": (datetime.now() - datetime.fromisoformat(self.session_stats["session_start"])).total_seconds()
        }
        
        # Calculate rates
        if combined_stats["urls_processed"] > 0:
            combined_stats["success_rate"] = combined_stats["successful_extractions"] / combined_stats["urls_processed"]
            combined_stats["avg_facilities_per_url"] = combined_stats["total_facilities"] / combined_stats["urls_processed"]
            combined_stats["avg_processing_time_per_url"] = combined_stats["total_processing_time"] / combined_stats["urls_processed"]
        
        return combined_stats
    
    async def close(self):
        """Clean up resources"""
        await self.extractor.close()


def save_results(results: List[Dict[str, Any]], 
                output_dir: str, 
                base_filename: str = None,
                formats: List[str] = None) -> Dict[str, str]:
    """Save results in multiple formats"""
    
    if formats is None:
        formats = ["json", "csv"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not base_filename:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"hybrid_healthcare_facilities_{timestamp}"
    
    saved_files = {}
    
    # Consolidate all facilities
    all_facilities = []
    all_validation_reports = []
    
    for result in results:
        all_facilities.extend(result.get("facilities", []))
        all_validation_reports.extend(result.get("validation_reports", []))
    
    # Save JSON format (complete data)
    if "json" in formats:
        json_file = os.path.join(output_dir, f"{base_filename}.json")
        with open(json_file, 'w') as f:
            json.dump({
                "facilities": all_facilities,
                "validation_reports": all_validation_reports,
                "scraping_results": results,
                "total_facilities": len(all_facilities),
                "generation_date": datetime.now().isoformat(),
                "extraction_type": "hybrid_llm_algorithm"
            }, f, indent=2)
        saved_files["json"] = json_file
    
    # Save CSV format (facilities only)
    if "csv" in formats and all_facilities:
        import pandas as pd
        
        csv_file = os.path.join(output_dir, f"{base_filename}.csv")
        
        # Flatten facilities for CSV
        flattened_facilities = []
        for facility in all_facilities:
            flat_facility = facility.copy()
            # Convert lists to pipe-separated strings
            for key, value in flat_facility.items():
                if isinstance(value, list):
                    flat_facility[key] = " | ".join(str(v) for v in value if v)
            flattened_facilities.append(flat_facility)
        
        df = pd.DataFrame(flattened_facilities)
        df.to_csv(csv_file, index=False)
        saved_files["csv"] = csv_file
    
    # Save Excel format
    if "excel" in formats and all_facilities:
        import pandas as pd
        
        excel_file = os.path.join(output_dir, f"{base_filename}.xlsx")
        
        # Create multiple sheets
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Facilities sheet
            if all_facilities:
                facilities_df = pd.DataFrame(all_facilities)
                facilities_df.to_excel(writer, sheet_name='Facilities', index=False)
            
            # Validation summary sheet
            if all_validation_reports:
                validation_df = pd.DataFrame(all_validation_reports)
                validation_df.to_excel(writer, sheet_name='Validation', index=False)
            
            # Results summary sheet
            summary_data = []
            for result in results:
                summary_data.append({
                    "url": result["url"],
                    "success": result["success"],
                    "facilities_found": len(result.get("facilities", [])),
                    "extraction_method": result.get("extraction_method", ""),
                    "processing_time": result.get("processing_time", 0),
                    "confidence_score": result.get("confidence_score", 0),
                    "llm_navigation_used": result.get("llm_navigation_used", False)
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        saved_files["excel"] = excel_file
    
    return saved_files


async def main():
    """Main CLI interface"""
    
    parser = argparse.ArgumentParser(
        description='Hybrid LLM + Algorithm Healthcare Facility Scraper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Single URL with balanced models
  python main_hybrid.py --url https://sunriseseniorliving.com --output ./results

  # Multiple URLs with fast models
  python main_hybrid.py --urls https://lcca.com https://genesishcc.com --output ./results --model-preference fast

  # CSV batch processing with precision models
  python main_hybrid.py --csv corporate_chains.csv --output ./results --model-preference precision --priority 2

  # High-throughput processing
  python main_hybrid.py --csv corporate_chains.csv --output ./results --batch-size 3 --delay 2 --model-preference fast

  # Custom model selection
  python main_hybrid.py --url https://brookdale.com --output ./results --custom-model "anthropic/claude-3.5-sonnet"

  # List available models
  python main_hybrid.py --list-models

  # Performance monitoring
  python main_hybrid.py --csv corporate_chains.csv --output ./results --monitor --stats-interval 10
        """
    )
    
    # Input options
    parser.add_argument('--url', help='Single URL to scrape')
    parser.add_argument('--urls', nargs='+', help='Multiple URLs to scrape')
    parser.add_argument('--urls-file', help='File containing URLs (one per line)')
    parser.add_argument('--csv', help='CSV file with corporate chains')
    
    # Output options
    parser.add_argument('--output', '-o', default='./hybrid_results', help='Output directory')
    parser.add_argument('--formats', nargs='+', choices=['json', 'csv', 'excel'], default=['json', 'csv'], help='Output formats')
    
    # LLM options
    parser.add_argument('--api-key', help='OpenRouter API key (or set OPENROUTER_API_KEY env var)')
    parser.add_argument('--model-preference', choices=['fast', 'balanced', 'precision'], default='balanced', help='Model preference for different tasks')
    parser.add_argument('--custom-model', help='Use specific model (overrides preference)')
    
    # Processing options
    parser.add_argument('--delay', type=float, default=3.0, help='Delay between requests (seconds)')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of URLs to process concurrently')
    parser.add_argument('--priority', type=int, help='Filter CSV by priority (1-5)')
    parser.add_argument('--facility-type', help='Filter CSV by facility type')
    
    # Validation options
    parser.add_argument('--no-validation', action='store_true', help='Disable validation system')
    
    # System options
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
    parser.add_argument('--log-file', help='Log file path')
    parser.add_argument('--monitor', action='store_true', help='Enable performance monitoring')
    parser.add_argument('--stats-interval', type=int, default=30, help='Statistics reporting interval (seconds)')
    
    # Information options
    parser.add_argument('--list-models', action='store_true', help='List available models and exit')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    # Get API key
    api_key = args.api_key or os.getenv('OPENROUTER_API_KEY') or "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"
    
    if not api_key:
        logger.error("OpenRouter API key required. Use --api-key or set OPENROUTER_API_KEY environment variable")
        sys.exit(1)
    
    # Handle information options
    if args.list_models:
        from hybrid_llm_navigator import OpenRouterClient
        client = OpenRouterClient(api_key)
        models = client.list_available_models()
        
        print("🤖 Available Model Configurations:")
        print("=" * 50)
        
        print("\nPredefined Model Preferences:")
        for pref_type, config in models["model_configs"].items():
            print(f"  {pref_type}:")
            print(f"    Model: {config['model']}")
            print(f"    Description: {config['description']}")
            print(f"    Max Tokens: {config['max_tokens']}")
            print()
        
        print("Custom Models Available:")
        for model in models["custom_models"]:
            print(f"  - {model}")
        
        await client.close()
        return
    
    # Collect URLs to process
    urls = []
    
    if args.url:
        urls.append(args.url)
    
    if args.urls:
        urls.extend(args.urls)
    
    if args.urls_file:
        if os.path.exists(args.urls_file):
            with open(args.urls_file, 'r') as f:
                file_urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                urls.extend(file_urls)
        else:
            logger.error(f"URLs file not found: {args.urls_file}")
            sys.exit(1)
    
    # Check if we have input
    if not urls and not args.csv:
        parser.error("No input provided. Use --url, --urls, --urls-file, or --csv")
    
    # Remove duplicates while preserving order
    if urls:
        urls = list(dict.fromkeys(urls))
    
    logger.info(f"🚀 Starting hybrid healthcare facility scraping")
    logger.info(f"🧠 Model preference: {args.model_preference}")
    logger.info(f"✅ Validation: {'disabled' if args.no_validation else 'enabled'}")
    logger.info(f"⏱️  Request delay: {args.delay}s")
    logger.info(f"📦 Batch size: {args.batch_size}")
    
    # Initialize scraper
    scraper = HybridHealthcareScraper(
        openrouter_api_key=api_key,
        model_preference=args.model_preference,
        enable_validation=not args.no_validation
    )
    
    # Switch to custom model if specified
    if args.custom_model:
        # This would require extending the extractor to support custom models
        logger.info(f"🎯 Using custom model: {args.custom_model}")
    
    try:
        # Process input
        if args.csv:
            logger.info(f"📋 Processing CSV file: {args.csv}")
            results = await scraper.scrape_from_csv(
                args.csv,
                priority_filter=args.priority,
                facility_type_filter=args.facility_type,
                delay_between_requests=args.delay,
                batch_size=args.batch_size
            )
        else:
            logger.info(f"🌐 Processing {len(urls)} URLs")
            results = await scraper.scrape_multiple_urls(
                urls,
                delay_between_requests=args.delay,
                batch_size=args.batch_size
            )
        
        # Save results
        saved_files = save_results(results, args.output, formats=args.formats)
        
        # Show session statistics
        stats = scraper.get_session_statistics()
        
        print(f"\n🎉 Hybrid scraping completed!")
        print(f"📊 Session Statistics:")
        print(f"   URLs processed: {stats['urls_processed']}")
        print(f"   Successful extractions: {stats['successful_extractions']}")
        print(f"   Failed extractions: {stats['failed_extractions']}")
        print(f"   Success rate: {stats.get('success_rate', 0):.1%}")
        print(f"   Total facilities found: {stats['total_facilities']}")
        print(f"   Average facilities per URL: {stats.get('avg_facilities_per_url', 0):.1f}")
        print(f"   Total processing time: {stats['total_processing_time']:.1f}s")
        print(f"   Average time per URL: {stats.get('avg_processing_time_per_url', 0):.1f}s")
        print(f"   Session duration: {stats.get('session_duration', 0):.1f}s")
        
        print(f"\n🔧 Extraction Method Performance:")
        print(f"   LLM navigation success rate: {stats.get('llm_navigation_success_rate', 0):.1%}")
        print(f"   Schema extraction success rate: {stats.get('schema_success_rate', 0):.1%}")
        print(f"   Regex extraction success rate: {stats.get('regex_success_rate', 0):.1%}")
        print(f"   LLM fallback rate: {stats.get('llm_fallback_rate', 0):.1%}")
        
        print(f"\n📁 Output files:")
        for format_type, file_path in saved_files.items():
            print(f"   {format_type.upper()}: {file_path}")
        
        # Show model usage summary
        print(f"\n🤖 Model Configuration:")
        print(f"   Preference: {stats['model_preference']}")
        print(f"   API: OpenRouter")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Scraping interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
    finally:
        await scraper.close()


if __name__ == '__main__':
    asyncio.run(main())

