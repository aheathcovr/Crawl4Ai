#!/usr/bin/env python3
"""
Healthcare Facility Scraper - Best Practices Implementation
Follows Crawl4AI recommendations: Schema/Regex first, LLM as fallback only
Includes free validation system for data quality assurance
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional

from schema_based_extractor import SchemaBasedExtractor, HealthcareSchemaLibrary
from free_validation import FreeValidationSystem
from healthcare_scraper import FacilityInfo
from config_optimized import get_dynamic_config, print_system_status


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


class BestPracticesHealthcareScraper:
    """Healthcare scraper implementing Crawl4AI best practices"""
    
    def __init__(self, 
                 use_llm_fallback: bool = False,
                 enable_validation: bool = True,
                 validation_flags_only: bool = False):
        
        self.extractor = SchemaBasedExtractor(use_llm_fallback=use_llm_fallback)
        self.validator = FreeValidationSystem() if enable_validation else None
        self.validation_flags_only = validation_flags_only
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.stats = {
            "urls_processed": 0,
            "facilities_found": 0,
            "facilities_validated": 0,
            "high_confidence_facilities": 0,
            "flagged_facilities": 0,
            "processing_time": 0.0
        }
    
    async def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape a single URL following best practices"""
        
        start_time = datetime.now()
        self.logger.info(f"🔍 Scraping: {url}")
        
        try:
            # Step 1: Extract facilities using schema-first approach
            facilities = await self.extractor.extract_facilities(url)
            
            if not facilities:
                self.logger.warning(f"No facilities found at {url}")
                return {
                    "url": url,
                    "success": False,
                    "facilities": [],
                    "validation_reports": [],
                    "error": "No facilities found"
                }
            
            self.logger.info(f"✅ Found {len(facilities)} facilities")
            self.stats["facilities_found"] += len(facilities)
            
            # Step 2: Validate facilities (if enabled)
            validation_reports = []
            if self.validator:
                self.logger.info("🔍 Validating facility data...")
                
                facility_dicts = [self._facility_to_dict(f) for f in facilities]
                validation_reports = await self.validator.validate_batch(facility_dicts)
                
                self.stats["facilities_validated"] += len(validation_reports)
                
                # Count high confidence and flagged facilities
                for report in validation_reports:
                    if report.overall_confidence > 0.8:
                        self.stats["high_confidence_facilities"] += 1
                    if report.flags:
                        self.stats["flagged_facilities"] += 1
                
                # Filter facilities based on validation if flags_only mode
                if self.validation_flags_only:
                    validated_facilities = []
                    for facility, report in zip(facilities, validation_reports):
                        if report.is_likely_valid and report.overall_confidence > 0.6:
                            validated_facilities.append(facility)
                    
                    if len(validated_facilities) < len(facilities):
                        self.logger.info(f"🚫 Filtered out {len(facilities) - len(validated_facilities)} low-quality facilities")
                        facilities = validated_facilities
            
            # Step 3: Prepare results
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats["processing_time"] += processing_time
            self.stats["urls_processed"] += 1
            
            result = {
                "url": url,
                "success": True,
                "facilities": [self._facility_to_dict(f) for f in facilities],
                "validation_reports": [self._validation_report_to_dict(r) for r in validation_reports],
                "extraction_stats": self.extractor.get_extraction_stats(),
                "processing_time_seconds": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Completed {url} in {processing_time:.1f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error scraping {url}: {e}")
            return {
                "url": url,
                "success": False,
                "facilities": [],
                "validation_reports": [],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def scrape_multiple_urls(self, urls: List[str], delay_between_requests: float = 2.0) -> List[Dict[str, Any]]:
        """Scrape multiple URLs with rate limiting"""
        
        results = []
        
        for i, url in enumerate(urls):
            self.logger.info(f"📊 Progress: {i+1}/{len(urls)}")
            
            result = await self.scrape_url(url)
            results.append(result)
            
            # Rate limiting (respect server resources)
            if i < len(urls) - 1:  # Don't delay after the last URL
                self.logger.info(f"⏱️  Waiting {delay_between_requests}s before next request...")
                await asyncio.sleep(delay_between_requests)
        
        return results
    
    def _facility_to_dict(self, facility: FacilityInfo) -> Dict[str, Any]:
        """Convert FacilityInfo to dictionary"""
        return {
            "name": facility.name,
            "facility_type": facility.facility_type,
            "address": facility.address,
            "city": facility.city,
            "state": facility.state,
            "zip_code": facility.zip_code,
            "phone": facility.phone,
            "email": facility.email,
            "website": facility.website,
            "administrator": facility.administrator,
            "beds": facility.beds,
            "services_offered": facility.services_offered,
            "source_url": facility.source_url,
            "extraction_method": getattr(facility, 'extraction_method', 'unknown'),
            "scraping_date": datetime.now().isoformat()
        }
    
    def _validation_report_to_dict(self, report) -> Dict[str, Any]:
        """Convert validation report to dictionary"""
        return {
            "facility_name": report.facility_name,
            "overall_confidence": report.overall_confidence,
            "is_likely_valid": report.is_likely_valid,
            "flags": report.flags,
            "validation_summary": report.validation_summary,
            "field_validations": [
                {
                    "field_name": v.field_name,
                    "is_valid": v.is_valid,
                    "confidence": v.confidence,
                    "validation_method": v.validation_method,
                    "corrected_value": v.corrected_value,
                    "error_message": v.error_message
                }
                for v in report.field_validations
            ]
        }
    
    async def close(self):
        """Clean up resources"""
        if self.validator:
            await self.validator.close()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = self.stats.copy()
        
        if stats["urls_processed"] > 0:
            stats["avg_facilities_per_url"] = stats["facilities_found"] / stats["urls_processed"]
            stats["avg_processing_time_per_url"] = stats["processing_time"] / stats["urls_processed"]
        
        if stats["facilities_validated"] > 0:
            stats["high_confidence_rate"] = stats["high_confidence_facilities"] / stats["facilities_validated"]
            stats["flagged_rate"] = stats["flagged_facilities"] / stats["facilities_validated"]
        
        return stats


def save_results(results: List[Dict[str, Any]], output_dir: str, base_filename: str = None):
    """Save results in multiple formats"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not base_filename:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"healthcare_facilities_{timestamp}"
    
    # Consolidate all facilities
    all_facilities = []
    all_validation_reports = []
    
    for result in results:
        all_facilities.extend(result.get("facilities", []))
        all_validation_reports.extend(result.get("validation_reports", []))
    
    # Save JSON format (complete data)
    json_file = os.path.join(output_dir, f"{base_filename}.json")
    with open(json_file, 'w') as f:
        json.dump({
            "facilities": all_facilities,
            "validation_reports": all_validation_reports,
            "scraping_results": results,
            "total_facilities": len(all_facilities),
            "generation_date": datetime.now().isoformat()
        }, f, indent=2)
    
    # Save CSV format (facilities only)
    csv_file = os.path.join(output_dir, f"{base_filename}.csv")
    if all_facilities:
        import pandas as pd
        
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
    
    # Save validation summary
    if all_validation_reports:
        validation_file = os.path.join(output_dir, f"{base_filename}_validation_summary.json")
        
        # Calculate validation statistics
        total_reports = len(all_validation_reports)
        valid_facilities = len([r for r in all_validation_reports if r["is_likely_valid"]])
        avg_confidence = sum(r["overall_confidence"] for r in all_validation_reports) / total_reports
        
        # Flag distribution
        all_flags = []
        for report in all_validation_reports:
            all_flags.extend(report["flags"])
        
        flag_counts = {}
        for flag in all_flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
        
        validation_summary = {
            "total_facilities": total_reports,
            "valid_facilities": valid_facilities,
            "validation_rate": valid_facilities / total_reports if total_reports > 0 else 0,
            "average_confidence": avg_confidence,
            "flag_distribution": flag_counts,
            "generation_date": datetime.now().isoformat()
        }
        
        with open(validation_file, 'w') as f:
            json.dump(validation_summary, f, indent=2)
    
    return {
        "json_file": json_file,
        "csv_file": csv_file,
        "validation_file": validation_file if all_validation_reports else None,
        "total_facilities": len(all_facilities)
    }


async def main():
    """Main CLI interface"""
    
    parser = argparse.ArgumentParser(
        description='Healthcare Facility Scraper - Best Practices Implementation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Schema-first extraction (recommended)
  python main_best_practices.py --url https://sunriseseniorliving.com --output ./results

  # With validation enabled
  python main_best_practices.py --url https://lcca.com --output ./results --validate

  # Multiple URLs with rate limiting
  python main_best_practices.py --urls https://lcca.com https://sunriseseniorliving.com --output ./results --delay 3

  # High-quality results only (validation filtering)
  python main_best_practices.py --url https://genesishcc.com --output ./results --validate --quality-filter

  # Enable LLM fallback for difficult sites
  python main_best_practices.py --url https://difficult-site.com --output ./results --llm-fallback

  # Show available schemas
  python main_best_practices.py --list-schemas

  # System status check
  python main_best_practices.py --status
        """
    )
    
    # Input options
    parser.add_argument('--url', help='Single URL to scrape')
    parser.add_argument('--urls', nargs='+', help='Multiple URLs to scrape')
    parser.add_argument('--urls-file', help='File containing URLs (one per line)')
    parser.add_argument('--output', '-o', default='./output', help='Output directory')
    
    # Extraction options
    parser.add_argument('--llm-fallback', action='store_true', help='Enable LLM fallback for difficult sites')
    parser.add_argument('--delay', type=float, default=2.0, help='Delay between requests (seconds)')
    
    # Validation options
    parser.add_argument('--validate', action='store_true', help='Enable free validation system')
    parser.add_argument('--quality-filter', action='store_true', help='Only return high-quality validated results')
    
    # System options
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
    parser.add_argument('--log-file', help='Log file path')
    
    # Information options
    parser.add_argument('--list-schemas', action='store_true', help='List available extraction schemas')
    parser.add_argument('--status', action='store_true', help='Show system status')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    # Handle information options
    if args.list_schemas:
        schema_library = HealthcareSchemaLibrary()
        print("📋 Available Extraction Schemas:")
        print("=" * 50)
        for schema in schema_library.schemas:
            print(f"Name: {schema.name}")
            print(f"Description: {schema.description}")
            print(f"Confidence: {schema.confidence_score}")
            print(f"Site Patterns: {', '.join(schema.site_patterns)}")
            print("-" * 30)
        return
    
    if args.status:
        print_system_status()
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
    
    if not urls:
        parser.error("No URLs provided. Use --url, --urls, or --urls-file")
    
    # Remove duplicates while preserving order
    urls = list(dict.fromkeys(urls))
    
    logger.info(f"🚀 Starting healthcare facility scraping")
    logger.info(f"📊 URLs to process: {len(urls)}")
    logger.info(f"🧠 LLM fallback: {'enabled' if args.llm_fallback else 'disabled'}")
    logger.info(f"✅ Validation: {'enabled' if args.validate else 'disabled'}")
    logger.info(f"🎯 Quality filter: {'enabled' if args.quality_filter else 'disabled'}")
    logger.info(f"⏱️  Request delay: {args.delay}s")
    
    # Initialize scraper
    scraper = BestPracticesHealthcareScraper(
        use_llm_fallback=args.llm_fallback,
        enable_validation=args.validate,
        validation_flags_only=args.quality_filter
    )
    
    try:
        # Process URLs
        if len(urls) == 1:
            results = [await scraper.scrape_url(urls[0])]
        else:
            results = await scraper.scrape_multiple_urls(urls, args.delay)
        
        # Save results
        saved_files = save_results(results, args.output)
        
        # Show performance statistics
        stats = scraper.get_performance_stats()
        
        print(f"\n🎉 Scraping completed!")
        print(f"📊 Performance Statistics:")
        print(f"   URLs processed: {stats['urls_processed']}")
        print(f"   Facilities found: {stats['facilities_found']}")
        print(f"   Average facilities per URL: {stats.get('avg_facilities_per_url', 0):.1f}")
        print(f"   Total processing time: {stats['processing_time']:.1f}s")
        print(f"   Average time per URL: {stats.get('avg_processing_time_per_url', 0):.1f}s")
        
        if args.validate:
            print(f"   Facilities validated: {stats['facilities_validated']}")
            print(f"   High confidence rate: {stats.get('high_confidence_rate', 0):.1%}")
            print(f"   Flagged rate: {stats.get('flagged_rate', 0):.1%}")
        
        print(f"\n📁 Output files:")
        print(f"   JSON: {saved_files['json_file']}")
        print(f"   CSV: {saved_files['csv_file']}")
        if saved_files['validation_file']:
            print(f"   Validation: {saved_files['validation_file']}")
        
        # Show extraction method statistics
        extraction_stats = scraper.extractor.get_extraction_stats()
        print(f"\n🔧 Extraction Method Performance:")
        print(f"   Schema-based success rate: {extraction_stats.get('schema_success_rate', 0):.1%}")
        print(f"   Regex-based success rate: {extraction_stats.get('regex_success_rate', 0):.1%}")
        if args.llm_fallback:
            print(f"   LLM fallback rate: {extraction_stats.get('llm_fallback_rate', 0):.1%}")
        print(f"   Overall success rate: {extraction_stats.get('overall_success_rate', 0):.1%}")
        
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

