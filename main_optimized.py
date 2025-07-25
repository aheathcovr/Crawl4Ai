#!/usr/bin/env python3
"""
Optimized Healthcare Facility Scraper for 2 vCPU / 4GB RAM Digital Ocean Droplet
Automatically adjusts settings based on available resources
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import gc
from datetime import datetime
from typing import List

from config_optimized import (
    get_dynamic_config, get_recommended_ollama_model, 
    check_system_resources, print_system_status,
    OLLAMA_MODELS_BY_MEMORY, OPTIMIZATION_TIPS
)
from llm_providers import (
    LLMConfig, LLMProvider, setup_openrouter, setup_ollama, setup_local_api
)
from llm_extractors_enhanced import EnhancedHealthcareScraper
from healthcare_scraper import HealthcareFacilityScraper


def setup_logging(log_level: str = 'INFO', log_file: str = None):
    """Setup optimized logging for resource-constrained environment"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Limit log file size to prevent disk issues
    handlers = []
    
    # Console handler with reduced verbosity
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    handlers.append(console_handler)
    
    # File handler with rotation
    if log_file:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=50*1024*1024,  # 50MB max
            backupCount=2
        )
        file_handler.setLevel(level)
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def get_optimal_llm_config() -> LLMConfig:
    """Get optimal LLM configuration for 4GB RAM system"""
    
    config = get_dynamic_config()
    
    # Check available providers in order of efficiency for 4GB RAM
    
    # 1. Try OpenRouter free models (most efficient for limited RAM)
    if os.getenv('OPENROUTER_API_KEY'):
        return LLMConfig(
            provider=LLMProvider.OPENROUTER,
            model=config['recommended_models']['openrouter_free'],
            api_key=os.getenv('OPENROUTER_API_KEY'),
            max_tokens=config['llm']['max_tokens'],
            temperature=config['llm']['temperature'],
            timeout=config['llm']['timeout']
        )
    
    # 2. Try local Ollama with appropriate model
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            
            # Find the best available model for our RAM
            recommended_model = get_recommended_ollama_model()
            available_model_names = [m['name'] for m in models]
            
            if recommended_model in available_model_names:
                return LLMConfig(
                    provider=LLMProvider.OLLAMA,
                    model=recommended_model,
                    base_url="http://localhost:11434",
                    max_tokens=config['llm']['max_tokens'],
                    temperature=config['llm']['temperature'],
                    timeout=config['llm']['timeout']
                )
            elif available_model_names:
                # Use any available model
                return LLMConfig(
                    provider=LLMProvider.OLLAMA,
                    model=available_model_names[0],
                    base_url="http://localhost:11434",
                    max_tokens=config['llm']['max_tokens'],
                    temperature=config['llm']['temperature'],
                    timeout=config['llm']['timeout']
                )
    except:
        pass
    
    # 3. Try local API servers
    for port in [8000, 5000, 7860]:
        try:
            import requests
            response = requests.get(f"http://localhost:{port}/v1/models", timeout=1)
            if response.status_code == 200:
                return LLMConfig(
                    provider=LLMProvider.LOCAL_API,
                    model="local-model",
                    base_url=f"http://localhost:{port}",
                    max_tokens=config['llm']['max_tokens'],
                    temperature=config['llm']['temperature'],
                    timeout=config['llm']['timeout']
                )
        except:
            continue
    
    raise Exception("No suitable LLM provider found for 4GB RAM system")


async def optimized_scrape(url: str, output_dir: str, use_llm: bool = True) -> dict:
    """Optimized scraping function for resource-constrained environment"""
    
    logger = logging.getLogger(__name__)
    config = get_dynamic_config()
    
    # Monitor resources before starting
    initial_status = check_system_resources()
    logger.info(f"🖥️  Starting with {initial_status['memory']['available_gb']:.1f}GB RAM available")
    
    try:
        if use_llm:
            # Get optimal LLM configuration
            llm_config = get_optimal_llm_config()
            logger.info(f"🤖 Using {llm_config.provider.value} with model {llm_config.model}")
            
            # Initialize scraper with resource limits
            scraper = EnhancedHealthcareScraper(url, output_dir, llm_config, use_llm=True)
            
            # Apply memory management
            if initial_status['memory']['used_percent'] > 70:
                logger.warning("⚠️  High memory usage detected - using conservative settings")
                # Force garbage collection
                gc.collect()
        
        else:
            # Traditional scraping with optimized settings
            scraper = HealthcareFacilityScraper(url, output_dir)
        
        # Run scraping with periodic resource monitoring
        facilities = []
        
        if use_llm:
            facilities = await scraper.scrape_with_llm_enhancement()
        else:
            facilities = await scraper.scrape_facilities()
        
        # Monitor resources after scraping
        final_status = check_system_resources()
        logger.info(f"🖥️  Completed with {final_status['memory']['available_gb']:.1f}GB RAM available")
        
        if facilities:
            # Save results with compression if needed
            if use_llm:
                traditional_scraper = HealthcareFacilityScraper(url, output_dir)
                traditional_scraper.facilities = facilities
                
                saved_files = []
                for format_type in ['json', 'csv']:
                    filename = traditional_scraper.save_results(format_type)
                    saved_files.append(filename)
                    logger.info(f"💾 Results saved to {filename}")
            else:
                saved_files = []
                for format_type in ['json', 'csv']:
                    filename = scraper.save_results(format_type)
                    saved_files.append(filename)
            
            # Force cleanup
            gc.collect()
            
            return {
                'success': True,
                'facilities_count': len(facilities),
                'files': saved_files,
                'memory_used_mb': (initial_status['memory']['available_gb'] - final_status['memory']['available_gb']) * 1024,
                'provider': llm_config.provider.value if use_llm else 'traditional'
            }
        
        else:
            return {
                'success': False,
                'error': 'No facilities found',
                'facilities_count': 0,
                'provider': llm_config.provider.value if use_llm else 'traditional'
            }
    
    except Exception as e:
        logger.error(f"💥 Scraping failed: {e}")
        
        # Check if it's a resource issue
        current_status = check_system_resources()
        if current_status['memory']['status'] == 'critical':
            logger.error("❌ Out of memory - try using a smaller model or traditional scraping")
        
        return {
            'success': False,
            'error': str(e),
            'facilities_count': 0,
            'provider': 'unknown'
        }


def print_optimization_guide():
    """Print optimization guide for 4GB RAM droplet"""
    
    print("\n🎯 Optimization Guide for Your 2 vCPU / 4GB RAM Droplet:")
    print("\n📊 Recommended LLM Models:")
    
    for level, info in OLLAMA_MODELS_BY_MEMORY.items():
        if info['ram_usage_gb'] <= 4.5:  # Only show models that fit
            status = "✅ RECOMMENDED" if info['ram_usage_gb'] <= 3 else "⚠️  MAXIMUM"
            print(f"   {status} {info['model']}")
            print(f"      RAM: {info['ram_usage_gb']}GB | Speed: {info['speed']} | Quality: {info['quality']}")
            print(f"      {info['description']}")
    
    print("\n🚀 Performance Tips:")
    for category, tips in OPTIMIZATION_TIPS.items():
        if category != 'network':  # Skip network tips since 4TB is plenty
            print(f"\n   {category.upper()}:")
            for tip in tips[:2]:  # Show top 2 tips
                print(f"   • {tip}")
    
    print("\n💰 Cost-Effective Strategies:")
    print("   • Use OpenRouter free models for best RAM efficiency")
    print("   • Use phi3:mini for fastest local processing")
    print("   • Process sites sequentially to avoid memory spikes")
    print("   • Your 4TB transfer limit allows ~1000+ healthcare sites/month")


def main():
    """Optimized main function for 4GB RAM droplet"""
    
    parser = argparse.ArgumentParser(
        description='Optimized Healthcare Facility Scraper for 2 vCPU / 4GB RAM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Optimized Examples for Your 4GB RAM Droplet:

  # Check system status and recommendations
  python main_optimized.py --status

  # Auto-optimized LLM scraping (recommended)
  python main_optimized.py --url https://lcca.com --output ./results --llm

  # Memory-efficient OpenRouter (best for 4GB)
  python main_optimized.py --url https://lcca.com --output ./results --provider openrouter

  # Local Ollama with small model
  python main_optimized.py --url https://lcca.com --output ./results --provider ollama --model phi3:mini

  # Traditional scraping (lowest memory usage)
  python main_optimized.py --url https://lcca.com --output ./results

  # Batch processing with memory management
  python main_optimized.py --urls-file sites.txt --output ./results --llm --sequential
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--url', '-u', help='Single URL to scrape')
    input_group.add_argument('--urls-file', '-f', help='File containing URLs to scrape')
    input_group.add_argument('--status', action='store_true', help='Check system status and optimization recommendations')
    
    # Output options
    parser.add_argument('--output', '-o', default='./output', help='Output directory')
    
    # LLM options optimized for 4GB RAM
    parser.add_argument('--llm', action='store_true', help='Use LLM extraction (auto-optimized for 4GB RAM)')
    parser.add_argument('--provider', choices=['auto', 'openrouter', 'ollama'], default='auto', help='LLM provider')
    parser.add_argument('--model', help='Specific model (will be validated against RAM limits)')
    parser.add_argument('--sequential', action='store_true', help='Process URLs sequentially (saves memory)')
    
    # System options
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
    parser.add_argument('--monitor', action='store_true', help='Enable resource monitoring during scraping')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Handle status check
    if args.status:
        print_system_status()
        print_optimization_guide()
        return
    
    # Require URL for scraping
    if not args.url and not args.urls_file:
        parser.error("Either --url or --urls-file is required")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Get URLs
    if args.url:
        urls = [args.url]
    else:
        try:
            with open(args.urls_file, 'r') as f:
                urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        except FileNotFoundError:
            logger.error(f"❌ File not found: {args.urls_file}")
            sys.exit(1)
    
    # Check system resources before starting
    print_system_status()
    
    # Validate model choice against RAM limits
    if args.model and args.provider == 'ollama':
        for level, info in OLLAMA_MODELS_BY_MEMORY.items():
            if info['model'] == args.model and info['ram_usage_gb'] > 4.5:
                logger.error(f"❌ Model {args.model} requires {info['ram_usage_gb']}GB RAM (you have 4GB)")
                logger.info("💡 Recommended models for 4GB: phi3:mini, llama3.2:3b")
                sys.exit(1)
    
    logger.info(f"🚀 Starting optimized scraper for {len(urls)} URLs")
    logger.info(f"📁 Output directory: {args.output}")
    logger.info(f"🔄 Processing mode: {'sequential' if args.sequential else 'optimized'}")
    
    # Process URLs
    try:
        total_facilities = 0
        
        for i, url in enumerate(urls, 1):
            logger.info(f"\n📋 Processing {i}/{len(urls)}: {url}")
            
            # Monitor resources if requested
            if args.monitor:
                status = check_system_resources()
                if status['memory']['status'] == 'warning':
                    logger.warning("⚠️  Memory usage high - forcing garbage collection")
                    gc.collect()
                elif status['memory']['status'] == 'critical':
                    logger.error("❌ Memory critical - skipping remaining URLs")
                    break
            
            # Run scraping
            result = asyncio.run(optimized_scrape(url, args.output, args.llm))
            
            if result['success']:
                total_facilities += result['facilities_count']
                print(f"✅ Found {result['facilities_count']} facilities")
                print(f"🤖 Provider: {result['provider']}")
                if 'memory_used_mb' in result:
                    print(f"💾 Memory used: {result['memory_used_mb']:.1f}MB")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
            
            # Cleanup between URLs if processing multiple
            if len(urls) > 1 and i < len(urls):
                gc.collect()
                if args.sequential:
                    import time
                    time.sleep(2)  # Brief pause for system recovery
        
        print(f"\n🎉 Completed! Total facilities found: {total_facilities}")
        
        # Final resource check
        if args.monitor:
            print("\n📊 Final System Status:")
            print_system_status()
    
    except KeyboardInterrupt:
        logger.info("\n⏹️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

