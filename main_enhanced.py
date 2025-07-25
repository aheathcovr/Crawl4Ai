#!/usr/bin/env python3
"""
Enhanced Healthcare Facility Scraper with Flexible LLM Support
Supports OpenRouter, Ollama, local models, and traditional extraction
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import List

from llm_providers import (
    LLMConfig, LLMProvider, LLMProviderFactory,
    setup_openrouter, setup_ollama, setup_local_api,
    test_provider
)
from llm_extractors_enhanced import EnhancedHealthcareScraper
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


async def auto_detect_llm() -> LLMConfig:
    """Auto-detect the best available LLM provider"""
    logger = logging.getLogger(__name__)
    
    # Test providers in order of preference
    test_configs = [
        # OpenRouter (free models available)
        setup_openrouter("meta-llama/llama-3.1-8b-instruct:free"),
        setup_openrouter("google/gemma-2-9b-it:free"),
        
        # Local Ollama
        setup_ollama("llama3.1:8b"),
        setup_ollama("llama3.2:3b"),
        setup_ollama("phi3:mini"),
        
        # Local API servers
        setup_local_api("local-model", "http://localhost:8000"),
        setup_local_api("local-model", "http://localhost:5000"),
        setup_local_api("local-model", "http://localhost:7860"),
    ]
    
    for config in test_configs:
        logger.info(f"🔍 Testing {config.provider.value} with model {config.model}...")
        
        if await test_provider(config):
            logger.info(f"✅ Found working provider: {config.provider.value} - {config.model}")
            return config
        else:
            logger.debug(f"❌ Provider {config.provider.value} not available")
    
    raise Exception("No working LLM provider found. Please set up OpenRouter, Ollama, or another provider.")


async def scrape_with_enhanced_llm(url: str, output_dir: str, llm_config: LLMConfig) -> dict:
    """Scrape using enhanced LLM extraction"""
    logger = logging.getLogger(__name__)
    logger.info(f"🧠 Starting enhanced LLM scraping of {url}")
    logger.info(f"🤖 Using provider: {llm_config.provider.value} - {llm_config.model}")
    
    try:
        # Initialize enhanced scraper
        scraper = EnhancedHealthcareScraper(url, output_dir, llm_config, use_llm=True)
        
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
                'facilities': facilities,
                'provider': f"{llm_config.provider.value} - {llm_config.model}"
            }
        else:
            logger.warning(f"⚠️  No facilities found for {url}")
            return {
                'success': False,
                'error': 'No facilities found',
                'facilities_count': 0,
                'provider': f"{llm_config.provider.value} - {llm_config.model}"
            }
            
    except Exception as e:
        logger.error(f"💥 Error in enhanced LLM scraping: {e}")
        return {
            'success': False,
            'error': str(e),
            'facilities_count': 0,
            'provider': f"{llm_config.provider.value} - {llm_config.model}"
        }


async def setup_ollama_model(model_name: str) -> bool:
    """Setup Ollama model if not available"""
    logger = logging.getLogger(__name__)
    
    try:
        from llm_providers import OllamaProvider
        
        config = setup_ollama(model_name)
        provider = OllamaProvider(config)
        
        if not provider.is_available():
            logger.error("❌ Ollama is not running. Please start Ollama first:")
            logger.info("💡 Install: curl -fsSL https://ollama.ai/install.sh | sh")
            logger.info("💡 Start: ollama serve")
            return False
        
        # Check if model is available
        available_models = provider.get_available_models()
        if model_name not in available_models:
            logger.info(f"📥 Model {model_name} not found locally. Pulling...")
            if provider.pull_model(model_name):
                logger.info(f"✅ Successfully pulled {model_name}")
                return True
            else:
                logger.error(f"❌ Failed to pull {model_name}")
                return False
        else:
            logger.info(f"✅ Model {model_name} already available")
            return True
    
    except Exception as e:
        logger.error(f"Error setting up Ollama: {e}")
        return False


def print_provider_status():
    """Print status of available LLM providers"""
    print("\n🔍 Checking LLM Provider Status:")
    
    # Check OpenRouter
    if os.getenv('OPENROUTER_API_KEY'):
        print("✅ OpenRouter: API key found")
        print("   💡 Free models: meta-llama/llama-3.1-8b-instruct:free, google/gemma-2-9b-it:free")
    else:
        print("❌ OpenRouter: No API key (set OPENROUTER_API_KEY)")
        print("   💡 Get key at: https://openrouter.ai/")
    
    # Check Ollama
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama: Running with {len(models)} models")
            if models:
                print(f"   📋 Available: {', '.join([m['name'] for m in models[:3]])}")
        else:
            print("❌ Ollama: Not responding")
    except:
        print("❌ Ollama: Not running")
        print("   💡 Install: curl -fsSL https://ollama.ai/install.sh | sh")
        print("   💡 Start: ollama serve")
    
    # Check local APIs
    local_ports = [8000, 5000, 7860]
    for port in local_ports:
        try:
            import requests
            response = requests.get(f"http://localhost:{port}/v1/models", timeout=1)
            if response.status_code == 200:
                print(f"✅ Local API: Found on port {port}")
                break
        except:
            continue
    else:
        print("❌ Local API: No servers found on common ports")
    
    print()


def main():
    """Enhanced CLI interface with flexible LLM provider support"""
    parser = argparse.ArgumentParser(
        description='Enhanced Healthcare Facility Scraper with Flexible LLM Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect best LLM provider
  python main_enhanced.py --url https://lcca.com --output ./results --llm

  # Use specific OpenRouter model
  python main_enhanced.py --url https://lcca.com --output ./results --provider openrouter --model meta-llama/llama-3.1-8b-instruct:free

  # Use local Ollama
  python main_enhanced.py --url https://lcca.com --output ./results --provider ollama --model llama3.1:8b

  # Use local API server
  python main_enhanced.py --url https://lcca.com --output ./results --provider local --base-url http://localhost:8000

  # Traditional scraping (no LLM)
  python main_enhanced.py --url https://lcca.com --output ./results

  # Check provider status
  python main_enhanced.py --status

Environment Variables:
  OPENROUTER_API_KEY    Your OpenRouter API key (for cloud models)
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        '--url', '-u',
        help='Single URL to scrape'
    )
    input_group.add_argument(
        '--urls-file', '-f',
        help='File containing URLs to scrape (one per line)'
    )
    input_group.add_argument(
        '--status',
        action='store_true',
        help='Check status of available LLM providers'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        default='./output',
        help='Output directory for results (default: ./output)'
    )
    
    # LLM provider options
    parser.add_argument(
        '--llm',
        action='store_true',
        help='Use LLM-enhanced extraction (auto-detect provider)'
    )
    parser.add_argument(
        '--provider',
        choices=['openrouter', 'ollama', 'local', 'auto'],
        default='auto',
        help='LLM provider to use (default: auto-detect)'
    )
    parser.add_argument(
        '--model',
        help='Specific model to use (provider-dependent)'
    )
    parser.add_argument(
        '--base-url',
        help='Base URL for local API providers'
    )
    parser.add_argument(
        '--api-key',
        help='API key for cloud providers'
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
    
    # Handle status check
    if args.status:
        print_provider_status()
        return
    
    # Require URL input for scraping
    if not args.url and not args.urls_file:
        parser.error("Either --url or --urls-file is required for scraping")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
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
    
    # Setup LLM configuration if requested
    llm_config = None
    if args.llm or args.provider != 'auto':
        try:
            if args.provider == 'auto' or (args.llm and args.provider == 'auto'):
                logger.info("🔍 Auto-detecting LLM provider...")
                llm_config = asyncio.run(auto_detect_llm())
            
            elif args.provider == 'openrouter':
                model = args.model or "meta-llama/llama-3.1-8b-instruct:free"
                api_key = args.api_key or os.getenv('OPENROUTER_API_KEY')
                if not api_key:
                    logger.error("❌ OpenRouter API key required. Set OPENROUTER_API_KEY or use --api-key")
                    sys.exit(1)
                llm_config = setup_openrouter(model, api_key)
            
            elif args.provider == 'ollama':
                model = args.model or "llama3.1:8b"
                base_url = args.base_url or "http://localhost:11434"
                
                # Setup Ollama model if needed
                if not asyncio.run(setup_ollama_model(model)):
                    logger.error("❌ Failed to setup Ollama model")
                    sys.exit(1)
                
                llm_config = setup_ollama(model, base_url)
            
            elif args.provider == 'local':
                model = args.model or "local-model"
                base_url = args.base_url or "http://localhost:8000"
                llm_config = setup_local_api(model, base_url)
            
            # Test the configuration
            if llm_config and not asyncio.run(test_provider(llm_config)):
                logger.error(f"❌ LLM provider {llm_config.provider.value} is not working")
                logger.info("💡 Run with --status to check available providers")
                sys.exit(1)
        
        except Exception as e:
            logger.error(f"❌ Failed to setup LLM provider: {e}")
            logger.info("💡 Run with --status to check available providers")
            sys.exit(1)
    
    # Run the scraper
    try:
        if llm_config:
            # LLM-enhanced mode
            for url in urls:
                logger.info(f"\n🧠 Enhanced LLM scraping: {url}")
                result = asyncio.run(scrape_with_enhanced_llm(url, args.output, llm_config))
                
                if result['success']:
                    print(f"\n✅ Successfully scraped {result['facilities_count']} facilities")
                    print(f"🤖 Provider: {result['provider']}")
                    print(f"📁 Results saved to: {', '.join(result['files'])}")
                else:
                    print(f"\n❌ Scraping failed: {result.get('error', 'Unknown error')}")
                    print(f"🤖 Provider: {result['provider']}")
        
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

