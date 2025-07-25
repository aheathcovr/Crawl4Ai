"""
Hybrid LLM Navigation System
Uses LLM intelligence for site navigation and section discovery,
then fast algorithms for data extraction
"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import aiohttp

from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy


@dataclass
class NavigationTarget:
    """Target page or section identified by LLM"""
    url: str
    page_type: str  # "facility_listing", "individual_facility", "directory", etc.
    confidence: float
    description: str
    expected_facility_count: Optional[int] = None
    css_selectors: Optional[List[str]] = None
    extraction_hints: Optional[Dict[str, Any]] = None


@dataclass
class SiteStructure:
    """Complete site structure analysis"""
    main_url: str
    navigation_targets: List[NavigationTarget]
    site_type: str  # "corporate_chain", "individual_facility", "directory"
    total_expected_facilities: int
    navigation_strategy: str
    analysis_confidence: float


class OpenRouterClient:
    """OpenRouter API client with model switching capabilities"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.session = None
        self.logger = logging.getLogger(__name__)
        
        # Model configurations optimized for different tasks
        self.models = {
            # Fast models for navigation
            "navigation": {
                "model": "meta-llama/llama-3.1-8b-instruct:free",
                "max_tokens": 1000,
                "temperature": 0.1,
                "description": "Fast, free model for site navigation"
            },
            
            # Balanced models for analysis
            "analysis": {
                "model": "microsoft/wizardlm-2-8x22b",
                "max_tokens": 2000,
                "temperature": 0.2,
                "description": "Balanced model for site structure analysis"
            },
            
            # Precise models for extraction
            "extraction": {
                "model": "anthropic/claude-3.5-sonnet",
                "max_tokens": 4000,
                "temperature": 0.0,
                "description": "High-precision model for data extraction"
            },
            
            # Fast extraction fallback
            "extraction_fast": {
                "model": "meta-llama/llama-3.1-70b-instruct",
                "max_tokens": 2000,
                "temperature": 0.1,
                "description": "Fast model for extraction fallback"
            }
        }
    
    async def _get_session(self):
        """Get or create HTTP session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def chat_completion(self, 
                            messages: List[Dict[str, str]], 
                            model_type: str = "analysis",
                            custom_model: str = None) -> Dict[str, Any]:
        """Make chat completion request to OpenRouter"""
        
        session = await self._get_session()
        
        # Use custom model or get from predefined types
        if custom_model:
            model_config = {"model": custom_model, "max_tokens": 2000, "temperature": 0.1}
        else:
            model_config = self.models.get(model_type, self.models["analysis"])
        
        payload = {
            "model": model_config["model"],
            "messages": messages,
            "max_tokens": model_config["max_tokens"],
            "temperature": model_config["temperature"]
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://healthcare-scraper.local",
            "X-Title": "Healthcare Facility Scraper"
        }
        
        try:
            async with session.post(f"{self.base_url}/chat/completions", 
                                  json=payload, 
                                  headers=headers) as response:
                
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "content": result["choices"][0]["message"]["content"],
                        "model": model_config["model"],
                        "usage": result.get("usage", {})
                    }
                else:
                    error_text = await response.text()
                    self.logger.error(f"OpenRouter API error {response.status}: {error_text}")
                    return {
                        "success": False,
                        "error": f"API error {response.status}: {error_text}"
                    }
                    
        except Exception as e:
            self.logger.error(f"OpenRouter request failed: {e}")
            return {
                "success": False,
                "error": f"Request failed: {e}"
            }
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    def list_available_models(self) -> Dict[str, Any]:
        """List available model configurations"""
        return {
            "predefined_types": list(self.models.keys()),
            "model_configs": self.models,
            "custom_models": [
                "meta-llama/llama-3.1-8b-instruct:free",
                "meta-llama/llama-3.1-70b-instruct",
                "anthropic/claude-3.5-sonnet",
                "openai/gpt-4o-mini",
                "google/gemini-pro-1.5",
                "microsoft/wizardlm-2-8x22b",
                "qwen/qwen-2-72b-instruct"
            ]
        }


class HybridLLMNavigator:
    """Hybrid system using LLM for navigation, algorithms for extraction"""
    
    def __init__(self, openrouter_api_key: str):
        self.openrouter = OpenRouterClient(openrouter_api_key)
        self.logger = logging.getLogger(__name__)
        
        # Navigation patterns for different healthcare site types
        self.site_patterns = {
            "corporate_chain": {
                "indicators": ["locations", "communities", "facilities", "find a location"],
                "navigation_selectors": [
                    "a[href*='location']", "a[href*='facilities']", 
                    "a[href*='communities']", "a[href*='find']"
                ]
            },
            "individual_facility": {
                "indicators": ["about us", "services", "contact", "staff"],
                "navigation_selectors": [
                    ".facility-info", ".contact-info", ".services"
                ]
            },
            "directory": {
                "indicators": ["search", "directory", "browse", "filter"],
                "navigation_selectors": [
                    ".search-form", ".directory-list", ".facility-grid"
                ]
            }
        }
    
    async def analyze_site_structure(self, url: str) -> SiteStructure:
        """Analyze site structure using LLM intelligence"""
        
        self.logger.info(f"🧠 Analyzing site structure: {url}")
        
        async with AsyncWebCrawler(headless=True, verbose=False) as crawler:
            # Get initial page content
            result = await crawler.arun(url=url)
            
            if not result.success:
                raise Exception(f"Failed to load {url}: {result.error_message}")
            
            # Extract navigation and content structure
            html_content = result.html
            page_text = getattr(result, 'text', getattr(result, 'markdown', result.html))[:8000]  # Limit for LLM processing
            
            # Use LLM to analyze site structure
            analysis_prompt = f"""
Analyze this healthcare website to understand its structure and identify where facility information is located.

Website URL: {url}
Page Content: {page_text}

Please analyze and provide a JSON response with:
1. site_type: "corporate_chain", "individual_facility", or "directory"
2. navigation_targets: List of URLs/sections where facility data might be found
3. expected_facility_count: Estimated number of facilities
4. navigation_strategy: How to best navigate this site
5. confidence: Your confidence in this analysis (0-1)

For each navigation target, include:
- url: Full URL to the target page
- page_type: Type of page ("facility_listing", "individual_facility", etc.)
- confidence: Confidence this page contains facility data (0-1)
- description: Brief description of what's expected on this page
- css_selectors: Suggested CSS selectors for facility containers

Focus on finding pages that list multiple healthcare facilities or contain detailed facility information.

Return only valid JSON.
"""
            
            messages = [
                {"role": "system", "content": "You are an expert web scraping analyst specializing in healthcare websites. Analyze site structures to identify where facility data is located."},
                {"role": "user", "content": analysis_prompt}
            ]
            
            llm_response = await self.openrouter.chat_completion(messages, model_type="analysis")
            
            if not llm_response["success"]:
                raise Exception(f"LLM analysis failed: {llm_response['error']}")
            
            try:
                analysis_data = json.loads(llm_response["content"])
                
                # Convert to NavigationTarget objects
                navigation_targets = []
                for target_data in analysis_data.get("navigation_targets", []):
                    # Resolve relative URLs
                    target_url = urljoin(url, target_data["url"])
                    
                    target = NavigationTarget(
                        url=target_url,
                        page_type=target_data.get("page_type", "unknown"),
                        confidence=target_data.get("confidence", 0.5),
                        description=target_data.get("description", ""),
                        expected_facility_count=target_data.get("expected_facility_count"),
                        css_selectors=target_data.get("css_selectors", [])
                    )
                    navigation_targets.append(target)
                
                site_structure = SiteStructure(
                    main_url=url,
                    navigation_targets=navigation_targets,
                    site_type=analysis_data.get("site_type", "unknown"),
                    total_expected_facilities=analysis_data.get("expected_facility_count", 0),
                    navigation_strategy=analysis_data.get("navigation_strategy", ""),
                    analysis_confidence=analysis_data.get("confidence", 0.5)
                )
                
                self.logger.info(f"✅ Site analysis complete: {site_structure.site_type}, {len(navigation_targets)} targets found")
                return site_structure
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse LLM response: {e}")
                # Fallback to pattern-based analysis
                return await self._fallback_site_analysis(url, html_content)
    
    async def _fallback_site_analysis(self, url: str, html_content: str) -> SiteStructure:
        """Fallback site analysis using pattern matching"""
        
        self.logger.info("🔄 Using fallback pattern-based analysis")
        
        # Detect site type based on content patterns
        site_type = "unknown"
        for pattern_type, patterns in self.site_patterns.items():
            for indicator in patterns["indicators"]:
                if indicator.lower() in html_content.lower():
                    site_type = pattern_type
                    break
            if site_type != "unknown":
                break
        
        # Find navigation links
        navigation_targets = []
        
        # Extract links that might lead to facility pages
        link_patterns = [
            r'href="([^"]*(?:location|facility|community|center)[^"]*)"',
            r'href="([^"]*(?:find|search|directory)[^"]*)"'
        ]
        
        found_urls = set()
        for pattern in link_patterns:
            matches = re.findall(pattern, html_content, re.IGNORECASE)
            for match in matches:
                full_url = urljoin(url, match)
                if full_url not in found_urls:
                    found_urls.add(full_url)
                    
                    target = NavigationTarget(
                        url=full_url,
                        page_type="facility_listing",
                        confidence=0.6,
                        description=f"Potential facility page: {match}"
                    )
                    navigation_targets.append(target)
        
        return SiteStructure(
            main_url=url,
            navigation_targets=navigation_targets,
            site_type=site_type,
            total_expected_facilities=len(navigation_targets) * 10,  # Rough estimate
            navigation_strategy="pattern_based_fallback",
            analysis_confidence=0.4
        )
    
    async def discover_facility_sections(self, url: str, html_content: str = None) -> List[Dict[str, Any]]:
        """Use LLM to discover specific sections containing facility data"""
        
        self.logger.info(f"🔍 Discovering facility sections: {url}")
        
        if not html_content:
            async with AsyncWebCrawler(headless=True, verbose=False) as crawler:
                result = await crawler.arun(url=url)
                if not result.success:
                    return []
                html_content = result.html
        
        # Use LLM to identify facility data sections
        discovery_prompt = f"""
Analyze this healthcare webpage to identify specific sections that contain facility information.

URL: {url}
HTML Content: {html_content[:6000]}

Find sections that contain:
- Facility names
- Addresses
- Phone numbers
- Services offered
- Administrator information

For each section found, provide:
1. css_selector: CSS selector to target this section
2. data_type: Type of data in this section
3. confidence: How confident you are this contains facility data (0-1)
4. extraction_hints: Specific selectors for individual data fields

Return a JSON array of sections found.
"""
        
        messages = [
            {"role": "system", "content": "You are an expert at analyzing HTML to find structured data sections. Focus on identifying precise CSS selectors for healthcare facility information."},
            {"role": "user", "content": discovery_prompt}
        ]
        
        llm_response = await self.openrouter.chat_completion(messages, model_type="navigation")
        
        if llm_response["success"]:
            try:
                sections = json.loads(llm_response["content"])
                self.logger.info(f"✅ Found {len(sections)} facility sections")
                return sections
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse section discovery response")
        
        return []
    
    async def generate_extraction_schema(self, url: str, sample_html: str) -> Dict[str, Any]:
        """Generate optimized extraction schema using LLM analysis"""
        
        self.logger.info(f"🛠️ Generating extraction schema for: {url}")
        
        schema_prompt = f"""
Create an optimized Crawl4AI extraction schema for this healthcare facility webpage.

URL: {url}
Sample HTML: {sample_html[:4000]}

Generate a JSON CSS extraction schema with:
1. baseSelector: CSS selector for facility containers
2. fields: Array of field definitions for facility data

Each field should have:
- name: Field name (facility_name, address, phone, etc.)
- selector: CSS selector relative to baseSelector
- type: "text", "attribute", or "html"
- attribute: If type is "attribute", specify which attribute
- multiple: true if this field can have multiple values

Focus on extracting:
- Facility name
- Address components (street, city, state, zip)
- Phone number
- Email
- Website
- Administrator/contact person
- Services offered
- Facility type

Return only the JSON schema, no explanation.
"""
        
        messages = [
            {"role": "system", "content": "You are an expert at creating Crawl4AI extraction schemas. Generate precise, working schemas for healthcare facility data."},
            {"role": "user", "content": schema_prompt}
        ]
        
        llm_response = await self.openrouter.chat_completion(messages, model_type="analysis")
        
        if llm_response["success"]:
            try:
                schema = json.loads(llm_response["content"])
                self.logger.info("✅ Generated extraction schema")
                return schema
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse schema generation response")
        
        # Fallback to default schema
        return self._get_default_schema()
    
    def _get_default_schema(self) -> Dict[str, Any]:
        """Default extraction schema as fallback"""
        return {
            "name": "facilities",
            "baseSelector": ".facility, .location, .community, .center, .card",
            "fields": [
                {"name": "facility_name", "selector": "h1, h2, h3, .name, .title", "type": "text"},
                {"name": "address", "selector": ".address, .street", "type": "text"},
                {"name": "city", "selector": ".city", "type": "text"},
                {"name": "state", "selector": ".state", "type": "text"},
                {"name": "zip_code", "selector": ".zip, .postal", "type": "text"},
                {"name": "phone", "selector": ".phone, a[href^='tel:']", "type": "text"},
                {"name": "email", "selector": ".email, a[href^='mailto:']", "type": "text"},
                {"name": "website", "selector": "a[href^='http']", "type": "attribute", "attribute": "href"}
            ]
        }
    
    async def smart_navigation_discovery(self, url: str) -> List[str]:
        """Discover additional facility pages through smart navigation"""
        
        self.logger.info(f"🗺️ Smart navigation discovery: {url}")
        
        async with AsyncWebCrawler(headless=True, verbose=False) as crawler:
            result = await crawler.arun(url=url)
            
            if not result.success:
                return []
            
            # Use LLM to find navigation patterns
            nav_prompt = f"""
Analyze this healthcare website's navigation to find all pages that might contain facility information.

URL: {url}
Page Content: {getattr(result, 'text', getattr(result, 'markdown', result.html))[:4000]}

Look for:
1. Navigation menus with facility-related links
2. Pagination for facility listings
3. State/region-based facility pages
4. Search or filter functionality
5. Sitemap references

Return a JSON array of URLs that likely contain facility data. Include:
- url: Full URL
- reason: Why this URL likely contains facility data
- priority: 1-5 (1=highest priority)

Focus on finding comprehensive facility listings rather than individual facility pages.
"""
            
            messages = [
                {"role": "system", "content": "You are an expert at website navigation analysis. Find all possible sources of healthcare facility data."},
                {"role": "user", "content": nav_prompt}
            ]
            
            llm_response = await self.openrouter.chat_completion(messages, model_type="navigation")
            
            discovered_urls = []
            
            if llm_response["success"]:
                try:
                    nav_data = json.loads(llm_response["content"])
                    for item in nav_data:
                        full_url = urljoin(url, item["url"])
                        discovered_urls.append(full_url)
                except json.JSONDecodeError:
                    self.logger.warning("Failed to parse navigation discovery response")
            
            self.logger.info(f"✅ Discovered {len(discovered_urls)} additional URLs")
            return discovered_urls
    
    async def close(self):
        """Close all connections"""
        await self.openrouter.close()


# Usage example and testing
async def test_hybrid_navigator():
    """Test the hybrid navigation system"""
    
    # Initialize with your OpenRouter API key
    navigator = HybridLLMNavigator("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")
    
    try:
        # Test site structure analysis
        test_url = "https://sunriseseniorliving.com"
        
        print(f"Testing site structure analysis for: {test_url}")
        site_structure = await navigator.analyze_site_structure(test_url)
        
        print(f"Site Type: {site_structure.site_type}")
        print(f"Expected Facilities: {site_structure.total_expected_facilities}")
        print(f"Navigation Targets: {len(site_structure.navigation_targets)}")
        
        for target in site_structure.navigation_targets[:3]:  # Show first 3
            print(f"  - {target.url} ({target.confidence:.2f} confidence)")
        
        # Test section discovery
        if site_structure.navigation_targets:
            target_url = site_structure.navigation_targets[0].url
            print(f"\nTesting section discovery for: {target_url}")
            
            sections = await navigator.discover_facility_sections(target_url)
            print(f"Found {len(sections)} facility sections")
            
            for section in sections[:2]:  # Show first 2
                print(f"  - {section.get('css_selector', 'N/A')} ({section.get('confidence', 0):.2f} confidence)")
        
        # Test model switching
        print(f"\nAvailable models:")
        models = navigator.openrouter.list_available_models()
        for model_type, config in models["model_configs"].items():
            print(f"  - {model_type}: {config['model']} ({config['description']})")
    
    finally:
        await navigator.close()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run test
    asyncio.run(test_hybrid_navigator())

