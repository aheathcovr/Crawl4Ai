"""
Enhanced LLM-powered extractors using flexible provider system
Supports OpenRouter, Ollama, and other local/cloud providers
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy, JsonCssExtractionStrategy
from crawl4ai.chunking_strategy import RegexChunking

from healthcare_scraper import FacilityInfo
from llm_providers import (
    LLMConfig, LLMProvider, LLMProviderFactory, MultiProviderLLM,
    setup_openrouter, setup_ollama, setup_local_api
)


class FlexibleLLMExtractor:
    """LLM extractor that works with multiple providers"""
    
    def __init__(self, llm_config: LLMConfig, fallback_configs: Optional[List[LLMConfig]] = None):
        self.llm = MultiProviderLLM(llm_config, fallback_configs)
        self.logger = logging.getLogger(__name__)
        
        # Define the extraction schema for facilities
        self.facility_schema = {
            "type": "object",
            "properties": {
                "facilities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Official facility name"},
                            "facility_type": {"type": "string", "description": "Type of healthcare facility (Skilled Nursing, Assisted Living, Memory Care, etc.)"},
                            "address": {"type": "string", "description": "Complete street address"},
                            "city": {"type": "string", "description": "City name"},
                            "state": {"type": "string", "description": "State abbreviation (e.g., CA, NY)"},
                            "zip_code": {"type": "string", "description": "ZIP or postal code"},
                            "phone": {"type": "string", "description": "Primary phone number"},
                            "email": {"type": "string", "description": "Contact email address"},
                            "website": {"type": "string", "description": "Facility website URL"},
                            "administrator": {"type": "string", "description": "Administrator or director name"},
                            "beds": {"type": "string", "description": "Number of beds or capacity"},
                            "services_offered": {"type": "array", "items": {"type": "string"}, "description": "List of services offered"},
                            "specialties": {"type": "array", "items": {"type": "string"}, "description": "Medical specialties or special programs"},
                            "amenities": {"type": "array", "items": {"type": "string"}, "description": "Facility amenities and features"},
                            "description": {"type": "string", "description": "Brief facility description"},
                            "license_number": {"type": "string", "description": "State license number"},
                            "medicare_provider_id": {"type": "string", "description": "Medicare provider ID"},
                            "accreditation": {"type": "string", "description": "Accreditation information"},
                            "visiting_hours": {"type": "string", "description": "Visiting hours information"}
                        },
                        "required": ["name", "facility_type"]
                    }
                }
            },
            "required": ["facilities"]
        }
    
    def create_facility_extraction_prompt(self, content: str) -> str:
        """Create extraction prompt for healthcare facilities"""
        
        prompt = f"""
You are an expert at extracting healthcare facility information from web pages.

TASK: Extract detailed information about healthcare facilities from the provided content.

FACILITY TYPES TO LOOK FOR:
- Skilled Nursing Facilities
- Assisted Living Communities  
- Memory Care Centers
- Continuing Care Retirement Communities (CCRC)
- Rehabilitation Centers
- Long-term Care Facilities
- Senior Living Communities

EXTRACTION GUIDELINES:
1. Extract ONLY actual healthcare facilities, not corporate offices or general pages
2. For addresses, extract complete street addresses, separate city/state/zip
3. Normalize phone numbers to (XXX) XXX-XXXX format when possible
4. For facility types, use standardized terms (e.g., "Skilled Nursing" not "skilled nursing facility")
5. Extract services as specific, actionable items (e.g., "Physical Therapy", "Memory Care")
6. Include license numbers, Medicare IDs, and accreditation info when available
7. If multiple facilities are on one page, extract each separately
8. Skip duplicate entries

QUALITY REQUIREMENTS:
- Facility name must be specific (not generic like "Our Facility")
- Address should be complete and valid
- Phone numbers should be properly formatted
- Services should be specific healthcare services, not marketing language

CONTENT TO ANALYZE:
{content[:8000]}  # Limit content to avoid token limits

Please respond with a JSON object containing an array of facilities. Each facility should have the fields: name, facility_type, address, city, state, zip_code, phone, email, website, administrator, beds, services_offered, specialties, amenities, description, license_number, medicare_provider_id, accreditation, visiting_hours.

If no valid facilities are found, return {{"facilities": []}}.

JSON Response:
"""
        return prompt
    
    def create_facility_listing_prompt(self, content: str) -> str:
        """Create prompt for finding facility listing pages"""
        
        prompt = f"""
You are tasked with finding links to individual healthcare facility pages.

TASK: Extract URLs and basic information for individual healthcare facilities from this listing page.

LOOK FOR:
- Links to individual facility detail pages
- Facility names and locations
- URLs that lead to specific facility information (not general corporate pages)

INCLUDE:
- Direct links to facility pages (e.g., /locations/facility-name, /communities/facility-name)
- Facility names as they appear on the page
- Location information (city, state) when available

EXCLUDE:
- Corporate "About Us" pages
- General service pages
- Contact forms or general inquiry pages
- Duplicate entries

CONTENT TO ANALYZE:
{content[:8000]}

Please respond with a JSON object containing an array of facility links. Each link should have: url, name, location, facility_type (if mentioned).

JSON Response:
"""
        return prompt
    
    async def extract_facilities_from_content(self, content: str, source_url: str) -> List[FacilityInfo]:
        """Extract facility information from content using LLM"""
        
        facilities = []
        
        try:
            prompt = self.create_facility_extraction_prompt(content)
            response = await self.llm.generate(prompt, self.facility_schema)
            
            # Parse the LLM response
            try:
                # Clean response - sometimes LLMs add extra text
                response = response.strip()
                if response.startswith('```json'):
                    response = response[7:]
                if response.endswith('```'):
                    response = response[:-3]
                
                extracted_data = json.loads(response)
                
                if 'facilities' in extracted_data:
                    for facility_data in extracted_data['facilities']:
                        facility = self._convert_to_facility_info(facility_data, source_url)
                        if facility and self._is_valid_facility(facility):
                            facilities.append(facility)
                            self.logger.info(f"LLM extracted facility: {facility.name}")
            
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse LLM response: {e}")
                self.logger.debug(f"Raw response: {response[:500]}...")
                # Try to extract JSON from response
                facilities.extend(self._extract_json_from_text(response, source_url))
        
        except Exception as e:
            self.logger.error(f"Error in LLM extraction: {e}")
        
        return facilities
    
    async def find_facility_links_from_content(self, content: str, base_url: str) -> List[Dict[str, str]]:
        """Use LLM to find facility listing links from content"""
        
        facility_links = []
        
        try:
            prompt = self.create_facility_listing_prompt(content)
            response = await self.llm.generate(prompt)
            
            # Parse response
            try:
                response = response.strip()
                if response.startswith('```json'):
                    response = response[7:]
                if response.endswith('```'):
                    response = response[:-3]
                
                extracted_data = json.loads(response)
                
                if 'facility_links' in extracted_data:
                    for link_data in extracted_data['facility_links']:
                        # Ensure URL is absolute
                        if link_data.get('url'):
                            if not link_data['url'].startswith('http'):
                                from urllib.parse import urljoin
                                link_data['url'] = urljoin(base_url, link_data['url'])
                            
                            facility_links.append(link_data)
                            self.logger.info(f"LLM found facility link: {link_data['name']} - {link_data['url']}")
            
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse facility links response: {e}")
        
        except Exception as e:
            self.logger.error(f"Error in LLM facility link extraction: {e}")
        
        return facility_links
    
    def _extract_json_from_text(self, text: str, source_url: str) -> List[FacilityInfo]:
        """Try to extract JSON from malformed response"""
        facilities = []
        
        # Look for JSON-like structures
        json_pattern = r'\{[^{}]*"facilities"[^{}]*\[[^\]]*\][^{}]*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                data = json.loads(match)
                if 'facilities' in data:
                    for facility_data in data['facilities']:
                        facility = self._convert_to_facility_info(facility_data, source_url)
                        if facility and self._is_valid_facility(facility):
                            facilities.append(facility)
            except:
                continue
        
        return facilities
    
    def _convert_to_facility_info(self, facility_data: Dict[str, Any], source_url: str) -> Optional[FacilityInfo]:
        """Convert LLM extracted data to FacilityInfo object"""
        
        try:
            facility = FacilityInfo(source_url=source_url)
            
            # Map LLM extracted fields to FacilityInfo fields
            facility.name = facility_data.get('name', '').strip()
            facility.facility_type = facility_data.get('facility_type', '').strip()
            facility.address = facility_data.get('address', '').strip()
            facility.city = facility_data.get('city', '').strip()
            facility.state = facility_data.get('state', '').strip()
            facility.zip_code = facility_data.get('zip_code', '').strip()
            facility.phone = self._clean_phone(facility_data.get('phone', ''))
            facility.email = facility_data.get('email', '').strip()
            facility.website = facility_data.get('website', '').strip()
            facility.administrator = facility_data.get('administrator', '').strip()
            facility.beds = facility_data.get('beds', '').strip()
            facility.description = facility_data.get('description', '').strip()
            facility.license_number = facility_data.get('license_number', '').strip()
            facility.medicare_provider_id = facility_data.get('medicare_provider_id', '').strip()
            facility.visiting_hours = facility_data.get('visiting_hours', '').strip()
            
            # Handle arrays
            facility.services_offered = facility_data.get('services_offered', [])
            facility.specialties = facility_data.get('specialties', [])
            facility.amenities = facility_data.get('amenities', [])
            
            return facility
            
        except Exception as e:
            self.logger.error(f"Error converting LLM data to FacilityInfo: {e}")
            return None
    
    def _is_valid_facility(self, facility: FacilityInfo) -> bool:
        """Validate that the extracted facility has minimum required information"""
        
        # Must have a name
        if not facility.name or len(facility.name.strip()) < 3:
            return False
        
        # Skip generic names
        generic_names = ['our facility', 'healthcare facility', 'nursing home', 'assisted living']
        if facility.name.lower().strip() in generic_names:
            return False
        
        # Must have some location information
        if not any([facility.address, facility.city, facility.state]):
            return False
        
        return True
    
    def _clean_phone(self, phone: str) -> str:
        """Clean and format phone number"""
        if not phone:
            return ""
        
        # Remove all non-digit characters except +
        cleaned = re.sub(r'[^\d+]', '', phone)
        
        # Format US phone numbers
        if len(cleaned) == 10:
            return f"({cleaned[:3]}) {cleaned[3:6]}-{cleaned[6:]}"
        elif len(cleaned) == 11 and cleaned.startswith('1'):
            return f"({cleaned[1:4]}) {cleaned[4:7]}-{cleaned[7:]}"
        
        return cleaned


class SmartWebsiteAnalyzer:
    """LLM-powered website structure analyzer"""
    
    def __init__(self, llm_config: LLMConfig):
        self.llm = MultiProviderLLM(llm_config)
        self.logger = logging.getLogger(__name__)
    
    async def analyze_website_structure(self, content: str, base_url: str) -> Dict[str, Any]:
        """Use LLM to understand website structure and find facility sections"""
        
        prompt = f"""
Analyze this healthcare website's structure to understand how facilities are organized.

TASK: Identify sections that contain individual healthcare facility information.

LOOK FOR:
- "Locations" or "Our Locations" sections
- "Communities" or "Our Communities" 
- "Facilities" or "Care Centers"
- "Find a Location" or "Facility Locator"
- State-by-state facility listings
- Individual facility pages

ANALYZE:
- Navigation menu structure
- How facilities are organized (by state, by type, alphabetically)
- Whether there's a central directory or individual pages
- Estimated number of facilities based on visible links

WEBSITE CONTENT:
{content[:6000]}

Please respond with a JSON object containing:
- facility_sections: array of sections with section_name, url, description, likely_facility_count
- navigation_pattern: description of how navigation works
- site_type: corporate chain, single facility, regional network, etc.
- recommended_strategy: how to best crawl this site

JSON Response:
"""
        
        try:
            response = await self.llm.generate(prompt)
            
            # Clean and parse response
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            
            return json.loads(response)
        
        except Exception as e:
            self.logger.error(f"Error analyzing website structure: {e}")
            return {
                "facility_sections": [],
                "navigation_pattern": "unknown",
                "site_type": "unknown",
                "recommended_strategy": "fallback to traditional crawling"
            }


class EnhancedHealthcareScraper:
    """Enhanced scraper using flexible LLM providers"""
    
    def __init__(self, base_url: str, output_dir: str, llm_config: LLMConfig, use_llm: bool = True):
        self.base_url = base_url
        self.output_dir = output_dir
        self.use_llm = use_llm
        self.logger = logging.getLogger(__name__)
        
        if use_llm:
            # Setup with fallback providers
            fallback_configs = []
            
            # Add OpenRouter as fallback if available
            if llm_config.provider != LLMProvider.OPENROUTER and os.getenv('OPENROUTER_API_KEY'):
                fallback_configs.append(setup_openrouter())
            
            # Add Ollama as fallback if available
            if llm_config.provider != LLMProvider.OLLAMA:
                try:
                    import requests
                    requests.get("http://localhost:11434/api/tags", timeout=2)
                    fallback_configs.append(setup_ollama())
                except:
                    pass
            
            self.llm_extractor = FlexibleLLMExtractor(llm_config, fallback_configs)
            self.website_analyzer = SmartWebsiteAnalyzer(llm_config)
    
    async def scrape_with_llm_enhancement(self) -> List[FacilityInfo]:
        """Run enhanced scraping with flexible LLM providers"""
        
        all_facilities = []
        
        async with AsyncWebCrawler(headless=True) as crawler:
            
            if self.use_llm:
                # Step 1: Get main page content
                self.logger.info("🧠 Analyzing website structure with LLM...")
                result = await crawler.arun(url=self.base_url)
                
                if result.success:
                    # Analyze structure
                    structure = await self.website_analyzer.analyze_website_structure(
                        result.html, self.base_url
                    )
                    
                    self.logger.info(f"Site type: {structure.get('site_type', 'unknown')}")
                    self.logger.info(f"Navigation pattern: {structure.get('navigation_pattern', 'unknown')}")
                    
                    # Extract facilities from main page
                    main_facilities = await self.llm_extractor.extract_facilities_from_content(
                        result.html, self.base_url
                    )
                    all_facilities.extend(main_facilities)
                    
                    # Find facility links
                    facility_links = await self.llm_extractor.find_facility_links_from_content(
                        result.html, self.base_url
                    )
                    
                    # Extract from each facility page
                    for link_data in facility_links[:20]:  # Limit to avoid overwhelming
                        try:
                            self.logger.info(f"🏥 Extracting from: {link_data['name']}")
                            page_result = await crawler.arun(url=link_data['url'])
                            
                            if page_result.success:
                                facilities = await self.llm_extractor.extract_facilities_from_content(
                                    page_result.html, link_data['url']
                                )
                                all_facilities.extend(facilities)
                        
                        except Exception as e:
                            self.logger.warning(f"Failed to extract from {link_data['url']}: {e}")
                    
                    # Process facility sections found by structure analysis
                    for section in structure.get('facility_sections', [])[:5]:  # Limit sections
                        if section.get('url'):
                            try:
                                self.logger.info(f"📋 Processing facility section: {section['section_name']}")
                                section_result = await crawler.arun(url=section['url'])
                                
                                if section_result.success:
                                    facilities = await self.llm_extractor.extract_facilities_from_content(
                                        section_result.html, section['url']
                                    )
                                    all_facilities.extend(facilities)
                            
                            except Exception as e:
                                self.logger.warning(f"Failed to process section {section['url']}: {e}")
            
            else:
                # Fallback to traditional scraping
                from healthcare_scraper import HealthcareFacilityScraper
                traditional_scraper = HealthcareFacilityScraper(self.base_url, self.output_dir)
                all_facilities = await traditional_scraper.scrape_facilities()
        
        # Remove duplicates
        unique_facilities = self._remove_duplicates(all_facilities)
        
        self.logger.info(f"🎉 Enhanced scraping completed: {len(unique_facilities)} unique facilities found")
        
        return unique_facilities
    
    def _remove_duplicates(self, facilities: List[FacilityInfo]) -> List[FacilityInfo]:
        """Remove duplicate facilities based on name and address"""
        
        seen = set()
        unique_facilities = []
        
        for facility in facilities:
            # Create a key based on name and address
            key = (
                facility.name.lower().strip(),
                facility.address.lower().strip(),
                facility.city.lower().strip()
            )
            
            if key not in seen:
                seen.add(key)
                unique_facilities.append(facility)
        
        return unique_facilities

