"""
LLM-powered extractors for healthcare facilities
Uses crawl4ai's LLM integration for surgical precision in data extraction
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


class LLMFacilityExtractor:
    """Advanced LLM-powered facility information extractor"""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
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
    
    def create_facility_extraction_strategy(self) -> LLMExtractionStrategy:
        """Create LLM extraction strategy specifically for healthcare facilities"""
        
        instruction = """
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
        
        Return a JSON object with an array of facilities matching the provided schema.
        If no valid facilities are found, return {"facilities": []}.
        """
        
        return LLMExtractionStrategy(
            provider="openai",
            api_token=None,  # Will use environment variable OPENAI_API_KEY
            schema=self.facility_schema,
            extraction_type="schema",
            instruction=instruction,
            model=self.model_name
        )
    
    def create_facility_listing_strategy(self) -> LLMExtractionStrategy:
        """Create strategy for finding facility listing pages"""
        
        listing_schema = {
            "type": "object",
            "properties": {
                "facility_links": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL to facility detail page"},
                            "name": {"type": "string", "description": "Facility name"},
                            "location": {"type": "string", "description": "City, State or general location"},
                            "facility_type": {"type": "string", "description": "Type of facility if mentioned"}
                        },
                        "required": ["url", "name"]
                    }
                }
            }
        }
        
        instruction = """
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
        
        Return URLs as complete, absolute URLs when possible.
        """
        
        return LLMExtractionStrategy(
            provider="openai",
            api_token=None,
            schema=listing_schema,
            extraction_type="schema",
            instruction=instruction,
            model=self.model_name
        )
    
    async def extract_facilities_from_page(self, crawler: AsyncWebCrawler, url: str) -> List[FacilityInfo]:
        """Extract facility information from a single page using LLM"""
        
        facilities = []
        
        try:
            # Use LLM extraction strategy
            extraction_strategy = self.create_facility_extraction_strategy()
            
            result = await crawler.arun(
                url=url,
                extraction_strategy=extraction_strategy,
                chunking_strategy=RegexChunking(patterns=[r'\n\n', r'\. ']),
                bypass_cache=True
            )
            
            if result.success and result.extracted_content:
                try:
                    # Parse the LLM response
                    extracted_data = json.loads(result.extracted_content)
                    
                    if 'facilities' in extracted_data:
                        for facility_data in extracted_data['facilities']:
                            facility = self._convert_to_facility_info(facility_data, url)
                            if facility and self._is_valid_facility(facility):
                                facilities.append(facility)
                                self.logger.info(f"LLM extracted facility: {facility.name}")
                    
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse LLM response for {url}: {e}")
                    # Fallback to regex extraction if LLM response is malformed
                    facilities.extend(await self._fallback_extraction(result.html, url))
            
            else:
                self.logger.warning(f"LLM extraction failed for {url}: {result.error_message}")
                # Fallback to traditional extraction
                facilities.extend(await self._fallback_extraction(result.html, url))
        
        except Exception as e:
            self.logger.error(f"Error in LLM extraction for {url}: {e}")
        
        return facilities
    
    async def find_facility_links(self, crawler: AsyncWebCrawler, url: str) -> List[Dict[str, str]]:
        """Use LLM to intelligently find facility listing links"""
        
        facility_links = []
        
        try:
            extraction_strategy = self.create_facility_listing_strategy()
            
            result = await crawler.arun(
                url=url,
                extraction_strategy=extraction_strategy,
                bypass_cache=True
            )
            
            if result.success and result.extracted_content:
                try:
                    extracted_data = json.loads(result.extracted_content)
                    
                    if 'facility_links' in extracted_data:
                        for link_data in extracted_data['facility_links']:
                            # Ensure URL is absolute
                            if link_data.get('url'):
                                if not link_data['url'].startswith('http'):
                                    from urllib.parse import urljoin
                                    link_data['url'] = urljoin(url, link_data['url'])
                                
                                facility_links.append(link_data)
                                self.logger.info(f"LLM found facility link: {link_data['name']} - {link_data['url']}")
                
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse LLM response for facility links: {e}")
        
        except Exception as e:
            self.logger.error(f"Error in LLM facility link extraction: {e}")
        
        return facility_links
    
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
    
    async def _fallback_extraction(self, html: str, url: str) -> List[FacilityInfo]:
        """Fallback to traditional extraction if LLM fails"""
        
        # Import traditional extractor
        from extractors import FacilityDataExtractor
        from bs4 import BeautifulSoup
        
        try:
            extractor = FacilityDataExtractor()
            soup = BeautifulSoup(html, 'html.parser')
            facility = extractor.extract_facility_data(soup, url)
            
            if self._is_valid_facility(facility):
                return [facility]
        
        except Exception as e:
            self.logger.error(f"Fallback extraction failed for {url}: {e}")
        
        return []


class SmartFacilityDiscovery:
    """LLM-powered facility discovery that understands website structure"""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
    
    async def analyze_website_structure(self, crawler: AsyncWebCrawler, base_url: str) -> Dict[str, Any]:
        """Use LLM to understand website structure and find facility sections"""
        
        structure_schema = {
            "type": "object",
            "properties": {
                "facility_sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "section_name": {"type": "string"},
                            "url": {"type": "string"},
                            "description": {"type": "string"},
                            "likely_facility_count": {"type": "integer"}
                        }
                    }
                },
                "navigation_pattern": {"type": "string"},
                "site_type": {"type": "string"},
                "recommended_strategy": {"type": "string"}
            }
        }
        
        instruction = """
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
        
        PROVIDE:
        - URLs to main facility listing pages
        - Description of how facilities are organized
        - Recommended crawling strategy
        - Site type (corporate chain, single facility, regional network, etc.)
        """
        
        extraction_strategy = LLMExtractionStrategy(
            provider="openai",
            api_token=None,
            schema=structure_schema,
            extraction_type="schema",
            instruction=instruction,
            model=self.model_name
        )
        
        try:
            result = await crawler.arun(
                url=base_url,
                extraction_strategy=extraction_strategy,
                bypass_cache=True
            )
            
            if result.success and result.extracted_content:
                return json.loads(result.extracted_content)
        
        except Exception as e:
            self.logger.error(f"Error analyzing website structure: {e}")
        
        return {"facility_sections": [], "navigation_pattern": "unknown", "site_type": "unknown"}


# Integration with existing scraper
class EnhancedHealthcareScraper:
    """Enhanced scraper that combines traditional and LLM extraction"""
    
    def __init__(self, base_url: str, output_dir: str, use_llm: bool = True):
        self.base_url = base_url
        self.output_dir = output_dir
        self.use_llm = use_llm
        self.logger = logging.getLogger(__name__)
        
        if use_llm:
            self.llm_extractor = LLMFacilityExtractor()
            self.smart_discovery = SmartFacilityDiscovery()
    
    async def scrape_with_llm_enhancement(self) -> List[FacilityInfo]:
        """Run enhanced scraping with LLM assistance"""
        
        all_facilities = []
        
        async with AsyncWebCrawler(headless=True) as crawler:
            
            if self.use_llm:
                # Step 1: Analyze website structure with LLM
                self.logger.info("🧠 Analyzing website structure with LLM...")
                structure = await self.smart_discovery.analyze_website_structure(crawler, self.base_url)
                
                self.logger.info(f"Site type: {structure.get('site_type', 'unknown')}")
                self.logger.info(f"Navigation pattern: {structure.get('navigation_pattern', 'unknown')}")
                
                # Step 2: Use LLM to find facility links
                facility_links = await self.llm_extractor.find_facility_links(crawler, self.base_url)
                
                # Step 3: Extract from each facility page using LLM
                for link_data in facility_links:
                    self.logger.info(f"🏥 Extracting from: {link_data['name']}")
                    facilities = await self.llm_extractor.extract_facilities_from_page(
                        crawler, link_data['url']
                    )
                    all_facilities.extend(facilities)
                
                # Step 4: Also try facility sections found by structure analysis
                for section in structure.get('facility_sections', []):
                    if section.get('url'):
                        self.logger.info(f"📋 Processing facility section: {section['section_name']}")
                        facilities = await self.llm_extractor.extract_facilities_from_page(
                            crawler, section['url']
                        )
                        all_facilities.extend(facilities)
            
            else:
                # Fallback to traditional scraping
                from healthcare_scraper import HealthcareFacilityScraper
                traditional_scraper = HealthcareFacilityScraper(self.base_url, self.output_dir)
                all_facilities = await traditional_scraper.scrape_facilities()
        
        # Remove duplicates
        unique_facilities = self._remove_duplicates(all_facilities)
        
        self.logger.info(f"🎉 LLM-enhanced scraping completed: {len(unique_facilities)} unique facilities found")
        
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

