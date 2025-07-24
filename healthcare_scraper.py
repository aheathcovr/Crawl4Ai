"""
Healthcare Facility Scraper - Main scraper class
"""

import asyncio
import json
import logging
import re
import time
from typing import Dict, List, Optional, Set, Tuple, Any
from urllib.parse import urljoin, urlparse, parse_qs
from dataclasses import dataclass, asdict
from datetime import datetime

import pandas as pd
from crawl4ai import AsyncWebCrawler, CrawlResult
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from bs4 import BeautifulSoup
import aiohttp

from config import (
    FACILITY_TYPES, FACILITY_URL_PATTERNS, FACILITY_KEYWORDS,
    FACILITY_DATA_FIELDS, CRAWL_CONFIG, BROWSER_CONFIG,
    CSS_SELECTORS, REGEX_PATTERNS, STATE_ABBREVIATIONS,
    EXCLUDE_PATTERNS
)


@dataclass
class FacilityInfo:
    """Data class for storing facility information"""
    name: str = ""
    facility_type: str = ""
    address: str = ""
    city: str = ""
    state: str = ""
    zip_code: str = ""
    phone: str = ""
    fax: str = ""
    email: str = ""
    website: str = ""
    administrator: str = ""
    director: str = ""
    medical_director: str = ""
    beds: str = ""
    capacity: str = ""
    license_number: str = ""
    medicare_provider_id: str = ""
    medicaid_certified: str = ""
    services_offered: List[str] = None
    specialties: List[str] = None
    amenities: List[str] = None
    care_levels: List[str] = None
    description: str = ""
    hours: str = ""
    visiting_hours: str = ""
    parking_info: str = ""
    source_url: str = ""
    scraped_at: str = ""
    
    def __post_init__(self):
        if self.services_offered is None:
            self.services_offered = []
        if self.specialties is None:
            self.specialties = []
        if self.amenities is None:
            self.amenities = []
        if self.care_levels is None:
            self.care_levels = []
        if not self.scraped_at:
            self.scraped_at = datetime.now().isoformat()


class HealthcareFacilityScraper:
    """Main scraper class for healthcare facilities"""
    
    def __init__(self, base_url: str, output_dir: str = "./output"):
        self.base_url = base_url
        self.output_dir = output_dir
        self.domain = urlparse(base_url).netloc
        self.visited_urls: Set[str] = set()
        self.facility_urls: Set[str] = set()
        self.facilities: List[FacilityInfo] = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('healthcare_scraper.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    async def scrape_facilities(self) -> List[FacilityInfo]:
        """Main method to scrape all facilities from the website"""
        self.logger.info(f"Starting scrape of {self.base_url}")
        
        async with AsyncWebCrawler(
            headless=BROWSER_CONFIG['headless'],
            viewport_width=BROWSER_CONFIG['viewport']['width'],
            viewport_height=BROWSER_CONFIG['viewport']['height']
        ) as crawler:
            
            # Step 1: Discover facility listing pages
            await self._discover_facility_pages(crawler)
            
            # Step 2: Extract facility URLs from listing pages
            await self._extract_facility_urls(crawler)
            
            # Step 3: Scrape individual facility pages
            await self._scrape_individual_facilities(crawler)
            
        self.logger.info(f"Scraping completed. Found {len(self.facilities)} facilities")
        return self.facilities
    
    async def _discover_facility_pages(self, crawler: AsyncWebCrawler):
        """Discover pages that likely contain facility listings"""
        self.logger.info("Discovering facility listing pages...")
        
        # Start with the main page
        result = await crawler.arun(url=self.base_url)
        if result.success:
            await self._analyze_page_for_facility_links(result, crawler)
        
        # Look for common facility listing patterns in navigation
        potential_urls = self._generate_potential_facility_urls()
        
        for url in potential_urls:
            if url not in self.visited_urls:
                try:
                    result = await crawler.arun(url=url)
                    if result.success:
                        await self._analyze_page_for_facility_links(result, crawler)
                        await asyncio.sleep(CRAWL_CONFIG['delay_between_requests'])
                except Exception as e:
                    self.logger.warning(f"Error crawling {url}: {e}")
    
    def _generate_potential_facility_urls(self) -> List[str]:
        """Generate potential URLs for facility listings"""
        urls = []
        for pattern in FACILITY_URL_PATTERNS:
            urls.extend([
                urljoin(self.base_url, pattern),
                urljoin(self.base_url, f"/{pattern}"),
                urljoin(self.base_url, f"/{pattern}/"),
                urljoin(self.base_url, f"/{pattern}.html"),
                urljoin(self.base_url, f"/{pattern}.php")
            ])
        return list(set(urls))
    
    async def _analyze_page_for_facility_links(self, result: CrawlResult, crawler: AsyncWebCrawler):
        """Analyze a page to find links to facility pages"""
        if not result.success or not result.html:
            return
            
        self.visited_urls.add(result.url)
        soup = BeautifulSoup(result.html, 'html.parser')
        
        # Look for facility-specific links
        facility_links = self._find_facility_links(soup, result.url)
        
        for link in facility_links:
            if self._is_facility_url(link) and link not in self.facility_urls:
                self.facility_urls.add(link)
                self.logger.debug(f"Found facility URL: {link}")
        
        # Look for pagination or "load more" functionality
        await self._handle_pagination(soup, result.url, crawler)
    
    def _find_facility_links(self, soup: BeautifulSoup, base_url: str) -> Set[str]:
        """Find links that likely point to facility pages"""
        links = set()
        
        # Use CSS selectors to find facility links
        for selector in CSS_SELECTORS['facility_links']:
            elements = soup.select(selector)
            for element in elements:
                href = element.get('href')
                if href:
                    full_url = urljoin(base_url, href)
                    if self._is_same_domain(full_url):
                        links.add(full_url)
        
        # Look for links containing facility keywords
        all_links = soup.find_all('a', href=True)
        for link in all_links:
            href = link.get('href', '')
            text = link.get_text(strip=True).lower()
            
            # Check if link text or href contains facility indicators
            if any(keyword in text or keyword in href.lower() 
                   for keyword in FACILITY_URL_PATTERNS):
                full_url = urljoin(base_url, href)
                if self._is_same_domain(full_url):
                    links.add(full_url)
        
        return links
    
    def _is_facility_url(self, url: str) -> bool:
        """Check if a URL likely points to a facility page"""
        url_lower = url.lower()
        
        # Check for facility patterns in URL
        if any(pattern in url_lower for pattern in FACILITY_URL_PATTERNS):
            return True
        
        # Check for exclusion patterns
        if any(pattern in url_lower for pattern in EXCLUDE_PATTERNS):
            return False
        
        # Check for facility-specific path segments
        path_segments = urlparse(url).path.lower().split('/')
        facility_indicators = ['location', 'facility', 'center', 'community', 'property']
        
        return any(indicator in segment for segment in path_segments 
                  for indicator in facility_indicators)
    
    def _is_same_domain(self, url: str) -> bool:
        """Check if URL belongs to the same domain"""
        return urlparse(url).netloc == self.domain
    
    async def _handle_pagination(self, soup: BeautifulSoup, base_url: str, crawler: AsyncWebCrawler):
        """Handle pagination on facility listing pages"""
        # Look for pagination links
        pagination_selectors = [
            'a[href*="page"]', '.pagination a', '.pager a',
            'a:contains("Next")', 'a:contains("More")', 'a:contains("Load")'
        ]
        
        for selector in pagination_selectors:
            try:
                elements = soup.select(selector)
                for element in elements:
                    href = element.get('href')
                    if href and 'page' in href.lower():
                        next_url = urljoin(base_url, href)
                        if next_url not in self.visited_urls and self._is_same_domain(next_url):
                            result = await crawler.arun(url=next_url)
                            if result.success:
                                await self._analyze_page_for_facility_links(result, crawler)
                                await asyncio.sleep(CRAWL_CONFIG['delay_between_requests'])
            except Exception as e:
                self.logger.debug(f"Error handling pagination: {e}")
    
    async def _extract_facility_urls(self, crawler: AsyncWebCrawler):
        """Extract individual facility URLs from listing pages"""
        self.logger.info(f"Extracting facility URLs from {len(self.facility_urls)} listing pages...")
        
        for listing_url in list(self.facility_urls):
            try:
                result = await crawler.arun(url=listing_url)
                if result.success:
                    soup = BeautifulSoup(result.html, 'html.parser')
                    
                    # Look for facility cards or listings
                    facility_cards = self._find_facility_cards(soup)
                    
                    for card in facility_cards:
                        facility_link = self._extract_facility_link_from_card(card, listing_url)
                        if facility_link and facility_link not in self.facility_urls:
                            self.facility_urls.add(facility_link)
                
                await asyncio.sleep(CRAWL_CONFIG['delay_between_requests'])
                
            except Exception as e:
                self.logger.warning(f"Error extracting from {listing_url}: {e}")
    
    def _find_facility_cards(self, soup: BeautifulSoup) -> List:
        """Find facility cards or listings on a page"""
        cards = []
        
        for selector in CSS_SELECTORS['facility_cards']:
            elements = soup.select(selector)
            cards.extend(elements)
        
        # If no cards found, look for list items or table rows
        if not cards:
            cards.extend(soup.select('li'))
            cards.extend(soup.select('tr'))
        
        return cards
    
    def _extract_facility_link_from_card(self, card, base_url: str) -> Optional[str]:
        """Extract facility link from a facility card"""
        # Look for links within the card
        links = card.find_all('a', href=True)
        
        for link in links:
            href = link.get('href')
            if href:
                full_url = urljoin(base_url, href)
                if self._is_facility_url(full_url) and self._is_same_domain(full_url):
                    return full_url
        
        return None
    
    async def _scrape_individual_facilities(self, crawler: AsyncWebCrawler):
        """Scrape individual facility pages for detailed information"""
        self.logger.info(f"Scraping {len(self.facility_urls)} individual facility pages...")
        
        for facility_url in self.facility_urls:
            try:
                result = await crawler.arun(url=facility_url)
                if result.success:
                    facility_info = await self._extract_facility_info(result)
                    if facility_info and facility_info.name:  # Only add if we found a name
                        self.facilities.append(facility_info)
                        self.logger.debug(f"Extracted facility: {facility_info.name}")
                
                await asyncio.sleep(CRAWL_CONFIG['delay_between_requests'])
                
            except Exception as e:
                self.logger.warning(f"Error scraping facility {facility_url}: {e}")
    
    async def _extract_facility_info(self, result: CrawlResult) -> Optional[FacilityInfo]:
        """Extract facility information from a facility page"""
        if not result.success or not result.html:
            return None
        
        soup = BeautifulSoup(result.html, 'html.parser')
        facility = FacilityInfo(source_url=result.url)
        
        # Extract basic information
        facility.name = self._extract_facility_name(soup)
        facility.facility_type = self._determine_facility_type(soup, result.html)
        
        # Extract contact information
        self._extract_contact_info(soup, facility)
        
        # Extract operational information
        self._extract_operational_info(soup, facility)
        
        # Extract services and amenities
        self._extract_services_info(soup, facility)
        
        return facility
    
    def _extract_facility_name(self, soup: BeautifulSoup) -> str:
        """Extract facility name from the page"""
        # Try various selectors for facility name
        name_selectors = [
            'h1', '.facility-name', '.location-name', '.community-name',
            '.property-name', '[class*="name"]', 'title'
        ]
        
        for selector in name_selectors:
            element = soup.select_one(selector)
            if element:
                name = element.get_text(strip=True)
                if name and len(name) > 3:  # Basic validation
                    return name
        
        # Fallback to page title
        title = soup.find('title')
        if title:
            return title.get_text(strip=True)
        
        return ""
    
    def _determine_facility_type(self, soup: BeautifulSoup, html: str) -> str:
        """Determine the type of healthcare facility"""
        text_content = soup.get_text().lower()
        
        # Check for facility type keywords
        for facility_type, keywords in FACILITY_TYPES.items():
            for keyword in keywords:
                if keyword in text_content:
                    return facility_type.replace('_', ' ').title()
        
        return "Healthcare Facility"
    
    def _extract_contact_info(self, soup: BeautifulSoup, facility: FacilityInfo):
        """Extract contact information"""
        text_content = soup.get_text()
        
        # Extract phone number
        phone_match = re.search(REGEX_PATTERNS['phone'], text_content)
        if phone_match:
            facility.phone = phone_match.group(1)
        
        # Extract email
        email_match = re.search(REGEX_PATTERNS['email'], text_content)
        if email_match:
            facility.email = email_match.group(1)
        
        # Extract address components
        self._extract_address_info(soup, facility)
    
    def _extract_address_info(self, soup: BeautifulSoup, facility: FacilityInfo):
        """Extract address information"""
        # Look for address in structured data or specific elements
        address_selectors = [
            '.address', '.location', '[class*="address"]',
            '[itemprop="address"]', '.contact-info'
        ]
        
        for selector in address_selectors:
            element = soup.select_one(selector)
            if element:
                address_text = element.get_text(strip=True)
                self._parse_address(address_text, facility)
                break
        
        # If no structured address found, try to extract from full text
        if not facility.address:
            text_content = soup.get_text()
            address_match = re.search(REGEX_PATTERNS['address'], text_content)
            if address_match:
                self._parse_address(address_match.group(1), facility)
    
    def _parse_address(self, address_text: str, facility: FacilityInfo):
        """Parse address text into components"""
        # This is a simplified parser - could be enhanced with more sophisticated logic
        lines = [line.strip() for line in address_text.split('\n') if line.strip()]
        
        if lines:
            facility.address = lines[0]
            
            # Look for city, state, zip in subsequent lines
            for line in lines[1:]:
                zip_match = re.search(REGEX_PATTERNS['zip_code'], line)
                if zip_match:
                    facility.zip_code = zip_match.group(1)
                    # Extract city and state from the same line
                    parts = line.replace(facility.zip_code, '').strip().split(',')
                    if len(parts) >= 2:
                        facility.city = parts[0].strip()
                        facility.state = parts[1].strip()
                    break
    
    def _extract_operational_info(self, soup: BeautifulSoup, facility: FacilityInfo):
        """Extract operational information like beds, administrator, etc."""
        text_content = soup.get_text()
        
        # Extract bed count
        beds_match = re.search(REGEX_PATTERNS['beds'], text_content, re.IGNORECASE)
        if beds_match:
            facility.beds = beds_match.group(1)
        
        # Extract license number
        license_match = re.search(REGEX_PATTERNS['license'], text_content, re.IGNORECASE)
        if license_match:
            facility.license_number = license_match.group(1)
        
        # Extract Medicare ID
        medicare_match = re.search(REGEX_PATTERNS['medicare_id'], text_content, re.IGNORECASE)
        if medicare_match:
            facility.medicare_provider_id = medicare_match.group(1)
    
    def _extract_services_info(self, soup: BeautifulSoup, facility: FacilityInfo):
        """Extract services and amenities information"""
        # Look for services lists
        service_selectors = [
            '.services', '.amenities', '.care-services',
            '[class*="service"]', '[class*="amenity"]'
        ]
        
        for selector in service_selectors:
            elements = soup.select(selector)
            for element in elements:
                # Extract list items
                list_items = element.find_all(['li', 'p'])
                for item in list_items:
                    text = item.get_text(strip=True)
                    if text and len(text) > 3:
                        facility.services_offered.append(text)
    
    def save_results(self, format_type: str = 'json'):
        """Save scraping results to file"""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        domain_name = self.domain.replace('.', '_')
        
        if format_type == 'json':
            filename = f"{self.output_dir}/{domain_name}_facilities_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump([asdict(facility) for facility in self.facilities], 
                         f, indent=2, ensure_ascii=False)
        
        elif format_type == 'csv':
            filename = f"{self.output_dir}/{domain_name}_facilities_{timestamp}.csv"
            df = pd.DataFrame([asdict(facility) for facility in self.facilities])
            df.to_csv(filename, index=False, encoding='utf-8')
        
        elif format_type == 'excel':
            filename = f"{self.output_dir}/{domain_name}_facilities_{timestamp}.xlsx"
            df = pd.DataFrame([asdict(facility) for facility in self.facilities])
            df.to_excel(filename, index=False, engine='openpyxl')
        
        self.logger.info(f"Results saved to {filename}")
        return filename

