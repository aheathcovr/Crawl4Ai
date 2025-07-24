"""
Data extraction and parsing components for healthcare facilities
Handles various page layouts and data formats
"""

import re
import json
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Tag, NavigableString
import pandas as pd

from config import (
    FACILITY_TYPES, FACILITY_DATA_FIELDS, CSS_SELECTORS, 
    REGEX_PATTERNS, STATE_ABBREVIATIONS
)
from healthcare_scraper import FacilityInfo


class StructuredDataExtractor:
    """Extract structured data (JSON-LD, microdata, etc.)"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_json_ld(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract JSON-LD structured data"""
        json_ld_data = {}
        
        scripts = soup.find_all('script', type='application/ld+json')
        for script in scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, list):
                    for item in data:
                        json_ld_data.update(self._process_json_ld_item(item))
                else:
                    json_ld_data.update(self._process_json_ld_item(data))
            except (json.JSONDecodeError, AttributeError) as e:
                self.logger.debug(f"Error parsing JSON-LD: {e}")
        
        return json_ld_data
    
    def _process_json_ld_item(self, item: Dict) -> Dict[str, Any]:
        """Process individual JSON-LD item"""
        processed = {}
        
        if not isinstance(item, dict):
            return processed
        
        # Look for organization/business data
        if item.get('@type') in ['Organization', 'LocalBusiness', 'HealthAndBeautyBusiness']:
            processed.update({
                'name': item.get('name', ''),
                'description': item.get('description', ''),
                'url': item.get('url', ''),
                'telephone': item.get('telephone', ''),
                'email': item.get('email', '')
            })
            
            # Extract address
            address = item.get('address', {})
            if isinstance(address, dict):
                processed.update({
                    'street_address': address.get('streetAddress', ''),
                    'city': address.get('addressLocality', ''),
                    'state': address.get('addressRegion', ''),
                    'zip_code': address.get('postalCode', ''),
                    'country': address.get('addressCountry', '')
                })
        
        return processed
    
    def extract_microdata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract microdata structured information"""
        microdata = {}
        
        # Look for itemscope elements
        items = soup.find_all(attrs={'itemscope': True})
        
        for item in items:
            item_type = item.get('itemtype', '')
            if 'Organization' in item_type or 'LocalBusiness' in item_type:
                microdata.update(self._extract_microdata_properties(item))
        
        return microdata
    
    def _extract_microdata_properties(self, element: Tag) -> Dict[str, Any]:
        """Extract properties from microdata element"""
        properties = {}
        
        # Find all elements with itemprop
        prop_elements = element.find_all(attrs={'itemprop': True})
        
        for prop_elem in prop_elements:
            prop_name = prop_elem.get('itemprop')
            
            # Get property value based on element type
            if prop_elem.name == 'meta':
                prop_value = prop_elem.get('content', '')
            elif prop_elem.name in ['input', 'textarea']:
                prop_value = prop_elem.get('value', '')
            elif prop_elem.name == 'a':
                prop_value = prop_elem.get('href', '')
            else:
                prop_value = prop_elem.get_text(strip=True)
            
            if prop_value:
                properties[prop_name] = prop_value
        
        return properties


class ContactInfoExtractor:
    """Extract contact information using various methods"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_contact_info(self, soup: BeautifulSoup, url: str) -> Dict[str, str]:
        """Extract all contact information from a page"""
        contact_info = {
            'phone': '',
            'fax': '',
            'email': '',
            'address': '',
            'city': '',
            'state': '',
            'zip_code': ''
        }
        
        # Method 1: Look for structured contact sections
        contact_info.update(self._extract_from_contact_sections(soup))
        
        # Method 2: Use regex patterns on full text
        contact_info.update(self._extract_with_regex(soup))
        
        # Method 3: Look for specific HTML elements
        contact_info.update(self._extract_from_elements(soup))
        
        # Method 4: Extract from links
        contact_info.update(self._extract_from_links(soup, url))
        
        return contact_info
    
    def _extract_from_contact_sections(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract from dedicated contact sections"""
        contact_info = {}
        
        contact_selectors = [
            '.contact', '.contact-info', '.contact-details',
            '.address', '.location-info', '.facility-contact',
            '[class*="contact"]', '[class*="address"]'
        ]
        
        for selector in contact_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text()
                
                # Extract phone
                phone_match = re.search(REGEX_PATTERNS['phone'], text)
                if phone_match and not contact_info.get('phone'):
                    contact_info['phone'] = self._clean_phone(phone_match.group(1))
                
                # Extract email
                email_match = re.search(REGEX_PATTERNS['email'], text)
                if email_match and not contact_info.get('email'):
                    contact_info['email'] = email_match.group(1)
                
                # Extract address
                if not contact_info.get('address'):
                    address_info = self._parse_address_text(text)
                    contact_info.update(address_info)
        
        return contact_info
    
    def _extract_with_regex(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract using regex patterns on full page text"""
        contact_info = {}
        text_content = soup.get_text()
        
        # Phone number
        phone_matches = re.findall(REGEX_PATTERNS['phone'], text_content)
        if phone_matches and not contact_info.get('phone'):
            # Take the first valid phone number
            for phone in phone_matches:
                cleaned_phone = self._clean_phone(phone)
                if self._is_valid_phone(cleaned_phone):
                    contact_info['phone'] = cleaned_phone
                    break
        
        # Email
        email_matches = re.findall(REGEX_PATTERNS['email'], text_content)
        if email_matches and not contact_info.get('email'):
            # Filter out common false positives
            for email in email_matches:
                if not any(exclude in email.lower() for exclude in ['example', 'test', 'noreply']):
                    contact_info['email'] = email
                    break
        
        return contact_info
    
    def _extract_from_elements(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract from specific HTML elements"""
        contact_info = {}
        
        # Look for tel: links
        tel_links = soup.find_all('a', href=re.compile(r'^tel:'))
        if tel_links and not contact_info.get('phone'):
            href = tel_links[0].get('href', '')
            phone = href.replace('tel:', '').strip()
            contact_info['phone'] = self._clean_phone(phone)
        
        # Look for mailto: links
        mailto_links = soup.find_all('a', href=re.compile(r'^mailto:'))
        if mailto_links and not contact_info.get('email'):
            href = mailto_links[0].get('href', '')
            email = href.replace('mailto:', '').strip()
            contact_info['email'] = email
        
        # Look for address in specific elements
        address_elements = soup.select('[itemprop="address"], .address, .location')
        for element in address_elements:
            if not contact_info.get('address'):
                address_info = self._parse_address_element(element)
                contact_info.update(address_info)
                break
        
        return contact_info
    
    def _extract_from_links(self, soup: BeautifulSoup, base_url: str) -> Dict[str, str]:
        """Extract contact info from various link types"""
        contact_info = {}
        
        # Look for Google Maps links (often contain address)
        maps_links = soup.find_all('a', href=re.compile(r'maps\.google|google\.com/maps'))
        for link in maps_links:
            href = link.get('href', '')
            address = self._extract_address_from_maps_url(href)
            if address and not contact_info.get('address'):
                contact_info['address'] = address
        
        return contact_info
    
    def _parse_address_text(self, text: str) -> Dict[str, str]:
        """Parse address information from text"""
        address_info = {}
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        for line in lines:
            # Look for zip code to identify city/state line
            zip_match = re.search(REGEX_PATTERNS['zip_code'], line)
            if zip_match:
                address_info['zip_code'] = zip_match.group(1)
                
                # Extract city and state from the same line
                line_without_zip = line.replace(zip_match.group(1), '').strip()
                parts = [part.strip() for part in line_without_zip.split(',')]
                
                if len(parts) >= 2:
                    address_info['city'] = parts[-2]
                    address_info['state'] = self._normalize_state(parts[-1])
                elif len(parts) == 1:
                    # Try to split by space for "City ST" format
                    city_state = parts[0].split()
                    if len(city_state) >= 2:
                        address_info['state'] = self._normalize_state(city_state[-1])
                        address_info['city'] = ' '.join(city_state[:-1])
                
                break
        
        # Look for street address (usually the first line with numbers)
        for line in lines:
            if re.match(r'^\d+', line) and not address_info.get('address'):
                address_info['address'] = line
                break
        
        return address_info
    
    def _parse_address_element(self, element: Tag) -> Dict[str, str]:
        """Parse address from a structured HTML element"""
        address_info = {}
        
        # Look for specific address components
        street = element.select_one('[itemprop="streetAddress"], .street, .street-address')
        if street:
            address_info['address'] = street.get_text(strip=True)
        
        city = element.select_one('[itemprop="addressLocality"], .city, .locality')
        if city:
            address_info['city'] = city.get_text(strip=True)
        
        state = element.select_one('[itemprop="addressRegion"], .state, .region')
        if state:
            address_info['state'] = self._normalize_state(state.get_text(strip=True))
        
        zip_code = element.select_one('[itemprop="postalCode"], .zip, .postal-code')
        if zip_code:
            address_info['zip_code'] = zip_code.get_text(strip=True)
        
        # If no structured data, parse as text
        if not any(address_info.values()):
            address_info = self._parse_address_text(element.get_text())
        
        return address_info
    
    def _extract_address_from_maps_url(self, url: str) -> str:
        """Extract address from Google Maps URL"""
        # Simple extraction - could be enhanced
        if 'q=' in url:
            parts = url.split('q=')
            if len(parts) > 1:
                address = parts[1].split('&')[0]
                return address.replace('+', ' ').replace('%20', ' ')
        
        return ''
    
    def _clean_phone(self, phone: str) -> str:
        """Clean and format phone number"""
        # Remove all non-digit characters except +
        cleaned = re.sub(r'[^\d+]', '', phone)
        
        # Format US phone numbers
        if len(cleaned) == 10:
            return f"({cleaned[:3]}) {cleaned[3:6]}-{cleaned[6:]}"
        elif len(cleaned) == 11 and cleaned.startswith('1'):
            return f"({cleaned[1:4]}) {cleaned[4:7]}-{cleaned[7:]}"
        
        return cleaned
    
    def _is_valid_phone(self, phone: str) -> bool:
        """Check if phone number is valid"""
        digits = re.sub(r'\D', '', phone)
        return len(digits) in [10, 11] and not digits.startswith('0')
    
    def _normalize_state(self, state: str) -> str:
        """Normalize state name to abbreviation"""
        state_clean = state.strip().lower()
        
        # If already an abbreviation
        if len(state_clean) == 2:
            return state_clean.upper()
        
        # Look up full state name
        return STATE_ABBREVIATIONS.get(state_clean, state_clean.upper())


class FacilityTypeClassifier:
    """Classify facility types based on content"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def classify_facility(self, soup: BeautifulSoup, url: str) -> str:
        """Classify the type of healthcare facility"""
        text_content = soup.get_text().lower()
        
        # Score each facility type based on keyword frequency
        type_scores = {}
        
        for facility_type, keywords in FACILITY_TYPES.items():
            score = 0
            for keyword in keywords:
                # Count occurrences, with higher weight for exact matches
                exact_matches = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_content))
                partial_matches = text_content.count(keyword) - exact_matches
                
                score += exact_matches * 3 + partial_matches
            
            type_scores[facility_type] = score
        
        # Return the type with the highest score
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            if type_scores[best_type] > 0:
                return best_type.replace('_', ' ').title()
        
        # Fallback classification based on URL
        url_lower = url.lower()
        for facility_type, keywords in FACILITY_TYPES.items():
            if any(keyword.replace(' ', '-') in url_lower for keyword in keywords):
                return facility_type.replace('_', ' ').title()
        
        return "Healthcare Facility"


class ServicesExtractor:
    """Extract services and amenities information"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_services(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """Extract services, amenities, and specialties"""
        services_info = {
            'services_offered': [],
            'amenities': [],
            'specialties': [],
            'care_levels': []
        }
        
        # Look for services sections
        services_sections = self._find_services_sections(soup)
        
        for section in services_sections:
            section_type = self._classify_services_section(section)
            items = self._extract_service_items(section)
            
            if section_type in services_info:
                services_info[section_type].extend(items)
        
        # Remove duplicates and clean up
        for key in services_info:
            services_info[key] = list(set(services_info[key]))
            services_info[key] = [item for item in services_info[key] if len(item.strip()) > 2]
        
        return services_info
    
    def _find_services_sections(self, soup: BeautifulSoup) -> List[Tag]:
        """Find sections that likely contain services information"""
        sections = []
        
        service_selectors = [
            '.services', '.amenities', '.care-services', '.specialties',
            '[class*="service"]', '[class*="amenity"]', '[class*="care"]',
            '[id*="service"]', '[id*="amenity"]'
        ]
        
        for selector in service_selectors:
            elements = soup.select(selector)
            sections.extend(elements)
        
        # Also look for sections with service-related headings
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        for heading in headings:
            heading_text = heading.get_text().lower()
            if any(keyword in heading_text for keyword in 
                   ['service', 'amenity', 'care', 'specialty', 'offer']):
                # Get the next sibling elements that might contain the list
                next_elem = heading.find_next_sibling()
                if next_elem:
                    sections.append(next_elem)
        
        return sections
    
    def _classify_services_section(self, section: Tag) -> str:
        """Classify what type of services section this is"""
        section_text = section.get_text().lower()
        class_names = ' '.join(section.get('class', [])).lower()
        
        if any(keyword in section_text or keyword in class_names 
               for keyword in ['amenity', 'amenities']):
            return 'amenities'
        elif any(keyword in section_text or keyword in class_names 
                 for keyword in ['specialty', 'specialties', 'specialize']):
            return 'specialties'
        elif any(keyword in section_text or keyword in class_names 
                 for keyword in ['care level', 'level of care']):
            return 'care_levels'
        else:
            return 'services_offered'
    
    def _extract_service_items(self, section: Tag) -> List[str]:
        """Extract individual service items from a section"""
        items = []
        
        # Look for list items
        list_items = section.find_all('li')
        for item in list_items:
            text = item.get_text(strip=True)
            if text:
                items.append(text)
        
        # Look for paragraphs or divs if no list items
        if not items:
            paragraphs = section.find_all(['p', 'div'])
            for para in paragraphs:
                text = para.get_text(strip=True)
                if text and len(text) < 200:  # Avoid long descriptions
                    items.append(text)
        
        # Look for comma-separated items
        if not items:
            text = section.get_text()
            if ',' in text:
                items = [item.strip() for item in text.split(',') if item.strip()]
        
        return items


class FacilityDataExtractor:
    """Main facility data extractor that combines all extraction methods"""
    
    def __init__(self):
        self.structured_extractor = StructuredDataExtractor()
        self.contact_extractor = ContactInfoExtractor()
        self.type_classifier = FacilityTypeClassifier()
        self.services_extractor = ServicesExtractor()
        self.logger = logging.getLogger(__name__)
    
    def extract_facility_data(self, soup: BeautifulSoup, url: str) -> FacilityInfo:
        """Extract comprehensive facility data from a page"""
        facility = FacilityInfo(source_url=url)
        
        try:
            # Extract structured data first (highest confidence)
            structured_data = self.structured_extractor.extract_json_ld(soup)
            microdata = self.structured_extractor.extract_microdata(soup)
            
            # Merge structured data
            all_structured = {**structured_data, **microdata}
            
            # Extract basic information
            facility.name = (all_structured.get('name') or 
                           self._extract_facility_name(soup))
            
            facility.description = (all_structured.get('description') or 
                                  self._extract_description(soup))
            
            # Extract contact information
            contact_info = self.contact_extractor.extract_contact_info(soup, url)
            
            # Prefer structured data, fallback to extracted
            facility.phone = all_structured.get('telephone') or contact_info.get('phone', '')
            facility.email = all_structured.get('email') or contact_info.get('email', '')
            facility.address = (all_structured.get('street_address') or 
                              contact_info.get('address', ''))
            facility.city = all_structured.get('city') or contact_info.get('city', '')
            facility.state = all_structured.get('state') or contact_info.get('state', '')
            facility.zip_code = (all_structured.get('zip_code') or 
                               contact_info.get('zip_code', ''))
            
            # Classify facility type
            facility.facility_type = self.type_classifier.classify_facility(soup, url)
            
            # Extract services information
            services_info = self.services_extractor.extract_services(soup)
            facility.services_offered = services_info['services_offered']
            facility.amenities = services_info['amenities']
            facility.specialties = services_info['specialties']
            facility.care_levels = services_info['care_levels']
            
            # Extract operational information
            self._extract_operational_info(soup, facility)
            
        except Exception as e:
            self.logger.error(f"Error extracting facility data from {url}: {e}")
        
        return facility
    
    def _extract_facility_name(self, soup: BeautifulSoup) -> str:
        """Extract facility name using multiple methods"""
        # Try various selectors in order of preference
        name_selectors = [
            'h1',
            '.facility-name', '.location-name', '.community-name',
            '[class*="name"]', '[class*="title"]',
            '.page-title', '.main-title'
        ]
        
        for selector in name_selectors:
            element = soup.select_one(selector)
            if element:
                name = element.get_text(strip=True)
                if name and len(name) > 3 and len(name) < 100:
                    return name
        
        # Fallback to page title
        title = soup.find('title')
        if title:
            title_text = title.get_text(strip=True)
            # Clean up common title patterns
            title_text = re.sub(r'\s*\|\s*.*$', '', title_text)  # Remove "| Company Name"
            title_text = re.sub(r'\s*-\s*.*$', '', title_text)   # Remove "- Company Name"
            return title_text
        
        return ""
    
    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract facility description"""
        description_selectors = [
            '.description', '.about', '.overview',
            '[class*="description"]', '[class*="about"]',
            'meta[name="description"]'
        ]
        
        for selector in description_selectors:
            element = soup.select_one(selector)
            if element:
                if element.name == 'meta':
                    return element.get('content', '')
                else:
                    text = element.get_text(strip=True)
                    if text and len(text) > 20:
                        return text
        
        return ""
    
    def _extract_operational_info(self, soup: BeautifulSoup, facility: FacilityInfo):
        """Extract operational information like beds, staff, etc."""
        text_content = soup.get_text()
        
        # Extract bed count
        beds_patterns = [
            r'(\d+)\s*beds?',
            r'(\d+)\s*bed\s+facility',
            r'bed\s+capacity[:\s]*(\d+)',
            r'(\d+)\s*licensed\s+beds?'
        ]
        
        for pattern in beds_patterns:
            match = re.search(pattern, text_content, re.IGNORECASE)
            if match:
                facility.beds = match.group(1)
                break
        
        # Extract administrator/director information
        staff_patterns = [
            r'administrator[:\s]*([^,\n]+)',
            r'executive\s+director[:\s]*([^,\n]+)',
            r'facility\s+director[:\s]*([^,\n]+)'
        ]
        
        for pattern in staff_patterns:
            match = re.search(pattern, text_content, re.IGNORECASE)
            if match:
                facility.administrator = match.group(1).strip()
                break
        
        # Extract license information
        license_patterns = [
            r'license\s*#?\s*([A-Z0-9-]+)',
            r'state\s+license[:\s]*([A-Z0-9-]+)',
            r'facility\s+id[:\s]*([A-Z0-9-]+)'
        ]
        
        for pattern in license_patterns:
            match = re.search(pattern, text_content, re.IGNORECASE)
            if match:
                facility.license_number = match.group(1)
                break
        
        # Extract Medicare provider ID
        medicare_patterns = [
            r'medicare\s+provider\s+(?:id|number)[:\s]*([A-Z0-9-]+)',
            r'cms\s+certification[:\s]*([A-Z0-9-]+)',
            r'provider\s+number[:\s]*([A-Z0-9-]+)'
        ]
        
        for pattern in medicare_patterns:
            match = re.search(pattern, text_content, re.IGNORECASE)
            if match:
                facility.medicare_provider_id = match.group(1)
                break

