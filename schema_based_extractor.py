"""
Schema-Based Healthcare Facility Extractor
Follows Crawl4AI best practices: Schema/Regex first, LLM as fallback only
Zero hallucination, guaranteed structure, high speed
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict

from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import (
    JsonCssExtractionStrategy, 
    RegexExtractionStrategy,
    LLMExtractionStrategy
)
from crawl4ai.chunking_strategy import RegexChunking

from healthcare_scraper import FacilityInfo


@dataclass
class ExtractionSchema:
    """Predefined extraction schemas for different healthcare site patterns"""
    
    name: str
    description: str
    css_schema: Dict[str, Any]
    regex_patterns: Dict[str, str]
    confidence_score: float
    site_patterns: List[str]  # URL patterns this schema works for


class HealthcareSchemaLibrary:
    """Library of proven extraction schemas for healthcare websites"""
    
    def __init__(self):
        self.schemas = self._load_predefined_schemas()
        self.logger = logging.getLogger(__name__)
    
    def _load_predefined_schemas(self) -> List[ExtractionSchema]:
        """Load predefined schemas for common healthcare site patterns"""
        
        schemas = []
        
        # Schema 1: Standard healthcare facility listing pages
        schemas.append(ExtractionSchema(
            name="standard_facility_listing",
            description="Standard facility listing with cards/tiles",
            css_schema={
                "name": "facilities",
                "baseSelector": ".facility-card, .location-card, .community-card, .center-card",
                "fields": [
                    {
                        "name": "facility_name",
                        "selector": "h1, h2, h3, .facility-name, .location-name, .community-name",
                        "type": "text"
                    },
                    {
                        "name": "address",
                        "selector": ".address, .location, .street-address, [itemprop='streetAddress']",
                        "type": "text"
                    },
                    {
                        "name": "city",
                        "selector": ".city, [itemprop='addressLocality']",
                        "type": "text"
                    },
                    {
                        "name": "state",
                        "selector": ".state, [itemprop='addressRegion']",
                        "type": "text"
                    },
                    {
                        "name": "zip_code",
                        "selector": ".zip, .postal-code, [itemprop='postalCode']",
                        "type": "text"
                    },
                    {
                        "name": "phone",
                        "selector": ".phone, .telephone, [itemprop='telephone'], a[href^='tel:']",
                        "type": "text"
                    },
                    {
                        "name": "website",
                        "selector": "a[href*='http'], .website-link, .facility-link",
                        "type": "attribute",
                        "attribute": "href"
                    },
                    {
                        "name": "facility_type",
                        "selector": ".facility-type, .care-type, .service-type",
                        "type": "text"
                    },
                    {
                        "name": "services",
                        "selector": ".services li, .amenities li, .care-services li",
                        "type": "text",
                        "multiple": True
                    }
                ]
            },
            regex_patterns={
                "phone": r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
                "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
                "zip_code": r'\b\d{5}(?:-\d{4})?\b',
                "state": r'\b[A-Z]{2}\b'
            },
            confidence_score=0.9,
            site_patterns=["*facility*", "*location*", "*community*", "*center*"]
        ))
        
        # Schema 2: Table-based facility listings
        schemas.append(ExtractionSchema(
            name="table_facility_listing",
            description="Table-based facility listings",
            css_schema={
                "name": "facilities",
                "baseSelector": "table tr, .facility-table tr, .location-table tr",
                "fields": [
                    {
                        "name": "facility_name",
                        "selector": "td:first-child, .facility-name, .name-column",
                        "type": "text"
                    },
                    {
                        "name": "address",
                        "selector": "td:nth-child(2), .address-column",
                        "type": "text"
                    },
                    {
                        "name": "phone",
                        "selector": "td:nth-child(3), .phone-column, a[href^='tel:']",
                        "type": "text"
                    },
                    {
                        "name": "facility_type",
                        "selector": "td:nth-child(4), .type-column",
                        "type": "text"
                    }
                ]
            },
            regex_patterns={
                "phone": r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
                "address": r'\d+\s+[A-Za-z0-9\s,.-]+',
            },
            confidence_score=0.85,
            site_patterns=["*table*", "*directory*", "*list*"]
        ))
        
        # Schema 3: Individual facility detail pages
        schemas.append(ExtractionSchema(
            name="individual_facility_detail",
            description="Individual facility detail pages",
            css_schema={
                "name": "facility",
                "baseSelector": "body, .facility-details, .location-details",
                "fields": [
                    {
                        "name": "facility_name",
                        "selector": "h1, .facility-name, .page-title",
                        "type": "text"
                    },
                    {
                        "name": "address",
                        "selector": ".address, .contact-info .address, [itemprop='streetAddress']",
                        "type": "text"
                    },
                    {
                        "name": "city",
                        "selector": ".city, [itemprop='addressLocality']",
                        "type": "text"
                    },
                    {
                        "name": "state",
                        "selector": ".state, [itemprop='addressRegion']",
                        "type": "text"
                    },
                    {
                        "name": "zip_code",
                        "selector": ".zip, [itemprop='postalCode']",
                        "type": "text"
                    },
                    {
                        "name": "phone",
                        "selector": ".phone, [itemprop='telephone'], a[href^='tel:']",
                        "type": "text"
                    },
                    {
                        "name": "email",
                        "selector": ".email, a[href^='mailto:']",
                        "type": "text"
                    },
                    {
                        "name": "administrator",
                        "selector": ".administrator, .director, .manager",
                        "type": "text"
                    },
                    {
                        "name": "beds",
                        "selector": ".beds, .capacity, .bed-count",
                        "type": "text"
                    },
                    {
                        "name": "services",
                        "selector": ".services li, .amenities li, .features li",
                        "type": "text",
                        "multiple": True
                    },
                    {
                        "name": "description",
                        "selector": ".description, .about, .overview",
                        "type": "text"
                    }
                ]
            },
            regex_patterns={
                "phone": r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
                "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
                "beds": r'\b\d+\s*(?:bed|room)s?\b',
                "license": r'[A-Z]{2}[-\s]?[A-Z0-9]+'
            },
            confidence_score=0.95,
            site_patterns=["*facility/*", "*location/*", "*community/*"]
        ))
        
        # Schema 4: Sunrise Senior Living specific
        schemas.append(ExtractionSchema(
            name="sunrise_senior_living",
            description="Sunrise Senior Living specific schema",
            css_schema={
                "name": "facilities",
                "baseSelector": ".community-card, .location-result",
                "fields": [
                    {
                        "name": "facility_name",
                        "selector": ".community-name, h3",
                        "type": "text"
                    },
                    {
                        "name": "address",
                        "selector": ".address-line-1",
                        "type": "text"
                    },
                    {
                        "name": "city_state_zip",
                        "selector": ".address-line-2",
                        "type": "text"
                    },
                    {
                        "name": "phone",
                        "selector": ".phone-number, a[href^='tel:']",
                        "type": "text"
                    },
                    {
                        "name": "care_types",
                        "selector": ".care-types li, .services li",
                        "type": "text",
                        "multiple": True
                    }
                ]
            },
            regex_patterns={
                "city_state_zip": r'([^,]+),\s*([A-Z]{2})\s*(\d{5}(?:-\d{4})?)'
            },
            confidence_score=0.98,
            site_patterns=["*sunriseseniorliving.com*"]
        ))
        
        return schemas
    
    def get_best_schema(self, url: str, html_content: str) -> Optional[ExtractionSchema]:
        """Select the best schema based on URL patterns and HTML content analysis"""
        
        # First, try URL pattern matching
        for schema in self.schemas:
            for pattern in schema.site_patterns:
                if self._matches_pattern(url, pattern):
                    self.logger.info(f"Selected schema '{schema.name}' based on URL pattern")
                    return schema
        
        # Then, analyze HTML content for structural patterns
        best_schema = None
        best_score = 0
        
        for schema in self.schemas:
            score = self._analyze_html_compatibility(html_content, schema)
            if score > best_score:
                best_score = score
                best_schema = schema
        
        if best_schema and best_score > 0.3:  # Minimum confidence threshold
            self.logger.info(f"Selected schema '{best_schema.name}' with compatibility score {best_score:.2f}")
            return best_schema
        
        self.logger.warning("No suitable schema found for this page")
        return None
    
    def _matches_pattern(self, url: str, pattern: str) -> bool:
        """Check if URL matches a pattern (supports wildcards)"""
        import fnmatch
        return fnmatch.fnmatch(url.lower(), pattern.lower())
    
    def _analyze_html_compatibility(self, html: str, schema: ExtractionSchema) -> float:
        """Analyze how well the HTML structure matches a schema"""
        
        score = 0
        total_checks = 0
        
        # Check for base selector presence
        base_selector = schema.css_schema.get("baseSelector", "")
        if base_selector:
            # Simple check for class/id patterns in HTML
            selectors = base_selector.split(", ")
            for selector in selectors:
                if "." in selector:
                    class_name = selector.split(".")[1].split(" ")[0]
                    if class_name in html:
                        score += 1
                elif "#" in selector:
                    id_name = selector.split("#")[1].split(" ")[0]
                    if f'id="{id_name}"' in html:
                        score += 1
                total_checks += 1
        
        # Check for field selector patterns
        for field in schema.css_schema.get("fields", []):
            selector = field.get("selector", "")
            if "." in selector:
                class_names = re.findall(r'\.([a-zA-Z0-9_-]+)', selector)
                for class_name in class_names:
                    if class_name in html:
                        score += 0.5
                    total_checks += 1
        
        return score / max(total_checks, 1)


class SchemaBasedExtractor:
    """Schema-based extractor following Crawl4AI best practices"""
    
    def __init__(self, use_llm_fallback: bool = False):
        self.schema_library = HealthcareSchemaLibrary()
        self.use_llm_fallback = use_llm_fallback
        self.logger = logging.getLogger(__name__)
        
        # Performance counters
        self.extraction_stats = {
            "schema_success": 0,
            "regex_success": 0,
            "llm_fallback": 0,
            "total_extractions": 0
        }
    
    async def extract_facilities(self, url: str, html_content: str = None) -> List[FacilityInfo]:
        """Extract facilities using schema-first approach"""
        
        self.extraction_stats["total_extractions"] += 1
        facilities = []
        
        async with AsyncWebCrawler(headless=True, verbose=True) as crawler:
            
            # Get HTML content if not provided
            if not html_content:
                result = await crawler.arun(url=url, bypass_cache=True)
                if not result.success:
                    self.logger.error(f"Failed to crawl {url}: {result.error_message}")
                    return []
                html_content = result.html
            
            # Step 1: Try schema-based extraction (primary method)
            schema = self.schema_library.get_best_schema(url, html_content)
            
            if schema:
                self.logger.info(f"Using schema-based extraction: {schema.name}")
                facilities = await self._extract_with_schema(crawler, url, schema)
                
                if facilities:
                    self.extraction_stats["schema_success"] += 1
                    self.logger.info(f"Schema extraction successful: {len(facilities)} facilities found")
                    return facilities
            
            # Step 2: Try regex-based extraction (fallback)
            self.logger.info("Trying regex-based extraction")
            facilities = await self._extract_with_regex(crawler, url, html_content)
            
            if facilities:
                self.extraction_stats["regex_success"] += 1
                self.logger.info(f"Regex extraction successful: {len(facilities)} facilities found")
                return facilities
            
            # Step 3: LLM fallback (only if enabled and other methods failed)
            if self.use_llm_fallback:
                self.logger.info("Falling back to LLM extraction")
                facilities = await self._extract_with_llm(crawler, url)
                
                if facilities:
                    self.extraction_stats["llm_fallback"] += 1
                    self.logger.info(f"LLM extraction successful: {len(facilities)} facilities found")
                    return facilities
            
            self.logger.warning(f"All extraction methods failed for {url}")
            return []
    
    async def _extract_with_schema(self, crawler, url: str, schema: ExtractionSchema) -> List[FacilityInfo]:
        """Extract using CSS schema (fastest, most reliable)"""
        
        try:
            extraction_strategy = JsonCssExtractionStrategy(schema.css_schema, verbose=True)
            
            result = await crawler.arun(
                url=url,
                extraction_strategy=extraction_strategy,
                bypass_cache=True
            )
            
            if result.success and result.extracted_content:
                extracted_data = json.loads(result.extracted_content)
                facilities = []
                
                # Handle both single facility and facility list
                facility_data = extracted_data.get(schema.css_schema["name"], [])
                if not isinstance(facility_data, list):
                    facility_data = [facility_data]
                
                for item in facility_data:
                    facility = self._convert_to_facility_info(item, url, schema.name)
                    if facility and self._is_valid_facility(facility):
                        facilities.append(facility)
                
                return facilities
            
        except Exception as e:
            self.logger.error(f"Schema extraction failed: {e}")
        
        return []
    
    async def _extract_with_regex(self, crawler, url: str, html_content: str) -> List[FacilityInfo]:
        """Extract using regex patterns (fast, reliable for structured data)"""
        
        try:
            # Define comprehensive regex patterns for healthcare facilities
            regex_patterns = {
                "facility_blocks": r'<div[^>]*(?:facility|location|community|center)[^>]*>.*?</div>',
                "phone_numbers": r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
                "addresses": r'\d+\s+[A-Za-z0-9\s,.-]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Boulevard|Blvd|Lane|Ln|Way|Circle|Cir)',
                "emails": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
                "facility_names": r'<h[1-6][^>]*>([^<]+(?:Care|Center|Living|Manor|Home|Facility|Community)[^<]*)</h[1-6]>',
                "zip_codes": r'\b\d{5}(?:-\d{4})?\b',
                "states": r'\b[A-Z]{2}\b'
            }
            
            facilities = []
            
            # Extract facility blocks first
            facility_blocks = re.findall(regex_patterns["facility_blocks"], html_content, re.IGNORECASE | re.DOTALL)
            
            if facility_blocks:
                # Process each facility block
                for block in facility_blocks:
                    facility = self._extract_from_block(block, url)
                    if facility and self._is_valid_facility(facility):
                        facilities.append(facility)
            else:
                # Fallback: extract from entire page
                facility = self._extract_from_block(html_content, url)
                if facility and self._is_valid_facility(facility):
                    facilities.append(facility)
            
            return facilities
            
        except Exception as e:
            self.logger.error(f"Regex extraction failed: {e}")
            return []
    
    def _extract_from_block(self, html_block: str, source_url: str) -> Optional[FacilityInfo]:
        """Extract facility information from an HTML block using regex"""
        
        facility = FacilityInfo(source_url=source_url)
        
        # Extract facility name
        name_patterns = [
            r'<h[1-6][^>]*>([^<]+)</h[1-6]>',
            r'class="[^"]*(?:name|title)[^"]*"[^>]*>([^<]+)</[^>]+>',
            r'<title>([^<]+)</title>'
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, html_block, re.IGNORECASE)
            if match:
                facility.name = match.group(1).strip()
                break
        
        # Extract phone number
        phone_match = re.search(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', html_block)
        if phone_match:
            facility.phone = phone_match.group(0)
        
        # Extract email
        email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', html_block)
        if email_match:
            facility.email = email_match.group(0)
        
        # Extract address components
        address_match = re.search(r'\d+\s+[A-Za-z0-9\s,.-]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Boulevard|Blvd|Lane|Ln|Way|Circle|Cir)', html_block, re.IGNORECASE)
        if address_match:
            facility.address = address_match.group(0)
        
        # Extract ZIP code
        zip_match = re.search(r'\b\d{5}(?:-\d{4})?\b', html_block)
        if zip_match:
            facility.zip_code = zip_match.group(0)
        
        # Extract state
        state_match = re.search(r'\b[A-Z]{2}\b', html_block)
        if state_match:
            facility.state = state_match.group(0)
        
        # Extract facility type from context
        type_patterns = [
            r'(?:skilled|nursing|assisted|living|memory|care|rehabilitation|senior)',
            r'(?:facility|center|home|manor|community|residence)'
        ]
        
        for pattern in type_patterns:
            if re.search(pattern, html_block, re.IGNORECASE):
                # Determine facility type based on keywords
                if re.search(r'skilled.*nursing', html_block, re.IGNORECASE):
                    facility.facility_type = "Skilled Nursing"
                elif re.search(r'assisted.*living', html_block, re.IGNORECASE):
                    facility.facility_type = "Assisted Living"
                elif re.search(r'memory.*care', html_block, re.IGNORECASE):
                    facility.facility_type = "Memory Care"
                elif re.search(r'senior.*living', html_block, re.IGNORECASE):
                    facility.facility_type = "Senior Living"
                else:
                    facility.facility_type = "Healthcare Facility"
                break
        
        return facility if facility.name else None
    
    async def _extract_with_llm(self, crawler, url: str) -> List[FacilityInfo]:
        """LLM extraction as last resort (slower, potential hallucination)"""
        
        try:
            # Only use LLM for highly unstructured content
            llm_schema = {
                "type": "object",
                "properties": {
                    "facilities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "facility_type": {"type": "string"},
                                "address": {"type": "string"},
                                "city": {"type": "string"},
                                "state": {"type": "string"},
                                "zip_code": {"type": "string"},
                                "phone": {"type": "string"},
                                "confidence": {"type": "number"}
                            }
                        }
                    }
                }
            }
            
            extraction_strategy = LLMExtractionStrategy(
                provider="ollama/llama3.2:3b",  # Use local model to avoid costs
                schema=llm_schema,
                extraction_type="schema",
                instruction="Extract ONLY healthcare facilities that are explicitly mentioned. Do not infer or generate any information. Mark confidence level for each extraction."
            )
            
            result = await crawler.arun(
                url=url,
                extraction_strategy=extraction_strategy,
                bypass_cache=True
            )
            
            if result.success and result.extracted_content:
                extracted_data = json.loads(result.extracted_content)
                facilities = []
                
                for item in extracted_data.get("facilities", []):
                    # Only accept high-confidence LLM extractions
                    if item.get("confidence", 0) > 0.8:
                        facility = self._convert_to_facility_info(item, url, "llm_extraction")
                        if facility and self._is_valid_facility(facility):
                            facilities.append(facility)
                
                return facilities
            
        except Exception as e:
            self.logger.error(f"LLM extraction failed: {e}")
        
        return []
    
    def _convert_to_facility_info(self, data: Dict[str, Any], source_url: str, extraction_method: str) -> Optional[FacilityInfo]:
        """Convert extracted data to FacilityInfo object"""
        
        try:
            facility = FacilityInfo(source_url=source_url)
            
            # Map common field variations
            field_mappings = {
                "name": ["facility_name", "name", "title"],
                "facility_type": ["facility_type", "care_type", "service_type", "type"],
                "address": ["address", "street_address", "address_line_1"],
                "city": ["city", "locality"],
                "state": ["state", "region", "province"],
                "zip_code": ["zip_code", "postal_code", "zip"],
                "phone": ["phone", "telephone", "phone_number"],
                "email": ["email", "email_address"],
                "website": ["website", "url", "link"]
            }
            
            for facility_field, possible_keys in field_mappings.items():
                for key in possible_keys:
                    if key in data and data[key]:
                        setattr(facility, facility_field, str(data[key]).strip())
                        break
            
            # Handle special cases
            if "city_state_zip" in data:
                # Parse combined city, state, zip
                match = re.match(r'([^,]+),\s*([A-Z]{2})\s*(\d{5}(?:-\d{4})?)', data["city_state_zip"])
                if match:
                    facility.city = match.group(1).strip()
                    facility.state = match.group(2).strip()
                    facility.zip_code = match.group(3).strip()
            
            # Handle services/amenities arrays
            if "services" in data and isinstance(data["services"], list):
                facility.services_offered = [str(s).strip() for s in data["services"] if s]
            
            if "care_types" in data and isinstance(data["care_types"], list):
                facility.services_offered = [str(s).strip() for s in data["care_types"] if s]
            
            # Add extraction metadata
            facility.extraction_method = extraction_method
            
            return facility
            
        except Exception as e:
            self.logger.error(f"Error converting data to FacilityInfo: {e}")
            return None
    
    def _is_valid_facility(self, facility: FacilityInfo) -> bool:
        """Validate that extracted facility has minimum required information"""
        
        # Must have a name
        if not facility.name or len(facility.name.strip()) < 3:
            return False
        
        # Skip generic/template names
        generic_names = [
            "our facility", "healthcare facility", "nursing home", 
            "assisted living", "care center", "location", "facility"
        ]
        if facility.name.lower().strip() in generic_names:
            return False
        
        # Must have some location information
        if not any([facility.address, facility.city, facility.state, facility.zip_code]):
            return False
        
        return True
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction performance statistics"""
        
        total = self.extraction_stats["total_extractions"]
        if total == 0:
            return self.extraction_stats
        
        return {
            **self.extraction_stats,
            "schema_success_rate": self.extraction_stats["schema_success"] / total,
            "regex_success_rate": self.extraction_stats["regex_success"] / total,
            "llm_fallback_rate": self.extraction_stats["llm_fallback"] / total,
            "overall_success_rate": (
                self.extraction_stats["schema_success"] + 
                self.extraction_stats["regex_success"] + 
                self.extraction_stats["llm_fallback"]
            ) / total
        }

