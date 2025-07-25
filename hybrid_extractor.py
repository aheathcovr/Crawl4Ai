"""
Hybrid Extraction System
Combines LLM intelligence for navigation with fast algorithms for data extraction
Optimal balance of accuracy and speed
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy, RegexExtractionStrategy

from hybrid_llm_navigator import HybridLLMNavigator, NavigationTarget, SiteStructure
from schema_based_extractor import SchemaBasedExtractor, HealthcareSchemaLibrary
from free_validation import FreeValidationSystem
from healthcare_scraper import FacilityInfo
from crawler_manager import EnhancedCrawlerSession
from crawl_config import EXTRACTION_CONFIG, SCRAPING_LIMITS
from deduplicator import FacilityDeduplicator


@dataclass
class ExtractionResult:
    """Result of hybrid extraction process"""
    url: str
    facilities: List[Dict[str, Any]]
    extraction_method: str
    processing_time: float
    confidence_score: float
    llm_navigation_used: bool
    validation_reports: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class HybridExtractionStats:
    """Statistics for hybrid extraction performance"""
    total_urls_processed: int = 0
    llm_navigation_success: int = 0
    schema_extraction_success: int = 0
    regex_extraction_success: int = 0
    llm_extraction_fallback: int = 0
    total_facilities_found: int = 0
    total_processing_time: float = 0.0
    avg_confidence_score: float = 0.0


class HybridExtractor:
    """Hybrid extraction system combining LLM navigation with algorithmic extraction"""
    
    def __init__(self, 
                 openrouter_api_key: str,
                 enable_validation: bool = True,
                 llm_model_preference: str = "balanced",
                 crawler_session: Optional[EnhancedCrawlerSession] = None):
        
        self.navigator = HybridLLMNavigator(openrouter_api_key)
        self.schema_extractor = SchemaBasedExtractor(use_llm_fallback=True)
        self.validator = FreeValidationSystem() if enable_validation else None
        self.logger = logging.getLogger(__name__)
        self.crawler_session = crawler_session or EnhancedCrawlerSession()
        self.deduplicator = FacilityDeduplicator()
        
        # Model preferences for different tasks
        self.model_preferences = {
            "fast": {
                "navigation": "meta-llama/llama-3.1-8b-instruct:free",
                "analysis": "meta-llama/llama-3.1-8b-instruct:free",
                "extraction": "meta-llama/llama-3.1-70b-instruct"
            },
            "balanced": {
                "navigation": "meta-llama/llama-3.1-8b-instruct:free",
                "analysis": "microsoft/wizardlm-2-8x22b",
                "extraction": "meta-llama/llama-3.1-70b-instruct"
            },
            "precision": {
                "navigation": "microsoft/wizardlm-2-8x22b",
                "analysis": "anthropic/claude-3.5-sonnet",
                "extraction": "anthropic/claude-3.5-sonnet"
            }
        }
        
        self.current_preference = llm_model_preference
        self.stats = HybridExtractionStats()
    
    async def extract_facilities_hybrid(self, url: str) -> ExtractionResult:
        """Main hybrid extraction method"""
        
        start_time = time.time()
        self.logger.info(f"🚀 Starting hybrid extraction: {url}")
        
        try:
            # Phase 1: LLM-powered site analysis and navigation
            self.logger.info("🧠 Phase 1: LLM site analysis")
            site_structure = await self.navigator.analyze_site_structure(url)
            
            llm_navigation_used = site_structure.analysis_confidence > 0.5
            self.stats.total_urls_processed += 1
            
            if llm_navigation_used:
                self.stats.llm_navigation_success += 1
            
            # Phase 2: Smart target selection
            extraction_targets = self._select_best_targets(site_structure)
            
            # Phase 3: Fast algorithmic extraction
            all_facilities = []
            extraction_methods_used = []
            
            for target in extraction_targets:
                self.logger.info(f"🔍 Extracting from: {target.url}")
                
                target_facilities, method = await self._extract_from_target(target, site_structure)
                
                if target_facilities:
                    # Deduplicate before adding
                    unique_facilities = self.deduplicator.deduplicate_facilities(target_facilities)
                    all_facilities.extend(unique_facilities)
                    extraction_methods_used.append(method)
                    self.logger.info(f"✅ Found {len(unique_facilities)} unique facilities using {method} (deduped from {len(target_facilities)})")
                
                # Rate limiting is handled by Crawl4AI, no manual sleep needed
            
            # Phase 4: Validation (if enabled)
            validation_reports = []
            if self.validator and all_facilities:
                self.logger.info("🔍 Phase 4: Validation")
                validation_reports = await self.validator.validate_batch(all_facilities)
            
            # Calculate metrics
            processing_time = time.time() - start_time
            confidence_score = self._calculate_confidence_score(
                site_structure, all_facilities, extraction_methods_used, validation_reports
            )
            
            # Update statistics
            self.stats.total_facilities_found += len(all_facilities)
            self.stats.total_processing_time += processing_time
            
            # Determine primary extraction method
            primary_method = self._determine_primary_method(extraction_methods_used)
            
            result = ExtractionResult(
                url=url,
                facilities=all_facilities,
                extraction_method=primary_method,
                processing_time=processing_time,
                confidence_score=confidence_score,
                llm_navigation_used=llm_navigation_used,
                validation_reports=[asdict(r) for r in validation_reports],
                metadata={
                    "site_structure": asdict(site_structure),
                    "extraction_targets": [asdict(t) for t in extraction_targets],
                    "extraction_methods_used": extraction_methods_used,
                    "model_preference": self.current_preference
                }
            )
            
            self.logger.info(f"✅ Hybrid extraction complete: {len(all_facilities)} facilities, {processing_time:.1f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Hybrid extraction failed: {e}")
            
            # Fallback to basic extraction
            return await self._fallback_extraction(url, start_time)
    
    def _select_best_targets(self, site_structure: SiteStructure) -> List[NavigationTarget]:
        """Select the best targets for extraction based on LLM analysis"""
        
        # Sort targets by confidence and expected facility count
        targets = sorted(
            site_structure.navigation_targets,
            key=lambda t: (t.confidence, t.expected_facility_count or 0),
            reverse=True
        )
        
        # Select top targets, but limit to avoid overloading
        max_targets = EXTRACTION_CONFIG["max_targets_corporate"] if site_structure.site_type == "corporate_chain" else EXTRACTION_CONFIG["max_targets_single"]
        selected_targets = targets[:max_targets]
        
        # Always include the main URL if not already included
        main_url_included = any(t.url == site_structure.main_url for t in selected_targets)
        if not main_url_included:
            main_target = NavigationTarget(
                url=site_structure.main_url,
                page_type="main_page",
                confidence=0.7,
                description="Main page fallback"
            )
            selected_targets.append(main_target)
        
        self.logger.info(f"📋 Selected {len(selected_targets)} extraction targets")
        return selected_targets
    
    async def _extract_from_target(self, target: NavigationTarget, site_structure: SiteStructure) -> Tuple[List[Dict[str, Any]], str]:
        """Extract facilities from a specific target using the best method"""
        
        try:
            # Method 1: LLM-generated schema extraction (fastest for structured data)
            if target.css_selectors or target.extraction_hints:
                facilities, method = await self._extract_with_llm_schema(target)
                if facilities:
                    self.stats.schema_extraction_success += 1
                    return facilities, method
            
            # Method 2: Predefined schema extraction
            facilities, method = await self._extract_with_predefined_schema(target)
            if facilities:
                self.stats.schema_extraction_success += 1
                return facilities, method
            
            # Method 3: Regex extraction
            facilities, method = await self._extract_with_regex(target)
            if facilities:
                self.stats.regex_extraction_success += 1
                return facilities, method
            
            # Method 4: LLM extraction fallback (slowest, but most flexible)
            facilities, method = await self._extract_with_llm_fallback(target)
            if facilities:
                self.stats.llm_extraction_fallback += 1
                return facilities, method
            
            return [], "no_extraction"
            
        except Exception as e:
            self.logger.error(f"Extraction failed for {target.url}: {e}")
            return [], "error"
    
    async def _extract_with_llm_schema(self, target: NavigationTarget) -> Tuple[List[Dict[str, Any]], str]:
        """Extract using LLM-generated schema (hybrid approach)"""
        
        if not target.extraction_hints and not target.css_selectors:
            return [], "no_llm_schema"
        
        # Check for cached schema first
        cached_schema = self.crawler_session.get_cached_schema(target.url)
        
        if cached_schema:
            schema = cached_schema
            self.logger.info(f"Using cached schema for {self.crawler_session.get_domain(target.url)}")
        else:
            # Get page content to generate schema
            result = await self.crawler_session.crawl_with_cache(url=target.url)
            if not result.success:
                return [], "page_load_failed"
            
            # Generate optimized schema using LLM
            schema = await self.navigator.generate_extraction_schema(target.url, result.html[:4000])
            
            if not schema:
                return [], "schema_generation_failed"
            
            # Cache the schema for future use
            self.crawler_session.cache_schema(target.url, schema)
        
        # Use the generated/cached schema for fast extraction
        extraction_strategy = JsonCssExtractionStrategy(schema, verbose=False)
        
        extract_result = await self.crawler_session.crawl_with_cache(
            url=target.url,
            extraction_strategy=extraction_strategy,
            chunking_strategy=self.crawler_session.get_html_chunking_strategy()
        )
        
        if extract_result.success and extract_result.extracted_content:
            try:
                extracted_data = json.loads(extract_result.extracted_content)
                facilities_data = extracted_data.get(schema.get("name", "facilities"), [])
                
                if not isinstance(facilities_data, list):
                    facilities_data = [facilities_data]
                
                # Convert to standard format
                facilities = []
                for item in facilities_data:
                    if self._is_valid_facility_data(item):
                        facility = self._normalize_facility_data(item, target.url)
                        facilities.append(facility)
                
                return facilities, "llm_generated_schema"
                
            except json.JSONDecodeError:
                pass
        
        return [], "llm_schema_extraction_failed"
    
    async def _extract_with_predefined_schema(self, target: NavigationTarget) -> Tuple[List[Dict[str, Any]], str]:
        """Extract using predefined schemas"""
        
        # Use the schema-based extractor
        facilities = await self.schema_extractor.extract_facilities(target.url)
        
        if facilities:
            facility_dicts = []
            for facility in facilities:
                facility_dict = self._facility_info_to_dict(facility)
                facility_dicts.append(facility_dict)
            
            return facility_dicts, "predefined_schema"
        
        return [], "predefined_schema_failed"
    
    async def _extract_with_regex(self, target: NavigationTarget) -> Tuple[List[Dict[str, Any]], str]:
        """Extract using regex patterns"""
        
        async with AsyncWebCrawler(headless=True, verbose=False) as crawler:
            result = await crawler.arun(url=target.url)
            
            if not result.success:
                return [], "page_load_failed"
            
            # Use comprehensive regex patterns
            regex_patterns = {
                "facility_names": r'<h[1-6][^>]*>([^<]*(?:care|center|living|manor|home|facility|community)[^<]*)</h[1-6]>',
                "phone_numbers": r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
                "addresses": r'\d+\s+[A-Za-z0-9\s,.-]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Boulevard|Blvd|Lane|Ln|Way|Circle|Cir)',
                "emails": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
                "zip_codes": r'\b\d{5}(?:-\d{4})?\b'
            }
            
            facilities = []
            html_content = result.html
            
            # Extract facility blocks
            facility_blocks = self._extract_facility_blocks(html_content)
            
            for block in facility_blocks:
                facility_data = {}
                
                # Extract data from each block
                for field, pattern in regex_patterns.items():
                    matches = re.findall(pattern, block, re.IGNORECASE)
                    if matches:
                        if field == "facility_names":
                            facility_data["name"] = matches[0].strip()
                        elif field == "phone_numbers":
                            facility_data["phone"] = matches[0]
                        elif field == "addresses":
                            facility_data["address"] = matches[0]
                        elif field == "emails":
                            facility_data["email"] = matches[0]
                        elif field == "zip_codes":
                            facility_data["zip_code"] = matches[0]
                
                if self._is_valid_facility_data(facility_data):
                    facility_data["source_url"] = target.url
                    facility_data["extraction_method"] = "regex"
                    facilities.append(facility_data)
            
            return facilities, "regex_extraction"
    
    async def _extract_with_llm_fallback(self, target: NavigationTarget) -> Tuple[List[Dict[str, Any]], str]:
        """LLM extraction as final fallback"""
        
        # Use the schema extractor's LLM fallback
        facilities = await self.schema_extractor._extract_with_llm(None, target.url)
        
        if facilities:
            facility_dicts = []
            for facility in facilities:
                facility_dict = self._facility_info_to_dict(facility)
                facility_dicts.append(facility_dict)
            
            return facility_dicts, "llm_fallback"
        
        return [], "llm_fallback_failed"
    
    def _extract_facility_blocks(self, html_content: str) -> List[str]:
        """Extract potential facility blocks from HTML"""
        
        import re
        
        # Patterns for facility containers
        block_patterns = [
            r'<div[^>]*(?:facility|location|community|center)[^>]*>.*?</div>',
            r'<article[^>]*>.*?</article>',
            r'<section[^>]*>.*?</section>',
            r'<li[^>]*(?:facility|location)[^>]*>.*?</li>'
        ]
        
        blocks = []
        for pattern in block_patterns:
            matches = re.findall(pattern, html_content, re.IGNORECASE | re.DOTALL)
            blocks.extend(matches)
        
        # If no blocks found, split by common separators
        if not blocks:
            # Split by major HTML elements
            potential_blocks = re.split(r'<(?:div|section|article|li)[^>]*>', html_content)
            blocks = [block for block in potential_blocks if len(block) > 100]
        
        return blocks[:50]  # Limit to avoid processing too many blocks
    
    def _is_valid_facility_data(self, data: Dict[str, Any]) -> bool:
        """Check if extracted data represents a valid facility"""
        
        if not isinstance(data, dict):
            return False
        
        # Must have a name
        name = data.get("name", "").strip()
        if not name or len(name) < 3:
            return False
        
        # Skip generic names
        generic_names = ["facility", "location", "center", "home", "care"]
        if name.lower() in generic_names:
            return False
        
        # Must have some contact or location info
        has_contact_info = any([
            data.get("phone"),
            data.get("email"),
            data.get("address"),
            data.get("city"),
            data.get("zip_code")
        ])
        
        return has_contact_info
    
    def _normalize_facility_data(self, data: Dict[str, Any], source_url: str) -> Dict[str, Any]:
        """Normalize facility data to standard format"""
        
        normalized = {
            "name": data.get("facility_name") or data.get("name", ""),
            "facility_type": data.get("facility_type", "Healthcare Facility"),
            "address": data.get("address", ""),
            "city": data.get("city", ""),
            "state": data.get("state", ""),
            "zip_code": data.get("zip_code", ""),
            "phone": data.get("phone", ""),
            "email": data.get("email", ""),
            "website": data.get("website", ""),
            "administrator": data.get("administrator", ""),
            "beds": data.get("beds", ""),
            "services_offered": data.get("services", []) or data.get("services_offered", []),
            "source_url": source_url,
            "scraping_date": datetime.now().isoformat(),
            "extraction_method": data.get("extraction_method", "hybrid")
        }
        
        # Clean up empty strings
        for key, value in normalized.items():
            if isinstance(value, str):
                normalized[key] = value.strip()
        
        return normalized
    
    def _facility_info_to_dict(self, facility: FacilityInfo) -> Dict[str, Any]:
        """Convert FacilityInfo object to dictionary"""
        
        return {
            "name": facility.name or "",
            "facility_type": facility.facility_type or "Healthcare Facility",
            "address": facility.address or "",
            "city": facility.city or "",
            "state": facility.state or "",
            "zip_code": facility.zip_code or "",
            "phone": facility.phone or "",
            "email": facility.email or "",
            "website": facility.website or "",
            "administrator": facility.administrator or "",
            "beds": facility.beds or "",
            "services_offered": facility.services_offered or [],
            "source_url": facility.source_url,
            "scraping_date": datetime.now().isoformat(),
            "extraction_method": getattr(facility, "extraction_method", "schema_based")
        }
    
    def _calculate_confidence_score(self, 
                                  site_structure: SiteStructure,
                                  facilities: List[Dict[str, Any]],
                                  extraction_methods: List[str],
                                  validation_reports: List[Any]) -> float:
        """Calculate overall confidence score for the extraction"""
        
        scores = []
        
        # Site analysis confidence
        scores.append(site_structure.analysis_confidence)
        
        # Extraction method confidence
        method_scores = {
            "llm_generated_schema": 0.9,
            "predefined_schema": 0.85,
            "regex_extraction": 0.7,
            "llm_fallback": 0.6
        }
        
        if extraction_methods:
            avg_method_score = sum(method_scores.get(method, 0.5) for method in extraction_methods) / len(extraction_methods)
            scores.append(avg_method_score)
        
        # Validation confidence
        if validation_reports:
            validation_scores = [r.overall_confidence for r in validation_reports if hasattr(r, 'overall_confidence')]
            if validation_scores:
                avg_validation_score = sum(validation_scores) / len(validation_scores)
                scores.append(avg_validation_score)
        
        # Data completeness score
        if facilities:
            completeness_scores = []
            for facility in facilities:
                required_fields = ["name", "address", "phone"]
                optional_fields = ["city", "state", "zip_code", "email", "website"]
                
                required_score = sum(1 for field in required_fields if facility.get(field))
                optional_score = sum(1 for field in optional_fields if facility.get(field))
                
                completeness = (required_score / len(required_fields)) * 0.7 + (optional_score / len(optional_fields)) * 0.3
                completeness_scores.append(completeness)
            
            avg_completeness = sum(completeness_scores) / len(completeness_scores)
            scores.append(avg_completeness)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _determine_primary_method(self, extraction_methods: List[str]) -> str:
        """Determine the primary extraction method used"""
        
        if not extraction_methods:
            return "no_extraction"
        
        # Count method usage
        method_counts = {}
        for method in extraction_methods:
            method_counts[method] = method_counts.get(method, 0) + 1
        
        # Return most used method
        return max(method_counts.items(), key=lambda x: x[1])[0]
    
    async def _fallback_extraction(self, url: str, start_time: float) -> ExtractionResult:
        """Fallback extraction when hybrid method fails"""
        
        self.logger.info("🔄 Using fallback extraction")
        
        try:
            # Use basic schema extraction
            facilities = await self.schema_extractor.extract_facilities(url)
            facility_dicts = [self._facility_info_to_dict(f) for f in facilities]
            
            processing_time = time.time() - start_time
            
            return ExtractionResult(
                url=url,
                facilities=facility_dicts,
                extraction_method="fallback_schema",
                processing_time=processing_time,
                confidence_score=0.4,
                llm_navigation_used=False,
                validation_reports=[],
                metadata={"fallback_used": True}
            )
            
        except Exception as e:
            self.logger.error(f"Fallback extraction also failed: {e}")
            
            processing_time = time.time() - start_time
            
            return ExtractionResult(
                url=url,
                facilities=[],
                extraction_method="failed",
                processing_time=processing_time,
                confidence_score=0.0,
                llm_navigation_used=False,
                validation_reports=[],
                metadata={"error": str(e)}
            )
    
    def switch_model_preference(self, preference: str):
        """Switch LLM model preference"""
        
        if preference in self.model_preferences:
            self.current_preference = preference
            self.logger.info(f"Switched to {preference} model preference")
        else:
            self.logger.warning(f"Unknown model preference: {preference}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        
        stats_dict = asdict(self.stats)
        
        if self.stats.total_urls_processed > 0:
            stats_dict["avg_facilities_per_url"] = self.stats.total_facilities_found / self.stats.total_urls_processed
            stats_dict["avg_processing_time"] = self.stats.total_processing_time / self.stats.total_urls_processed
            stats_dict["llm_navigation_success_rate"] = self.stats.llm_navigation_success / self.stats.total_urls_processed
        
        # Method success rates
        total_extractions = (
            self.stats.schema_extraction_success + 
            self.stats.regex_extraction_success + 
            self.stats.llm_extraction_fallback
        )
        
        if total_extractions > 0:
            stats_dict["schema_success_rate"] = self.stats.schema_extraction_success / total_extractions
            stats_dict["regex_success_rate"] = self.stats.regex_extraction_success / total_extractions
            stats_dict["llm_fallback_rate"] = self.stats.llm_extraction_fallback / total_extractions
        
        return stats_dict
    
    async def close(self):
        """Close all connections"""
        await self.navigator.close()
        if self.validator:
            await self.validator.close()


# CLI interface for hybrid extraction
async def main():
    """CLI interface for hybrid extraction"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Hybrid LLM + Algorithm Healthcare Facility Extractor')
    parser.add_argument('--url', required=True, help='URL to extract facilities from')
    parser.add_argument('--api-key', help='OpenRouter API key (or set OPENROUTER_API_KEY env var)')
    parser.add_argument('--model-preference', choices=['fast', 'balanced', 'precision'], default='balanced')
    parser.add_argument('--output', help='Output JSON file')
    parser.add_argument('--validate', action='store_true', help='Enable validation')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Get API key
    api_key = args.api_key or "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"
    
    if not api_key:
        print("Error: OpenRouter API key required")
        return
    
    # Initialize hybrid extractor
    extractor = HybridExtractor(
        openrouter_api_key=api_key,
        enable_validation=args.validate,
        llm_model_preference=args.model_preference
    )
    
    try:
        # Extract facilities
        print(f"🚀 Starting hybrid extraction: {args.url}")
        result = await extractor.extract_facilities_hybrid(args.url)
        
        # Show results
        print(f"\n✅ Extraction completed!")
        print(f"   Facilities found: {len(result.facilities)}")
        print(f"   Processing time: {result.processing_time:.1f}s")
        print(f"   Confidence score: {result.confidence_score:.2f}")
        print(f"   Extraction method: {result.extraction_method}")
        print(f"   LLM navigation used: {result.llm_navigation_used}")
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(asdict(result), f, indent=2)
            print(f"   Results saved to: {args.output}")
        
        # Show performance stats
        stats = extractor.get_performance_stats()
        print(f"\n📊 Performance Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")
    
    finally:
        await extractor.close()


if __name__ == "__main__":
    asyncio.run(main())

