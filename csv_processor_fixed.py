"""
CSV Processing System for Healthcare Corporate Chains
Handles large-scale scraping operations with priority management
"""

import csv
import json
import logging
import os
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd


@dataclass
class CorporateChain:
    """Data class for healthcare corporate chain information"""
    corporation_name: str
    primary_url: str
    secondary_urls: str = ""
    facility_types: str = ""
    priority: int = 1
    notes: str = ""
    estimated_facilities: str = ""
    last_updated: str = ""
    
    def get_all_urls(self) -> List[str]:
        """Get all URLs for this corporation"""
        urls = [self.primary_url]
        if self.secondary_urls:
            secondary = [url.strip() for url in self.secondary_urls.split('|') if url.strip()]
            urls.extend(secondary)
        return urls
    
    def get_facility_types_list(self) -> List[str]:
        """Get facility types as a list"""
        if not self.facility_types:
            return []
        return [ft.strip() for ft in self.facility_types.split('|') if ft.strip()]
    
    def is_due_for_update(self, days_threshold: int = 30) -> bool:
        """Check if this chain is due for an update"""
        if not self.last_updated:
            return True
        
        try:
            last_update = datetime.strptime(self.last_updated, '%Y-%m-%d')
            threshold_date = datetime.now() - timedelta(days=days_threshold)
            return last_update < threshold_date
        except:
            return True


class CorporateChainManager:
    """Manager for healthcare corporate chains CSV processing"""
    
    def __init__(self, csv_file: str = "corporate_chains.csv"):
        self.csv_file = csv_file
        self.logger = logging.getLogger(__name__)
        self.chains: List[CorporateChain] = []
        
        # CSV size limits and recommendations
        self.max_recommended_chains = 1000  # Reasonable for most use cases
        self.max_absolute_chains = 10000    # Technical limit
        self.max_file_size_mb = 50          # 50MB CSV limit
    
    def load_chains(self) -> List[CorporateChain]:
        """Load corporate chains from CSV file"""
        
        if not os.path.exists(self.csv_file):
            self.logger.error(f"CSV file not found: {self.csv_file}")
            return []
        
        # Check file size
        file_size_mb = os.path.getsize(self.csv_file) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            self.logger.warning(f"Large CSV file: {file_size_mb:.1f}MB (max recommended: {self.max_file_size_mb}MB)")
        
        chains = []
        
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row_num, row in enumerate(reader, 1):
                    try:
                        # Validate required fields
                        if not row.get('corporation_name') or not row.get('primary_url'):
                            self.logger.warning(f"Row {row_num}: Missing required fields")
                            continue
                        
                        # Create chain object
                        chain = CorporateChain(
                            corporation_name=row.get('corporation_name', '').strip(),
                            primary_url=row.get('primary_url', '').strip(),
                            secondary_urls=row.get('secondary_urls', '').strip(),
                            facility_types=row.get('facility_types', '').strip(),
                            priority=int(row.get('priority', 1)),
                            notes=row.get('notes', '').strip(),
                            estimated_facilities=row.get('estimated_facilities', '').strip(),
                            last_updated=row.get('last_updated', '').strip()
                        )
                        
                        chains.append(chain)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing row {row_num}: {e}")
                        continue
            
            self.chains = chains
            self.logger.info(f"Loaded {len(chains)} corporate chains from {self.csv_file}")
            
            # Provide size recommendations
            if len(chains) > self.max_recommended_chains:
                self.logger.warning(f"Large number of chains: {len(chains)} (recommended max: {self.max_recommended_chains})")
                self.logger.info("Consider using priority filtering or batch processing")
            
            return chains
            
        except Exception as e:
            self.logger.error(f"Error loading CSV file: {e}")
            return []
    
    def get_chains_by_priority(self, max_priority: int = 3) -> List[CorporateChain]:
        """Get chains filtered by priority level"""
        return [chain for chain in self.chains if chain.priority <= max_priority]
    
    def get_chains_by_facility_type(self, facility_type: str) -> List[CorporateChain]:
        """Get chains that operate specific facility types"""
        matching_chains = []
        for chain in self.chains:
            if facility_type.lower() in [ft.lower() for ft in chain.get_facility_types_list()]:
                matching_chains.append(chain)
        return matching_chains
    
    def get_chains_due_for_update(self, days_threshold: int = 30) -> List[CorporateChain]:
        """Get chains that haven't been updated recently"""
        return [chain for chain in self.chains if chain.is_due_for_update(days_threshold)]
    
    def get_processing_batches(self, batch_size: int = 10) -> List[List[CorporateChain]]:
        """Split chains into processing batches"""
        batches = []
        for i in range(0, len(self.chains), batch_size):
            batch = self.chains[i:i + batch_size]
            batches.append(batch)
        return batches
    
    def update_chain_timestamp(self, corporation_name: str):
        """Update the last_updated timestamp for a chain"""
        # This would update the CSV file - implementation depends on your needs
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded chains"""
        if not self.chains:
            return {}
        
        # Priority distribution
        priority_counts = {}
        for chain in self.chains:
            priority_counts[chain.priority] = priority_counts.get(chain.priority, 0) + 1
        
        # Facility type distribution
        facility_type_counts = {}
        for chain in self.chains:
            for ft in chain.get_facility_types_list():
                facility_type_counts[ft] = facility_type_counts.get(ft, 0) + 1
        
        # Estimated facilities
        total_estimated = 0
        for chain in self.chains:
            if chain.estimated_facilities:
                # Extract number from strings like "200+", "100-150", etc.
                import re
                numbers = re.findall(r'\d+', chain.estimated_facilities)
                if numbers:
                    total_estimated += int(numbers[0])
        
        return {
            'total_chains': len(self.chains),
            'priority_distribution': priority_counts,
            'facility_type_distribution': facility_type_counts,
            'estimated_total_facilities': total_estimated,
            'chains_due_for_update': len(self.get_chains_due_for_update()),
            'file_size_mb': round(os.path.getsize(self.csv_file) / (1024 * 1024), 2)
        }
    
    def export_results_summary(self, results: List[Dict], output_file: str):
        """Export scraping results summary"""
        
        summary_data = []
        
        for result in results:
            summary_data.append({
                'corporation_name': result.get('corporation_name', ''),
                'url_scraped': result.get('url', ''),
                'facilities_found': result.get('facilities_count', 0),
                'scraping_success': result.get('success', False),
                'scraping_date': result.get('scraping_date', ''),
                'processing_time_minutes': result.get('processing_time_minutes', 0),
                'error_message': result.get('error', ''),
                'output_files': '|'.join(result.get('files', []))
            })
        
        # Save as CSV
        df = pd.DataFrame(summary_data)
        df.to_csv(output_file, index=False)
        
        self.logger.info(f"Results summary exported to {output_file}")


class CSVProcessor:
    """
    Simple CSV processor for backward compatibility with main_hybrid.py and other scripts
    This class provides the interface that the main scripts expect
    """
    
    def __init__(self, csv_file: str = "corporate_chains.csv"):
        self.csv_file = csv_file
        self.manager = CorporateChainManager(csv_file)
        self.chains: List[CorporateChain] = []
        self.logger = logging.getLogger(__name__)
    
    def load_chains(self) -> List[CorporateChain]:
        """Load chains from CSV file"""
        self.chains = self.manager.load_chains()
        return self.chains
    
    def get_chains_by_priority(self, priority: int) -> List[CorporateChain]:
        """Get chains filtered by priority level"""
        if not self.chains:
            self.load_chains()
        return self.manager.get_chains_by_priority(priority)
    
    def get_chains_by_facility_type(self, facility_type: str) -> List[CorporateChain]:
        """Get chains that operate specific facility types"""
        if not self.chains:
            self.load_chains()
        return self.manager.get_chains_by_facility_type(facility_type)
    
    def get_urls_for_processing(self, priority: int = None, facility_type: str = None) -> List[str]:
        """Get URLs for processing based on filters"""
        if not self.chains:
            self.load_chains()
        
        chains = self.chains
        
        # Apply priority filter
        if priority:
            chains = [c for c in chains if c.priority <= priority]
        
        # Apply facility type filter
        if facility_type:
            chains = [c for c in chains if facility_type.lower() in [ft.lower() for ft in c.get_facility_types_list()]]
        
        # Extract all URLs
        urls = []
        for chain in chains:
            urls.extend(chain.get_all_urls())
        
        return urls
    
    def get_chain_info(self, url: str) -> Optional[CorporateChain]:
        """Get chain information for a specific URL"""
        if not self.chains:
            self.load_chains()
        
        for chain in self.chains:
            if url in chain.get_all_urls():
                return chain
        return None
    
    def get_processing_plan(self, priority: int = None, facility_type: str = None, batch_size: int = 10) -> Dict[str, Any]:
        """Get a processing plan for the filtered chains"""
        if not self.chains:
            self.load_chains()
        
        # Get filtered chains
        chains = self.chains
        
        if priority:
            chains = [c for c in chains if c.priority <= priority]
        
        if facility_type:
            chains = [c for c in chains if facility_type.lower() in [ft.lower() for ft in c.get_facility_types_list()]]
        
        # Calculate estimated processing time and facilities
        total_estimated_facilities = 0
        for chain in chains:
            if chain.estimated_facilities:
                import re
                numbers = re.findall(r'\d+', chain.estimated_facilities)
                if numbers:
                    total_estimated_facilities += int(numbers[0])
        
        # Create batches
        batches = []
        for i in range(0, len(chains), batch_size):
            batch = chains[i:i + batch_size]
            batches.append(batch)
        
        return {
            'total_chains': len(chains),
            'total_batches': len(batches),
            'batch_size': batch_size,
            'estimated_facilities': total_estimated_facilities,
            'estimated_time_hours': len(chains) * 0.5,  # Rough estimate: 30 minutes per chain
            'chains': [{'name': c.corporation_name, 'url': c.primary_url, 'priority': c.priority} for c in chains]
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded chains"""
        if not self.chains:
            self.load_chains()
        return self.manager.get_statistics()


class BatchProcessor:
    """Batch processor for large-scale corporate chain scraping"""
    
    def __init__(self, chain_manager: CorporateChainManager, output_dir: str = "./batch_output"):
        self.chain_manager = chain_manager
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Create output directory structure
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/individual_chains", exist_ok=True)
        os.makedirs(f"{output_dir}/consolidated", exist_ok=True)
        os.makedirs(f"{output_dir}/logs", exist_ok=True)
    
    async def process_all_chains(self, 
                                use_llm: bool = True, 
                                batch_size: int = 5,
                                priority_filter: int = 3,
                                delay_between_batches: int = 60) -> Dict[str, Any]:
        """Process all chains in batches"""
        
        # Filter chains by priority
        chains_to_process = self.chain_manager.get_chains_by_priority(priority_filter)
        
        self.logger.info(f"Processing {len(chains_to_process)} chains (priority <= {priority_filter})")
        
        # Split into batches
        batches = []
        for i in range(0, len(chains_to_process), batch_size):
            batch = chains_to_process[i:i + batch_size]
            batches.append(batch)
        
        self.logger.info(f"Created {len(batches)} batches of size {batch_size}")
        
        # Process batches
        all_results = []
        total_facilities = 0
        
        for batch_num, batch in enumerate(batches, 1):
            self.logger.info(f"Processing batch {batch_num}/{len(batches)}")
            
            batch_results = await self.process_batch(batch, batch_num, use_llm)
            all_results.extend(batch_results)
            
            # Count facilities
            batch_facilities = sum(r.get('facilities_count', 0) for r in batch_results)
            total_facilities += batch_facilities
            
            self.logger.info(f"Batch {batch_num} completed: {batch_facilities} facilities found")
            
            # Delay between batches (except for last batch)
            if batch_num < len(batches):
                self.logger.info(f"Waiting {delay_between_batches} seconds before next batch...")
                await asyncio.sleep(delay_between_batches)
        
        # Consolidate results
        consolidated_file = await self.consolidate_results(all_results)
        
        # Generate summary
        summary = {
            'total_chains_processed': len(chains_to_process),
            'total_batches': len(batches),
            'total_facilities_found': total_facilities,
            'successful_chains': len([r for r in all_results if r.get('success', False)]),
            'failed_chains': len([r for r in all_results if not r.get('success', False)]),
            'consolidated_file': consolidated_file,
            'processing_date': datetime.now().isoformat(),
            'individual_results': all_results
        }
        
        # Save summary
        summary_file = f"{self.output_dir}/batch_processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Export CSV summary
        csv_summary_file = f"{self.output_dir}/batch_processing_summary.csv"
        self.chain_manager.export_results_summary(all_results, csv_summary_file)
        
        self.logger.info(f"Batch processing completed: {total_facilities} total facilities found")
        
        return summary
    
    async def process_batch(self, batch: List[CorporateChain], batch_num: int, use_llm: bool) -> List[Dict]:
        """Process a single batch of chains"""
        
        batch_results = []
        
        for chain in batch:
            self.logger.info(f"Processing: {chain.corporation_name}")
            
            start_time = datetime.now()
            
            try:
                # Import the scraper here to avoid circular imports
                if use_llm:
                    from main_optimized import optimized_scrape
                    
                    # Create chain-specific output directory
                    chain_output_dir = f"{self.output_dir}/individual_chains/{chain.corporation_name.replace(' ', '_')}"
                    os.makedirs(chain_output_dir, exist_ok=True)
                    
                    # Process primary URL
                    result = await optimized_scrape(chain.primary_url, chain_output_dir, use_llm=True)
                    
                    # Process secondary URLs if any
                    secondary_results = []
                    for url in chain.get_all_urls()[1:]:  # Skip primary URL
                        secondary_result = await optimized_scrape(url, chain_output_dir, use_llm=True)
                        secondary_results.append(secondary_result)
                
                else:
                    # Traditional scraping
                    from healthcare_scraper import HealthcareFacilityScraper
                    
                    chain_output_dir = f"{self.output_dir}/individual_chains/{chain.corporation_name.replace(' ', '_')}"
                    os.makedirs(chain_output_dir, exist_ok=True)
                    
                    scraper = HealthcareFacilityScraper(chain.primary_url, chain_output_dir)
                    facilities = await scraper.scrape_facilities()
                    
                    result = {
                        'success': len(facilities) > 0,
                        'facilities_count': len(facilities),
                        'files': [scraper.save_results('json'), scraper.save_results('csv')] if facilities else []
                    }
                
                processing_time = (datetime.now() - start_time).total_seconds() / 60
                
                # Add metadata to result
                result.update({
                    'corporation_name': chain.corporation_name,
                    'url': chain.primary_url,
                    'facility_types': chain.get_facility_types_list(),
                    'priority': chain.priority,
                    'estimated_facilities': chain.estimated_facilities,
                    'processing_time_minutes': round(processing_time, 2),
                    'scraping_date': datetime.now().isoformat(),
                    'batch_number': batch_num
                })
                
                batch_results.append(result)
                
                self.logger.info(f"Completed {chain.corporation_name}: {result.get('facilities_count', 0)} facilities")
                
            except Exception as e:
                self.logger.error(f"Error processing {chain.corporation_name}: {e}")
                
                batch_results.append({
                    'corporation_name': chain.corporation_name,
                    'url': chain.primary_url,
                    'success': False,
                    'facilities_count': 0,
                    'error': str(e),
                    'processing_time_minutes': (datetime.now() - start_time).total_seconds() / 60,
                    'scraping_date': datetime.now().isoformat(),
                    'batch_number': batch_num
                })
        
        return batch_results
    
    async def consolidate_results(self, all_results: List[Dict]) -> str:
        """Consolidate all individual results into master files"""
        
        # Collect all facilities from successful scrapes
        all_facilities = []
        
        for result in all_results:
            if result.get('success') and result.get('files'):
                # Read JSON files and combine
                for file_path in result.get('files', []):
                    if file_path.endswith('.json'):
                        try:
                            with open(file_path, 'r') as f:
                                facilities_data = json.load(f)
                                
                                # Add corporation metadata to each facility
                                for facility in facilities_data:
                                    facility['source_corporation'] = result.get('corporation_name')
                                    facility['corporation_priority'] = result.get('priority')
                                    facility['corporation_facility_types'] = result.get('facility_types', [])
                                
                                all_facilities.extend(facilities_data)
                        
                        except Exception as e:
                            self.logger.error(f"Error reading {file_path}: {e}")
        
        # Save consolidated results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON format
        json_file = f"{self.output_dir}/consolidated/all_facilities_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(all_facilities, f, indent=2, default=str)
        
        # CSV format
        if all_facilities:
            df = pd.json_normalize(all_facilities)
            csv_file = f"{self.output_dir}/consolidated/all_facilities_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
        
        self.logger.info(f"Consolidated {len(all_facilities)} facilities into {json_file}")
        
        return json_file


# Convenience functions for quick access
def load_corporate_chains(csv_file: str = "corporate_chains.csv") -> List[CorporateChain]:
    """Quick function to load corporate chains"""
    processor = CSVProcessor(csv_file)
    return processor.load_chains()


def get_priority_chains(priority: int, csv_file: str = "corporate_chains.csv") -> List[CorporateChain]:
    """Quick function to get chains by priority"""
    processor = CSVProcessor(csv_file)
    return processor.get_chains_by_priority(priority)


def get_processing_urls(priority: int = None, facility_type: str = None, csv_file: str = "corporate_chains.csv") -> List[str]:
    """Quick function to get URLs for processing"""
    processor = CSVProcessor(csv_file)
    return processor.get_urls_for_processing(priority, facility_type)


if __name__ == "__main__":
    # Test the CSV processor
    processor = CSVProcessor()
    chains = processor.load_chains()
    
    print(f"Loaded {len(chains)} corporate chains")
    
    # Show statistics
    stats = processor.get_statistics()
    print("Statistics:", json.dumps(stats, indent=2))
    
    # Show processing plan for priority 1-2 chains
    plan = processor.get_processing_plan(priority=2)
    print("Processing Plan:", json.dumps(plan, indent=2))

