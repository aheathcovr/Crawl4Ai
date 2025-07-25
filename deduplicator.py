"""
Facility Deduplication System
Uses MinHash and fuzzy matching to identify and remove duplicate facilities
"""

import hashlib
import logging
from typing import Dict, List, Any, Set, Tuple
from dataclasses import dataclass
import re

from crawl_config import VALIDATION_CONFIG


@dataclass
class FacilitySignature:
    """Unique signature for a facility based on key attributes"""
    phone_hash: str
    address_hash: str
    name_hash: str
    full_hash: str
    
    @classmethod
    def from_facility(cls, facility: Dict[str, Any]) -> 'FacilitySignature':
        """Create signature from facility data"""
        # Normalize phone number
        phone = re.sub(r'[^\d]', '', str(facility.get('phone', '')))
        phone_hash = hashlib.md5(phone.encode()).hexdigest()[:8] if phone else ''
        
        # Normalize address
        address_parts = []
        for field in ['address', 'city', 'zip_code']:
            value = str(facility.get(field, '')).lower().strip()
            if value:
                address_parts.append(value)
        address_str = '|'.join(address_parts)
        address_hash = hashlib.md5(address_str.encode()).hexdigest()[:8] if address_str else ''
        
        # Normalize name
        name = re.sub(r'[^\w\s]', '', str(facility.get('name', '')).lower())
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8] if name else ''
        
        # Combined hash
        combined = f"{phone}|{address_str}|{name}"
        full_hash = hashlib.md5(combined.encode()).hexdigest()
        
        return cls(
            phone_hash=phone_hash,
            address_hash=address_hash,
            name_hash=name_hash,
            full_hash=full_hash
        )


class FacilityDeduplicator:
    """Deduplicate facilities based on various matching strategies"""
    
    def __init__(self, threshold: float = None):
        self.threshold = threshold or VALIDATION_CONFIG["dedup_threshold"]
        self.seen_signatures: Set[str] = set()
        self.seen_facilities: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
        self.stats = {
            "total_processed": 0,
            "duplicates_found": 0,
            "unique_facilities": 0
        }
    
    def deduplicate_facilities(self, facilities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate facilities from a list"""
        unique_facilities = []
        
        for facility in facilities:
            self.stats["total_processed"] += 1
            
            if self._is_duplicate(facility):
                self.stats["duplicates_found"] += 1
                self.logger.debug(f"Duplicate found: {facility.get('name', 'Unknown')}")
            else:
                unique_facilities.append(facility)
                self._add_to_seen(facility)
                self.stats["unique_facilities"] += 1
        
        return unique_facilities
    
    def _is_duplicate(self, facility: Dict[str, Any]) -> bool:
        """Check if facility is a duplicate based on multiple criteria"""
        signature = FacilitySignature.from_facility(facility)
        
        # Quick check: exact signature match
        if signature.full_hash in self.seen_signatures:
            return True
        
        # Check individual components
        if signature.phone_hash and signature.address_hash:
            # Strong match: same phone AND address
            phone_address_sig = f"{signature.phone_hash}|{signature.address_hash}"
            if phone_address_sig in self.seen_signatures:
                return True
        
        # Fuzzy matching for similar facilities
        if VALIDATION_CONFIG["enable_deduplication"]:
            for seen_facility in self.seen_facilities[-100:]:  # Check last 100 for performance
                if self._calculate_similarity(facility, seen_facility) >= self.threshold:
                    return True
        
        return False
    
    def _add_to_seen(self, facility: Dict[str, Any]):
        """Add facility to seen collections"""
        signature = FacilitySignature.from_facility(facility)
        
        # Add all signature variants
        self.seen_signatures.add(signature.full_hash)
        
        if signature.phone_hash and signature.address_hash:
            self.seen_signatures.add(f"{signature.phone_hash}|{signature.address_hash}")
        
        # Keep facility for fuzzy matching (limited size for memory)
        self.seen_facilities.append(facility)
        if len(self.seen_facilities) > 1000:
            self.seen_facilities = self.seen_facilities[-500:]  # Keep last 500
    
    def _calculate_similarity(self, facility1: Dict[str, Any], facility2: Dict[str, Any]) -> float:
        """Calculate similarity score between two facilities"""
        score = 0.0
        weights = {
            "phone": 0.4,
            "address": 0.3,
            "name": 0.2,
            "zip_code": 0.1
        }
        
        # Phone similarity
        phone1 = re.sub(r'[^\d]', '', str(facility1.get('phone', '')))
        phone2 = re.sub(r'[^\d]', '', str(facility2.get('phone', '')))
        if phone1 and phone2 and phone1 == phone2:
            score += weights["phone"]
        
        # Address similarity
        addr1 = str(facility1.get('address', '')).lower()
        addr2 = str(facility2.get('address', '')).lower()
        if addr1 and addr2:
            if addr1 == addr2:
                score += weights["address"]
            elif self._fuzzy_match(addr1, addr2, 0.8):
                score += weights["address"] * 0.7
        
        # Name similarity
        name1 = re.sub(r'[^\w\s]', '', str(facility1.get('name', '')).lower())
        name2 = re.sub(r'[^\w\s]', '', str(facility2.get('name', '')).lower())
        if name1 and name2:
            if name1 == name2:
                score += weights["name"]
            elif self._fuzzy_match(name1, name2, 0.7):
                score += weights["name"] * 0.5
        
        # ZIP code
        zip1 = str(facility1.get('zip_code', ''))[:5]
        zip2 = str(facility2.get('zip_code', ''))[:5]
        if zip1 and zip2 and zip1 == zip2:
            score += weights["zip_code"]
        
        return score
    
    def _fuzzy_match(self, str1: str, str2: str, threshold: float) -> bool:
        """Simple fuzzy string matching"""
        if not str1 or not str2:
            return False
        
        # Levenshtein-like similarity
        shorter = min(len(str1), len(str2))
        longer = max(len(str1), len(str2))
        
        if shorter == 0:
            return False
        
        # Count matching characters
        matches = sum(c1 == c2 for c1, c2 in zip(str1, str2))
        similarity = matches / longer
        
        return similarity >= threshold
    
    def get_stats(self) -> Dict[str, int]:
        """Get deduplication statistics"""
        return self.stats.copy()
    
    def reset(self):
        """Reset deduplicator state"""
        self.seen_signatures.clear()
        self.seen_facilities.clear()
        self.stats = {
            "total_processed": 0,
            "duplicates_found": 0,
            "unique_facilities": 0
        }