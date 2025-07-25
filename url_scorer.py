"""
URL Scoring and Prioritization System
Intelligently scores URLs to prioritize facility pages and avoid irrelevant content
"""

import re
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse, parse_qs
import logging

from crawl_config import URL_SCORING


class URLScorer:
    """Score and prioritize URLs based on facility relevance"""
    
    def __init__(self):
        self.keyword_weights = URL_SCORING["keyword_weights"]
        self.depth_penalty = URL_SCORING["depth_penalty"]
        self.max_depth = URL_SCORING["max_depth"]
        self.logger = logging.getLogger(__name__)
        
        # Compile regex patterns for efficiency
        self.facility_patterns = [
            re.compile(r'/location[s]?/', re.I),
            re.compile(r'/facilit(y|ies)/', re.I),
            re.compile(r'/communit(y|ies)/', re.I),
            re.compile(r'/center[s]?/', re.I),
            re.compile(r'/find.*location', re.I),
            re.compile(r'/directory/', re.I)
        ]
        
        self.negative_patterns = [
            re.compile(r'/career[s]?/', re.I),
            re.compile(r'/job[s]?/', re.I),
            re.compile(r'/news/', re.I),
            re.compile(r'/blog/', re.I),
            re.compile(r'/investor[s]?/', re.I),
            re.compile(r'/press/', re.I),
            re.compile(r'/media/', re.I),
            re.compile(r'\.pdf$', re.I)
        ]
    
    def score_url(self, url: str, base_url: str = None) -> Tuple[float, Dict[str, any]]:
        """
        Score a URL based on facility relevance
        Returns (score, metadata)
        """
        parsed = urlparse(url)
        path = parsed.path.lower()
        query = parsed.query.lower()
        
        score = 0.0
        metadata = {
            "depth": self._calculate_depth(url, base_url),
            "has_facility_keywords": False,
            "has_negative_keywords": False,
            "keyword_matches": []
        }
        
        # Check URL depth
        if metadata["depth"] > self.max_depth:
            return -100.0, metadata  # Strongly discourage deep URLs
        
        # Apply depth penalty
        score -= metadata["depth"] * self.depth_penalty
        
        # Check positive patterns
        for pattern in self.facility_patterns:
            if pattern.search(path) or pattern.search(query):
                score += 20
                metadata["has_facility_keywords"] = True
                break
        
        # Check negative patterns
        for pattern in self.negative_patterns:
            if pattern.search(path):
                score -= 50
                metadata["has_negative_keywords"] = True
                break
        
        # Check keywords in URL
        url_lower = url.lower()
        for keyword, weight in self.keyword_weights.items():
            if keyword in url_lower:
                score += weight
                metadata["keyword_matches"].append(keyword)
        
        # Bonus for specific URL structures
        if '/our-' in path and any(kw in path for kw in ['location', 'facilities', 'communities']):
            score += 15
        
        # Check for pagination (usually good for listings)
        if 'page=' in query or '/page/' in path:
            score += 5
        
        # Penalize file downloads (except HTML)
        if any(path.endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx']):
            score -= 30
        
        return score, metadata
    
    def _calculate_depth(self, url: str, base_url: str = None) -> int:
        """Calculate URL depth from base"""
        if not base_url:
            return 0
        
        base_parts = urlparse(base_url).path.strip('/').split('/')
        url_parts = urlparse(url).path.strip('/').split('/')
        
        # Remove empty parts
        base_parts = [p for p in base_parts if p]
        url_parts = [p for p in url_parts if p]
        
        return len(url_parts) - len(base_parts)
    
    def rank_urls(self, urls: List[str], base_url: str = None) -> List[Tuple[str, float, Dict]]:
        """
        Rank a list of URLs by score
        Returns list of (url, score, metadata) tuples sorted by score
        """
        scored_urls = []
        
        for url in urls:
            score, metadata = self.score_url(url, base_url)
            scored_urls.append((url, score, metadata))
        
        # Sort by score (descending)
        scored_urls.sort(key=lambda x: x[1], reverse=True)
        
        self.logger.info(f"Ranked {len(urls)} URLs")
        
        # Log top and bottom URLs for debugging
        if scored_urls:
            self.logger.debug(f"Top URL: {scored_urls[0][0]} (score: {scored_urls[0][1]:.1f})")
            if len(scored_urls) > 1:
                self.logger.debug(f"Bottom URL: {scored_urls[-1][0]} (score: {scored_urls[-1][1]:.1f})")
        
        return scored_urls
    
    def filter_facility_urls(self, urls: List[str], base_url: str = None, min_score: float = 0) -> List[str]:
        """
        Filter URLs to only include likely facility pages
        """
        ranked = self.rank_urls(urls, base_url)
        filtered = [url for url, score, _ in ranked if score >= min_score]
        
        self.logger.info(f"Filtered {len(urls)} URLs to {len(filtered)} facility-related URLs")
        
        return filtered


class SmartURLFrontier:
    """
    Manages URL frontier with intelligent prioritization
    """
    
    def __init__(self, scorer: Optional[URLScorer] = None):
        self.scorer = scorer or URLScorer()
        self.visited: set = set()
        self.frontier: List[Tuple[str, float]] = []
        self.logger = logging.getLogger(__name__)
    
    def add_urls(self, urls: List[str], base_url: str = None):
        """Add URLs to frontier with scoring"""
        new_urls = [url for url in urls if url not in self.visited]
        
        if not new_urls:
            return
        
        # Score and add to frontier
        for url in new_urls:
            score, _ = self.scorer.score_url(url, base_url)
            self.frontier.append((url, score))
        
        # Re-sort frontier by score
        self.frontier.sort(key=lambda x: x[1], reverse=True)
        
        self.logger.debug(f"Added {len(new_urls)} URLs to frontier, total: {len(self.frontier)}")
    
    def get_next(self) -> Optional[str]:
        """Get next highest-scored URL"""
        while self.frontier:
            url, score = self.frontier.pop(0)
            
            if url not in self.visited:
                self.visited.add(url)
                self.logger.debug(f"Next URL: {url} (score: {score:.1f})")
                return url
        
        return None
    
    def has_urls(self) -> bool:
        """Check if frontier has unvisited URLs"""
        return bool(self.frontier)
    
    def get_stats(self) -> Dict[str, int]:
        """Get frontier statistics"""
        return {
            "visited": len(self.visited),
            "frontier_size": len(self.frontier),
            "avg_score": sum(s for _, s in self.frontier) / len(self.frontier) if self.frontier else 0
        }