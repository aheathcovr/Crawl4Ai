"""
Enhanced Crawler Manager with session reuse and caching
Manages browser instances efficiently across scraping sessions
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import gc

from crawl4ai import AsyncWebCrawler, LLMConfig
from crawl4ai.chunking_strategy import HtmlTagChunking, RegexChunking

from crawl_config import CRAWLER_CONFIG, LLM_CONFIG, EXTRACTION_CONFIG


class CrawlerPool:
    """Manages a pool of reusable crawler instances"""
    
    def __init__(self, max_crawlers: int = 3):
        self.max_crawlers = max_crawlers
        self.crawlers: List[AsyncWebCrawler] = []
        self.available: asyncio.Queue = asyncio.Queue()
        self.created = 0
        self.logger = logging.getLogger(__name__)
        self._initialized = False
        
    async def initialize(self):
        """Initialize the crawler pool"""
        if self._initialized:
            return
            
        # Create initial crawlers
        for _ in range(min(2, self.max_crawlers)):
            crawler = await self._create_crawler()
            if crawler:
                await self.available.put(crawler)
                
        self._initialized = True
        self.logger.info(f"Initialized crawler pool with {self.available.qsize()} crawlers")
    
    async def _create_crawler(self) -> Optional[AsyncWebCrawler]:
        """Create a new crawler instance with optimized settings"""
        try:
            crawler = AsyncWebCrawler(
                headless=CRAWLER_CONFIG["headless"],
                verbose=CRAWLER_CONFIG["verbose"],
                cache_dir=CRAWLER_CONFIG["cache_dir"],
                viewport_width=CRAWLER_CONFIG["viewport"]["width"],
                viewport_height=CRAWLER_CONFIG["viewport"]["height"],
                requests_per_minute=CRAWLER_CONFIG["requests_per_minute"],
                concurrent_tabs=CRAWLER_CONFIG["concurrent_tabs"],
                session_cache=CRAWLER_CONFIG["session_cache"]
            )
            
            await crawler.__aenter__()
            self.crawlers.append(crawler)
            self.created += 1
            
            return crawler
            
        except Exception as e:
            self.logger.error(f"Failed to create crawler: {e}")
            return None
    
    @asynccontextmanager
    async def get_crawler(self):
        """Get a crawler from the pool"""
        if not self._initialized:
            await self.initialize()
            
        # Try to get an available crawler
        crawler = None
        try:
            crawler = await asyncio.wait_for(self.available.get(), timeout=5.0)
        except asyncio.TimeoutError:
            # Create a new one if needed and under limit
            if self.created < self.max_crawlers:
                crawler = await self._create_crawler()
            else:
                # Wait for one to become available
                crawler = await self.available.get()
        
        if not crawler:
            raise RuntimeError("Could not obtain crawler from pool")
            
        try:
            yield crawler
        finally:
            # Return to pool
            await self.available.put(crawler)
    
    async def cleanup(self):
        """Clean up all crawlers"""
        for crawler in self.crawlers:
            try:
                await crawler.__aexit__(None, None, None)
            except Exception as e:
                self.logger.error(f"Error closing crawler: {e}")
                
        self.crawlers.clear()
        self._initialized = False


class EnhancedCrawlerSession:
    """Enhanced crawler session with built-in optimizations"""
    
    def __init__(self, pool: Optional[CrawlerPool] = None):
        self.pool = pool or CrawlerPool()
        self.schema_cache: Dict[str, Any] = {}
        self.seen_urls: set = set()
        self.session_stats = {
            "pages_crawled": 0,
            "cache_hits": 0,
            "schemas_generated": 0,
            "schemas_reused": 0
        }
        self.logger = logging.getLogger(__name__)
        
    async def __aenter__(self):
        await self.pool.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.pool.cleanup()
        
    def get_domain(self, url: str) -> str:
        """Extract domain from URL for caching"""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc.lower()
    
    def get_cached_schema(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached schema for domain if available"""
        domain = self.get_domain(url)
        if domain in self.schema_cache:
            schema_entry = self.schema_cache[domain]
            # Check if schema is still valid (TTL)
            if datetime.now() - schema_entry["created"] < timedelta(hours=168):
                self.session_stats["schemas_reused"] += 1
                return schema_entry["schema"]
        return None
    
    def cache_schema(self, url: str, schema: Dict[str, Any]):
        """Cache schema for domain"""
        domain = self.get_domain(url)
        self.schema_cache[domain] = {
            "schema": schema,
            "created": datetime.now()
        }
        self.session_stats["schemas_generated"] += 1
    
    def get_html_chunking_strategy(self):
        """Get optimized HTML chunking strategy"""
        return HtmlTagChunking(
            tags=EXTRACTION_CONFIG["html_chunk_tags"],
            max_chunk_tokens=EXTRACTION_CONFIG["chunk_token_threshold"],
            overlap_rate=EXTRACTION_CONFIG["chunk_overlap_rate"]
        )
    
    def get_regex_chunking_strategy(self):
        """Get regex-based chunking strategy"""
        return RegexChunking(
            patterns=EXTRACTION_CONFIG["regex_chunk_patterns"],
            max_chunk_tokens=EXTRACTION_CONFIG["chunk_token_threshold"],
            overlap_rate=EXTRACTION_CONFIG["chunk_overlap_rate"]
        )
    
    def create_llm_config(self, provider: str = "openai", **kwargs) -> LLMConfig:
        """Create optimized LLM configuration"""
        config_params = {
            "provider": provider,
            "temperature": LLM_CONFIG["temperature"],
            "top_p": LLM_CONFIG["top_p"],
            "max_tokens": LLM_CONFIG["max_tokens"]
        }
        config_params.update(kwargs)
        return LLMConfig(**config_params)
    
    async def crawl_with_cache(self, url: str, **kwargs) -> Any:
        """Crawl URL with caching and session reuse"""
        async with self.pool.get_crawler() as crawler:
            # Update stats
            self.session_stats["pages_crawled"] += 1
            
            # Check if we've seen this URL recently
            if url in self.seen_urls and "force_refresh" not in kwargs:
                self.session_stats["cache_hits"] += 1
                
            self.seen_urls.add(url)
            
            # Perform crawl
            result = await crawler.arun(url=url, **kwargs)
            
            # Memory management
            if self.session_stats["pages_crawled"] % CRAWLER_CONFIG.get("clear_cookies_interval", 50) == 0:
                try:
                    await crawler.context.clear_cookies()
                    gc.collect()
                    self.logger.info("Cleared cookies and ran GC")
                except Exception as e:
                    self.logger.error(f"Error clearing cookies: {e}")
                    
            return result


# Global instance for reuse
_global_crawler_pool = None

def get_global_crawler_pool() -> CrawlerPool:
    """Get or create global crawler pool"""
    global _global_crawler_pool
    if _global_crawler_pool is None:
        _global_crawler_pool = CrawlerPool()
    return _global_crawler_pool