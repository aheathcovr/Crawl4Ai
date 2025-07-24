"""
Navigation module for healthcare facility discovery
Handles different website architectures and navigation patterns
"""

import asyncio
import json
import logging
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, parse_qs
from dataclasses import dataclass

import aiohttp
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, CrawlResult

from config import FACILITY_URL_PATTERNS, EXCLUDE_PATTERNS, CRAWL_CONFIG


@dataclass
class NavigationResult:
    """Result of navigation discovery"""
    facility_urls: Set[str]
    listing_pages: Set[str]
    sitemap_urls: Set[str]
    search_endpoints: List[str]
    pagination_info: Dict[str, any]


class SiteNavigator:
    """Advanced site navigation for facility discovery"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.logger = logging.getLogger(__name__)
        self.visited_urls: Set[str] = set()
        
    async def discover_facilities(self, crawler: AsyncWebCrawler) -> NavigationResult:
        """Main method to discover all facility-related URLs"""
        result = NavigationResult(
            facility_urls=set(),
            listing_pages=set(),
            sitemap_urls=set(),
            search_endpoints=[],
            pagination_info={}
        )
        
        # Step 1: Check for sitemaps
        await self._discover_sitemaps(result)
        
        # Step 2: Analyze main navigation
        await self._analyze_main_navigation(crawler, result)
        
        # Step 3: Look for search functionality
        await self._discover_search_endpoints(crawler, result)
        
        # Step 4: Recursive link discovery
        await self._recursive_link_discovery(crawler, result)
        
        # Step 5: Handle JavaScript-rendered content
        await self._handle_spa_content(crawler, result)
        
        return result
    
    async def _discover_sitemaps(self, result: NavigationResult):
        """Discover and parse XML sitemaps"""
        sitemap_urls = [
            urljoin(self.base_url, '/sitemap.xml'),
            urljoin(self.base_url, '/sitemap_index.xml'),
            urljoin(self.base_url, '/sitemaps.xml'),
            urljoin(self.base_url, '/robots.txt')  # Check robots.txt for sitemap references
        ]
        
        async with aiohttp.ClientSession() as session:
            for sitemap_url in sitemap_urls:
                try:
                    async with session.get(sitemap_url, timeout=10) as response:
                        if response.status == 200:
                            content = await response.text()
                            
                            if sitemap_url.endswith('robots.txt'):
                                await self._parse_robots_txt(content, result)
                            else:
                                await self._parse_sitemap(content, result)
                                
                except Exception as e:
                    self.logger.debug(f"Could not access {sitemap_url}: {e}")
    
    async def _parse_robots_txt(self, content: str, result: NavigationResult):
        """Parse robots.txt for sitemap references"""
        lines = content.split('\n')
        for line in lines:
            if line.strip().lower().startswith('sitemap:'):
                sitemap_url = line.split(':', 1)[1].strip()
                result.sitemap_urls.add(sitemap_url)
                
                # Parse the discovered sitemap
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(sitemap_url, timeout=10) as response:
                            if response.status == 200:
                                sitemap_content = await response.text()
                                await self._parse_sitemap(sitemap_content, result)
                except Exception as e:
                    self.logger.debug(f"Could not parse sitemap {sitemap_url}: {e}")
    
    async def _parse_sitemap(self, content: str, result: NavigationResult):
        """Parse XML sitemap for facility URLs"""
        try:
            root = ET.fromstring(content)
            
            # Handle sitemap index files
            if 'sitemapindex' in root.tag:
                for sitemap in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap'):
                    loc = sitemap.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                    if loc is not None:
                        result.sitemap_urls.add(loc.text)
            
            # Handle regular sitemaps
            else:
                for url in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                    loc = url.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                    if loc is not None:
                        url_text = loc.text
                        if self._is_facility_related_url(url_text):
                            if self._is_facility_detail_url(url_text):
                                result.facility_urls.add(url_text)
                            else:
                                result.listing_pages.add(url_text)
                                
        except ET.ParseError as e:
            self.logger.debug(f"Could not parse sitemap XML: {e}")
    
    async def _analyze_main_navigation(self, crawler: AsyncWebCrawler, result: NavigationResult):
        """Analyze main website navigation for facility links"""
        try:
            crawl_result = await crawler.arun(url=self.base_url)
            if not crawl_result.success:
                return
                
            soup = BeautifulSoup(crawl_result.html, 'html.parser')
            
            # Analyze main navigation menu
            nav_selectors = [
                'nav', '.navigation', '.nav', '.menu', '.main-menu',
                '[role="navigation"]', '.navbar', '.header-nav'
            ]
            
            for selector in nav_selectors:
                nav_elements = soup.select(selector)
                for nav in nav_elements:
                    await self._extract_nav_links(nav, result)
            
            # Look for footer links (often contain location links)
            footer_selectors = ['footer', '.footer', '.site-footer']
            for selector in footer_selectors:
                footer_elements = soup.select(selector)
                for footer in footer_elements:
                    await self._extract_nav_links(footer, result)
                    
        except Exception as e:
            self.logger.warning(f"Error analyzing main navigation: {e}")
    
    async def _extract_nav_links(self, element, result: NavigationResult):
        """Extract facility-related links from navigation elements"""
        links = element.find_all('a', href=True)
        
        for link in links:
            href = link.get('href')
            text = link.get_text(strip=True).lower()
            
            if href:
                full_url = urljoin(self.base_url, href)
                
                if self._is_same_domain(full_url) and self._is_facility_related_url(full_url):
                    if self._is_facility_detail_url(full_url):
                        result.facility_urls.add(full_url)
                    else:
                        result.listing_pages.add(full_url)
    
    async def _discover_search_endpoints(self, crawler: AsyncWebCrawler, result: NavigationResult):
        """Discover search functionality that might help find facilities"""
        try:
            crawl_result = await crawler.arun(url=self.base_url)
            if not crawl_result.success:
                return
                
            soup = BeautifulSoup(crawl_result.html, 'html.parser')
            
            # Look for search forms
            forms = soup.find_all('form')
            for form in forms:
                action = form.get('action', '')
                method = form.get('method', 'get').lower()
                
                # Check if form might be for location search
                form_text = form.get_text().lower()
                location_keywords = ['location', 'find', 'search', 'facility', 'center']
                
                if any(keyword in form_text for keyword in location_keywords):
                    search_endpoint = {
                        'url': urljoin(self.base_url, action) if action else self.base_url,
                        'method': method,
                        'fields': []
                    }
                    
                    # Extract form fields
                    inputs = form.find_all(['input', 'select'])
                    for input_elem in inputs:
                        field_info = {
                            'name': input_elem.get('name', ''),
                            'type': input_elem.get('type', 'text'),
                            'required': input_elem.has_attr('required')
                        }
                        search_endpoint['fields'].append(field_info)
                    
                    result.search_endpoints.append(search_endpoint)
            
            # Look for AJAX search endpoints in JavaScript
            await self._discover_ajax_endpoints(soup, result)
            
        except Exception as e:
            self.logger.warning(f"Error discovering search endpoints: {e}")
    
    async def _discover_ajax_endpoints(self, soup: BeautifulSoup, result: NavigationResult):
        """Discover AJAX endpoints from JavaScript code"""
        script_tags = soup.find_all('script')
        
        for script in script_tags:
            if script.string:
                script_content = script.string
                
                # Look for common AJAX patterns
                ajax_patterns = [
                    r'ajax.*?url.*?["\']([^"\']+)["\']',
                    r'fetch\(["\']([^"\']+)["\']',
                    r'\.get\(["\']([^"\']+)["\']',
                    r'\.post\(["\']([^"\']+)["\']'
                ]
                
                for pattern in ajax_patterns:
                    matches = re.findall(pattern, script_content, re.IGNORECASE)
                    for match in matches:
                        if 'location' in match.lower() or 'facility' in match.lower():
                            endpoint_url = urljoin(self.base_url, match)
                            if self._is_same_domain(endpoint_url):
                                result.search_endpoints.append({
                                    'url': endpoint_url,
                                    'method': 'get',
                                    'type': 'ajax'
                                })
    
    async def _recursive_link_discovery(self, crawler: AsyncWebCrawler, result: NavigationResult, max_depth: int = 3):
        """Recursively discover facility links with depth limit"""
        current_depth = 0
        urls_to_visit = {self.base_url}
        
        while current_depth < max_depth and urls_to_visit:
            next_urls = set()
            
            for url in urls_to_visit:
                if url in self.visited_urls:
                    continue
                    
                try:
                    crawl_result = await crawler.arun(url=url)
                    if crawl_result.success:
                        self.visited_urls.add(url)
                        discovered_urls = await self._extract_links_from_page(crawl_result, result)
                        next_urls.update(discovered_urls)
                        
                    await asyncio.sleep(CRAWL_CONFIG['delay_between_requests'])
                    
                except Exception as e:
                    self.logger.debug(f"Error crawling {url}: {e}")
            
            urls_to_visit = next_urls - self.visited_urls
            current_depth += 1
    
    async def _extract_links_from_page(self, crawl_result: CrawlResult, result: NavigationResult) -> Set[str]:
        """Extract relevant links from a page"""
        discovered_urls = set()
        soup = BeautifulSoup(crawl_result.html, 'html.parser')
        
        links = soup.find_all('a', href=True)
        for link in links:
            href = link.get('href')
            if href:
                full_url = urljoin(crawl_result.url, href)
                
                if self._is_same_domain(full_url) and not self._should_exclude_url(full_url):
                    if self._is_facility_related_url(full_url):
                        if self._is_facility_detail_url(full_url):
                            result.facility_urls.add(full_url)
                        else:
                            result.listing_pages.add(full_url)
                    else:
                        # Add to discovered URLs for further exploration
                        discovered_urls.add(full_url)
        
        return discovered_urls
    
    async def _handle_spa_content(self, crawler: AsyncWebCrawler, result: NavigationResult):
        """Handle Single Page Application content that loads dynamically"""
        try:
            # Look for common SPA patterns in listing pages
            for listing_url in list(result.listing_pages):
                crawl_result = await crawler.arun(
                    url=listing_url,
                    wait_for="networkidle",
                    js_code="""
                    // Wait for potential AJAX content to load
                    await new Promise(resolve => setTimeout(resolve, 3000));
                    
                    // Try to trigger "load more" buttons
                    const loadMoreButtons = document.querySelectorAll('[class*="load"], [class*="more"], [class*="show"]');
                    for (let button of loadMoreButtons) {
                        if (button.textContent.toLowerCase().includes('more') || 
                            button.textContent.toLowerCase().includes('load')) {
                            button.click();
                            await new Promise(resolve => setTimeout(resolve, 2000));
                        }
                    }
                    """
                )
                
                if crawl_result.success:
                    soup = BeautifulSoup(crawl_result.html, 'html.parser')
                    
                    # Extract any new facility links that appeared
                    links = soup.find_all('a', href=True)
                    for link in links:
                        href = link.get('href')
                        if href:
                            full_url = urljoin(listing_url, href)
                            if (self._is_same_domain(full_url) and 
                                self._is_facility_detail_url(full_url)):
                                result.facility_urls.add(full_url)
                
                await asyncio.sleep(CRAWL_CONFIG['delay_between_requests'])
                
        except Exception as e:
            self.logger.warning(f"Error handling SPA content: {e}")
    
    def _is_facility_related_url(self, url: str) -> bool:
        """Check if URL is related to facilities"""
        url_lower = url.lower()
        return any(pattern in url_lower for pattern in FACILITY_URL_PATTERNS)
    
    def _is_facility_detail_url(self, url: str) -> bool:
        """Check if URL points to a specific facility detail page"""
        url_lower = url.lower()
        
        # Look for patterns that suggest individual facility pages
        detail_patterns = [
            r'/location/[^/]+/?$',
            r'/facility/[^/]+/?$',
            r'/center/[^/]+/?$',
            r'/community/[^/]+/?$',
            r'/property/[^/]+/?$',
            r'/locations/[^/]+/?$',
            r'/facilities/[^/]+/?$'
        ]
        
        return any(re.search(pattern, url_lower) for pattern in detail_patterns)
    
    def _should_exclude_url(self, url: str) -> bool:
        """Check if URL should be excluded from crawling"""
        url_lower = url.lower()
        return any(pattern in url_lower for pattern in EXCLUDE_PATTERNS)
    
    def _is_same_domain(self, url: str) -> bool:
        """Check if URL belongs to the same domain"""
        return urlparse(url).netloc == self.domain


class PaginationHandler:
    """Handle pagination on facility listing pages"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)
    
    async def handle_pagination(self, crawler: AsyncWebCrawler, listing_url: str) -> List[str]:
        """Handle pagination and return all page URLs"""
        page_urls = [listing_url]
        
        try:
            crawl_result = await crawler.arun(url=listing_url)
            if not crawl_result.success:
                return page_urls
            
            soup = BeautifulSoup(crawl_result.html, 'html.parser')
            
            # Method 1: Look for numbered pagination
            page_urls.extend(await self._handle_numbered_pagination(soup, listing_url))
            
            # Method 2: Look for "Next" button pagination
            page_urls.extend(await self._handle_next_button_pagination(crawler, soup, listing_url))
            
            # Method 3: Look for infinite scroll/load more
            page_urls.extend(await self._handle_load_more_pagination(crawler, listing_url))
            
        except Exception as e:
            self.logger.warning(f"Error handling pagination for {listing_url}: {e}")
        
        return list(set(page_urls))  # Remove duplicates
    
    async def _handle_numbered_pagination(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Handle numbered pagination (1, 2, 3, ...)"""
        page_urls = []
        
        pagination_selectors = [
            '.pagination a', '.pager a', '.page-numbers a',
            '[class*="pagination"] a', '[class*="pager"] a'
        ]
        
        for selector in pagination_selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get('href')
                if href and re.search(r'page=\d+|p=\d+|\d+', href):
                    full_url = urljoin(base_url, href)
                    page_urls.append(full_url)
        
        return page_urls
    
    async def _handle_next_button_pagination(self, crawler: AsyncWebCrawler, soup: BeautifulSoup, current_url: str) -> List[str]:
        """Handle next button pagination"""
        page_urls = []
        
        # Look for "Next" links
        next_selectors = [
            'a:contains("Next")', 'a:contains(">")', 
            '.next', '[class*="next"]'
        ]
        
        for selector in next_selectors:
            try:
                next_links = soup.select(selector)
                for next_link in next_links:
                    href = next_link.get('href')
                    if href:
                        next_url = urljoin(current_url, href)
                        page_urls.append(next_url)
                        
                        # Recursively follow next links (with limit)
                        if len(page_urls) < 50:  # Prevent infinite loops
                            next_result = await crawler.arun(url=next_url)
                            if next_result.success:
                                next_soup = BeautifulSoup(next_result.html, 'html.parser')
                                more_pages = await self._handle_next_button_pagination(
                                    crawler, next_soup, next_url
                                )
                                page_urls.extend(more_pages)
            except Exception as e:
                self.logger.debug(f"Error with next button pagination: {e}")
        
        return page_urls
    
    async def _handle_load_more_pagination(self, crawler: AsyncWebCrawler, listing_url: str) -> List[str]:
        """Handle load more / infinite scroll pagination"""
        page_urls = []
        
        try:
            # Use JavaScript to trigger load more buttons
            crawl_result = await crawler.arun(
                url=listing_url,
                js_code="""
                const loadMoreButtons = document.querySelectorAll(
                    '[class*="load"], [class*="more"], [class*="show"], button'
                );
                
                let clickedButtons = 0;
                for (let button of loadMoreButtons) {
                    const text = button.textContent.toLowerCase();
                    if (text.includes('more') || text.includes('load') || text.includes('show')) {
                        button.click();
                        await new Promise(resolve => setTimeout(resolve, 2000));
                        clickedButtons++;
                        if (clickedButtons >= 5) break; // Limit clicks
                    }
                }
                
                // Return the current URL (content has been loaded dynamically)
                window.location.href;
                """
            )
            
            if crawl_result.success:
                # The content has been loaded dynamically, so we return the same URL
                # but the content will be different when scraped again
                page_urls.append(listing_url + "#loaded")
                
        except Exception as e:
            self.logger.debug(f"Error with load more pagination: {e}")
        
        return page_urls

