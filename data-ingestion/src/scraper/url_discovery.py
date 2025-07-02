"""
URL discovery system for finding all relevant pages on tekyz.com
"""

import re
import xml.etree.ElementTree as ET
from typing import List, Set, Optional, Dict
from urllib.parse import urljoin, urlparse, urlunparse
import requests
from loguru import logger
import time
from bs4 import BeautifulSoup

from config.urls import TEKYZ_URLS, EXCLUDE_PATTERNS, INCLUDE_PATTERNS, SITEMAP_LOCATIONS, is_valid_tekyz_url, prioritize_urls
from config.settings import get_settings


class URLDiscovery:
    """Comprehensive URL discovery system for tekyz.com"""
    
    def __init__(self, config=None):
        """Initialize URL discovery"""
        self.config = config or get_settings()
        self.discovered_urls: Set[str] = set()
        self.processed_pages: Set[str] = set()
        self.session = self._setup_session()
        
        logger.info("Enhanced URLDiscovery initialized")
    
    def _setup_session(self) -> requests.Session:
        """Set up requests session with better headers and retry strategy"""
        session = requests.Session()
        
        # Better headers to appear more like a real browser
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        })
        
        return session
    
    def get_sitemap_urls(self, base_url: str = None) -> List[str]:
        """
        Comprehensive sitemap discovery and parsing
        
        Args:
            base_url: Base URL for the website
            
        Returns:
            List of discovered URLs from all sitemaps
        """
        if not base_url:
            base_url = self.config.tekyz_base_url
            
        all_sitemap_urls = []
        found_sitemaps = []
        
        # Try multiple sitemap locations
        for location in SITEMAP_LOCATIONS:
            sitemap_url = urljoin(base_url, location)
            
            try:
                logger.info(f"Checking sitemap location: {sitemap_url}")
                
                response = self.session.get(
                    sitemap_url,
                    timeout=self.config.request_timeout
                )
                
                if response.status_code == 200:
                    if location == '/robots.txt':
                        # Extract sitemap URLs from robots.txt
                        sitemap_urls = self._extract_sitemaps_from_robots(response.text)
                        for sitemap_url in sitemap_urls:
                            if sitemap_url not in found_sitemaps:
                                found_sitemaps.append(sitemap_url)
                                urls = self._fetch_and_parse_sitemap(sitemap_url)
                                all_sitemap_urls.extend(urls)
                    else:
                        # Parse XML sitemap directly
                        if sitemap_url not in found_sitemaps:
                            found_sitemaps.append(sitemap_url)
                            urls = self._parse_sitemap_xml(response.content)
                            all_sitemap_urls.extend(urls)
                            logger.success(f"Found {len(urls)} URLs in {sitemap_url}")
                    
            except Exception as e:
                logger.warning(f"Could not fetch sitemap {sitemap_url}: {str(e)}")
                continue
        
        # Remove duplicates and validate URLs
        valid_urls = []
        seen_urls = set()
        
        for url in all_sitemap_urls:
            if url not in seen_urls and self.validate_url(url):
                valid_urls.append(url)
                seen_urls.add(url)
        
        logger.info(f"Discovered {len(valid_urls)} valid URLs from {len(found_sitemaps)} sitemaps")
        return valid_urls
    
    def _extract_sitemaps_from_robots(self, robots_content: str) -> List[str]:
        """Extract sitemap URLs from robots.txt content"""
        sitemap_urls = []
        
        for line in robots_content.split('\n'):
            line = line.strip()
            if line.lower().startswith('sitemap:'):
                sitemap_url = line.split(':', 1)[1].strip()
                sitemap_urls.append(sitemap_url)
                logger.info(f"Found sitemap in robots.txt: {sitemap_url}")
        
        return sitemap_urls
    
    def _fetch_and_parse_sitemap(self, sitemap_url: str) -> List[str]:
        """Fetch and parse a sitemap URL"""
        try:
            response = self.session.get(sitemap_url, timeout=self.config.request_timeout)
            if response.status_code == 200:
                return self._parse_sitemap_xml(response.content)
        except Exception as e:
            logger.warning(f"Failed to fetch sitemap {sitemap_url}: {str(e)}")
        return []
    
    def _parse_sitemap_xml(self, xml_content: bytes) -> List[str]:
        """
        Enhanced XML sitemap parsing with better error handling
        
        Args:
            xml_content: Raw XML content
            
        Returns:
            List of URLs found in sitemap
        """
        urls = []
        
        try:
            # Try to decode if it's compressed
            if xml_content.startswith(b'\x1f\x8b'):
                import gzip
                xml_content = gzip.decompress(xml_content)
            
            root = ET.fromstring(xml_content)
            
            # Handle different sitemap formats with namespaces
            namespaces = {
                'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9',
                'news': 'http://www.google.com/schemas/sitemap-news/0.9',
                'image': 'http://www.google.com/schemas/sitemap-image/1.1'
            }
            
            # Standard sitemap format
            for url_elem in root.findall('.//sitemap:url', namespaces):
                loc_elem = url_elem.find('sitemap:loc', namespaces)
                if loc_elem is not None and loc_elem.text:
                    urls.append(loc_elem.text.strip())
            
            # Sitemap index format (recursive sitemaps)
            for sitemap_elem in root.findall('.//sitemap:sitemap', namespaces):
                loc_elem = sitemap_elem.find('sitemap:loc', namespaces)
                if loc_elem is not None and loc_elem.text:
                    nested_urls = self._fetch_and_parse_sitemap(loc_elem.text.strip())
                    urls.extend(nested_urls)
            
            # Fallback for non-standard formats
            if not urls:
                # Try without namespaces
                for url_elem in root.findall('.//url'):
                    loc_elem = url_elem.find('loc')
                    if loc_elem is not None and loc_elem.text:
                        urls.append(loc_elem.text.strip())
                
                # Look for sitemap elements
                for sitemap_elem in root.findall('.//sitemap'):
                    loc_elem = sitemap_elem.find('loc')
                    if loc_elem is not None and loc_elem.text:
                        nested_urls = self._fetch_and_parse_sitemap(loc_elem.text.strip())
                        urls.extend(nested_urls)
                
                # Last resort - any loc tags
                if not urls:
                    for loc in root.iter():
                        if loc.tag.endswith('loc') and loc.text:
                            urls.append(loc.text.strip())
                        
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {str(e)}")
        except Exception as e:
            logger.error(f"Error parsing sitemap: {str(e)}")
        
        return urls
    
    def get_manual_urls(self) -> List[str]:
        """
        Return comprehensive manual URL list
        
        Returns:
            List of manually defined URLs
        """
        manual_urls = []
        
        for category, urls in TEKYZ_URLS.items():
            manual_urls.extend(urls)
        
        logger.info(f"Retrieved {len(manual_urls)} manual URLs from {len(TEKYZ_URLS)} categories")
        return manual_urls
    
    def discover_urls_from_page(self, page_content: str, base_url: str, max_depth: int = 3) -> List[str]:
        """
        Comprehensive URL discovery from page content
        
        Args:
            page_content: HTML content of the page
            base_url: Base URL for resolving relative links
            max_depth: Maximum depth for URL discovery
            
        Returns:
            List of discovered URLs
        """
        discovered = []
        
        try:
            soup = BeautifulSoup(page_content, 'lxml')
            base_domain = urlparse(base_url).netloc
            
            # Find all possible links
            link_selectors = [
                'a[href]',              # Standard links
                'link[href]',           # Link elements
                'area[href]',           # Image map areas
                '[data-href]',          # Data attributes
                '[data-url]',           # Data URL attributes
            ]
            
            for selector in link_selectors:
                elements = soup.select(selector)
                for element in elements:
                    href = element.get('href') or element.get('data-href') or element.get('data-url')
                    
                    if href:
                        # Clean and resolve URL
                        href = href.strip()
                        if href.startswith('javascript:') or href.startswith('mailto:') or href.startswith('tel:'):
                            continue
                            
                        full_url = urljoin(base_url, href)
                        
                        # Validate and add
                        if self.validate_url(full_url, base_domain):
                            discovered.append(full_url)
            
            # Look for URLs in JavaScript code and data attributes
            script_tags = soup.find_all('script')
            for script in script_tags:
                if script.string:
                    # Extract URLs from JavaScript
                    js_urls = re.findall(r'["\']https?://[^"\']+["\']', script.string)
                    for url in js_urls:
                        url = url.strip('"\'')
                        if self.validate_url(url, base_domain):
                            discovered.append(url)
            
            # Look for URLs in meta tags
            meta_tags = soup.find_all('meta')
            for meta in meta_tags:
                content = meta.get('content', '')
                if content and ('http://' in content or 'https://' in content):
                    urls = re.findall(r'https?://[^\s<>"\']+', content)
                    for url in urls:
                        if self.validate_url(url, base_domain):
                            discovered.append(url)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_discovered = []
            for url in discovered:
                if url not in seen:
                    seen.add(url)
                    unique_discovered.append(url)
            
            logger.info(f"Discovered {len(unique_discovered)} unique URLs from page: {base_url}")
            
        except Exception as e:
            logger.error(f"Error discovering URLs from page {base_url}: {str(e)}")
            return []
        
        return unique_discovered
    
    def crawl_for_urls(self, start_urls: List[str], max_pages: int = 50, max_depth: int = 3) -> List[str]:
        """
        Recursive crawling to discover more URLs
        
        Args:
            start_urls: URLs to start crawling from
            max_pages: Maximum number of pages to crawl
            max_depth: Maximum crawl depth
            
        Returns:
            List of all discovered URLs
        """
        all_discovered = set(start_urls)
        to_crawl = start_urls.copy()
        crawled = set()
        current_depth = 0
        
        logger.info(f"Starting recursive URL crawl with {len(start_urls)} seed URLs")
        
        while to_crawl and len(crawled) < max_pages and current_depth < max_depth:
            current_batch = to_crawl[:10]  # Process in batches
            to_crawl = to_crawl[10:]
            current_depth += 1
            
            logger.info(f"Crawl depth {current_depth}: Processing {len(current_batch)} URLs")
            
            for url in current_batch:
                if url in crawled:
                    continue
                    
                try:
                    # Rate limiting
                    time.sleep(self.config.scraping_delay)
                    
                    response = self.session.get(url, timeout=self.config.request_timeout)
                    
                    if response.status_code == 200 and 'text/html' in response.headers.get('content-type', ''):
                        # Discover URLs from this page
                        new_urls = self.discover_urls_from_page(response.text, url)
                        
                        # Add new URLs to our sets
                        for new_url in new_urls:
                            if new_url not in all_discovered:
                                all_discovered.add(new_url)
                                if new_url not in crawled:
                                    to_crawl.append(new_url)
                        
                        logger.debug(f"Found {len(new_urls)} URLs on {url}")
                    
                    crawled.add(url)
                    
                except Exception as e:
                    logger.warning(f"Failed to crawl {url}: {str(e)}")
                    crawled.add(url)  # Mark as processed to avoid retrying
        
        final_urls = list(all_discovered)
        logger.info(f"Recursive crawl completed: {len(crawled)} pages crawled, {len(final_urls)} total URLs discovered")
        
        return final_urls
    
    def validate_url(self, url: str, base_domain: str = None) -> bool:
        """
        Enhanced URL validation
        
        Args:
            url: URL to validate
            base_domain: Base domain to restrict crawling to (optional)
            
        Returns:
            True if URL is valid for scraping
        """
        if not url or not isinstance(url, str):
            return False
        
        # Parse URL
        try:
            parsed = urlparse(url)
        except Exception:
            return False
        
        # Must be HTTP/HTTPS
        if parsed.scheme not in ['http', 'https']:
            return False
        
        # If base_domain is provided, restrict to that domain
        if base_domain:
            if not parsed.netloc.endswith(base_domain):
                return False
        else:
            # Default: must be tekyz.com or subdomain
            if not re.match(r'^(.*\.)?tekyz\.com$', parsed.netloc):
                return False
        
        # Check exclude patterns first
        for pattern in EXCLUDE_PATTERNS:
            if re.match(pattern, url, re.IGNORECASE):
                return False
        
        # Check include patterns
        for pattern in INCLUDE_PATTERNS:
            if re.match(pattern, url, re.IGNORECASE):
                return True
        
        # Additional validation for common patterns
        path = parsed.path.lower()
        
        # Allow common content paths
        allowed_patterns = [
            r'^/$',                           # Home page
            r'^/[^/]+/?$',                   # Top-level pages
            r'^/[^/]+/[^/]+/?$',            # Second-level pages
            r'^/services/',                  # Services pages
            r'^/portfolio/',                 # Portfolio pages
            r'^/blog/',                      # Blog pages
            r'^/about',                      # About pages
            r'^/contact',                    # Contact pages
            r'^/team',                       # Team pages
            r'^/careers',                    # Career pages
            r'^/technologies/',              # Technology pages
            r'^/industries/',                # Industry pages
            r'^/case-studies',               # Case studies
            r'^/projects',                   # Projects
            r'^/work',                       # Work pages
        ]
        
        for pattern in allowed_patterns:
            if re.match(pattern, path):
                return True
        
        return False
    
    def get_all_urls(self, enable_crawling: bool = True, max_crawl_pages: int = 100) -> List[str]:
        """
        Comprehensive URL discovery using all available methods
        
        Args:
            enable_crawling: Whether to enable recursive crawling
            max_crawl_pages: Maximum pages to crawl if crawling is enabled
            
        Returns:
            Complete list of discovered URLs, prioritized
        """
        all_urls = set()
        
        # Step 1: Get manual URLs first (guaranteed to work)
        logger.info("Step 1: Getting manual URL list")
        manual_urls = self.get_manual_urls()
        all_urls.update(manual_urls)
        logger.info(f"Added {len(manual_urls)} manual URLs")
        
        # Step 2: Try to get sitemap URLs
        logger.info("Step 2: Discovering URLs from sitemaps")
        try:
            sitemap_urls = self.get_sitemap_urls()
            new_sitemap_urls = [url for url in sitemap_urls if url not in all_urls]
            all_urls.update(sitemap_urls)
            logger.info(f"Added {len(new_sitemap_urls)} new URLs from sitemaps")
        except Exception as e:
            logger.warning(f"Could not get sitemap URLs: {str(e)}")
        
        # Step 3: Recursive crawling (if enabled)
        if enable_crawling:
            logger.info("Step 3: Starting recursive crawling for additional URLs")
            try:
                # Use high-priority URLs as starting points
                from config.urls import get_high_priority_urls
                seed_urls = get_high_priority_urls()
                
                crawled_urls = self.crawl_for_urls(
                    start_urls=seed_urls,
                    max_pages=max_crawl_pages,
                    max_depth=3
                )
                
                new_crawled_urls = [url for url in crawled_urls if url not in all_urls]
                all_urls.update(crawled_urls)
                logger.info(f"Added {len(new_crawled_urls)} new URLs from crawling")
                
            except Exception as e:
                logger.warning(f"Crawling failed: {str(e)}")
        
        # Convert to sorted list and prioritize
        final_urls = list(all_urls)
        final_urls = prioritize_urls(final_urls)
        
        logger.success(f"Total URL discovery completed: {len(final_urls)} unique URLs found")
        
        # Log breakdown by category
        self._log_url_breakdown(final_urls)
        
        return final_urls
    
    def _log_url_breakdown(self, urls: List[str]):
        """Log breakdown of URLs by category"""
        categories = {
            'Core Pages': 0,
            'Service Pages': 0,
            'Portfolio Pages': 0,
            'Technology Pages': 0,
            'Industry Pages': 0,
            'Blog Pages': 0,
            'Company Pages': 0,
            'Career Pages': 0,
            'Other Pages': 0
        }
        
        for url in urls:
            url_lower = url.lower()
            
            if any(core in url_lower for core in ['/', '/about', '/contact', '/team']):
                categories['Core Pages'] += 1
            elif '/services' in url_lower:
                categories['Service Pages'] += 1
            elif any(keyword in url_lower for keyword in ['/portfolio', '/work', '/case-studies', '/projects']):
                categories['Portfolio Pages'] += 1
            elif '/technologies' in url_lower:
                categories['Technology Pages'] += 1
            elif '/industries' in url_lower:
                categories['Industry Pages'] += 1
            elif any(keyword in url_lower for keyword in ['/blog', '/news', '/insights', '/articles']):
                categories['Blog Pages'] += 1
            elif any(keyword in url_lower for keyword in ['/company', '/leadership', '/founders', '/culture']):
                categories['Company Pages'] += 1
            elif any(keyword in url_lower for keyword in ['/careers', '/jobs', '/join']):
                categories['Career Pages'] += 1
            else:
                categories['Other Pages'] += 1
        
        logger.info("URL Discovery Breakdown:")
        for category, count in categories.items():
            if count > 0:
                logger.info(f"  {category}: {count} URLs")
    
    def filter_urls_by_pattern(self, urls: List[str], patterns: List[str]) -> List[str]:
        """
        Filter URLs by regex patterns
        
        Args:
            urls: List of URLs to filter
            patterns: List of regex patterns
            
        Returns:
            Filtered list of URLs
        """
        filtered = []
        
        for url in urls:
            matches = False
            for pattern in patterns:
                if re.search(pattern, url, re.IGNORECASE):
                    matches = True
                    break
            
            if matches:
                filtered.append(url)
        
        return filtered 