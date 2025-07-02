"""
Enhanced URL Discovery System for Comprehensive Page Coverage
This module provides advanced URL discovery techniques to find every possible page
"""

import re
import asyncio
import aiohttp
import json
from typing import Set, List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse
from pathlib import Path
import time
from dataclasses import dataclass
from loguru import logger
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.settings import get_settings
from config.urls import is_valid_tekyz_url, EXCLUDE_PATTERNS


@dataclass
class DiscoveryStats:
    """Statistics for URL discovery operation"""
    total_discovered: int = 0
    sitemap_urls: int = 0
    crawled_urls: int = 0
    api_discovered: int = 0
    form_discovered: int = 0
    js_discovered: int = 0
    unique_urls: int = 0
    processing_time: float = 0.0
    failed_requests: int = 0


class EnhancedURLDiscovery:
    """Comprehensive URL discovery with multiple advanced techniques"""
    
    def __init__(self, config=None):
        self.config = config or get_settings()
        self.discovered_urls: Set[str] = set()
        self.processed_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        self.stats = DiscoveryStats()
        
        # Enhanced discovery patterns
        self.url_patterns = [
            r'https?://[^"\s<>]+',  # Basic HTTP URLs
            r'(?:href|src|action|data-href|data-url)=["\']([^"\']+)["\']',  # Attribute URLs
            r'window\.location\s*=\s*["\']([^"\']+)["\']',  # JavaScript redirects
            r'\.href\s*=\s*["\']([^"\']+)["\']',  # JavaScript href assignments
            r'fetch\(["\']([^"\']+)["\']',  # Fetch API calls
            r'axios\.[get|post]+\(["\']([^"\']+)["\']',  # Axios calls
            r'url:\s*["\']([^"\']+)["\']',  # AJAX URL patterns
        ]
        
        logger.info("Enhanced URL Discovery initialized")
    
    async def discover_all_urls(self, max_depth: int = 5, max_pages: int = 500) -> List[str]:
        """
        Comprehensive URL discovery using all available methods
        """
        start_time = time.time()
        logger.info("🚀 Starting comprehensive URL discovery")
        
        try:
            # Step 1: Get seed URLs
            seed_urls = await self._get_seed_urls()
            logger.info(f"Found {len(seed_urls)} seed URLs")
            
            # Step 2: Enhanced sitemap discovery
            sitemap_urls = await self._enhanced_sitemap_discovery()
            self.stats.sitemap_urls = len(sitemap_urls)
            self.discovered_urls.update(sitemap_urls)
            logger.info(f"Found {len(sitemap_urls)} URLs from enhanced sitemap discovery")
            
            # Step 3: Deep recursive crawling
            crawled_urls = await self._deep_crawl(seed_urls, max_depth, max_pages)
            self.stats.crawled_urls = len(crawled_urls)
            self.discovered_urls.update(crawled_urls)
            logger.info(f"Found {len(crawled_urls)} URLs from deep crawling")
            
            # Step 4: Pattern-based discovery
            pattern_urls = await self._discover_by_patterns()
            self.discovered_urls.update(pattern_urls)
            logger.info(f"Found {len(pattern_urls)} URLs from pattern discovery")
            
            # Step 5: Filter and validate
            valid_urls = self._filter_and_validate_urls(list(self.discovered_urls))
            final_urls = self._prioritize_urls(valid_urls)
            
            self.stats.total_discovered = len(self.discovered_urls)
            self.stats.unique_urls = len(final_urls)
            self.stats.processing_time = time.time() - start_time
            
            # Save discovery report
            await self._save_discovery_report(final_urls)
            
            logger.success(f"✅ URL discovery completed: {len(final_urls)} unique URLs in {self.stats.processing_time:.2f}s")
            
            return final_urls
            
        except Exception as e:
            logger.error(f"❌ URL discovery failed: {str(e)}")
            raise
    
    async def _get_seed_urls(self) -> List[str]:
        """Get initial seed URLs from multiple sources"""
        seed_urls = set()
        
        # Manual URLs from config
        from config.urls import get_all_urls
        manual_urls = get_all_urls()
        seed_urls.update(manual_urls)
        
        # Common patterns
        base_patterns = [
            'https://tekyz.com/',
            'https://tekyz.com/about',
            'https://tekyz.com/services',
            'https://tekyz.com/portfolio',
            'https://tekyz.com/contact',
            'https://tekyz.com/blog',
        ]
        seed_urls.update(base_patterns)
        
        return list(seed_urls)
    
    async def _enhanced_sitemap_discovery(self) -> List[str]:
        """Enhanced sitemap discovery with recursive parsing"""
        all_urls = set()
        
        sitemap_locations = [
            '/sitemap.xml',
            '/sitemap_index.xml',
            '/sitemap/sitemap.xml',
            '/robots.txt',
        ]
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            for location in sitemap_locations:
                url = f"https://tekyz.com{location}"
                try:
                    urls = await self._parse_sitemap_recursive(session, url)
                    all_urls.update(urls)
                    logger.debug(f"Found {len(urls)} URLs in {url}")
                except Exception as e:
                    logger.debug(f"Failed to parse {url}: {str(e)}")
        
        return list(all_urls)
    
    async def _parse_sitemap_recursive(self, session: aiohttp.ClientSession, url: str) -> List[str]:
        """Recursively parse sitemap with better error handling"""
        urls = []
        
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    return urls
                
                content = await response.text()
                
                if url.endswith('robots.txt'):
                    # Extract sitemap URLs from robots.txt
                    sitemap_urls = re.findall(r'Sitemap:\s*(.+)', content, re.IGNORECASE)
                    for sitemap_url in sitemap_urls:
                        nested_urls = await self._parse_sitemap_recursive(session, sitemap_url.strip())
                        urls.extend(nested_urls)
                else:
                    # Parse XML sitemap
                    try:
                        root = ET.fromstring(content)
                        
                        # Handle different namespaces
                        namespaces = {
                            '': 'http://www.sitemaps.org/schemas/sitemap/0.9',
                            'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'
                        }
                        
                        # Extract URLs from sitemap
                        for url_element in root.findall('.//url', namespaces) + root.findall('.//sitemap:url', namespaces):
                            loc = url_element.find('loc', namespaces) or url_element.find('sitemap:loc', namespaces)
                            if loc is not None and loc.text:
                                urls.append(loc.text.strip())
                        
                        # Handle sitemap index files
                        for sitemap_element in root.findall('.//sitemap', namespaces) + root.findall('.//sitemap:sitemap', namespaces):
                            loc = sitemap_element.find('loc', namespaces) or sitemap_element.find('sitemap:loc', namespaces)
                            if loc is not None and loc.text:
                                nested_urls = await self._parse_sitemap_recursive(session, loc.text.strip())
                                urls.extend(nested_urls)
                    
                    except ET.ParseError:
                        # Try parsing as plain text for non-XML sitemaps
                        url_matches = re.findall(r'https?://[^\s<>"\']+', content)
                        urls.extend(url_matches)
        
        except Exception as e:
            logger.debug(f"Error parsing sitemap {url}: {str(e)}")
        
        return urls
    
    async def _deep_crawl(self, seed_urls: List[str], max_depth: int, max_pages: int) -> List[str]:
        """Deep recursive crawling with intelligent link discovery"""
        crawled_urls = set()
        to_crawl = set(seed_urls)
        depth = 0
        
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        ) as session:
            
            while to_crawl and len(crawled_urls) < max_pages and depth < max_depth:
                logger.info(f"Crawling depth {depth + 1}: {len(to_crawl)} URLs to process")
                
                current_batch = list(to_crawl)[:20]  # Process in smaller batches
                to_crawl.difference_update(current_batch)
                
                # Crawl batch concurrently
                tasks = [self._crawl_single_page(session, url) for url in current_batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                new_urls = set()
                for url, result in zip(current_batch, results):
                    if isinstance(result, Exception):
                        logger.debug(f"Failed to crawl {url}: {str(result)}")
                        self.stats.failed_requests += 1
                        continue
                    
                    crawled_urls.add(url)
                    if result:
                        new_urls.update(result)
                
                # Filter new URLs
                valid_new_urls = {url for url in new_urls 
                                if url not in crawled_urls and url not in self.processed_urls 
                                and is_valid_tekyz_url(url)}
                
                to_crawl.update(valid_new_urls)
                depth += 1
                
                # Rate limiting
                await asyncio.sleep(1)
        
        logger.info(f"Deep crawl completed: {len(crawled_urls)} pages crawled")
        return list(crawled_urls)
    
    async def _crawl_single_page(self, session: aiohttp.ClientSession, url: str) -> List[str]:
        """Crawl a single page and extract all possible URLs"""
        discovered_urls = []
        
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    return discovered_urls
                
                content = await response.text()
                soup = BeautifulSoup(content, 'lxml')
                
                # Extract URLs from various sources
                urls = set()
                
                # Standard links
                for tag in soup.find_all(['a', 'link', 'area']):
                    href = tag.get('href')
                    if href:
                        full_url = urljoin(url, href)
                        urls.add(full_url)
                
                # Form actions
                for form in soup.find_all('form'):
                    action = form.get('action')
                    if action:
                        full_url = urljoin(url, action)
                        urls.add(full_url)
                
                # Data attributes
                for element in soup.find_all(attrs={'data-href': True}):
                    href = element.get('data-href')
                    if href:
                        full_url = urljoin(url, href)
                        urls.add(full_url)
                
                # JavaScript URLs
                script_tags = soup.find_all('script')
                for script in script_tags:
                    if script.string:
                        # Extract URLs from JavaScript
                        for pattern in self.url_patterns:
                            matches = re.findall(pattern, script.string)
                            for match in matches:
                                if isinstance(match, tuple):
                                    match = match[0]
                                full_url = urljoin(url, match)
                                urls.add(full_url)
                
                # Filter and validate URLs
                for found_url in urls:
                    if is_valid_tekyz_url(found_url):
                        discovered_urls.append(found_url)
                
        except Exception as e:
            logger.debug(f"Error crawling {url}: {str(e)}")
        
        return discovered_urls
    
    async def _discover_by_patterns(self) -> List[str]:
        """Discover URLs using intelligent pattern matching"""
        pattern_urls = set()
        
        # Generate URLs based on content type patterns
        base_url = "https://tekyz.com"
        
        # Service variations
        services = [
            'web-development', 'mobile-development', 'app-development',
            'software-development', 'ui-ux-design', 'digital-marketing',
            'seo', 'ecommerce-development', 'cloud-solutions', 'consulting'
        ]
        
        for service in services:
            pattern_urls.add(f"{base_url}/services/{service}")
            pattern_urls.add(f"{base_url}/{service}")
        
        # Industry variations
        industries = [
            'healthcare', 'finance', 'education', 'retail', 'real-estate',
            'non-profit', 'startups', 'enterprise'
        ]
        
        for industry in industries:
            pattern_urls.add(f"{base_url}/industries/{industry}")
            pattern_urls.add(f"{base_url}/solutions/{industry}")
        
        # Technology variations
        technologies = [
            'react', 'angular', 'vue', 'node-js', 'python', 'php',
            'java', 'wordpress', 'aws', 'azure'
        ]
        
        for tech in technologies:
            pattern_urls.add(f"{base_url}/technologies/{tech}")
            pattern_urls.add(f"{base_url}/expertise/{tech}")
        
        return list(pattern_urls)
    
    def _filter_and_validate_urls(self, urls: List[str]) -> List[str]:
        """Filter and validate discovered URLs"""
        valid_urls = []
        
        for url in urls:
            if not url or not isinstance(url, str):
                continue
            
            url = url.strip()
            
            # Remove fragments
            parsed = urlparse(url)
            if parsed.fragment:
                url = urlunparse(parsed._replace(fragment=''))
            
            # Skip invalid URLs
            if not is_valid_tekyz_url(url):
                continue
            
            # Skip excluded patterns
            skip = False
            for pattern in EXCLUDE_PATTERNS:
                if re.search(pattern, url, re.IGNORECASE):
                    skip = True
                    break
            
            if not skip:
                valid_urls.append(url)
        
        # Remove duplicates
        seen = set()
        unique_urls = []
        for url in valid_urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        
        return unique_urls
    
    def _prioritize_urls(self, urls: List[str]) -> List[str]:
        """Prioritize URLs for optimal scraping order"""
        
        def get_priority_score(url: str) -> int:
            score = 0
            url_lower = url.lower()
            
            # High priority pages
            if any(page in url_lower for page in ['/', '/about', '/services', '/portfolio']):
                score += 100
            
            # Important content types
            if '/services' in url_lower:
                score += 80
            elif any(keyword in url_lower for keyword in ['/portfolio', '/work']):
                score += 70
            elif '/blog' in url_lower:
                score += 60
            
            # Prefer shorter URLs
            path_length = len(urlparse(url).path.strip('/').split('/'))
            score += max(0, 20 - path_length * 5)
            
            return score
        
        return sorted(urls, key=get_priority_score, reverse=True)
    
    async def _save_discovery_report(self, urls: List[str]):
        """Save comprehensive discovery report"""
        report = {
            'timestamp': time.time(),
            'total_urls': len(urls),
            'stats': {
                'sitemap_urls': self.stats.sitemap_urls,
                'crawled_urls': self.stats.crawled_urls,
                'processing_time': self.stats.processing_time
            },
            'urls': urls
        }
        
        report_file = Path("data/url_discovery_report.json")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Discovery report saved to {report_file}") 