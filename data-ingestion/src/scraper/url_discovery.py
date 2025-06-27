"""
URL discovery system for finding all relevant pages on tekyz.com
"""

import re
import xml.etree.ElementTree as ET
from typing import List, Set, Optional
from urllib.parse import urljoin, urlparse
import requests
from loguru import logger

from config.urls import TEKYZ_URLS, EXCLUDE_PATTERNS, INCLUDE_PATTERNS, is_valid_tekyz_url
from config.settings import get_settings


class URLDiscovery:
    """Discovers URLs from sitemaps and manual lists"""
    
    def __init__(self, config=None):
        """Initialize URL discovery"""
        self.config = config or get_settings()
        self.discovered_urls: Set[str] = set()
        
        logger.info("URLDiscovery initialized")
    
    def get_sitemap_urls(self, base_url: str = None) -> List[str]:
        """
        Parse XML sitemap and extract all URLs
        
        Args:
            base_url: Base URL for the website
            
        Returns:
            List of discovered URLs
        """
        if not base_url:
            base_url = self.config.tekyz_base_url
            
        sitemap_urls = []
        
        # Common sitemap locations
        sitemap_locations = [
            f"{base_url}/sitemap.xml",
            f"{base_url}/sitemap_index.xml",
            f"{base_url}/wp-sitemap.xml",
            f"{base_url}/sitemap.xml.gz"
        ]
        
        for sitemap_url in sitemap_locations:
            try:
                logger.info(f"Checking sitemap: {sitemap_url}")
                
                response = requests.get(
                    sitemap_url,
                    timeout=self.config.request_timeout,
                    headers={'User-Agent': 'Mozilla/5.0 (compatible; TekyzBot/1.0)'}
                )
                
                if response.status_code == 200:
                    urls = self._parse_sitemap_xml(response.content)
                    sitemap_urls.extend(urls)
                    logger.success(f"Found {len(urls)} URLs in {sitemap_url}")
                    break  # Use first available sitemap
                    
            except Exception as e:
                logger.warning(f"Could not fetch sitemap {sitemap_url}: {str(e)}")
                continue
        
        # Filter and validate URLs
        valid_urls = []
        for url in sitemap_urls:
            if self.validate_url(url):
                valid_urls.append(url)
        
        logger.info(f"Discovered {len(valid_urls)} valid URLs from sitemap")
        return valid_urls
    
    def _parse_sitemap_xml(self, xml_content: bytes) -> List[str]:
        """
        Parse XML sitemap content and extract URLs
        
        Args:
            xml_content: Raw XML content
            
        Returns:
            List of URLs found in sitemap
        """
        urls = []
        
        try:
            root = ET.fromstring(xml_content)
            
            # Handle different sitemap formats
            # Standard sitemap format
            for url_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                loc_elem = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                if loc_elem is not None and loc_elem.text:
                    urls.append(loc_elem.text.strip())
            
            # Sitemap index format
            for sitemap_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap'):
                loc_elem = sitemap_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                if loc_elem is not None and loc_elem.text:
                    # Recursively fetch nested sitemaps
                    nested_urls = self._fetch_nested_sitemap(loc_elem.text.strip())
                    urls.extend(nested_urls)
            
            # Fallback for non-standard formats
            if not urls:
                # Try to find any <loc> tags
                for loc in root.iter():
                    if loc.tag.endswith('loc') and loc.text:
                        urls.append(loc.text.strip())
                        
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {str(e)}")
        except Exception as e:
            logger.error(f"Error parsing sitemap: {str(e)}")
        
        return urls
    
    def _fetch_nested_sitemap(self, sitemap_url: str) -> List[str]:
        """
        Fetch and parse nested sitemap
        
        Args:
            sitemap_url: URL of nested sitemap
            
        Returns:
            List of URLs from nested sitemap
        """
        try:
            response = requests.get(
                sitemap_url,
                timeout=self.config.request_timeout,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; TekyzBot/1.0)'}
            )
            
            if response.status_code == 200:
                return self._parse_sitemap_xml(response.content)
                
        except Exception as e:
            logger.warning(f"Could not fetch nested sitemap {sitemap_url}: {str(e)}")
        
        return []
    
    def get_manual_urls(self) -> List[str]:
        """
        Return predefined manual URL list
        
        Returns:
            List of manually defined URLs
        """
        manual_urls = []
        
        for category, urls in TEKYZ_URLS.items():
            manual_urls.extend(urls)
        
        logger.info(f"Retrieved {len(manual_urls)} manual URLs")
        return manual_urls
    
    def discover_urls_from_page(self, page_content: str, base_url: str) -> List[str]:
        """
        Discover additional URLs from a page's content
        
        Args:
            page_content: HTML content of the page
            base_url: Base URL for resolving relative links
            
        Returns:
            List of discovered URLs
        """
        from bs4 import BeautifulSoup
        
        discovered = []
        
        try:
            soup = BeautifulSoup(page_content, 'lxml')
            
            # Find all links
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Resolve relative URLs
                full_url = urljoin(base_url, href)
                
                # Validate and add
                if self.validate_url(full_url):
                    discovered.append(full_url)
            
            # Remove duplicates
            discovered = list(set(discovered))
            
        except Exception as e:
            logger.error(f"Error discovering URLs from page: {str(e)}")
        
        return discovered
    
    def validate_url(self, url: str) -> bool:
        """
        Validate if URL should be scraped
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL is valid for scraping
        """
        if not url or not isinstance(url, str):
            return False
        
        # Use the validation function from config
        return is_valid_tekyz_url(url)
    
    def get_all_urls(self) -> List[str]:
        """
        Get all URLs from both sitemap and manual sources
        
        Returns:
            Combined list of all discovered URLs
        """
        all_urls = set()
        
        # Get manual URLs first (guaranteed to work)
        manual_urls = self.get_manual_urls()
        all_urls.update(manual_urls)
        
        # Try to get sitemap URLs
        try:
            sitemap_urls = self.get_sitemap_urls()
            all_urls.update(sitemap_urls)
        except Exception as e:
            logger.warning(f"Could not get sitemap URLs: {str(e)}")
        
        # Convert to sorted list
        final_urls = sorted(list(all_urls))
        
        logger.info(f"Total discovered URLs: {len(final_urls)}")
        
        # Log URL breakdown
        for category, urls in TEKYZ_URLS.items():
            category_count = sum(1 for url in final_urls if any(u in url for u in urls))
            logger.info(f"  {category}: {category_count} URLs")
        
        return final_urls
    
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
    
    def prioritize_urls(self, urls: List[str]) -> List[str]:
        """
        Prioritize URLs based on importance
        
        Args:
            urls: List of URLs to prioritize
            
        Returns:
            URLs sorted by priority
        """
        def get_priority(url: str) -> int:
            """Get priority score for URL (lower = higher priority)"""
            # Core pages have highest priority
            if any(core_url in url for core_url in TEKYZ_URLS['core_pages']):
                return 1
            
            # Service pages
            if '/services' in url:
                return 2
            
            # Portfolio pages
            if '/portfolio' in url:
                return 3
            
            # Team pages
            if '/team' in url:
                return 4
            
            # Other pages
            return 5
        
        # Sort by priority, then alphabetically
        return sorted(urls, key=lambda x: (get_priority(x), x)) 