"""
Base scraper class for extracting content from Tekyz website
"""

import asyncio
import time
from typing import Optional, Dict, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fake_useragent import UserAgent
from loguru import logger
from bs4 import BeautifulSoup

from config.settings import get_settings


class TekyzScraper:
    """Main scraper class for tekyz.com with error handling and rate limiting"""
    
    def __init__(self, config=None):
        """Initialize the scraper with configuration"""
        self.config = config or get_settings()
        self.session = self._setup_session()
        self.user_agent = UserAgent()
        self.last_request_time = 0
        
        logger.info("TekyzScraper initialized")
    
    def _setup_session(self) -> requests.Session:
        """Set up requests session with retry strategy"""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with rotating user agent"""
        return {
            'User-Agent': self.user_agent.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def _rate_limit(self):
        """Implement rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.config.scraping_delay:
            sleep_time = self.config.scraping_delay - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def scrape_page(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Main scraping method for a single page
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary containing page data or None if failed
        """
        logger.info(f"Scraping page: {url}")
        
        try:
            # Apply rate limiting
            self._rate_limit()
            
            # Make request
            response = self.session.get(
                url,
                headers=self._get_headers(),
                timeout=self.config.request_timeout
            )
            
            # Validate response
            if not self.is_valid_page(response):
                logger.warning(f"Invalid page response for {url}")
                return None
            
            # Extract content
            page_data = self.get_page_content(url, response)
            
            if page_data:
                logger.success(f"Successfully scraped: {url}")
                return page_data
            else:
                logger.warning(f"No content extracted from: {url}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error scraping {url}: {str(e)}")
            return None
    
    def get_page_content(self, url: str, response: requests.Response) -> Optional[Dict[str, Any]]:
        """
        Extract raw HTML content and basic metadata
        
        Args:
            url: Source URL
            response: HTTP response object
            
        Returns:
            Dictionary with page content and metadata
        """
        try:
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Extract basic metadata
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ""
            
            meta_description = soup.find('meta', attrs={'name': 'description'})
            description = meta_description.get('content', '') if meta_description else ""
            
            # Get all text content
            text_content = soup.get_text()
            
            page_data = {
                'url': url,
                'title': title_text,
                'description': description,
                'html': str(soup),
                'text': text_content,
                'status_code': response.status_code,
                'scraped_at': time.time(),
                'content_length': len(response.content),
                'response_headers': dict(response.headers)
            }
            
            return page_data
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return None
    
    def is_valid_page(self, response: requests.Response) -> bool:
        """
        Check if page response is valid Tekyz content
        
        Args:
            response: HTTP response object
            
        Returns:
            True if valid page, False otherwise
        """
        # Check status code
        if response.status_code != 200:
            logger.warning(f"Invalid status code: {response.status_code}")
            return False
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type:
            logger.warning(f"Invalid content type: {content_type}")
            return False
        
        # Check if it's actually tekyz.com content
        response_text_lower = response.text.lower()
        if 'tekyz' not in response_text_lower:
            logger.warning("Page doesn't appear to be Tekyz content")
            logger.debug(f"Response text preview: {response.text[:200]}...")
            return False
        
        # Check for error pages
        try:
            soup = BeautifulSoup(response.content, 'lxml')
            title_element = soup.find('title')
            if title_element:
                title_text = title_element.get_text().lower()
                if '404' in title_text or 'not found' in title_text:
                    logger.warning(f"404 error page detected, title: {title_text}")
                    return False
        except Exception as e:
            logger.warning(f"Error checking for 404 page: {str(e)}")
        
        logger.debug("Page validation passed")
        return True
    
    def close(self):
        """Close the session"""
        if self.session:
            self.session.close()
            logger.info("Scraper session closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close() 