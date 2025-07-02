"""
Enhanced Scraper with Dynamic Content Support and Advanced Features
This module provides comprehensive scraping capabilities with fallback strategies
"""

import asyncio
import aiohttp
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass, asdict
from loguru import logger
from bs4 import BeautifulSoup
import re
from concurrent.futures import ThreadPoolExecutor
import hashlib

from config.settings import get_settings


@dataclass
class ScrapingResult:
    """Result of a scraping operation"""
    url: str
    success: bool
    status_code: Optional[int] = None
    title: str = ""
    description: str = ""
    content: str = ""
    html: str = ""
    word_count: int = 0
    content_length: int = 0
    scraped_at: float = 0.0
    processing_time: float = 0.0
    method_used: str = "static"  # static, dynamic, fallback
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ScrapingStats:
    """Statistics for scraping session"""
    total_urls: int = 0
    successful: int = 0
    failed: int = 0
    static_success: int = 0
    dynamic_success: int = 0
    fallback_success: int = 0
    total_processing_time: float = 0.0
    average_content_length: int = 0
    total_content_length: int = 0


class EnhancedScraper:
    """Enhanced scraper with multiple strategies and robust error handling"""
    
    def __init__(self, config=None):
        self.config = config or get_settings()
        self.stats = ScrapingStats()
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Enhanced headers to appear more browser-like
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
        
        # Content quality thresholds
        self.min_content_length = 100
        self.min_word_count = 20
        
        logger.info("Enhanced Scraper initialized")
    
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(
            limit=10,  # Connection pool size
            limit_per_host=5,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        timeout = aiohttp.ClientTimeout(
            total=60,  # Total timeout
            connect=10,  # Connection timeout
            sock_read=30  # Socket read timeout
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=self.headers
        )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def scrape_urls(self, urls: List[str], max_concurrent: int = 5) -> List[ScrapingResult]:
        """
        Scrape multiple URLs concurrently with rate limiting
        
        Args:
            urls: List of URLs to scrape
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of scraping results
        """
        logger.info(f"🚀 Starting enhanced scraping of {len(urls)} URLs")
        start_time = time.time()
        
        self.stats.total_urls = len(urls)
        results = []
        
        # Process URLs in batches to avoid overwhelming the server
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_with_semaphore(url: str) -> ScrapingResult:
            async with semaphore:
                result = await self.scrape_single_url(url)
                # Rate limiting between requests
                await asyncio.sleep(0.5)
                return result
        
        # Execute scraping tasks
        tasks = [scrape_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to scrape {urls[i]}: {str(result)}")
                error_result = ScrapingResult(
                    url=urls[i],
                    success=False,
                    error=str(result),
                    scraped_at=time.time()
                )
                final_results.append(error_result)
                self.stats.failed += 1
            else:
                final_results.append(result)
                if result.success:
                    self.stats.successful += 1
                    self.stats.total_content_length += result.content_length
                    
                    # Track method used
                    if result.method_used == "static":
                        self.stats.static_success += 1
                    elif result.method_used == "dynamic":
                        self.stats.dynamic_success += 1
                    elif result.method_used == "fallback":
                        self.stats.fallback_success += 1
                else:
                    self.stats.failed += 1
        
        # Calculate final statistics
        self.stats.total_processing_time = time.time() - start_time
        if self.stats.successful > 0:
            self.stats.average_content_length = self.stats.total_content_length // self.stats.successful
        
        logger.success(f"✅ Scraping completed: {self.stats.successful}/{self.stats.total_urls} successful "
                      f"in {self.stats.total_processing_time:.2f}s")
        
        # Save detailed results
        await self._save_scraping_report(final_results)
        
        return final_results
    
    async def scrape_single_url(self, url: str) -> ScrapingResult:
        """
        Scrape a single URL with multiple fallback strategies
        
        Args:
            url: URL to scrape
            
        Returns:
            ScrapingResult object
        """
        start_time = time.time()
        logger.debug(f"Scraping: {url}")
        
        # Try static scraping first (fastest)
        result = await self._static_scrape(url)
        
        if result and result.success and self._is_quality_content(result):
            result.processing_time = time.time() - start_time
            return result
        
        # If static scraping failed or low quality, try enhanced parsing
        result = await self._enhanced_static_scrape(url)
        
        if result and result.success and self._is_quality_content(result):
            result.processing_time = time.time() - start_time
            return result
        
        # If all methods failed, return the best attempt
        if result is None:
            result = ScrapingResult(
                url=url,
                success=False,
                error="All scraping methods failed",
                scraped_at=time.time(),
                processing_time=time.time() - start_time
            )
        
        result.processing_time = time.time() - start_time
        return result
    
    async def _static_scrape(self, url: str) -> Optional[ScrapingResult]:
        """Basic static scraping with BeautifulSoup"""
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.debug(f"HTTP {response.status} for {url}")
                    return ScrapingResult(
                        url=url,
                        success=False,
                        status_code=response.status,
                        error=f"HTTP {response.status}",
                        method_used="static",
                        scraped_at=time.time()
                    )
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' not in content_type:
                    return ScrapingResult(
                        url=url,
                        success=False,
                        error=f"Invalid content type: {content_type}",
                        method_used="static",
                        scraped_at=time.time()
                    )
                
                html = await response.text()
                soup = BeautifulSoup(html, 'lxml')
                
                # Extract content
                extracted = self._extract_content(soup, url, html)
                
                return ScrapingResult(
                    url=url,
                    success=True,
                    status_code=response.status,
                    title=extracted['title'],
                    description=extracted['description'],
                    content=extracted['content'],
                    html=html,
                    word_count=extracted['word_count'],
                    content_length=len(html),
                    method_used="static",
                    scraped_at=time.time(),
                    metadata=extracted['metadata']
                )
                
        except asyncio.TimeoutError:
            logger.debug(f"Timeout for {url}")
            return ScrapingResult(
                url=url,
                success=False,
                error="Request timeout",
                method_used="static",
                scraped_at=time.time()
            )
        except Exception as e:
            logger.debug(f"Static scraping failed for {url}: {str(e)}")
            return ScrapingResult(
                url=url,
                success=False,
                error=f"Static scraping error: {str(e)}",
                method_used="static",
                scraped_at=time.time()
            )
    
    async def _enhanced_static_scrape(self, url: str) -> Optional[ScrapingResult]:
        """Enhanced static scraping with retry and better parsing"""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                # Wait between retries
                if attempt > 0:
                    await asyncio.sleep(2)
                
                # Try with different headers
                enhanced_headers = self.headers.copy()
                if attempt == 1:
                    enhanced_headers['User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                
                async with self.session.get(url, headers=enhanced_headers) as response:
                    if response.status != 200:
                        continue
                    
                    html = await response.text()
                    
                    # Try different parsers
                    parsers = ['lxml', 'html.parser', 'html5lib']
                    
                    for parser in parsers:
                        try:
                            soup = BeautifulSoup(html, parser)
                            extracted = self._extract_content(soup, url, html)
                            
                            # Check if we got quality content
                            if extracted['word_count'] >= self.min_word_count:
                                return ScrapingResult(
                                    url=url,
                                    success=True,
                                    status_code=response.status,
                                    title=extracted['title'],
                                    description=extracted['description'],
                                    content=extracted['content'],
                                    html=html,
                                    word_count=extracted['word_count'],
                                    content_length=len(html),
                                    method_used="enhanced_static",
                                    scraped_at=time.time(),
                                    metadata=extracted['metadata']
                                )
                        except Exception:
                            continue
                
            except Exception as e:
                logger.debug(f"Enhanced scraping attempt {attempt + 1} failed for {url}: {str(e)}")
                continue
        
        return None
    
    def _extract_content(self, soup: BeautifulSoup, url: str, html: str) -> Dict[str, Any]:
        """
        Extract and clean content from BeautifulSoup object
        
        Args:
            soup: BeautifulSoup object
            url: Source URL
            html: Raw HTML
            
        Returns:
            Dictionary with extracted content
        """
        # Remove unwanted elements
        self._remove_unwanted_elements(soup)
        
        # Extract title
        title = ""
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
        
        # Extract meta description
        description = ""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            description = meta_desc['content'].strip()
        
        # Extract main content using multiple strategies
        content = self._extract_main_content(soup)
        
        # Calculate word count
        word_count = len(content.split()) if content else 0
        
        # Extract metadata
        metadata = self._extract_metadata(soup, url)
        
        return {
            'title': title,
            'description': description,
            'content': content,
            'word_count': word_count,
            'metadata': metadata
        }
    
    def _remove_unwanted_elements(self, soup: BeautifulSoup):
        """Remove unwanted HTML elements"""
        # Elements to remove completely
        unwanted_tags = [
            'script', 'style', 'nav', 'header', 'footer', 'aside',
            'menu', 'noscript', 'iframe', 'object', 'embed'
        ]
        
        for tag_name in unwanted_tags:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        
        # Remove elements with common unwanted classes/IDs
        unwanted_selectors = [
            '.nav', '.navbar', '.navigation', '.menu', '.sidebar',
            '.footer', '.header', '.advertisement', '.ads', '.ad',
            '.social', '.share', '.comments', '.comment', '.pagination',
            '.breadcrumb', '.cookie', '.popup', '.modal', '.overlay',
            '#nav', '#navbar', '#navigation', '#menu', '#sidebar',
            '#footer', '#header', '#ads', '#social', '#comments'
        ]
        
        for selector in unwanted_selectors:
            for element in soup.select(selector):
                element.decompose()
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content using multiple strategies"""
        content_texts = []
        
        # Strategy 1: Look for main content containers
        main_selectors = [
            'main', 'article', '[role="main"]', '.main-content',
            '.content', '#main', '#content', '.post-content',
            '.entry-content', '.page-content', '.article-content'
        ]
        
        for selector in main_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(separator=' ', strip=True)
                if len(text) > 100:  # Minimum content length
                    content_texts.append(text)
        
        # Strategy 2: Extract from paragraphs and headings
        if not content_texts:
            paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 20:  # Skip very short paragraphs
                    content_texts.append(text)
        
        # Strategy 3: Fallback to body content
        if not content_texts:
            body = soup.find('body')
            if body:
                content_texts.append(body.get_text(separator=' ', strip=True))
        
        # Combine and clean content
        combined_content = ' '.join(content_texts)
        
        # Clean up whitespace
        cleaned_content = re.sub(r'\s+', ' ', combined_content).strip()
        
        return cleaned_content
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract additional metadata from the page"""
        metadata = {
            'url': url,
            'domain': urlparse(url).netloc,
            'path': urlparse(url).path
        }
        
        # Extract Open Graph metadata
        og_tags = soup.find_all('meta', attrs={'property': re.compile(r'^og:')})
        for tag in og_tags:
            property_name = tag.get('property', '').replace('og:', '')
            content = tag.get('content', '')
            if property_name and content:
                metadata[f'og_{property_name}'] = content
        
        # Extract Twitter Card metadata
        twitter_tags = soup.find_all('meta', attrs={'name': re.compile(r'^twitter:')})
        for tag in twitter_tags:
            name = tag.get('name', '').replace('twitter:', '')
            content = tag.get('content', '')
            if name and content:
                metadata[f'twitter_{name}'] = content
        
        # Extract canonical URL
        canonical = soup.find('link', attrs={'rel': 'canonical'})
        if canonical and canonical.get('href'):
            metadata['canonical_url'] = canonical['href']
        
        # Extract language
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang'):
            metadata['language'] = html_tag['lang']
        
        # Classify page type
        metadata['page_type'] = self._classify_page_type(url)
        
        return metadata
    
    def _classify_page_type(self, url: str) -> str:
        """Classify the type of page based on URL"""
        url_lower = url.lower()
        path = urlparse(url).path.lower()
        
        if path == '/' or path == '':
            return 'homepage'
        elif any(keyword in path for keyword in ['/about', '/company', '/team']):
            return 'about'
        elif '/services' in path or '/solutions' in path:
            return 'services'
        elif any(keyword in path for keyword in ['/portfolio', '/work', '/projects', '/case-studies']):
            return 'portfolio'
        elif any(keyword in path for keyword in ['/blog', '/news', '/articles', '/insights']):
            return 'blog'
        elif '/contact' in path:
            return 'contact'
        elif '/careers' in path or '/jobs' in path:
            return 'careers'
        elif '/technologies' in path or '/expertise' in path:
            return 'technology'
        elif '/industries' in path:
            return 'industry'
        else:
            return 'other'
    
    def _is_quality_content(self, result: ScrapingResult) -> bool:
        """Check if the scraped content meets quality thresholds"""
        if not result.success:
            return False
        
        # Check minimum content length
        if result.content_length < self.min_content_length:
            return False
        
        # Check minimum word count
        if result.word_count < self.min_word_count:
            return False
        
        # Check for actual content (not just HTML)
        if not result.content.strip():
            return False
        
        # Check for meaningful title
        if not result.title.strip():
            return False
        
        return True
    
    async def _save_scraping_report(self, results: List[ScrapingResult]):
        """Save detailed scraping report"""
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            result_dict = asdict(result)
            # Handle potential non-serializable metadata
            if result_dict['metadata']:
                result_dict['metadata'] = {k: str(v) for k, v in result_dict['metadata'].items()}
            serializable_results.append(result_dict)
        
        report = {
            'timestamp': time.time(),
            'stats': asdict(self.stats),
            'results': serializable_results,
            'summary': {
                'total_urls': self.stats.total_urls,
                'successful': self.stats.successful,
                'failed': self.stats.failed,
                'success_rate': (self.stats.successful / max(self.stats.total_urls, 1)) * 100,
                'average_content_length': self.stats.average_content_length,
                'total_processing_time': self.stats.total_processing_time
            }
        }
        
        # Save report
        report_file = Path("data/enhanced_scraping_report.json")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Enhanced scraping report saved to {report_file}") 