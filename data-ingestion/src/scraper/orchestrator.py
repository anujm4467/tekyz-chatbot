"""
Scraping orchestrator that coordinates all scraping components
"""

import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from loguru import logger
from tqdm import tqdm

from .base_scraper import TekyzScraper
from .url_discovery import URLDiscovery
from .content_extractor import ContentExtractor
from config.settings import get_settings


@dataclass
class ScrapingStats:
    """Statistics for scraping operation"""
    total_urls: int = 0
    successful_scrapes: int = 0
    failed_scrapes: int = 0
    start_time: float = 0
    end_time: float = 0
    total_content_length: int = 0
    average_page_size: int = 0
    
    @property
    def success_rate(self) -> float:
        if self.total_urls == 0:
            return 0.0
        return (self.successful_scrapes / self.total_urls) * 100
    
    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time if self.end_time > self.start_time else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['success_rate'] = self.success_rate
        data['duration_seconds'] = self.duration_seconds
        return data


class ScrapingOrchestrator:
    """Orchestrates the complete scraping process"""
    
    def __init__(self, config=None):
        """Initialize the orchestrator"""
        self.config = config or get_settings()
        self.url_discovery = URLDiscovery(config)
        self.content_extractor = ContentExtractor()
        self.stats = ScrapingStats()
        
        # Create output directories
        self.raw_data_dir = Path("data/raw")
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("ScrapingOrchestrator initialized")
    
    async def run_full_scrape(self) -> Dict[str, Any]:
        """
        Execute complete scraping process
        
        Returns:
            Dictionary containing scraping results and statistics
        """
        logger.info("Starting full scraping process")
        self.stats.start_time = time.time()
        
        try:
            # Step 1: Discover URLs
            logger.info("Step 1: Discovering URLs")
            urls = self.url_discovery.get_all_urls()
            
            if not urls:
                logger.error("No URLs discovered for scraping")
                return self._create_empty_result("No URLs discovered")
            
            # Prioritize URLs
            urls = self.url_discovery.prioritize_urls(urls)
            self.stats.total_urls = len(urls)
            
            logger.info(f"Found {len(urls)} URLs to scrape")
            
            # Step 2: Scrape pages with parallel processing
            logger.info("Step 2: Scraping pages")
            scraped_data = await self.parallel_scraping(urls, max_workers=3)
            
            # Step 3: Extract content from scraped pages
            logger.info("Step 3: Extracting content")
            extracted_data = self.extract_content_from_scraped_data(scraped_data)
            
            # Step 4: Save results
            logger.info("Step 4: Saving results")
            output_file = self.save_scraped_data(extracted_data)
            
            # Step 5: Generate report
            self.stats.end_time = time.time()
            report = self.generate_scraping_report(extracted_data)
            
            result = {
                'success': True,
                'stats': self.stats.to_dict(),
                'output_file': str(output_file),
                'scraped_pages': len(extracted_data),
                'report': report
            }
            
            logger.success(f"Scraping completed successfully! "
                          f"Scraped {self.stats.successful_scrapes}/{self.stats.total_urls} pages "
                          f"in {self.stats.duration_seconds:.2f} seconds")
            
            return result
            
        except Exception as e:
            logger.error(f"Scraping process failed: {str(e)}")
            self.stats.end_time = time.time()
            return self._create_error_result(str(e))
    
    async def scrape_single_page(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Scrape a single page with fallback strategies
        
        Args:
            url: URL to scrape
            
        Returns:
            Scraped page data or None if failed
        """
        logger.debug(f"Scraping single page: {url}")
        
        try:
            # Try static scraping first
            with TekyzScraper(self.config) as scraper:
                page_data = scraper.scrape_page(url)
                
                if page_data:
                    self.stats.successful_scrapes += 1
                    self.stats.total_content_length += page_data.get('content_length', 0)
                    return page_data
                else:
                    logger.warning(f"Static scraping failed for {url}")
            
            # TODO: Add dynamic scraping fallback here (Selenium)
            # For now, mark as failed
            self.stats.failed_scrapes += 1
            return None
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            self.stats.failed_scrapes += 1
            return None
    
    async def parallel_scraping(self, urls: List[str], max_workers: int = 3) -> List[Dict[str, Any]]:
        """
        Scrape multiple URLs in parallel
        
        Args:
            urls: List of URLs to scrape
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of scraped page data
        """
        logger.info(f"Starting parallel scraping of {len(urls)} URLs with {max_workers} workers")
        
        scraped_data = []
        
        # Create progress bar
        with tqdm(total=len(urls), desc="Scraping pages") as pbar:
            
            # Use ThreadPoolExecutor for I/O bound tasks
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                
                # Submit all scraping tasks
                future_to_url = {}
                
                for url in urls:
                    # Convert async function to sync for ThreadPoolExecutor
                    future = executor.submit(self._sync_scrape_wrapper, url)
                    future_to_url[future] = url
                
                # Process completed tasks
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    
                    try:
                        result = future.result()
                        if result:
                            scraped_data.append(result)
                            logger.debug(f"Successfully scraped: {url}")
                        else:
                            logger.warning(f"Failed to scrape: {url}")
                    
                    except Exception as e:
                        logger.error(f"Error processing {url}: {str(e)}")
                    
                    finally:
                        pbar.update(1)
                        
                        # Respect rate limiting
                        time.sleep(self.config.scraping_delay)
        
        logger.info(f"Parallel scraping completed. Successfully scraped {len(scraped_data)} pages")
        return scraped_data
    
    def _sync_scrape_wrapper(self, url: str) -> Optional[Dict[str, Any]]:
        """Synchronous wrapper for async scraping function"""
        return asyncio.run(self.scrape_single_page(url))
    
    def extract_content_from_scraped_data(self, scraped_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract clean content from scraped HTML data
        
        Args:
            scraped_data: List of scraped page data
            
        Returns:
            List of extracted content data
        """
        logger.info(f"Extracting content from {len(scraped_data)} scraped pages")
        
        extracted_data = []
        
        with tqdm(scraped_data, desc="Extracting content") as pbar:
            for page_data in pbar:
                try:
                    url = page_data.get('url', '')
                    html = page_data.get('html', '')
                    
                    if html:
                        # Extract content using content extractor
                        extracted_content = self.content_extractor.extract_main_content(html, url)
                        
                        # Combine with original scraped metadata
                        combined_data = {
                            **page_data,  # Original scraped data
                            **extracted_content,  # Extracted content
                            'processing_timestamp': time.time()
                        }
                        
                        extracted_data.append(combined_data)
                        
                    else:
                        logger.warning(f"No HTML content for {url}")
                        
                except Exception as e:
                    logger.error(f"Error extracting content from {page_data.get('url', 'unknown')}: {str(e)}")
        
        logger.success(f"Content extraction completed for {len(extracted_data)} pages")
        return extracted_data
    
    def save_scraped_data(self, data: List[Dict[str, Any]]) -> Path:
        """
        Save scraped data to JSON file
        
        Args:
            data: Scraped and processed data
            
        Returns:
            Path to saved file
        """
        timestamp = int(time.time())
        filename = f"tekyz_scraped_data_{timestamp}.json"
        output_file = self.raw_data_dir / filename
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.success(f"Scraped data saved to: {output_file}")
            
            # Also save a summary file
            summary_file = self.raw_data_dir / f"scraping_summary_{timestamp}.json"
            self._save_scraping_summary(data, summary_file)
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error saving scraped data: {str(e)}")
            raise
    
    def _save_scraping_summary(self, data: List[Dict[str, Any]], summary_file: Path):
        """Save a summary of scraped data"""
        
        summary = {
            'scraping_timestamp': time.time(),
            'total_pages': len(data),
            'successful_extractions': sum(1 for d in data if d.get('extraction_success', False)),
            'failed_extractions': sum(1 for d in data if not d.get('extraction_success', False)),
            'total_content_length': sum(len(d.get('clean_text', '')) for d in data),
            'average_content_length': 0,
            'pages_by_type': {},
            'stats': self.stats.to_dict()
        }
        
        # Calculate averages
        if len(data) > 0:
            summary['average_content_length'] = summary['total_content_length'] // len(data)
        
        # Count pages by type
        for item in data:
            page_type = item.get('metadata', {}).get('page_type', 'unknown')
            summary['pages_by_type'][page_type] = summary['pages_by_type'].get(page_type, 0) + 1
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Scraping summary saved to: {summary_file}")
            
        except Exception as e:
            logger.error(f"Error saving summary: {str(e)}")
    
    def generate_scraping_report(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a comprehensive scraping report"""
        
        report = {
            'overview': {
                'total_pages_scraped': len(data),
                'successful_extractions': sum(1 for d in data if d.get('extraction_success', False)),
                'failed_extractions': sum(1 for d in data if not d.get('extraction_success', False)),
                'success_rate': self.stats.success_rate,
                'total_duration': self.stats.duration_seconds
            },
            'content_stats': {
                'total_content_length': sum(len(d.get('clean_text', '')) for d in data),
                'average_content_length': 0,
                'total_words': sum(d.get('content_metrics', {}).get('word_count', 0) for d in data),
                'average_words_per_page': 0
            },
            'page_types': {},
            'errors': []
        }
        
        # Calculate averages
        if len(data) > 0:
            report['content_stats']['average_content_length'] = (
                report['content_stats']['total_content_length'] // len(data)
            )
            report['content_stats']['average_words_per_page'] = (
                report['content_stats']['total_words'] // len(data)
            )
        
        # Analyze page types
        for item in data:
            page_type = item.get('metadata', {}).get('page_type', 'unknown')
            if page_type not in report['page_types']:
                report['page_types'][page_type] = {
                    'count': 0,
                    'total_content_length': 0,
                    'total_words': 0
                }
            
            report['page_types'][page_type]['count'] += 1
            report['page_types'][page_type]['total_content_length'] += len(item.get('clean_text', ''))
            report['page_types'][page_type]['total_words'] += item.get('content_metrics', {}).get('word_count', 0)
        
        # Collect errors
        for item in data:
            if not item.get('extraction_success', False) and item.get('error'):
                report['errors'].append({
                    'url': item.get('url'),
                    'error': item.get('error')
                })
        
        return report
    
    def _create_empty_result(self, reason: str) -> Dict[str, Any]:
        """Create empty result with error reason"""
        return {
            'success': False,
            'error': reason,
            'stats': self.stats.to_dict(),
            'scraped_pages': 0
        }
    
    def _create_error_result(self, error: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            'success': False,
            'error': error,
            'stats': self.stats.to_dict(),
            'scraped_pages': self.stats.successful_scrapes
        } 