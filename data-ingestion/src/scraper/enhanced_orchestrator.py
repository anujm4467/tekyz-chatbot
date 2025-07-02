"""
Enhanced Scraping Orchestrator for Comprehensive Page Coverage
This orchestrator coordinates advanced URL discovery and enhanced scraping
"""

import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from loguru import logger
from tqdm import tqdm

from .enhanced_discovery import EnhancedURLDiscovery
from .enhanced_scraper import EnhancedScraper, ScrapingResult
from .content_extractor import ContentExtractor
from config.settings import get_settings


@dataclass
class ComprehensiveStats:
    """Comprehensive statistics for the entire scraping operation"""
    discovery_time: float = 0.0
    scraping_time: float = 0.0
    processing_time: float = 0.0
    total_time: float = 0.0
    
    urls_discovered: int = 0
    urls_attempted: int = 0
    urls_successful: int = 0
    urls_failed: int = 0
    
    total_content_length: int = 0
    average_content_length: int = 0
    total_word_count: int = 0
    average_word_count: int = 0
    
    pages_by_type: Dict[str, int] = None
    success_rate: float = 0.0
    
    def __post_init__(self):
        if self.pages_by_type is None:
            self.pages_by_type = {}


class EnhancedScrapingOrchestrator:
    """Orchestrates comprehensive scraping with enhanced discovery and extraction"""
    
    def __init__(self, config=None):
        self.config = config or get_settings()
        self.stats = ComprehensiveStats()
        
        # Create output directories
        self.raw_data_dir = Path("data/raw")
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Enhanced Scraping Orchestrator initialized")
    
    async def run_comprehensive_scrape(
        self,
        max_discovery_depth: int = 5,
        max_discovery_pages: int = 500,
        max_scraping_concurrent: int = 5,
        enable_content_extraction: bool = True
    ) -> Dict[str, Any]:
        """
        Execute comprehensive scraping with advanced discovery and extraction
        
        Args:
            max_discovery_depth: Maximum crawling depth for URL discovery
            max_discovery_pages: Maximum pages to discover
            max_scraping_concurrent: Maximum concurrent scraping requests
            enable_content_extraction: Whether to run content extraction
            
        Returns:
            Comprehensive results dictionary
        """
        start_time = time.time()
        logger.info("🚀 Starting comprehensive scraping operation")
        
        try:
            # Phase 1: Enhanced URL Discovery
            logger.info("📡 Phase 1: Enhanced URL Discovery")
            discovery_start = time.time()
            
            discovery = EnhancedURLDiscovery(self.config)
            discovered_urls = await discovery.discover_all_urls(
                max_depth=max_discovery_depth,
                max_pages=max_discovery_pages
            )
            
            self.stats.discovery_time = time.time() - discovery_start
            self.stats.urls_discovered = len(discovered_urls)
            
            if not discovered_urls:
                logger.error("❌ No URLs discovered for scraping")
                return self._create_empty_result("No URLs discovered")
            
            logger.success(f"✅ Discovered {len(discovered_urls)} URLs in {self.stats.discovery_time:.2f}s")
            
            # Phase 2: Enhanced Scraping
            logger.info("🕷️ Phase 2: Enhanced Scraping")
            scraping_start = time.time()
            
            async with EnhancedScraper(self.config) as scraper:
                scraping_results = await scraper.scrape_urls(
                    urls=discovered_urls,
                    max_concurrent=max_scraping_concurrent
                )
            
            self.stats.scraping_time = time.time() - scraping_start
            self.stats.urls_attempted = len(scraping_results)
            
            # Filter successful results
            successful_results = [r for r in scraping_results if r.success]
            self.stats.urls_successful = len(successful_results)
            self.stats.urls_failed = self.stats.urls_attempted - self.stats.urls_successful
            
            if self.stats.urls_attempted > 0:
                self.stats.success_rate = (self.stats.urls_successful / self.stats.urls_attempted) * 100
            
            logger.success(f"✅ Scraping completed: {self.stats.urls_successful}/{self.stats.urls_attempted} "
                          f"successful ({self.stats.success_rate:.1f}%) in {self.stats.scraping_time:.2f}s")
            
            # Phase 3: Content Processing (if enabled)
            processed_results = successful_results
            
            if enable_content_extraction and successful_results:
                logger.info("🔄 Phase 3: Content Processing")
                processing_start = time.time()
                
                processed_results = await self._process_scraped_content(successful_results)
                self.stats.processing_time = time.time() - processing_start
                
                logger.success(f"✅ Content processing completed in {self.stats.processing_time:.2f}s")
            
            # Phase 4: Calculate final statistics
            self._calculate_final_stats(processed_results)
            
            # Phase 5: Save results
            logger.info("💾 Phase 4: Saving Results")
            output_file = await self._save_comprehensive_results(processed_results, discovered_urls)
            
            # Generate final report
            self.stats.total_time = time.time() - start_time
            report = self._generate_comprehensive_report(processed_results)
            
            result = {
                'success': True,
                'stats': asdict(self.stats),
                'output_file': str(output_file),
                'scraped_pages': len(processed_results),
                'discovered_urls': len(discovered_urls),
                'report': report,
                'processing_phases': {
                    'discovery_time': self.stats.discovery_time,
                    'scraping_time': self.stats.scraping_time,
                    'processing_time': self.stats.processing_time,
                    'total_time': self.stats.total_time
                }
            }
            
            logger.success(f"🎉 Comprehensive scraping completed successfully!")
            logger.success(f"   📊 Results: {self.stats.urls_successful} pages scraped from {self.stats.urls_discovered} discovered URLs")
            logger.success(f"   ⏱️ Total time: {self.stats.total_time:.2f}s")
            logger.success(f"   📈 Success rate: {self.stats.success_rate:.1f}%")
            logger.success(f"   📄 Average content length: {self.stats.average_content_length:,} characters")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Comprehensive scraping failed: {str(e)}")
            self.stats.total_time = time.time() - start_time
            return self._create_error_result(str(e))
    
    async def _process_scraped_content(self, scraping_results: List[ScrapingResult]) -> List[Dict[str, Any]]:
        """Process scraped content for better extraction and cleaning"""
        processed_results = []
        content_extractor = ContentExtractor()
        
        logger.info(f"Processing content from {len(scraping_results)} scraped pages")
        
        with tqdm(scraping_results, desc="Processing content") as pbar:
            for result in pbar:
                try:
                    # Extract enhanced content if HTML is available
                    if result.html:
                        extracted_content = content_extractor.extract_main_content(result.html, result.url)
                        
                        # Combine scraping result with enhanced extraction
                        processed_data = {
                            # Original scraping data
                            'url': result.url,
                            'title': result.title,
                            'description': result.description,
                            'status_code': result.status_code,
                            'scraped_at': result.scraped_at,
                            'processing_time': result.processing_time,
                            'method_used': result.method_used,
                            
                            # Enhanced extracted content
                            'content': extracted_content.get('clean_text', result.content),
                            'html': result.html,
                            'word_count': len(extracted_content.get('clean_text', result.content).split()),
                            'content_length': len(result.html),
                            
                            # Metadata combination
                            'metadata': {
                                **result.metadata,
                                **extracted_content.get('metadata', {}),
                                'page_type': extracted_content.get('metadata', {}).get('page_type', result.metadata.get('page_type', 'unknown')),
                                'extraction_success': extracted_content.get('extraction_success', True),
                                'content_metrics': extracted_content.get('content_metrics', {})
                            },
                            
                            # Additional extracted data
                            'headings': extracted_content.get('headings', []),
                            'lists': extracted_content.get('lists', []),
                            
                            # Processing timestamp
                            'enhanced_processing_timestamp': time.time()
                        }
                    else:
                        # Fallback for results without HTML
                        processed_data = {
                            'url': result.url,
                            'title': result.title,
                            'description': result.description,
                            'content': result.content,
                            'html': result.html,
                            'word_count': result.word_count,
                            'content_length': result.content_length,
                            'status_code': result.status_code,
                            'scraped_at': result.scraped_at,
                            'processing_time': result.processing_time,
                            'method_used': result.method_used,
                            'metadata': result.metadata,
                            'headings': [],
                            'lists': [],
                            'enhanced_processing_timestamp': time.time()
                        }
                    
                    processed_results.append(processed_data)
                    
                except Exception as e:
                    logger.warning(f"Failed to process content from {result.url}: {str(e)}")
                    # Add the original result as fallback
                    processed_results.append({
                        'url': result.url,
                        'title': result.title,
                        'description': result.description,
                        'content': result.content,
                        'html': result.html,
                        'word_count': result.word_count,
                        'content_length': result.content_length,
                        'status_code': result.status_code,
                        'scraped_at': result.scraped_at,
                        'processing_time': result.processing_time,
                        'method_used': result.method_used,
                        'metadata': result.metadata,
                        'headings': [],
                        'lists': [],
                        'processing_error': str(e),
                        'enhanced_processing_timestamp': time.time()
                    })
        
        logger.success(f"Content processing completed for {len(processed_results)} pages")
        return processed_results
    
    def _calculate_final_stats(self, results: List[Dict[str, Any]]):
        """Calculate comprehensive final statistics"""
        if not results:
            return
        
        # Content statistics
        total_content_length = sum(r.get('content_length', 0) for r in results)
        total_word_count = sum(r.get('word_count', 0) for r in results)
        
        self.stats.total_content_length = total_content_length
        self.stats.total_word_count = total_word_count
        self.stats.average_content_length = total_content_length // len(results)
        self.stats.average_word_count = total_word_count // len(results)
        
        # Page type distribution
        page_types = {}
        for result in results:
            page_type = result.get('metadata', {}).get('page_type', 'unknown')
            page_types[page_type] = page_types.get(page_type, 0) + 1
        
        self.stats.pages_by_type = page_types
    
    async def _save_comprehensive_results(self, results: List[Dict[str, Any]], discovered_urls: List[str]) -> Path:
        """Save comprehensive results with detailed metadata"""
        timestamp = int(time.time())
        output_file = self.raw_data_dir / f"tekyz_comprehensive_scraped_data_{timestamp}.json"
        
        comprehensive_data = {
            'scraping_metadata': {
                'timestamp': timestamp,
                'scraping_method': 'enhanced_comprehensive',
                'total_discovered_urls': len(discovered_urls),
                'total_scraped_pages': len(results),
                'success_rate': self.stats.success_rate,
                'processing_stats': asdict(self.stats)
            },
            'discovered_urls': discovered_urls,
            'scraped_pages': results,
            'url_breakdown_by_type': self._categorize_urls(discovered_urls),
            'content_summary': self._create_content_summary(results)
        }
        
        # Save with pretty formatting
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Comprehensive results saved to {output_file}")
        logger.info(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        return output_file
    
    def _categorize_urls(self, urls: List[str]) -> Dict[str, List[str]]:
        """Categorize URLs by type for better organization"""
        categories = {
            'core_pages': [],
            'services': [],
            'portfolio': [],
            'blog': [],
            'company': [],
            'technology': [],
            'industry': [],
            'contact': [],
            'careers': [],
            'other': []
        }
        
        for url in urls:
            url_lower = url.lower()
            
            if url_lower.endswith('/') and url_lower.count('/') <= 3:
                categories['core_pages'].append(url)
            elif any(keyword in url_lower for keyword in ['/services', '/solutions']):
                categories['services'].append(url)
            elif any(keyword in url_lower for keyword in ['/portfolio', '/work', '/projects', '/case-studies']):
                categories['portfolio'].append(url)
            elif any(keyword in url_lower for keyword in ['/blog', '/news', '/articles', '/insights']):
                categories['blog'].append(url)
            elif any(keyword in url_lower for keyword in ['/about', '/team', '/company', '/culture']):
                categories['company'].append(url)
            elif any(keyword in url_lower for keyword in ['/technologies', '/expertise', '/skills']):
                categories['technology'].append(url)
            elif '/industries' in url_lower:
                categories['industry'].append(url)
            elif '/contact' in url_lower:
                categories['contact'].append(url)
            elif any(keyword in url_lower for keyword in ['/careers', '/jobs', '/join']):
                categories['careers'].append(url)
            else:
                categories['other'].append(url)
        
        return categories
    
    def _create_content_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a summary of content characteristics"""
        if not results:
            return {}
        
        return {
            'total_pages': len(results),
            'total_words': self.stats.total_word_count,
            'average_words_per_page': self.stats.average_word_count,
            'total_content_length': self.stats.total_content_length,
            'average_content_length': self.stats.average_content_length,
            'pages_by_type': self.stats.pages_by_type,
            'content_distribution': {
                'short_pages': len([r for r in results if r.get('word_count', 0) < 100]),
                'medium_pages': len([r for r in results if 100 <= r.get('word_count', 0) < 500]),
                'long_pages': len([r for r in results if r.get('word_count', 0) >= 500])
            }
        }
    
    def _generate_comprehensive_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive scraping report"""
        return {
            'executive_summary': {
                'total_urls_discovered': self.stats.urls_discovered,
                'total_pages_scraped': self.stats.urls_successful,
                'success_rate': f"{self.stats.success_rate:.1f}%",
                'total_processing_time': f"{self.stats.total_time:.2f}s",
                'average_content_per_page': f"{self.stats.average_content_length:,} chars"
            },
            'performance_metrics': {
                'discovery_time': f"{self.stats.discovery_time:.2f}s",
                'scraping_time': f"{self.stats.scraping_time:.2f}s",
                'processing_time': f"{self.stats.processing_time:.2f}s",
                'average_time_per_page': f"{(self.stats.scraping_time / max(self.stats.urls_successful, 1)):.2f}s"
            },
            'content_analysis': {
                'total_words_extracted': f"{self.stats.total_word_count:,}",
                'average_words_per_page': f"{self.stats.average_word_count:,}",
                'pages_by_type': self.stats.pages_by_type
            },
            'quality_metrics': {
                'successful_extractions': len([r for r in results if r.get('content', '').strip()]),
                'pages_with_metadata': len([r for r in results if r.get('metadata')]),
                'pages_with_headings': len([r for r in results if r.get('headings')])
            }
        }
    
    def _create_empty_result(self, reason: str) -> Dict[str, Any]:
        """Create empty result with error reason"""
        return {
            'success': False,
            'error': reason,
            'stats': asdict(self.stats),
            'scraped_pages': 0,
            'discovered_urls': 0
        }
    
    def _create_error_result(self, error: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            'success': False,
            'error': error,
            'stats': asdict(self.stats),
            'scraped_pages': 0,
            'discovered_urls': 0,
            'total_time': self.stats.total_time
        } 