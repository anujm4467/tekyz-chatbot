"""
Content extraction engine for extracting clean text from HTML pages
"""

import re
from typing import Dict, List, Optional, Any
from bs4 import BeautifulSoup, Tag, NavigableString
from loguru import logger
from urllib.parse import urlparse


class ContentExtractor:
    """Extracts main content from HTML and removes unwanted elements"""
    
    def __init__(self):
        """Initialize the content extractor"""
        self.unwanted_tags = [
            'script', 'style', 'nav', 'header', 'footer', 'aside', 'menu',
            'noscript', 'iframe', 'object', 'embed', 'form', 'input', 'button',
            'select', 'textarea', 'label', 'fieldset', 'legend', 'link', 'meta'
        ]
        
        self.unwanted_classes = [
            'nav', 'navbar', 'navigation', 'menu', 'sidebar', 'footer', 'header',
            'advertisement', 'ads', 'ad', 'social', 'share', 'comments', 'comment',
            'pagination', 'breadcrumb', 'cookie', 'popup', 'modal', 'overlay'
        ]
        
        self.unwanted_ids = [
            'nav', 'navbar', 'navigation', 'menu', 'sidebar', 'footer', 'header',
            'advertisement', 'ads', 'social', 'comments', 'pagination', 'breadcrumb'
        ]
        
        self.content_selectors = [
            'main', 'article', '[role="main"]', '.main-content', '.content',
            '#main', '#content', '.post-content', '.entry-content', '.page-content'
        ]
        
        logger.info("ContentExtractor initialized")
    
    def extract_main_content(self, html: str, url: str) -> Dict[str, Any]:
        """
        Extract main content from HTML page
        
        Args:
            html: Raw HTML content
            url: Source URL for context
            
        Returns:
            Dictionary containing extracted content and metadata
        """
        try:
            soup = BeautifulSoup(html, 'lxml')
            
            # Remove unwanted elements first
            self._remove_unwanted_elements(soup)
            
            # Extract metadata
            metadata = self.extract_metadata(soup, url)
            
            # Find main content area
            main_content = self._find_main_content(soup)
            
            if not main_content:
                # Fallback to body content
                main_content = soup.find('body') or soup
            
            # Extract clean text
            clean_text = self._extract_clean_text(main_content)
            
            # Extract headings structure
            headings = self._extract_headings(main_content)
            
            # Extract lists and structured content
            lists = self._extract_lists(main_content)
            
            # Calculate content metrics
            content_metrics = self._calculate_content_metrics(clean_text)
            
            extracted_content = {
                'url': url,
                'metadata': metadata,
                'clean_text': clean_text,
                'headings': headings,
                'lists': lists,
                'content_metrics': content_metrics,
                'extraction_success': True
            }
            
            logger.success(f"Successfully extracted content from {url}")
            return extracted_content
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return {
                'url': url,
                'metadata': {},
                'clean_text': '',
                'headings': [],
                'lists': [],
                'content_metrics': {},
                'extraction_success': False,
                'error': str(e)
            }
    
    def extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Extract page metadata
        
        Args:
            soup: BeautifulSoup object
            url: Source URL
            
        Returns:
            Dictionary containing metadata
        """
        metadata = {
            'url': url,
            'domain': urlparse(url).netloc,
            'path': urlparse(url).path
        }
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text().strip()
        
        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            metadata['description'] = meta_desc['content'].strip()
        
        # Extract meta keywords
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords and meta_keywords.get('content'):
            metadata['keywords'] = [k.strip() for k in meta_keywords['content'].split(',')]
        
        # Extract Open Graph metadata
        og_title = soup.find('meta', attrs={'property': 'og:title'})
        if og_title and og_title.get('content'):
            metadata['og_title'] = og_title['content'].strip()
        
        og_description = soup.find('meta', attrs={'property': 'og:description'})
        if og_description and og_description.get('content'):
            metadata['og_description'] = og_description['content'].strip()
        
        # Extract canonical URL
        canonical = soup.find('link', attrs={'rel': 'canonical'})
        if canonical and canonical.get('href'):
            metadata['canonical_url'] = canonical['href']
        
        # Extract language
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang'):
            metadata['language'] = html_tag['lang']
        
        # Classify page type based on URL
        metadata['page_type'] = self._classify_page_type(url)
        
        return metadata
    
    def _remove_unwanted_elements(self, soup: BeautifulSoup):
        """Remove unwanted HTML elements"""
        
        # Remove unwanted tags
        for tag_name in self.unwanted_tags:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        
        # Remove elements with unwanted classes
        for class_name in self.unwanted_classes:
            for tag in soup.find_all(class_=re.compile(class_name, re.IGNORECASE)):
                tag.decompose()
        
        # Remove elements with unwanted IDs
        for id_name in self.unwanted_ids:
            for tag in soup.find_all(id=re.compile(id_name, re.IGNORECASE)):
                tag.decompose()
        
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, NavigableString) and 
                                     str(text).strip().startswith('<!--')):
            comment.extract()
    
    def _find_main_content(self, soup: BeautifulSoup) -> Optional[Tag]:
        """Find the main content area of the page"""
        
        # Try content selectors in order of preference
        for selector in self.content_selectors:
            try:
                content = soup.select_one(selector)
                if content and self._has_substantial_content(content):
                    logger.debug(f"Found main content using selector: {selector}")
                    return content
            except Exception as e:
                logger.debug(f"Selector {selector} failed: {str(e)}")
                continue
        
        # If no main content found, try to find the largest content block
        return self._find_largest_content_block(soup)
    
    def _find_largest_content_block(self, soup: BeautifulSoup) -> Optional[Tag]:
        """Find the largest content block by text length"""
        
        candidates = []
        
        # Look for divs with substantial content
        for div in soup.find_all('div'):
            text = div.get_text().strip()
            if len(text) > 100:  # Minimum content length
                candidates.append((div, len(text)))
        
        if candidates:
            # Sort by text length and return the largest
            candidates.sort(key=lambda x: x[1], reverse=True)
            logger.debug(f"Found largest content block with {candidates[0][1]} characters")
            return candidates[0][0]
        
        return None
    
    def _has_substantial_content(self, element: Tag) -> bool:
        """Check if element has substantial text content"""
        text = element.get_text().strip()
        return len(text) > 50  # Minimum threshold
    
    def _extract_clean_text(self, element: Tag) -> str:
        """Extract clean text from HTML element"""
        
        # Get all text content
        text_content = []
        
        for item in element.descendants:
            if isinstance(item, NavigableString):
                text = str(item).strip()
                if text:
                    text_content.append(text)
            elif isinstance(item, Tag):
                # Add extra spacing for block elements
                if item.name in ['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'br']:
                    text_content.append('\n')
        
        # Join and clean text
        raw_text = ' '.join(text_content)
        
        # Clean up text
        clean_text = self.clean_text(raw_text)
        
        return clean_text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        # Clean up common HTML artifacts
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&quot;', '"', text)
        text = re.sub(r'&apos;', "'", text)
        
        # Remove URLs that might be left in text
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remove phone numbers (basic pattern)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '---', text)
        
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        
        # Final cleanup
        text = text.strip()
        
        return text
    
    def _extract_headings(self, element: Tag) -> List[Dict[str, Any]]:
        """Extract heading structure from content"""
        
        headings = []
        
        for level in range(1, 7):  # h1 to h6
            for heading in element.find_all(f'h{level}'):
                heading_text = heading.get_text().strip()
                if heading_text:
                    headings.append({
                        'level': level,
                        'text': heading_text,
                        'clean_text': self.clean_text(heading_text)
                    })
        
        return headings
    
    def _extract_lists(self, element: Tag) -> List[Dict[str, Any]]:
        """Extract list content from HTML"""
        
        lists = []
        
        # Extract unordered lists
        for ul in element.find_all('ul'):
            items = []
            for li in ul.find_all('li', recursive=False):  # Direct children only
                item_text = li.get_text().strip()
                if item_text:
                    items.append(self.clean_text(item_text))
            
            if items:
                lists.append({
                    'type': 'unordered',
                    'items': items
                })
        
        # Extract ordered lists
        for ol in element.find_all('ol'):
            items = []
            for li in ol.find_all('li', recursive=False):  # Direct children only
                item_text = li.get_text().strip()
                if item_text:
                    items.append(self.clean_text(item_text))
            
            if items:
                lists.append({
                    'type': 'ordered',
                    'items': items
                })
        
        return lists
    
    def _calculate_content_metrics(self, text: str) -> Dict[str, Any]:
        """Calculate content quality metrics"""
        
        metrics = {
            'character_count': len(text),
            'word_count': len(text.split()) if text else 0,
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
            'sentence_count': len([s for s in re.split(r'[.!?]+', text) if s.strip()]),
            'avg_sentence_length': 0,
            'reading_time_minutes': 0
        }
        
        # Calculate average sentence length
        if metrics['sentence_count'] > 0:
            metrics['avg_sentence_length'] = metrics['word_count'] / metrics['sentence_count']
        
        # Estimate reading time (average 200 words per minute)
        if metrics['word_count'] > 0:
            metrics['reading_time_minutes'] = round(metrics['word_count'] / 200, 1)
        
        return metrics
    
    def _classify_page_type(self, url: str) -> str:
        """Classify page type based on URL pattern"""
        
        url_lower = url.lower()
        
        if url_lower.endswith('/') and url_lower.count('/') <= 3:
            return 'homepage'
        elif '/about' in url_lower:
            return 'about'
        elif '/services' in url_lower:
            return 'services'
        elif '/portfolio' in url_lower:
            return 'portfolio'
        elif '/team' in url_lower:
            return 'team'
        elif '/contact' in url_lower:
            return 'contact'
        elif '/blog' in url_lower or '/news' in url_lower:
            return 'blog'
        else:
            return 'other' 