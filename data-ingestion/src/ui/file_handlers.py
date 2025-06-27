"""
File Handlers for Data Ingestion UI

Handles file processing and URL validation for the Streamlit interface.
"""

import re
import io
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import streamlit as st
from urllib.parse import urlparse
import tempfile
import os

try:
    import docx
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    st.warning("python-docx not installed. Word document processing will be limited.")

try:
    import docx2txt
    DOCX2TXT_AVAILABLE = True
except ImportError:
    DOCX2TXT_AVAILABLE = False

import requests
from bs4 import BeautifulSoup


class WordDocumentProcessor:
    """Handles Word document processing for the data ingestion pipeline"""
    
    @staticmethod
    def is_supported_format(file_type: str) -> bool:
        """Check if the file format is supported"""
        supported_types = [
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword',
            'application/vnd.ms-word'
        ]
        return file_type in supported_types
    
    @staticmethod
    def extract_text_from_uploaded_file(uploaded_file) -> Dict[str, Any]:
        """
        Extract text from an uploaded Word document
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Extract text using available library
            text_content = ""
            metadata = {
                'filename': uploaded_file.name,
                'size': uploaded_file.size,
                'type': uploaded_file.type,
                'extraction_method': None,
                'paragraphs': [],
                'word_count': 0,
                'char_count': 0
            }
            
            if DOCX_AVAILABLE:
                try:
                    doc = Document(tmp_path)
                    paragraphs = []
                    
                    for paragraph in doc.paragraphs:
                        if paragraph.text.strip():
                            paragraphs.append(paragraph.text.strip())
                    
                    text_content = '\n\n'.join(paragraphs)
                    metadata['extraction_method'] = 'python-docx'
                    metadata['paragraphs'] = paragraphs
                    
                except Exception as e:
                    st.warning(f"python-docx extraction failed: {e}")
                    text_content = WordDocumentProcessor._fallback_extraction(tmp_path)
                    metadata['extraction_method'] = 'fallback'
            
            elif DOCX2TXT_AVAILABLE:
                try:
                    text_content = docx2txt.process(tmp_path)
                    metadata['extraction_method'] = 'docx2txt'
                    metadata['paragraphs'] = [p.strip() for p in text_content.split('\n') if p.strip()]
                except Exception as e:
                    st.warning(f"docx2txt extraction failed: {e}")
                    text_content = WordDocumentProcessor._fallback_extraction(tmp_path)
                    metadata['extraction_method'] = 'fallback'
            
            else:
                text_content = WordDocumentProcessor._fallback_extraction(tmp_path)
                metadata['extraction_method'] = 'fallback'
            
            # Update metadata
            metadata['word_count'] = len(text_content.split())
            metadata['char_count'] = len(text_content)
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            return {
                'success': True,
                'text': text_content,
                'metadata': metadata,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'text': '',
                'metadata': metadata,
                'error': str(e)
            }
    
    @staticmethod
    def _fallback_extraction(file_path: str) -> str:
        """
        Fallback extraction method when specialized libraries fail
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text (may be limited)
        """
        try:
            # Try to read as plain text (will only work for some formats)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                # Basic cleanup for binary content
                content = re.sub(r'[^\x20-\x7E\n\r\t]', '', content)
                return content
        except Exception:
            return "Failed to extract text from document. Please ensure the file is a valid Word document."
    
    @staticmethod
    def process_multiple_files(uploaded_files: List) -> List[Dict[str, Any]]:
        """
        Process multiple uploaded Word documents
        
        Args:
            uploaded_files: List of Streamlit uploaded file objects
            
        Returns:
            List of processing results
        """
        results = []
        
        for file in uploaded_files:
            if WordDocumentProcessor.is_supported_format(file.type):
                result = WordDocumentProcessor.extract_text_from_uploaded_file(file)
                results.append(result)
            else:
                results.append({
                    'success': False,
                    'text': '',
                    'metadata': {'filename': file.name, 'type': file.type},
                    'error': f'Unsupported file type: {file.type}'
                })
        
        return results


class URLValidator:
    """Handles URL validation and preprocessing"""
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """
        Validate if a URL is properly formatted
        
        Args:
            url: URL string to validate
            
        Returns:
            True if URL is valid, False otherwise
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    @staticmethod
    def normalize_url(url: str) -> str:
        """
        Normalize a URL by adding missing schemes, etc.
        
        Args:
            url: URL string to normalize
            
        Returns:
            Normalized URL string
        """
        url = url.strip()
        
        # Add http:// if no scheme is present
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        return url
    
    @staticmethod
    def validate_and_normalize_urls(urls: List[str]) -> Dict[str, List[str]]:
        """
        Validate and normalize a list of URLs
        
        Args:
            urls: List of URL strings
            
        Returns:
            Dictionary with 'valid' and 'invalid' URL lists
        """
        valid_urls = []
        invalid_urls = []
        
        for url in urls:
            normalized_url = URLValidator.normalize_url(url)
            
            if URLValidator.is_valid_url(normalized_url):
                valid_urls.append(normalized_url)
            else:
                invalid_urls.append(url)
        
        return {
            'valid': valid_urls,
            'invalid': invalid_urls
        }
    
    @staticmethod
    def check_url_accessibility(url: str, timeout: int = 10) -> Dict[str, Any]:
        """
        Check if a URL is accessible
        
        Args:
            url: URL to check
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary with accessibility information
        """
        try:
            response = requests.head(url, timeout=timeout, allow_redirects=True)
            
            return {
                'accessible': response.status_code < 400,
                'status_code': response.status_code,
                'content_type': response.headers.get('content-type', 'unknown'),
                'final_url': response.url,
                'error': None
            }
            
        except requests.RequestException as e:
            return {
                'accessible': False,
                'status_code': None,
                'content_type': None,
                'final_url': None,
                'error': str(e)
            }
    
    @staticmethod
    def batch_check_urls(urls: List[str], timeout: int = 10) -> List[Dict[str, Any]]:
        """
        Check accessibility of multiple URLs
        
        Args:
            urls: List of URLs to check
            timeout: Request timeout in seconds
            
        Returns:
            List of accessibility results
        """
        results = []
        
        for url in urls:
            result = URLValidator.check_url_accessibility(url, timeout)
            result['url'] = url
            results.append(result)
        
        return results


class FilePreviewGenerator:
    """Generate previews for uploaded files and URLs"""
    
    @staticmethod
    def generate_text_preview(text: str, max_length: int = 500) -> str:
        """
        Generate a text preview of specified length
        
        Args:
            text: Full text content
            max_length: Maximum length of preview
            
        Returns:
            Truncated text preview
        """
        if len(text) <= max_length:
            return text
        
        # Try to break at sentence boundary
        truncated = text[:max_length]
        last_sentence_end = max(
            truncated.rfind('.'),
            truncated.rfind('!'),
            truncated.rfind('?')
        )
        
        if last_sentence_end > max_length * 0.7:  # If we found a good break point
            return truncated[:last_sentence_end + 1] + "..."
        else:
            # Break at word boundary
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.8:
                return truncated[:last_space] + "..."
            else:
                return truncated + "..."
    
    @staticmethod
    def generate_file_summary(file_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of processed files
        
        Args:
            file_results: List of file processing results
            
        Returns:
            Summary dictionary
        """
        total_files = len(file_results)
        successful_files = sum(1 for result in file_results if result['success'])
        failed_files = total_files - successful_files
        
        total_words = sum(
            result['metadata'].get('word_count', 0) 
            for result in file_results 
            if result['success']
        )
        
        total_chars = sum(
            result['metadata'].get('char_count', 0) 
            for result in file_results 
            if result['success']
        )
        
        return {
            'total_files': total_files,
            'successful_files': successful_files,
            'failed_files': failed_files,
            'success_rate': (successful_files / total_files * 100) if total_files > 0 else 0,
            'total_words': total_words,
            'total_characters': total_chars,
            'average_words_per_file': total_words / successful_files if successful_files > 0 else 0
        } 