"""
Text cleaning and normalization module for processing scraped content
"""

import re
import html
import unicodedata
from typing import str, List, Dict, Any
from loguru import logger


class TextCleaner:
    """Handles text cleaning, normalization, and noise removal"""
    
    def __init__(self):
        """Initialize the text cleaner with common patterns"""
        
        # Common noise patterns to remove
        self.noise_patterns = [
            # Common website elements
            r'(?i)copyright\s*©?\s*\d{4}.*',
            r'(?i)all rights reserved.*',
            r'(?i)privacy policy.*',
            r'(?i)terms of service.*',
            r'(?i)cookie policy.*',
            r'(?i)subscribe to newsletter.*',
            r'(?i)follow us on.*',
            r'(?i)share this.*',
            r'(?i)like us on facebook.*',
            r'(?i)connect with us.*',
            
            # Navigation elements
            r'(?i)home\s*\|\s*about\s*\|\s*services.*',
            r'(?i)skip to main content.*',
            r'(?i)back to top.*',
            r'(?i)breadcrumb.*',
            
            # Advertisement patterns
            r'(?i)advertisement.*',
            r'(?i)sponsored.*',
            r'(?i)ads by.*',
            
            # Social media patterns
            r'(?i)tweet\s*share\s*like.*',
            r'(?i)facebook\s*twitter\s*linkedin.*',
        ]
        
        # HTML entities mapping
        self.html_entities = {
            '&nbsp;': ' ',
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&apos;': "'",
            '&cent;': '¢',
            '&pound;': '£',
            '&yen;': '¥',
            '&euro;': '€',
            '&copy;': '©',
            '&reg;': '®',
            '&trade;': '™',
            '&mdash;': '—',
            '&ndash;': '–',
            '&hellip;': '…',
            '&lsquo;': ''',
            '&rsquo;': ''',
            '&ldquo;': '"',
            '&rdquo;': '"',
        }
        
        logger.info("TextCleaner initialized")
    
    def clean_html_artifacts(self, text: str) -> str:
        """
        Remove HTML artifacts and entities
        
        Args:
            text: Raw text with potential HTML artifacts
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove any remaining HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Decode HTML entities using html.unescape first
        text = html.unescape(text)
        
        # Handle custom/additional entities
        for entity, replacement in self.html_entities.items():
            text = text.replace(entity, replacement)
        
        # Remove HTML-style comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        
        # Remove CSS and JavaScript remnants
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove broken HTML attributes that might remain
        text = re.sub(r'\s+(class|id|style|href|src|alt|title)=["\'][^"\']*["\']', '', text)
        
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace and line breaks
        
        Args:
            text: Text with irregular whitespace
            
        Returns:
            Text with normalized whitespace
        """
        if not text:
            return ""
        
        # Replace multiple spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Replace multiple newlines with maximum of 2 newlines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove trailing/leading whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Remove empty lines at start and end
        text = text.strip()
        
        # Ensure proper spacing around punctuation
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        text = re.sub(r'([,;:])\s*([A-Za-z])', r'\1 \2', text)
        
        return text
    
    def normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode characters
        
        Args:
            text: Text with potential Unicode issues
            
        Returns:
            Unicode-normalized text
        """
        if not text:
            return ""
        
        # Normalize Unicode to NFC form (composed form)
        text = unicodedata.normalize('NFC', text)
        
        # Handle common Unicode quotes and dashes
        replacements = {
            # Quotes
            '"': '"',  # Left double quotation mark
            '"': '"',  # Right double quotation mark
            ''': "'",  # Left single quotation mark
            ''': "'",  # Right single quotation mark
            '´': "'",  # Acute accent
            '`': "'",  # Grave accent
            
            # Dashes
            '–': '-',  # En dash
            '—': '-',  # Em dash
            '―': '-',  # Horizontal bar
            
            # Spaces
            ' ': ' ',  # Non-breaking space
            ' ': ' ',  # En space
            ' ': ' ',  # Em space
            '	': ' ',  # Tab character
            
            # Other punctuation
            '…': '...',  # Horizontal ellipsis
            '•': '*',    # Bullet
            '◦': '*',    # White bullet
            '‣': '*',    # Triangular bullet
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove or replace other problematic Unicode characters
        # Keep only printable ASCII and common Unicode characters
        text = ''.join(char for char in text if (
            char.isprintable() or char in '\n\t'
        ))
        
        return text
    
    def remove_noise(self, text: str) -> str:
        """
        Remove common website noise and boilerplate text
        
        Args:
            text: Text potentially containing noise
            
        Returns:
            Text with noise removed
        """
        if not text:
            return ""
        
        # Apply noise removal patterns
        for pattern in self.noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove URLs
        text = re.sub(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            '',
            text
        )
        
        # Remove email addresses
        text = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '',
            text
        )
        
        # Remove phone numbers (various formats)
        phone_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # 123-456-7890
            r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',   # (123) 456-7890
            r'\+\d{1,3}\s*\d{3}\s*\d{3}\s*\d{4}',  # +1 123 456 7890
        ]
        
        for pattern in phone_patterns:
            text = re.sub(pattern, '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)  # Multiple periods
        text = re.sub(r'[-]{3,}', '---', text)  # Multiple dashes
        text = re.sub(r'[!]{2,}', '!', text)    # Multiple exclamations
        text = re.sub(r'[?]{2,}', '?', text)    # Multiple question marks
        
        # Remove standalone punctuation lines
        text = re.sub(r'\n[.!?*-]+\n', '\n', text)
        
        return text
    
    def remove_extra_punctuation(self, text: str) -> str:
        """
        Clean up excessive or misplaced punctuation
        
        Args:
            text: Text with potential punctuation issues
            
        Returns:
            Text with cleaned punctuation
        """
        if not text:
            return ""
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([,.!?;:])\s*([^\s])', r'\1 \2', text)  # Add space after punctuation
        
        # Remove duplicate punctuation
        text = re.sub(r'([,.!?;:])\1+', r'\1', text)
        
        # Fix comma and period placement
        text = re.sub(r'\s*,\s*', ', ', text)
        text = re.sub(r'\s*\.\s*', '. ', text)
        
        # Remove punctuation at line starts
        text = re.sub(r'\n[,.!?;:]+\s*', '\n', text)
        
        return text
    
    def clean_text(self, text: str) -> str:
        """
        Apply all cleaning operations to text
        
        Args:
            text: Raw text to clean
            
        Returns:
            Fully cleaned and normalized text
        """
        if not text or not isinstance(text, str):
            return ""
        
        logger.debug(f"Cleaning text of length: {len(text)}")
        
        # Step 1: Remove HTML artifacts
        text = self.clean_html_artifacts(text)
        
        # Step 2: Normalize Unicode
        text = self.normalize_unicode(text)
        
        # Step 3: Remove noise patterns
        text = self.remove_noise(text)
        
        # Step 4: Normalize whitespace
        text = self.normalize_whitespace(text)
        
        # Step 5: Clean punctuation
        text = self.remove_extra_punctuation(text)
        
        # Final trim
        text = text.strip()
        
        logger.debug(f"Cleaned text length: {len(text)}")
        
        return text
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Clean a batch of texts
        
        Args:
            texts: List of raw texts to clean
            
        Returns:
            List of cleaned texts
        """
        logger.info(f"Cleaning batch of {len(texts)} texts")
        
        cleaned_texts = []
        for i, text in enumerate(texts):
            try:
                cleaned_text = self.clean_text(text)
                cleaned_texts.append(cleaned_text)
            except Exception as e:
                logger.error(f"Error cleaning text {i}: {str(e)}")
                cleaned_texts.append("")  # Add empty string for failed cleaning
        
        logger.success(f"Batch cleaning completed")
        return cleaned_texts
    
    def get_cleaning_stats(self, original_text: str, cleaned_text: str) -> Dict[str, Any]:
        """
        Get statistics about the cleaning process
        
        Args:
            original_text: Original text before cleaning
            cleaned_text: Text after cleaning
            
        Returns:
            Dictionary containing cleaning statistics
        """
        return {
            'original_length': len(original_text) if original_text else 0,
            'cleaned_length': len(cleaned_text) if cleaned_text else 0,
            'reduction_ratio': 1 - (len(cleaned_text) / len(original_text)) if original_text else 0,
            'original_lines': original_text.count('\n') + 1 if original_text else 0,
            'cleaned_lines': cleaned_text.count('\n') + 1 if cleaned_text else 0,
            'original_words': len(original_text.split()) if original_text else 0,
            'cleaned_words': len(cleaned_text.split()) if cleaned_text else 0,
        } 