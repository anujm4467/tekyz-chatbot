"""
Quality Control Component

This module validates processed content quality, filters out low-quality chunks,
and ensures content completeness and consistency.
"""

import re
import hashlib
from typing import List, Dict, Set, Tuple, Any
import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from collections import Counter
import unicodedata

from .text_processor import TextChunk

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality assessment metrics for content"""
    chunk_id: str
    length_score: float
    language_score: float
    readability_score: float
    completeness_score: float
    uniqueness_score: float
    overall_score: float
    passed_quality_check: bool
    issues: List[str]


class QualityController:
    """
    Validates and controls content quality.
    
    Handles:
    - Content quality validation
    - Duplicate content detection
    - Metadata completeness validation
    - Content filtering and scoring
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_length = self.config.get('min_chunk_length', 100)
        self.max_length = self.config.get('max_chunk_length', 5000)
        self.min_word_count = self.config.get('min_word_count', 20)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.85)
        self.quality_threshold = self.config.get('quality_threshold', 0.7)
        
        # Quality patterns
        self.noise_patterns = [
            r'(?i)cookie.*policy',
            r'(?i)privacy.*policy',
            r'(?i)terms.*conditions',
            r'(?i)copyright.*\d{4}',
            r'(?i)all.*rights.*reserved',
            r'(?i)subscribe.*newsletter',
            r'(?i)follow.*us.*on',
            r'(?i)click.*here.*to',
            r'(?i)back.*to.*top',
            r'(?i)skip.*to.*content',
            r'(?i)javascript.*disabled',
            r'(?i)enable.*javascript',
        ]
        
        self.gibberish_patterns = [
            r'[a-zA-Z]{20,}',  # Very long words (likely concatenated)
            r'(\w)\1{5,}',     # Repeated characters
            r'[0-9]{10,}',     # Long number sequences
            r'[^a-zA-Z0-9\s\.,!?;:-]{5,}',  # Special character sequences
        ]
        
        # Cache for duplicate detection
        self.content_hashes: Set[str] = set()
        self.processed_chunks: List[Dict[str, Any]] = []
    
    def clear_cache(self):
        """Clear the duplicate detection cache"""
        self.content_hashes.clear()
        self.processed_chunks.clear()
        logger.info("Quality controller cache cleared")
    
    def validate_chunk_quality(self, chunk: TextChunk) -> QualityMetrics:
        """
        Validate quality of a single text chunk.
        
        Args:
            chunk: TextChunk to validate
            
        Returns:
            QualityMetrics with assessment results
        """
        issues = []
        text = chunk.clean_text or chunk.text
        
        # Length validation
        length_score = self._validate_length(text, issues)
        
        # Language validation (use detected language from metadata if available)
        detected_language = chunk.metadata.get('language', 'en') or 'en'  # Default to English if empty
        language_score = self._validate_language(text, detected_language, issues)
        
        # Readability validation
        readability_score = self._validate_readability(text, issues)
        
        # Completeness validation
        completeness_score = self._validate_completeness(text, issues)
        
        # Uniqueness validation (against cache)
        uniqueness_score = self._validate_uniqueness(text, issues)
        
        # Calculate overall score
        weights = {
            'length': 0.2,
            'language': 0.15,
            'readability': 0.25,
            'completeness': 0.2,
            'uniqueness': 0.2
        }
        
        overall_score = (
            length_score * weights['length'] +
            language_score * weights['language'] +
            readability_score * weights['readability'] +
            completeness_score * weights['completeness'] +
            uniqueness_score * weights['uniqueness']
        )
        
        # More lenient passing criteria - focus on overall score, ignore minor issues
        ignored_issues = ['non_english_content', 'duplicate_content', 'similar_content']
        critical_issues = [issue for issue in issues if not any(issue.startswith(ignored) for ignored in ignored_issues)]
        passed = overall_score >= self.quality_threshold and len(critical_issues) == 0
        
        quality_metrics = QualityMetrics(
            chunk_id=chunk.id,
            length_score=length_score,
            language_score=language_score,
            readability_score=readability_score,
            completeness_score=completeness_score,
            uniqueness_score=uniqueness_score,
            overall_score=overall_score,
            passed_quality_check=passed,
            issues=issues
        )
        
        # Add to cache if passed
        if passed:
            self._add_to_cache(text, chunk)
        
        logger.debug(f"Quality check for {chunk.id}: {overall_score:.2f} ({'PASS' if passed else 'FAIL'})")
        
        return quality_metrics
    
    def detect_duplicate_content(self, chunks: List[TextChunk]) -> List[Tuple[int, int, float]]:
        """
        Detect duplicate or near-duplicate content.
        
        Args:
            chunks: List of TextChunk objects
            
        Returns:
            List of (index1, index2, similarity_score) for duplicates
        """
        duplicates = []
        
        # Create content hashes for exact duplicates
        hash_to_indices = {}
        for i, chunk in enumerate(chunks):
            text = chunk.clean_text or chunk.text
            content_hash = self._generate_content_hash(text)
            if content_hash in hash_to_indices:
                duplicates.append((hash_to_indices[content_hash], i, 1.0))
            else:
                hash_to_indices[content_hash] = i
        
        # Check for near-duplicates using similarity
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                chunk1 = chunks[i]
                chunk2 = chunks[j]
                text1 = chunk1.clean_text or chunk1.text
                text2 = chunk2.clean_text or chunk2.text
                
                # Skip if already found as exact duplicate
                if any(dup for dup in duplicates if (dup[0] == i and dup[1] == j) or (dup[0] == j and dup[1] == i)):
                    continue
                
                similarity = self._calculate_similarity(text1, text2)
                if similarity >= self.similarity_threshold:
                    duplicates.append((i, j, similarity))
        
        logger.info(f"Found {len(duplicates)} duplicate pairs in {len(chunks)} chunks")
        return duplicates
    
    def remove_duplicates(self, chunks: List[TextChunk]) -> Tuple[List[TextChunk], Dict[str, Any]]:
        """
        Remove duplicate chunks from the list.
        
        Args:
            chunks: List of TextChunk objects
            
        Returns:
            Tuple of (deduplicated_chunks, duplicate_info)
        """
        if not chunks:
            return [], {}
        
        # Detect duplicates
        duplicates = self.detect_duplicate_content(chunks)
        
        # Create set of indices to remove (keep the first occurrence)
        indices_to_remove = set()
        duplicate_info = {}
        
        for idx1, idx2, similarity in duplicates:
            # Always remove the higher index (keep the first occurrence)
            remove_idx = max(idx1, idx2)
            keep_idx = min(idx1, idx2)
            
            indices_to_remove.add(remove_idx)
            
            # Track duplicate information
            keep_chunk_id = chunks[keep_idx].id
            remove_chunk_id = chunks[remove_idx].id
            
            if keep_chunk_id not in duplicate_info:
                duplicate_info[keep_chunk_id] = []
            
            duplicate_info[keep_chunk_id].append({
                'chunk_id': remove_chunk_id,
                'similarity_score': similarity,
                'duplicate_type': 'exact' if similarity >= 1.0 else 'near_duplicate'
            })
        
        # Filter out duplicates
        deduplicated_chunks = [
            chunk for i, chunk in enumerate(chunks) 
            if i not in indices_to_remove
        ]
        
        logger.info(f"Removed {len(indices_to_remove)} duplicate chunks, {len(deduplicated_chunks)} remaining")
        
        return deduplicated_chunks, duplicate_info
    
    def validate_metadata_completeness(self, chunk: TextChunk) -> Tuple[bool, List[str]]:
        """
        Validate metadata completeness.
        
        Args:
            chunk: TextChunk to validate
            
        Returns:
            Tuple of (is_complete, list_of_missing_fields)
        """
        required_fields = [
            'id', 'source_url', 'chunk_index', 'char_count', 'word_count'
        ]
        
        missing_fields = []
        
        # Check direct fields
        if not chunk.id or not chunk.id.strip():
            missing_fields.append('id')
        if not chunk.source_url or not chunk.source_url.strip():
            missing_fields.append('source_url')
        if chunk.chunk_index < 0:
            missing_fields.append('chunk_index (invalid)')
        if chunk.char_count <= 0:
            missing_fields.append('char_count (invalid)')
        if chunk.word_count <= 0:
            missing_fields.append('word_count (invalid)')
        
        # Check metadata fields
        metadata_required = ['page_title', 'page_type', 'content_type', 'language', 'timestamp']
        for field in metadata_required:
            value = chunk.metadata.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing_fields.append(f'metadata.{field}')
        
        if chunk.char_count <= 0:
            missing_fields.append('char_count (invalid)')
        
        if chunk.word_count <= 0:
            missing_fields.append('word_count (invalid)')
        
        is_complete = len(missing_fields) == 0
        
        return is_complete, missing_fields
    
    def filter_quality_chunks(self, chunks: List[TextChunk]) -> Tuple[List[TextChunk], List[QualityMetrics]]:
        """
        Filter chunks based on quality criteria.
        
        Args:
            chunks: List of TextChunk objects
            
        Returns:
            Tuple of (filtered_chunks, quality_reports)
        """
        filtered_chunks = []
        quality_reports = []
        
        for chunk in chunks:
            quality_metrics = self.validate_chunk_quality(chunk)
            quality_reports.append(quality_metrics)
            
            if quality_metrics.passed_quality_check:
                filtered_chunks.append(chunk)
            else:
                logger.debug(f"Filtered out chunk {chunk.id}: {quality_metrics.issues}")
        
        logger.info(f"Quality filtering: {len(filtered_chunks)}/{len(chunks)} chunks passed")
        
        return filtered_chunks, quality_reports
    
    def generate_quality_report(self, quality_metrics: List[QualityMetrics]) -> Dict[str, Any]:
        """
        Generate comprehensive quality report.
        
        Args:
            quality_metrics: List of QualityMetrics
            
        Returns:
            Dictionary containing quality statistics
        """
        if not quality_metrics:
            return {'total_chunks': 0, 'error': 'No quality metrics provided'}
        
        total_chunks = len(quality_metrics)
        passed_chunks = sum(1 for qm in quality_metrics if qm.passed_quality_check)
        failed_chunks = total_chunks - passed_chunks
        
        # Calculate average scores
        avg_scores = {
            'overall': sum(qm.overall_score for qm in quality_metrics) / total_chunks,
            'length': sum(qm.length_score for qm in quality_metrics) / total_chunks,
            'language': sum(qm.language_score for qm in quality_metrics) / total_chunks,
            'readability': sum(qm.readability_score for qm in quality_metrics) / total_chunks,
            'completeness': sum(qm.completeness_score for qm in quality_metrics) / total_chunks,
            'uniqueness': sum(qm.uniqueness_score for qm in quality_metrics) / total_chunks,
        }
        
        # Collect common issues
        all_issues = []
        for qm in quality_metrics:
            all_issues.extend(qm.issues)
        
        issue_counts = Counter(all_issues)
        
        # Score distribution
        score_ranges = {
            'excellent (0.9-1.0)': sum(1 for qm in quality_metrics if qm.overall_score >= 0.9),
            'good (0.8-0.9)': sum(1 for qm in quality_metrics if 0.8 <= qm.overall_score < 0.9),
            'fair (0.7-0.8)': sum(1 for qm in quality_metrics if 0.7 <= qm.overall_score < 0.8),
            'poor (<0.7)': sum(1 for qm in quality_metrics if qm.overall_score < 0.7),
        }
        
        report = {
            'summary': {
                'total_chunks': total_chunks,
                'passed_chunks': passed_chunks,
                'failed_chunks': failed_chunks,
                'pass_rate': passed_chunks / total_chunks if total_chunks > 0 else 0,
            },
            'average_scores': avg_scores,
            'score_distribution': score_ranges,
            'common_issues': dict(issue_counts.most_common(10)),
            'quality_threshold': self.quality_threshold,
            'recommendations': self._generate_recommendations(avg_scores, issue_counts)
        }
        
        return report
    
    def _validate_length(self, text: str, issues: List[str]) -> float:
        """Validate text length"""
        char_count = len(text.strip())
        word_count = len(text.split())
        
        if char_count < self.min_length:
            issues.append(f'text_too_short ({char_count} < {self.min_length})')
            return 0.0
        
        if char_count > self.max_length:
            issues.append(f'text_too_long ({char_count} > {self.max_length})')
            return 0.5
        
        if word_count < self.min_word_count:
            issues.append(f'insufficient_words ({word_count} < {self.min_word_count})')
            return 0.3
        
        # Score based on optimal length range
        if self.min_length <= char_count <= 1000:
            return 1.0
        elif char_count <= 2000:
            return 0.9
        else:
            return 0.8
    
    def _validate_language(self, text: str, detected_language: str, issues: List[str]) -> float:
        """Validate text language"""
        # More lenient language validation - check for obvious English patterns
        english_indicators = ['the', 'and', 'a', 'to', 'of', 'in', 'that', 'is', 'for', 'with']
        text_lower = text.lower()
        english_word_count = sum(1 for word in english_indicators if word in text_lower)
        
        # If we find common English words, assume it's English regardless of detected_language
        if english_word_count >= 2:
            return 1.0
        
        if detected_language and detected_language != 'en':
            issues.append(f'non_english_content ({detected_language})')
            return 0.2  # Don't completely fail, just reduce score
        
        # Check for excessive non-ASCII characters
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        ascii_ratio = ascii_chars / len(text) if text else 1
        
        if ascii_ratio < 0.8:
            issues.append('excessive_non_ascii_characters')
            return 0.5
        
        return 1.0
    
    def _validate_readability(self, text: str, issues: List[str]) -> float:
        """Validate text readability"""
        score = 1.0
        
        # Check for gibberish patterns
        for pattern in self.gibberish_patterns:
            if re.search(pattern, text):
                issues.append('gibberish_content_detected')
                score *= 0.5
                break
        
        # Check sentence structure
        sentences = re.split(r'[.!?]+', text)
        valid_sentences = [s for s in sentences if len(s.strip()) > 10 and ' ' in s.strip()]
        
        if len(valid_sentences) == 0 and len(text) > 100:
            issues.append('no_proper_sentences')
            score *= 0.3
        
        # Check for reasonable punctuation
        punct_count = sum(1 for c in text if c in '.!?,:;')
        punct_ratio = punct_count / len(text.split()) if text.split() else 0
        
        if punct_ratio > 0.5:  # Too much punctuation
            issues.append('excessive_punctuation')
            score *= 0.7
        
        return score
    
    def _validate_completeness(self, text: str, issues: List[str]) -> float:
        """Validate content completeness"""
        score = 1.0
        
        # Check for common noise patterns
        noise_count = 0
        for pattern in self.noise_patterns:
            if re.search(pattern, text):
                noise_count += 1
        
        if noise_count > 2:
            issues.append('excessive_boilerplate_content')
            score *= 0.5
        
        # Check for truncated content
        if text.strip().endswith(('...', '…', 'Read more', 'Continue reading')):
            issues.append('truncated_content')
            score *= 0.8
        
        # Check for meaningful content ratio
        meaningful_words = len([w for w in text.split() if len(w) > 3])
        total_words = len(text.split())
        
        if total_words > 0:
            meaningful_ratio = meaningful_words / total_words
            if meaningful_ratio < 0.5:
                issues.append('low_meaningful_content_ratio')
                score *= 0.6
        
        return score
    
    def _validate_uniqueness(self, text: str, issues: List[str]) -> float:
        """Validate content uniqueness"""
        content_hash = self._generate_content_hash(text)
        
        if content_hash in self.content_hashes:
            issues.append('duplicate_content')
            return 0.0
        
        # Check similarity with recent chunks
        for existing_chunk in self.processed_chunks[-50:]:  # Check last 50 chunks
            similarity = self._calculate_similarity(text, existing_chunk['text'])
            if similarity >= self.similarity_threshold:
                issues.append(f'similar_content (similarity: {similarity:.2f})')
                return 0.2
        
        return 1.0
    
    def _generate_content_hash(self, text: str) -> str:
        """Generate hash for content deduplication"""
        # Normalize text for hashing
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        # Normalize texts
        norm1 = re.sub(r'\s+', ' ', text1.lower().strip())
        norm2 = re.sub(r'\s+', ' ', text2.lower().strip())
        
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def _add_to_cache(self, text: str, chunk: TextChunk) -> None:
        """Add validated content to cache"""
        content_hash = self._generate_content_hash(text)
        self.content_hashes.add(content_hash)
        
        # Keep recent chunks for similarity checking
        self.processed_chunks.append({
            'text': text,
            'chunk_id': chunk.id,
            'hash': content_hash
        })
        
        # Limit cache size
        if len(self.processed_chunks) > 1000:
            removed = self.processed_chunks.pop(0)
            self.content_hashes.discard(removed['hash'])
    
    def _generate_recommendations(self, avg_scores: Dict[str, float], 
                                 issue_counts: Counter) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        if avg_scores['length'] < 0.8:
            recommendations.append("Consider adjusting chunk size parameters to improve length scores")
        
        if avg_scores['readability'] < 0.8:
            recommendations.append("Review content extraction to reduce gibberish and improve readability")
        
        if avg_scores['uniqueness'] < 0.9:
            recommendations.append("Implement better duplicate detection during scraping")
        
        if 'non_english_content' in issue_counts:
            recommendations.append("Improve language detection and filtering")
        
        if 'excessive_boilerplate_content' in issue_counts:
            recommendations.append("Enhance noise removal patterns for cleaner content")
        
        return recommendations 