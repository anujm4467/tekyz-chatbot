"""
Metadata Extraction Module

This module extracts metadata from text chunks including keywords,
language detection, content classification, and semantic features.
"""

import re
from typing import List, Dict, Any, Optional, Set, Tuple
import logging
from collections import Counter
from dataclasses import dataclass
import string
from datetime import datetime

# Optional imports with fallbacks
try:
    from langdetect import detect, detect_langs
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("langdetect not available, language detection disabled")

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    from nltk.tree import Tree
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available, advanced NLP features disabled")

logger = logging.getLogger(__name__)


@dataclass
class ExtractedMetadata:
    """Container for extracted metadata."""
    keywords: List[str]
    language: str
    language_confidence: float
    content_type: str
    named_entities: List[Dict[str, str]]
    pos_tags: List[Tuple[str, str]]
    readability_score: float
    sentiment_indicators: Dict[str, int]
    technical_terms: List[str]
    business_terms: List[str]
    extraction_timestamp: str


class MetadataExtractor:
    """
    Advanced metadata extraction from text content.
    
    Features:
    - Keyword extraction with TF-IDF-like scoring
    - Language detection with confidence
    - Content type classification
    - Named entity recognition
    - Technical and business term identification
    - Readability assessment
    - Sentiment indicators
    """
    
    def __init__(self):
        """Initialize the metadata extractor."""
        self._initialize_resources()
        self._load_domain_vocabularies()
        logger.info("MetadataExtractor initialized")
    
    def _initialize_resources(self):
        """Initialize NLP resources."""
        self.stopwords_en = set()
        
        if NLTK_AVAILABLE:
            try:
                self.stopwords_en = set(stopwords.words('english'))
            except LookupError:
                nltk.download('stopwords', quiet=True)
                self.stopwords_en = set(stopwords.words('english'))
            
            # Download other required NLTK data
            for resource in ['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']:
                try:
                    nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else 
                                 f'taggers/{resource}' if 'tagger' in resource else
                                 f'chunkers/{resource}' if 'chunker' in resource else
                                 f'corpora/{resource}')
                except LookupError:
                    nltk.download(resource, quiet=True)
        
        # Compile regex patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for text analysis."""
        # Technical terms patterns
        self.tech_patterns = [
            re.compile(r'\b(?:API|SDK|UI|UX|AI|ML|SaaS|PaaS|IaaS|DevOps|CI/CD)\b', re.IGNORECASE),
            re.compile(r'\b(?:JavaScript|Python|React|Node\.js|Docker|Kubernetes|AWS|Azure|GCP)\b', re.IGNORECASE),
            re.compile(r'\b(?:database|server|cloud|microservices|architecture|framework|library)\b', re.IGNORECASE),
            re.compile(r'\b(?:frontend|backend|fullstack|mobile|web|application|software|platform)\b', re.IGNORECASE)
        ]
        
        # Business terms patterns
        self.business_patterns = [
            re.compile(r'\b(?:startup|entrepreneur|business|company|enterprise|corporation)\b', re.IGNORECASE),
            re.compile(r'\b(?:revenue|profit|ROI|KPI|metrics|analytics|growth|scale|scaling)\b', re.IGNORECASE),
            re.compile(r'\b(?:market|customer|client|user|stakeholder|investor|funding)\b', re.IGNORECASE),
            re.compile(r'\b(?:strategy|planning|execution|management|leadership|team|collaboration)\b', re.IGNORECASE),
            re.compile(r'\b(?:product|service|solution|offering|portfolio|brand|marketing)\b', re.IGNORECASE)
        ]
        
        # Content type indicators
        self.content_type_patterns = {
            'service_description': re.compile(r'\b(?:we offer|our services|we provide|we specialize|we help)\b', re.IGNORECASE),
            'team_bio': re.compile(r'\b(?:years of experience|background in|expertise in|specializes in|founded|CEO|CTO|developer)\b', re.IGNORECASE),
            'case_study': re.compile(r'\b(?:case study|project|client|challenge|solution|results|outcome)\b', re.IGNORECASE),
            'blog_post': re.compile(r'\b(?:in this post|today we|recently|update|announcement|insights)\b', re.IGNORECASE),
            'technical_doc': re.compile(r'\b(?:documentation|guide|tutorial|how to|step by step|implementation)\b', re.IGNORECASE),
            'about_page': re.compile(r'\b(?:about us|our story|our mission|our vision|who we are|founded in)\b', re.IGNORECASE)
        }
        
        # Sentiment indicators
        self.positive_words = {
            'excellent', 'outstanding', 'amazing', 'great', 'fantastic', 'wonderful',
            'successful', 'innovative', 'cutting-edge', 'leading', 'expert', 'professional',
            'reliable', 'trusted', 'proven', 'effective', 'efficient', 'optimized',
            'scalable', 'robust', 'secure', 'fast', 'easy', 'simple', 'powerful'
        }
        
        self.negative_words = {
            'difficult', 'challenging', 'complex', 'complicated', 'hard', 'tough',
            'problem', 'issue', 'bug', 'error', 'fail', 'failure', 'broken',
            'slow', 'expensive', 'costly', 'limited', 'restricted', 'outdated'
        }
    
    def _load_domain_vocabularies(self):
        """Load domain-specific vocabularies."""
        # Technology vocabulary
        self.tech_vocabulary = {
            'programming_languages': {
                'python', 'javascript', 'java', 'c++', 'c#', 'ruby', 'php', 'go', 'rust', 'swift',
                'kotlin', 'typescript', 'scala', 'r', 'matlab', 'sql'
            },
            'frameworks': {
                'react', 'angular', 'vue', 'django', 'flask', 'express', 'spring', 'laravel',
                'rails', 'nextjs', 'nuxt', 'gatsby', 'svelte', 'ember'
            },
            'tools': {
                'docker', 'kubernetes', 'jenkins', 'git', 'github', 'gitlab', 'bitbucket',
                'jira', 'confluence', 'slack', 'teams', 'zoom', 'figma', 'sketch'
            },
            'cloud_platforms': {
                'aws', 'azure', 'gcp', 'google cloud', 'amazon web services', 'microsoft azure',
                'heroku', 'digitalocean', 'linode', 'vultr'
            }
        }
        
        # Business vocabulary
        self.business_vocabulary = {
            'roles': {
                'ceo', 'cto', 'cfo', 'coo', 'founder', 'co-founder', 'director', 'manager',
                'lead', 'senior', 'junior', 'developer', 'engineer', 'designer', 'analyst'
            },
            'industries': {
                'fintech', 'healthtech', 'edtech', 'proptech', 'insurtech', 'regtech',
                'martech', 'adtech', 'cleantech', 'biotech', 'nanotech'
            },
            'business_models': {
                'saas', 'paas', 'iaas', 'b2b', 'b2c', 'b2b2c', 'marketplace', 'platform',
                'subscription', 'freemium', 'enterprise', 'startup', 'scaleup'
            }
        }
    
    def extract_metadata(self, text: str, context: Optional[Dict[str, Any]] = None) -> ExtractedMetadata:
        """
        Extract comprehensive metadata from text.
        
        Args:
            text: Text to analyze
            context: Optional context information (URL, page type, etc.)
            
        Returns:
            Extracted metadata
        """
        context = context or {}
        
        # Extract keywords
        keywords = self.extract_keywords(text)
        
        # Detect language
        language, language_confidence = self.detect_language(text)
        
        # Classify content type
        content_type = self.classify_content_type(text, context)
        
        # Extract named entities
        named_entities = self.extract_named_entities(text)
        
        # Get POS tags (sample for performance)
        pos_tags = self.get_pos_tags(text[:500])  # First 500 chars for performance
        
        # Calculate readability
        readability_score = self.calculate_readability(text)
        
        # Analyze sentiment indicators
        sentiment_indicators = self.analyze_sentiment_indicators(text)
        
        # Extract domain-specific terms
        technical_terms = self.extract_technical_terms(text)
        business_terms = self.extract_business_terms(text)
        
        return ExtractedMetadata(
            keywords=keywords,
            language=language,
            language_confidence=language_confidence,
            content_type=content_type,
            named_entities=named_entities,
            pos_tags=pos_tags,
            readability_score=readability_score,
            sentiment_indicators=sentiment_indicators,
            technical_terms=technical_terms,
            business_terms=business_terms,
            extraction_timestamp=datetime.now().isoformat()
        )
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords using frequency and importance scoring."""
        if not text:
            return []
        
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove stopwords
        words = [word for word in words if word not in self.stopwords_en]
        
        # Count frequencies
        word_freq = Counter(words)
        
        # Score words based on frequency and length
        scored_words = []
        for word, freq in word_freq.items():
            # Boost score for longer words and domain terms
            score = freq * (len(word) / 5.0)  # Length bonus
            
            # Boost for technical/business terms
            if self._is_domain_term(word):
                score *= 1.5
            
            # Boost for capitalized words in original text
            if word.title() in text or word.upper() in text:
                score *= 1.2
            
            scored_words.append((word, score))
        
        # Sort by score and return top keywords
        scored_words.sort(key=lambda x: x[1], reverse=True)
        return [word for word, score in scored_words[:max_keywords]]
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect text language with confidence score."""
        if not LANGDETECT_AVAILABLE or not text:
            return 'en', 0.5  # Default to English
        
        try:
            # Get language probabilities
            lang_probs = detect_langs(text)
            
            if lang_probs:
                top_lang = lang_probs[0]
                return top_lang.lang, top_lang.prob
            else:
                return 'en', 0.5
                
        except Exception as e:
            logger.debug(f"Language detection failed: {str(e)}")
            return 'en', 0.5
    
    def classify_content_type(self, text: str, context: Dict[str, Any]) -> str:
        """Classify the type of content."""
        # Check URL-based classification first
        url = context.get('url', '').lower()
        
        if '/about' in url:
            return 'about_page'
        elif '/team' in url or '/people' in url:
            return 'team_bio'
        elif '/services' in url or '/solutions' in url:
            return 'service_description'
        elif '/blog' in url or '/news' in url or '/insights' in url:
            return 'blog_post'
        elif '/case-study' in url or '/portfolio' in url:
            return 'case_study'
        elif '/docs' in url or '/documentation' in url:
            return 'technical_doc'
        
        # Content-based classification
        text_lower = text.lower()
        
        for content_type, pattern in self.content_type_patterns.items():
            if pattern.search(text_lower):
                return content_type
        
        # Default classification based on content characteristics
        if len(text) > 2000 and 'we' in text_lower[:200]:
            return 'service_description'
        elif 'experience' in text_lower and len(text) < 1000:
            return 'team_bio'
        else:
            return 'general_content'
    
    def extract_named_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities from text."""
        if not NLTK_AVAILABLE or not text:
            return []
        
        try:
            # Tokenize and tag
            tokens = word_tokenize(text[:1000])  # Limit for performance
            pos_tags = pos_tag(tokens)
            
            # Named entity chunking
            chunks = ne_chunk(pos_tags)
            
            entities = []
            for chunk in chunks:
                if isinstance(chunk, Tree):
                    entity_name = ' '.join([token for token, pos in chunk.leaves()])
                    entity_type = chunk.label()
                    entities.append({
                        'text': entity_name,
                        'type': entity_type
                    })
            
            return entities
            
        except Exception as e:
            logger.debug(f"Named entity extraction failed: {str(e)}")
            return []
    
    def get_pos_tags(self, text: str) -> List[Tuple[str, str]]:
        """Get part-of-speech tags for text sample."""
        if not NLTK_AVAILABLE or not text:
            return []
        
        try:
            tokens = word_tokenize(text)
            return pos_tag(tokens)
        except Exception as e:
            logger.debug(f"POS tagging failed: {str(e)}")
            return []
    
    def calculate_readability(self, text: str) -> float:
        """Calculate readability score (simplified Flesch Reading Ease)."""
        if not text:
            return 0.0
        
        # Count sentences, words, and syllables
        sentences = len(re.findall(r'[.!?]+', text))
        words = len(text.split())
        
        if sentences == 0 or words == 0:
            return 0.0
        
        # Estimate syllables (simplified)
        syllables = sum(self._count_syllables(word) for word in text.split())
        
        # Flesch Reading Ease formula (simplified)
        if sentences > 0 and words > 0:
            avg_sentence_length = words / sentences
            avg_syllables_per_word = syllables / words
            
            score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            return max(0.0, min(100.0, score))
        
        return 50.0  # Default middle score
    
    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count for a word."""
        word = word.lower().strip(string.punctuation)
        if not word:
            return 0
        
        # Simple syllable counting heuristic
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def analyze_sentiment_indicators(self, text: str) -> Dict[str, int]:
        """Analyze sentiment indicators in text."""
        text_lower = text.lower()
        words = set(re.findall(r'\b[a-zA-Z]+\b', text_lower))
        
        positive_count = len(words.intersection(self.positive_words))
        negative_count = len(words.intersection(self.negative_words))
        
        return {
            'positive_indicators': positive_count,
            'negative_indicators': negative_count,
            'sentiment_ratio': positive_count / max(negative_count, 1)
        }
    
    def extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms from text."""
        technical_terms = set()
        text_lower = text.lower()
        
        # Pattern-based extraction
        for pattern in self.tech_patterns:
            matches = pattern.findall(text)
            technical_terms.update(match.lower() for match in matches)
        
        # Vocabulary-based extraction
        for category, terms in self.tech_vocabulary.items():
            for term in terms:
                if term in text_lower:
                    technical_terms.add(term)
        
        return list(technical_terms)
    
    def extract_business_terms(self, text: str) -> List[str]:
        """Extract business terms from text."""
        business_terms = set()
        text_lower = text.lower()
        
        # Pattern-based extraction
        for pattern in self.business_patterns:
            matches = pattern.findall(text)
            business_terms.update(match.lower() for match in matches)
        
        # Vocabulary-based extraction
        for category, terms in self.business_vocabulary.items():
            for term in terms:
                if term in text_lower:
                    business_terms.add(term)
        
        return list(business_terms)
    
    def _is_domain_term(self, word: str) -> bool:
        """Check if word is a domain-specific term."""
        word_lower = word.lower()
        
        # Check technical vocabulary
        for category, terms in self.tech_vocabulary.items():
            if word_lower in terms:
                return True
        
        # Check business vocabulary
        for category, terms in self.business_vocabulary.items():
            if word_lower in terms:
                return True
        
        return False 