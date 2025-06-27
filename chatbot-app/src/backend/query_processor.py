"""
Query Processing Engine

This module handles query validation, intent classification, and routing for the Tekyz chatbot.
It determines if queries are Tekyz-related and prepares them for vector search.
"""

import re
import time
import json
from typing import Dict, Any, Optional
from groq import Groq

from ..models.data_models import ClassificationResult, QueryCategory
from ..utils.config_manager import ConfigManager
from ..utils.logger import get_logger


class QueryProcessor:
    """
    Main query processing engine for the Tekyz chatbot.
    
    Handles:
    - Query validation and preprocessing
    - Intent classification using Groq AI
    - Query routing and filtering
    """
    
    def __init__(self):
        self.config = ConfigManager()
        self.logger = get_logger()
        self.groq_config = self.config.get_groq_config()
        
        # Initialize Groq client
        if self.groq_config["api_key"]:
            try:
                self.groq_client = Groq(api_key=self.groq_config["api_key"])
                self.logger.logger.info("Groq client initialized successfully")
            except Exception as e:
                self.logger.logger.warning(f"Failed to initialize Groq client: {e}. Using fallback logic.")
                self.groq_client = None
        else:
            self.groq_client = None
            self.logger.logger.warning("Groq API key not provided. Classification will use fallback logic.")
    
    def process_query(self, user_query: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Main query processing pipeline.
        
        Args:
            user_query: Raw user input
            session_id: Session identifier for logging
            
        Returns:
            Dictionary containing processing results
        """
        start_time = time.time()
        
        try:
            # Log incoming query
            self.logger.log_query(user_query, session_id)
            
            # Step 1: Validate query
            validation_result = self.validate_query(user_query)
            if not validation_result["is_valid"]:
                return {
                    "success": False,
                    "error": validation_result["error"],
                    "processing_time": time.time() - start_time
                }
            
            # Step 2: Preprocess query
            processed_query = self.preprocess_query(user_query)
            
            # Step 3: Classify intent
            classification = self.classify_intent(processed_query)
            
            # Log classification results
            self.logger.log_classification(
                processed_query, 
                classification.is_tekyz_related, 
                classification.confidence
            )
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "original_query": user_query,
                "processed_query": processed_query,
                "classification": classification,
                "processing_time": processing_time
            }
            
        except Exception as e:
            self.logger.log_error(e, "query_processing")
            return {
                "success": False,
                "error": f"Query processing failed: {str(e)}",
                "processing_time": time.time() - start_time
            }
    
    def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Validate user query for basic requirements.
        
        Args:
            query: User input to validate
            
        Returns:
            Validation result dictionary
        """
        # Check if query is empty or just whitespace
        if not query or not query.strip():
            return {
                "is_valid": False,
                "error": "Query cannot be empty"
            }
        
        # Check query length
        max_length = self.config.max_query_length
        if len(query) > max_length:
            return {
                "is_valid": False,
                "error": f"Query too long. Maximum {max_length} characters allowed."
            }
        
        # Check for minimum length
        if len(query.strip()) < 3:
            return {
                "is_valid": False,
                "error": "Query too short. Please provide more details."
            }
        
        # Check for potentially harmful content (basic filter)
        harmful_patterns = [
            r'\bsql\s+injection\b',
            r'\bscript\b.*\balert\b',
            r'<script.*?>',
            r'javascript:',
        ]
        
        query_lower = query.lower()
        for pattern in harmful_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return {
                    "is_valid": False,
                    "error": "Query contains potentially harmful content"
                }
        
        return {
            "is_valid": True,
            "cleaned_query": query.strip()
        }
    
    def preprocess_query(self, query: str) -> str:
        """
        Clean and normalize query text for processing.
        
        Args:
            query: Raw query text
            
        Returns:
            Processed query text
        """
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', query.strip())
        
        # Remove special characters that might interfere with processing
        # Keep letters, numbers, spaces, and basic punctuation
        cleaned = re.sub(r'[^\w\s\.\?\!\,\-\:]', '', cleaned)
        
        # Convert to lower case for consistent processing
        # (Note: We'll preserve original case for display)
        processed = cleaned.lower()
        
        return processed
    
    def classify_intent(self, query: str) -> ClassificationResult:
        """
        Classify query intent using Groq AI or fallback logic.
        
        Args:
            query: Preprocessed query text
            
        Returns:
            Classification result
        """
        if self.groq_client:
            return self._classify_with_groq(query)
        else:
            return self._classify_with_fallback(query)
    
    def _classify_with_groq(self, query: str) -> ClassificationResult:
        """
        Use Groq AI for intent classification.
        
        Args:
            query: Query text to classify
            
        Returns:
            Classification result from Groq
        """
        try:
            prompt = self._build_classification_prompt(query)
            
            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.groq_config["models"]["classifier"],
                temperature=self.groq_config["default_params"]["temperature"],
                max_tokens=self.groq_config["default_params"]["max_tokens"],
                top_p=self.groq_config["default_params"]["top_p"]
            )
            
            # Parse the JSON response
            result_text = response.choices[0].message.content.strip()
            
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result_json = json.loads(json_match.group())
            else:
                # Fallback if JSON parsing fails
                return self._classify_with_fallback(query)
            
            # Map category string to enum
            category_map = {
                "services": QueryCategory.SERVICES,
                "portfolio": QueryCategory.PORTFOLIO,
                "company": QueryCategory.COMPANY,
                "contact": QueryCategory.CONTACT,
                "general": QueryCategory.GENERAL,
                "off-topic": QueryCategory.OFF_TOPIC
            }
            
            category = category_map.get(
                result_json.get("category", "off-topic").lower(),
                QueryCategory.OFF_TOPIC
            )
            
            return ClassificationResult(
                is_tekyz_related=result_json.get("is_tekyz_related", False),
                confidence=float(result_json.get("confidence", 0.0)),
                category=category,
                reasoning=result_json.get("reasoning", "")
            )
            
        except Exception as e:
            self.logger.log_error(e, "groq_classification")
            # Fallback to rule-based classification
            return self._classify_with_fallback(query)
    
    def _classify_with_fallback(self, query: str) -> ClassificationResult:
        """
        Fallback classification using keyword matching.
        
        Args:
            query: Query text to classify
            
        Returns:
            Rule-based classification result
        """
        query_lower = query.lower()
        
        # Define keyword patterns for different categories
        tekyz_keywords = [
            "tekyz", "tekyz.com", "your company", "your team", "your services"
        ]
        
        service_keywords = [
            "web development", "mobile app", "software development", 
            "custom software", "development services", "programming",
            "website", "app development", "software solutions"
        ]
        
        portfolio_keywords = [
            "portfolio", "projects", "case studies", "previous work",
            "examples", "clients", "past projects"
        ]
        
        company_keywords = [
            "about", "team", "company", "who are you", "background",
            "history", "founders", "experience"
        ]
        
        contact_keywords = [
            "contact", "reach out", "get in touch", "email", "phone",
            "consultation", "quote", "pricing"
        ]
        
        # Check for Tekyz-specific mentions
        is_tekyz_related = any(keyword in query_lower for keyword in tekyz_keywords)
        
        # Determine category based on keywords
        if any(keyword in query_lower for keyword in service_keywords):
            category = QueryCategory.SERVICES
            is_tekyz_related = True  # Services questions are Tekyz-related
        elif any(keyword in query_lower for keyword in portfolio_keywords):
            category = QueryCategory.PORTFOLIO
            is_tekyz_related = True
        elif any(keyword in query_lower for keyword in company_keywords):
            category = QueryCategory.COMPANY
            is_tekyz_related = True
        elif any(keyword in query_lower for keyword in contact_keywords):
            category = QueryCategory.CONTACT
            is_tekyz_related = True
        else:
            # Check if it's a general business/tech question that might be relevant
            general_keywords = [
                "how to", "what is", "best practices", "advice",
                "recommendations", "help", "question"
            ]
            
            if any(keyword in query_lower for keyword in general_keywords):
                category = QueryCategory.GENERAL
                # General questions might be Tekyz-related if they mention relevant topics
                tech_keywords = [
                    "software", "development", "technology", "digital",
                    "programming", "coding", "business"
                ]
                is_tekyz_related = any(keyword in query_lower for keyword in tech_keywords)
            else:
                category = QueryCategory.OFF_TOPIC
                is_tekyz_related = False
        
        # Calculate confidence based on keyword matches
        confidence = 0.8 if is_tekyz_related else 0.6
        
        reasoning = f"Classified as {category.value} based on keyword analysis"
        
        return ClassificationResult(
            is_tekyz_related=is_tekyz_related,
            confidence=confidence,
            category=category,
            reasoning=reasoning
        )
    
    def _build_classification_prompt(self, query: str) -> str:
        """
        Build the prompt for Groq classification.
        
        Args:
            query: Query to classify
            
        Returns:
            Formatted prompt string
        """
        return f"""You are a query classifier for Tekyz company chatbot.

Analyze the user query and determine if it's related to Tekyz company:
- Tekyz services (web development, mobile apps, software solutions)
- Tekyz portfolio and projects  
- Tekyz team and company information
- Tekyz contact and business details
- General questions about Tekyz

Respond with a JSON object:
{{
    "is_tekyz_related": boolean,
    "confidence": float (0.0 to 1.0),
    "category": "string (services|portfolio|company|contact|general|off-topic)",
    "reasoning": "brief explanation"
}}

User Query: {query}""" 