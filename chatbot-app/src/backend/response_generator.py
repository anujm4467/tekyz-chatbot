"""
Response Generation Engine

This module implements RAG (Retrieval Augmented Generation) using Groq AI.
It combines retrieved context with user queries to generate accurate, Tekyz-specific responses.
"""

import time
import json
from typing import Dict, Any, List, Optional
from groq import Groq

from ..models.data_models import SearchResult, ClassificationResult, ChatResponse
from ..utils.config_manager import ConfigManager
from ..utils.logger import get_logger


class ResponseGenerator:
    """
    RAG-based response generator using Groq AI.
    
    Features:
    - Combines user queries with retrieved context
    - Generates responses using Groq AI models
    - Validates response quality and relevance
    - Handles off-topic queries gracefully
    """
    
    
    def __init__(self):
        self.config = ConfigManager()
        self.logger = get_logger()
        self.groq_config = self.config.get_groq_config()
        
        # Debug logging for Groq configuration
        self.logger.info(f"DEBUG: Groq config retrieved: api_key={'present' if self.groq_config['api_key'] else 'MISSING'}")
        if self.groq_config["api_key"]:
            self.logger.info(f"DEBUG: API key length: {len(self.groq_config['api_key'])}")
            self.logger.info(f"DEBUG: API key prefix: {self.groq_config['api_key'][:10]}...")
        
        # Initialize Groq client (Groq 0.29.0+ fixes proxies parameter issue)
        if self.groq_config["api_key"]:
            try:
                self.groq_client = Groq(api_key=self.groq_config["api_key"])
                self.logger.info("✅ Groq client initialized successfully for response generation")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize Groq client: {e}")
                self.groq_client = None
        else:
            self.groq_client = None
            self.logger.warning("❌ Groq API key not provided. Response generation will be limited.")
    
    def generate_response(
        self,
        query: str,
        context_results: List[SearchResult],
        classification: Optional[ClassificationResult] = None,
        session_id: str = "default"
    ) -> ChatResponse:
        """
        Generate response using retrieved context.
        
        Args:
            query: User query
            context_results: Retrieved context chunks
            classification: Query classification result
            session_id: Session identifier for logging
            
        Returns:
            Generated response with metadata
        """
        start_time = time.time()
        
        try:
            # Log response generation start
            self.logger.info(f"Generating response for: '{query[:50]}...'")
            
            # Attempt to use Groq AI for response generation
            if self.groq_client:
                self.logger.info("Attempting to generate response with Groq AI")
                try:
                    response_text = self._generate_with_groq(query, context_results, classification)
                    self.logger.info(f"Successfully generated response with Groq AI")
                except Exception as e:
                    self.logger.error(f"Groq generation failed: {str(e)}")
                    self.logger.info("Falling back to template-based response")
                    response_text = self._generate_fallback_response(query, context_results)
            else:
                self.logger.info("Groq client not available, using fallback response")
                response_text = self._generate_fallback_response(query, context_results)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Determine confidence based on available context
            confidence = self._calculate_confidence(context_results, classification)
            
            # Create response metadata
            metadata = {
                "session_id": session_id,
                "processing_time": processing_time,
                "context_sources": len(context_results),
                "tekyz_related": classification.is_tekyz_related if classification else True,
                "classification_confidence": classification.confidence if classification else 0.5,
                "sources": [
                    {
                        "url": result.metadata.get("source_url", "Unknown"),
                        "score": result.score,
                        "snippet": result.text[:100] + "..." if len(result.text) > 100 else result.text
                    } for result in context_results[:3]  # Top 3 sources
                ]
            }
            
            # Log successful response generation
            self.logger.log_response(response_text, session_id, processing_time)
            
            return ChatResponse(
                response_text=response_text,
                sources=context_results,
                classification=classification,
                confidence=confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            # Log error and return fallback response
            processing_time = time.time() - start_time
            self.logger.error(f"Error in response_generation: {str(e)}")
            
            return ChatResponse(
                response_text="I apologize, but I'm experiencing technical difficulties. Please try again.",
                sources=[],
                classification=classification,
                confidence=0.0,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def _generate_with_groq(self, query: str, context_results: List[SearchResult], classification: Optional[ClassificationResult] = None) -> str:
        """
        Generate response using Groq AI with retrieved context.
        
        Args:
            query: User query
            context_results: Retrieved context chunks
            classification: Query classification result
            
        Returns:
            Generated response text
        """
        try:
            # Build context prompt
            context_prompt = self._build_context_prompt(query, context_results)
            
            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": context_prompt
                    }
                ],
                model=self.groq_config["models"]["generator"],
                temperature=self.groq_config["default_params"]["temperature"],
                max_tokens=self.groq_config["default_params"]["max_tokens"],
                top_p=self.groq_config["default_params"]["top_p"]
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"Error in groq_response_generation: {str(e)}")
            return self._generate_fallback_response(query, context_results)
    
    def _build_context_prompt(self, query: str, context_results: List[SearchResult]) -> str:
        """
        Build the prompt with context for Groq AI.
        
        Args:
            query: User query
            context_results: Retrieved context chunks
            
        Returns:
            Formatted prompt with context
        """
        # Build context section
        context_parts = []
        for i, result in enumerate(context_results[:5], 1):  # Limit to top 5 results
            context_parts.append(f"Context {i}:\n{result.text}\nSource: {result.source_url}\n")
        
        context_text = "\n".join(context_parts)
        
        prompt = f"""You are Tekyz's helpful AI assistant. Your job is to provide detailed, specific answers using the context provided below.

CONTEXT FROM TEKYZ KNOWLEDGE BASE:
{context_text}

CRITICAL INSTRUCTIONS:
1. Use the specific information from the context above to answer the question
2. Be detailed and comprehensive - extract key facts, services, capabilities mentioned in the context
3. Include relevant URLs from the context as clickable links in your response where appropriate
4. Structure your response clearly with bullet points or paragraphs as appropriate
5. Reference specific details from the context (services offered, technologies used, project details, etc.)
6. When mentioning information from a source, include the relevant URL as a link: [relevant text](URL)
7. Be confident in presenting information that's clearly stated in the context
8. Only say you don't have information if the context truly doesn't contain relevant details

USER QUESTION: {query}

Based on the context provided above, here's what I can tell you about Tekyz:"""
        
        return prompt
    
    def _generate_fallback_response(self, query: str, context_results: List[SearchResult]) -> str:
        """
        Generate a fallback response when Groq is not available.
        
        Args:
            query: User query
            context_results: Retrieved context chunks
            
        Returns:
            Template-based response
        """
        if not context_results:
            return "I'm sorry, but I don't have specific information about that topic in my knowledge base."
        
        # Extract key information from search results
        all_text = []
        sources = []
        for result in context_results[:3]:  # Use top 3 results
            # Clean and extract text from result
            text_content = result.text
            if isinstance(text_content, str):
                # Remove dictionary formatting if present
                if text_content.startswith("{'text':"):
                    try:
                        import ast
                        parsed = ast.literal_eval(text_content)
                        if isinstance(parsed, dict) and 'text' in parsed:
                            text_content = parsed['text']
                    except:
                        # If parsing fails, use string manipulation
                        start = text_content.find("'text': '") + 9
                        end = text_content.find("',", start)
                        if start > 8 and end > start:
                            text_content = text_content[start:end]
                
                all_text.append(text_content)
                if hasattr(result, 'source_url') and result.source_url:
                    sources.append(result.source_url)
        
        # Create a contextual response based on query type
        query_lower = query.lower()
        
        # Determine response style based on query
        if any(word in query_lower for word in ['services', 'offer', 'provide', 'do']):
            intro = "Based on the information I have about Tekyz's services:"
        elif any(word in query_lower for word in ['team', 'who', 'people', 'staff']):
            intro = "Here's what I know about the Tekyz team:"
        elif any(word in query_lower for word in ['portfolio', 'projects', 'work', 'examples']):
            intro = "Based on Tekyz's portfolio and project information:"
        elif any(word in query_lower for word in ['contact', 'reach', 'address', 'phone', 'email']):
            intro = "For contacting Tekyz:"
        else:
            intro = "Based on the information I have about Tekyz:"
        
        # Combine relevant text snippets
        if all_text:
            # Take the most relevant information
            combined_text = ' '.join(all_text[:2])  # Limit to prevent too long responses
            
            # Clean up the text
            combined_text = combined_text.replace('...', '. ')
            if len(combined_text) > 500:
                combined_text = combined_text[:500] + "..."
            
            response = f"{intro}\n\n{combined_text}"
            
            # Add source information if available
            if sources:
                unique_sources = list(set(sources))[:2]  # Max 2 unique sources
                source_text = "\n\nFor more information, you can visit: " + ", ".join(unique_sources)
                response += source_text
                
            return response
        else:
            return "I found some information in our knowledge base, but I'm having trouble processing it properly. Could you please rephrase your question?"
    
    def _create_decline_response(
        self, 
        query: str, 
        classification: ClassificationResult, 
        start_time: float
    ) -> ChatResponse:
        """
        Create a polite decline response for off-topic queries.
        
        Args:
            query: Original user query
            classification: Classification result
            start_time: Processing start time
            
        Returns:
            Decline response
        """
        decline_messages = [
            "I'm Tekyz's AI assistant, and I'm designed to help with questions about Tekyz - our services, portfolio, team, and company information.",
            "I'd be happy to tell you about Tekyz's web development services, mobile app solutions, software development expertise, or our portfolio of projects.",
            "Is there anything specific about Tekyz that you'd like to know?"
        ]
        
        response_text = " ".join(decline_messages)
        processing_time = time.time() - start_time
        
        return ChatResponse(
            response_text=response_text,
            sources=[],
            classification=classification,
            confidence=0.9,  # High confidence in decline
            processing_time=processing_time
        )
    
    def _create_no_context_response(self, query: str, start_time: float) -> ChatResponse:
        """
        Create response when no relevant context is found.
        
        Args:
            query: Original user query
            start_time: Processing start time
            
        Returns:
            No context response
        """
        response_text = (
            "I don't have specific information about that topic in my current knowledge base. "
            "However, I'd be happy to help you with questions about Tekyz's services, portfolio, "
            "team, or company information. Is there something specific about Tekyz you'd like to know?"
        )
        
        processing_time = time.time() - start_time
        
        return ChatResponse(
            response_text=response_text,
            sources=[],
            confidence=0.7,
            processing_time=processing_time
        )
    
    def validate_response(self, response: str, query: str) -> Dict[str, Any]:
        """
        Validate the quality of the generated response.
        
        Args:
            response: Generated response text
            query: Original user query
            
        Returns:
            Validation result dictionary
        """
        validation_result = {
            "is_valid": True,
            "quality_score": 0.8,  # Default score
            "issues": []
        }
        
        # Check response length
        if len(response) < 10:
            validation_result["issues"].append("Response too short")
            validation_result["quality_score"] -= 0.3
        
        if len(response) > self.config.max_response_length:
            validation_result["issues"].append("Response too long")
            validation_result["quality_score"] -= 0.2
        
        # Check for generic responses
        generic_phrases = [
            "i don't know", "not sure", "maybe", "possibly", 
            "i think", "i believe", "probably"
        ]
        
        response_lower = response.lower()
        generic_count = sum(1 for phrase in generic_phrases if phrase in response_lower)
        if generic_count > 2:
            validation_result["issues"].append("Too many uncertain phrases")
            validation_result["quality_score"] -= 0.2
        
        # Check for Tekyz relevance
        tekyz_keywords = ["tekyz", "our", "we", "company", "services", "team"]
        tekyz_mentions = sum(1 for keyword in tekyz_keywords if keyword in response_lower)
        
        if tekyz_mentions == 0:
            validation_result["issues"].append("No Tekyz context in response")
            validation_result["quality_score"] -= 0.3
        
        # Determine overall validity
        validation_result["is_valid"] = validation_result["quality_score"] > 0.3
        
        return validation_result
    
    def _calculate_confidence(
        self, 
        context_results: List[SearchResult], 
        classification: Optional[ClassificationResult] = None
    ) -> float:
        """
        Calculate confidence score for the response.
        
        Args:
            context_results: Retrieved context chunks
            classification: Query classification result
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not context_results:
            return 0.1
        
        # Base confidence from top search result
        base_confidence = context_results[0].score
        
        # Adjust based on number of supporting results
        support_factor = min(len(context_results) / 3.0, 1.0)  # Max boost at 3+ results
        
        # Adjust based on classification
        classification_factor = classification.confidence if classification else 0.5
        
        # Calculate final confidence
        confidence = base_confidence * 0.6 + support_factor * 0.2 + classification_factor * 0.2
        
        return min(max(confidence, 0.0), 1.0)  # Clamp between 0 and 1
    
    def get_suggested_follow_ups(self, category: str, response: str) -> List[str]:
        """
        Generate follow-up question suggestions based on the response category.
        
        Args:
            category: Response category (services, portfolio, etc.)
            response: Generated response text
            
        Returns:
            List of suggested follow-up questions
        """
        suggestions = {
            "services": [
                "What is the typical timeline for these services?",
                "How much do these services cost?",
                "Can you show me examples of similar projects?"
            ],
            "portfolio": [
                "Tell me more about this project",
                "What technologies were used?",
                "How long did this project take?"
            ],
            "company": [
                "How can I get in touch with the team?",
                "What is Tekyz's experience in my industry?",
                "Do you offer consultation calls?"
            ],
            "contact": [
                "What's the best way to reach out?",
                "Do you offer free consultations?",
                "What information should I prepare for our discussion?"
            ]
        }
        
        return suggestions.get(category, [
            "Tell me more about Tekyz's services",
            "Can you show me some portfolio examples?",
            "How can I get in touch with Tekyz?"
        ]) 