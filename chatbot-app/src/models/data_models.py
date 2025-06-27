"""
Data Models

This module defines data structures used throughout the Tekyz chatbot application.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import uuid


class MessageType(Enum):
    """Message types for chat interface."""
    USER = "user"
    BOT = "bot"
    SYSTEM = "system"


class QueryCategory(Enum):
    """Categories for query classification."""
    SERVICES = "services"
    PORTFOLIO = "portfolio"
    COMPANY = "company"
    CONTACT = "contact"
    GENERAL = "general"
    OFF_TOPIC = "off_topic"


class ConfidenceLevel(Enum):
    """Confidence levels for responses."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class SearchResult:
    """Result from vector similarity search."""
    id: str
    score: float
    text: str
    source_url: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get confidence level based on score."""
        if self.score >= 0.8:
            return ConfidenceLevel.HIGH
        elif self.score >= 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW


@dataclass
class ClassificationResult:
    """Result from query intent classification."""
    is_tekyz_related: bool
    confidence: float
    category: QueryCategory
    reasoning: str
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get confidence level based on score."""
        if self.confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW


@dataclass
class Message:
    """Chat message data structure."""
    content: str
    message_type: MessageType
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'message_id': self.message_id,
            'content': self.content,
            'message_type': self.message_type.value,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create from dictionary."""
        return cls(
            content=data['content'],
            message_type=MessageType(data['message_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            message_id=data.get('message_id', str(uuid.uuid4())),
            metadata=data.get('metadata', {})
        )


@dataclass
class ChatResponse:
    """Complete response from chatbot."""
    response_text: str
    sources: List[SearchResult] = field(default_factory=list)
    classification: Optional[ClassificationResult] = None
    confidence: float = 0.0
    processing_time: float = 0.0
    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get confidence level based on score."""
        if self.confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    @property
    def has_sources(self) -> bool:
        """Check if response has sources."""
        return len(self.sources) > 0
    
    @property
    def is_successful(self) -> bool:
        """Check if response generation was successful."""
        return self.error_message is None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        classification_dict = None
        if self.classification:
            classification_dict = {
                'is_tekyz_related': self.classification.is_tekyz_related,
                'confidence': self.classification.confidence,
                'category': self.classification.category.value,
                'reasoning': self.classification.reasoning
            }
        
        return {
            'response_id': self.response_id,
            'response_text': self.response_text,
            'sources': [source.__dict__ for source in self.sources],
            'classification': classification_dict,
            'confidence': self.confidence,
            'confidence_level': self.confidence_level.value,
            'processing_time': self.processing_time,
            'timestamp': self.timestamp.isoformat(),
            'error_message': self.error_message,
            'has_sources': self.has_sources,
            'is_successful': self.is_successful
        }


@dataclass
class ChatSession:
    """Chat session data structure."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    query_count: int = 0
    
    def add_message(self, message: Message):
        """Add a message to the session."""
        self.messages.append(message)
        self.last_activity = datetime.now()
        if message.message_type == MessageType.USER:
            self.query_count += 1
    
    def get_recent_messages(self, count: int = 10) -> List[Message]:
        """Get recent messages."""
        return self.messages[-count:] if count > 0 else self.messages
    
    def get_conversation_context(self, max_messages: int = 5) -> str:
        """Get conversation context as string."""
        recent_messages = self.get_recent_messages(max_messages)
        context_parts = []
        
        for msg in recent_messages:
            role = "Human" if msg.message_type == MessageType.USER else "Assistant"
            context_parts.append(f"{role}: {msg.content}")
        
        return "\n".join(context_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'session_id': self.session_id,
            'messages': [msg.to_dict() for msg in self.messages],
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'user_preferences': self.user_preferences,
            'query_count': self.query_count
        }


@dataclass
class AnalyticsData:
    """Analytics data structure."""
    timestamp: datetime
    session_id: str
    user_query: str
    query_length: int
    classification_result: Optional[ClassificationResult]
    search_results_count: int
    response_generated: str
    response_time_ms: int
    confidence_score: float
    user_feedback: Optional[str] = None
    error_occurred: bool = False
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        classification_dict = None
        if self.classification_result:
            classification_dict = {
                'is_tekyz_related': self.classification_result.is_tekyz_related,
                'confidence': self.classification_result.confidence,
                'category': self.classification_result.category.value,
                'reasoning': self.classification_result.reasoning
            }
        
        return {
            'timestamp': self.timestamp.isoformat(),
            'session_id': self.session_id,
            'user_query': self.user_query,
            'query_length': self.query_length,
            'classification_result': classification_dict,
            'search_results_count': self.search_results_count,
            'response_generated': self.response_generated,
            'response_time_ms': self.response_time_ms,
            'confidence_score': self.confidence_score,
            'user_feedback': self.user_feedback,
            'error_occurred': self.error_occurred,
            'error_message': self.error_message
        } 