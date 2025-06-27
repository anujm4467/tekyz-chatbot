"""
Analytics Manager

This module handles analytics collection, performance monitoring, and usage tracking
for the Tekyz chatbot application.
"""

import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

from ..models.data_models import AnalyticsData, ChatResponse, ClassificationResult
from ..utils.config_manager import ConfigManager
from ..utils.logger import get_logger


class AnalyticsManager:
    """
    Analytics and monitoring manager for the chatbot.
    
    Features:
    - Query interaction logging
    - Performance metrics tracking
    - Usage statistics generation
    - System health monitoring
    """
    
    def __init__(self):
        self.config = ConfigManager()
        self.logger = get_logger()
        self.analytics_file = Path("logs/analytics.jsonl")
        self.analytics_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log_query_interaction(
        self,
        session_id: str,
        user_query: str,
        chat_response: ChatResponse,
        classification_result: Optional[ClassificationResult] = None,
        user_feedback: Optional[str] = None
    ):
        """
        Log a complete query interaction.
        
        Args:
            session_id: Session identifier
            user_query: User's original query
            chat_response: Generated response
            classification_result: Query classification
            user_feedback: Optional user feedback
        """
        try:
            analytics_data = AnalyticsData(
                timestamp=datetime.now(),
                session_id=session_id,
                user_query=user_query,
                query_length=len(user_query),
                classification_result=classification_result,
                search_results_count=len(chat_response.sources),
                response_generated=chat_response.response_text[:200] + "..." if len(chat_response.response_text) > 200 else chat_response.response_text,
                response_time_ms=int(chat_response.processing_time * 1000),
                confidence_score=chat_response.confidence,
                user_feedback=user_feedback,
                error_occurred=not chat_response.is_successful,
                error_message=chat_response.error_message
            )
            
            # Write to analytics file
            with open(self.analytics_file, 'a') as f:
                f.write(json.dumps(analytics_data.to_dict()) + '\n')
            
            # Log performance metrics
            self.log_performance(
                "query_interaction",
                chat_response.processing_time,
                {
                    "query_length": len(user_query),
                    "response_length": len(chat_response.response_text),
                    "sources_count": len(chat_response.sources),
                    "confidence": chat_response.confidence
                }
            )
            
        except Exception as e:
            self.logger.log_error(e, "analytics_logging")
    
    def track_search_performance(self, search_metrics: Dict[str, Any]):
        """
        Track vector search performance metrics.
        
        Args:
            search_metrics: Dictionary containing search performance data
        """
        try:
            self.logger.log_performance(
                "vector_search",
                search_metrics.get("duration", 0),
                {
                    "results_count": search_metrics.get("results_count", 0),
                    "query_length": search_metrics.get("query_length", 0),
                    "average_score": search_metrics.get("average_score", 0)
                }
            )
            
        except Exception as e:
            self.logger.log_error(e, "search_performance_tracking")
    
    def generate_usage_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Generate usage statistics for the specified time period.
        
        Args:
            hours: Number of hours to analyze (default: 24)
            
        Returns:
            Dictionary containing usage statistics
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Read analytics data
            analytics_data = self._read_analytics_data(cutoff_time)
            
            if not analytics_data:
                return {"error": "No data available for the specified period"}
            
            stats = {
                "time_period": f"Last {hours} hours",
                "total_queries": len(analytics_data),
                "unique_sessions": len(set(data.session_id for data in analytics_data)),
                "average_response_time": sum(data.response_time_ms for data in analytics_data) / len(analytics_data),
                "success_rate": sum(1 for data in analytics_data if not data.error_occurred) / len(analytics_data),
                "average_confidence": sum(data.confidence_score for data in analytics_data) / len(analytics_data),
                "query_categories": self._analyze_query_categories(analytics_data),
                "response_times": self._analyze_response_times(analytics_data),
                "error_analysis": self._analyze_errors(analytics_data),
                "popular_queries": self._analyze_popular_queries(analytics_data)
            }
            
            return stats
            
        except Exception as e:
            self.logger.log_error(e, "usage_statistics_generation")
            return {"error": f"Failed to generate statistics: {str(e)}"}
    
    def monitor_system_health(self) -> Dict[str, Any]:
        """
        Monitor system health and performance.
        
        Returns:
            System health status dictionary
        """
        try:
            cutoff_time = datetime.now() - timedelta(minutes=5)  # Last 5 minutes
            recent_data = self._read_analytics_data(cutoff_time)
            
            health_status = {
                "timestamp": datetime.now().isoformat(),
                "status": "healthy",
                "recent_activity": {
                    "queries_last_5min": len(recent_data),
                    "average_response_time": 0,
                    "error_rate": 0
                },
                "alerts": []
            }
            
            if recent_data:
                # Calculate metrics
                avg_response_time = sum(data.response_time_ms for data in recent_data) / len(recent_data)
                error_rate = sum(1 for data in recent_data if data.error_occurred) / len(recent_data)
                
                health_status["recent_activity"]["average_response_time"] = avg_response_time
                health_status["recent_activity"]["error_rate"] = error_rate
                
                # Check for alerts
                if avg_response_time > 5000:  # 5 seconds
                    health_status["alerts"].append("High response times detected")
                    health_status["status"] = "warning"
                
                if error_rate > 0.1:  # 10% error rate
                    health_status["alerts"].append("High error rate detected")
                    health_status["status"] = "critical"
                
                if len(recent_data) > 100:  # High query volume
                    health_status["alerts"].append("High query volume detected")
            
            return health_status
            
        except Exception as e:
            self.logger.log_error(e, "system_health_monitoring")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def log_performance(self, operation: str, duration: float, details: Optional[Dict] = None):
        """
        Log performance metrics for specific operations.
        
        Args:
            operation: Name of the operation
            duration: Operation duration in seconds
            details: Optional additional details
        """
        self.logger.log_performance(operation, duration, details)
    
    def _read_analytics_data(self, cutoff_time: datetime) -> List[AnalyticsData]:
        """
        Read analytics data from file filtered by time.
        
        Args:
            cutoff_time: Only include data after this time
            
        Returns:
            List of AnalyticsData objects
        """
        analytics_data = []
        
        try:
            if not self.analytics_file.exists():
                return analytics_data
            
            with open(self.analytics_file, 'r') as f:
                for line in f:
                    try:
                        data_dict = json.loads(line.strip())
                        timestamp = datetime.fromisoformat(data_dict['timestamp'])
                        
                        if timestamp >= cutoff_time:
                            # Convert back to AnalyticsData object
                            classification_result = None
                            if data_dict.get('classification_result'):
                                classification_result = ClassificationResult(**data_dict['classification_result'])
                            
                            analytics_data.append(AnalyticsData(
                                timestamp=timestamp,
                                session_id=data_dict['session_id'],
                                user_query=data_dict['user_query'],
                                query_length=data_dict['query_length'],
                                classification_result=classification_result,
                                search_results_count=data_dict['search_results_count'],
                                response_generated=data_dict['response_generated'],
                                response_time_ms=data_dict['response_time_ms'],
                                confidence_score=data_dict['confidence_score'],
                                user_feedback=data_dict.get('user_feedback'),
                                error_occurred=data_dict['error_occurred'],
                                error_message=data_dict.get('error_message')
                            ))
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        self.logger.logger.warning(f"Failed to parse analytics line: {e}")
                        continue
            
        except Exception as e:
            self.logger.log_error(e, "analytics_data_reading")
        
        return analytics_data
    
    def _analyze_query_categories(self, analytics_data: List[AnalyticsData]) -> Dict[str, int]:
        """Analyze query categories distribution."""
        categories = {}
        for data in analytics_data:
            if data.classification_result:
                category = data.classification_result.category.value
                categories[category] = categories.get(category, 0) + 1
        return categories
    
    def _analyze_response_times(self, analytics_data: List[AnalyticsData]) -> Dict[str, float]:
        """Analyze response time statistics."""
        response_times = [data.response_time_ms for data in analytics_data]
        
        if not response_times:
            return {}
        
        response_times.sort()
        n = len(response_times)
        
        return {
            "min": min(response_times),
            "max": max(response_times),
            "average": sum(response_times) / n,
            "median": response_times[n // 2],
            "p95": response_times[int(n * 0.95)] if n > 0 else 0
        }
    
    def _analyze_errors(self, analytics_data: List[AnalyticsData]) -> Dict[str, Any]:
        """Analyze error patterns."""
        errors = [data for data in analytics_data if data.error_occurred]
        error_messages = {}
        
        for error_data in errors:
            if error_data.error_message:
                error_messages[error_data.error_message] = error_messages.get(error_data.error_message, 0) + 1
        
        return {
            "total_errors": len(errors),
            "error_rate": len(errors) / len(analytics_data) if analytics_data else 0,
            "error_messages": error_messages
        }
    
    def _analyze_popular_queries(self, analytics_data: List[AnalyticsData], limit: int = 10) -> List[Dict[str, Any]]:
        """Analyze most popular queries."""
        query_counts = {}
        query_details = {}
        
        for data in analytics_data:
            query = data.user_query.lower().strip()
            if len(query) > 5:  # Only meaningful queries
                query_counts[query] = query_counts.get(query, 0) + 1
                if query not in query_details:
                    query_details[query] = {
                        "average_confidence": data.confidence_score,
                        "average_response_time": data.response_time_ms,
                        "count": 1
                    }
                else:
                    details = query_details[query]
                    details["average_confidence"] = (details["average_confidence"] * details["count"] + data.confidence_score) / (details["count"] + 1)
                    details["average_response_time"] = (details["average_response_time"] * details["count"] + data.response_time_ms) / (details["count"] + 1)
                    details["count"] += 1
        
        # Sort by frequency and take top N
        popular_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        return [
            {
                "query": query,
                "count": count,
                "average_confidence": query_details[query]["average_confidence"],
                "average_response_time": query_details[query]["average_response_time"]
            }
            for query, count in popular_queries
        ] 