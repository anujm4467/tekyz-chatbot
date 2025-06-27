"""
Pipeline tracking database for storing progress, metrics, and ETAs
"""

import sqlite3
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from loguru import logger

@dataclass
class PipelineStep:
    """Pipeline step information"""
    step_name: str
    step_order: int
    status: str  # 'pending', 'running', 'completed', 'failed', 'skipped'
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    progress_percentage: float = 0.0
    items_total: int = 0
    items_processed: int = 0
    error_message: Optional[str] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Get step duration in seconds"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def is_active(self) -> bool:
        """Check if step is currently running"""
        return self.status == 'running'
    
    @property
    def is_complete(self) -> bool:
        """Check if step is completed"""
        return self.status in ['completed', 'failed', 'skipped']

@dataclass
class PipelineMetrics:
    """Pipeline performance metrics"""
    total_urls: int = 0
    processed_urls: int = 0
    successful_urls: int = 0
    failed_urls: int = 0
    total_chunks: int = 0
    total_embeddings: int = 0
    total_vectors: int = 0
    duplicate_count: int = 0
    start_time: float = 0.0
    end_time: Optional[float] = None
    estimated_completion: Optional[float] = None
    current_step: str = ""
    current_step_progress: float = 0.0
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def processing_rate(self) -> float:
        """Get URLs processed per second"""
        if self.elapsed_time <= 0:
            return 0.0
        return self.processed_urls / self.elapsed_time
    
    @property
    def eta_seconds(self) -> Optional[float]:
        """Calculate ETA in seconds"""
        if self.processing_rate <= 0 or self.processed_urls >= self.total_urls:
            return None
        
        remaining_urls = self.total_urls - self.processed_urls
        return remaining_urls / self.processing_rate
    
    @property
    def progress_percentage(self) -> float:
        """Get progress as percentage"""
        if self.total_urls <= 0:
            return 0.0
        return (self.processed_urls / self.total_urls) * 100

@dataclass
class PipelineJob:
    """Pipeline job record"""
    job_id: str
    status: str  # 'running', 'completed', 'failed', 'stopped'
    start_time: float
    end_time: Optional[float] = None
    input_urls: List[str] = None
    metrics: PipelineMetrics = None
    logs: List[str] = None
    steps: List[PipelineStep] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.input_urls is None:
            self.input_urls = []
        if self.metrics is None:
            self.metrics = PipelineMetrics(start_time=self.start_time)
        if self.logs is None:
            self.logs = []
        if self.steps is None:
            self.steps = []

class PipelineTracker:
    """SQLite-based pipeline tracking system"""
    
    # Define standard pipeline steps
    PIPELINE_STEPS = [
        ("initialization", 1, "Initializing pipeline"),
        ("file_processing", 2, "Processing uploaded files"),
        ("url_discovery", 3, "Discovering URLs"),
        ("web_scraping", 4, "Scraping web content"),
        ("content_cleaning", 5, "Cleaning content"),
        ("text_chunking", 6, "Chunking text"),
        ("embedding_generation", 7, "Generating embeddings"),
        ("deduplication", 8, "Checking for duplicates"),
        ("database_upload", 9, "Uploading to database"),
        ("finalization", 10, "Finalizing pipeline")
    ]
    
    def __init__(self, db_path: str = "data/pipeline_tracker.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        logger.info(f"Pipeline tracker initialized: {self.db_path}")
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    input_urls TEXT,  -- JSON array
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    step_name TEXT NOT NULL,
                    step_order INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    start_time REAL,
                    end_time REAL,
                    progress_percentage REAL DEFAULT 0.0,
                    items_total INTEGER DEFAULT 0,
                    items_processed INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES pipeline_jobs (job_id),
                    UNIQUE(job_id, step_name)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_metrics (
                    job_id TEXT,
                    timestamp REAL NOT NULL,
                    total_urls INTEGER DEFAULT 0,
                    processed_urls INTEGER DEFAULT 0,
                    successful_urls INTEGER DEFAULT 0,
                    failed_urls INTEGER DEFAULT 0,
                    total_chunks INTEGER DEFAULT 0,
                    total_embeddings INTEGER DEFAULT 0,
                    total_vectors INTEGER DEFAULT 0,
                    duplicate_count INTEGER DEFAULT 0,
                    current_step TEXT DEFAULT '',
                    current_step_progress REAL DEFAULT 0.0,
                    estimated_completion REAL,
                    FOREIGN KEY (job_id) REFERENCES pipeline_jobs (job_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT,
                    timestamp REAL NOT NULL,
                    log_level TEXT DEFAULT 'INFO',
                    message TEXT NOT NULL,
                    step_name TEXT,
                    FOREIGN KEY (job_id) REFERENCES pipeline_jobs (job_id)
                )
            """)
            
            # Handle database migrations
            self._migrate_database(conn)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON pipeline_jobs(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_steps_job_order ON pipeline_steps(job_id, step_order)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_job_timestamp ON pipeline_metrics(job_id, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_job_timestamp ON pipeline_logs(job_id, timestamp)")
            
            conn.commit()
    
    def _migrate_database(self, conn):
        """Handle database schema migrations"""
        try:
            # Check if step_name column exists in pipeline_logs
            cursor = conn.execute("PRAGMA table_info(pipeline_logs)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'step_name' not in columns:
                logger.info("Adding step_name column to pipeline_logs table")
                conn.execute("ALTER TABLE pipeline_logs ADD COLUMN step_name TEXT")
            
            # Check if current_step and current_step_progress exist in pipeline_metrics
            cursor = conn.execute("PRAGMA table_info(pipeline_metrics)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'current_step' not in columns:
                logger.info("Adding current_step column to pipeline_metrics table")
                conn.execute("ALTER TABLE pipeline_metrics ADD COLUMN current_step TEXT DEFAULT ''")
            
            if 'current_step_progress' not in columns:
                logger.info("Adding current_step_progress column to pipeline_metrics table")
                conn.execute("ALTER TABLE pipeline_metrics ADD COLUMN current_step_progress REAL DEFAULT 0.0")
                
        except Exception as e:
            logger.warning(f"Migration warning (this is usually safe to ignore): {e}")
    
    def start_job(self, job_id: str, input_urls: List[str]) -> PipelineJob:
        """Start a new pipeline job"""
        job = PipelineJob(
            job_id=job_id,
            status='running',
            start_time=time.time(),
            input_urls=input_urls
        )
        
        with sqlite3.connect(self.db_path) as conn:
            # Create job record
            conn.execute("""
                INSERT INTO pipeline_jobs (job_id, status, start_time, input_urls)
                VALUES (?, ?, ?, ?)
            """, (job_id, job.status, job.start_time, json.dumps(input_urls)))
            
            # Initialize all pipeline steps
            for step_name, step_order, description in self.PIPELINE_STEPS:
                conn.execute("""
                    INSERT INTO pipeline_steps (job_id, step_name, step_order, status)
                    VALUES (?, ?, ?, 'pending')
                """, (job_id, step_name, step_order))
            
            conn.commit()
        
        self.add_log(job_id, f"Pipeline started with {len(input_urls)} input URLs")
        logger.info(f"Started pipeline job: {job_id}")
        return job
    
    def start_step(self, job_id: str, step_name: str, items_total: int = 0):
        """Start a pipeline step"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE pipeline_steps 
                SET status = 'running', start_time = ?, items_total = ?, updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ? AND step_name = ?
            """, (time.time(), items_total, job_id, step_name))
            conn.commit()
        
        self.add_log(job_id, f"Started step: {step_name}", step_name=step_name)
        logger.info(f"Started step {step_name} for job {job_id}")
    
    def update_step_progress(self, job_id: str, step_name: str, items_processed: int, progress_percentage: float = None):
        """Update step progress"""
        if progress_percentage is None and items_processed > 0:
            # Calculate progress from items if not provided
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT items_total FROM pipeline_steps 
                    WHERE job_id = ? AND step_name = ?
                """, (job_id, step_name))
                result = cursor.fetchone()
                if result and result[0] > 0:
                    progress_percentage = (items_processed / result[0]) * 100
                else:
                    progress_percentage = 0
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE pipeline_steps 
                SET items_processed = ?, progress_percentage = ?, updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ? AND step_name = ?
            """, (items_processed, progress_percentage or 0, job_id, step_name))
            conn.commit()
    
    def complete_step(self, job_id: str, step_name: str, success: bool = True, error_message: str = None):
        """Complete a pipeline step"""
        status = 'completed' if success else 'failed'
        end_time = time.time()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE pipeline_steps 
                SET status = ?, end_time = ?, error_message = ?, progress_percentage = ?, updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ? AND step_name = ?
            """, (status, end_time, error_message, 100.0 if success else None, job_id, step_name))
            conn.commit()
        
        result_msg = f"Completed step: {step_name}" + (f" - {error_message}" if error_message else "")
        self.add_log(job_id, result_msg, level='ERROR' if not success else 'INFO', step_name=step_name)
        logger.info(f"Completed step {step_name} for job {job_id}: {status}")
    
    def skip_step(self, job_id: str, step_name: str, reason: str = "Not applicable"):
        """Skip a pipeline step"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE pipeline_steps 
                SET status = 'skipped', error_message = ?, progress_percentage = 100.0, updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ? AND step_name = ?
            """, (reason, job_id, step_name))
            conn.commit()
        
        self.add_log(job_id, f"Skipped step: {step_name} - {reason}", step_name=step_name)
    
    def get_current_step(self, job_id: str) -> Optional[PipelineStep]:
        """Get the current active step"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT step_name, step_order, status, start_time, end_time, 
                       progress_percentage, items_total, items_processed, error_message
                FROM pipeline_steps 
                WHERE job_id = ? AND status = 'running'
                ORDER BY step_order
                LIMIT 1
            """, (job_id,))
            result = cursor.fetchone()
            
            if result:
                return PipelineStep(
                    step_name=result[0],
                    step_order=result[1],
                    status=result[2],
                    start_time=result[3],
                    end_time=result[4],
                    progress_percentage=result[5] or 0,
                    items_total=result[6] or 0,
                    items_processed=result[7] or 0,
                    error_message=result[8]
                )
            return None
    
    def get_job_steps(self, job_id: str) -> List[PipelineStep]:
        """Get all steps for a job"""
        steps = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT step_name, step_order, status, start_time, end_time, 
                       progress_percentage, items_total, items_processed, error_message
                FROM pipeline_steps 
                WHERE job_id = ? 
                ORDER BY step_order
            """, (job_id,))
            
            for row in cursor.fetchall():
                steps.append(PipelineStep(
                    step_name=row[0],
                    step_order=row[1],
                    status=row[2],
                    start_time=row[3],
                    end_time=row[4],
                    progress_percentage=row[5] or 0,
                    items_total=row[6] or 0,
                    items_processed=row[7] or 0,
                    error_message=row[8]
                ))
        
        return steps
    
    def get_step_logs(self, job_id: str, step_name: str = None) -> List[Dict[str, Any]]:
        """Get logs for a specific step or all steps"""
        logs = []
        with sqlite3.connect(self.db_path) as conn:
            if step_name:
                cursor = conn.execute("""
                    SELECT timestamp, log_level, message, step_name
                    FROM pipeline_logs 
                    WHERE job_id = ? AND step_name = ?
                    ORDER BY timestamp
                """, (job_id, step_name))
            else:
                cursor = conn.execute("""
                    SELECT timestamp, log_level, message, step_name
                    FROM pipeline_logs 
                    WHERE job_id = ?
                    ORDER BY timestamp
                """, (job_id,))
            
            for row in cursor.fetchall():
                logs.append({
                    'timestamp': row[0],
                    'level': row[1],
                    'message': row[2],
                    'step_name': row[3],
                    'formatted_time': datetime.fromtimestamp(row[0]).strftime('%H:%M:%S')
                })
        
        return logs
    
    def get_pipeline_visualization_data(self, job_id: str) -> Dict[str, Any]:
        """Get comprehensive pipeline visualization data"""
        job = self.get_job(job_id)
        if not job:
            return None
        
        # Get steps with their logs
        steps_with_logs = []
        for step in job.steps:
            step_logs = self.get_step_logs(job_id, step.step_name)
            steps_with_logs.append({
                'step': step,
                'logs': step_logs
            })
        
        # Get step progression timeline
        timeline = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT step_name, step_order, status, start_time, end_time, progress_percentage
                FROM pipeline_steps 
                WHERE job_id = ?
                ORDER BY step_order
            """, (job_id,))
            
            for row in cursor.fetchall():
                timeline.append({
                    'step_name': row[0],
                    'step_order': row[1],
                    'status': row[2],
                    'start_time': row[3],
                    'end_time': row[4],
                    'progress': row[5] or 0,
                    'start_formatted': datetime.fromtimestamp(row[3]).strftime('%H:%M:%S') if row[3] else None,
                    'end_formatted': datetime.fromtimestamp(row[4]).strftime('%H:%M:%S') if row[4] else None,
                    'duration': (row[4] - row[3]) if (row[3] and row[4]) else None
                })
        
        # Get metrics progression over time
        metrics_history = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT timestamp, current_step, current_step_progress, processed_urls, total_urls
                FROM pipeline_metrics 
                WHERE job_id = ?
                ORDER BY timestamp
            """, (job_id,))
            
            for row in cursor.fetchall():
                metrics_history.append({
                    'timestamp': row[0],
                    'current_step': row[1],
                    'step_progress': row[2],
                    'processed_urls': row[3],
                    'total_urls': row[4],
                    'formatted_time': datetime.fromtimestamp(row[0]).strftime('%H:%M:%S')
                })
        
        return {
            'job': job,
            'steps_with_logs': steps_with_logs,
            'timeline': timeline,
            'metrics_history': metrics_history,
            'summary': {
                'total_duration': (job.end_time - job.start_time) if job.end_time else None,
                'completed_steps': len([s for s in job.steps if s.status == 'completed']),
                'failed_steps': len([s for s in job.steps if s.status == 'failed']),
                'skipped_steps': len([s for s in job.steps if s.status == 'skipped']),
                'total_steps': len(job.steps)
            }
        }
    
    def update_metrics(self, job_id: str, metrics: PipelineMetrics):
        """Update job metrics"""
        # Get current step info
        current_step = self.get_current_step(job_id)
        if current_step:
            metrics.current_step = current_step.step_name
            metrics.current_step_progress = current_step.progress_percentage
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO pipeline_metrics (
                    job_id, timestamp, total_urls, processed_urls, successful_urls,
                    failed_urls, total_chunks, total_embeddings, total_vectors,
                    duplicate_count, current_step, current_step_progress, estimated_completion
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job_id, time.time(), metrics.total_urls, metrics.processed_urls,
                metrics.successful_urls, metrics.failed_urls, metrics.total_chunks,
                metrics.total_embeddings, metrics.total_vectors, metrics.duplicate_count,
                metrics.current_step, metrics.current_step_progress, metrics.estimated_completion
            ))
            conn.commit()
    
    def add_log(self, job_id: str, message: str, level: str = 'INFO', step_name: str = None):
        """Add a log entry for a job"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO pipeline_logs (job_id, timestamp, log_level, message, step_name)
                VALUES (?, ?, ?, ?, ?)
            """, (job_id, time.time(), level, message, step_name))
            conn.commit()
    
    def complete_job(self, job_id: str, success: bool = True, error_message: str = None):
        """Mark a job as completed"""
        status = 'completed' if success else 'failed'
        end_time = time.time()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE pipeline_jobs 
                SET status = ?, end_time = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ?
            """, (status, end_time, error_message, job_id))
            conn.commit()
        
        final_message = f"Pipeline {status}"
        if error_message:
            final_message += f": {error_message}"
        
        self.add_log(job_id, final_message, level='ERROR' if not success else 'INFO')
        logger.info(f"Completed pipeline job: {job_id} - {status}")

    def stop_job(self, job_id: str):
        """Stop a running job"""
        with sqlite3.connect(self.db_path) as conn:
            # Update job status
            conn.execute("""
                UPDATE pipeline_jobs 
                SET status = 'stopped', end_time = ?, updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ?
            """, (time.time(), job_id))
            
            # Update any running steps
            conn.execute("""
                UPDATE pipeline_steps 
                SET status = 'stopped', end_time = ?, updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ? AND status = 'running'
            """, (time.time(), job_id))
            
            conn.commit()
        
        self.add_log(job_id, "Pipeline stopped by user")
        logger.info(f"Stopped pipeline job: {job_id}")

    def get_job(self, job_id: str) -> Optional[PipelineJob]:
        """Get a specific job with all details"""
        with sqlite3.connect(self.db_path) as conn:
            # Get job basic info
            cursor = conn.execute("""
                SELECT job_id, status, start_time, end_time, input_urls, error_message
                FROM pipeline_jobs WHERE job_id = ?
            """, (job_id,))
            job_row = cursor.fetchone()
            
            if not job_row:
                return None
            
            # Get latest metrics
            cursor = conn.execute("""
                SELECT total_urls, processed_urls, successful_urls, failed_urls,
                       total_chunks, total_embeddings, total_vectors, duplicate_count,
                       current_step, current_step_progress, estimated_completion
                FROM pipeline_metrics 
                WHERE job_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """, (job_id,))
            metrics_row = cursor.fetchone()
            
            # Create metrics object
            if metrics_row:
                metrics = PipelineMetrics(
                    total_urls=metrics_row[0] or 0,
                    processed_urls=metrics_row[1] or 0,
                    successful_urls=metrics_row[2] or 0,
                    failed_urls=metrics_row[3] or 0,
                    total_chunks=metrics_row[4] or 0,
                    total_embeddings=metrics_row[5] or 0,
                    total_vectors=metrics_row[6] or 0,
                    duplicate_count=metrics_row[7] or 0,
                    start_time=job_row[2],
                    end_time=job_row[3],
                    current_step=metrics_row[8] or "",
                    current_step_progress=metrics_row[9] or 0,
                    estimated_completion=metrics_row[10]
                )
            else:
                metrics = PipelineMetrics(start_time=job_row[2], end_time=job_row[3])
            
            # Get recent logs
            cursor = conn.execute("""
                SELECT message FROM pipeline_logs 
                WHERE job_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 50
            """, (job_id,))
            logs = [row[0] for row in cursor.fetchall()]
            
            # Get steps
            steps = self.get_job_steps(job_id)
            
            return PipelineJob(
                job_id=job_row[0],
                status=job_row[1],
                start_time=job_row[2],
                end_time=job_row[3],
                input_urls=json.loads(job_row[4] or '[]'),
                metrics=metrics,
                logs=logs,
                steps=steps,
                error_message=job_row[5]
            )

    def get_active_jobs(self) -> List[PipelineJob]:
        """Get all currently active jobs"""
        active_jobs = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT job_id FROM pipeline_jobs 
                WHERE status = 'running' 
                ORDER BY start_time DESC
            """)
            for row in cursor.fetchall():
                job = self.get_job(row[0])
                if job:
                    active_jobs.append(job)
        return active_jobs

    def get_job_history(self, limit: int = 10) -> List[PipelineJob]:
        """Get job history"""
        jobs = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT job_id FROM pipeline_jobs 
                ORDER BY start_time DESC 
                LIMIT ?
            """, (limit,))
            for row in cursor.fetchall():
                job = self.get_job(row[0])
                if job:
                    jobs.append(job)
        return jobs

    def get_job_stats(self) -> Dict[str, Any]:
        """Get overall job statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_jobs,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_jobs,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_jobs,
                    SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as running_jobs,
                    AVG(CASE WHEN end_time IS NOT NULL THEN end_time - start_time END) as avg_duration
                FROM pipeline_jobs
            """)
            row = cursor.fetchone()
            
            cursor = conn.execute("""
                SELECT 
                    SUM(total_vectors) as total_documents_processed,
                    SUM(successful_urls) as total_urls_processed
                FROM pipeline_metrics m
                INNER JOIN (
                    SELECT job_id, MAX(timestamp) as max_timestamp
                    FROM pipeline_metrics
                    GROUP BY job_id
                ) latest ON m.job_id = latest.job_id AND m.timestamp = latest.max_timestamp
            """)
            metrics_row = cursor.fetchone()
            
            return {
                'total_jobs': row[0] or 0,
                'completed_jobs': row[1] or 0,
                'failed_jobs': row[2] or 0,
                'running_jobs': row[3] or 0,
                'success_rate': (row[1] / row[0] * 100) if row[0] > 0 else 0,
                'average_duration_minutes': (row[4] / 60) if row[4] else 0,
                'total_documents_processed': metrics_row[0] or 0,
                'total_urls_processed': metrics_row[1] or 0
            }

    def cleanup_old_jobs(self, days: int = 30):
        """Clean up old job records"""
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        with sqlite3.connect(self.db_path) as conn:
            # Get jobs to delete
            cursor = conn.execute("""
                SELECT job_id FROM pipeline_jobs 
                WHERE start_time < ? AND status IN ('completed', 'failed', 'stopped')
            """, (cutoff_time,))
            old_job_ids = [row[0] for row in cursor.fetchall()]
            
            if old_job_ids:
                # Delete related records
                placeholders = ','.join('?' * len(old_job_ids))
                conn.execute(f"DELETE FROM pipeline_logs WHERE job_id IN ({placeholders})", old_job_ids)
                conn.execute(f"DELETE FROM pipeline_metrics WHERE job_id IN ({placeholders})", old_job_ids)
                conn.execute(f"DELETE FROM pipeline_steps WHERE job_id IN ({placeholders})", old_job_ids)
                conn.execute(f"DELETE FROM pipeline_jobs WHERE job_id IN ({placeholders})", old_job_ids)
                
                conn.commit()
                logger.info(f"Cleaned up {len(old_job_ids)} old jobs")
                
                return len(old_job_ids)
        
        return 0

# Global instance
pipeline_tracker = PipelineTracker() 