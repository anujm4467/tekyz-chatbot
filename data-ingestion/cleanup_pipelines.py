#!/usr/bin/env python3

"""
Script to cleanup pipeline data and close running pipelines
"""

import sqlite3
import time
import argparse
from pathlib import Path
from src.database.pipeline_tracker import PipelineTracker

def close_running_pipelines(db_path: str = "data/pipeline_tracker.db"):
    """Close all running pipelines"""
    print("🔄 Closing all running pipelines...")
    
    tracker = PipelineTracker(db_path)
    
    # Get all running jobs
    running_jobs = tracker.get_active_jobs()
    
    if not running_jobs:
        print("✅ No running pipelines found")
        return
    
    print(f"📋 Found {len(running_jobs)} running pipeline(s)")
    
    # Stop each running job
    for job in running_jobs:
        print(f"  🛑 Stopping job: {job.job_id}")
        tracker.stop_job(job.job_id)
    
    print("✅ All running pipelines have been stopped")

def clear_all_data(db_path: str = "data/pipeline_tracker.db"):
    """Clear all pipeline data from database"""
    print("🗑️  Clearing all pipeline data...")
    
    with sqlite3.connect(db_path) as conn:
        # Get counts before deletion
        cursor = conn.execute("SELECT COUNT(*) FROM pipeline_jobs")
        job_count = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT COUNT(*) FROM pipeline_logs")
        log_count = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT COUNT(*) FROM pipeline_steps")
        step_count = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT COUNT(*) FROM pipeline_metrics")
        metric_count = cursor.fetchone()[0]
        
        print(f"📊 Current data:")
        print(f"  - Jobs: {job_count}")
        print(f"  - Logs: {log_count}")
        print(f"  - Steps: {step_count}")
        print(f"  - Metrics: {metric_count}")
        
        if job_count == 0:
            print("✅ Database is already empty")
            return
        
        # Delete all data
        print("🗑️  Deleting all data...")
        conn.execute("DELETE FROM pipeline_logs")
        conn.execute("DELETE FROM pipeline_metrics")
        conn.execute("DELETE FROM pipeline_steps")
        conn.execute("DELETE FROM pipeline_jobs")
        
        # Reset auto-increment counters
        conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('pipeline_steps', 'pipeline_logs')")
        
        conn.commit()
        
        print("✅ All pipeline data has been cleared")

def clear_old_data(days: int, db_path: str = "data/pipeline_tracker.db"):
    """Clear old completed/failed jobs"""
    print(f"🧹 Clearing jobs older than {days} days...")
    
    tracker = PipelineTracker(db_path)
    deleted_count = tracker.cleanup_old_jobs(days)
    
    print(f"✅ Cleaned up {deleted_count} old jobs")

def show_status(db_path: str = "data/pipeline_tracker.db"):
    """Show current database status"""
    print("📊 Current pipeline status:")
    
    if not Path(db_path).exists():
        print("❌ Database file does not exist")
        return
    
    with sqlite3.connect(db_path) as conn:
        # Job counts by status
        cursor = conn.execute("""
            SELECT status, COUNT(*) 
            FROM pipeline_jobs 
            GROUP BY status
        """)
        
        status_counts = dict(cursor.fetchall())
        
        if not status_counts:
            print("✅ No jobs in database")
            return
        
        print("Jobs by status:")
        for status, count in status_counts.items():
            print(f"  - {status}: {count}")
        
        # Recent jobs
        cursor = conn.execute("""
            SELECT job_id, status, start_time 
            FROM pipeline_jobs 
            ORDER BY start_time DESC 
            LIMIT 5
        """)
        
        recent_jobs = cursor.fetchall()
        
        if recent_jobs:
            print("\n🕒 Recent jobs:")
            for job_id, status, start_time in recent_jobs:
                start_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
                print(f"  - {job_id[:8]}... ({status}) - {start_str}")

def main():
    parser = argparse.ArgumentParser(description='Cleanup pipeline data')
    parser.add_argument('--db', default='data/pipeline_tracker.db', help='Database path')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--status', action='store_true', help='Show current status')
    group.add_argument('--close', action='store_true', help='Close running pipelines only')
    group.add_argument('--clear-old', type=int, metavar='DAYS', help='Clear jobs older than N days')
    group.add_argument('--clear-all', action='store_true', help='Clear ALL data (destructive!)')
    
    args = parser.parse_args()
    
    print("🚀 Pipeline Cleanup Tool")
    print("=" * 40)
    
    if args.status:
        show_status(args.db)
    
    elif args.close:
        close_running_pipelines(args.db)
        print("\n📊 Updated status:")
        show_status(args.db)
    
    elif args.clear_old:
        clear_old_data(args.clear_old, args.db)
        print("\n📊 Updated status:")
        show_status(args.db)
    
    elif args.clear_all:
        print("⚠️  WARNING: This will delete ALL pipeline data!")
        print("This action cannot be undone.")
        
        confirm = input("Type 'yes' to confirm: ")
        if confirm.lower() == 'yes':
            clear_all_data(args.db)
            print("\n📊 Updated status:")
            show_status(args.db)
        else:
            print("❌ Operation cancelled")

if __name__ == "__main__":
    main() 