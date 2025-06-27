#!/usr/bin/env python3
"""
Performance Monitor for Tekyz Data Ingestion Pipeline

Real-time monitoring of pipeline performance, resource usage, and system metrics.
"""

import time
import psutil
import threading
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque
import logging

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int


@dataclass
class PipelineMetrics:
    """Pipeline-specific performance metrics."""
    timestamp: datetime
    component: str
    operation: str
    duration_seconds: float
    items_processed: int
    throughput_per_second: float
    memory_peak_mb: float
    error_count: int
    success_rate: float


@dataclass
class PerformanceAlert:
    """Performance alert for threshold violations."""
    timestamp: datetime
    severity: str  # 'warning', 'critical'
    component: str
    metric: str
    current_value: float
    threshold_value: float
    message: str


class PerformanceMonitor:
    """
    Real-time performance monitoring for the data ingestion pipeline.
    
    Features:
    - System resource monitoring
    - Pipeline component metrics
    - Performance alerting
    - Historical data tracking
    - Report generation
    """
    
    def __init__(self, 
                 monitoring_interval: float = 1.0,
                 history_size: int = 1000,
                 output_dir: Path = None):
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.output_dir = output_dir or Path("monitoring_data")
        self.output_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.system_metrics_history = deque(maxlen=history_size)
        self.pipeline_metrics_history = deque(maxlen=history_size)
        self.alerts_history = deque(maxlen=history_size)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Performance thresholds
        self.thresholds = {
            'cpu_percent': {'warning': 80.0, 'critical': 95.0},
            'memory_percent': {'warning': 85.0, 'critical': 95.0},
            'disk_usage_percent': {'warning': 85.0, 'critical': 95.0},
            'throughput_per_second': {'warning': 1.0, 'critical': 0.1},  # Minimum expected
            'error_rate': {'warning': 0.05, 'critical': 0.1},  # Maximum acceptable
            'response_time': {'warning': 5.0, 'critical': 10.0}  # Seconds
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.output_dir / 'performance_monitor.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        if self.is_monitoring:
            self.logger.warning("Monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Performance monitoring started")
        print("🔍 Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        if not self.is_monitoring:
            self.logger.warning("Monitoring is not running")
            return
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("Performance monitoring stopped")
        print("⏹️ Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.system_metrics_history.append(system_metrics)
                
                # Check for alerts
                self._check_system_alerts(system_metrics)
                
                # Save periodic snapshots
                if len(self.system_metrics_history) % 60 == 0:  # Every minute
                    self._save_snapshot()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage_percent = (disk.used / disk.total) * 100
        disk_free_gb = disk.free / (1024**3)
        
        # Network I/O
        network = psutil.net_io_counters()
        network_bytes_sent = network.bytes_sent
        network_bytes_recv = network.bytes_recv
        
        # Process count
        process_count = len(psutil.pids())
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            memory_total_gb=memory_total_gb,
            disk_usage_percent=disk_usage_percent,
            disk_free_gb=disk_free_gb,
            network_bytes_sent=network_bytes_sent,
            network_bytes_recv=network_bytes_recv,
            process_count=process_count
        )
    
    def record_pipeline_metric(self,
                             component: str,
                             operation: str,
                             duration_seconds: float,
                             items_processed: int = 1,
                             memory_peak_mb: float = 0.0,
                             error_count: int = 0):
        """Record a pipeline performance metric."""
        throughput = items_processed / duration_seconds if duration_seconds > 0 else 0
        success_rate = (items_processed - error_count) / items_processed if items_processed > 0 else 0
        
        metric = PipelineMetrics(
            timestamp=datetime.now(),
            component=component,
            operation=operation,
            duration_seconds=duration_seconds,
            items_processed=items_processed,
            throughput_per_second=throughput,
            memory_peak_mb=memory_peak_mb,
            error_count=error_count,
            success_rate=success_rate
        )
        
        self.pipeline_metrics_history.append(metric)
        
        # Check for pipeline alerts
        self._check_pipeline_alerts(metric)
        
        self.logger.info(f"Pipeline metric recorded: {component}.{operation} - "
                        f"{duration_seconds:.2f}s, {throughput:.2f} items/s")
    
    def _check_system_alerts(self, metrics: SystemMetrics):
        """Check system metrics against thresholds."""
        checks = [
            ('cpu_percent', metrics.cpu_percent, 'CPU Usage'),
            ('memory_percent', metrics.memory_percent, 'Memory Usage'),
            ('disk_usage_percent', metrics.disk_usage_percent, 'Disk Usage')
        ]
        
        for metric_name, value, display_name in checks:
            thresholds = self.thresholds.get(metric_name, {})
            
            if value >= thresholds.get('critical', float('inf')):
                self._create_alert('critical', 'system', metric_name, value, 
                                 thresholds['critical'], 
                                 f"{display_name} is critically high: {value:.1f}%")
            elif value >= thresholds.get('warning', float('inf')):
                self._create_alert('warning', 'system', metric_name, value,
                                 thresholds['warning'],
                                 f"{display_name} is high: {value:.1f}%")
    
    def _check_pipeline_alerts(self, metric: PipelineMetrics):
        """Check pipeline metrics against thresholds."""
        # Check throughput
        if metric.throughput_per_second < self.thresholds['throughput_per_second']['critical']:
            self._create_alert('critical', metric.component, 'throughput', 
                             metric.throughput_per_second,
                             self.thresholds['throughput_per_second']['critical'],
                             f"Very low throughput in {metric.component}.{metric.operation}: "
                             f"{metric.throughput_per_second:.2f} items/s")
        elif metric.throughput_per_second < self.thresholds['throughput_per_second']['warning']:
            self._create_alert('warning', metric.component, 'throughput',
                             metric.throughput_per_second,
                             self.thresholds['throughput_per_second']['warning'],
                             f"Low throughput in {metric.component}.{metric.operation}: "
                             f"{metric.throughput_per_second:.2f} items/s")
        
        # Check error rate
        error_rate = 1.0 - metric.success_rate
        if error_rate >= self.thresholds['error_rate']['critical']:
            self._create_alert('critical', metric.component, 'error_rate',
                             error_rate, self.thresholds['error_rate']['critical'],
                             f"High error rate in {metric.component}.{metric.operation}: "
                             f"{error_rate:.1%}")
        elif error_rate >= self.thresholds['error_rate']['warning']:
            self._create_alert('warning', metric.component, 'error_rate',
                             error_rate, self.thresholds['error_rate']['warning'],
                             f"Elevated error rate in {metric.component}.{metric.operation}: "
                             f"{error_rate:.1%}")
        
        # Check response time
        if metric.duration_seconds >= self.thresholds['response_time']['critical']:
            self._create_alert('critical', metric.component, 'response_time',
                             metric.duration_seconds, self.thresholds['response_time']['critical'],
                             f"Very slow response in {metric.component}.{metric.operation}: "
                             f"{metric.duration_seconds:.2f}s")
        elif metric.duration_seconds >= self.thresholds['response_time']['warning']:
            self._create_alert('warning', metric.component, 'response_time',
                             metric.duration_seconds, self.thresholds['response_time']['warning'],
                             f"Slow response in {metric.component}.{metric.operation}: "
                             f"{metric.duration_seconds:.2f}s")
    
    def _create_alert(self, severity: str, component: str, metric: str,
                     current_value: float, threshold_value: float, message: str):
        """Create and log a performance alert."""
        alert = PerformanceAlert(
            timestamp=datetime.now(),
            severity=severity,
            component=component,
            metric=metric,
            current_value=current_value,
            threshold_value=threshold_value,
            message=message
        )
        
        self.alerts_history.append(alert)
        
        # Log alert
        log_level = logging.CRITICAL if severity == 'critical' else logging.WARNING
        self.logger.log(log_level, f"ALERT [{severity.upper()}]: {message}")
        
        # Print to console
        icon = "🚨" if severity == 'critical' else "⚠️"
        print(f"{icon} {severity.upper()}: {message}")
    
    def _save_snapshot(self):
        """Save current monitoring data snapshot."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare data
        snapshot_data = {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': [asdict(m) for m in list(self.system_metrics_history)],
            'pipeline_metrics': [asdict(m) for m in list(self.pipeline_metrics_history)],
            'alerts': [asdict(a) for a in list(self.alerts_history)]
        }
        
        # Save to file
        snapshot_file = self.output_dir / f'snapshot_{timestamp}.json'
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot_data, f, indent=2, default=str)
        
        self.logger.info(f"Monitoring snapshot saved: {snapshot_file}")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        if not self.system_metrics_history:
            return {'error': 'No monitoring data available'}
        
        # Latest system metrics
        latest_system = self.system_metrics_history[-1]
        
        # Recent pipeline metrics (last 5 minutes)
        recent_cutoff = datetime.now() - timedelta(minutes=5)
        recent_pipeline = [m for m in self.pipeline_metrics_history 
                          if m.timestamp >= recent_cutoff]
        
        # Recent alerts (last hour)
        alert_cutoff = datetime.now() - timedelta(hours=1)
        recent_alerts = [a for a in self.alerts_history 
                        if a.timestamp >= alert_cutoff]
        
        # Calculate averages
        avg_throughput = (sum(m.throughput_per_second for m in recent_pipeline) / 
                         len(recent_pipeline)) if recent_pipeline else 0
        
        avg_response_time = (sum(m.duration_seconds for m in recent_pipeline) / 
                           len(recent_pipeline)) if recent_pipeline else 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'cpu_percent': latest_system.cpu_percent,
                'memory_percent': latest_system.memory_percent,
                'memory_used_gb': latest_system.memory_used_gb,
                'disk_usage_percent': latest_system.disk_usage_percent,
                'disk_free_gb': latest_system.disk_free_gb
            },
            'pipeline': {
                'avg_throughput_5min': avg_throughput,
                'avg_response_time_5min': avg_response_time,
                'operations_count_5min': len(recent_pipeline),
                'total_errors_5min': sum(m.error_count for m in recent_pipeline)
            },
            'alerts': {
                'total_alerts_1hour': len(recent_alerts),
                'critical_alerts_1hour': len([a for a in recent_alerts if a.severity == 'critical']),
                'warning_alerts_1hour': len([a for a in recent_alerts if a.severity == 'warning'])
            }
        }
    
    def generate_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter data by time range
        system_data = [m for m in self.system_metrics_history if m.timestamp >= cutoff_time]
        pipeline_data = [m for m in self.pipeline_metrics_history if m.timestamp >= cutoff_time]
        alert_data = [a for a in self.alerts_history if a.timestamp >= cutoff_time]
        
        if not system_data:
            return {'error': f'No data available for the last {hours} hours'}
        
        # System statistics
        cpu_values = [m.cpu_percent for m in system_data]
        memory_values = [m.memory_percent for m in system_data]
        
        system_stats = {
            'cpu': {
                'avg': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            },
            'memory': {
                'avg': sum(memory_values) / len(memory_values),
                'max': max(memory_values),
                'min': min(memory_values)
            }
        }
        
        # Pipeline statistics
        pipeline_stats = {}
        if pipeline_data:
            components = set(m.component for m in pipeline_data)
            for component in components:
                component_data = [m for m in pipeline_data if m.component == component]
                throughput_values = [m.throughput_per_second for m in component_data]
                duration_values = [m.duration_seconds for m in component_data]
                
                pipeline_stats[component] = {
                    'operations_count': len(component_data),
                    'avg_throughput': sum(throughput_values) / len(throughput_values),
                    'max_throughput': max(throughput_values),
                    'avg_duration': sum(duration_values) / len(duration_values),
                    'max_duration': max(duration_values),
                    'total_errors': sum(m.error_count for m in component_data)
                }
        
        # Alert statistics
        alert_stats = {
            'total_alerts': len(alert_data),
            'critical_alerts': len([a for a in alert_data if a.severity == 'critical']),
            'warning_alerts': len([a for a in alert_data if a.severity == 'warning']),
            'alerts_by_component': {}
        }
        
        # Group alerts by component
        for alert in alert_data:
            component = alert.component
            if component not in alert_stats['alerts_by_component']:
                alert_stats['alerts_by_component'][component] = {'critical': 0, 'warning': 0}
            alert_stats['alerts_by_component'][component][alert.severity] += 1
        
        report = {
            'report_period_hours': hours,
            'generated_at': datetime.now().isoformat(),
            'data_points': len(system_data),
            'system_performance': system_stats,
            'pipeline_performance': pipeline_stats,
            'alerts_summary': alert_stats
        }
        
        # Save report
        report_file = self.output_dir / f'performance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Performance report generated: {report_file}")
        
        return report
    
    def print_current_status(self):
        """Print current monitoring status to console."""
        stats = self.get_current_stats()
        
        if 'error' in stats:
            print(f"❌ {stats['error']}")
            return
        
        print("\n" + "="*50)
        print("📊 PERFORMANCE MONITOR STATUS")
        print("="*50)
        print(f"Timestamp: {stats['timestamp']}")
        
        print("\n🖥️ System Resources:")
        system = stats['system']
        print(f"  CPU Usage: {system['cpu_percent']:.1f}%")
        print(f"  Memory Usage: {system['memory_percent']:.1f}% ({system['memory_used_gb']:.1f} GB)")
        print(f"  Disk Usage: {system['disk_usage_percent']:.1f}% ({system['disk_free_gb']:.1f} GB free)")
        
        print("\n⚡ Pipeline Performance (Last 5 min):")
        pipeline = stats['pipeline']
        print(f"  Average Throughput: {pipeline['avg_throughput_5min']:.2f} items/sec")
        print(f"  Average Response Time: {pipeline['avg_response_time_5min']:.2f} seconds")
        print(f"  Operations Count: {pipeline['operations_count_5min']}")
        print(f"  Total Errors: {pipeline['total_errors_5min']}")
        
        print("\n🚨 Alerts (Last 1 hour):")
        alerts = stats['alerts']
        print(f"  Total Alerts: {alerts['total_alerts_1hour']}")
        print(f"  Critical: {alerts['critical_alerts_1hour']}")
        print(f"  Warnings: {alerts['warning_alerts_1hour']}")
        
        print("="*50)


class PerformanceContext:
    """Context manager for tracking operation performance."""
    
    def __init__(self, monitor: PerformanceMonitor, component: str, operation: str):
        self.monitor = monitor
        self.component = component
        self.operation = operation
        self.start_time = None
        self.start_memory = None
        self.items_processed = 0
        self.error_count = 0
    
    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        current_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
        memory_peak = max(current_memory, self.start_memory)
        
        if exc_type is not None:
            self.error_count += 1
        
        self.monitor.record_pipeline_metric(
            component=self.component,
            operation=self.operation,
            duration_seconds=duration,
            items_processed=max(1, self.items_processed),
            memory_peak_mb=memory_peak,
            error_count=self.error_count
        )
    
    def add_processed_items(self, count: int):
        """Add to the count of processed items."""
        self.items_processed += count
    
    def add_error(self):
        """Add to the error count."""
        self.error_count += 1


def main():
    """Main function for standalone monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Tekyz Performance Monitor')
    parser.add_argument('--interval', type=float, default=1.0, 
                       help='Monitoring interval in seconds')
    parser.add_argument('--output-dir', type=str, default='monitoring_data',
                       help='Output directory for monitoring data')
    parser.add_argument('--duration', type=int, default=0,
                       help='Monitoring duration in seconds (0 = indefinite)')
    parser.add_argument('--report', action='store_true',
                       help='Generate performance report and exit')
    parser.add_argument('--report-hours', type=int, default=24,
                       help='Hours of data to include in report')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = PerformanceMonitor(
        monitoring_interval=args.interval,
        output_dir=Path(args.output_dir)
    )
    
    if args.report:
        # Generate report and exit
        print("📊 Generating performance report...")
        report = monitor.generate_performance_report(args.report_hours)
        
        if 'error' in report:
            print(f"❌ {report['error']}")
            return
        
        print(f"✅ Report generated for last {args.report_hours} hours")
        print(f"📄 Data points: {report['data_points']}")
        print(f"🚨 Total alerts: {report['alerts_summary']['total_alerts']}")
        
        return
    
    # Start monitoring
    try:
        monitor.start_monitoring()
        
        if args.duration > 0:
            print(f"⏱️ Monitoring for {args.duration} seconds...")
            time.sleep(args.duration)
        else:
            print("⏱️ Monitoring indefinitely (Ctrl+C to stop)...")
            while True:
                time.sleep(10)
                monitor.print_current_status()
                
    except KeyboardInterrupt:
        print("\n⏹️ Stopping monitoring...")
    finally:
        monitor.stop_monitoring()
        
        # Generate final report
        print("📊 Generating final report...")
        report = monitor.generate_performance_report(1)  # Last hour
        if 'error' not in report:
            print(f"✅ Final report generated with {report['data_points']} data points")


if __name__ == '__main__':
    main() 