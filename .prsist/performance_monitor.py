#!/usr/bin/env python3
"""
Performance monitoring system for Prsist Memory System Phase 3.
Monitors system performance, resource usage, and provides alerts.
"""

import logging
import psutil
import time
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import threading
from dataclasses import dataclass, asdict
from collections import deque, defaultdict

from database import MemoryDatabase
from utils import setup_logging

@dataclass
class PerformanceMetric:
    """Represents a performance metric."""
    metric_type: str
    value: float
    timestamp: datetime
    component: str
    context: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

@dataclass
class AlertThreshold:
    """Represents an alert threshold."""
    metric_type: str
    warning_value: float
    critical_value: float = None
    comparison: str = "greater"  # greater, less, equal
    window_size: int = 1  # Number of measurements to consider
    
class PerformanceAlert:
    """Represents a performance alert."""
    
    def __init__(self, alert_type: str, level: str, message: str, 
                 metric_value: float, threshold_value: float, 
                 component: str = None, context: Dict[str, Any] = None):
        self.alert_type = alert_type
        self.level = level  # warning, critical
        self.message = message
        self.metric_value = metric_value
        self.threshold_value = threshold_value
        self.component = component
        self.context = context or {}
        self.timestamp = datetime.now()
        self.alert_id = f"{alert_type}_{int(self.timestamp.timestamp())}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type,
            'level': self.level,
            'message': self.message,
            'metric_value': self.metric_value,
            'threshold_value': self.threshold_value,
            'component': self.component,
            'context': self.context,
            'timestamp': self.timestamp.isoformat()
        }

class MetricCollector:
    """Collects various performance metrics."""
    
    def __init__(self, memory_db: MemoryDatabase):
        self.memory_db = memory_db
        self.process = psutil.Process()
        self.start_time = datetime.now()
        
    def collect_system_metrics(self) -> List[PerformanceMetric]:
        """Collect system-level performance metrics."""
        metrics = []
        now = datetime.now()
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            metrics.append(PerformanceMetric(
                metric_type="cpu_usage_percent",
                value=cpu_percent,
                timestamp=now,
                component="system"
            ))
            
            # Memory usage
            memory_info = psutil.virtual_memory()
            metrics.append(PerformanceMetric(
                metric_type="system_memory_usage_percent",
                value=memory_info.percent,
                timestamp=now,
                component="system",
                context={"available_gb": memory_info.available / (1024**3)}
            ))
            
            # Disk usage
            disk_usage = psutil.disk_usage('.')
            metrics.append(PerformanceMetric(
                metric_type="disk_usage_percent",
                value=(disk_usage.used / disk_usage.total) * 100,
                timestamp=now,
                component="system",
                context={"free_gb": disk_usage.free / (1024**3)}
            ))
            
        except Exception as e:
            logging.error(f"Failed to collect system metrics: {e}")
        
        return metrics
    
    def collect_process_metrics(self) -> List[PerformanceMetric]:
        """Collect process-specific performance metrics."""
        metrics = []
        now = datetime.now()
        
        try:
            # Process memory usage
            memory_info = self.process.memory_info()
            metrics.append(PerformanceMetric(
                metric_type="process_memory_usage_mb",
                value=memory_info.rss / (1024 * 1024),
                timestamp=now,
                component="memory_system",
                context={"vms_mb": memory_info.vms / (1024 * 1024)}
            ))
            
            # Process CPU usage
            cpu_percent = self.process.cpu_percent()
            metrics.append(PerformanceMetric(
                metric_type="process_cpu_usage_percent",
                value=cpu_percent,
                timestamp=now,
                component="memory_system"
            ))
            
            # Number of threads
            num_threads = self.process.num_threads()
            metrics.append(PerformanceMetric(
                metric_type="process_thread_count",
                value=num_threads,
                timestamp=now,
                component="memory_system"
            ))
            
            # Process uptime
            uptime_seconds = (now - self.start_time).total_seconds()
            metrics.append(PerformanceMetric(
                metric_type="process_uptime_hours",
                value=uptime_seconds / 3600,
                timestamp=now,
                component="memory_system"
            ))
            
            # Open file descriptors (Unix-like systems)
            try:
                num_fds = self.process.num_fds()
                metrics.append(PerformanceMetric(
                    metric_type="process_open_files",
                    value=num_fds,
                    timestamp=now,
                    component="memory_system"
                ))
            except (AttributeError, psutil.AccessDenied):
                # Not available on Windows or access denied
                pass
            
        except Exception as e:
            logging.error(f"Failed to collect process metrics: {e}")
        
        return metrics
    
    def collect_database_metrics(self) -> List[PerformanceMetric]:
        """Collect database performance metrics."""
        metrics = []
        now = datetime.now()
        
        try:
            # Database file size
            db_path = Path(self.memory_db.db_path)
            if db_path.exists():
                db_size_mb = db_path.stat().st_size / (1024 * 1024)
                metrics.append(PerformanceMetric(
                    metric_type="database_size_mb",
                    value=db_size_mb,
                    timestamp=now,
                    component="database"
                ))
            
            # Count records in major tables
            table_counts = self._get_table_counts()
            for table_name, count in table_counts.items():
                metrics.append(PerformanceMetric(
                    metric_type=f"table_row_count_{table_name}",
                    value=count,
                    timestamp=now,
                    component="database",
                    context={"table": table_name}
                ))
            
            # Active sessions count
            active_sessions = len(self.memory_db.get_active_sessions())
            metrics.append(PerformanceMetric(
                metric_type="active_sessions_count",
                value=active_sessions,
                timestamp=now,
                component="database"
            ))
            
        except Exception as e:
            logging.error(f"Failed to collect database metrics: {e}")
        
        return metrics
    
    def _get_table_counts(self) -> Dict[str, int]:
        """Get row counts for major tables."""
        counts = {}
        major_tables = [
            'sessions', 'tool_usage', 'file_interactions', 'context_entries',
            'git_commits', 'git_file_changes', 'context_snapshots', 
            'file_relevance', 'performance_metrics'
        ]
        
        try:
            import sqlite3
            with sqlite3.connect(self.memory_db.db_path) as conn:
                for table in major_tables:
                    try:
                        cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                        counts[table] = cursor.fetchone()[0]
                    except sqlite3.OperationalError:
                        # Table might not exist
                        counts[table] = 0
        except Exception as e:
            logging.error(f"Failed to get table counts: {e}")
        
        return counts

class PerformanceAnalyzer:
    """Analyzes performance metrics and detects trends."""
    
    def __init__(self, memory_db: MemoryDatabase):
        self.memory_db = memory_db
        self.metric_history = defaultdict(lambda: deque(maxlen=100))
        
    def analyze_metrics(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Analyze performance metrics for trends and anomalies."""
        analysis = {
            'trends': {},
            'anomalies': [],
            'recommendations': [],
            'health_score': 1.0
        }
        
        try:
            # Update metric history
            for metric in metrics:
                self.metric_history[metric.metric_type].append(metric)
            
            # Analyze trends for each metric type
            for metric_type, history in self.metric_history.items():
                if len(history) >= 2:
                    trend_analysis = self._analyze_trend(metric_type, history)
                    analysis['trends'][metric_type] = trend_analysis
            
            # Detect anomalies
            anomalies = self._detect_anomalies(metrics)
            analysis['anomalies'] = anomalies
            
            # Generate recommendations
            recommendations = self._generate_recommendations(analysis)
            analysis['recommendations'] = recommendations
            
            # Calculate overall health score
            health_score = self._calculate_health_score(analysis)
            analysis['health_score'] = health_score
            
        except Exception as e:
            logging.error(f"Failed to analyze metrics: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _analyze_trend(self, metric_type: str, history: deque) -> Dict[str, Any]:
        """Analyze trend for a specific metric."""
        if len(history) < 2:
            return {'trend': 'unknown', 'change_rate': 0.0}
        
        try:
            recent_values = [m.value for m in list(history)[-10:]]  # Last 10 values
            
            if len(recent_values) < 2:
                return {'trend': 'unknown', 'change_rate': 0.0}
            
            # Simple linear trend analysis
            x = list(range(len(recent_values)))
            y = recent_values
            
            # Calculate slope (simplified)
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(xi * yi for xi, yi in zip(x, y))
            sum_x2 = sum(xi * xi for xi in x)
            
            if n * sum_x2 - sum_x * sum_x == 0:
                slope = 0
            else:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            # Determine trend direction
            if abs(slope) < 0.01:
                trend = 'stable'
            elif slope > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'
            
            # Calculate percentage change rate
            if recent_values[0] != 0:
                change_rate = ((recent_values[-1] - recent_values[0]) / recent_values[0]) * 100
            else:
                change_rate = 0.0
            
            return {
                'trend': trend,
                'change_rate': change_rate,
                'slope': slope,
                'current_value': recent_values[-1],
                'average_value': sum(recent_values) / len(recent_values)
            }
            
        except Exception as e:
            logging.error(f"Failed to analyze trend for {metric_type}: {e}")
            return {'trend': 'error', 'change_rate': 0.0, 'error': str(e)}
    
    def _detect_anomalies(self, metrics: List[PerformanceMetric]) -> List[Dict[str, Any]]:
        """Detect anomalies in metrics."""
        anomalies = []
        
        for metric in metrics:
            try:
                history = self.metric_history[metric.metric_type]
                if len(history) < 10:  # Need enough history
                    continue
                
                # Calculate statistics
                values = [m.value for m in history]
                mean_value = sum(values) / len(values)
                variance = sum((v - mean_value) ** 2 for v in values) / len(values)
                std_dev = variance ** 0.5
                
                if std_dev == 0:
                    continue
                
                # Z-score anomaly detection
                z_score = abs(metric.value - mean_value) / std_dev
                
                if z_score > 3:  # More than 3 standard deviations
                    anomalies.append({
                        'metric_type': metric.metric_type,
                        'current_value': metric.value,
                        'expected_value': mean_value,
                        'z_score': z_score,
                        'severity': 'high' if z_score > 4 else 'medium',
                        'component': metric.component,
                        'timestamp': metric.timestamp.isoformat()
                    })
                    
            except Exception as e:
                logging.error(f"Failed to detect anomalies for {metric.metric_type}: {e}")
        
        return anomalies
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        try:
            trends = analysis.get('trends', {})
            anomalies = analysis.get('anomalies', [])
            
            # Memory usage recommendations
            if 'process_memory_usage_mb' in trends:
                memory_trend = trends['process_memory_usage_mb']
                if memory_trend['trend'] == 'increasing' and memory_trend['change_rate'] > 20:
                    recommendations.append(
                        "Memory usage is increasing rapidly. Consider enabling auto-compression "
                        "or reducing context cache size."
                    )
            
            # CPU usage recommendations
            if 'process_cpu_usage_percent' in trends:
                cpu_trend = trends['process_cpu_usage_percent']
                if cpu_trend['current_value'] > 80:
                    recommendations.append(
                        "High CPU usage detected. Consider reducing file watch frequency "
                        "or disabling expensive features."
                    )
            
            # Database size recommendations
            if 'database_size_mb' in trends:
                db_trend = trends['database_size_mb']
                if db_trend['current_value'] > 1000:  # 1GB
                    recommendations.append(
                        "Database size is large. Consider running cleanup operations "
                        "or archiving old data."
                    )
            
            # Thread count recommendations
            if 'process_thread_count' in trends:
                thread_trend = trends['process_thread_count']
                if thread_trend['current_value'] > 50:
                    recommendations.append(
                        "High thread count detected. Check for thread leaks "
                        "or reduce thread pool sizes."
                    )
            
            # Anomaly-based recommendations
            for anomaly in anomalies:
                if anomaly['severity'] == 'high':
                    recommendations.append(
                        f"Anomaly detected in {anomaly['metric_type']}: "
                        f"value is {anomaly['z_score']:.1f} standard deviations "
                        f"from normal. Investigate {anomaly['component']} component."
                    )
            
        except Exception as e:
            logging.error(f"Failed to generate recommendations: {e}")
            recommendations.append(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def _calculate_health_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall system health score (0.0 to 1.0)."""
        try:
            health_score = 1.0
            
            # Penalize for anomalies
            anomalies = analysis.get('anomalies', [])
            for anomaly in anomalies:
                if anomaly['severity'] == 'high':
                    health_score -= 0.2
                elif anomaly['severity'] == 'medium':
                    health_score -= 0.1
            
            # Penalize for concerning trends
            trends = analysis.get('trends', {})
            
            # Memory usage trend
            if 'process_memory_usage_mb' in trends:
                memory_trend = trends['process_memory_usage_mb']
                if memory_trend['trend'] == 'increasing' and memory_trend['change_rate'] > 50:
                    health_score -= 0.3
            
            # CPU usage trend
            if 'process_cpu_usage_percent' in trends:
                cpu_trend = trends['process_cpu_usage_percent']
                if cpu_trend['current_value'] > 90:
                    health_score -= 0.2
            
            return max(0.0, health_score)
            
        except Exception as e:
            logging.error(f"Failed to calculate health score: {e}")
            return 0.5

class AlertManager:
    """Manages performance alerts and notifications."""
    
    def __init__(self, memory_db: MemoryDatabase, config: Dict[str, Any]):
        self.memory_db = memory_db
        self.config = config
        self.thresholds = self._load_thresholds()
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        
    def _load_thresholds(self) -> Dict[str, AlertThreshold]:
        """Load alert thresholds from configuration."""
        thresholds = {}
        
        try:
            alert_config = self.config.get('monitoring', {}).get('alerts', {})
            threshold_config = alert_config.get('thresholds', {})
            
            # Define default thresholds
            default_thresholds = {
                'process_memory_usage_mb': AlertThreshold(
                    'process_memory_usage_mb', 400, 500, 'greater'
                ),
                'process_cpu_usage_percent': AlertThreshold(
                    'process_cpu_usage_percent', 80, 95, 'greater'
                ),
                'database_size_mb': AlertThreshold(
                    'database_size_mb', 500, 1000, 'greater'
                ),
                'active_sessions_count': AlertThreshold(
                    'active_sessions_count', 50, 100, 'greater'
                ),
                'queue_size': AlertThreshold(
                    'queue_size', 500, 800, 'greater'
                ),
                'error_rate_percent': AlertThreshold(
                    'error_rate_percent', 10, 25, 'greater'
                )
            }
            
            # Override with config values
            for metric_type, default_threshold in default_thresholds.items():
                warning_value = threshold_config.get(f'{metric_type}_warning', default_threshold.warning_value)
                critical_value = threshold_config.get(f'{metric_type}_critical', default_threshold.critical_value)
                
                thresholds[metric_type] = AlertThreshold(
                    metric_type, warning_value, critical_value, default_threshold.comparison
                )
            
        except Exception as e:
            logging.error(f"Failed to load alert thresholds: {e}")
        
        return thresholds
    
    def check_alerts(self, metrics: List[PerformanceMetric]) -> List[PerformanceAlert]:
        """Check metrics against thresholds and generate alerts."""
        new_alerts = []
        
        for metric in metrics:
            try:
                threshold = self.thresholds.get(metric.metric_type)
                if not threshold:
                    continue
                
                alerts = self._check_metric_threshold(metric, threshold)
                new_alerts.extend(alerts)
                
            except Exception as e:
                logging.error(f"Failed to check alerts for {metric.metric_type}: {e}")
        
        # Update active alerts
        self._update_active_alerts(new_alerts)
        
        return new_alerts
    
    def _check_metric_threshold(self, metric: PerformanceMetric, 
                               threshold: AlertThreshold) -> List[PerformanceAlert]:
        """Check a single metric against its threshold."""
        alerts = []
        
        try:
            value = metric.value
            
            # Determine if threshold is exceeded
            exceeded = False
            if threshold.comparison == 'greater':
                exceeded = value > threshold.warning_value
            elif threshold.comparison == 'less':
                exceeded = value < threshold.warning_value
            elif threshold.comparison == 'equal':
                exceeded = abs(value - threshold.warning_value) < 0.01
            
            if exceeded:
                # Determine alert level
                level = 'warning'
                threshold_value = threshold.warning_value
                
                if threshold.critical_value is not None:
                    if threshold.comparison == 'greater' and value > threshold.critical_value:
                        level = 'critical'
                        threshold_value = threshold.critical_value
                    elif threshold.comparison == 'less' and value < threshold.critical_value:
                        level = 'critical'
                        threshold_value = threshold.critical_value
                
                # Create alert
                alert = PerformanceAlert(
                    alert_type=metric.metric_type,
                    level=level,
                    message=self._generate_alert_message(metric, threshold, level),
                    metric_value=value,
                    threshold_value=threshold_value,
                    component=metric.component,
                    context=metric.context
                )
                
                alerts.append(alert)
                
        except Exception as e:
            logging.error(f"Failed to check threshold for {metric.metric_type}: {e}")
        
        return alerts
    
    def _generate_alert_message(self, metric: PerformanceMetric, 
                               threshold: AlertThreshold, level: str) -> str:
        """Generate alert message."""
        try:
            component_str = f" in {metric.component}" if metric.component else ""
            value_str = f"{metric.value:.2f}"
            threshold_str = f"{threshold.warning_value:.2f}"
            
            if level == 'critical' and threshold.critical_value:
                threshold_str = f"{threshold.critical_value:.2f}"
            
            return (f"{level.title()}: {metric.metric_type}{component_str} "
                   f"is {value_str} (threshold: {threshold_str})")
                   
        except Exception as e:
            logging.error(f"Failed to generate alert message: {e}")
            return f"{level.title()}: {metric.metric_type} threshold exceeded"
    
    def _update_active_alerts(self, new_alerts: List[PerformanceAlert]):
        """Update active alerts tracking."""
        try:
            # Add new alerts to active alerts
            for alert in new_alerts:
                alert_key = f"{alert.alert_type}_{alert.component}_{alert.level}"
                self.active_alerts[alert_key] = alert
                self.alert_history.append(alert)
            
            # Remove resolved alerts (simplified - just clear old ones)
            current_time = datetime.now()
            resolved_keys = []
            
            for alert_key, alert in self.active_alerts.items():
                # Consider alert resolved if it's older than 5 minutes
                if (current_time - alert.timestamp).total_seconds() > 300:
                    resolved_keys.append(alert_key)
            
            for key in resolved_keys:
                del self.active_alerts[key]
                
        except Exception as e:
            logging.error(f"Failed to update active alerts: {e}")
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get currently active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics."""
        try:
            active_alerts = list(self.active_alerts.values())
            recent_alerts = [a for a in self.alert_history 
                           if (datetime.now() - a.timestamp).total_seconds() < 3600]  # Last hour
            
            return {
                'active_alerts_count': len(active_alerts),
                'critical_alerts_count': len([a for a in active_alerts if a.level == 'critical']),
                'warning_alerts_count': len([a for a in active_alerts if a.level == 'warning']),
                'recent_alerts_count': len(recent_alerts),
                'active_alerts': [a.to_dict() for a in active_alerts],
                'alert_types': list(set(a.alert_type for a in recent_alerts))
            }
            
        except Exception as e:
            logging.error(f"Failed to get alert summary: {e}")
            return {'error': str(e)}

class PerformanceMonitor:
    """Main performance monitoring system."""
    
    def __init__(self, memory_db: MemoryDatabase, config: Dict[str, Any]):
        self.memory_db = memory_db
        self.config = config
        self.collector = MetricCollector(memory_db)
        self.analyzer = PerformanceAnalyzer(memory_db)
        self.alert_manager = AlertManager(memory_db, config)
        self.running = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start performance monitoring."""
        try:
            if self.running:
                logging.warning("Performance monitoring is already running")
                return
            
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            
            logging.info("Performance monitoring started")
            
        except Exception as e:
            logging.error(f"Failed to start performance monitoring: {e}")
            self.running = False
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        try:
            self.running = False
            
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
            
            logging.info("Performance monitoring stopped")
            
        except Exception as e:
            logging.error(f"Failed to stop performance monitoring: {e}")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        try:
            interval = self.config.get('monitoring', {}).get('performance_monitoring', {}).get(
                'metric_collection_interval', 60
            )
            
            while self.running:
                try:
                    # Collect metrics
                    all_metrics = []
                    all_metrics.extend(self.collector.collect_system_metrics())
                    all_metrics.extend(self.collector.collect_process_metrics())
                    all_metrics.extend(self.collector.collect_database_metrics())
                    
                    # Store metrics in database
                    for metric in all_metrics:
                        self.memory_db.record_performance_metric(
                            metric_type=metric.metric_type,
                            metric_value=metric.value,
                            measurement_context={
                                'component': metric.component,
                                'context': metric.context
                            }
                        )
                    
                    # Analyze metrics
                    analysis = self.analyzer.analyze_metrics(all_metrics)
                    
                    # Check for alerts
                    new_alerts = self.alert_manager.check_alerts(all_metrics)
                    
                    # Log alerts
                    for alert in new_alerts:
                        if alert.level == 'critical':
                            logging.error(f"CRITICAL ALERT: {alert.message}")
                        else:
                            logging.warning(f"WARNING ALERT: {alert.message}")
                    
                    # Log health score
                    health_score = analysis.get('health_score', 1.0)
                    if health_score < 0.8:
                        logging.warning(f"System health score: {health_score:.2f}")
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    logging.error(f"Error in monitoring loop: {e}")
                    time.sleep(interval)
                    
        except Exception as e:
            logging.error(f"Monitoring loop error: {e}")
        finally:
            self.running = False
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        try:
            # Collect current metrics
            all_metrics = []
            all_metrics.extend(self.collector.collect_system_metrics())
            all_metrics.extend(self.collector.collect_process_metrics())
            all_metrics.extend(self.collector.collect_database_metrics())
            
            # Convert to dictionary format
            metrics_dict = {}
            for metric in all_metrics:
                metrics_dict[metric.metric_type] = {
                    'value': metric.value,
                    'component': metric.component,
                    'context': metric.context,
                    'timestamp': metric.timestamp.isoformat()
                }
            
            return metrics_dict
            
        except Exception as e:
            logging.error(f"Failed to get current metrics: {e}")
            return {'error': str(e)}
    
    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        try:
            # Get metrics from database
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # This would typically query the database for historical metrics
            # For now, return current state
            current_metrics = self.get_current_metrics()
            alert_summary = self.alert_manager.get_alert_summary()
            
            return {
                'report_period_hours': hours,
                'generated_at': datetime.now().isoformat(),
                'current_metrics': current_metrics,
                'alert_summary': alert_summary,
                'health_status': self._get_health_status(),
                'recommendations': self._get_recommendations()
            }
            
        except Exception as e:
            logging.error(f"Failed to generate performance report: {e}")
            return {'error': str(e)}
    
    def _get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        try:
            active_alerts = self.alert_manager.get_active_alerts()
            critical_count = len([a for a in active_alerts if a.level == 'critical'])
            warning_count = len([a for a in active_alerts if a.level == 'warning'])
            
            if critical_count > 0:
                status = 'critical'
            elif warning_count > 0:
                status = 'warning'
            else:
                status = 'healthy'
            
            return {
                'status': status,
                'critical_alerts': critical_count,
                'warning_alerts': warning_count,
                'monitoring_active': self.running
            }
            
        except Exception as e:
            logging.error(f"Failed to get health status: {e}")
            return {'status': 'unknown', 'error': str(e)}
    
    def _get_recommendations(self) -> List[str]:
        """Get current performance recommendations."""
        try:
            # Collect current metrics for analysis
            all_metrics = []
            all_metrics.extend(self.collector.collect_system_metrics())
            all_metrics.extend(self.collector.collect_process_metrics())
            all_metrics.extend(self.collector.collect_database_metrics())
            
            # Analyze and get recommendations
            analysis = self.analyzer.analyze_metrics(all_metrics)
            return analysis.get('recommendations', [])
            
        except Exception as e:
            logging.error(f"Failed to get recommendations: {e}")
            return [f"Error generating recommendations: {e}"]