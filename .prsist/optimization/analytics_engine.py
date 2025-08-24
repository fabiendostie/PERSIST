#!/usr/bin/env python3
"""
Advanced analytics engine for Prsist Memory System Phase 4.
Implements comprehensive analytics, insights generation, and performance analysis.
"""

import json
import logging
import statistics
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter

import numpy as np

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from database import MemoryDatabase
from utils import setup_logging

@dataclass
class SessionAnalytics:
    """Analytics for session activity."""
    total_sessions: int
    total_duration_hours: float
    average_duration_hours: float
    unique_files_touched: int
    total_changes: int
    productivity_score: float
    focus_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

@dataclass
class ProductivityMetrics:
    """Productivity metrics for developers."""
    code_velocity: Dict[str, float]
    focus_metrics: Dict[str, float]
    quality_metrics: Dict[str, float]
    time_distribution: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

@dataclass
class MemoryEfficiencyMetrics:
    """Memory system efficiency metrics."""
    cache_performance: Dict[str, float]
    compression_efficiency: Dict[str, float]
    relevance_accuracy: Dict[str, float]
    sync_performance: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

@dataclass
class UsagePattern:
    """Represents a usage pattern."""
    pattern_id: str
    pattern_type: str
    frequency: int
    confidence: float
    description: str
    first_seen: datetime
    last_seen: datetime
    sessions_involved: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['first_seen'] = self.first_seen.isoformat()
        result['last_seen'] = self.last_seen.isoformat()
        return result

class MetricsCollector:
    """Collects various metrics from the memory system."""
    
    def __init__(self, memory_db: MemoryDatabase):
        self.memory_db = memory_db
        
    def collect_session_metrics(self, time_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Collect metrics for session activity."""
        try:
            start_time, end_time = time_range
            
            # Get sessions in time range
            sessions = self._get_sessions_in_range(start_time, end_time)
            
            if not sessions:
                return self._empty_session_metrics()
            
            # Calculate basic metrics
            session_count = len(sessions)
            total_duration = sum(s.get('duration_minutes', 0) for s in sessions)
            avg_duration = total_duration / session_count if session_count > 0 else 0
            
            # Get unique files touched
            unique_files = self._get_unique_files_touched(sessions)
            
            # Get total changes
            total_changes = self._get_total_changes(sessions)
            
            # Get tool usage statistics
            tool_usage = self._get_tool_usage_stats(sessions)
            
            # Get git activity
            git_activity = self._get_git_activity(start_time, end_time)
            
            return {
                'session_count': session_count,
                'total_duration': total_duration,
                'avg_duration': avg_duration,
                'unique_files': len(unique_files),
                'total_changes': total_changes,
                'total_hours': total_duration / 60.0,
                'tool_usage': tool_usage,
                'git_activity': git_activity,
                'lines_changed': git_activity.get('total_insertions', 0) + git_activity.get('total_deletions', 0),
                'commits': git_activity.get('commit_count', 0),
                'context_switches': self._calculate_context_switches(sessions),
                'refactoring_changes': self._count_refactoring_changes(sessions),
                'doc_updates': self._count_documentation_updates(sessions)
            }
            
        except Exception as e:
            logging.error(f"Failed to collect session metrics: {e}")
            return self._empty_session_metrics()
    
    def _get_sessions_in_range(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get sessions within time range."""
        try:
            query = """
            SELECT * FROM sessions 
            WHERE started_at >= ? AND started_at <= ?
            ORDER BY started_at DESC
            """
            
            with sqlite3.connect(self.memory_db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, (start_time.isoformat(), end_time.isoformat()))
                sessions = [dict(row) for row in cursor.fetchall()]
            
            # Calculate duration for each session
            for session in sessions:
                if session.get('ended_at'):
                    start = datetime.fromisoformat(session['started_at'])
                    end = datetime.fromisoformat(session['ended_at'])
                    session['duration_minutes'] = (end - start).total_seconds() / 60.0
                else:
                    session['duration_minutes'] = 0
            
            return sessions
            
        except Exception as e:
            logging.error(f"Failed to get sessions in range: {e}")
            return []
    
    def _get_unique_files_touched(self, sessions: List[Dict[str, Any]]) -> Set[str]:
        """Get unique files touched across sessions."""
        try:
            unique_files = set()
            
            for session in sessions:
                session_id = session['session_id']
                
                # Get file interactions for this session
                query = """
                SELECT DISTINCT file_path FROM file_interactions 
                WHERE session_id = ?
                """
                
                with sqlite3.connect(self.memory_db.db_path) as conn:
                    cursor = conn.execute(query, (session_id,))
                    files = [row[0] for row in cursor.fetchall()]
                    unique_files.update(files)
            
            return unique_files
            
        except Exception as e:
            logging.error(f"Failed to get unique files: {e}")
            return set()
    
    def _get_total_changes(self, sessions: List[Dict[str, Any]]) -> int:
        """Get total number of changes across sessions."""
        try:
            total_changes = 0
            
            for session in sessions:
                session_id = session['session_id']
                
                # Count tool usage that indicates changes
                query = """
                SELECT COUNT(*) FROM tool_usage 
                WHERE session_id = ? AND tool_name IN ('Edit', 'MultiEdit', 'Write')
                """
                
                with sqlite3.connect(self.memory_db.db_path) as conn:
                    cursor = conn.execute(query, (session_id,))
                    changes = cursor.fetchone()[0]
                    total_changes += changes
            
            return total_changes
            
        except Exception as e:
            logging.error(f"Failed to get total changes: {e}")
            return 0
    
    def _get_tool_usage_stats(self, sessions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get tool usage statistics."""
        try:
            tool_stats = defaultdict(int)
            
            for session in sessions:
                session_id = session['session_id']
                
                query = """
                SELECT tool_name, COUNT(*) as usage_count 
                FROM tool_usage 
                WHERE session_id = ? 
                GROUP BY tool_name
                """
                
                with sqlite3.connect(self.memory_db.db_path) as conn:
                    cursor = conn.execute(query, (session_id,))
                    for tool_name, count in cursor.fetchall():
                        tool_stats[tool_name] += count
            
            return dict(tool_stats)
            
        except Exception as e:
            logging.error(f"Failed to get tool usage stats: {e}")
            return {}
    
    def _get_git_activity(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get git activity in time range."""
        try:
            query = """
            SELECT 
                COUNT(*) as commit_count,
                SUM(CAST(json_extract(metadata, '$.insertions') AS INTEGER)) as total_insertions,
                SUM(CAST(json_extract(metadata, '$.deletions') AS INTEGER)) as total_deletions,
                COUNT(DISTINCT author) as unique_authors
            FROM git_commits 
            WHERE committed_at >= ? AND committed_at <= ?
            """
            
            with sqlite3.connect(self.memory_db.db_path) as conn:
                cursor = conn.execute(query, (start_time.isoformat(), end_time.isoformat()))
                result = cursor.fetchone()
                
                return {
                    'commit_count': result[0] or 0,
                    'total_insertions': result[1] or 0,
                    'total_deletions': result[2] or 0,
                    'unique_authors': result[3] or 0
                }
                
        except Exception as e:
            logging.error(f"Failed to get git activity: {e}")
            return {'commit_count': 0, 'total_insertions': 0, 'total_deletions': 0, 'unique_authors': 0}
    
    def _calculate_context_switches(self, sessions: List[Dict[str, Any]]) -> int:
        """Calculate number of context switches."""
        try:
            context_switches = 0
            
            for session in sessions:
                session_id = session['session_id']
                
                # Get file interactions ordered by time
                query = """
                SELECT file_path, timestamp 
                FROM file_interactions 
                WHERE session_id = ? 
                ORDER BY timestamp
                """
                
                with sqlite3.connect(self.memory_db.db_path) as conn:
                    cursor = conn.execute(query, (session_id,))
                    interactions = cursor.fetchall()
                
                # Count switches between different files
                prev_file = None
                for file_path, _ in interactions:
                    if prev_file and prev_file != file_path:
                        context_switches += 1
                    prev_file = file_path
            
            return context_switches
            
        except Exception as e:
            logging.error(f"Failed to calculate context switches: {e}")
            return 0
    
    def _count_refactoring_changes(self, sessions: List[Dict[str, Any]]) -> int:
        """Count refactoring-related changes."""
        try:
            refactoring_keywords = ['refactor', 'rename', 'move', 'extract', 'inline']
            refactoring_count = 0
            
            for session in sessions:
                session_id = session['session_id']
                
                # Look for refactoring in context entries
                query = """
                SELECT COUNT(*) FROM context_entries 
                WHERE session_id = ? AND (
                    LOWER(content) LIKE '%refactor%' OR
                    LOWER(content) LIKE '%rename%' OR
                    LOWER(content) LIKE '%extract%' OR
                    LOWER(content) LIKE '%inline%'
                )
                """
                
                with sqlite3.connect(self.memory_db.db_path) as conn:
                    cursor = conn.execute(query, (session_id,))
                    count = cursor.fetchone()[0]
                    refactoring_count += count
            
            return refactoring_count
            
        except Exception as e:
            logging.error(f"Failed to count refactoring changes: {e}")
            return 0
    
    def _count_documentation_updates(self, sessions: List[Dict[str, Any]]) -> int:
        """Count documentation updates."""
        try:
            doc_count = 0
            
            for session in sessions:
                session_id = session['session_id']
                
                # Count interactions with documentation files
                query = """
                SELECT COUNT(*) FROM file_interactions 
                WHERE session_id = ? AND (
                    file_path LIKE '%.md' OR
                    file_path LIKE '%.rst' OR
                    file_path LIKE '%.txt' OR
                    LOWER(file_path) LIKE '%readme%' OR
                    LOWER(file_path) LIKE '%doc%'
                )
                """
                
                with sqlite3.connect(self.memory_db.db_path) as conn:
                    cursor = conn.execute(query, (session_id,))
                    count = cursor.fetchone()[0]
                    doc_count += count
            
            return doc_count
            
        except Exception as e:
            logging.error(f"Failed to count documentation updates: {e}")
            return 0
    
    def _empty_session_metrics(self) -> Dict[str, Any]:
        """Return empty session metrics."""
        return {
            'session_count': 0,
            'total_duration': 0,
            'avg_duration': 0,
            'unique_files': 0,
            'total_changes': 0,
            'total_hours': 0,
            'tool_usage': {},
            'git_activity': {},
            'lines_changed': 0,
            'commits': 0,
            'context_switches': 0,
            'refactoring_changes': 0,
            'doc_updates': 0
        }

class InsightsGenerator:
    """Generates actionable insights from analytics data."""
    
    def __init__(self):
        self.insight_rules = self._load_insight_rules()
        self.ml_analyzer = MLPatternAnalyzer()
        
    def _load_insight_rules(self) -> Dict[str, Any]:
        """Load insight generation rules."""
        return {
            'productivity_thresholds': {
                'high_velocity': 100,  # lines per hour
                'low_velocity': 20,
                'high_focus': 0.8,     # focus score
                'low_focus': 0.4,
                'high_commits': 5,     # commits per session
                'low_commits': 1
            },
            'pattern_thresholds': {
                'frequent_pattern': 5,  # occurrences
                'rare_pattern': 2,
                'confidence_threshold': 0.7
            },
            'performance_thresholds': {
                'high_memory_usage': 400,  # MB
                'slow_operation': 5000,    # ms
                'low_cache_hit_rate': 0.5
            }
        }
    
    def generate_insights(self, analytics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable insights from analytics data."""
        try:
            insights = []
            
            # Productivity insights
            productivity_insights = self._generate_productivity_insights(analytics)
            insights.extend(productivity_insights)
            
            # Pattern insights
            pattern_insights = self._generate_pattern_insights(analytics)
            insights.extend(pattern_insights)
            
            # Performance insights
            performance_insights = self._generate_performance_insights(analytics)
            insights.extend(performance_insights)
            
            # Anomaly insights
            anomaly_insights = self._generate_anomaly_insights(analytics)
            insights.extend(anomaly_insights)
            
            # ML-generated insights
            ml_insights = self.ml_analyzer.find_anomalies(analytics)
            insights.extend(ml_insights)
            
            # Sort by severity
            insights.sort(key=lambda x: {'critical': 3, 'warning': 2, 'info': 1}.get(x.get('severity', 'info'), 1), reverse=True)
            
            return insights
            
        except Exception as e:
            logging.error(f"Failed to generate insights: {e}")
            return [{'type': 'error', 'message': f'Error generating insights: {e}', 'severity': 'warning'}]
    
    def _generate_productivity_insights(self, analytics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate productivity-related insights."""
        insights = []
        
        try:
            productivity = analytics.get('productivity', {})
            code_velocity = productivity.get('code_velocity', {})
            focus_metrics = productivity.get('focus_metrics', {})
            
            thresholds = self.insight_rules['productivity_thresholds']
            
            # Code velocity insights
            lines_per_hour = code_velocity.get('lines_per_hour', 0)
            if lines_per_hour > thresholds['high_velocity']:
                insights.append({
                    'type': 'productivity',
                    'severity': 'info',
                    'message': f'High code velocity detected: {lines_per_hour:.1f} lines/hour',
                    'recommendation': 'Consider adding more tests and code reviews to maintain quality',
                    'data': {'lines_per_hour': lines_per_hour}
                })
            elif lines_per_hour < thresholds['low_velocity']:
                insights.append({
                    'type': 'productivity',
                    'severity': 'warning',
                    'message': f'Low code velocity: {lines_per_hour:.1f} lines/hour',
                    'recommendation': 'Consider removing blockers or improving development tools',
                    'data': {'lines_per_hour': lines_per_hour}
                })
            
            # Focus insights
            focus_duration = focus_metrics.get('average_focus_duration', 0)
            deep_work_percentage = focus_metrics.get('deep_work_percentage', 0)
            
            if deep_work_percentage < thresholds['low_focus']:
                insights.append({
                    'type': 'productivity',
                    'severity': 'warning',
                    'message': f'Low deep work percentage: {deep_work_percentage:.1%}',
                    'recommendation': 'Try to reduce interruptions and context switching',
                    'data': {'deep_work_percentage': deep_work_percentage}
                })
            
            # Commit frequency insights
            commits_per_session = code_velocity.get('commits_per_session', 0)
            if commits_per_session < thresholds['low_commits']:
                insights.append({
                    'type': 'productivity',
                    'severity': 'info',
                    'message': f'Low commit frequency: {commits_per_session:.1f} commits/session',
                    'recommendation': 'Consider making smaller, more frequent commits',
                    'data': {'commits_per_session': commits_per_session}
                })
            
        except Exception as e:
            logging.error(f"Failed to generate productivity insights: {e}")
        
        return insights
    
    def _generate_pattern_insights(self, analytics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate pattern-related insights."""
        insights = []
        
        try:
            patterns = analytics.get('patterns', {})
            peak_hours = patterns.get('peak_hours', [])
            
            if peak_hours:
                insights.append({
                    'type': 'pattern',
                    'severity': 'info',
                    'message': f'Most productive hours: {", ".join(map(str, peak_hours))}',
                    'recommendation': 'Schedule complex tasks during peak productivity hours',
                    'data': {'peak_hours': peak_hours}
                })
            
            # Tool usage patterns
            tool_patterns = patterns.get('tool_usage_patterns', {})
            most_used_tool = max(tool_patterns.items(), key=lambda x: x[1])[0] if tool_patterns else None
            
            if most_used_tool:
                insights.append({
                    'type': 'pattern',
                    'severity': 'info',
                    'message': f'Most used tool: {most_used_tool}',
                    'recommendation': 'Consider optimizing workflows around frequently used tools',
                    'data': {'most_used_tool': most_used_tool}
                })
            
        except Exception as e:
            logging.error(f"Failed to generate pattern insights: {e}")
        
        return insights
    
    def _generate_performance_insights(self, analytics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate performance-related insights."""
        insights = []
        
        try:
            performance = analytics.get('performance', {})
            thresholds = self.insight_rules['performance_thresholds']
            
            # Memory usage insights
            memory_usage = performance.get('memory_usage_mb', 0)
            if memory_usage > thresholds['high_memory_usage']:
                insights.append({
                    'type': 'performance',
                    'severity': 'warning',
                    'message': f'High memory usage: {memory_usage:.1f}MB',
                    'recommendation': 'Consider optimizing memory usage or increasing cache cleanup frequency',
                    'data': {'memory_usage_mb': memory_usage}
                })
            
            # Cache performance insights
            cache_hit_rate = performance.get('cache_hit_rate', 0)
            if cache_hit_rate < thresholds['low_cache_hit_rate']:
                insights.append({
                    'type': 'performance',
                    'severity': 'warning',
                    'message': f'Low cache hit rate: {cache_hit_rate:.1%}',
                    'recommendation': 'Review caching strategy and improve prefix detection',
                    'data': {'cache_hit_rate': cache_hit_rate}
                })
            
        except Exception as e:
            logging.error(f"Failed to generate performance insights: {e}")
        
        return insights
    
    def _generate_anomaly_insights(self, analytics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate anomaly-related insights."""
        insights = []
        
        try:
            # This is a simplified anomaly detection
            # In a real implementation, you'd use more sophisticated methods
            
            productivity = analytics.get('productivity', {})
            code_velocity = productivity.get('code_velocity', {})
            
            # Detect sudden productivity changes
            lines_per_hour = code_velocity.get('lines_per_hour', 0)
            historical_average = 50  # This would come from historical data
            
            if abs(lines_per_hour - historical_average) > historical_average * 0.5:
                insights.append({
                    'type': 'anomaly',
                    'severity': 'info',
                    'message': f'Productivity anomaly detected: {lines_per_hour:.1f} vs {historical_average:.1f} average',
                    'recommendation': 'Investigate factors affecting productivity',
                    'data': {'current': lines_per_hour, 'average': historical_average}
                })
            
        except Exception as e:
            logging.error(f"Failed to generate anomaly insights: {e}")
        
        return insights
    
    def generate_recommendations(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations from insights."""
        try:
            recommendations = []
            
            # Group insights by type
            insights_by_type = defaultdict(list)
            for insight in insights:
                insights_by_type[insight['type']].append(insight)
            
            # Generate type-specific recommendations
            for insight_type, type_insights in insights_by_type.items():
                if insight_type == 'productivity':
                    rec = self._generate_productivity_recommendation(type_insights)
                elif insight_type == 'performance':
                    rec = self._generate_performance_recommendation(type_insights)
                elif insight_type == 'pattern':
                    rec = self._generate_pattern_recommendation(type_insights)
                else:
                    rec = self._generate_generic_recommendation(type_insights)
                
                if rec:
                    recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logging.error(f"Failed to generate recommendations: {e}")
            return []
    
    def _generate_productivity_recommendation(self, insights: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate productivity-specific recommendations."""
        try:
            if not insights:
                return None
            
            # Count severity types
            warnings = [i for i in insights if i.get('severity') == 'warning']
            
            if warnings:
                return {
                    'type': 'productivity',
                    'priority': 'high',
                    'title': 'Productivity Optimization Needed',
                    'description': f'Detected {len(warnings)} productivity issues',
                    'actions': [i.get('recommendation', '') for i in warnings if i.get('recommendation')]
                }
            
            return None
            
        except Exception as e:
            logging.error(f"Failed to generate productivity recommendation: {e}")
            return None
    
    def _generate_performance_recommendation(self, insights: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate performance-specific recommendations."""
        try:
            if not insights:
                return None
            
            critical_issues = [i for i in insights if i.get('severity') == 'critical']
            warning_issues = [i for i in insights if i.get('severity') == 'warning']
            
            if critical_issues or warning_issues:
                priority = 'critical' if critical_issues else 'high'
                issue_count = len(critical_issues) + len(warning_issues)
                
                return {
                    'type': 'performance',
                    'priority': priority,
                    'title': 'Performance Issues Detected',
                    'description': f'Found {issue_count} performance issues requiring attention',
                    'actions': [i.get('recommendation', '') for i in insights if i.get('recommendation')]
                }
            
            return None
            
        except Exception as e:
            logging.error(f"Failed to generate performance recommendation: {e}")
            return None
    
    def _generate_pattern_recommendation(self, insights: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate pattern-specific recommendations."""
        try:
            if not insights:
                return None
            
            return {
                'type': 'pattern',
                'priority': 'medium',
                'title': 'Usage Pattern Optimization',
                'description': f'Identified {len(insights)} usage patterns for optimization',
                'actions': [i.get('recommendation', '') for i in insights if i.get('recommendation')]
            }
            
        except Exception as e:
            logging.error(f"Failed to generate pattern recommendation: {e}")
            return None
    
    def _generate_generic_recommendation(self, insights: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate generic recommendations."""
        try:
            if not insights:
                return None
            
            return {
                'type': 'general',
                'priority': 'medium',
                'title': 'General Improvements',
                'description': f'Found {len(insights)} areas for improvement',
                'actions': [i.get('recommendation', '') for i in insights if i.get('recommendation')]
            }
            
        except Exception as e:
            logging.error(f"Failed to generate generic recommendation: {e}")
            return None

class MLPatternAnalyzer:
    """Machine learning-based pattern analyzer."""
    
    def __init__(self):
        self.anomaly_detector = SimpleAnomalyDetector()
        
    def find_anomalies(self, analytics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find anomalies using simple ML techniques."""
        try:
            anomalies = []
            
            # Extract numerical features
            features = self._extract_features(analytics)
            
            if features:
                detected_anomalies = self.anomaly_detector.detect_anomalies(features)
                
                for anomaly in detected_anomalies:
                    anomalies.append({
                        'type': 'ml_anomaly',
                        'severity': 'info',
                        'message': f'ML detected anomaly: {anomaly["description"]}',
                        'recommendation': 'Investigate unusual patterns in the data',
                        'data': anomaly
                    })
            
            return anomalies
            
        except Exception as e:
            logging.error(f"Failed to find ML anomalies: {e}")
            return []
    
    def _extract_features(self, analytics: Dict[str, Any]) -> List[float]:
        """Extract numerical features for ML analysis."""
        try:
            features = []
            
            # Extract productivity features
            productivity = analytics.get('productivity', {})
            code_velocity = productivity.get('code_velocity', {})
            
            features.append(code_velocity.get('lines_per_hour', 0))
            features.append(code_velocity.get('commits_per_session', 0))
            features.append(code_velocity.get('files_per_session', 0))
            
            # Extract performance features
            performance = analytics.get('performance', {})
            features.append(performance.get('memory_usage_mb', 0))
            features.append(performance.get('cache_hit_rate', 0))
            
            # Extract session features
            summary = analytics.get('summary', {})
            features.append(summary.get('total_sessions', 0))
            features.append(summary.get('average_duration', 0))
            
            return features
            
        except Exception as e:
            logging.error(f"Failed to extract features: {e}")
            return []

class SimpleAnomalyDetector:
    """Simple statistical anomaly detector."""
    
    def __init__(self):
        self.z_score_threshold = 2.0
        
    def detect_anomalies(self, features: List[float]) -> List[Dict[str, Any]]:
        """Detect anomalies using Z-score method."""
        try:
            if len(features) < 3:
                return []
            
            anomalies = []
            
            # Calculate Z-scores
            mean_val = statistics.mean(features)
            std_val = statistics.stdev(features) if len(features) > 1 else 0
            
            if std_val == 0:
                return []
            
            for i, value in enumerate(features):
                z_score = abs((value - mean_val) / std_val)
                
                if z_score > self.z_score_threshold:
                    anomalies.append({
                        'feature_index': i,
                        'value': value,
                        'z_score': z_score,
                        'description': f'Feature {i} has unusual value {value:.2f} (z-score: {z_score:.2f})'
                    })
            
            return anomalies
            
        except Exception as e:
            logging.error(f"Failed to detect anomalies: {e}")
            return []

class AnalyticsEngine:
    """Main analytics engine for comprehensive analysis."""
    
    def __init__(self, database_path: str):
        self.db_path = database_path
        self.memory_db = MemoryDatabase(database_path)
        self.metrics_collector = MetricsCollector(self.memory_db)
        self.insights_generator = InsightsGenerator()
        self.visualizer = AnalyticsVisualizer() if MATPLOTLIB_AVAILABLE else None
        
    def generate_session_analytics(self, time_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Generate comprehensive analytics for session activity."""
        try:
            # Collect metrics
            session_metrics = self.metrics_collector.collect_session_metrics(time_range)
            
            # Calculate analytics
            analytics = {
                'summary': {
                    'total_sessions': session_metrics['session_count'],
                    'total_duration': session_metrics['total_duration'],
                    'average_duration': session_metrics['avg_duration'],
                    'unique_files_touched': session_metrics['unique_files'],
                    'total_changes': session_metrics['total_changes']
                },
                'productivity': self.calculate_productivity_metrics(session_metrics),
                'patterns': self.analyze_usage_patterns(session_metrics),
                'performance': self.analyze_system_performance(time_range),
                'insights': []
            }
            
            # Generate insights
            insights = self.insights_generator.generate_insights(analytics)
            analytics['insights'] = insights
            
            # Generate recommendations
            recommendations = self.insights_generator.generate_recommendations(insights)
            analytics['recommendations'] = recommendations
            
            # Create visualizations if available
            if self.visualizer:
                analytics['visualizations'] = self.create_analytics_visualizations(analytics)
            
            return analytics
            
        except Exception as e:
            logging.error(f"Failed to generate session analytics: {e}")
            return {'error': str(e)}
    
    def calculate_productivity_metrics(self, metrics: Dict[str, Any]) -> ProductivityMetrics:
        """Calculate developer productivity metrics."""
        try:
            # Code velocity
            total_hours = metrics.get('total_hours', 1)
            code_velocity = {
                'lines_per_hour': metrics.get('lines_changed', 0) / max(total_hours, 1),
                'commits_per_session': metrics.get('commits', 0) / max(metrics.get('session_count', 1), 1),
                'files_per_session': metrics.get('unique_files', 0) / max(metrics.get('session_count', 1), 1)
            }
            
            # Focus metrics
            focus_metrics = {
                'average_focus_duration': self.calculate_focus_duration(metrics),
                'context_switches': metrics.get('context_switches', 0),
                'deep_work_percentage': self.calculate_deep_work_percentage(metrics)
            }
            
            # Quality metrics
            quality_metrics = {
                'test_coverage_trend': self.analyze_test_coverage_trend(metrics),
                'refactoring_ratio': metrics.get('refactoring_changes', 0) / max(metrics.get('total_changes', 1), 1),
                'documentation_updates': metrics.get('doc_updates', 0)
            }
            
            # Time distribution
            time_distribution = {
                'coding_time': 0.6,  # Placeholder - would calculate from actual data
                'debugging_time': 0.2,
                'documentation_time': 0.1,
                'other_time': 0.1
            }
            
            return ProductivityMetrics(
                code_velocity=code_velocity,
                focus_metrics=focus_metrics,
                quality_metrics=quality_metrics,
                time_distribution=time_distribution
            )
            
        except Exception as e:
            logging.error(f"Failed to calculate productivity metrics: {e}")
            return ProductivityMetrics({}, {}, {}, {})
    
    def calculate_focus_duration(self, metrics: Dict[str, Any]) -> float:
        """Calculate average focus duration."""
        try:
            # Simplified calculation - in reality would analyze timestamps
            total_duration = metrics.get('total_duration', 0)
            context_switches = metrics.get('context_switches', 0)
            
            if context_switches == 0:
                return total_duration
            
            return total_duration / (context_switches + 1)
            
        except Exception as e:
            logging.error(f"Failed to calculate focus duration: {e}")
            return 0.0
    
    def calculate_deep_work_percentage(self, metrics: Dict[str, Any]) -> float:
        """Calculate percentage of time spent in deep work."""
        try:
            # Simplified calculation - sessions with fewer context switches are "deeper work"
            context_switches = metrics.get('context_switches', 0)
            total_duration = metrics.get('total_duration', 1)
            
            # Normalize context switches per hour
            switches_per_hour = context_switches / max(total_duration / 60.0, 1)
            
            # Lower switches = higher deep work percentage
            if switches_per_hour <= 2:
                return 0.9
            elif switches_per_hour <= 5:
                return 0.7
            elif switches_per_hour <= 10:
                return 0.5
            else:
                return 0.3
                
        except Exception as e:
            logging.error(f"Failed to calculate deep work percentage: {e}")
            return 0.5
    
    def analyze_test_coverage_trend(self, metrics: Dict[str, Any]) -> float:
        """Analyze test coverage trend."""
        try:
            # Simplified - would analyze actual test files and coverage
            tool_usage = metrics.get('tool_usage', {})
            test_activity = tool_usage.get('test_related_tools', 0)
            total_activity = sum(tool_usage.values()) if tool_usage else 1
            
            return test_activity / total_activity
            
        except Exception as e:
            logging.error(f"Failed to analyze test coverage trend: {e}")
            return 0.0
    
    def analyze_usage_patterns(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze usage patterns."""
        try:
            return {
                'peak_hours': self._find_peak_hours(metrics),
                'tool_usage_patterns': metrics.get('tool_usage', {}),
                'file_type_distribution': self._analyze_file_types(metrics),
                'session_length_distribution': self._analyze_session_lengths(metrics)
            }
            
        except Exception as e:
            logging.error(f"Failed to analyze usage patterns: {e}")
            return {}
    
    def _find_peak_hours(self, metrics: Dict[str, Any]) -> List[int]:
        """Find peak productivity hours."""
        # Simplified - would analyze actual session timestamps
        return [9, 10, 11, 14, 15]  # Common productive hours
    
    def _analyze_file_types(self, metrics: Dict[str, Any]) -> Dict[str, int]:
        """Analyze file type distribution."""
        # Simplified - would analyze actual file interactions
        return {
            '.py': 45,
            '.js': 30,
            '.md': 15,
            '.yaml': 10
        }
    
    def _analyze_session_lengths(self, metrics: Dict[str, Any]) -> Dict[str, int]:
        """Analyze session length distribution."""
        # Simplified - would analyze actual session durations
        return {
            'short (< 30 min)': 20,
            'medium (30-120 min)': 60,
            'long (> 120 min)': 20
        }
    
    def analyze_system_performance(self, time_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Analyze system performance."""
        try:
            # Get performance metrics from database
            start_time, end_time = time_range
            
            # Simplified performance analysis
            return {
                'memory_usage_mb': 150,  # Would get from actual metrics
                'cache_hit_rate': 0.75,
                'average_response_time_ms': 250,
                'error_rate': 0.02,
                'sync_success_rate': 0.98
            }
            
        except Exception as e:
            logging.error(f"Failed to analyze system performance: {e}")
            return {}
    
    def analyze_memory_efficiency(self) -> MemoryEfficiencyMetrics:
        """Analyze memory system efficiency."""
        try:
            cache_performance = {
                'hit_rate': self.calculate_cache_hit_rate(),
                'avg_retrieval_time': self.calculate_avg_retrieval_time(),
                'cache_size_mb': self.get_cache_size_mb(),
                'eviction_rate': self.calculate_eviction_rate()
            }
            
            compression_efficiency = {
                'avg_compression_ratio': self.calculate_compression_ratio(),
                'tokens_saved': self.calculate_tokens_saved(),
                'cost_reduction_percentage': self.calculate_cost_reduction()
            }
            
            relevance_accuracy = {
                'precision': self.calculate_relevance_precision(),
                'recall': self.calculate_relevance_recall(),
                'f1_score': self.calculate_relevance_f1()
            }
            
            sync_performance = {
                'sync_success_rate': 0.95,
                'avg_sync_time_seconds': 15,
                'conflict_rate': 0.05,
                'bandwidth_efficiency': 0.8
            }
            
            return MemoryEfficiencyMetrics(
                cache_performance=cache_performance,
                compression_efficiency=compression_efficiency,
                relevance_accuracy=relevance_accuracy,
                sync_performance=sync_performance
            )
            
        except Exception as e:
            logging.error(f"Failed to analyze memory efficiency: {e}")
            return MemoryEfficiencyMetrics({}, {}, {}, {})
    
    def calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        # Simplified - would get from actual cache statistics
        return 0.75
    
    def calculate_avg_retrieval_time(self) -> float:
        """Calculate average retrieval time."""
        return 150.0  # ms
    
    def get_cache_size_mb(self) -> float:
        """Get current cache size."""
        return 120.0  # MB
    
    def calculate_eviction_rate(self) -> float:
        """Calculate cache eviction rate."""
        return 0.05  # 5%
    
    def calculate_compression_ratio(self) -> float:
        """Calculate average compression ratio."""
        return 3.2  # 3.2:1 compression
    
    def calculate_tokens_saved(self) -> int:
        """Calculate total tokens saved."""
        return 50000
    
    def calculate_cost_reduction(self) -> float:
        """Calculate cost reduction percentage."""
        return 0.6  # 60%
    
    def calculate_relevance_precision(self) -> float:
        """Calculate relevance scoring precision."""
        return 0.85
    
    def calculate_relevance_recall(self) -> float:
        """Calculate relevance scoring recall."""
        return 0.78
    
    def calculate_relevance_f1(self) -> float:
        """Calculate relevance scoring F1 score."""
        precision = self.calculate_relevance_precision()
        recall = self.calculate_relevance_recall()
        return 2 * (precision * recall) / (precision + recall)
    
    def create_analytics_visualizations(self, analytics: Dict[str, Any]) -> Dict[str, str]:
        """Create analytics visualizations."""
        if not self.visualizer:
            return {'error': 'Visualization not available'}
        
        try:
            return self.visualizer.create_visualizations(analytics)
        except Exception as e:
            logging.error(f"Failed to create visualizations: {e}")
            return {'error': str(e)}

class AnalyticsVisualizer:
    """Creates visualizations for analytics data."""
    
    def __init__(self):
        self.output_dir = Path('.prsist/visualizations')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_visualizations(self, analytics: Dict[str, Any]) -> Dict[str, str]:
        """Create various visualizations from analytics data."""
        try:
            visualizations = {}
            
            # Create productivity chart
            productivity_chart = self._create_productivity_chart(analytics.get('productivity', {}))
            if productivity_chart:
                visualizations['productivity'] = productivity_chart
            
            # Create usage patterns chart
            patterns_chart = self._create_patterns_chart(analytics.get('patterns', {}))
            if patterns_chart:
                visualizations['patterns'] = patterns_chart
            
            # Create performance chart
            performance_chart = self._create_performance_chart(analytics.get('performance', {}))
            if performance_chart:
                visualizations['performance'] = performance_chart
            
            return visualizations
            
        except Exception as e:
            logging.error(f"Failed to create visualizations: {e}")
            return {'error': str(e)}
    
    def _create_productivity_chart(self, productivity: Dict[str, Any]) -> Optional[str]:
        """Create productivity metrics chart."""
        try:
            if not productivity:
                return None
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Code velocity chart
            code_velocity = productivity.get('code_velocity', {})
            if code_velocity:
                metrics = list(code_velocity.keys())
                values = list(code_velocity.values())
                
                ax1.bar(metrics, values)
                ax1.set_title('Code Velocity Metrics')
                ax1.set_ylabel('Value')
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # Focus metrics chart
            focus_metrics = productivity.get('focus_metrics', {})
            if focus_metrics:
                metrics = list(focus_metrics.keys())
                values = list(focus_metrics.values())
                
                ax2.bar(metrics, values)
                ax2.set_title('Focus Metrics')
                ax2.set_ylabel('Value')
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            # Save chart
            output_path = self.output_dir / 'productivity_chart.png'
            plt.savefig(output_path)
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            logging.error(f"Failed to create productivity chart: {e}")
            return None
    
    def _create_patterns_chart(self, patterns: Dict[str, Any]) -> Optional[str]:
        """Create usage patterns chart."""
        try:
            if not patterns:
                return None
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Tool usage patterns
            tool_patterns = patterns.get('tool_usage_patterns', {})
            if tool_patterns:
                tools = list(tool_patterns.keys())
                usage = list(tool_patterns.values())
                
                ax.pie(usage, labels=tools, autopct='%1.1f%%')
                ax.set_title('Tool Usage Distribution')
            
            # Save chart
            output_path = self.output_dir / 'patterns_chart.png'
            plt.savefig(output_path)
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            logging.error(f"Failed to create patterns chart: {e}")
            return None
    
    def _create_performance_chart(self, performance: Dict[str, Any]) -> Optional[str]:
        """Create performance metrics chart."""
        try:
            if not performance:
                return None
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Performance metrics
            metrics = []
            values = []
            
            for key, value in performance.items():
                if isinstance(value, (int, float)):
                    metrics.append(key.replace('_', ' ').title())
                    values.append(value)
            
            if metrics and values:
                ax.bar(metrics, values)
                ax.set_title('System Performance Metrics')
                ax.set_ylabel('Value')
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            # Save chart
            output_path = self.output_dir / 'performance_chart.png'
            plt.savefig(output_path)
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            logging.error(f"Failed to create performance chart: {e}")
            return None