#!/usr/bin/env python3
"""
Productivity Tracker for Prsist Memory System - Phase 2.
Tracks development velocity, patterns, and memory system effectiveness.
"""

import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter

from database import MemoryDatabase
from git_integration import GitMetadataExtractor


@dataclass
class ProductivityMetric:
    """Data class for productivity metrics."""
    metric_type: str
    value: float
    timestamp: datetime
    context: Dict[str, Any] = None
    session_id: str = None


class ProductivityTracker:
    """Tracks development productivity and memory system effectiveness."""
    
    def __init__(self, memory_dir: str, repo_path: str = "."):
        """Initialize productivity tracker."""
        self.memory_dir = Path(memory_dir)
        self.repo_path = Path(repo_path).resolve()
        
        # Initialize components
        self.db = MemoryDatabase(self.memory_dir / "storage" / "sessions.db")
        
        try:
            self.git_extractor = GitMetadataExtractor(str(self.repo_path))
            self.git_available = True
        except Exception as e:
            logging.warning(f"Git integration unavailable for productivity tracking: {e}")
            self.git_available = False
        
        logging.info("Productivity Tracker initialized")
    
    def measure_development_velocity(self, time_period_days: int = 7, session_id: str = None) -> Dict[str, Any]:
        """Measure development velocity over a time period."""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=time_period_days)
            
            # Get sessions in time period
            sessions = self._get_sessions_in_period(start_time, end_time, session_id)
            
            # Get git commits in time period
            commits = []
            if self.git_available:
                commits = self._get_commits_in_period(start_time, end_time)
            
            # Calculate velocity metrics
            velocity_metrics = {
                "time_period_days": time_period_days,
                "measurement_date": end_time.isoformat(),
                "session_metrics": self._calculate_session_velocity(sessions),
                "git_metrics": self._calculate_git_velocity(commits) if commits else {},
                "combined_metrics": {},
                "productivity_trends": self._analyze_productivity_trends(sessions, commits),
                "memory_impact": self._calculate_memory_system_impact(sessions)
            }
            
            # Combine session and git metrics
            if commits:
                velocity_metrics["combined_metrics"] = self._combine_velocity_metrics(
                    velocity_metrics["session_metrics"], 
                    velocity_metrics["git_metrics"]
                )
            
            # Store the metrics
            self._store_velocity_metrics(velocity_metrics, session_id)
            
            return velocity_metrics
            
        except Exception as e:
            logging.error(f"Failed to measure development velocity: {e}")
            return {"error": str(e)}
    
    def _get_sessions_in_period(self, start_time: datetime, end_time: datetime, 
                               session_id: str = None) -> List[Dict]:
        """Get sessions within time period."""
        try:
            if session_id:
                # Get specific session if it's in period
                session = self.db.get_session(session_id)
                if session and self._is_session_in_period(session, start_time, end_time):
                    return [session]
                return []
            else:
                # Get all sessions in period
                all_sessions = self.db.get_recent_sessions(limit=100)  # Get more sessions to filter
                return [s for s in all_sessions if self._is_session_in_period(s, start_time, end_time)]
                
        except Exception as e:
            logging.error(f"Failed to get sessions in period: {e}")
            return []
    
    def _is_session_in_period(self, session: Dict, start_time: datetime, end_time: datetime) -> bool:
        """Check if session falls within time period."""
        try:
            session_time = datetime.fromisoformat(session.get('created_at', ''))
            return start_time <= session_time <= end_time
        except:
            return False
    
    def _get_commits_in_period(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Get git commits within time period."""
        try:
            # Get recent commits from database
            commits = self.db.get_recent_commits(limit=100)
            
            # Filter by time period
            period_commits = []
            for commit in commits:
                try:
                    commit_time = datetime.fromisoformat(commit.get('commit_timestamp', ''))
                    if start_time <= commit_time <= end_time:
                        period_commits.append(commit)
                except:
                    continue
            
            return period_commits
            
        except Exception as e:
            logging.error(f"Failed to get commits in period: {e}")
            return []
    
    def _calculate_session_velocity(self, sessions: List[Dict]) -> Dict[str, Any]:
        """Calculate velocity metrics from sessions."""
        if not sessions:
            return {"total_sessions": 0}
        
        total_sessions = len(sessions)
        total_tools_used = 0
        total_files_touched = 0
        session_durations = []
        
        for session in sessions:
            # Get tool usage for session
            tools = self.db.get_session_tool_usage(session['id'])
            total_tools_used += len(tools)
            
            # Calculate session duration
            created = datetime.fromisoformat(session.get('created_at', ''))
            updated = datetime.fromisoformat(session.get('updated_at', session.get('created_at', '')))
            duration = (updated - created).total_seconds() / 3600  # hours
            session_durations.append(duration)
        
        avg_duration = sum(session_durations) / len(session_durations) if session_durations else 0
        
        return {
            "total_sessions": total_sessions,
            "avg_session_duration_hours": round(avg_duration, 2),
            "total_tools_used": total_tools_used,
            "avg_tools_per_session": round(total_tools_used / total_sessions, 2),
            "total_files_touched": total_files_touched,
            "productivity_score": self._calculate_session_productivity_score(sessions)
        }
    
    def _calculate_git_velocity(self, commits: List[Dict]) -> Dict[str, Any]:
        """Calculate velocity metrics from git commits."""
        if not commits:
            return {"total_commits": 0}
        
        total_commits = len(commits)
        total_files_changed = sum(c.get('changed_files_count', 0) for c in commits)
        total_lines_added = sum(c.get('lines_added', 0) for c in commits)
        total_lines_deleted = sum(c.get('lines_deleted', 0) for c in commits)
        
        # Analyze commit patterns
        commit_types = Counter()
        branches = set()
        
        for commit in commits:
            metadata = commit.get('commit_metadata', {})
            commit_type = metadata.get('commit_type', 'unknown')
            commit_types[commit_type] += 1
            
            if commit.get('branch_name'):
                branches.add(commit.get('branch_name'))
        
        return {
            "total_commits": total_commits,
            "total_files_changed": total_files_changed,
            "total_lines_added": total_lines_added,
            "total_lines_deleted": total_lines_deleted,
            "net_lines_added": total_lines_added - total_lines_deleted,
            "avg_files_per_commit": round(total_files_changed / total_commits, 2),
            "avg_lines_per_commit": round((total_lines_added + total_lines_deleted) / total_commits, 2),
            "commit_types": dict(commit_types),
            "branches_active": len(branches),
            "code_churn": round(total_lines_deleted / max(total_lines_added, 1), 2)
        }
    
    def _combine_velocity_metrics(self, session_metrics: Dict, git_metrics: Dict) -> Dict[str, Any]:
        """Combine session and git velocity metrics."""
        return {
            "commits_per_session": round(git_metrics.get('total_commits', 0) / max(session_metrics.get('total_sessions', 1), 1), 2),
            "lines_per_session": round((git_metrics.get('total_lines_added', 0) + git_metrics.get('total_lines_deleted', 0)) / max(session_metrics.get('total_sessions', 1), 1), 2),
            "files_per_session": round(git_metrics.get('total_files_changed', 0) / max(session_metrics.get('total_sessions', 1), 1), 2),
            "development_intensity": self._calculate_development_intensity(session_metrics, git_metrics)
        }
    
    def _calculate_development_intensity(self, session_metrics: Dict, git_metrics: Dict) -> float:
        """Calculate development intensity score."""
        # Normalize different metrics to 0-1 scale and combine
        session_factor = min(session_metrics.get('avg_tools_per_session', 0) / 10, 1.0)
        commit_factor = min(git_metrics.get('total_commits', 0) / 10, 1.0)
        line_factor = min(git_metrics.get('net_lines_added', 0) / 100, 1.0)
        
        return round((session_factor + commit_factor + line_factor) / 3, 2)
    
    def _analyze_productivity_trends(self, sessions: List[Dict], commits: List[Dict]) -> Dict[str, Any]:
        """Analyze productivity trends over time."""
        trends = {
            "session_trend": self._calculate_session_trend(sessions),
            "commit_trend": self._calculate_commit_trend(commits) if commits else {},
            "peak_productivity_hours": self._find_peak_hours(sessions, commits),
            "productivity_patterns": self._identify_patterns(sessions, commits)
        }
        
        return trends
    
    def _calculate_session_trend(self, sessions: List[Dict]) -> Dict[str, Any]:
        """Calculate session activity trends."""
        if len(sessions) < 2:
            return {"trend": "insufficient_data"}
        
        # Group sessions by day
        daily_sessions = defaultdict(int)
        for session in sessions:
            date = datetime.fromisoformat(session.get('created_at', '')).date()
            daily_sessions[date] += 1
        
        # Calculate trend
        dates = sorted(daily_sessions.keys())
        values = [daily_sessions[date] for date in dates]
        
        if len(values) >= 3:
            # Simple trend calculation (linear regression would be better)
            first_half = values[:len(values)//2]
            second_half = values[len(values)//2:]
            
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            
            trend_direction = "increasing" if second_avg > first_avg else "decreasing" if second_avg < first_avg else "stable"
            trend_strength = abs(second_avg - first_avg) / max(first_avg, 0.1)
            
            return {
                "trend": trend_direction,
                "strength": round(trend_strength, 2),
                "daily_average": round(sum(values) / len(values), 2),
                "peak_day": max(daily_sessions, key=daily_sessions.get).isoformat(),
                "peak_sessions": max(daily_sessions.values())
            }
        
        return {"trend": "insufficient_data"}
    
    def _calculate_commit_trend(self, commits: List[Dict]) -> Dict[str, Any]:
        """Calculate commit activity trends."""
        if len(commits) < 2:
            return {"trend": "insufficient_data"}
        
        # Group commits by day
        daily_commits = defaultdict(int)
        daily_lines = defaultdict(int)
        
        for commit in commits:
            try:
                date = datetime.fromisoformat(commit.get('commit_timestamp', '')).date()
                daily_commits[date] += 1
                daily_lines[date] += commit.get('lines_added', 0) + commit.get('lines_deleted', 0)
            except:
                continue
        
        return {
            "avg_commits_per_day": round(sum(daily_commits.values()) / len(daily_commits), 2),
            "avg_lines_per_day": round(sum(daily_lines.values()) / len(daily_lines), 2),
            "most_active_day": max(daily_commits, key=daily_commits.get).isoformat() if daily_commits else None,
            "commit_consistency": self._calculate_consistency_score(list(daily_commits.values()))
        }
    
    def _calculate_consistency_score(self, values: List[int]) -> float:
        """Calculate consistency score (lower variance = higher consistency)."""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5
        
        # Consistency score: 1.0 for no variation, approaches 0 for high variation
        consistency = 1.0 / (1.0 + std_dev / max(mean, 0.1))
        return round(consistency, 2)
    
    def _find_peak_hours(self, sessions: List[Dict], commits: List[Dict]) -> Dict[str, Any]:
        """Find peak productivity hours."""
        session_hours = defaultdict(int)
        commit_hours = defaultdict(int)
        
        # Analyze session times
        for session in sessions:
            try:
                hour = datetime.fromisoformat(session.get('created_at', '')).hour
                session_hours[hour] += 1
            except:
                continue
        
        # Analyze commit times
        for commit in commits:
            try:
                hour = datetime.fromisoformat(commit.get('commit_timestamp', '')).hour
                commit_hours[hour] += 1
            except:
                continue
        
        peak_session_hour = max(session_hours, key=session_hours.get) if session_hours else None
        peak_commit_hour = max(commit_hours, key=commit_hours.get) if commit_hours else None
        
        return {
            "peak_session_hour": peak_session_hour,
            "peak_commit_hour": peak_commit_hour,
            "session_hour_distribution": dict(session_hours),
            "commit_hour_distribution": dict(commit_hours)
        }
    
    def _identify_patterns(self, sessions: List[Dict], commits: List[Dict]) -> List[str]:
        """Identify productivity patterns."""
        patterns = []
        
        # Session patterns
        if len(sessions) >= 5:
            avg_duration = sum(self._get_session_duration(s) for s in sessions) / len(sessions)
            if avg_duration > 4:  # hours
                patterns.append("Long development sessions (>4h average)")
            elif avg_duration < 0.5:
                patterns.append("Short development bursts (<30m average)")
        
        # Commit patterns
        if commits:
            commit_types = Counter()
            for commit in commits:
                metadata = commit.get('commit_metadata', {})
                commit_type = metadata.get('commit_type', 'unknown')
                commit_types[commit_type] += 1
            
            most_common = commit_types.most_common(1)
            if most_common:
                common_type, count = most_common[0]
                if count / len(commits) > 0.5:
                    patterns.append(f"Focuses primarily on {common_type} commits")
        
        # Cross-correlation patterns
        if sessions and commits:
            sessions_with_commits = sum(1 for s in sessions if self._session_has_commits(s))
            if sessions_with_commits / len(sessions) > 0.8:
                patterns.append("High git correlation - most sessions result in commits")
            elif sessions_with_commits / len(sessions) < 0.3:
                patterns.append("Low git correlation - many sessions without commits")
        
        return patterns
    
    def _get_session_duration(self, session: Dict) -> float:
        """Get session duration in hours."""
        try:
            created = datetime.fromisoformat(session.get('created_at', ''))
            updated = datetime.fromisoformat(session.get('updated_at', session.get('created_at', '')))
            return (updated - created).total_seconds() / 3600
        except:
            return 0.0
    
    def _session_has_commits(self, session: Dict) -> bool:
        """Check if session has associated commits."""
        commits = self.db.get_session_git_commits(session['id'])
        return len(commits) > 0
    
    def _calculate_memory_system_impact(self, sessions: List[Dict]) -> Dict[str, Any]:
        """Calculate impact of memory system on productivity."""
        if not sessions:
            return {"impact": "no_data"}
        
        # Analyze context injection effectiveness
        context_injections = 0
        total_context_size = 0
        
        for session in sessions:
            injections = self.db.get_context_injections(session['id'])
            context_injections += len(injections)
            total_context_size += sum(inj.get('token_count', 0) for inj in injections)
        
        avg_context_per_session = context_injections / len(sessions) if sessions else 0
        avg_context_size = total_context_size / context_injections if context_injections else 0
        
        return {
            "total_context_injections": context_injections,
            "avg_context_per_session": round(avg_context_per_session, 2),
            "avg_context_size_tokens": round(avg_context_size, 0),
            "memory_utilization": self._calculate_memory_utilization(sessions),
            "effectiveness_score": self._calculate_memory_effectiveness(sessions)
        }
    
    def _calculate_memory_utilization(self, sessions: List[Dict]) -> float:
        """Calculate memory system utilization score."""
        if not sessions:
            return 0.0
        
        # Count sessions that actively used memory features
        memory_active_sessions = 0
        for session in sessions:
            # Check if session has context injections, checkpoints, or project memory updates
            has_context = len(self.db.get_context_injections(session['id'])) > 0
            has_tools = len(self.db.get_session_tool_usage(session['id'])) > 0
            
            if has_context or has_tools:
                memory_active_sessions += 1
        
        return round(memory_active_sessions / len(sessions), 2)
    
    def _calculate_memory_effectiveness(self, sessions: List[Dict]) -> float:
        """Calculate memory system effectiveness score."""
        # This would ideally compare productivity before/after memory system
        # For now, use proxy metrics
        effectiveness_factors = []
        
        # Factor 1: Context injection usage
        total_injections = sum(len(self.db.get_context_injections(s['id'])) for s in sessions)
        if total_injections > 0:
            effectiveness_factors.append(0.3)  # Base effectiveness for using context
        
        # Factor 2: Session continuity (checkpoints, cross-references)
        sessions_with_continuity = sum(1 for s in sessions if len(self.db.get_session_tool_usage(s['id'])) > 5)
        if sessions_with_continuity / len(sessions) > 0.5:
            effectiveness_factors.append(0.4)  # Good session engagement
        
        # Factor 3: Git correlation
        sessions_with_git = sum(1 for s in sessions if self._session_has_commits(s))
        if sessions_with_git / len(sessions) > 0.6:
            effectiveness_factors.append(0.3)  # Good git integration
        
        return round(sum(effectiveness_factors), 2)
    
    def _calculate_session_productivity_score(self, sessions: List[Dict]) -> float:
        """Calculate overall session productivity score."""
        if not sessions:
            return 0.0
        
        scores = []
        for session in sessions:
            session_score = 0.0
            
            # Tool usage factor
            tools = self.db.get_session_tool_usage(session['id'])
            tool_score = min(len(tools) * 0.1, 1.0)
            session_score += tool_score * 0.4
            
            # Duration factor (optimal around 2-4 hours)
            duration = self._get_session_duration(session)
            if 1.0 <= duration <= 6.0:
                duration_score = 1.0 - abs(duration - 3.0) / 3.0  # Peak at 3 hours
            else:
                duration_score = 0.2
            session_score += duration_score * 0.3
            
            # Git correlation factor
            has_commits = self._session_has_commits(session)
            commit_score = 1.0 if has_commits else 0.3
            session_score += commit_score * 0.3
            
            scores.append(session_score)
        
        return round(sum(scores) / len(scores), 2)
    
    def _store_velocity_metrics(self, velocity_metrics: Dict, session_id: str = None):
        """Store velocity metrics in database."""
        try:
            # Store overall productivity score
            overall_score = velocity_metrics.get('session_metrics', {}).get('productivity_score', 0)
            self.db.record_performance_metric(
                'development_velocity',
                overall_score,
                session_id,
                velocity_metrics
            )
            
            # Store specific metrics
            session_metrics = velocity_metrics.get('session_metrics', {})
            git_metrics = velocity_metrics.get('git_metrics', {})
            
            if session_metrics.get('total_sessions', 0) > 0:
                self.db.record_performance_metric(
                    'sessions_per_period',
                    session_metrics['total_sessions'],
                    session_id,
                    {'period_days': velocity_metrics['time_period_days']}
                )
            
            if git_metrics.get('total_commits', 0) > 0:
                self.db.record_performance_metric(
                    'commits_per_period',
                    git_metrics['total_commits'],
                    session_id,
                    {'period_days': velocity_metrics['time_period_days']}
                )
            
        except Exception as e:
            logging.error(f"Failed to store velocity metrics: {e}")
    
    def identify_recurring_patterns(self, lookback_days: int = 30) -> Dict[str, Any]:
        """Identify recurring development patterns."""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=lookback_days)
            
            # Get data for analysis
            sessions = self._get_sessions_in_period(start_time, end_time)
            commits = self._get_commits_in_period(start_time, end_time) if self.git_available else []
            
            patterns = {
                "analysis_period_days": lookback_days,
                "temporal_patterns": self._analyze_temporal_patterns(sessions, commits),
                "workflow_patterns": self._analyze_workflow_patterns(sessions),
                "collaboration_patterns": self._analyze_collaboration_patterns(commits),
                "productivity_cycles": self._identify_productivity_cycles(sessions, commits),
                "recommendations": []
            }
            
            # Generate recommendations based on patterns
            patterns["recommendations"] = self._generate_pattern_recommendations(patterns)
            
            return patterns
            
        except Exception as e:
            logging.error(f"Failed to identify recurring patterns: {e}")
            return {"error": str(e)}
    
    def _analyze_temporal_patterns(self, sessions: List[Dict], commits: List[Dict]) -> Dict[str, Any]:
        """Analyze temporal patterns in development activity."""
        # Day of week patterns
        weekday_sessions = defaultdict(int)
        weekday_commits = defaultdict(int)
        
        for session in sessions:
            try:
                weekday = datetime.fromisoformat(session.get('created_at', '')).weekday()
                weekday_sessions[weekday] += 1
            except:
                continue
        
        for commit in commits:
            try:
                weekday = datetime.fromisoformat(commit.get('commit_timestamp', '')).weekday()
                weekday_commits[weekday] += 1
            except:
                continue
        
        # Time of day patterns (already implemented in _find_peak_hours)
        peak_hours = self._find_peak_hours(sessions, commits)
        
        return {
            "weekday_distribution": {
                "sessions": dict(weekday_sessions),
                "commits": dict(weekday_commits)
            },
            "peak_hours": peak_hours,
            "most_active_weekday": max(weekday_sessions, key=weekday_sessions.get) if weekday_sessions else None,
            "weekend_activity": {
                "sessions": weekday_sessions.get(5, 0) + weekday_sessions.get(6, 0),
                "commits": weekday_commits.get(5, 0) + weekday_commits.get(6, 0)
            }
        }
    
    def _analyze_workflow_patterns(self, sessions: List[Dict]) -> Dict[str, Any]:
        """Analyze workflow patterns from sessions."""
        tool_sequences = []
        session_lengths = []
        
        for session in sessions:
            tools = self.db.get_session_tool_usage(session['id'])
            if tools:
                tool_sequence = [tool.get('tool_name', 'unknown') for tool in tools]
                tool_sequences.append(tool_sequence)
            
            duration = self._get_session_duration(session)
            session_lengths.append(duration)
        
        # Analyze common tool patterns
        tool_transitions = defaultdict(int)
        for sequence in tool_sequences:
            for i in range(len(sequence) - 1):
                transition = (sequence[i], sequence[i + 1])
                tool_transitions[transition] += 1
        
        common_transitions = sorted(tool_transitions.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "avg_session_length_hours": round(sum(session_lengths) / len(session_lengths), 2) if session_lengths else 0,
            "common_tool_transitions": [
                {"from": t[0][0], "to": t[0][1], "count": t[1]} 
                for t in common_transitions
            ],
            "session_length_distribution": self._categorize_session_lengths(session_lengths)
        }
    
    def _categorize_session_lengths(self, lengths: List[float]) -> Dict[str, int]:
        """Categorize session lengths."""
        categories = {
            "short (< 30m)": 0,
            "medium (30m-2h)": 0,
            "long (2h-4h)": 0,
            "extended (> 4h)": 0
        }
        
        for length in lengths:
            if length < 0.5:
                categories["short (< 30m)"] += 1
            elif length < 2.0:
                categories["medium (30m-2h)"] += 1
            elif length < 4.0:
                categories["long (2h-4h)"] += 1
            else:
                categories["extended (> 4h)"] += 1
        
        return categories
    
    def _analyze_collaboration_patterns(self, commits: List[Dict]) -> Dict[str, Any]:
        """Analyze collaboration patterns from commits."""
        if not commits:
            return {"collaboration_score": 0}
        
        authors = Counter()
        branches = Counter()
        
        for commit in commits:
            author = commit.get('author_email', 'unknown')
            authors[author] += 1
            
            branch = commit.get('branch_name', 'unknown')
            branches[branch] += 1
        
        return {
            "unique_authors": len(authors),
            "author_distribution": dict(authors.most_common(10)),
            "branch_diversity": len(branches),
            "branch_distribution": dict(branches.most_common(10)),
            "collaboration_score": round(1.0 / len(authors), 2) if len(authors) > 1 else 0.0
        }
    
    def _identify_productivity_cycles(self, sessions: List[Dict], commits: List[Dict]) -> Dict[str, Any]:
        """Identify productivity cycles and rhythms."""
        # Group activity by week
        weekly_activity = defaultdict(lambda: {"sessions": 0, "commits": 0, "lines": 0})
        
        for session in sessions:
            try:
                week_start = datetime.fromisoformat(session.get('created_at', '')).date()
                # Get Monday of that week
                week_start = week_start - timedelta(days=week_start.weekday())
                weekly_activity[week_start]["sessions"] += 1
            except:
                continue
        
        for commit in commits:
            try:
                week_start = datetime.fromisoformat(commit.get('commit_timestamp', '')).date()
                week_start = week_start - timedelta(days=week_start.weekday())
                weekly_activity[week_start]["commits"] += 1
                weekly_activity[week_start]["lines"] += commit.get('lines_added', 0) + commit.get('lines_deleted', 0)
            except:
                continue
        
        # Identify patterns in weekly activity
        weeks = sorted(weekly_activity.keys())
        if len(weeks) >= 3:
            session_values = [weekly_activity[week]["sessions"] for week in weeks]
            commit_values = [weekly_activity[week]["commits"] for week in weeks]
            
            return {
                "cycle_length_weeks": len(weeks),
                "avg_sessions_per_week": round(sum(session_values) / len(session_values), 2),
                "avg_commits_per_week": round(sum(commit_values) / len(commit_values), 2),
                "productivity_consistency": self._calculate_consistency_score(session_values),
                "peak_productivity_week": weeks[session_values.index(max(session_values))].isoformat() if session_values else None,
                "weekly_trend": self._calculate_weekly_trend(session_values)
            }
        
        return {"insufficient_data": True}
    
    def _calculate_weekly_trend(self, values: List[int]) -> str:
        """Calculate trend over weekly values."""
        if len(values) < 3:
            return "insufficient_data"
        
        # Simple trend calculation
        first_third = values[:len(values)//3] or [0]
        last_third = values[-len(values)//3:] or [0]
        
        first_avg = sum(first_third) / len(first_third)
        last_avg = sum(last_third) / len(last_third)
        
        if last_avg > first_avg * 1.2:
            return "increasing"
        elif last_avg < first_avg * 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def _generate_pattern_recommendations(self, patterns: Dict) -> List[str]:
        """Generate recommendations based on identified patterns."""
        recommendations = []
        
        temporal = patterns.get("temporal_patterns", {})
        workflow = patterns.get("workflow_patterns", {})
        cycles = patterns.get("productivity_cycles", {})
        
        # Temporal recommendations
        peak_session_hour = temporal.get("peak_hours", {}).get("peak_session_hour")
        if peak_session_hour is not None:
            recommendations.append(f"Your peak productivity time is {peak_session_hour}:00. Schedule important work during this hour.")
        
        weekend_activity = temporal.get("weekend_activity", {}).get("sessions", 0)
        weekday_total = sum(temporal.get("weekday_distribution", {}).get("sessions", {}).values()) - weekend_activity
        if weekend_activity / max(weekday_total, 1) > 0.3:
            recommendations.append("You're quite active on weekends. Consider taking breaks to prevent burnout.")
        
        # Workflow recommendations
        avg_length = workflow.get("avg_session_length_hours", 0)
        if avg_length > 4:
            recommendations.append("Your sessions are quite long. Consider taking breaks every 2-3 hours.")
        elif avg_length < 0.5:
            recommendations.append("Your sessions are very short. Consider batching similar tasks for better flow.")
        
        # Productivity cycle recommendations
        consistency = cycles.get("productivity_consistency", 0)
        if consistency < 0.5:
            recommendations.append("Your productivity varies significantly. Try to establish more consistent work rhythms.")
        
        trend = cycles.get("weekly_trend", "")
        if trend == "decreasing":
            recommendations.append("Productivity trend is declining. Consider reviewing your current workflow and tools.")
        elif trend == "increasing":
            recommendations.append("Great! Your productivity is trending upward. Keep up the current practices.")
        
        if not recommendations:
            recommendations.append("Keep up the good development practices. Your patterns show consistent productive work.")
        
        return recommendations
    
    def generate_productivity_report(self, time_period_days: int = 7, session_id: str = None) -> Dict[str, Any]:
        """Generate comprehensive productivity report."""
        try:
            # Get velocity metrics
            velocity = self.measure_development_velocity(time_period_days, session_id)
            
            # Get pattern analysis
            patterns = self.identify_recurring_patterns(time_period_days * 2)  # Look back twice as far for patterns
            
            # Combine into comprehensive report
            report = {
                "report_date": datetime.now().isoformat(),
                "analysis_period_days": time_period_days,
                "session_focus": session_id,
                "velocity_analysis": velocity,
                "pattern_analysis": patterns,
                "summary": self._generate_productivity_summary(velocity, patterns),
                "action_items": self._generate_action_items(velocity, patterns)
            }
            
            return report
            
        except Exception as e:
            logging.error(f"Failed to generate productivity report: {e}")
            return {"error": str(e)}
    
    def _generate_productivity_summary(self, velocity: Dict, patterns: Dict) -> Dict[str, Any]:
        """Generate productivity summary."""
        session_metrics = velocity.get('session_metrics', {})
        git_metrics = velocity.get('git_metrics', {})
        
        summary = {
            "overall_score": session_metrics.get('productivity_score', 0),
            "activity_level": self._categorize_activity_level(session_metrics, git_metrics),
            "key_strengths": [],
            "improvement_areas": []
        }
        
        # Identify strengths
        if session_metrics.get('productivity_score', 0) > 0.7:
            summary["key_strengths"].append("High session productivity")
        
        if git_metrics.get('code_churn', 1) < 0.3:
            summary["key_strengths"].append("Low code churn - quality focused development")
        
        memory_impact = velocity.get('memory_impact', {})
        if memory_impact.get('effectiveness_score', 0) > 0.6:
            summary["key_strengths"].append("Effective memory system utilization")
        
        # Identify improvement areas
        if session_metrics.get('avg_session_duration_hours', 0) > 5:
            summary["improvement_areas"].append("Session lengths may be too long")
        
        if git_metrics.get('total_commits', 0) == 0:
            summary["improvement_areas"].append("No git commits in period - consider more frequent commits")
        
        if memory_impact.get('memory_utilization', 0) < 0.3:
            summary["improvement_areas"].append("Low memory system utilization")
        
        return summary
    
    def _categorize_activity_level(self, session_metrics: Dict, git_metrics: Dict) -> str:
        """Categorize overall activity level."""
        sessions = session_metrics.get('total_sessions', 0)
        commits = git_metrics.get('total_commits', 0)
        
        total_activity = sessions + commits * 2  # Weight commits higher
        
        if total_activity >= 20:
            return "very_high"
        elif total_activity >= 10:
            return "high"
        elif total_activity >= 5:
            return "moderate"
        elif total_activity >= 2:
            return "low"
        else:
            return "very_low"
    
    def _generate_action_items(self, velocity: Dict, patterns: Dict) -> List[str]:
        """Generate actionable improvement items."""
        action_items = []
        
        # From velocity analysis
        memory_impact = velocity.get('memory_impact', {})
        if memory_impact.get('memory_utilization', 0) < 0.5:
            action_items.append("Increase usage of memory system features like checkpoints and context injection")
        
        session_metrics = velocity.get('session_metrics', {})
        if session_metrics.get('avg_session_duration_hours', 0) > 4:
            action_items.append("Break long sessions into shorter focused blocks with breaks")
        
        # From pattern analysis
        recommendations = patterns.get('recommendations', [])
        action_items.extend(recommendations[:3])  # Take top 3 recommendations
        
        # Add general productivity improvements
        git_metrics = velocity.get('git_metrics', {})
        if git_metrics.get('total_commits', 0) > 0:
            commit_types = git_metrics.get('commit_types', {})
            if commit_types.get('feature', 0) > commit_types.get('test', 0) * 2:
                action_items.append("Consider writing more tests relative to features")
        
        return action_items[:5]  # Limit to 5 action items


# CLI interface for productivity tracking
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: productivity_tracker.py <command> [args...]")
        sys.exit(1)
    
    command = sys.argv[1]
    memory_dir = os.environ.get("PRSIST_MEMORY_DIR", os.path.dirname(__file__))
    
    tracker = ProductivityTracker(memory_dir)
    
    if command == "velocity":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
        session_id = sys.argv[3] if len(sys.argv) > 3 else None
        result = tracker.measure_development_velocity(days, session_id)
        print(json.dumps(result, indent=2))
    
    elif command == "patterns":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        result = tracker.identify_recurring_patterns(days)
        print(json.dumps(result, indent=2))
    
    elif command == "report":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
        session_id = sys.argv[3] if len(sys.argv) > 3 else None
        result = tracker.generate_productivity_report(days, session_id)
        print(json.dumps(result, indent=2))
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)