#!/usr/bin/env python3
"""
Correlation engine for Prsist Memory System.
Memory-git correlation algorithms and relationship analysis.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import re

from database import MemoryDatabase
from git_integration import GitMetadataExtractor, ChangeImpactAnalyzer
from utils import sanitize_input

class CorrelationEngine:
    """Correlates memory sessions with git commits and analyzes relationships."""
    
    def __init__(self, memory_db: MemoryDatabase, git_extractor: GitMetadataExtractor):
        """Initialize correlation engine."""
        self.db = memory_db
        self.git_extractor = git_extractor
        self.impact_analyzer = ChangeImpactAnalyzer(git_extractor)
        
    def correlate_commit_with_sessions(self, commit_sha: str) -> Dict[str, Any]:
        """Correlate a commit with memory sessions."""
        try:
            # Get commit metadata
            commit_metadata = self.git_extractor.get_commit_metadata(commit_sha)
            if not commit_metadata:
                logging.error(f"Could not extract metadata for commit {commit_sha}")
                return {"success": False, "error": "Failed to extract commit metadata"}
            
            # Analyze commit impact
            impact_analysis = self.impact_analyzer.analyze_commit_impact(commit_metadata)
            
            # Find related sessions
            related_sessions = self.find_related_sessions(commit_metadata)
            
            # Store commit in database
            success = self.store_commit_data(commit_metadata, impact_analysis)
            if not success:
                logging.error(f"Failed to store commit data for {commit_sha}")
                return {"success": False, "error": "Failed to store commit data"}
            
            # Create correlations with sessions
            correlations_created = 0
            for session_data in related_sessions:
                correlation_created = self.create_session_commit_correlation(
                    session_data, commit_metadata, impact_analysis
                )
                if correlation_created:
                    correlations_created += 1
            
            result = {
                "success": True,
                "commit_sha": commit_sha,
                "commit_metadata": commit_metadata,
                "impact_analysis": impact_analysis,
                "related_sessions": len(related_sessions),
                "correlations_created": correlations_created,
                "correlation_summary": self.generate_correlation_summary(
                    commit_metadata, related_sessions, impact_analysis
                )
            }
            
            logging.info(f"Successfully correlated commit {commit_sha[:8]} with {len(related_sessions)} sessions")
            return result
            
        except Exception as e:
            logging.error(f"Failed to correlate commit {commit_sha}: {e}")
            return {"success": False, "error": str(e)}
    
    def find_related_sessions(self, commit_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find memory sessions related to a commit."""
        commit_timestamp = datetime.fromisoformat(commit_metadata["timestamp"])
        
        # Search window: 6 hours before and 1 hour after commit
        search_start = commit_timestamp - timedelta(hours=6)
        search_end = commit_timestamp + timedelta(hours=1)
        
        related_sessions = []
        
        try:
            # Get sessions in time window
            time_window_sessions = self.get_sessions_in_timeframe(search_start, search_end)
            
            for session in time_window_sessions:
                correlation_strength = self.calculate_session_commit_correlation(
                    session, commit_metadata
                )
                
                if correlation_strength > 0.3:  # Minimum correlation threshold
                    session["correlation_strength"] = correlation_strength
                    related_sessions.append(session)
            
            # Sort by correlation strength
            related_sessions.sort(key=lambda x: x["correlation_strength"], reverse=True)
            
            return related_sessions
            
        except Exception as e:
            logging.error(f"Failed to find related sessions: {e}")
            return []
    
    def calculate_session_commit_correlation(self, session: Dict[str, Any], 
                                           commit_metadata: Dict[str, Any]) -> float:
        """Calculate correlation strength between a session and commit."""
        correlation_score = 0.0
        
        try:
            # Time correlation (closer in time = higher correlation)
            time_correlation = self.calculate_time_correlation(session, commit_metadata)
            correlation_score += time_correlation * 0.3
            
            # File interaction correlation
            file_correlation = self.calculate_file_correlation(session, commit_metadata)
            correlation_score += file_correlation * 0.4
            
            # Tool usage correlation
            tool_correlation = self.calculate_tool_correlation(session, commit_metadata)
            correlation_score += tool_correlation * 0.2
            
            # Branch correlation
            branch_correlation = self.calculate_branch_correlation(session, commit_metadata)
            correlation_score += branch_correlation * 0.1
            
            return min(correlation_score, 1.0)
            
        except Exception as e:
            logging.error(f"Failed to calculate correlation: {e}")
            return 0.0
    
    def calculate_time_correlation(self, session: Dict[str, Any], 
                                 commit_metadata: Dict[str, Any]) -> float:
        """Calculate time-based correlation."""
        try:
            commit_time = datetime.fromisoformat(commit_metadata["timestamp"])
            session_start = datetime.fromisoformat(session["created_at"])
            session_end = session.get("ended_at")
            
            if session_end:
                session_end = datetime.fromisoformat(session_end)
            else:
                # Assume session is still active or ended recently
                session_end = session_start + timedelta(hours=4)
            
            # Check if commit happened during session
            if session_start <= commit_time <= session_end:
                return 1.0
            
            # Calculate proximity to session
            if commit_time < session_start:
                time_diff = (session_start - commit_time).total_seconds()
            else:
                time_diff = (commit_time - session_end).total_seconds()
            
            # Decay based on time difference (max 6 hours)
            max_diff = 6 * 3600  # 6 hours in seconds
            if time_diff > max_diff:
                return 0.0
            
            return 1.0 - (time_diff / max_diff)
            
        except Exception as e:
            logging.error(f"Failed to calculate time correlation: {e}")
            return 0.0
    
    def calculate_file_correlation(self, session: Dict[str, Any], 
                                 commit_metadata: Dict[str, Any]) -> float:
        """Calculate file-based correlation."""
        try:
            # Get session file interactions
            session_id = session["id"]
            session_files = self.get_session_file_interactions(session_id)
            
            if not session_files:
                return 0.0
            
            # Get commit file changes
            commit_files = {fc["file_path"] for fc in commit_metadata.get("file_changes", [])}
            
            if not commit_files:
                return 0.0
            
            # Calculate overlap
            session_file_paths = {fi["file_path"] for fi in session_files}
            overlap = len(session_file_paths.intersection(commit_files))
            total_unique = len(session_file_paths.union(commit_files))
            
            if total_unique == 0:
                return 0.0
            
            # Jaccard similarity
            jaccard = overlap / total_unique
            
            # Boost for exact matches
            exact_matches = overlap
            boost = min(exact_matches * 0.2, 0.5)
            
            return min(jaccard + boost, 1.0)
            
        except Exception as e:
            logging.error(f"Failed to calculate file correlation: {e}")
            return 0.0
    
    def calculate_tool_correlation(self, session: Dict[str, Any], 
                                 commit_metadata: Dict[str, Any]) -> float:
        """Calculate tool usage correlation."""
        try:
            # Get session tool usage
            session_id = session["id"]
            tool_usage = self.db.get_session_tool_usage(session_id)
            
            if not tool_usage:
                return 0.0
            
            # Analyze tool patterns that suggest commit preparation
            commit_related_tools = {
                "Edit", "MultiEdit", "Write", "Read", "Bash"
            }
            
            commit_tools = [tu for tu in tool_usage 
                          if tu["tool_name"] in commit_related_tools]
            
            if not commit_tools:
                return 0.2  # Low baseline for any session activity
            
            # Calculate intensity near commit time
            commit_time = datetime.fromisoformat(commit_metadata["timestamp"])
            
            recent_tools = []
            for tool in commit_tools:
                tool_time = datetime.fromisoformat(tool["timestamp"])
                time_diff = abs((commit_time - tool_time).total_seconds())
                
                if time_diff <= 3600:  # Within 1 hour
                    recent_tools.append(tool)
            
            if not recent_tools:
                return 0.3
            
            # Score based on tool diversity and recency
            tool_types = set(rt["tool_name"] for rt in recent_tools)
            diversity_score = min(len(tool_types) * 0.2, 0.6)
            
            # Boost for edit-heavy activity
            edit_tools = sum(1 for rt in recent_tools 
                           if rt["tool_name"] in ["Edit", "MultiEdit", "Write"])
            edit_score = min(edit_tools * 0.1, 0.4)
            
            return min(diversity_score + edit_score, 1.0)
            
        except Exception as e:
            logging.error(f"Failed to calculate tool correlation: {e}")
            return 0.0
    
    def calculate_branch_correlation(self, session: Dict[str, Any], 
                                   commit_metadata: Dict[str, Any]) -> float:
        """Calculate branch-based correlation."""
        try:
            # Get session git info
            session_git_info = session.get("git_info", {})
            session_branch = session_git_info.get("branch", "")
            
            # Get commit branches
            commit_branches = commit_metadata.get("branches", [])
            
            if not session_branch or not commit_branches:
                return 0.5  # Neutral if branch info unavailable
            
            # Check for exact branch match
            if session_branch in commit_branches:
                return 1.0
            
            # Check for related branches (feature branches, etc.)
            branch_similarity = self.calculate_branch_similarity(session_branch, commit_branches)
            return branch_similarity
            
        except Exception as e:
            logging.error(f"Failed to calculate branch correlation: {e}")
            return 0.5
    
    def calculate_branch_similarity(self, session_branch: str, commit_branches: List[str]) -> float:
        """Calculate similarity between session branch and commit branches."""
        max_similarity = 0.0
        
        for commit_branch in commit_branches:
            # Remove remote prefixes for comparison
            clean_session = session_branch.replace("origin/", "").replace("remotes/", "")
            clean_commit = commit_branch.replace("origin/", "").replace("remotes/", "")
            
            # Exact match
            if clean_session == clean_commit:
                return 1.0
            
            # Check for common patterns
            similarity = 0.0
            
            # Same feature prefix (feature/xxx)
            if "/" in clean_session and "/" in clean_commit:
                session_prefix = clean_session.split("/")[0]
                commit_prefix = clean_commit.split("/")[0]
                if session_prefix == commit_prefix:
                    similarity = 0.7
            
            # Similar names (substring match)
            if clean_session in clean_commit or clean_commit in clean_session:
                similarity = max(similarity, 0.5)
            
            # Common words
            session_words = set(re.findall(r'\w+', clean_session.lower()))
            commit_words = set(re.findall(r'\w+', clean_commit.lower()))
            
            if session_words and commit_words:
                word_overlap = len(session_words.intersection(commit_words))
                word_union = len(session_words.union(commit_words))
                word_similarity = word_overlap / word_union if word_union > 0 else 0
                similarity = max(similarity, word_similarity * 0.6)
            
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def store_commit_data(self, commit_metadata: Dict[str, Any], 
                         impact_analysis: Dict[str, Any]) -> bool:
        """Store commit data in the database."""
        try:
            # Store main commit record
            success = self.db.record_commit(
                commit_sha=commit_metadata["commit_sha"],
                session_id=None,  # Will be set via correlations
                branch_name=commit_metadata.get("branches", ["unknown"])[0],
                commit_message=sanitize_input(commit_metadata["subject"]),
                author_email=sanitize_input(commit_metadata["author_email"]),
                commit_timestamp=commit_metadata["timestamp"],
                changed_files_count=len(commit_metadata.get("file_changes", [])),
                lines_added=commit_metadata.get("stats", {}).get("insertions", 0),
                lines_deleted=commit_metadata.get("stats", {}).get("deletions", 0),
                memory_impact_score=commit_metadata["impact_score"],
                commit_metadata={
                    "author_name": commit_metadata["author_name"],
                    "commit_type": commit_metadata["commit_type"],
                    "parent_commits": commit_metadata["parent_commits"],
                    "impact_analysis": impact_analysis
                }
            )
            
            if not success:
                return False
            
            # Store file changes
            for file_change in commit_metadata.get("file_changes", []):
                self.db.record_file_change(
                    commit_sha=commit_metadata["commit_sha"],
                    file_path=file_change["file_path"],
                    change_type=file_change["change_type"],
                    lines_added=file_change.get("lines_added", 0),
                    lines_deleted=file_change.get("lines_deleted", 0),
                    significance_score=file_change.get("significance_score", 0),
                    context_summary=f"{file_change['file_type']} file, "
                                  f"test: {file_change.get('is_test', False)}, "
                                  f"config: {file_change.get('is_config', False)}, "
                                  f"docs: {file_change.get('is_documentation', False)}"
                )
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to store commit data: {e}")
            return False
    
    def create_session_commit_correlation(self, session_data: Dict[str, Any], 
                                        commit_metadata: Dict[str, Any],
                                        impact_analysis: Dict[str, Any]) -> bool:
        """Create correlation between session and commit."""
        try:
            correlation_strength = session_data["correlation_strength"]
            
            # Determine correlation type
            if correlation_strength >= 0.8:
                correlation_type = "direct"
            elif correlation_strength >= 0.5:
                correlation_type = "related"
            else:
                correlation_type = "background"
            
            # Prepare context overlap data
            context_overlap = {
                "time_correlation": self.calculate_time_correlation(session_data, commit_metadata),
                "file_correlation": self.calculate_file_correlation(session_data, commit_metadata),
                "tool_correlation": self.calculate_tool_correlation(session_data, commit_metadata),
                "branch_correlation": self.calculate_branch_correlation(session_data, commit_metadata)
            }
            
            # Analysis metadata
            analysis_metadata = {
                "commit_type": commit_metadata["commit_type"],
                "impact_score": commit_metadata["impact_score"],
                "files_changed": len(commit_metadata.get("file_changes", [])),
                "risk_level": impact_analysis.get("risk_assessment", {}).get("level", "unknown"),
                "quality_score": impact_analysis.get("quality_indicators", {}).get("quality_score", 0)
            }
            
            return self.db.create_commit_correlation(
                session_id=session_data["id"],
                commit_sha=commit_metadata["commit_sha"],
                correlation_type=correlation_type,
                correlation_strength=correlation_strength,
                context_overlap=context_overlap,
                analysis_metadata=analysis_metadata
            )
            
        except Exception as e:
            logging.error(f"Failed to create session-commit correlation: {e}")
            return False
    
    def generate_correlation_summary(self, commit_metadata: Dict[str, Any], 
                                   related_sessions: List[Dict[str, Any]],
                                   impact_analysis: Dict[str, Any]) -> str:
        """Generate human-readable correlation summary."""
        try:
            commit_sha = commit_metadata["commit_sha"][:8]
            commit_message = commit_metadata["subject"]
            commit_type = commit_metadata["commit_type"]
            
            summary_parts = [
                f"Commit {commit_sha}: {commit_message}",
                f"Type: {commit_type.title()}"
            ]
            
            # Add impact information
            impact_score = commit_metadata["impact_score"]
            risk_level = impact_analysis.get("risk_assessment", {}).get("level", "unknown")
            
            summary_parts.append(f"Impact: {impact_score:.2f}, Risk: {risk_level}")
            
            # Add session correlations
            if related_sessions:
                strong_correlations = [s for s in related_sessions if s["correlation_strength"] >= 0.7]
                moderate_correlations = [s for s in related_sessions if 0.4 <= s["correlation_strength"] < 0.7]
                
                if strong_correlations:
                    summary_parts.append(f"Strong correlations: {len(strong_correlations)} sessions")
                
                if moderate_correlations:
                    summary_parts.append(f"Moderate correlations: {len(moderate_correlations)} sessions")
            else:
                summary_parts.append("No session correlations found")
            
            # Add file change summary
            file_changes = commit_metadata.get("file_changes", [])
            if file_changes:
                test_files = sum(1 for fc in file_changes if fc.get("is_test", False))
                doc_files = sum(1 for fc in file_changes if fc.get("is_documentation", False))
                config_files = sum(1 for fc in file_changes if fc.get("is_config", False))
                
                change_summary = f"{len(file_changes)} files"
                if test_files:
                    change_summary += f", {test_files} tests"
                if doc_files:
                    change_summary += f", {doc_files} docs"
                if config_files:
                    change_summary += f", {config_files} config"
                
                summary_parts.append(f"Changes: {change_summary}")
            
            return " | ".join(summary_parts)
            
        except Exception as e:
            logging.error(f"Failed to generate correlation summary: {e}")
            return f"Commit {commit_metadata.get('commit_sha', 'unknown')[:8]} processed"
    
    def get_sessions_in_timeframe(self, start_time: datetime, 
                                end_time: datetime) -> List[Dict[str, Any]]:
        """Get sessions that overlap with the given timeframe."""
        try:
            # Get recent sessions (broader than timeframe for safety)
            recent_sessions = self.db.get_recent_sessions(50)
            
            relevant_sessions = []
            
            for session in recent_sessions:
                session_start = datetime.fromisoformat(session["created_at"])
                session_end = session.get("updated_at")
                
                if session_end:
                    session_end = datetime.fromisoformat(session_end)
                else:
                    # Assume ongoing session or default duration
                    session_end = session_start + timedelta(hours=4)
                
                # Check for overlap
                if (session_start <= end_time and session_end >= start_time):
                    relevant_sessions.append(session)
            
            return relevant_sessions
            
        except Exception as e:
            logging.error(f"Failed to get sessions in timeframe: {e}")
            return []
    
    def get_session_file_interactions(self, session_id: str) -> List[Dict[str, Any]]:
        """Get file interactions for a session."""
        try:
            import sqlite3
            with sqlite3.connect(self.db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM file_interactions 
                    WHERE session_id = ?
                    ORDER BY timestamp ASC
                """, (session_id,))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logging.error(f"Failed to get session file interactions: {e}")
            return []
    
    def analyze_productivity_trends(self, days: int = 30) -> Dict[str, Any]:
        """Analyze productivity trends over time."""
        try:
            recent_commits = self.db.get_recent_commits(limit=days * 10)  # Estimate
            
            if not recent_commits:
                return {"error": "No commits found for analysis"}
            
            # Group by day
            daily_stats = {}
            
            for commit in recent_commits:
                commit_date = datetime.fromisoformat(commit["commit_timestamp"]).date()
                
                if commit_date not in daily_stats:
                    daily_stats[commit_date] = {
                        "commits": 0,
                        "files_changed": 0,
                        "lines_added": 0,
                        "lines_deleted": 0,
                        "impact_score": 0.0
                    }
                
                stats = daily_stats[commit_date]
                stats["commits"] += 1
                stats["files_changed"] += commit.get("changed_files_count", 0)
                stats["lines_added"] += commit.get("lines_added", 0)
                stats["lines_deleted"] += commit.get("lines_deleted", 0)
                stats["impact_score"] += commit.get("memory_impact_score", 0)
            
            # Calculate averages
            total_days = len(daily_stats)
            if total_days == 0:
                return {"error": "No data available for analysis"}
            
            avg_stats = {
                "avg_commits_per_day": sum(day["commits"] for day in daily_stats.values()) / total_days,
                "avg_files_per_day": sum(day["files_changed"] for day in daily_stats.values()) / total_days,
                "avg_lines_added_per_day": sum(day["lines_added"] for day in daily_stats.values()) / total_days,
                "avg_lines_deleted_per_day": sum(day["lines_deleted"] for day in daily_stats.values()) / total_days,
                "avg_impact_per_day": sum(day["impact_score"] for day in daily_stats.values()) / total_days,
                "active_days": total_days,
                "total_commits": sum(day["commits"] for day in daily_stats.values()),
                "total_files": sum(day["files_changed"] for day in daily_stats.values()),
                "total_lines_added": sum(day["lines_added"] for day in daily_stats.values()),
                "total_lines_deleted": sum(day["lines_deleted"] for day in daily_stats.values())
            }
            
            return {
                "analysis_period_days": days,
                "daily_stats": daily_stats,
                "summary": avg_stats
            }
            
        except Exception as e:
            logging.error(f"Failed to analyze productivity trends: {e}")
            return {"error": str(e)}