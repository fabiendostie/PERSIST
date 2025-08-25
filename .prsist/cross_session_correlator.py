#!/usr/bin/env python3
"""
Cross-Session Correlator for Prsist Memory System - Phase 3.
Builds relationships between sessions, commits, and context across time.
"""

import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter

from database import MemoryDatabase
from semantic_analyzer import SemanticAnalyzer


@dataclass
class CorrelationScore:
    """Represents a correlation score between two entities."""
    source_id: str
    source_type: str  # session, commit, file
    target_id: str
    target_type: str
    correlation_strength: float
    correlation_type: str  # semantic, temporal, contextual, behavioral
    factors: Dict[str, float]
    created_at: datetime
    metadata: Dict[str, Any] = None


@dataclass
class CorrelationCluster:
    """Represents a cluster of related sessions/commits."""
    cluster_id: str
    entity_ids: List[str]
    entity_types: List[str]
    cluster_strength: float
    primary_theme: str
    temporal_span: Dict[str, str]  # start_date, end_date
    shared_elements: List[str]


class CrossSessionCorrelator:
    """Correlates sessions, commits, and context across development timeline."""
    
    def __init__(self, memory_dir: str, repo_path: str = "."):
        """Initialize cross-session correlator."""
        self.memory_dir = Path(memory_dir)
        self.repo_path = Path(repo_path).resolve()
        
        # Initialize components
        self.db = MemoryDatabase(self.memory_dir / "storage" / "sessions.db")
        self.semantic_analyzer = SemanticAnalyzer(str(self.memory_dir), str(self.repo_path))
        
        # Correlation cache
        self._correlation_cache = {}
        self._cluster_cache = {}
        
        # Correlation thresholds
        self.correlation_thresholds = {
            "weak": 0.2,
            "moderate": 0.4,
            "strong": 0.7,
            "very_strong": 0.9
        }
        
        logging.info("Cross-Session Correlator initialized")
    
    def build_session_correlations(self, session_id: str = None, lookback_days: int = 30) -> Dict[str, Any]:
        """Build comprehensive correlations for a session or all recent sessions."""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=lookback_days)
            
            if session_id:
                target_sessions = [self.db.get_session(session_id)]
                target_sessions = [s for s in target_sessions if s]
            else:
                # Get all sessions in timeframe
                all_sessions = self.db.get_recent_sessions(limit=200)
                target_sessions = []
                for session in all_sessions:
                    try:
                        session_time = datetime.fromisoformat(session.get('created_at', ''))
                        if start_time <= session_time <= end_time:
                            target_sessions.append(session)
                    except:
                        continue
            
            if not target_sessions:
                return {"error": "No sessions found in timeframe"}
            
            correlations = []
            correlation_matrix = {}
            
            # Build correlations between all session pairs
            for i, session_a in enumerate(target_sessions):
                session_a_id = session_a['id']
                correlation_matrix[session_a_id] = {}
                
                for j, session_b in enumerate(target_sessions[i+1:], i+1):
                    session_b_id = session_b['id']
                    
                    # Calculate correlation
                    correlation = self._calculate_session_correlation(session_a, session_b)
                    
                    if correlation.correlation_strength >= self.correlation_thresholds["weak"]:
                        correlations.append(correlation)
                        correlation_matrix[session_a_id][session_b_id] = correlation.correlation_strength
                        
                        # Store in database
                        self.db.create_cross_session_relationship(
                            source_session_id=session_a_id,
                            target_session_id=session_b_id,
                            relationship_type=correlation.correlation_type,
                            relationship_strength=correlation.correlation_strength,
                            shared_elements=list(correlation.factors.keys())
                        )
            
            # Identify correlation clusters
            clusters = self._identify_correlation_clusters(correlations, target_sessions)
            
            # Build comprehensive correlation report
            correlation_report = {
                "analysis_date": end_time.isoformat(),
                "lookback_days": lookback_days,
                "session_focus": session_id,
                "sessions_analyzed": len(target_sessions),
                "correlations_found": len(correlations),
                "correlation_summary": self._summarize_correlations(correlations),
                "correlation_clusters": [asdict(cluster) for cluster in clusters],
                "strongest_correlations": self._get_strongest_correlations(correlations, limit=10),
                "correlation_insights": self._generate_correlation_insights(correlations, clusters)
            }
            
            return correlation_report
            
        except Exception as e:
            logging.error(f"Failed to build session correlations: {e}")
            return {"error": str(e)}
    
    def _calculate_session_correlation(self, session_a: Dict, session_b: Dict) -> CorrelationScore:
        """Calculate correlation between two sessions."""
        factors = {}
        
        # Temporal correlation
        temporal_factor = self._calculate_temporal_correlation(session_a, session_b)
        factors["temporal"] = temporal_factor
        
        # Semantic correlation
        semantic_factor = self._calculate_semantic_session_correlation(session_a, session_b)
        factors["semantic"] = semantic_factor
        
        # Contextual correlation (shared files, tools, etc.)
        contextual_factor = self._calculate_contextual_correlation(session_a, session_b)
        factors["contextual"] = contextual_factor
        
        # Behavioral correlation (similar patterns)
        behavioral_factor = self._calculate_behavioral_correlation(session_a, session_b)
        factors["behavioral"] = behavioral_factor
        
        # Git correlation (shared commits, branches)
        git_factor = self._calculate_git_correlation(session_a, session_b)
        factors["git"] = git_factor
        
        # Calculate weighted overall correlation
        weights = {
            "temporal": 0.15,
            "semantic": 0.30,
            "contextual": 0.25,
            "behavioral": 0.20,
            "git": 0.10
        }
        
        overall_strength = sum(factors[factor] * weights[factor] for factor in factors)
        
        # Determine correlation type
        max_factor = max(factors, key=factors.get)
        correlation_type = max_factor if factors[max_factor] > 0.3 else "weak"
        
        return CorrelationScore(
            source_id=session_a['id'],
            source_type="session",
            target_id=session_b['id'],
            target_type="session",
            correlation_strength=round(overall_strength, 3),
            correlation_type=correlation_type,
            factors=factors,
            created_at=datetime.now(),
            metadata={
                "session_a_date": session_a.get('created_at'),
                "session_b_date": session_b.get('created_at'),
                "time_gap_hours": self._calculate_time_gap(session_a, session_b)
            }
        )
    
    def _calculate_temporal_correlation(self, session_a: Dict, session_b: Dict) -> float:
        """Calculate temporal correlation between sessions."""
        try:
            time_a = datetime.fromisoformat(session_a.get('created_at', ''))
            time_b = datetime.fromisoformat(session_b.get('created_at', ''))
            
            time_gap = abs((time_a - time_b).total_seconds())
            
            # Higher correlation for sessions closer in time
            if time_gap < 3600:  # Within 1 hour
                return 0.9
            elif time_gap < 86400:  # Within 1 day
                return 0.7
            elif time_gap < 604800:  # Within 1 week
                return 0.5
            elif time_gap < 2592000:  # Within 1 month
                return 0.3
            else:
                return 0.1
                
        except:
            return 0.0
    
    def _calculate_semantic_session_correlation(self, session_a: Dict, session_b: Dict) -> float:
        """Calculate semantic correlation between sessions."""
        try:
            # Generate embeddings
            embedding_a = self.semantic_analyzer.generate_session_embedding(session_a['id'])
            embedding_b = self.semantic_analyzer.generate_session_embedding(session_b['id'])
            
            if not embedding_a or not embedding_b:
                return 0.0
            
            # Calculate similarity
            similarity = self.semantic_analyzer.calculate_similarity(embedding_a, embedding_b)
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logging.debug(f"Failed to calculate semantic correlation: {e}")
            return 0.0
    
    def _calculate_contextual_correlation(self, session_a: Dict, session_b: Dict) -> float:
        """Calculate contextual correlation (shared files, tools, directories)."""
        try:
            # Get tool usage for both sessions
            tools_a = self.db.get_session_tool_usage(session_a['id'])
            tools_b = self.db.get_session_tool_usage(session_b['id'])
            
            # Extract file paths and tool names
            files_a = set()
            tools_names_a = set()
            for tool in tools_a:
                tools_names_a.add(tool.get('tool_name', ''))
                tool_input = tool.get('input_data', {})
                if isinstance(tool_input, dict) and 'file_path' in tool_input:
                    files_a.add(tool_input['file_path'])
            
            files_b = set()
            tools_names_b = set()
            for tool in tools_b:
                tools_names_b.add(tool.get('tool_name', ''))
                tool_input = tool.get('input_data', {})
                if isinstance(tool_input, dict) and 'file_path' in tool_input:
                    files_b.add(tool_input['file_path'])
            
            # Calculate overlap
            file_overlap = len(files_a & files_b) / max(len(files_a | files_b), 1)
            tool_overlap = len(tools_names_a & tools_names_b) / max(len(tools_names_a | tools_names_b), 1)
            
            # Combine overlaps
            contextual_score = (file_overlap * 0.6 + tool_overlap * 0.4)
            return round(contextual_score, 3)
            
        except Exception as e:
            logging.debug(f"Failed to calculate contextual correlation: {e}")
            return 0.0
    
    def _calculate_behavioral_correlation(self, session_a: Dict, session_b: Dict) -> float:
        """Calculate behavioral correlation (similar usage patterns)."""
        try:
            # Get session durations
            duration_a = self._calculate_session_duration(session_a)
            duration_b = self._calculate_session_duration(session_b)
            
            # Duration similarity
            if duration_a > 0 and duration_b > 0:
                duration_similarity = 1.0 - min(abs(duration_a - duration_b) / max(duration_a, duration_b), 1.0)
            else:
                duration_similarity = 0.0
            
            # Tool usage patterns
            tools_a = self.db.get_session_tool_usage(session_a['id'])
            tools_b = self.db.get_session_tool_usage(session_b['id'])
            
            # Tool sequence similarity (simplified)
            tool_sequence_a = [tool.get('tool_name', '') for tool in tools_a]
            tool_sequence_b = [tool.get('tool_name', '') for tool in tools_b]
            
            sequence_similarity = self._calculate_sequence_similarity(tool_sequence_a, tool_sequence_b)
            
            # Activity level similarity
            activity_a = len(tools_a)
            activity_b = len(tools_b)
            if activity_a > 0 and activity_b > 0:
                activity_similarity = 1.0 - min(abs(activity_a - activity_b) / max(activity_a, activity_b), 1.0)
            else:
                activity_similarity = 0.0
            
            # Combine behavioral factors
            behavioral_score = (duration_similarity * 0.3 + sequence_similarity * 0.4 + activity_similarity * 0.3)
            return round(behavioral_score, 3)
            
        except Exception as e:
            logging.debug(f"Failed to calculate behavioral correlation: {e}")
            return 0.0
    
    def _calculate_git_correlation(self, session_a: Dict, session_b: Dict) -> float:
        """Calculate git-based correlation between sessions."""
        try:
            # Get commits for both sessions
            commits_a = self.db.get_session_git_commits(session_a['id'])
            commits_b = self.db.get_session_git_commits(session_b['id'])
            
            if not commits_a or not commits_b:
                return 0.0
            
            # Extract branches and commit types
            branches_a = set(c.get('branch_name', '') for c in commits_a if c.get('branch_name'))
            branches_b = set(c.get('branch_name', '') for c in commits_b if c.get('branch_name'))
            
            # Branch overlap
            branch_overlap = len(branches_a & branches_b) / max(len(branches_a | branches_b), 1)
            
            # Commit type similarity
            types_a = []
            types_b = []
            for commit in commits_a:
                metadata = commit.get('commit_metadata', {})
                if isinstance(metadata, dict):
                    types_a.append(metadata.get('commit_type', 'unknown'))
            
            for commit in commits_b:
                metadata = commit.get('commit_metadata', {})
                if isinstance(metadata, dict):
                    types_b.append(metadata.get('commit_type', 'unknown'))
            
            type_overlap = len(set(types_a) & set(types_b)) / max(len(set(types_a) | set(types_b)), 1)
            
            # Combine git factors
            git_score = (branch_overlap * 0.6 + type_overlap * 0.4)
            return round(git_score, 3)
            
        except Exception as e:
            logging.debug(f"Failed to calculate git correlation: {e}")
            return 0.0
    
    def _calculate_session_duration(self, session: Dict) -> float:
        """Calculate session duration in hours."""
        try:
            created = datetime.fromisoformat(session.get('created_at', ''))
            updated = datetime.fromisoformat(session.get('updated_at', session.get('created_at', '')))
            return (updated - created).total_seconds() / 3600
        except:
            return 0.0
    
    def _calculate_time_gap(self, session_a: Dict, session_b: Dict) -> float:
        """Calculate time gap between sessions in hours."""
        try:
            time_a = datetime.fromisoformat(session_a.get('created_at', ''))
            time_b = datetime.fromisoformat(session_b.get('created_at', ''))
            return abs((time_a - time_b).total_seconds()) / 3600
        except:
            return 0.0
    
    def _calculate_sequence_similarity(self, seq_a: List[str], seq_b: List[str]) -> float:
        """Calculate similarity between two sequences."""
        if not seq_a or not seq_b:
            return 0.0
        
        # Simple Jaccard similarity for now
        set_a = set(seq_a)
        set_b = set(seq_b)
        
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        
        return intersection / union if union > 0 else 0.0
    
    def _identify_correlation_clusters(self, correlations: List[CorrelationScore], sessions: List[Dict]) -> List[CorrelationCluster]:
        """Identify clusters of highly correlated sessions."""
        try:
            # Build adjacency graph
            graph = defaultdict(list)
            for correlation in correlations:
                if correlation.correlation_strength >= self.correlation_thresholds["moderate"]:
                    graph[correlation.source_id].append((correlation.target_id, correlation.correlation_strength))
                    graph[correlation.target_id].append((correlation.source_id, correlation.correlation_strength))
            
            # Find connected components (clusters)
            visited = set()
            clusters = []
            
            for session_id in graph:
                if session_id in visited:
                    continue
                
                # BFS to find connected component
                cluster_nodes = []
                queue = [session_id]
                visited.add(session_id)
                
                while queue:
                    current = queue.pop(0)
                    cluster_nodes.append(current)
                    
                    for neighbor, strength in graph[current]:
                        if neighbor not in visited and strength >= self.correlation_thresholds["moderate"]:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                if len(cluster_nodes) >= 2:  # Only clusters with 2+ nodes
                    cluster = self._build_cluster(cluster_nodes, correlations, sessions)
                    clusters.append(cluster)
            
            # Sort clusters by strength
            clusters.sort(key=lambda c: c.cluster_strength, reverse=True)
            return clusters
            
        except Exception as e:
            logging.error(f"Failed to identify correlation clusters: {e}")
            return []
    
    def _build_cluster(self, node_ids: List[str], correlations: List[CorrelationScore], sessions: List[Dict]) -> CorrelationCluster:
        """Build cluster object from node IDs."""
        # Calculate cluster strength (average correlation within cluster)
        cluster_correlations = []
        for correlation in correlations:
            if correlation.source_id in node_ids and correlation.target_id in node_ids:
                cluster_correlations.append(correlation.correlation_strength)
        
        cluster_strength = sum(cluster_correlations) / len(cluster_correlations) if cluster_correlations else 0.0
        
        # Find temporal span
        cluster_sessions = [s for s in sessions if s['id'] in node_ids]
        dates = []
        for session in cluster_sessions:
            try:
                dates.append(datetime.fromisoformat(session.get('created_at', '')))
            except:
                continue
        
        if dates:
            start_date = min(dates).isoformat()
            end_date = max(dates).isoformat()
        else:
            start_date = end_date = datetime.now().isoformat()
        
        # Identify primary theme (most common correlation type)
        theme_counts = Counter()
        for correlation in correlations:
            if correlation.source_id in node_ids and correlation.target_id in node_ids:
                theme_counts[correlation.correlation_type] += 1
        
        primary_theme = theme_counts.most_common(1)[0][0] if theme_counts else "mixed"
        
        # Find shared elements
        shared_elements = self._find_shared_cluster_elements(node_ids)
        
        return CorrelationCluster(
            cluster_id=f"cluster_{hash(''.join(sorted(node_ids)))}"[:16],
            entity_ids=node_ids,
            entity_types=["session"] * len(node_ids),
            cluster_strength=round(cluster_strength, 3),
            primary_theme=primary_theme,
            temporal_span={"start_date": start_date, "end_date": end_date},
            shared_elements=shared_elements
        )
    
    def _find_shared_cluster_elements(self, session_ids: List[str]) -> List[str]:
        """Find elements shared across cluster sessions."""
        shared_elements = set()
        
        # Find shared tools and files
        all_files = defaultdict(int)
        all_tools = defaultdict(int)
        
        for session_id in session_ids:
            tools = self.db.get_session_tool_usage(session_id)
            session_files = set()
            session_tools = set()
            
            for tool in tools:
                session_tools.add(tool.get('tool_name', ''))
                tool_input = tool.get('input_data', {})
                if isinstance(tool_input, dict) and 'file_path' in tool_input:
                    session_files.add(tool_input['file_path'])
            
            for file_path in session_files:
                all_files[file_path] += 1
            
            for tool_name in session_tools:
                all_tools[tool_name] += 1
        
        # Elements shared by most sessions in cluster
        threshold = max(len(session_ids) // 2, 1)
        
        for file_path, count in all_files.items():
            if count >= threshold:
                shared_elements.add(f"file:{Path(file_path).name}")
        
        for tool_name, count in all_tools.items():
            if count >= threshold:
                shared_elements.add(f"tool:{tool_name}")
        
        return list(shared_elements)[:10]  # Limit to top 10
    
    def _summarize_correlations(self, correlations: List[CorrelationScore]) -> Dict[str, Any]:
        """Summarize correlation statistics."""
        if not correlations:
            return {"total": 0}
        
        strengths = [c.correlation_strength for c in correlations]
        types = Counter(c.correlation_type for c in correlations)
        
        return {
            "total": len(correlations),
            "average_strength": round(sum(strengths) / len(strengths), 3),
            "strength_distribution": {
                "weak": sum(1 for s in strengths if s < self.correlation_thresholds["moderate"]),
                "moderate": sum(1 for s in strengths if self.correlation_thresholds["moderate"] <= s < self.correlation_thresholds["strong"]),
                "strong": sum(1 for s in strengths if s >= self.correlation_thresholds["strong"])
            },
            "correlation_types": dict(types),
            "strongest_correlation": max(strengths),
            "weakest_correlation": min(strengths)
        }
    
    def _get_strongest_correlations(self, correlations: List[CorrelationScore], limit: int = 10) -> List[Dict[str, Any]]:
        """Get strongest correlations for display."""
        sorted_correlations = sorted(correlations, key=lambda c: c.correlation_strength, reverse=True)
        
        strongest = []
        for correlation in sorted_correlations[:limit]:
            strongest.append({
                "source_id": correlation.source_id,
                "target_id": correlation.target_id,
                "strength": correlation.correlation_strength,
                "type": correlation.correlation_type,
                "primary_factors": {k: v for k, v in correlation.factors.items() if v > 0.3},
                "time_gap_hours": correlation.metadata.get('time_gap_hours', 0) if correlation.metadata else 0
            })
        
        return strongest
    
    def _generate_correlation_insights(self, correlations: List[CorrelationScore], clusters: List[CorrelationCluster]) -> List[str]:
        """Generate insights from correlation analysis."""
        insights = []
        
        if not correlations:
            insights.append("No significant correlations found in the analyzed timeframe")
            return insights
        
        # Overall patterns
        avg_strength = sum(c.correlation_strength for c in correlations) / len(correlations)
        if avg_strength > 0.6:
            insights.append("High correlation density detected - sessions show strong interconnectedness")
        elif avg_strength > 0.4:
            insights.append("Moderate correlation patterns - some related development themes")
        else:
            insights.append("Low correlation patterns - diverse development activities")
        
        # Correlation type insights
        type_counts = Counter(c.correlation_type for c in correlations)
        most_common_type = type_counts.most_common(1)[0][0] if type_counts else None
        
        if most_common_type == "semantic":
            insights.append("Sessions primarily related through semantic similarity - consistent problem domains")
        elif most_common_type == "contextual":
            insights.append("Sessions primarily related through shared files/tools - focused work areas")
        elif most_common_type == "behavioral":
            insights.append("Sessions primarily related through similar patterns - consistent development habits")
        elif most_common_type == "temporal":
            insights.append("Sessions primarily related through timing - burst-like development activity")
        elif most_common_type == "git":
            insights.append("Sessions primarily related through git activity - structured commit workflows")
        
        # Cluster insights
        if clusters:
            largest_cluster = max(clusters, key=lambda c: len(c.entity_ids))
            insights.append(f"Largest development theme involves {len(largest_cluster.entity_ids)} related sessions")
            
            if largest_cluster.shared_elements:
                common_elements = [elem.split(':', 1)[1] for elem in largest_cluster.shared_elements[:3]]
                insights.append(f"Most active shared elements: {', '.join(common_elements)}")
        
        # Temporal insights
        time_gaps = []
        for correlation in correlations:
            if correlation.metadata and 'time_gap_hours' in correlation.metadata:
                time_gaps.append(correlation.metadata['time_gap_hours'])
        
        if time_gaps:
            avg_gap = sum(time_gaps) / len(time_gaps)
            if avg_gap < 24:
                insights.append("Strong same-day correlation patterns - intensive development periods")
            elif avg_gap < 168:  # 1 week
                insights.append("Weekly correlation patterns - regular development cycles")
            else:
                insights.append("Long-term correlation patterns - extended project themes")
        
        return insights
    
    def find_development_patterns(self, lookback_days: int = 60) -> Dict[str, Any]:
        """Find recurring development patterns across sessions and commits."""
        try:
            # Build comprehensive correlations
            correlations_data = self.build_session_correlations(lookback_days=lookback_days)
            
            if "error" in correlations_data:
                return correlations_data
            
            # Analyze patterns
            patterns = {
                "analysis_date": datetime.now().isoformat(),
                "lookback_days": lookback_days,
                "workflow_patterns": self._analyze_workflow_patterns(correlations_data),
                "temporal_patterns": self._analyze_temporal_patterns(correlations_data),
                "collaboration_patterns": self._analyze_collaboration_patterns(correlations_data),
                "productivity_cycles": self._analyze_productivity_cycles(correlations_data),
                "development_themes": self._extract_development_themes(correlations_data),
                "pattern_recommendations": []
            }
            
            # Generate recommendations
            patterns["pattern_recommendations"] = self._generate_pattern_recommendations(patterns)
            
            return patterns
            
        except Exception as e:
            logging.error(f"Failed to find development patterns: {e}")
            return {"error": str(e)}
    
    def _analyze_workflow_patterns(self, correlations_data: Dict) -> Dict[str, Any]:
        """Analyze workflow patterns from correlations."""
        clusters = correlations_data.get("correlation_clusters", [])
        
        workflow_types = Counter()
        cluster_sizes = []
        shared_tools = Counter()
        
        for cluster in clusters:
            workflow_types[cluster.get("primary_theme", "unknown")] += 1
            cluster_sizes.append(len(cluster.get("entity_ids", [])))
            
            for element in cluster.get("shared_elements", []):
                if element.startswith("tool:"):
                    shared_tools[element[5:]] += 1  # Remove "tool:" prefix
        
        return {
            "cluster_count": len(clusters),
            "avg_cluster_size": round(sum(cluster_sizes) / len(cluster_sizes), 2) if cluster_sizes else 0,
            "workflow_types": dict(workflow_types),
            "most_common_tools": dict(shared_tools.most_common(10)),
            "clustering_tendency": "high" if len(clusters) > 3 else "moderate" if len(clusters) > 1 else "low"
        }
    
    def _analyze_temporal_patterns(self, correlations_data: Dict) -> Dict[str, Any]:
        """Analyze temporal patterns from correlations."""
        strongest_correlations = correlations_data.get("strongest_correlations", [])
        
        time_gaps = []
        for correlation in strongest_correlations:
            gap = correlation.get("time_gap_hours", 0)
            if gap > 0:
                time_gaps.append(gap)
        
        if not time_gaps:
            return {"pattern": "insufficient_data"}
        
        # Categorize time gaps
        same_day = sum(1 for gap in time_gaps if gap < 24)
        same_week = sum(1 for gap in time_gaps if 24 <= gap < 168)
        same_month = sum(1 for gap in time_gaps if 168 <= gap < 720)
        longer = len(time_gaps) - same_day - same_week - same_month
        
        return {
            "avg_correlation_gap_hours": round(sum(time_gaps) / len(time_gaps), 2),
            "gap_distribution": {
                "same_day": same_day,
                "same_week": same_week,
                "same_month": same_month,
                "longer": longer
            },
            "dominant_pattern": self._determine_dominant_temporal_pattern(same_day, same_week, same_month, longer)
        }
    
    def _determine_dominant_temporal_pattern(self, same_day: int, same_week: int, same_month: int, longer: int) -> str:
        """Determine dominant temporal pattern."""
        total = same_day + same_week + same_month + longer
        if total == 0:
            return "unknown"
        
        percentages = {
            "intensive": same_day / total,
            "weekly_cycles": same_week / total,
            "monthly_cycles": same_month / total,
            "long_term": longer / total
        }
        
        return max(percentages, key=percentages.get)
    
    def _analyze_collaboration_patterns(self, correlations_data: Dict) -> Dict[str, Any]:
        """Analyze collaboration patterns (placeholder - would need git author data)."""
        # This would be enhanced with actual git author analysis
        return {
            "collaboration_detected": False,
            "note": "Collaboration analysis requires git author data correlation"
        }
    
    def _analyze_productivity_cycles(self, correlations_data: Dict) -> Dict[str, Any]:
        """Analyze productivity cycles from correlation data."""
        correlation_summary = correlations_data.get("correlation_summary", {})
        
        total_correlations = correlation_summary.get("total", 0)
        avg_strength = correlation_summary.get("average_strength", 0)
        
        # Determine productivity cycle characteristics
        if total_correlations > 20 and avg_strength > 0.5:
            cycle_type = "high_intensity"
        elif total_correlations > 10 and avg_strength > 0.4:
            cycle_type = "sustained_productivity"
        elif total_correlations > 5:
            cycle_type = "moderate_activity"
        else:
            cycle_type = "low_correlation"
        
        return {
            "cycle_type": cycle_type,
            "correlation_density": total_correlations,
            "relationship_strength": avg_strength,
            "productivity_indicators": self._assess_productivity_indicators(correlations_data)
        }
    
    def _assess_productivity_indicators(self, correlations_data: Dict) -> List[str]:
        """Assess productivity indicators from correlations."""
        indicators = []
        
        summary = correlations_data.get("correlation_summary", {})
        clusters = correlations_data.get("correlation_clusters", [])
        
        # High correlation count indicates active development
        if summary.get("total", 0) > 15:
            indicators.append("High development activity detected")
        
        # Strong average correlations indicate focused work
        if summary.get("average_strength", 0) > 0.6:
            indicators.append("Highly focused development themes")
        
        # Large clusters indicate sustained work on related topics
        if clusters:
            largest_cluster_size = max(len(c.get("entity_ids", [])) for c in clusters)
            if largest_cluster_size > 5:
                indicators.append("Sustained focus on major development theme")
        
        # Diverse correlation types indicate well-rounded development
        types = summary.get("correlation_types", {})
        if len(types) >= 3:
            indicators.append("Diverse development approach across multiple dimensions")
        
        return indicators
    
    def _extract_development_themes(self, correlations_data: Dict) -> List[Dict[str, Any]]:
        """Extract major development themes from clusters."""
        clusters = correlations_data.get("correlation_clusters", [])
        
        themes = []
        for cluster in clusters:
            theme = {
                "theme_name": cluster.get("primary_theme", "unknown"),
                "session_count": len(cluster.get("entity_ids", [])),
                "strength": cluster.get("cluster_strength", 0),
                "duration_days": self._calculate_theme_duration(cluster),
                "key_elements": cluster.get("shared_elements", [])[:5]
            }
            themes.append(theme)
        
        # Sort by strength and session count
        themes.sort(key=lambda t: (t["strength"], t["session_count"]), reverse=True)
        return themes[:10]  # Top 10 themes
    
    def _calculate_theme_duration(self, cluster: Dict) -> int:
        """Calculate duration of a development theme."""
        try:
            temporal_span = cluster.get("temporal_span", {})
            start_date = datetime.fromisoformat(temporal_span.get("start_date", ""))
            end_date = datetime.fromisoformat(temporal_span.get("end_date", ""))
            return (end_date - start_date).days
        except:
            return 0
    
    def _generate_pattern_recommendations(self, patterns: Dict) -> List[str]:
        """Generate recommendations based on identified patterns."""
        recommendations = []
        
        workflow = patterns.get("workflow_patterns", {})
        temporal = patterns.get("temporal_patterns", {})
        productivity = patterns.get("productivity_cycles", {})
        themes = patterns.get("development_themes", [])
        
        # Workflow recommendations
        clustering_tendency = workflow.get("clustering_tendency", "low")
        if clustering_tendency == "low":
            recommendations.append("Consider working on related tasks in the same session to build stronger development themes")
        elif clustering_tendency == "high":
            recommendations.append("Excellent clustering patterns - continue focusing on related development tasks")
        
        # Temporal recommendations
        temporal_pattern = temporal.get("dominant_pattern", "unknown")
        if temporal_pattern == "intensive":
            recommendations.append("You work in intensive bursts - ensure you're taking adequate breaks")
        elif temporal_pattern == "weekly_cycles":
            recommendations.append("Strong weekly development cycles - consider scheduling consistent development blocks")
        elif temporal_pattern == "long_term":
            recommendations.append("Long-term project focus - consider more frequent intermediate checkpoints")
        
        # Productivity recommendations
        cycle_type = productivity.get("cycle_type", "unknown")
        if cycle_type == "low_correlation":
            recommendations.append("Low session correlation suggests diverse work - consider documenting connections between tasks")
        elif cycle_type == "high_intensity":
            recommendations.append("High-intensity development detected - excellent focus and productivity")
        
        # Theme recommendations
        if len(themes) > 5:
            recommendations.append("Many development themes detected - consider prioritizing to maintain focus")
        elif len(themes) < 2:
            recommendations.append("Limited theme diversity - consider exploring related development areas")
        
        return recommendations[:5]  # Limit to 5 recommendations


# CLI interface for cross-session correlation
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: cross_session_correlator.py <command> [args...]")
        sys.exit(1)
    
    command = sys.argv[1]
    memory_dir = os.environ.get("PRSIST_MEMORY_DIR", os.path.dirname(__file__))
    
    correlator = CrossSessionCorrelator(memory_dir)
    
    if command == "correlate":
        session_id = sys.argv[2] if len(sys.argv) > 2 else None
        days = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        result = correlator.build_session_correlations(session_id, days)
        print(json.dumps(result, indent=2))
    
    elif command == "patterns":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        result = correlator.find_development_patterns(days)
        print(json.dumps(result, indent=2))
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)