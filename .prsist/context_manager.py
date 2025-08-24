#!/usr/bin/env python3
"""
Dynamic context manager for Prsist Memory System Phase 3.
Handles intelligent context injection and compression.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import re

from database import MemoryDatabase
from utils import setup_logging

class ContextManager:
    """Manages dynamic context injection and compression."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize context manager."""
        self.config = config
        self.context_cache = {}
        self.compression_engine = ContextCompressor()
        self.relevance_scorer = None  # Will be set when relevance scorer is available
        
    def get_dynamic_context(self, session_id: str, current_task: str) -> Dict[str, Any]:
        """Get just-in-time context injection."""
        try:
            logging.debug(f"Getting dynamic context for session {session_id}")
            
            # Get base context
            base_context = self.get_base_context(session_id)
            
            # Check if context is getting too large
            if self.is_context_full(base_context):
                logging.info("Context approaching capacity, applying compression")
                compressed_context = self.compression_engine.compress(
                    base_context,
                    preserve_critical=True
                )
                return compressed_context
            
            # Add relevant context incrementally
            enhanced_context = self.expand_context(
                base_context,
                current_task,
                relevance_threshold=0.7
            )
            
            return enhanced_context
            
        except Exception as e:
            logging.error(f"Failed to get dynamic context: {e}")
            return self.get_base_context(session_id)
    
    def get_base_context(self, session_id: str) -> Dict[str, Any]:
        """Get base context for a session."""
        try:
            # Check cache first
            cache_key = f"base_context_{session_id}"
            if cache_key in self.context_cache:
                cached = self.context_cache[cache_key]
                # Use cached context if it's less than 5 minutes old
                if (datetime.now() - cached["timestamp"]).total_seconds() < 300:
                    return cached["context"]
            
            # Build base context
            base_context = {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "current_session": self.get_current_session_context(session_id),
                "recent_files": self.get_recent_files(session_id),
                "recent_tools": self.get_recent_tool_usage(session_id),
                "project_context": self.get_project_context(),
                "critical_info": self.get_critical_information(session_id)
            }
            
            # Cache the context
            self.context_cache[cache_key] = {
                "context": base_context,
                "timestamp": datetime.now()
            }
            
            return base_context
            
        except Exception as e:
            logging.error(f"Failed to get base context: {e}")
            return {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "error": "Failed to load context"
            }
    
    def expand_context(self, base_context: Dict[str, Any], current_task: str, 
                      relevance_threshold: float = 0.7) -> Dict[str, Any]:
        """Expand context with task-relevant information."""
        try:
            expanded_context = base_context.copy()
            
            # Add task-specific context
            task_context = self.get_task_relevant_context(current_task)
            if task_context:
                expanded_context["task_context"] = task_context
            
            # Add related sessions if relevance scorer is available
            if self.relevance_scorer:
                related_sessions = self.get_related_sessions(
                    base_context["session_id"], 
                    current_task, 
                    relevance_threshold
                )
                if related_sessions:
                    expanded_context["related_sessions"] = related_sessions
            
            # Add pattern-based context
            pattern_context = self.get_pattern_context(current_task)
            if pattern_context:
                expanded_context["patterns"] = pattern_context
            
            return expanded_context
            
        except Exception as e:
            logging.error(f"Failed to expand context: {e}")
            return base_context
    
    def compress_at_95_percent(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Compress context when approaching 95% capacity."""
        if self.calculate_token_usage(context) >= 0.95:
            return self.compression_engine.hierarchical_compress(
                context,
                phases=['completed', 'background', 'reference']
            )
        return context
    
    def is_context_full(self, context: Dict[str, Any]) -> bool:
        """Check if context is approaching capacity."""
        token_usage = self.calculate_token_usage(context)
        return token_usage >= 0.90  # 90% threshold
    
    def calculate_token_usage(self, context: Dict[str, Any]) -> float:
        """Estimate token usage of context (simplified)."""
        try:
            # Rough estimation: 4 characters per token on average
            context_str = json.dumps(context, ensure_ascii=False)
            estimated_tokens = len(context_str) / 4
            
            # Assume max context window of 75,000 tokens
            max_tokens = self.config.get("context_management", {}).get("max_context_tokens", 75000)
            
            return min(estimated_tokens / max_tokens, 1.0)
            
        except Exception as e:
            logging.error(f"Failed to calculate token usage: {e}")
            return 0.5  # Conservative estimate
    
    def get_current_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get context for the current session."""
        try:
            # This would typically query the database for session info
            return {
                "session_id": session_id,
                "started_at": datetime.now().isoformat(),
                "status": "active",
                "goals": [],
                "progress": {}
            }
        except Exception as e:
            logging.error(f"Failed to get current session context: {e}")
            return {}
    
    def get_recent_files(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recently accessed files for session."""
        try:
            # This would query file interactions from database
            return []
        except Exception as e:
            logging.error(f"Failed to get recent files: {e}")
            return []
    
    def get_recent_tool_usage(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent tool usage for session."""
        try:
            # This would query tool usage from database
            return []
        except Exception as e:
            logging.error(f"Failed to get recent tool usage: {e}")
            return []
    
    def get_project_context(self) -> Dict[str, Any]:
        """Get overall project context."""
        try:
            return {
                "project_type": "bmad-method",
                "framework_version": "3.0",
                "key_components": [
                    "memory_system",
                    "git_integration", 
                    "file_watching",
                    "context_management"
                ]
            }
        except Exception as e:
            logging.error(f"Failed to get project context: {e}")
            return {}
    
    def get_critical_information(self, session_id: str) -> Dict[str, Any]:
        """Get critical information that should always be preserved."""
        try:
            return {
                "system_status": "operational",
                "active_features": ["file_watching", "git_integration", "memory_persistence"],
                "important_notes": []
            }
        except Exception as e:
            logging.error(f"Failed to get critical information: {e}")
            return {}
    
    def get_task_relevant_context(self, current_task: str) -> Optional[Dict[str, Any]]:
        """Get context relevant to the current task."""
        try:
            if not current_task:
                return None
            
            # Analyze task for context needs
            task_analysis = self.analyze_task(current_task)
            
            context = {
                "task_type": task_analysis.get("type", "unknown"),
                "complexity": task_analysis.get("complexity", "medium"),
                "required_knowledge": task_analysis.get("knowledge_areas", []),
                "suggested_files": task_analysis.get("relevant_files", [])
            }
            
            return context
            
        except Exception as e:
            logging.error(f"Failed to get task relevant context: {e}")
            return None
    
    def analyze_task(self, task: str) -> Dict[str, Any]:
        """Analyze a task to determine context requirements."""
        try:
            task_lower = task.lower()
            
            # Determine task type
            task_type = "general"
            if any(word in task_lower for word in ["implement", "create", "build", "develop"]):
                task_type = "implementation"
            elif any(word in task_lower for word in ["fix", "debug", "resolve", "error"]):
                task_type = "debugging"
            elif any(word in task_lower for word in ["refactor", "optimize", "improve"]):
                task_type = "optimization"
            elif any(word in task_lower for word in ["test", "validate", "verify"]):
                task_type = "testing"
            elif any(word in task_lower for word in ["document", "explain", "describe"]):
                task_type = "documentation"
            
            # Determine complexity
            complexity = "medium"
            complexity_indicators = {
                "simple": ["simple", "basic", "quick", "small"],
                "medium": ["medium", "moderate", "standard"],
                "complex": ["complex", "advanced", "comprehensive", "full", "complete"]
            }
            
            for level, indicators in complexity_indicators.items():
                if any(indicator in task_lower for indicator in indicators):
                    complexity = level
                    break
            
            # Identify knowledge areas
            knowledge_areas = []
            knowledge_patterns = {
                "git": ["git", "commit", "branch", "merge", "repository"],
                "database": ["database", "sql", "table", "query", "db"],
                "file_system": ["file", "directory", "path", "folder"],
                "memory_system": ["memory", "context", "session", "tracking"],
                "testing": ["test", "spec", "verify", "validate"],
                "configuration": ["config", "settings", "yaml", "json"]
            }
            
            for area, patterns in knowledge_patterns.items():
                if any(pattern in task_lower for pattern in patterns):
                    knowledge_areas.append(area)
            
            return {
                "type": task_type,
                "complexity": complexity,
                "knowledge_areas": knowledge_areas,
                "relevant_files": self.suggest_relevant_files(task_type, knowledge_areas)
            }
            
        except Exception as e:
            logging.error(f"Failed to analyze task: {e}")
            return {"type": "unknown", "complexity": "medium"}
    
    def suggest_relevant_files(self, task_type: str, knowledge_areas: List[str]) -> List[str]:
        """Suggest files that might be relevant to the task."""
        suggestions = []
        
        # File suggestions based on knowledge areas
        file_mappings = {
            "git": ["git_integration.py", "correlation_engine.py", ".lefthook.yml"],
            "database": ["database.py", "session_tracker.py"],
            "memory_system": ["memory_manager.py", "context_manager.py"],
            "testing": ["test_system.py", "test_git_integration.py"],
            "configuration": ["memory-config.yaml", "watch-config.yaml"]
        }
        
        for area in knowledge_areas:
            if area in file_mappings:
                suggestions.extend(file_mappings[area])
        
        # Remove duplicates and limit results
        return list(set(suggestions))[:10]
    
    def get_related_sessions(self, session_id: str, current_task: str, 
                           threshold: float) -> List[Dict[str, Any]]:
        """Get sessions related to current task (requires relevance scorer)."""
        try:
            if not self.relevance_scorer:
                return []
            
            # This would use the relevance scorer to find related sessions
            return []
            
        except Exception as e:
            logging.error(f"Failed to get related sessions: {e}")
            return []
    
    def get_pattern_context(self, current_task: str) -> Optional[Dict[str, Any]]:
        """Get context based on recognized patterns."""
        try:
            patterns = {
                "implementation_pattern": self.detect_implementation_pattern(current_task),
                "workflow_pattern": self.detect_workflow_pattern(current_task),
                "file_pattern": self.detect_file_pattern(current_task)
            }
            
            # Filter out None patterns
            active_patterns = {k: v for k, v in patterns.items() if v is not None}
            
            return active_patterns if active_patterns else None
            
        except Exception as e:
            logging.error(f"Failed to get pattern context: {e}")
            return None
    
    def detect_implementation_pattern(self, task: str) -> Optional[str]:
        """Detect implementation patterns in the task."""
        patterns = {
            "phase_implementation": r"phase\s+\d+|implement.*phase",
            "feature_addition": r"add.*feature|implement.*feature|new.*feature",
            "bug_fix": r"fix.*bug|resolve.*issue|debug",
            "refactoring": r"refactor|restructure|reorganize",
            "integration": r"integrate|connect|link"
        }
        
        for pattern_name, pattern_regex in patterns.items():
            if re.search(pattern_regex, task.lower()):
                return pattern_name
        
        return None
    
    def detect_workflow_pattern(self, task: str) -> Optional[str]:
        """Detect workflow patterns in the task."""
        patterns = {
            "agile_development": r"sprint|scrum|agile|story",
            "git_workflow": r"commit|merge|branch|pull request",
            "testing_workflow": r"test.*develop|tdd|bdd",
            "documentation_workflow": r"document.*implement|doc.*first"
        }
        
        for pattern_name, pattern_regex in patterns.items():
            if re.search(pattern_regex, task.lower()):
                return pattern_name
        
        return None
    
    def detect_file_pattern(self, task: str) -> Optional[str]:
        """Detect file-related patterns in the task."""
        patterns = {
            "multi_file": r"files|multiple.*file|across.*file",
            "config_change": r"config|configuration|settings",
            "code_generation": r"generate.*code|create.*file",
            "file_modification": r"modify.*file|update.*file|edit.*file"
        }
        
        for pattern_name, pattern_regex in patterns.items():
            if re.search(pattern_regex, task.lower()):
                return pattern_name
        
        return None
    
    def invalidate_cache(self, session_id: Optional[str] = None):
        """Invalidate context cache."""
        if session_id:
            cache_key = f"base_context_{session_id}"
            if cache_key in self.context_cache:
                del self.context_cache[cache_key]
        else:
            self.context_cache.clear()
        
        logging.debug(f"Context cache invalidated for session: {session_id or 'all'}")


class ContextCompressor:
    """Handles context compression to manage token limits."""
    
    def __init__(self):
        """Initialize context compressor."""
        self.compression_strategies = {
            'hierarchical': self.hierarchical_compress,
            'semantic': self.semantic_compress,
            'temporal': self.temporal_compress,
            'importance': self.importance_compress
        }
    
    def compress(self, context: Dict[str, Any], target_reduction: float = 0.3,
                preserve_critical: bool = True) -> Dict[str, Any]:
        """Compress context using multiple strategies."""
        try:
            compressed = context.copy()
            
            # Always preserve critical information
            if preserve_critical:
                critical_info = context.get('critical_info', {})
                current_session = context.get('current_session', {})
            
            # Apply hierarchical compression first
            compressed = self.hierarchical_compress(compressed, preserve_critical)
            
            # Apply temporal compression for older data
            compressed = self.temporal_compress(compressed)
            
            # Apply importance-based compression
            compressed = self.importance_compress(compressed)
            
            # Restore critical information
            if preserve_critical:
                compressed['critical_info'] = critical_info
                compressed['current_session'] = current_session
            
            return compressed
            
        except Exception as e:
            logging.error(f"Failed to compress context: {e}")
            return context
    
    def hierarchical_compress(self, context: Dict[str, Any], 
                            preserve_critical: bool = True,
                            phases: List[str] = None) -> Dict[str, Any]:
        """Phase-by-phase summarization."""
        try:
            compressed = {}
            
            # Always preserve critical information
            if preserve_critical:
                compressed['critical'] = context.get('critical_info', {})
                compressed['current_session'] = context.get('current_session', {})
            
            # Compress completed phases
            completed_phases = context.get('completed_phases', [])
            if completed_phases:
                compressed['completed_summary'] = self.summarize_phases(completed_phases)
            
            # Compress background context
            background = context.get('background', {})
            if background:
                compressed['background_summary'] = self.compress_background(background)
            
            # Compress recent files (keep only most important)
            recent_files = context.get('recent_files', [])
            if recent_files:
                compressed['recent_files'] = recent_files[:5]  # Keep top 5
            
            # Compress tool usage (summarize)
            recent_tools = context.get('recent_tools', [])
            if recent_tools:
                compressed['tool_usage_summary'] = self.summarize_tool_usage(recent_tools)
            
            # Preserve other important fields
            for key in ['session_id', 'timestamp', 'project_context']:
                if key in context:
                    compressed[key] = context[key]
            
            return compressed
            
        except Exception as e:
            logging.error(f"Failed to apply hierarchical compression: {e}")
            return context
    
    def semantic_compress(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Compress based on semantic similarity."""
        # This would require NLP models to identify similar content
        # For now, return unchanged
        return context
    
    def temporal_compress(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Compress older temporal data."""
        try:
            compressed = context.copy()
            
            # Compress older file interactions
            recent_files = context.get('recent_files', [])
            if recent_files:
                # Keep only files from last 24 hours
                cutoff = datetime.now() - timedelta(hours=24)
                filtered_files = []
                
                for file_info in recent_files:
                    try:
                        file_time = datetime.fromisoformat(file_info.get('timestamp', ''))
                        if file_time > cutoff:
                            filtered_files.append(file_info)
                    except:
                        # Keep file if timestamp is invalid
                        filtered_files.append(file_info)
                
                compressed['recent_files'] = filtered_files
            
            return compressed
            
        except Exception as e:
            logging.error(f"Failed to apply temporal compression: {e}")
            return context
    
    def importance_compress(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Compress based on importance scores."""
        try:
            compressed = context.copy()
            
            # This would use importance scores to filter content
            # For now, apply simple heuristics
            
            # Limit related sessions to most relevant
            related_sessions = context.get('related_sessions', [])
            if len(related_sessions) > 3:
                # Sort by relevance score if available
                sorted_sessions = sorted(
                    related_sessions,
                    key=lambda x: x.get('relevance_score', 0),
                    reverse=True
                )
                compressed['related_sessions'] = sorted_sessions[:3]
            
            return compressed
            
        except Exception as e:
            logging.error(f"Failed to apply importance compression: {e}")
            return context
    
    def summarize_phases(self, phases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize completed phases."""
        try:
            summary = {
                "total_phases": len(phases),
                "completed_at": datetime.now().isoformat(),
                "key_outcomes": [],
                "major_changes": []
            }
            
            for phase in phases:
                if phase.get('key_outcomes'):
                    summary['key_outcomes'].extend(phase['key_outcomes'])
                if phase.get('major_changes'):
                    summary['major_changes'].extend(phase['major_changes'])
            
            # Limit to prevent explosion
            summary['key_outcomes'] = summary['key_outcomes'][:10]
            summary['major_changes'] = summary['major_changes'][:10]
            
            return summary
            
        except Exception as e:
            logging.error(f"Failed to summarize phases: {e}")
            return {"total_phases": len(phases)}
    
    def compress_background(self, background: Dict[str, Any]) -> Dict[str, Any]:
        """Compress background context."""
        try:
            # Keep only essential background information
            essential_keys = ['project_type', 'framework_version', 'key_components']
            
            compressed = {}
            for key in essential_keys:
                if key in background:
                    compressed[key] = background[key]
            
            return compressed
            
        except Exception as e:
            logging.error(f"Failed to compress background: {e}")
            return background
    
    def summarize_tool_usage(self, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize tool usage patterns."""
        try:
            summary = {
                "total_operations": len(tools),
                "most_used_tools": {},
                "recent_activity": []
            }
            
            # Count tool usage
            tool_counts = {}
            for tool in tools:
                tool_name = tool.get('tool_name', 'unknown')
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
            
            # Get top 5 most used tools
            sorted_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)
            summary['most_used_tools'] = dict(sorted_tools[:5])
            
            # Get recent activity (last 5 operations)
            summary['recent_activity'] = tools[-5:] if len(tools) > 5 else tools
            
            return summary
            
        except Exception as e:
            logging.error(f"Failed to summarize tool usage: {e}")
            return {"total_operations": len(tools)}
    
    def auto_compact(self, context: Dict[str, Any], 
                    capacity_threshold: float = 0.95) -> Dict[str, Any]:
        """Auto-compact when approaching capacity."""
        try:
            # Rough token calculation (simplified)
            context_str = json.dumps(context, ensure_ascii=False)
            estimated_tokens = len(context_str) / 4
            
            # Assuming 75k token limit
            current_usage = estimated_tokens / 75000
            
            if current_usage >= capacity_threshold:
                logging.info(f"Auto-compacting context (usage: {current_usage:.1%})")
                return self.hierarchical_compress(context)
            
            return context
            
        except Exception as e:
            logging.error(f"Failed to auto-compact context: {e}")
            return context