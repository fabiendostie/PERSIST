#!/usr/bin/env python3
"""
Context building module for Prsist Memory System.
Handles context injection and relevance scoring.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

from database import MemoryDatabase
from session_tracker import SessionTracker
from utils import (
    load_yaml_config,
    load_json_file,
    get_git_info,
    get_project_root,
    truncate_content
)

class ContextBuilder:
    """Builds context for Claude Code sessions."""
    
    def __init__(self, memory_dir: str = None, config: Dict[str, Any] = None):
        """Initialize context builder."""
        if memory_dir is None:
            memory_dir = Path(__file__).parent
        
        self.memory_dir = Path(memory_dir)
        self.db = MemoryDatabase(self.memory_dir / "storage" / "sessions.db")
        self.session_tracker = SessionTracker(memory_dir)
        self.project_root = get_project_root()
        
        # Load configuration
        if config:
            self.config = config
        else:
            yaml_config_path = self.memory_dir / "config" / "memory-config.yaml"
            json_config_path = self.memory_dir / "config" / "memory-config.json"
            
            # Try YAML first, fallback to JSON
            if yaml_config_path.exists():
                self.config = load_yaml_config(str(yaml_config_path))
            elif json_config_path.exists():
                self.config = load_json_file(str(json_config_path))
            else:
                self.config = {}
        
        # Context configuration
        self.max_context_tokens = self.config.get("context", {}).get("max_size_tokens", 50000)
        self.relevance_threshold = self.config.get("context", {}).get("relevance_threshold", 0.3)
        self.auto_inject = self.config.get("context", {}).get("auto_inject", True)
    
    def build_session_context(self, include_history: bool = True) -> Dict[str, Any]:
        """Build comprehensive context for new session."""
        context = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "git_info": get_git_info(str(self.project_root)),
            "project_memory": self._load_project_memory(),
            "recent_decisions": self._load_recent_decisions(),
            "memory_system_info": {
                "active": True,
                "version": "1.0",
                "features": ["session_tracking", "context_injection", "tool_logging"]
            }
        }
        
        if include_history:
            context["recent_sessions"] = self._get_recent_session_summaries()
            context["relevant_context"] = self._get_relevant_context()
        
        # Truncate if needed
        context_text = self._format_context_for_claude(context)
        if len(context_text) > self.max_context_tokens * 4:  # Rough token estimation
            context = self._truncate_context(context)
        
        return context
    
    def _load_project_memory(self) -> str:
        """Load persistent project memory."""
        memory_file = self.memory_dir / "context" / "project-memory.md"
        
        try:
            if memory_file.exists():
                with open(memory_file, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                # Create default project memory
                default_memory = self._create_default_project_memory()
                memory_file.parent.mkdir(parents=True, exist_ok=True)
                with open(memory_file, 'w', encoding='utf-8') as f:
                    f.write(default_memory)
                return default_memory
        except Exception as e:
            logging.error(f"Failed to load project memory: {e}")
            return ""
    
    def _create_default_project_memory(self) -> str:
        """Create default project memory content."""
        return f"""# Project Memory

This file contains persistent project context and learned information that should be preserved across Claude Code sessions.

## Project Overview

Project Path: {self.project_root}
Memory System: Prsist Memory System v1.0
Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Key Decisions and Patterns

(This section will be populated as decisions are made and patterns are identified)

## Important Context

(This section contains important information that should be remembered across sessions)

## Development Notes

(Ongoing notes about the development process, gotchas, and lessons learned)

## Architecture Notes

(Key architectural decisions and design patterns used in this project)
"""
    
    def _load_recent_decisions(self) -> List[Dict[str, Any]]:
        """Load recent project decisions."""
        decisions_yaml_file = self.memory_dir / "context" / "decisions.yaml"
        decisions_json_file = self.memory_dir / "context" / "decisions.json"
        
        try:
            # Try YAML first, then JSON
            if decisions_yaml_file.exists():
                decisions_data = load_yaml_config(str(decisions_yaml_file))
                return decisions_data.get("decisions", [])
            elif decisions_json_file.exists():
                decisions_data = load_json_file(str(decisions_json_file))
                return decisions_data.get("decisions", [])
            else:
                # Create empty decisions file (prefer JSON if YAML not available)
                default_decisions = {
                    "decisions": [],
                    "last_updated": datetime.now().isoformat()
                }
                decisions_file = decisions_yaml_file
                save_func = save_yaml_config
                
                # Use JSON if YAML not available
                from utils import YAML_AVAILABLE
                if not YAML_AVAILABLE:
                    decisions_file = decisions_json_file
                    save_func = save_json_file
                
                decisions_file.parent.mkdir(parents=True, exist_ok=True)
                save_func(default_decisions, str(decisions_file))
                return []
        except Exception as e:
            logging.error(f"Failed to load decisions: {e}")
            return []
    
    def _get_recent_session_summaries(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get summaries of recent sessions."""
        try:
            recent_sessions = self.db.get_recent_sessions(limit)
            summaries = []
            
            for session in recent_sessions:
                summary = self.session_tracker.get_session_summary(session["id"])
                if summary:
                    summaries.append(summary)
            
            return summaries
        except Exception as e:
            logging.error(f"Failed to get recent session summaries: {e}")
            return []
    
    def _get_relevant_context(self) -> List[Dict[str, Any]]:
        """Get relevant context entries based on current project state."""
        try:
            # Get recent context entries from database
            # This is a simplified implementation - in the future, this could use
            # more sophisticated relevance scoring based on current task context
            
            current_session = self.session_tracker.get_current_session()
            if not current_session:
                return []
            
            # For now, just return recent context entries
            # In Phase 2, this would include semantic similarity scoring
            return []
            
        except Exception as e:
            logging.error(f"Failed to get relevant context: {e}")
            return []
    
    def _format_context_for_claude(self, context: Dict[str, Any]) -> str:
        """Format context data for Claude consumption."""
        formatted_sections = []
        
        # Project information
        formatted_sections.append(f"# Project Context\n")
        formatted_sections.append(f"**Project Root:** {context['project_root']}\n")
        formatted_sections.append(f"**Timestamp:** {context['timestamp']}\n")
        
        # Git information
        git_info = context.get("git_info", {})
        if git_info:
            formatted_sections.append(f"**Git Branch:** {git_info.get('branch', 'unknown')}\n")
            formatted_sections.append(f"**Git Hash:** {git_info.get('hash', 'unknown')}\n")
            if git_info.get("dirty"):
                formatted_sections.append("**Git Status:** Working directory has uncommitted changes\n")
        
        # Project memory
        project_memory = context.get("project_memory", "")
        if project_memory:
            formatted_sections.append(f"\n## Project Memory\n\n{project_memory}\n")
        
        # Recent decisions
        decisions = context.get("recent_decisions", [])
        if decisions:
            formatted_sections.append(f"\n## Recent Decisions\n\n")
            for decision in decisions[-5:]:  # Last 5 decisions
                formatted_sections.append(f"- **{decision.get('title', 'Untitled')}** ({decision.get('date', 'No date')}): {decision.get('description', 'No description')}\n")
        
        # Recent sessions
        recent_sessions = context.get("recent_sessions", [])
        if recent_sessions:
            formatted_sections.append(f"\n## Recent Sessions\n\n")
            for session in recent_sessions[:3]:  # Last 3 sessions
                formatted_sections.append(f"- **Session {session['session_id'][:8]}** ({session.get('created_at', 'Unknown time')}): ")
                formatted_sections.append(f"{session.get('tool_usage_count', 0)} tools used, ")
                formatted_sections.append(f"{session.get('files_interacted', 0)} files modified\n")
        
        # Memory system info
        memory_info = context.get("memory_system_info", {})
        if memory_info.get("active"):
            formatted_sections.append(f"\n## Memory System Status\n\n")
            formatted_sections.append(f"- **Status:** Active (Version {memory_info.get('version', 'Unknown')})\n")
            formatted_sections.append(f"- **Features:** {', '.join(memory_info.get('features', []))}\n")
        
        return "".join(formatted_sections)
    
    def _truncate_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Truncate context to fit within token limits."""
        # Priority order for truncation:
        # 1. Keep memory system info and git info (essential)
        # 2. Keep recent project memory (high priority)
        # 3. Truncate recent sessions (medium priority)
        # 4. Truncate relevant context (lower priority)
        
        # Start by truncating project memory if it's too long
        project_memory = context.get("project_memory", "")
        if len(project_memory) > self.max_context_tokens * 2:  # Half the tokens
            context["project_memory"] = truncate_content(project_memory, self.max_context_tokens // 2)
        
        # Limit recent sessions
        recent_sessions = context.get("recent_sessions", [])
        if len(recent_sessions) > 3:
            context["recent_sessions"] = recent_sessions[:3]
        
        # Limit recent decisions
        decisions = context.get("recent_decisions", [])
        if len(decisions) > 5:
            context["recent_decisions"] = decisions[-5:]
        
        return context
    
    def add_context_entry(self, session_id: str, context_type: str, 
                         content: str, relevance_score: float = 1.0) -> bool:
        """Add a context entry for future reference."""
        try:
            return self.db.add_context_entry(
                session_id=session_id,
                context_type=context_type,
                content=content,
                relevance_score=relevance_score
            )
        except Exception as e:
            logging.error(f"Failed to add context entry: {e}")
            return False
    
    def update_project_memory(self, new_content: str, append: bool = False) -> bool:
        """Update persistent project memory."""
        try:
            memory_file = self.memory_dir / "context" / "project-memory.md"
            memory_file.parent.mkdir(parents=True, exist_ok=True)
            
            if append and memory_file.exists():
                with open(memory_file, 'r', encoding='utf-8') as f:
                    existing_content = f.read()
                content = f"{existing_content}\n\n## Updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{new_content}"
            else:
                content = new_content
            
            with open(memory_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logging.info("Updated project memory")
            return True
            
        except Exception as e:
            logging.error(f"Failed to update project memory: {e}")
            return False
    
    def add_decision(self, title: str, description: str, 
                    category: str = "general", impact: str = "medium") -> bool:
        """Add a decision to the project decisions log."""
        try:
            decisions_yaml_file = self.memory_dir / "context" / "decisions.yaml"
            decisions_json_file = self.memory_dir / "context" / "decisions.json"
            
            # Determine which file to use
            decisions_file = decisions_yaml_file
            load_func = load_yaml_config
            save_func = save_yaml_config
            
            from utils import YAML_AVAILABLE
            if not YAML_AVAILABLE or decisions_json_file.exists():
                decisions_file = decisions_json_file
                load_func = load_json_file
                save_func = save_json_file
            
            # Load existing decisions
            decisions_data = {"decisions": []}
            if decisions_file.exists():
                decisions_data = load_func(str(decisions_file))
            
            # Add new decision
            new_decision = {
                "title": title,
                "description": description,
                "category": category,
                "impact": impact,
                "date": datetime.now().isoformat(),
                "timestamp": datetime.now().timestamp()
            }
            
            decisions_data["decisions"].append(new_decision)
            decisions_data["last_updated"] = datetime.now().isoformat()
            
            # Save decisions
            save_func(decisions_data, str(decisions_file))
            
            logging.info(f"Added decision: {title}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to add decision: {e}")
            return False
    
    def get_context_for_claude(self) -> str:
        """Get formatted context string for Claude Code."""
        if not self.auto_inject:
            return ""
        
        try:
            context = self.build_session_context()
            return self._format_context_for_claude(context)
        except Exception as e:
            logging.error(f"Failed to build context for Claude: {e}")
            return ""