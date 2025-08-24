#!/usr/bin/env python3
"""
Change processor for Prsist Memory System Phase 3.
Processes file changes and updates memory system.
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add memory system to Python path
memory_dir = Path(__file__).parent.parent
sys.path.insert(0, str(memory_dir))

try:
    from memory_manager import MemoryManager
    from utils import setup_logging
except ImportError as e:
    print(f"Memory system not available: {e}")
    sys.exit(1)

class ChangeProcessor:
    """Processes file changes and updates memory system."""
    
    def __init__(self):
        """Initialize change processor."""
        setup_logging("INFO")
        self.memory_manager = MemoryManager()
        
    def process_changes(self, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch of file changes."""
        try:
            logging.info(f"Processing {len(changes)} file changes")
            
            results = []
            
            for change in changes:
                result = self.process_single_change(change)
                results.append(result)
            
            # Update memory relevance based on changes
            self.update_memory_relevance(changes)
            
            # Trigger context updates if needed
            self.trigger_context_updates(changes)
            
            summary = {
                "success": True,
                "changes_processed": len(changes),
                "successful_updates": sum(1 for r in results if r.get("success", False)),
                "failed_updates": sum(1 for r in results if not r.get("success", False)),
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
            
            logging.info(f"Change processing complete: {summary['successful_updates']}/{len(changes)} successful")
            return summary
            
        except Exception as e:
            logging.error(f"Failed to process changes: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def process_single_change(self, change: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single file change."""
        try:
            file_path = change["path"]
            change_type = change["type"]
            relevance = change.get("relevance", {})
            
            logging.debug(f"Processing {change_type} for {file_path}")
            
            # Analyze file change impact
            impact_analysis = self.analyze_file_change_impact(change)
            
            # Update memory database
            update_result = self.update_memory_database(change, impact_analysis)
            
            # Update file interaction tracking
            self.track_file_interaction(change)
            
            return {
                "success": True,
                "file_path": file_path,
                "change_type": change_type,
                "relevance_score": relevance.get("score", 0),
                "impact_analysis": impact_analysis,
                "update_result": update_result
            }
            
        except Exception as e:
            logging.error(f"Failed to process change for {change.get('path', 'unknown')}: {e}")
            return {
                "success": False,
                "file_path": change.get("path", "unknown"),
                "error": str(e)
            }
    
    def analyze_file_change_impact(self, change: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the impact of a file change on memory."""
        file_path = change["path"]
        change_type = change["type"]
        relevance = change.get("relevance", {})
        
        impact = {
            "file_type": self.get_file_type(file_path),
            "change_significance": self.calculate_change_significance(change),
            "memory_update_required": relevance.get("score", 0) > 0.5,
            "context_invalidation": self.should_invalidate_context(change),
            "session_impact": self.calculate_session_impact(change)
        }
        
        # Determine priority
        if impact["change_significance"] > 0.8:
            impact["priority"] = "high"
        elif impact["change_significance"] > 0.5:
            impact["priority"] = "medium"
        else:
            impact["priority"] = "low"
        
        return impact
    
    def calculate_change_significance(self, change: Dict[str, Any]) -> float:
        """Calculate the significance of a file change."""
        file_path = change["path"]
        change_type = change["type"]
        relevance = change.get("relevance", {})
        
        significance = relevance.get("score", 0.0)
        
        # Boost significance for certain file types
        file_ext = Path(file_path).suffix.lower()
        critical_extensions = {".py", ".js", ".ts", ".yaml", ".json"}
        if file_ext in critical_extensions:
            significance += 0.2
        
        # Boost significance for critical directories
        path_parts = Path(file_path).parts
        critical_dirs = {"bmad-core", "src", "lib", "core"}
        if any(part in critical_dirs for part in path_parts):
            significance += 0.1
        
        # Adjust based on change type
        change_type_multipliers = {
            "add": 1.0,
            "change": 0.8,
            "delete": 0.6,
            "add_dir": 0.3,
            "delete_dir": 0.2
        }
        
        significance *= change_type_multipliers.get(change_type, 0.5)
        
        return min(significance, 1.0)
    
    def should_invalidate_context(self, change: Dict[str, Any]) -> bool:
        """Determine if the change should invalidate existing context."""
        file_path = change["path"]
        change_type = change["type"]
        
        # Configuration files always invalidate context
        config_files = {".yaml", ".json", ".toml", ".ini", ".cfg"}
        if Path(file_path).suffix.lower() in config_files:
            return True
        
        # Core system files invalidate context
        if "bmad-core" in file_path or "config" in file_path:
            return True
        
        # File deletions might invalidate context
        if change_type == "delete":
            return True
        
        return False
    
    def calculate_session_impact(self, change: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate how the change impacts active sessions."""
        try:
            # Get current active sessions
            active_sessions = self.memory_manager.get_active_sessions()
            
            impact = {
                "affected_sessions": 0,
                "requires_context_refresh": False,
                "session_ids": []
            }
            
            for session in active_sessions:
                if self.change_affects_session(change, session):
                    impact["affected_sessions"] += 1
                    impact["session_ids"].append(session["id"])
            
            # Determine if context refresh is needed
            if impact["affected_sessions"] > 0:
                significance = self.calculate_change_significance(change)
                if significance > 0.6:
                    impact["requires_context_refresh"] = True
            
            return impact
            
        except Exception as e:
            logging.error(f"Failed to calculate session impact: {e}")
            return {"affected_sessions": 0, "requires_context_refresh": False}
    
    def change_affects_session(self, change: Dict[str, Any], session: Dict[str, Any]) -> bool:
        """Check if a file change affects a specific session."""
        file_path = change["path"]
        
        # Check if session has interacted with this file
        session_context = session.get("context_data", {})
        recent_files = session_context.get("recent_files", [])
        
        return file_path in recent_files
    
    def update_memory_database(self, change: Dict[str, Any], impact_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Update memory database with file change information."""
        try:
            # Record file interaction
            interaction_id = self.memory_manager.record_file_interaction(
                file_path=change["path"],
                interaction_type=change["type"],
                timestamp=datetime.now().isoformat(),
                context_data={
                    "relevance": change.get("relevance", {}),
                    "impact_analysis": impact_analysis,
                    "change_source": "file_watcher"
                }
            )
            
            # Update file relevance scores
            if impact_analysis.get("memory_update_required"):
                self.update_file_relevance(change, impact_analysis)
            
            return {
                "success": True,
                "interaction_id": interaction_id,
                "relevance_updated": impact_analysis.get("memory_update_required", False)
            }
            
        except Exception as e:
            logging.error(f"Failed to update memory database: {e}")
            return {"success": False, "error": str(e)}
    
    def track_file_interaction(self, change: Dict[str, Any]):
        """Track file interaction for memory system."""
        try:
            # This would typically call the memory manager to record
            # the file interaction in the database
            logging.debug(f"Tracking file interaction: {change['path']} ({change['type']})")
            
        except Exception as e:
            logging.error(f"Failed to track file interaction: {e}")
    
    def update_file_relevance(self, change: Dict[str, Any], impact_analysis: Dict[str, Any]):
        """Update file relevance scores in memory system."""
        try:
            file_path = change["path"]
            relevance_score = change.get("relevance", {}).get("score", 0)
            
            # Apply relevance boost for recent changes
            boost_duration_hours = 24
            if impact_analysis.get("priority") == "high":
                boost_duration_hours = 48
            elif impact_analysis.get("priority") == "medium":
                boost_duration_hours = 36
            
            # This would update relevance in the database
            logging.debug(f"Updating file relevance: {file_path} -> {relevance_score:.2f}")
            
        except Exception as e:
            logging.error(f"Failed to update file relevance: {e}")
    
    def update_memory_relevance(self, changes: List[Dict[str, Any]]):
        """Update memory relevance based on file changes."""
        try:
            high_impact_changes = [
                c for c in changes 
                if self.calculate_change_significance(c) > 0.7
            ]
            
            if high_impact_changes:
                logging.info(f"Found {len(high_impact_changes)} high-impact changes")
                # This would trigger memory relevance recalculation
                
        except Exception as e:
            logging.error(f"Failed to update memory relevance: {e}")
    
    def trigger_context_updates(self, changes: List[Dict[str, Any]]):
        """Trigger context updates for affected sessions."""
        try:
            context_invalidating_changes = [
                c for c in changes 
                if self.should_invalidate_context(c)
            ]
            
            if context_invalidating_changes:
                logging.info(f"Found {len(context_invalidating_changes)} context-invalidating changes")
                # This would trigger context refresh for affected sessions
                
        except Exception as e:
            logging.error(f"Failed to trigger context updates: {e}")
    
    def get_file_type(self, file_path: str) -> str:
        """Get file type from file path."""
        ext = Path(file_path).suffix.lower()
        
        type_mapping = {
            ".py": "python",
            ".js": "javascript", 
            ".ts": "typescript",
            ".go": "go",
            ".java": "java",
            ".cpp": "cpp", ".cxx": "cpp", ".cc": "cpp",
            ".cs": "csharp",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".md": "markdown",
            ".rst": "restructuredtext",
            ".txt": "text",
            ".yaml": "yaml", ".yml": "yaml",
            ".json": "json",
            ".toml": "toml",
            ".ini": "ini",
            ".cfg": "config",
            ".dockerfile": "dockerfile",
            ".makefile": "makefile"
        }
        
        return type_mapping.get(ext, "unknown")

def main():
    """Main execution for command line usage."""
    if len(sys.argv) != 2:
        print("Usage: python change-processor.py <changes_json>")
        sys.exit(1)
    
    try:
        changes_json = sys.argv[1]
        changes = json.loads(changes_json)
        
        processor = ChangeProcessor()
        result = processor.process_changes(changes)
        
        print(json.dumps(result, indent=2))
        
        sys.exit(0 if result["success"] else 1)
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()