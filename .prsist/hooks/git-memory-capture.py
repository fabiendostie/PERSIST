#!/usr/bin/env python3
"""
Git memory capture hook for Prsist Memory System.
Captures current development context before commit.
"""

import sys
import os
import json
import logging
from datetime import datetime
from pathlib import Path

# Add memory system to Python path
memory_dir = Path(__file__).parent.parent
sys.path.insert(0, str(memory_dir))

try:
    from git_integration import GitMetadataExtractor, ChangeImpactAnalyzer
    from session_tracker import SessionTracker
    from utils import setup_logging, get_project_root
except ImportError as e:
    print(f"Memory system not available: {e}", file=sys.stderr)
    sys.exit(0)

def capture_pre_commit_context(staged_files: list) -> dict:
    """Capture development context before commit."""
    try:
        setup_logging("WARNING")  # Quiet for hooks
        
        project_root = get_project_root()
        git_extractor = GitMetadataExtractor(str(project_root))
        session_tracker = SessionTracker(str(memory_dir))
        
        # Get current session
        current_session = session_tracker.get_current_session()
        
        # Analyze staged changes
        staged_analysis = analyze_staged_changes(git_extractor, staged_files)
        
        # Capture context
        context = {
            "timestamp": datetime.now().isoformat(),
            "event": "pre-commit",
            "project_path": str(project_root),
            "current_branch": git_extractor.get_current_branch(),
            "staged_files": staged_files,
            "staged_analysis": staged_analysis,
            "active_session": current_session["id"] if current_session else None,
            "session_summary": session_tracker.get_session_summary() if current_session else None
        }
        
        # Update session with pre-commit context
        if current_session:
            session_tracker.update_session(
                context_data=current_session.get("context_data", {}).update({
                    "pre_commit_context": context
                })
            )
        
        # Create checkpoint before commit
        if current_session:
            checkpoint_name = f"pre_commit_{datetime.now().strftime('%H%M%S')}"
            session_tracker.create_checkpoint(checkpoint_name)
        
        logging.info(f"Captured pre-commit context: {len(staged_files)} files staged")
        return {"success": True, "context": context}
        
    except Exception as e:
        logging.error(f"Failed to capture pre-commit context: {e}")
        return {"success": False, "error": str(e)}

def analyze_staged_changes(git_extractor: GitMetadataExtractor, staged_files: list) -> dict:
    """Analyze the staged changes."""
    try:
        analysis = {
            "total_files": len(staged_files),
            "file_types": {},
            "significance_scores": [],
            "risk_indicators": [],
            "change_categories": {
                "source": 0,
                "tests": 0,
                "docs": 0,
                "config": 0,
                "other": 0
            }
        }
        
        for file_path in staged_files:
            # File type analysis
            file_type = git_extractor.get_file_type(file_path)
            analysis["file_types"][file_type] = analysis["file_types"].get(file_type, 0) + 1
            
            # Categorize changes
            if git_extractor.is_test_file(file_path):
                analysis["change_categories"]["tests"] += 1
            elif git_extractor.is_documentation_file(file_path):
                analysis["change_categories"]["docs"] += 1
            elif git_extractor.is_config_file(file_path):
                analysis["change_categories"]["config"] += 1
            elif file_type in ["python", "javascript", "typescript", "go", "java", "cpp", "rust"]:
                analysis["change_categories"]["source"] += 1
            else:
                analysis["change_categories"]["other"] += 1
            
            # Calculate significance
            significance = git_extractor.calculate_file_significance(file_path, "modified")
            analysis["significance_scores"].append(significance)
            
            # Risk indicators
            if git_extractor.is_config_file(file_path):
                analysis["risk_indicators"].append(f"Configuration file: {file_path}")
            elif "core" in file_path.lower():
                analysis["risk_indicators"].append(f"Core file: {file_path}")
        
        # Calculate overall metrics
        if analysis["significance_scores"]:
            analysis["avg_significance"] = sum(analysis["significance_scores"]) / len(analysis["significance_scores"])
            analysis["max_significance"] = max(analysis["significance_scores"])
        else:
            analysis["avg_significance"] = 0
            analysis["max_significance"] = 0
        
        # Risk assessment
        risk_score = 0
        if analysis["change_categories"]["config"] > 0:
            risk_score += 0.3
        if analysis["change_categories"]["source"] > 5:
            risk_score += 0.2
        if len(analysis["risk_indicators"]) > 0:
            risk_score += 0.4
        
        analysis["risk_score"] = min(risk_score, 1.0)
        analysis["risk_level"] = "high" if risk_score > 0.6 else "medium" if risk_score > 0.3 else "low"
        
        return analysis
        
    except Exception as e:
        logging.error(f"Failed to analyze staged changes: {e}")
        return {"error": str(e)}

def main():
    """Main hook execution."""
    try:
        # Parse arguments
        if len(sys.argv) < 2:
            print("Usage: git-memory-capture.py <staged_files...>", file=sys.stderr)
            sys.exit(0)
        
        staged_files = sys.argv[1:] if len(sys.argv) > 1 else []
        
        # Skip if no relevant files
        if not staged_files:
            sys.exit(0)
        
        # Filter for relevant file types only
        relevant_files = []
        relevant_extensions = {
            '.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.java', '.cpp', '.c', 
            '.h', '.hpp', '.cs', '.rs', '.php', '.rb', '.swift', '.kt', '.scala',
            '.md', '.json', '.yaml', '.yml', '.xml', '.html', '.css', '.scss'
        }
        
        for file_path in staged_files:
            if any(file_path.endswith(ext) for ext in relevant_extensions):
                relevant_files.append(file_path)
        
        if not relevant_files:
            sys.exit(0)
        
        # Capture context
        result = capture_pre_commit_context(relevant_files)
        
        if not result["success"]:
            print(f"Warning: Failed to capture pre-commit context: {result.get('error', 'Unknown error')}", 
                  file=sys.stderr)
        
        # Exit successfully to allow commit to proceed
        sys.exit(0)
        
    except Exception as e:
        print(f"Pre-commit hook error: {e}", file=sys.stderr)
        # Don't block commit on hook failure
        sys.exit(0)

if __name__ == "__main__":
    main()