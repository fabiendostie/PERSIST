#!/usr/bin/env python3
"""
Git memory correlation hook for Prsist Memory System.
Correlates commits with memory sessions and generates documentation.
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
    from correlation_engine import CorrelationEngine
    from git_integration import GitMetadataExtractor
    from database import MemoryDatabase
    from session_tracker import SessionTracker
    from utils import setup_logging, get_project_root
except ImportError as e:
    print(f"Memory system not available: {e}", file=sys.stderr)
    sys.exit(0)

def correlate_commit(commit_sha: str) -> dict:
    """Correlate commit with memory system."""
    try:
        setup_logging("WARNING")  # Quiet for hooks
        
        project_root = get_project_root()
        
        # Initialize components
        git_extractor = GitMetadataExtractor(str(project_root))
        memory_db = MemoryDatabase(str(memory_dir / "storage" / "sessions.db"))
        correlation_engine = CorrelationEngine(memory_db, git_extractor)
        session_tracker = SessionTracker(str(memory_dir))
        
        # Perform correlation
        correlation_result = correlation_engine.correlate_commit_with_sessions(commit_sha)
        
        if not correlation_result["success"]:
            logging.error(f"Correlation failed: {correlation_result.get('error')}")
            return correlation_result
        
        # Update current session with commit information
        current_session = session_tracker.get_current_session()
        if current_session:
            update_session_with_commit(session_tracker, current_session, correlation_result)
        
        # Generate documentation if enabled
        if should_generate_documentation(correlation_result):
            doc_result = generate_commit_documentation(correlation_result)
            correlation_result["documentation"] = doc_result
        
        logging.info(f"Successfully correlated commit {commit_sha[:8]}")
        return correlation_result
        
    except Exception as e:
        logging.error(f"Failed to correlate commit {commit_sha}: {e}")
        return {"success": False, "error": str(e)}

def update_session_with_commit(session_tracker: SessionTracker, 
                              current_session: dict, 
                              correlation_result: dict) -> bool:
    """Update current session with commit information."""
    try:
        commit_metadata = correlation_result["commit_metadata"]
        
        # Add commit to session context
        session_context = current_session.get("context_data", {})
        session_commits = session_context.get("session_commits", [])
        
        commit_summary = {
            "commit_sha": commit_metadata["commit_sha"],
            "short_sha": commit_metadata["short_sha"],
            "message": commit_metadata["subject"],
            "timestamp": commit_metadata["timestamp"],
            "impact_score": commit_metadata["impact_score"],
            "files_changed": len(commit_metadata.get("file_changes", [])),
            "commit_type": commit_metadata["commit_type"]
        }
        
        session_commits.append(commit_summary)
        session_context["session_commits"] = session_commits
        session_context["last_commit"] = commit_summary
        
        # Update session
        return session_tracker.update_session(context_data=session_context)
        
    except Exception as e:
        logging.error(f"Failed to update session with commit: {e}")
        return False

def should_generate_documentation(correlation_result: dict) -> bool:
    """Determine if documentation should be generated."""
    try:
        impact_analysis = correlation_result.get("impact_analysis", {})
        commit_metadata = correlation_result.get("commit_metadata", {})
        
        # Generate docs for significant commits
        impact_score = commit_metadata.get("impact_score", 0)
        if impact_score > 0.6:
            return True
        
        # Generate docs for breaking changes
        breaking_potential = impact_analysis.get("breaking_change_potential", {})
        if breaking_potential.get("potential") in ["medium", "high"]:
            return True
        
        # Generate docs for feature commits
        commit_type = commit_metadata.get("commit_type", "")
        if commit_type in ["feature", "bugfix"]:
            return True
        
        # Generate docs for configuration changes
        file_changes = commit_metadata.get("file_changes", [])
        config_changes = sum(1 for fc in file_changes if fc.get("is_config", False))
        if config_changes > 0:
            return True
        
        return False
        
    except Exception as e:
        logging.error(f"Failed to determine documentation generation: {e}")
        return False

def generate_commit_documentation(correlation_result: dict) -> dict:
    """Generate automated documentation for the commit."""
    try:
        commit_metadata = correlation_result["commit_metadata"]
        impact_analysis = correlation_result.get("impact_analysis", {})
        
        doc_entries = []
        
        # Generate commit summary
        commit_summary = generate_commit_summary(commit_metadata, impact_analysis)
        if commit_summary:
            doc_entries.append({
                "type": "commit_summary",
                "content": commit_summary,
                "metadata": {
                    "commit_sha": commit_metadata["commit_sha"],
                    "generated_at": datetime.now().isoformat()
                }
            })
        
        # Generate changelog entry
        changelog_entry = generate_changelog_entry(commit_metadata, impact_analysis)
        if changelog_entry:
            doc_entries.append({
                "type": "changelog",
                "content": changelog_entry,
                "metadata": {
                    "commit_sha": commit_metadata["commit_sha"],
                    "commit_type": commit_metadata["commit_type"]
                }
            })
        
        # Generate breaking change notice if needed
        breaking_potential = impact_analysis.get("breaking_change_potential", {})
        if breaking_potential.get("potential") in ["medium", "high"]:
            breaking_notice = generate_breaking_change_notice(commit_metadata, breaking_potential)
            if breaking_notice:
                doc_entries.append({
                    "type": "breaking_change",
                    "content": breaking_notice,
                    "metadata": {
                        "commit_sha": commit_metadata["commit_sha"],
                        "risk_level": breaking_potential.get("potential")
                    }
                })
        
        # Store documentation entries
        memory_db = MemoryDatabase(str(memory_dir / "storage" / "sessions.db"))
        stored_count = 0
        
        for entry in doc_entries:
            success = memory_db.record_documentation_entry(
                commit_sha=commit_metadata["commit_sha"],
                doc_type=entry["type"],
                content=entry["content"],
                metadata=entry["metadata"]
            )
            if success:
                stored_count += 1
        
        return {
            "generated": len(doc_entries),
            "stored": stored_count,
            "entries": doc_entries
        }
        
    except Exception as e:
        logging.error(f"Failed to generate documentation: {e}")
        return {"error": str(e)}

def generate_commit_summary(commit_metadata: dict, impact_analysis: dict) -> str:
    """Generate a comprehensive commit summary."""
    try:
        lines = [
            f"## Commit Summary: {commit_metadata['short_sha']}",
            "",
            f"**Message:** {commit_metadata['subject']}",
            f"**Author:** {commit_metadata['author_name']} <{commit_metadata['author_email']}>",
            f"**Date:** {commit_metadata['timestamp']}",
            f"**Type:** {commit_metadata['commit_type'].title()}",
            f"**Impact Score:** {commit_metadata['impact_score']:.2f}",
            ""
        ]
        
        # File changes summary
        file_changes = commit_metadata.get("file_changes", [])
        if file_changes:
            lines.extend([
                "### Changes:",
                f"- **Files modified:** {len(file_changes)}",
                f"- **Lines added:** {commit_metadata.get('stats', {}).get('insertions', 0)}",
                f"- **Lines deleted:** {commit_metadata.get('stats', {}).get('deletions', 0)}"
            ])
            
            # Categorize changes
            categories = {"source": 0, "tests": 0, "docs": 0, "config": 0}
            for fc in file_changes:
                if fc.get("is_test", False):
                    categories["tests"] += 1
                elif fc.get("is_documentation", False):
                    categories["docs"] += 1
                elif fc.get("is_config", False):
                    categories["config"] += 1
                else:
                    categories["source"] += 1
            
            for category, count in categories.items():
                if count > 0:
                    lines.append(f"- **{category.title()} files:** {count}")
        
        # Risk assessment
        risk_assessment = impact_analysis.get("risk_assessment", {})
        if risk_assessment:
            lines.extend([
                "",
                "### Risk Assessment:",
                f"- **Level:** {risk_assessment.get('level', 'unknown').title()}",
                f"- **Score:** {risk_assessment.get('score', 0):.2f}"
            ])
            
            risk_factors = risk_assessment.get("factors", [])
            if risk_factors:
                lines.append("- **Factors:**")
                for factor in risk_factors:
                    lines.append(f"  - {factor}")
        
        # Quality indicators
        quality = impact_analysis.get("quality_indicators", {})
        if quality:
            lines.extend([
                "",
                "### Quality Indicators:",
                f"- **Test coverage:** {'Yes' if quality.get('has_tests', False) else 'No'}",
                f"- **Documentation:** {'Yes' if quality.get('has_documentation', False) else 'No'}",
                f"- **Message quality:** {quality.get('commit_message_quality', 0):.2f}",
                f"- **Overall quality:** {quality.get('quality_score', 0):.2f}"
            ])
        
        return "\n".join(lines)
        
    except Exception as e:
        logging.error(f"Failed to generate commit summary: {e}")
        return ""

def generate_changelog_entry(commit_metadata: dict, impact_analysis: dict) -> str:
    """Generate changelog entry."""
    try:
        commit_type = commit_metadata["commit_type"]
        subject = commit_metadata["subject"]
        short_sha = commit_metadata["short_sha"]
        
        # Format based on commit type
        type_prefixes = {
            "feature": "✨ **Added:**",
            "bugfix": "🐛 **Fixed:**", 
            "documentation": "📚 **Docs:**",
            "refactor": "♻️ **Refactored:**",
            "test": "✅ **Tests:**",
            "chore": "🔧 **Chore:**",
            "style": "💄 **Style:**"
        }
        
        prefix = type_prefixes.get(commit_type, "🔄 **Changed:**")
        
        # Clean up commit message
        clean_subject = subject
        if ":" in clean_subject:
            clean_subject = clean_subject.split(":", 1)[1].strip()
        
        entry = f"{prefix} {clean_subject} ([{short_sha}])"
        
        # Add breaking change indicator
        breaking_potential = impact_analysis.get("breaking_change_potential", {})
        if breaking_potential.get("potential") in ["medium", "high"]:
            entry += " ⚠️ **BREAKING**"
        
        return entry
        
    except Exception as e:
        logging.error(f"Failed to generate changelog entry: {e}")
        return ""

def generate_breaking_change_notice(commit_metadata: dict, breaking_potential: dict) -> str:
    """Generate breaking change notice."""
    try:
        lines = [
            f"# ⚠️ Potential Breaking Change: {commit_metadata['short_sha']}",
            "",
            f"**Commit:** {commit_metadata['subject']}",
            f"**Risk Level:** {breaking_potential.get('potential', 'unknown').title()}",
            f"**Score:** {breaking_potential.get('score', 0):.2f}",
            ""
        ]
        
        indicators = breaking_potential.get("indicators", [])
        if indicators:
            lines.extend([
                "## Risk Indicators:",
                ""
            ])
            for indicator in indicators:
                lines.append(f"- {indicator}")
            lines.append("")
        
        lines.extend([
            "## Recommended Actions:",
            "",
            "1. Review the changes carefully before deployment",
            "2. Update documentation if APIs have changed", 
            "3. Test thoroughly in staging environment",
            "4. Consider versioning implications",
            "5. Communicate changes to team members"
        ])
        
        return "\n".join(lines)
        
    except Exception as e:
        logging.error(f"Failed to generate breaking change notice: {e}")
        return ""

def main():
    """Main hook execution."""
    try:
        if len(sys.argv) < 2:
            print("Usage: git-memory-correlate.py <commit_sha>", file=sys.stderr)
            sys.exit(0)
        
        commit_sha = sys.argv[1]
        
        # Perform correlation
        result = correlate_commit(commit_sha)
        
        if not result["success"]:
            print(f"Warning: Failed to correlate commit: {result.get('error', 'Unknown error')}", 
                  file=sys.stderr)
        else:
            logging.info(f"Successfully processed commit {commit_sha[:8]}")
        
        # Exit successfully
        sys.exit(0)
        
    except Exception as e:
        print(f"Post-commit hook error: {e}", file=sys.stderr)
        # Don't block on hook failure
        sys.exit(0)

if __name__ == "__main__":
    main()