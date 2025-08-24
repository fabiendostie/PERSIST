#!/usr/bin/env python3
"""
Git integration module for Prsist Memory System.
Core git operations and metadata extraction.
"""

import os
import re
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

class GitMetadataExtractor:
    """Extracts comprehensive metadata from git operations."""
    
    def __init__(self, repo_path: str = "."):
        """Initialize git metadata extractor."""
        self.repo_path = Path(repo_path).resolve()
        self.git_dir = self.repo_path / ".git"
        
        if not self.git_dir.exists():
            raise ValueError(f"Not a git repository: {self.repo_path}")
    
    def get_commit_metadata(self, commit_sha: str) -> Optional[Dict[str, Any]]:
        """Extract comprehensive metadata for a commit."""
        try:
            # Get basic commit information
            cmd = [
                "git", "show", "--format=%H|%s|%an|%ae|%at|%P", 
                "--name-status", "--no-patch", commit_sha
            ]
            result = subprocess.run(
                cmd, cwd=self.repo_path, capture_output=True, text=True, timeout=30
            )
            
            if result.returncode != 0:
                logging.error(f"Git show failed for {commit_sha}: {result.stderr}")
                return None
            
            lines = result.stdout.strip().split('\n')
            if not lines:
                return None
            
            # Parse commit header
            header_parts = lines[0].split('|')
            if len(header_parts) < 6:
                return None
            
            full_sha, subject, author_name, author_email, timestamp, parents = header_parts
            
            # Get commit stats
            stats = self.get_commit_stats(commit_sha)
            
            # Get file changes
            file_changes = self.get_file_changes(commit_sha)
            
            # Get branch information
            branches = self.get_commit_branches(commit_sha)
            
            # Calculate impact score
            impact_score = self.calculate_commit_impact(stats, file_changes)
            
            metadata = {
                "commit_sha": full_sha,
                "short_sha": full_sha[:8],
                "subject": subject,
                "author_name": author_name,
                "author_email": author_email,
                "timestamp": datetime.fromtimestamp(int(timestamp)).isoformat(),
                "unix_timestamp": int(timestamp),
                "parent_commits": parents.split() if parents else [],
                "branches": branches,
                "stats": stats,
                "file_changes": file_changes,
                "impact_score": impact_score,
                "commit_type": self.classify_commit_type(subject, file_changes)
            }
            
            return metadata
            
        except Exception as e:
            logging.error(f"Failed to extract commit metadata for {commit_sha}: {e}")
            return None
    
    def get_commit_stats(self, commit_sha: str) -> Dict[str, int]:
        """Get commit statistics (files changed, insertions, deletions)."""
        try:
            cmd = ["git", "show", "--stat", "--format=", commit_sha]
            result = subprocess.run(
                cmd, cwd=self.repo_path, capture_output=True, text=True, timeout=10
            )
            
            if result.returncode != 0:
                return {"files_changed": 0, "insertions": 0, "deletions": 0}
            
            # Parse the summary line (e.g., "3 files changed, 45 insertions(+), 12 deletions(-)")
            lines = result.stdout.strip().split('\n')
            summary_line = lines[-1] if lines else ""
            
            stats = {"files_changed": 0, "insertions": 0, "deletions": 0}
            
            # Extract numbers using regex
            files_match = re.search(r'(\d+) files? changed', summary_line)
            if files_match:
                stats["files_changed"] = int(files_match.group(1))
            
            insertions_match = re.search(r'(\d+) insertions?\(\+\)', summary_line)
            if insertions_match:
                stats["insertions"] = int(insertions_match.group(1))
            
            deletions_match = re.search(r'(\d+) deletions?\(-\)', summary_line)
            if deletions_match:
                stats["deletions"] = int(deletions_match.group(1))
            
            return stats
            
        except Exception as e:
            logging.error(f"Failed to get commit stats for {commit_sha}: {e}")
            return {"files_changed": 0, "insertions": 0, "deletions": 0}
    
    def get_file_changes(self, commit_sha: str) -> List[Dict[str, Any]]:
        """Get detailed file changes for a commit."""
        try:
            cmd = ["git", "show", "--name-status", "--format=", commit_sha]
            result = subprocess.run(
                cmd, cwd=self.repo_path, capture_output=True, text=True, timeout=15
            )
            
            if result.returncode != 0:
                return []
            
            changes = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                
                status = parts[0]
                file_path = parts[1]
                
                change_type = self.parse_change_type(status)
                significance = self.calculate_file_significance(file_path, change_type)
                
                change = {
                    "file_path": file_path,
                    "change_type": change_type,
                    "status": status,
                    "significance_score": significance,
                    "file_type": self.get_file_type(file_path),
                    "is_test": self.is_test_file(file_path),
                    "is_config": self.is_config_file(file_path),
                    "is_documentation": self.is_documentation_file(file_path)
                }
                
                # Get line-level changes for this file
                line_changes = self.get_file_line_changes(commit_sha, file_path)
                change.update(line_changes)
                
                changes.append(change)
            
            return changes
            
        except Exception as e:
            logging.error(f"Failed to get file changes for {commit_sha}: {e}")
            return []
    
    def get_file_line_changes(self, commit_sha: str, file_path: str) -> Dict[str, int]:
        """Get line-level changes for a specific file."""
        try:
            cmd = ["git", "show", "--numstat", "--format=", commit_sha, "--", file_path]
            result = subprocess.run(
                cmd, cwd=self.repo_path, capture_output=True, text=True, timeout=10
            )
            
            if result.returncode != 0 or not result.stdout.strip():
                return {"lines_added": 0, "lines_deleted": 0}
            
            line = result.stdout.strip()
            parts = line.split('\t')
            
            if len(parts) >= 2:
                added = int(parts[0]) if parts[0] != '-' else 0
                deleted = int(parts[1]) if parts[1] != '-' else 0
                return {"lines_added": added, "lines_deleted": deleted}
            
            return {"lines_added": 0, "lines_deleted": 0}
            
        except Exception as e:
            logging.error(f"Failed to get line changes for {file_path}: {e}")
            return {"lines_added": 0, "lines_deleted": 0}
    
    def get_commit_branches(self, commit_sha: str) -> List[str]:
        """Get branches that contain this commit."""
        try:
            cmd = ["git", "branch", "-a", "--contains", commit_sha]
            result = subprocess.run(
                cmd, cwd=self.repo_path, capture_output=True, text=True, timeout=10
            )
            
            if result.returncode != 0:
                return []
            
            branches = []
            for line in result.stdout.strip().split('\n'):
                branch = line.strip().lstrip('* ').strip()
                if branch and not branch.startswith('('):
                    # Clean up remote branch names
                    if branch.startswith('remotes/'):
                        branch = branch.replace('remotes/', '')
                    branches.append(branch)
            
            return list(set(branches))  # Remove duplicates
            
        except Exception as e:
            logging.error(f"Failed to get branches for {commit_sha}: {e}")
            return []
    
    def get_current_branch(self) -> str:
        """Get the current branch name."""
        try:
            cmd = ["git", "branch", "--show-current"]
            result = subprocess.run(
                cmd, cwd=self.repo_path, capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            
            # Fallback for detached HEAD
            cmd = ["git", "rev-parse", "--abbrev-ref", "HEAD"]
            result = subprocess.run(
                cmd, cwd=self.repo_path, capture_output=True, text=True, timeout=5
            )
            
            return result.stdout.strip() if result.returncode == 0 else "unknown"
            
        except Exception as e:
            logging.error(f"Failed to get current branch: {e}")
            return "unknown"
    
    def get_latest_commit_sha(self) -> Optional[str]:
        """Get the SHA of the latest commit."""
        try:
            cmd = ["git", "rev-parse", "HEAD"]
            result = subprocess.run(
                cmd, cwd=self.repo_path, capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            return None
            
        except Exception as e:
            logging.error(f"Failed to get latest commit SHA: {e}")
            return None
    
    def get_staged_files(self) -> List[str]:
        """Get list of staged files."""
        try:
            cmd = ["git", "diff", "--cached", "--name-only"]
            result = subprocess.run(
                cmd, cwd=self.repo_path, capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                return [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
            return []
            
        except Exception as e:
            logging.error(f"Failed to get staged files: {e}")
            return []
    
    def parse_change_type(self, status: str) -> str:
        """Parse git status code to change type."""
        if status.startswith('A'):
            return 'added'
        elif status.startswith('M'):
            return 'modified'
        elif status.startswith('D'):
            return 'deleted'
        elif status.startswith('R'):
            return 'renamed'
        elif status.startswith('C'):
            return 'copied'
        elif status.startswith('T'):
            return 'type_changed'
        else:
            return 'unknown'
    
    def calculate_file_significance(self, file_path: str, change_type: str) -> float:
        """Calculate significance score for a file change."""
        score = 0.5  # Base score
        
        # File type significance
        if self.is_config_file(file_path):
            score += 0.3
        elif self.is_documentation_file(file_path):
            score += 0.1
        elif self.is_test_file(file_path):
            score += 0.2
        else:
            score += 0.4  # Source code files
        
        # Change type significance
        if change_type == 'added':
            score += 0.3
        elif change_type == 'deleted':
            score += 0.4
        elif change_type == 'modified':
            score += 0.2
        elif change_type == 'renamed':
            score += 0.1
        
        # File location significance
        if 'core' in file_path.lower() or 'main' in file_path.lower():
            score += 0.2
        elif 'test' in file_path.lower():
            score += 0.1
        
        return min(score, 1.0)
    
    def calculate_commit_impact(self, stats: Dict[str, int], file_changes: List[Dict]) -> float:
        """Calculate overall impact score for a commit."""
        if not file_changes:
            return 0.0
        
        # Base score from statistics
        files_score = min(stats.get("files_changed", 0) * 0.1, 0.5)
        lines_score = min((stats.get("insertions", 0) + stats.get("deletions", 0)) * 0.001, 0.3)
        
        # Significance from file changes
        avg_significance = sum(fc.get("significance_score", 0) for fc in file_changes) / len(file_changes)
        
        # Special file types boost
        has_config = any(fc.get("is_config", False) for fc in file_changes)
        has_core = any("core" in fc.get("file_path", "").lower() for fc in file_changes)
        
        boost = 0.0
        if has_config:
            boost += 0.1
        if has_core:
            boost += 0.15
        
        total_score = files_score + lines_score + avg_significance + boost
        return min(total_score, 1.0)
    
    def classify_commit_type(self, subject: str, file_changes: List[Dict]) -> str:
        """Classify commit type based on message and changes."""
        subject_lower = subject.lower()
        
        # Check commit message patterns
        if any(pattern in subject_lower for pattern in ['feat:', 'feature:']):
            return 'feature'
        elif any(pattern in subject_lower for pattern in ['fix:', 'bugfix:', 'hotfix:']):
            return 'bugfix'
        elif any(pattern in subject_lower for pattern in ['docs:', 'doc:']):
            return 'documentation'
        elif any(pattern in subject_lower for pattern in ['test:', 'tests:']):
            return 'test'
        elif any(pattern in subject_lower for pattern in ['refactor:', 'refact:']):
            return 'refactor'
        elif any(pattern in subject_lower for pattern in ['style:', 'format:']):
            return 'style'
        elif any(pattern in subject_lower for pattern in ['chore:', 'build:', 'ci:']):
            return 'chore'
        
        # Analyze file changes
        if not file_changes:
            return 'unknown'
        
        test_files = sum(1 for fc in file_changes if fc.get("is_test", False))
        doc_files = sum(1 for fc in file_changes if fc.get("is_documentation", False))
        config_files = sum(1 for fc in file_changes if fc.get("is_config", False))
        
        total_files = len(file_changes)
        
        if test_files / total_files > 0.7:
            return 'test'
        elif doc_files / total_files > 0.7:
            return 'documentation'
        elif config_files / total_files > 0.5:
            return 'configuration'
        
        return 'development'
    
    def get_file_type(self, file_path: str) -> str:
        """Get file type based on extension."""
        ext = Path(file_path).suffix.lower()
        
        type_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.jsx': 'javascript',
            '.go': 'go',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'header',
            '.hpp': 'header',
            '.cs': 'csharp',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.md': 'markdown',
            '.txt': 'text',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.xml': 'xml',
            '.html': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.sql': 'sql'
        }
        
        return type_map.get(ext, 'unknown')
    
    def is_test_file(self, file_path: str) -> bool:
        """Check if file is a test file."""
        path_lower = file_path.lower()
        return any(pattern in path_lower for pattern in [
            'test', 'tests', 'spec', 'specs', '__test__', '.test.', '.spec.'
        ])
    
    def is_config_file(self, file_path: str) -> bool:
        """Check if file is a configuration file."""
        path_lower = file_path.lower()
        return any(pattern in path_lower for pattern in [
            'config', 'configuration', 'settings', 'package.json', 'requirements.txt',
            'dockerfile', 'docker-compose', '.env', 'makefile', 'cmake', 'build.gradle',
            'pom.xml', 'cargo.toml', 'pyproject.toml', 'setup.py', '.gitignore', '.github'
        ])
    
    def is_documentation_file(self, file_path: str) -> bool:
        """Check if file is documentation."""
        path_lower = file_path.lower()
        return any(pattern in path_lower for pattern in [
            'readme', 'docs', 'documentation', '.md', 'changelog', 'license', 'contributing'
        ]) or file_path.endswith('.md')


class ChangeImpactAnalyzer:
    """Analyzes the impact of code changes."""
    
    def __init__(self, git_extractor: GitMetadataExtractor):
        """Initialize change impact analyzer."""
        self.git_extractor = git_extractor
        
    def analyze_commit_impact(self, commit_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the overall impact of a commit."""
        file_changes = commit_metadata.get("file_changes", [])
        stats = commit_metadata.get("stats", {})
        
        analysis = {
            "overall_impact": commit_metadata.get("impact_score", 0),
            "change_complexity": self.calculate_change_complexity(file_changes, stats),
            "risk_assessment": self.assess_change_risk(file_changes),
            "productivity_metrics": self.calculate_productivity_metrics(stats),
            "quality_indicators": self.assess_quality_indicators(commit_metadata),
            "breaking_change_potential": self.assess_breaking_changes(file_changes),
            "test_coverage_impact": self.assess_test_coverage_impact(file_changes),
            "documentation_impact": self.assess_documentation_impact(file_changes)
        }
        
        return analysis
    
    def calculate_change_complexity(self, file_changes: List[Dict], stats: Dict[str, int]) -> float:
        """Calculate complexity score based on changes."""
        if not file_changes:
            return 0.0
        
        # File count complexity
        file_complexity = min(len(file_changes) * 0.1, 0.4)
        
        # Line changes complexity
        total_lines = stats.get("insertions", 0) + stats.get("deletions", 0)
        line_complexity = min(total_lines * 0.001, 0.3)
        
        # File type diversity
        file_types = set(fc.get("file_type", "unknown") for fc in file_changes)
        type_complexity = min(len(file_types) * 0.05, 0.2)
        
        # Configuration file changes add complexity
        config_changes = sum(1 for fc in file_changes if fc.get("is_config", False))
        config_complexity = min(config_changes * 0.1, 0.1)
        
        return min(file_complexity + line_complexity + type_complexity + config_complexity, 1.0)
    
    def assess_change_risk(self, file_changes: List[Dict]) -> Dict[str, Any]:
        """Assess risk level of changes."""
        if not file_changes:
            return {"level": "none", "score": 0.0, "factors": []}
        
        risk_factors = []
        risk_score = 0.0
        
        # Check for high-risk file types
        config_files = [fc for fc in file_changes if fc.get("is_config", False)]
        if config_files:
            risk_score += 0.3
            risk_factors.append(f"Configuration files modified: {len(config_files)}")
        
        # Check for core file modifications
        core_files = [fc for fc in file_changes if "core" in fc.get("file_path", "").lower()]
        if core_files:
            risk_score += 0.4
            risk_factors.append(f"Core files modified: {len(core_files)}")
        
        # Check for deletions
        deletions = [fc for fc in file_changes if fc.get("change_type") == "deleted"]
        if deletions:
            risk_score += 0.2
            risk_factors.append(f"Files deleted: {len(deletions)}")
        
        # Check for large files
        large_changes = [fc for fc in file_changes 
                        if fc.get("lines_added", 0) + fc.get("lines_deleted", 0) > 100]
        if large_changes:
            risk_score += 0.2
            risk_factors.append(f"Large file changes: {len(large_changes)}")
        
        # Determine risk level
        if risk_score < 0.3:
            level = "low"
        elif risk_score < 0.6:
            level = "medium"
        else:
            level = "high"
        
        return {
            "level": level,
            "score": min(risk_score, 1.0),
            "factors": risk_factors
        }
    
    def calculate_productivity_metrics(self, stats: Dict[str, int]) -> Dict[str, Any]:
        """Calculate productivity metrics from commit stats."""
        files_changed = stats.get("files_changed", 0)
        insertions = stats.get("insertions", 0)
        deletions = stats.get("deletions", 0)
        
        net_lines = insertions - deletions
        total_lines = insertions + deletions
        
        return {
            "files_modified": files_changed,
            "lines_added": insertions,
            "lines_removed": deletions,
            "net_line_change": net_lines,
            "total_line_change": total_lines,
            "average_lines_per_file": total_lines / files_changed if files_changed > 0 else 0,
            "code_churn": deletions / max(insertions, 1),
            "productivity_score": min(total_lines * 0.01, 1.0)
        }
    
    def assess_quality_indicators(self, commit_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Assess code quality indicators."""
        file_changes = commit_metadata.get("file_changes", [])
        commit_message = commit_metadata.get("subject", "")
        
        # Test file ratio
        test_files = sum(1 for fc in file_changes if fc.get("is_test", False))
        total_files = len(file_changes)
        test_ratio = test_files / total_files if total_files > 0 else 0
        
        # Documentation ratio
        doc_files = sum(1 for fc in file_changes if fc.get("is_documentation", False))
        doc_ratio = doc_files / total_files if total_files > 0 else 0
        
        # Commit message quality
        message_quality = self.assess_commit_message_quality(commit_message)
        
        return {
            "test_file_ratio": test_ratio,
            "documentation_ratio": doc_ratio,
            "commit_message_quality": message_quality,
            "has_tests": test_files > 0,
            "has_documentation": doc_files > 0,
            "quality_score": (test_ratio * 0.4 + doc_ratio * 0.2 + message_quality * 0.4)
        }
    
    def assess_commit_message_quality(self, message: str) -> float:
        """Assess quality of commit message."""
        if not message:
            return 0.0
        
        score = 0.0
        
        # Length check
        if 10 <= len(message) <= 72:
            score += 0.3
        elif len(message) > 5:
            score += 0.1
        
        # Conventional commit format
        if re.match(r'^(feat|fix|docs|style|refactor|test|chore)(\(.+\))?: .+', message):
            score += 0.4
        
        # Capitalization
        if message[0].isupper():
            score += 0.1
        
        # No period at end
        if not message.endswith('.'):
            score += 0.1
        
        # Descriptive content
        if len(message.split()) >= 3:
            score += 0.1
        
        return min(score, 1.0)
    
    def assess_breaking_changes(self, file_changes: List[Dict]) -> Dict[str, Any]:
        """Assess potential for breaking changes."""
        if not file_changes:
            return {"potential": "none", "score": 0.0, "indicators": []}
        
        indicators = []
        score = 0.0
        
        # API file changes
        api_files = [fc for fc in file_changes 
                    if any(term in fc.get("file_path", "").lower() 
                          for term in ["api", "interface", "contract", "schema"])]
        if api_files:
            score += 0.4
            indicators.append(f"API files modified: {len(api_files)}")
        
        # Configuration changes
        config_files = [fc for fc in file_changes if fc.get("is_config", False)]
        if config_files:
            score += 0.3
            indicators.append(f"Configuration files modified: {len(config_files)}")
        
        # Database migration files
        migration_files = [fc for fc in file_changes 
                          if any(term in fc.get("file_path", "").lower() 
                                for term in ["migration", "schema", "database"])]
        if migration_files:
            score += 0.5
            indicators.append(f"Database files modified: {len(migration_files)}")
        
        # File deletions
        deletions = [fc for fc in file_changes if fc.get("change_type") == "deleted"]
        if deletions:
            score += 0.3
            indicators.append(f"Files deleted: {len(deletions)}")
        
        # Determine potential level
        if score < 0.3:
            potential = "low"
        elif score < 0.6:
            potential = "medium"
        else:
            potential = "high"
        
        return {
            "potential": potential,
            "score": min(score, 1.0),
            "indicators": indicators
        }
    
    def assess_test_coverage_impact(self, file_changes: List[Dict]) -> Dict[str, Any]:
        """Assess impact on test coverage."""
        if not file_changes:
            return {"impact": "none", "test_files": 0, "source_files": 0, "ratio": 0.0}
        
        test_files = [fc for fc in file_changes if fc.get("is_test", False)]
        source_files = [fc for fc in file_changes if not fc.get("is_test", False) 
                       and not fc.get("is_documentation", False) 
                       and not fc.get("is_config", False)]
        
        test_count = len(test_files)
        source_count = len(source_files)
        ratio = test_count / source_count if source_count > 0 else 0
        
        if ratio >= 0.8:
            impact = "excellent"
        elif ratio >= 0.5:
            impact = "good"
        elif ratio >= 0.2:
            impact = "moderate"
        elif test_count > 0:
            impact = "minimal"
        else:
            impact = "none"
        
        return {
            "impact": impact,
            "test_files": test_count,
            "source_files": source_count,
            "ratio": ratio
        }
    
    def assess_documentation_impact(self, file_changes: List[Dict]) -> Dict[str, Any]:
        """Assess impact on documentation."""
        if not file_changes:
            return {"impact": "none", "doc_files": 0, "source_files": 0}
        
        doc_files = [fc for fc in file_changes if fc.get("is_documentation", False)]
        source_files = [fc for fc in file_changes if not fc.get("is_documentation", False)]
        
        doc_count = len(doc_files)
        source_count = len(source_files)
        
        if doc_count > 0 and source_count > 0:
            impact = "updated"
        elif doc_count > 0:
            impact = "documentation_only"
        elif source_count > 0:
            impact = "needs_documentation"
        else:
            impact = "none"
        
        return {
            "impact": impact,
            "doc_files": doc_count,
            "source_files": source_count
        }