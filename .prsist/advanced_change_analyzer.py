#!/usr/bin/env python3
"""
Advanced change impact analyzer for Prsist Memory System Phase 3.
Analyzes file changes and their impact on memory and sessions.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import re
import hashlib

from database import MemoryDatabase
from utils import setup_logging

class FileTypeAnalyzer:
    """Base class for file type specific analyzers."""
    
    def analyze(self, file_path: str, change_type: str, content_diff: str = None) -> Dict[str, Any]:
        """Analyze file change impact."""
        return {
            'impact_score': 0.5,
            'priority': 'medium',
            'analysis_type': 'generic'
        }

class CodeFileAnalyzer(FileTypeAnalyzer):
    """Analyzer for code files."""
    
    def __init__(self):
        """Initialize code file analyzer."""
        self.critical_patterns = {
            'function_signature': r'def\s+\w+\(|function\s+\w+\(|class\s+\w+',
            'import_changes': r'import\s+|from\s+\w+\s+import',
            'api_changes': r'@app\.route|@api\.|@endpoint',
            'config_changes': r'config\.|settings\.|CONFIG_',
            'database_changes': r'CREATE\s+TABLE|ALTER\s+TABLE|DROP\s+TABLE',
            'error_handling': r'try:|except|catch|throw|raise'
        }
        
        self.complexity_indicators = {
            'class_definition': 0.8,
            'function_definition': 0.6,
            'import_statement': 0.7,
            'loop_structure': 0.5,
            'conditional_logic': 0.4,
            'error_handling': 0.6
        }
    
    def analyze(self, file_path: str, change_type: str, content_diff: str = None) -> Dict[str, Any]:
        """Analyze code file changes."""
        try:
            impact_score = 0.5
            analysis_details = {
                'file_type': 'code',
                'language': self._detect_language(file_path),
                'change_categories': [],
                'complexity_factors': [],
                'breaking_change_risk': False
            }
            
            # Base impact based on change type
            change_impact = {
                'add': 0.7,
                'change': 0.6,
                'delete': 0.8,  # Deletions can break things
                'rename': 0.5
            }
            impact_score = change_impact.get(change_type, 0.5)
            
            # Analyze content if available
            if content_diff:
                content_impact = self._analyze_content_changes(content_diff)
                impact_score = max(impact_score, content_impact['impact_score'])
                analysis_details.update(content_impact)
            
            # Adjust based on file characteristics
            file_impact = self._analyze_file_characteristics(file_path)
            impact_score = min(1.0, impact_score + file_impact['adjustment'])
            analysis_details.update(file_impact)
            
            # Determine priority
            if impact_score > 0.8:
                priority = 'high'
            elif impact_score > 0.6:
                priority = 'medium'
            else:
                priority = 'low'
            
            return {
                'impact_score': impact_score,
                'priority': priority,
                'analysis_type': 'code',
                'details': analysis_details
            }
            
        except Exception as e:
            logging.error(f"Failed to analyze code file {file_path}: {e}")
            return {'impact_score': 0.5, 'priority': 'medium', 'error': str(e)}
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        ext = Path(file_path).suffix.lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.go': 'go',
            '.java': 'java',
            '.cpp': 'cpp', '.cxx': 'cpp', '.cc': 'cpp',
            '.cs': 'csharp',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php'
        }
        return language_map.get(ext, 'unknown')
    
    def _analyze_content_changes(self, content_diff: str) -> Dict[str, Any]:
        """Analyze the content of changes."""
        impact_score = 0.5
        change_categories = []
        complexity_factors = []
        breaking_change_risk = False
        
        try:
            # Check for critical patterns
            for pattern_name, pattern in self.critical_patterns.items():
                if re.search(pattern, content_diff, re.IGNORECASE):
                    change_categories.append(pattern_name)
                    
                    # Adjust impact based on pattern criticality
                    if pattern_name in ['function_signature', 'api_changes']:
                        impact_score = max(impact_score, 0.8)
                        breaking_change_risk = True
                    elif pattern_name in ['import_changes', 'config_changes']:
                        impact_score = max(impact_score, 0.7)
                    elif pattern_name in ['database_changes']:
                        impact_score = max(impact_score, 0.9)
                        breaking_change_risk = True
            
            # Analyze complexity indicators
            lines = content_diff.split('\n')
            added_lines = [line for line in lines if line.startswith('+')]
            removed_lines = [line for line in lines if line.startswith('-')]
            
            # Count complexity factors
            for line in added_lines + removed_lines:
                if re.search(r'class\s+\w+', line):
                    complexity_factors.append('class_definition')
                    impact_score = max(impact_score, 0.8)
                elif re.search(r'def\s+\w+|function\s+\w+', line):
                    complexity_factors.append('function_definition')
                    impact_score = max(impact_score, 0.6)
                elif re.search(r'for\s+|while\s+', line):
                    complexity_factors.append('loop_structure')
                elif re.search(r'if\s+|elif\s+|else:', line):
                    complexity_factors.append('conditional_logic')
            
            # Large changes are more impactful
            change_size = len(added_lines) + len(removed_lines)
            if change_size > 50:
                impact_score = min(1.0, impact_score + 0.2)
                complexity_factors.append('large_change')
            elif change_size > 20:
                impact_score = min(1.0, impact_score + 0.1)
                complexity_factors.append('medium_change')
            
        except Exception as e:
            logging.error(f"Failed to analyze content changes: {e}")
        
        return {
            'impact_score': impact_score,
            'change_categories': list(set(change_categories)),
            'complexity_factors': list(set(complexity_factors)),
            'breaking_change_risk': breaking_change_risk
        }
    
    def _analyze_file_characteristics(self, file_path: str) -> Dict[str, Any]:
        """Analyze file-specific characteristics."""
        adjustment = 0.0
        characteristics = []
        
        try:
            path_parts = Path(file_path).parts
            file_name = Path(file_path).name.lower()
            
            # Critical file patterns
            if any(part in ['core', 'main', 'index', 'app'] for part in path_parts):
                adjustment += 0.2
                characteristics.append('core_file')
            
            # Test files are less critical for immediate impact
            if 'test' in file_name or any('test' in part for part in path_parts):
                adjustment -= 0.1
                characteristics.append('test_file')
            
            # Configuration files can be highly impactful
            if file_name in ['config.py', 'settings.py', '__init__.py']:
                adjustment += 0.15
                characteristics.append('configuration_file')
            
            # API or route files are critical
            if any(keyword in file_name for keyword in ['api', 'route', 'endpoint', 'handler']):
                adjustment += 0.2
                characteristics.append('api_file')
            
            # Database related files
            if any(keyword in file_name for keyword in ['model', 'schema', 'migration', 'db']):
                adjustment += 0.15
                characteristics.append('database_file')
            
        except Exception as e:
            logging.error(f"Failed to analyze file characteristics: {e}")
        
        return {
            'adjustment': adjustment,
            'characteristics': characteristics
        }

class ConfigFileAnalyzer(FileTypeAnalyzer):
    """Analyzer for configuration files."""
    
    def analyze(self, file_path: str, change_type: str, content_diff: str = None) -> Dict[str, Any]:
        """Analyze configuration file changes."""
        try:
            # Config changes are generally high impact
            base_impact = 0.8
            
            analysis_details = {
                'file_type': 'configuration',
                'config_type': self._detect_config_type(file_path),
                'critical_sections': [],
                'requires_restart': False
            }
            
            if content_diff:
                content_analysis = self._analyze_config_content(content_diff)
                base_impact = max(base_impact, content_analysis['impact_score'])
                analysis_details.update(content_analysis)
            
            return {
                'impact_score': base_impact,
                'priority': 'high',
                'analysis_type': 'configuration',
                'details': analysis_details
            }
            
        except Exception as e:
            logging.error(f"Failed to analyze config file {file_path}: {e}")
            return {'impact_score': 0.8, 'priority': 'high', 'error': str(e)}
    
    def _detect_config_type(self, file_path: str) -> str:
        """Detect configuration file type."""
        ext = Path(file_path).suffix.lower()
        name = Path(file_path).name.lower()
        
        if ext == '.yaml' or ext == '.yml':
            return 'yaml'
        elif ext == '.json':
            return 'json'
        elif ext == '.toml':
            return 'toml'
        elif ext == '.ini' or ext == '.cfg':
            return 'ini'
        elif 'docker' in name:
            return 'docker'
        elif 'makefile' in name:
            return 'makefile'
        else:
            return 'unknown'
    
    def _analyze_config_content(self, content_diff: str) -> Dict[str, Any]:
        """Analyze configuration content changes."""
        critical_sections = []
        requires_restart = False
        impact_score = 0.8
        
        # Critical configuration patterns
        critical_patterns = {
            'database': ['database', 'db_', 'connection', 'datasource'],
            'security': ['password', 'secret', 'key', 'token', 'auth'],
            'networking': ['host', 'port', 'url', 'endpoint', 'proxy'],
            'performance': ['cache', 'pool', 'timeout', 'limit', 'memory'],
            'logging': ['log', 'debug', 'level', 'output'],
            'features': ['enable', 'disable', 'feature', 'flag']
        }
        
        content_lower = content_diff.lower()
        
        for section, patterns in critical_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                critical_sections.append(section)
                
                if section in ['database', 'security', 'networking']:
                    impact_score = 0.9
                    requires_restart = True
        
        return {
            'impact_score': impact_score,
            'critical_sections': critical_sections,
            'requires_restart': requires_restart
        }

class DocumentationAnalyzer(FileTypeAnalyzer):
    """Analyzer for documentation files."""
    
    def analyze(self, file_path: str, change_type: str, content_diff: str = None) -> Dict[str, Any]:
        """Analyze documentation file changes."""
        try:
            # Documentation changes are generally lower impact
            base_impact = 0.3
            
            analysis_details = {
                'file_type': 'documentation',
                'doc_type': self._detect_doc_type(file_path),
                'update_type': change_type
            }
            
            # Some documentation is more critical
            file_name = Path(file_path).name.lower()
            if file_name in ['readme.md', 'api.md', 'changelog.md']:
                base_impact = 0.6
                analysis_details['critical_doc'] = True
            
            return {
                'impact_score': base_impact,
                'priority': 'low' if base_impact < 0.5 else 'medium',
                'analysis_type': 'documentation',
                'details': analysis_details
            }
            
        except Exception as e:
            logging.error(f"Failed to analyze documentation file {file_path}: {e}")
            return {'impact_score': 0.3, 'priority': 'low', 'error': str(e)}
    
    def _detect_doc_type(self, file_path: str) -> str:
        """Detect documentation type."""
        ext = Path(file_path).suffix.lower()
        name = Path(file_path).name.lower()
        
        if ext == '.md':
            return 'markdown'
        elif ext == '.rst':
            return 'restructuredtext'
        elif 'readme' in name:
            return 'readme'
        elif 'changelog' in name:
            return 'changelog'
        elif 'api' in name:
            return 'api_documentation'
        else:
            return 'general'

class TestFileAnalyzer(FileTypeAnalyzer):
    """Analyzer for test files."""
    
    def analyze(self, file_path: str, change_type: str, content_diff: str = None) -> Dict[str, Any]:
        """Analyze test file changes."""
        try:
            # Test changes indicate code changes
            base_impact = 0.5
            
            analysis_details = {
                'file_type': 'test',
                'test_type': self._detect_test_type(file_path),
                'indicates_code_changes': True
            }
            
            # Integration tests are more critical than unit tests
            if 'integration' in file_path.lower() or 'e2e' in file_path.lower():
                base_impact = 0.7
                analysis_details['test_criticality'] = 'high'
            elif 'unit' in file_path.lower():
                base_impact = 0.4
                analysis_details['test_criticality'] = 'medium'
            
            return {
                'impact_score': base_impact,
                'priority': 'medium',
                'analysis_type': 'test',
                'details': analysis_details
            }
            
        except Exception as e:
            logging.error(f"Failed to analyze test file {file_path}: {e}")
            return {'impact_score': 0.5, 'priority': 'medium', 'error': str(e)}
    
    def _detect_test_type(self, file_path: str) -> str:
        """Detect test type."""
        path_lower = file_path.lower()
        
        if 'unit' in path_lower:
            return 'unit'
        elif 'integration' in path_lower:
            return 'integration'
        elif 'e2e' in path_lower or 'end-to-end' in path_lower:
            return 'end_to_end'
        elif 'performance' in path_lower or 'perf' in path_lower:
            return 'performance'
        else:
            return 'general'

class AdvancedChangeImpactAnalyzer:
    """Advanced file change impact analyzer with intelligent analysis."""
    
    def __init__(self, memory_db: MemoryDatabase):
        """Initialize advanced change impact analyzer."""
        self.memory_db = memory_db
        self.file_type_analyzers = {
            'code': CodeFileAnalyzer(),
            'config': ConfigFileAnalyzer(),
            'documentation': DocumentationAnalyzer(),
            'test': TestFileAnalyzer()
        }
    
    def analyze_change_impact(self, file_path: str, change_type: str, 
                            content_diff: str = None, session_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze the comprehensive impact of a file change."""
        try:
            logging.info(f"Analyzing change impact for {file_path} ({change_type})")
            
            # Detect file type and get appropriate analyzer
            file_type = self.detect_file_type(file_path)
            analyzer = self.file_type_analyzers.get(file_type, FileTypeAnalyzer())
            
            # Perform base analysis
            base_analysis = analyzer.analyze(file_path, change_type, content_diff)
            
            # Enhance with context analysis
            context_analysis = self.analyze_context_impact(file_path, session_context)
            
            # Analyze session impact
            session_impact = self.analyze_session_impact(file_path, change_type)
            
            # Calculate memory implications
            memory_implications = self.calculate_memory_implications(file_path, base_analysis)
            
            # Combine all analyses
            comprehensive_analysis = {
                'file_path': file_path,
                'change_type': change_type,
                'overall_impact': self._calculate_overall_impact(
                    base_analysis, context_analysis, session_impact
                ),
                'base_analysis': base_analysis,
                'context_analysis': context_analysis,
                'session_impact': session_impact,
                'memory_implications': memory_implications,
                'recommendations': self._generate_recommendations(
                    base_analysis, context_analysis, session_impact
                ),
                'analyzed_at': datetime.now().isoformat()
            }
            
            # Store analysis in database
            self._store_analysis(comprehensive_analysis)
            
            return comprehensive_analysis
            
        except Exception as e:
            logging.error(f"Failed to analyze change impact for {file_path}: {e}")
            return {
                'file_path': file_path,
                'change_type': change_type,
                'error': str(e),
                'overall_impact': 0.5
            }
    
    def detect_file_type(self, file_path: str) -> str:
        """Detect the type of file for analysis."""
        ext = Path(file_path).suffix.lower()
        name = Path(file_path).name.lower()
        path_parts = Path(file_path).parts
        
        # Code files
        code_extensions = {'.py', '.js', '.ts', '.go', '.java', '.cpp', '.cs', '.rs', '.rb', '.php'}
        if ext in code_extensions:
            return 'code'
        
        # Configuration files
        config_extensions = {'.yaml', '.yml', '.json', '.toml', '.ini', '.cfg'}
        config_names = {'dockerfile', 'makefile', 'requirements.txt', 'package.json'}
        if ext in config_extensions or name in config_names:
            return 'config'
        
        # Documentation files
        doc_extensions = {'.md', '.rst', '.txt'}
        if ext in doc_extensions or 'readme' in name or 'changelog' in name:
            return 'documentation'
        
        # Test files
        if 'test' in name or any('test' in part for part in path_parts):
            return 'test'
        
        return 'unknown'
    
    def analyze_context_impact(self, file_path: str, session_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze impact based on current session context."""
        try:
            context_impact = {
                'relevance_to_current_work': 0.5,
                'related_files': [],
                'workflow_disruption': False,
                'context_invalidation_required': False
            }
            
            if not session_context:
                return context_impact
            
            # Check if file is related to current work
            current_files = session_context.get('recent_files', [])
            if any(file_path in f or f in file_path for f in current_files):
                context_impact['relevance_to_current_work'] = 0.9
                context_impact['workflow_disruption'] = True
            
            # Check if change affects current session goals
            current_goals = session_context.get('goals', [])
            if current_goals:
                # Simple keyword matching for relevance
                file_keywords = set(Path(file_path).stem.split('_'))
                for goal in current_goals:
                    if isinstance(goal, str):
                        goal_keywords = set(goal.lower().split())
                        if file_keywords.intersection(goal_keywords):
                            context_impact['relevance_to_current_work'] = max(
                                context_impact['relevance_to_current_work'], 0.7
                            )
            
            # Determine if context invalidation is needed
            if context_impact['relevance_to_current_work'] > 0.7:
                context_impact['context_invalidation_required'] = True
            
            return context_impact
            
        except Exception as e:
            logging.error(f"Failed to analyze context impact: {e}")
            return {'relevance_to_current_work': 0.5, 'error': str(e)}
    
    def analyze_session_impact(self, file_path: str, change_type: str) -> Dict[str, Any]:
        """Analyze impact on active sessions."""
        try:
            # Get active sessions
            active_sessions = self.memory_db.get_active_sessions()
            
            affected_sessions = []
            total_impact = 0.0
            
            for session in active_sessions:
                session_impact = self._calculate_session_file_impact(
                    session, file_path, change_type
                )
                
                if session_impact['impact_score'] > 0.3:
                    affected_sessions.append({
                        'session_id': session['id'],
                        'impact_score': session_impact['impact_score'],
                        'impact_reasons': session_impact['reasons']
                    })
                    total_impact += session_impact['impact_score']
            
            avg_impact = total_impact / len(active_sessions) if active_sessions else 0.0
            
            return {
                'affected_sessions': affected_sessions,
                'total_sessions_affected': len(affected_sessions),
                'average_impact': avg_impact,
                'requires_session_refresh': avg_impact > 0.6
            }
            
        except Exception as e:
            logging.error(f"Failed to analyze session impact: {e}")
            return {'affected_sessions': [], 'error': str(e)}
    
    def _calculate_session_file_impact(self, session: Dict[str, Any], 
                                     file_path: str, change_type: str) -> Dict[str, Any]:
        """Calculate impact of file change on a specific session."""
        impact_score = 0.0
        reasons = []
        
        try:
            session_id = session['id']
            
            # Check recent file interactions
            recent_files = self.memory_db.get_recent_file_interactions(session_id, limit=20)
            file_interactions = [f for f in recent_files if file_path in f.get('file_path', '')]
            
            if file_interactions:
                impact_score = 0.8
                reasons.append(f"Recent interactions ({len(file_interactions)} times)")
            
            # Check tool usage patterns
            recent_tools = self.memory_db.get_session_tool_usage(session_id)
            tool_file_interactions = [
                t for t in recent_tools 
                if file_path in str(t.get('input_data', '')) or file_path in str(t.get('output_data', ''))
            ]
            
            if tool_file_interactions:
                impact_score = max(impact_score, 0.6)
                reasons.append(f"Tool usage involving file ({len(tool_file_interactions)} operations)")
            
            # Check context data
            context_data = session.get('context_data', {})
            context_str = json.dumps(context_data, default=str)
            if file_path in context_str:
                impact_score = max(impact_score, 0.5)
                reasons.append("File mentioned in session context")
            
            return {
                'impact_score': impact_score,
                'reasons': reasons
            }
            
        except Exception as e:
            logging.error(f"Failed to calculate session file impact: {e}")
            return {'impact_score': 0.0, 'reasons': [f"Error: {e}"]}
    
    def calculate_memory_implications(self, file_path: str, base_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate implications for memory system."""
        try:
            implications = {
                'relevance_update_required': False,
                'context_compression_triggered': False,
                'cross_session_impact': False,
                'priority_boost_required': False
            }
            
            impact_score = base_analysis.get('impact_score', 0.5)
            
            # High impact changes require relevance updates
            if impact_score > 0.7:
                implications['relevance_update_required'] = True
                implications['priority_boost_required'] = True
            
            # Configuration changes might trigger compression
            if base_analysis.get('analysis_type') == 'configuration':
                implications['context_compression_triggered'] = True
            
            # API or core changes have cross-session impact
            details = base_analysis.get('details', {})
            if 'api_file' in details.get('characteristics', []) or \
               'core_file' in details.get('characteristics', []):
                implications['cross_session_impact'] = True
            
            return implications
            
        except Exception as e:
            logging.error(f"Failed to calculate memory implications: {e}")
            return {'error': str(e)}
    
    def _calculate_overall_impact(self, base_analysis: Dict[str, Any], 
                                context_analysis: Dict[str, Any], 
                                session_impact: Dict[str, Any]) -> float:
        """Calculate overall impact score."""
        try:
            base_score = base_analysis.get('impact_score', 0.5)
            context_relevance = context_analysis.get('relevance_to_current_work', 0.5)
            session_avg_impact = session_impact.get('average_impact', 0.0)
            
            # Weighted combination
            overall = (base_score * 0.5) + (context_relevance * 0.3) + (session_avg_impact * 0.2)
            
            return min(1.0, overall)
            
        except Exception as e:
            logging.error(f"Failed to calculate overall impact: {e}")
            return 0.5
    
    def _generate_recommendations(self, base_analysis: Dict[str, Any], 
                                context_analysis: Dict[str, Any], 
                                session_impact: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        try:
            overall_impact = self._calculate_overall_impact(base_analysis, context_analysis, session_impact)
            
            if overall_impact > 0.8:
                recommendations.append("High impact change detected - consider immediate context refresh")
            
            if context_analysis.get('context_invalidation_required'):
                recommendations.append("Context invalidation required for current session")
            
            if session_impact.get('requires_session_refresh'):
                recommendations.append("Multiple sessions affected - consider broadcast refresh")
            
            if base_analysis.get('details', {}).get('breaking_change_risk'):
                recommendations.append("Breaking change risk - verify dependent systems")
            
            if base_analysis.get('analysis_type') == 'configuration':
                recommendations.append("Configuration change - may require service restart")
            
            affected_count = session_impact.get('total_sessions_affected', 0)
            if affected_count > 3:
                recommendations.append(f"Wide impact: {affected_count} sessions affected")
            
        except Exception as e:
            logging.error(f"Failed to generate recommendations: {e}")
            recommendations.append("Error generating recommendations")
        
        return recommendations
    
    def _store_analysis(self, analysis: Dict[str, Any]):
        """Store analysis results in database."""
        try:
            file_path = analysis['file_path']
            change_type = analysis['change_type']
            overall_impact = analysis['overall_impact']
            
            # Get affected session IDs
            affected_sessions = [
                s['session_id'] for s in analysis.get('session_impact', {}).get('affected_sessions', [])
            ]
            
            # Determine flags
            memory_invalidation = analysis.get('memory_implications', {}).get('relevance_update_required', False)
            context_refresh = analysis.get('context_analysis', {}).get('context_invalidation_required', False)
            
            # Store in database
            self.memory_db.record_change_impact(
                file_path=file_path,
                change_type=change_type,
                impact_score=overall_impact,
                affected_sessions=affected_sessions,
                memory_invalidation=memory_invalidation,
                context_refresh_required=context_refresh,
                analysis_metadata=analysis
            )
            
        except Exception as e:
            logging.error(f"Failed to store analysis: {e}")
    
    def get_file_change_history(self, file_path: str, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent change history for a file."""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            impacts = self.memory_db.get_change_impacts(file_path=file_path)
            recent_impacts = [
                impact for impact in impacts 
                if impact['created_at'] >= cutoff_date
            ]
            
            return recent_impacts
            
        except Exception as e:
            logging.error(f"Failed to get file change history: {e}")
            return []
    
    def get_high_impact_changes(self, hours: int = 24, min_impact: float = 0.7) -> List[Dict[str, Any]]:
        """Get recent high-impact changes."""
        try:
            cutoff_date = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            impacts = self.memory_db.get_change_impacts(min_impact=min_impact)
            recent_high_impacts = [
                impact for impact in impacts 
                if impact['created_at'] >= cutoff_date
            ]
            
            return sorted(recent_high_impacts, key=lambda x: x['impact_score'], reverse=True)
            
        except Exception as e:
            logging.error(f"Failed to get high impact changes: {e}")
            return []
    
    def analyze_change_patterns(self, days: int = 30) -> Dict[str, Any]:
        """Analyze patterns in file changes."""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            impacts = self.memory_db.get_change_impacts()
            recent_impacts = [
                impact for impact in impacts 
                if impact['created_at'] >= cutoff_date
            ]
            
            patterns = {
                'total_changes': len(recent_impacts),
                'high_impact_changes': len([i for i in recent_impacts if i['impact_score'] > 0.7]),
                'most_changed_files': {},
                'change_types': {},
                'average_impact': 0.0
            }
            
            if recent_impacts:
                # Count file changes
                for impact in recent_impacts:
                    file_path = impact['file_path']
                    change_type = impact['change_type']
                    
                    patterns['most_changed_files'][file_path] = \
                        patterns['most_changed_files'].get(file_path, 0) + 1
                    
                    patterns['change_types'][change_type] = \
                        patterns['change_types'].get(change_type, 0) + 1
                
                # Calculate average impact
                patterns['average_impact'] = sum(i['impact_score'] for i in recent_impacts) / len(recent_impacts)
                
                # Get top changed files
                patterns['most_changed_files'] = dict(
                    sorted(patterns['most_changed_files'].items(), 
                          key=lambda x: x[1], reverse=True)[:10]
                )
            
            return patterns
            
        except Exception as e:
            logging.error(f"Failed to analyze change patterns: {e}")
            return {'error': str(e)}