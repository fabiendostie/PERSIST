#!/usr/bin/env python3
"""
Activity Analyzer for Prsist Memory System
Analyzes session tool usage and file interactions to generate meaningful summaries.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging


class ActivityAnalyzer:
    """Analyzes session activity to generate meaningful summaries."""
    
    def __init__(self):
        self.bug_fix_patterns = [
            r'fix.*bug', r'fix.*issue', r'fix.*error', r'fix.*problem',
            r'bug.*fix', r'error.*fix', r'issue.*fix',
            r'correct.*bug', r'resolve.*issue', r'patch.*bug'
        ]
        
        self.feature_patterns = [
            r'add.*feature', r'implement.*feature', r'create.*feature',
            r'new.*feature', r'build.*feature', r'develop.*feature',
            r'add.*function', r'implement.*function', r'create.*function'
        ]
        
        self.refactor_patterns = [
            r'refactor', r'reorganize', r'restructure', r'cleanup',
            r'improve.*structure', r'optimize.*code', r'clean.*up'
        ]
        
        self.config_patterns = [
            r'config', r'settings', r'configuration', r'setup',
            r'install', r'deploy', r'environment'
        ]
        
        self.test_patterns = [
            r'test', r'spec', r'unit.*test', r'integration.*test',
            r'validate', r'verify', r'check'
        ]
        
        self.doc_patterns = [
            r'document', r'readme', r'docs?/', r'\.md$', r'comment',
            r'documentation', r'guide', r'manual'
        ]
    
    def analyze_session_activity(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze session activity and generate meaningful summary."""
        try:
            tool_usage = session_data.get('tool_usage', [])
            file_interactions = session_data.get('file_interactions', [])
            
            # Extract activity patterns
            patterns = self._extract_activity_patterns(tool_usage, file_interactions)
            
            # Analyze files modified
            file_analysis = self._analyze_file_modifications(tool_usage, file_interactions)
            
            # Generate activity summary
            activity_summary = self._generate_activity_summary(patterns, file_analysis)
            
            # Generate human-readable description
            description = self._generate_human_description(activity_summary, patterns, file_analysis)
            
            return {
                'activity_summary': activity_summary,
                'patterns_detected': patterns,
                'file_analysis': file_analysis,
                'human_description': description,
                'confidence_score': self._calculate_confidence(patterns, file_analysis)
            }
            
        except Exception as e:
            logging.error(f"Failed to analyze session activity: {e}")
            return {
                'activity_summary': 'Unknown activity',
                'human_description': 'Session activity could not be analyzed',
                'confidence_score': 0.0
            }
    
    def _extract_activity_patterns(self, tool_usage: List[Dict], file_interactions: List[Dict]) -> Dict[str, Any]:
        """Extract patterns from tool usage and file interactions."""
        patterns = {
            'bug_fixes': [],
            'features': [],
            'refactoring': [],
            'configuration': [],
            'testing': [],
            'documentation': [],
            'file_operations': [],
            'code_exploration': False,
            'command_execution': [],
            'search_queries': [],
            'todo_activities': [],
            'project_context': []
        }
        
        # Analyze tool usage
        for tool in tool_usage:
            tool_name = tool.get('tool_name', '')
            input_data = tool.get('input_data', {})
            output_data = tool.get('output_data', {})
            
            # Extract text content for pattern matching
            text_content = self._extract_text_content(input_data, output_data)
            
            # Check for activity patterns
            if self._matches_patterns(text_content, self.bug_fix_patterns):
                patterns['bug_fixes'].append({
                    'tool': tool_name,
                    'context': text_content[:200],
                    'timestamp': tool.get('timestamp')
                })
            
            if self._matches_patterns(text_content, self.feature_patterns):
                patterns['features'].append({
                    'tool': tool_name,
                    'context': text_content[:200],
                    'timestamp': tool.get('timestamp')
                })
            
            if self._matches_patterns(text_content, self.refactor_patterns):
                patterns['refactoring'].append({
                    'tool': tool_name,
                    'context': text_content[:200],
                    'timestamp': tool.get('timestamp')
                })
            
            # Track specific tool patterns
            if tool_name in ['Read', 'Grep', 'CodebaseSearch']:
                patterns['code_exploration'] = True
            
            if tool_name == 'Bash':
                command = input_data.get('command', '')
                patterns['command_execution'].append(command)
            
            if tool_name in ['Write', 'Edit', 'MultiEdit']:
                file_path = input_data.get('file_path', '')
                patterns['file_operations'].append({
                    'operation': tool_name,
                    'file': file_path,
                    'timestamp': tool.get('timestamp')
                })
            
            # Extract context from WebSearch
            if tool_name == 'WebSearch':
                query = input_data.get('query', '')
                if query:
                    patterns['search_queries'].append(query)
                    # Extract project context from search queries
                    if 'mcp' in query.lower() or 'context7' in query.lower():
                        patterns['project_context'].append('MCP server research and setup')
                    elif 'memory' in query.lower():
                        patterns['project_context'].append('memory system research')
            
            # Extract context from TodoWrite
            if tool_name == 'TodoWrite':
                todos = input_data.get('todos', [])
                for todo in todos:
                    content = todo.get('content', '').lower()
                    patterns['todo_activities'].append(content)
                    # Extract project context from todo items
                    if 'mcp' in content or 'context7' in content:
                        patterns['project_context'].append('MCP server integration tasks')
                    elif 'memory' in content or 'session' in content:
                        patterns['project_context'].append('memory system development')
                    elif 'enhance' in content or 'improve' in content:
                        patterns['project_context'].append('feature enhancement work')
        
        return patterns
    
    def _analyze_file_modifications(self, tool_usage: List[Dict], file_interactions: List[Dict]) -> Dict[str, Any]:
        """Analyze what files were modified and how."""
        analysis = {
            'files_created': [],
            'files_modified': [],
            'files_read': [],
            'file_types': {},
            'key_files': [],
            'modification_scope': 'unknown'
        }
        
        # Track file operations from tool usage
        for tool in tool_usage:
            tool_name = tool.get('tool_name', '')
            input_data = tool.get('input_data', {})
            
            file_path = input_data.get('file_path', '')
            if not file_path:
                continue
                
            file_path = Path(file_path)
            file_ext = file_path.suffix
            
            # Track file types
            if file_ext:
                analysis['file_types'][file_ext] = analysis['file_types'].get(file_ext, 0) + 1
            
            # Categorize operations
            if tool_name == 'Write':
                analysis['files_created'].append(str(file_path))
            elif tool_name in ['Edit', 'MultiEdit']:
                analysis['files_modified'].append(str(file_path))
            elif tool_name == 'Read':
                analysis['files_read'].append(str(file_path))
            
            # Identify key files
            if self._is_key_file(file_path):
                analysis['key_files'].append(str(file_path))
        
        # Determine modification scope
        total_files = len(set(analysis['files_created'] + analysis['files_modified']))
        if total_files == 0:
            analysis['modification_scope'] = 'read_only'
        elif total_files == 1:
            analysis['modification_scope'] = 'single_file'
        elif total_files <= 3:
            analysis['modification_scope'] = 'focused'
        else:
            analysis['modification_scope'] = 'broad'
        
        return analysis
    
    def _generate_activity_summary(self, patterns: Dict, file_analysis: Dict) -> str:
        """Generate a structured activity summary."""
        activities = []
        
        # Primary activities
        if patterns['bug_fixes']:
            activities.append(f"bug fixes ({len(patterns['bug_fixes'])} instances)")
        
        if patterns['features']:
            activities.append(f"feature development ({len(patterns['features'])} instances)")
        
        if patterns['refactoring']:
            activities.append(f"code refactoring ({len(patterns['refactoring'])} instances)")
        
        # File operations
        files_created = len(file_analysis['files_created'])
        files_modified = len(file_analysis['files_modified'])
        
        if files_created > 0:
            activities.append(f"file creation ({files_created} files)")
        
        if files_modified > 0:
            activities.append(f"file editing ({files_modified} files)")
        
        # Exploration
        if patterns['code_exploration'] and not activities:
            activities.append("code exploration")
        
        # Commands
        if patterns['command_execution'] and not activities:
            activities.append(f"command execution ({len(patterns['command_execution'])} commands)")
        
        return ", ".join(activities) if activities else "general development activity"
    
    def _generate_human_description(self, activity_summary: str, patterns: Dict, file_analysis: Dict) -> str:
        """Generate human-readable description."""
        
        # Extract meaningful context from tool usage patterns
        context_clues = self._extract_context_clues(patterns, file_analysis)
        
        # Handle specific patterns with context
        if patterns['bug_fixes']:
            bug_context = patterns['bug_fixes'][0].get('context', '')
            if context_clues['project_focus']:
                return f"Fixed {context_clues['project_focus']} bugs and issues"
            elif 'session tracker' in bug_context.lower():
                return "Fixed session tracker bug and updated file interaction logging"
            elif 'logging' in bug_context.lower():
                return "Fixed logging issues and corrected data tracking"
            else:
                return f"Fixed bugs in {self._get_primary_component(file_analysis)}"
        
        if patterns['features']:
            if context_clues['project_focus']:
                return f"Implemented {context_clues['project_focus']} features and functionality"
            else:
                component = self._get_primary_component(file_analysis)
                return f"Implemented new features in {component}"
        
        # Handle specific project activities based on context clues
        if context_clues['main_activity']:
            if context_clues['files_worked_on']:
                return f"{context_clues['main_activity']} - worked on {', '.join(context_clues['files_worked_on'][:3])}"
            else:
                return context_clues['main_activity']
        
        # Handle file-based activities with specific context
        key_files = file_analysis.get('key_files', [])
        if key_files:
            filenames = [Path(f).name for f in key_files[:2]]
            if context_clues['activity_type']:
                return f"{context_clues['activity_type']} in {', '.join(filenames)}"
            elif any('session' in f.lower() for f in key_files):
                return f"Enhanced session management - modified {', '.join(filenames)}"
            elif any('memory' in f.lower() for f in key_files):
                return f"Improved memory system - updated {', '.join(filenames)}"
            elif any('config' in f.lower() for f in key_files):
                return f"Updated configuration - modified {', '.join(filenames)}"
        
        # Handle scope-based activities with file context
        scope = file_analysis.get('modification_scope', 'unknown')
        if scope == 'single_file':
            file_path = (file_analysis.get('files_modified', []) + 
                        file_analysis.get('files_created', []))
            if file_path:
                filename = Path(file_path[0]).name
                if context_clues['activity_type']:
                    return f"{context_clues['activity_type']} - focused work on {filename}"
                else:
                    return f"Focused development work on {filename}"
        
        elif scope == 'focused':
            files_list = (file_analysis.get('files_modified', []) + 
                         file_analysis.get('files_created', []))
            if files_list:
                filenames = [Path(f).name for f in files_list[:2]]
                if context_clues['activity_type']:
                    return f"{context_clues['activity_type']} across {', '.join(filenames)}"
                
            file_types = file_analysis.get('file_types', {})
            if '.py' in file_types:
                return "Python development and code improvements"
            elif '.md' in file_types:
                return "Documentation updates and improvements"
        
        # Command execution activities with context
        commands = patterns.get('command_execution', [])
        if commands and context_clues['command_focus']:
            return context_clues['command_focus']
        elif commands:
            if any('mem' in cmd for cmd in commands):
                return "Memory system testing and CLI operations"
            elif any('python' in cmd for cmd in commands):
                return "Python script execution and testing"
            elif any('mcp' in cmd.lower() for cmd in commands):
                return "MCP server configuration and setup"
        
        # Exploration activities with context
        if patterns.get('code_exploration'):
            if context_clues['exploration_focus']:
                return context_clues['exploration_focus']
            elif not file_analysis.get('files_modified'):
                return "Code exploration and analysis"
        
        # Default with context if available
        if context_clues['project_focus']:
            return f"Development work on {context_clues['project_focus']}"
        
        return f"Development session - {activity_summary}"
    
    def _extract_context_clues(self, patterns: Dict, file_analysis: Dict) -> Dict[str, Any]:
        """Extract meaningful context clues from session patterns and file analysis."""
        context = {
            'main_activity': None,
            'project_focus': None,
            'activity_type': None,
            'files_worked_on': [],
            'command_focus': None,
            'exploration_focus': None,
            'technologies': []
        }
        
        # Analyze file operations for context
        files_modified = file_analysis.get('files_modified', [])
        files_created = file_analysis.get('files_created', [])
        all_files = files_modified + files_created
        
        if all_files:
            context['files_worked_on'] = [Path(f).name for f in all_files]
            
            # Extract technologies and frameworks from file extensions and names
            for file_path in all_files:
                file_path_lower = file_path.lower()
                filename = Path(file_path).name.lower()
                
                # Technology detection
                if file_path.endswith('.py'):
                    context['technologies'].append('Python')
                elif file_path.endswith(('.js', '.ts', '.jsx', '.tsx')):
                    context['technologies'].append('JavaScript/TypeScript')
                elif file_path.endswith(('.md', '.rst')):
                    context['technologies'].append('Documentation')
                elif file_path.endswith('.json'):
                    context['technologies'].append('Configuration')
                
                # Project-specific patterns
                if 'mcp' in filename or 'context7' in filename:
                    context['project_focus'] = 'MCP server integration'
                elif 'memory' in filename or 'session' in filename:
                    context['project_focus'] = 'memory system'
                elif 'activity' in filename or 'analyzer' in filename:
                    context['project_focus'] = 'session analysis'
                elif 'prsist' in filename:
                    context['project_focus'] = 'Prsist memory system'
        
        # Analyze tool usage for activity context
        file_ops = patterns.get('file_operations', [])
        if file_ops:
            if any(op['operation'] == 'Write' for op in file_ops):
                if any(op['operation'] == 'Edit' for op in file_ops):
                    context['activity_type'] = 'Created and refined files'
                else:
                    context['activity_type'] = 'Created new files'
            elif any(op['operation'] in ['Edit', 'MultiEdit'] for op in file_ops):
                context['activity_type'] = 'Enhanced existing code'
        
        # Analyze commands for specific activities
        commands = patterns.get('command_execution', [])
        command_text = ' '.join(commands).lower()
        
        if 'mcp' in command_text and 'add' in command_text:
            context['command_focus'] = 'MCP server installation and configuration'
            context['main_activity'] = 'Set up MCP server integration'
            context['project_focus'] = 'MCP server integration'
        elif 'mem' in command_text and 'recent' in command_text:
            context['command_focus'] = 'Memory system testing and analysis'
            context['main_activity'] = 'Tested memory system functionality'
            context['project_focus'] = 'memory system'
        elif 'python' in command_text and 'prsist' in command_text:
            context['command_focus'] = 'Memory CLI testing and validation'
            context['main_activity'] = 'Validated memory system CLI commands'
            context['project_focus'] = 'Prsist memory system'
        elif 'npm' in command_text or 'npx' in command_text:
            context['command_focus'] = 'Package installation and setup'
            context['main_activity'] = 'Installed and configured development tools'
        
        # Use project context from patterns
        project_contexts = patterns.get('project_context', [])
        if project_contexts:
            # Use the most recent/frequent project context
            context['project_focus'] = project_contexts[-1]  # Most recent
        
        # Enhance activity description with search queries and todos
        search_queries = patterns.get('search_queries', [])
        todo_activities = patterns.get('todo_activities', [])
        
        if search_queries and not context['main_activity']:
            query = search_queries[0].lower()
            if 'mcp' in query or 'context7' in query:
                context['main_activity'] = 'Researched and set up Context7 MCP server integration'
            elif 'memory' in query:
                context['main_activity'] = 'Researched memory system implementation'
        
        if todo_activities and not context['main_activity']:
            # Combine todo activities for description
            if len(todo_activities) > 1:
                context['main_activity'] = f"Worked on multiple tasks: {', '.join(todo_activities[:2])}"
            else:
                context['main_activity'] = f"Worked on: {todo_activities[0]}"
        
        # Analyze search patterns for exploration context
        if patterns.get('code_exploration'):
            files_read = file_analysis.get('files_read', [])
            if files_read:
                read_contexts = []
                for file_path in files_read[:3]:
                    filename = Path(file_path).name
                    if 'config' in filename.lower():
                        read_contexts.append('configuration files')
                    elif 'memory' in filename.lower():
                        read_contexts.append('memory system')
                    elif 'session' in filename.lower():
                        read_contexts.append('session tracking')
                    else:
                        read_contexts.append(filename)
                
                if read_contexts:
                    context['exploration_focus'] = f"Explored {', '.join(read_contexts[:2])}"
        
        # Set main activity based on strongest patterns
        if not context['main_activity']:
            if context['project_focus'] and context['activity_type']:
                context['main_activity'] = f"{context['activity_type']} for {context['project_focus']}"
            elif patterns.get('features'):
                context['main_activity'] = f"Feature development in {context['project_focus'] or 'project'}"
            elif patterns.get('bug_fixes'):
                context['main_activity'] = f"Bug fixes in {context['project_focus'] or 'codebase'}"
            elif len(all_files) == 1:
                filename = Path(all_files[0]).name
                context['main_activity'] = f"Focused work on {filename}"
            elif len(all_files) > 1:
                context['main_activity'] = f"Multi-file development across {len(all_files)} files"
        
        return context
    
    def _get_primary_component(self, file_analysis: Dict) -> str:
        """Determine the primary component being worked on."""
        files = file_analysis.get('files_modified', []) + file_analysis.get('files_created', [])
        
        if not files:
            return "system components"
        
        # Check for common patterns
        for file_path in files:
            path_lower = file_path.lower()
            if 'session' in path_lower:
                return "session management"
            elif 'memory' in path_lower:
                return "memory system"
            elif 'database' in path_lower:
                return "database layer"
            elif 'config' in path_lower:
                return "configuration"
            elif 'hook' in path_lower:
                return "integration hooks"
        
        # Fallback to file type
        file_types = file_analysis.get('file_types', {})
        if '.py' in file_types:
            return "Python modules"
        elif '.md' in file_types:
            return "documentation"
        elif '.yaml' in file_types or '.json' in file_types:
            return "configuration files"
        
        return "project files"
    
    def _extract_text_content(self, input_data: Any, output_data: Any) -> str:
        """Extract searchable text content from tool data."""
        content = []
        
        # Handle input data safely
        try:
            if isinstance(input_data, dict):
                for key, value in input_data.items():
                    if isinstance(value, str):
                        content.append(value)
                    elif isinstance(value, list) and value:
                        content.extend([str(item) for item in value if isinstance(item, str)])
            elif isinstance(input_data, str):
                content.append(input_data)
            elif input_data is not None:
                content.append(str(input_data))
        except Exception:
            pass  # Skip problematic input data
        
        # Handle output data safely
        try:
            if isinstance(output_data, str):
                content.append(output_data[:500])  # Limit output data
            elif isinstance(output_data, dict):
                if 'stdout' in output_data:
                    content.append(str(output_data['stdout'])[:500])
                elif 'content' in output_data:
                    content.append(str(output_data['content'])[:500])
            elif output_data is not None:
                content.append(str(output_data)[:500])
        except Exception:
            pass  # Skip problematic output data
        
        return " ".join(content).lower() if content else ""
    
    def _matches_patterns(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any of the given patterns."""
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _is_key_file(self, file_path: Path) -> bool:
        """Determine if a file is a key project file."""
        key_patterns = [
            r'config', r'settings', r'main', r'app', r'index',
            r'manager', r'tracker', r'database', r'core'
        ]
        
        file_name = file_path.name.lower()
        return any(re.search(pattern, file_name) for pattern in key_patterns)
    
    def _calculate_confidence(self, patterns: Dict, file_analysis: Dict) -> float:
        """Calculate confidence score for the analysis."""
        confidence = 0.5  # Base confidence
        
        # Higher confidence if we detected specific patterns
        if patterns['bug_fixes'] or patterns['features'] or patterns['refactoring']:
            confidence += 0.3
        
        # Higher confidence if we have file modifications
        if file_analysis['files_modified'] or file_analysis['files_created']:
            confidence += 0.2
        
        # Higher confidence if we identified key files
        if file_analysis['key_files']:
            confidence += 0.1
        
        return min(confidence, 1.0)
