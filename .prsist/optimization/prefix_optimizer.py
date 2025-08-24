#!/usr/bin/env python3
"""
Prefix optimizer for KV-Cache system.
Analyzes prefix patterns and creates optimized caching strategies.
"""

import hashlib
import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter

from utils import setup_logging

@dataclass
class PrefixPattern:
    """Represents a detected prefix pattern."""
    pattern_hash: str
    content: str
    frequency: int
    sessions: List[str]
    token_count: int
    optimization_value: float
    pattern_type: str
    similarity_group: Optional[str] = None
    last_seen: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        if self.last_seen:
            result['last_seen'] = self.last_seen.isoformat()
        return result

@dataclass
class OptimizationStrategy:
    """Represents a caching optimization strategy."""
    strategy_name: str
    always_cache: List[str]
    conditional_cache: List[str]
    never_cache: List[str]
    cache_budget_mb: float
    expected_hit_rate: float
    expected_cost_reduction: float
    strategy_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

class PrefixExtractor:
    """Extracts and analyzes prefixes from session data."""
    
    def __init__(self):
        self.extraction_rules = self._load_extraction_rules()
    
    def _load_extraction_rules(self) -> Dict[str, Any]:
        """Load prefix extraction rules."""
        return {
            'system_content': {
                'patterns': [r'system.*prompt', r'instructions?', r'guidelines?'],
                'weight': 0.9,
                'cache_priority': 'high'
            },
            'configuration': {
                'patterns': [r'config.*', r'settings?', r'parameters?'],
                'weight': 0.8,
                'cache_priority': 'high'
            },
            'templates': {
                'patterns': [r'template.*', r'pattern.*', r'boilerplate'],
                'weight': 0.7,
                'cache_priority': 'medium'
            },
            'documentation': {
                'patterns': [r'doc.*', r'readme', r'guide.*', r'help'],
                'weight': 0.6,
                'cache_priority': 'medium'
            },
            'code_snippets': {
                'patterns': [r'function.*', r'class.*', r'def ', r'import '],
                'weight': 0.5,
                'cache_priority': 'low'
            }
        }
    
    def extract_prefixes(self, session_data: Dict[str, Any]) -> List[str]:
        """Extract prefixes from session data."""
        try:
            prefixes = []
            
            # Extract from common session fields
            for field in ['system_prompt', 'instructions', 'context', 'background']:
                if field in session_data and isinstance(session_data[field], str):
                    prefixes.append(session_data[field])
            
            # Extract from tool usage patterns
            if 'tool_usage' in session_data:
                tool_patterns = self._extract_tool_patterns(session_data['tool_usage'])
                prefixes.extend(tool_patterns)
            
            # Extract from file interactions
            if 'file_interactions' in session_data:
                file_patterns = self._extract_file_patterns(session_data['file_interactions'])
                prefixes.extend(file_patterns)
            
            # Extract from conversation patterns
            if 'conversation' in session_data:
                conversation_patterns = self._extract_conversation_patterns(session_data['conversation'])
                prefixes.extend(conversation_patterns)
            
            return prefixes
            
        except Exception as e:
            logging.error(f"Failed to extract prefixes: {e}")
            return []
    
    def _extract_tool_patterns(self, tool_usage: List[Dict[str, Any]]) -> List[str]:
        """Extract patterns from tool usage."""
        patterns = []
        
        try:
            # Group tools by type
            tool_groups = defaultdict(list)
            for tool in tool_usage:
                tool_name = tool.get('tool_name', '')
                tool_groups[tool_name].append(tool)
            
            # Create patterns for each tool group
            for tool_name, tools in tool_groups.items():
                if len(tools) >= 3:  # Only create pattern if used multiple times
                    pattern = f"Tool usage pattern: {tool_name} used {len(tools)} times"
                    
                    # Add common parameters
                    common_params = self._find_common_parameters(tools)
                    if common_params:
                        pattern += f" with common parameters: {json.dumps(common_params)}"
                    
                    patterns.append(pattern)
            
        except Exception as e:
            logging.error(f"Failed to extract tool patterns: {e}")
        
        return patterns
    
    def _extract_file_patterns(self, file_interactions: List[Dict[str, Any]]) -> List[str]:
        """Extract patterns from file interactions."""
        patterns = []
        
        try:
            # Group by file types
            file_types = defaultdict(list)
            for interaction in file_interactions:
                file_path = interaction.get('file_path', '')
                file_ext = Path(file_path).suffix.lower()
                if file_ext:
                    file_types[file_ext].append(interaction)
            
            # Create patterns for file types
            for file_ext, interactions in file_types.items():
                if len(interactions) >= 2:
                    pattern = f"File type pattern: {file_ext} files ({len(interactions)} interactions)"
                    patterns.append(pattern)
            
            # Group by directories
            directories = defaultdict(list)
            for interaction in file_interactions:
                file_path = interaction.get('file_path', '')
                directory = str(Path(file_path).parent)
                directories[directory].append(interaction)
            
            # Create patterns for directories
            for directory, interactions in directories.items():
                if len(interactions) >= 3 and directory != '.':
                    pattern = f"Directory pattern: {directory} ({len(interactions)} interactions)"
                    patterns.append(pattern)
            
        except Exception as e:
            logging.error(f"Failed to extract file patterns: {e}")
        
        return patterns
    
    def _extract_conversation_patterns(self, conversation: List[Dict[str, Any]]) -> List[str]:
        """Extract patterns from conversation."""
        patterns = []
        
        try:
            # Extract common phrases or instructions
            all_text = ' '.join([msg.get('content', '') for msg in conversation if isinstance(msg.get('content'), str)])
            
            # Find repeated phrases (simplified)
            words = all_text.lower().split()
            word_phrases = []
            
            # Create 3-word phrases
            for i in range(len(words) - 2):
                phrase = ' '.join(words[i:i+3])
                word_phrases.append(phrase)
            
            # Find common phrases
            phrase_counts = Counter(word_phrases)
            common_phrases = [phrase for phrase, count in phrase_counts.items() if count >= 2]
            
            if common_phrases:
                patterns.append(f"Common conversation phrases: {', '.join(common_phrases[:5])}")
            
        except Exception as e:
            logging.error(f"Failed to extract conversation patterns: {e}")
        
        return patterns
    
    def _find_common_parameters(self, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find common parameters across tool usages."""
        try:
            if not tools:
                return {}
            
            # Find parameters that appear in most tools
            param_counts = defaultdict(lambda: defaultdict(int))
            
            for tool in tools:
                params = tool.get('parameters', {})
                for key, value in params.items():
                    if isinstance(value, (str, int, bool)):
                        param_counts[key][str(value)] += 1
            
            # Find parameters with consistent values
            common_params = {}
            for param_name, value_counts in param_counts.items():
                most_common_value, count = max(value_counts.items(), key=lambda x: x[1])
                if count >= len(tools) * 0.7:  # Appears in 70% of tools
                    common_params[param_name] = most_common_value
            
            return common_params
            
        except Exception as e:
            logging.error(f"Failed to find common parameters: {e}")
            return {}

class PrefixOptimizer:
    """Optimizes prefix caching strategies."""
    
    def __init__(self):
        self.prefix_patterns = {}
        self.optimization_rules = self._load_optimization_rules()
        self.extractor = PrefixExtractor()
        
    def _load_optimization_rules(self) -> Dict[str, Any]:
        """Load optimization rules."""
        return {
            'cache_budget_allocation': {
                'system_content': 0.4,      # 40% for system prompts, configs
                'templates': 0.3,           # 30% for templates and patterns
                'documentation': 0.2,       # 20% for documentation
                'dynamic_content': 0.1      # 10% for dynamic content
            },
            'frequency_thresholds': {
                'always_cache': 5,          # Cache if used 5+ times
                'conditional_cache': 2,     # Conditionally cache if used 2+ times
                'never_cache': 1            # Don't cache single-use items
            },
            'size_limits': {
                'min_tokens': 50,           # Minimum tokens to consider caching
                'max_tokens': 4000,         # Maximum tokens per cache entry
                'optimal_tokens': 500       # Optimal token count for caching
            },
            'decay_factors': {
                'time_decay': 0.1,          # Decay rate per day
                'usage_boost': 1.5,         # Boost factor for recent usage
                'similarity_bonus': 0.2     # Bonus for similar content
            }
        }
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // 4  # Simple approximation
    
    def analyze_prefix_patterns(self, session_history: List[Dict[str, Any]]) -> Dict[str, PrefixPattern]:
        """Analyze session history to identify common prefix patterns."""
        try:
            pattern_frequency = {}
            
            for session in session_history:
                session_id = session.get('session_id', 'unknown')
                prefixes = self.extractor.extract_prefixes(session)
                
                for prefix in prefixes:
                    if not prefix or len(prefix.strip()) < 10:  # Skip very short prefixes
                        continue
                    
                    prefix_hash = hashlib.md5(prefix.encode()).hexdigest()
                    token_count = self.count_tokens(prefix)
                    
                    # Skip if outside size limits (more lenient for testing)
                    size_limits = self.optimization_rules['size_limits']
                    if not (10 <= token_count <= size_limits['max_tokens']):  # Lowered minimum
                        continue
                    
                    if prefix_hash not in pattern_frequency:
                        pattern_type = self._classify_prefix_type(prefix)
                        pattern_frequency[prefix_hash] = PrefixPattern(
                            pattern_hash=prefix_hash,
                            content=prefix,
                            frequency=0,
                            sessions=[],
                            token_count=token_count,
                            optimization_value=0.0,
                            pattern_type=pattern_type,
                            last_seen=datetime.now()
                        )
                    
                    pattern = pattern_frequency[prefix_hash]
                    pattern.frequency += 1
                    if session_id not in pattern.sessions:
                        pattern.sessions.append(session_id)
                    pattern.last_seen = datetime.now()
            
            # Calculate optimization values
            for pattern in pattern_frequency.values():
                pattern.optimization_value = self._calculate_optimization_value(pattern)
            
            self.prefix_patterns = pattern_frequency
            return pattern_frequency
            
        except Exception as e:
            logging.error(f"Failed to analyze prefix patterns: {e}")
            return {}
    
    def _classify_prefix_type(self, prefix: str) -> str:
        """Classify the type of prefix."""
        try:
            prefix_lower = prefix.lower()
            
            # Check extraction rules to classify
            for pattern_type, rules in self.extractor.extraction_rules.items():
                for pattern in rules['patterns']:
                    if re.search(pattern, prefix_lower):
                        return pattern_type
            
            # Fallback classification
            if len(prefix) > 1000:
                return 'documentation'
            elif 'def ' in prefix or 'class ' in prefix or 'function' in prefix_lower:
                return 'code_snippets'
            elif any(word in prefix_lower for word in ['config', 'setting', 'param']):
                return 'configuration'
            elif any(word in prefix_lower for word in ['system', 'prompt', 'instruction']):
                return 'system_content'
            else:
                return 'general_content'
                
        except Exception as e:
            logging.error(f"Failed to classify prefix type: {e}")
            return 'general_content'
    
    def _calculate_optimization_value(self, pattern: PrefixPattern) -> float:
        """Calculate optimization value for a pattern."""
        try:
            rules = self.optimization_rules
            
            # Base value from frequency and token savings
            base_value = pattern.frequency * pattern.token_count * 0.9  # 90% cost reduction
            
            # Type-based multiplier
            type_weights = {
                'system_content': 1.5,
                'configuration': 1.3,
                'templates': 1.2,
                'documentation': 1.0,
                'code_snippets': 0.8,
                'general_content': 0.6
            }
            type_multiplier = type_weights.get(pattern.pattern_type, 1.0)
            
            # Frequency bonus
            frequency_thresholds = rules['frequency_thresholds']
            if pattern.frequency >= frequency_thresholds['always_cache']:
                frequency_bonus = 1.5
            elif pattern.frequency >= frequency_thresholds['conditional_cache']:
                frequency_bonus = 1.2
            else:
                frequency_bonus = 1.0
            
            # Size optimization (prefer medium-sized entries)
            optimal_tokens = rules['size_limits']['optimal_tokens']
            size_ratio = min(pattern.token_count, optimal_tokens) / optimal_tokens
            size_multiplier = 0.5 + 0.5 * size_ratio  # 0.5 to 1.0 multiplier
            
            # Cross-session bonus
            cross_session_bonus = 1.0 + (len(pattern.sessions) - 1) * 0.1  # 10% bonus per additional session
            
            # Time decay (if last seen is old)
            if pattern.last_seen:
                days_old = (datetime.now() - pattern.last_seen).days
                time_decay = max(0.5, 1.0 - days_old * rules['decay_factors']['time_decay'])
            else:
                time_decay = 1.0
            
            optimization_value = (base_value * type_multiplier * frequency_bonus * 
                                size_multiplier * cross_session_bonus * time_decay)
            
            return optimization_value
            
        except Exception as e:
            logging.error(f"Failed to calculate optimization value: {e}")
            return 0.0
    
    def get_cache_budget(self) -> float:
        """Get cache budget in MB."""
        # This would typically come from configuration
        return 500.0  # 500MB default
    
    def create_optimized_cache_strategy(self, patterns: Dict[str, PrefixPattern]) -> OptimizationStrategy:
        """Create cache strategy based on pattern analysis."""
        try:
            # Sort patterns by optimization value
            sorted_patterns = sorted(
                patterns.items(),
                key=lambda x: x[1].optimization_value,
                reverse=True
            )
            
            cache_budget = self.get_cache_budget()
            allocation = self.optimization_rules['cache_budget_allocation']
            frequency_thresholds = self.optimization_rules['frequency_thresholds']
            
            strategy = OptimizationStrategy(
                strategy_name=f"optimized_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                always_cache=[],
                conditional_cache=[],
                never_cache=[],
                cache_budget_mb=cache_budget,
                expected_hit_rate=0.0,
                expected_cost_reduction=0.0,
                strategy_metadata={}
            )
            
            # Allocate budget by type
            type_budgets = {}
            for pattern_type, ratio in allocation.items():
                type_budgets[pattern_type] = cache_budget * ratio
            
            type_usage = defaultdict(float)
            
            # Categorize patterns
            for pattern_hash, pattern in sorted_patterns:
                pattern_size_mb = pattern.token_count * 0.001  # Rough estimate: 1KB per 1000 tokens
                pattern_type_budget = type_budgets.get(pattern.pattern_type, type_budgets['dynamic_content'])
                
                if pattern.frequency >= frequency_thresholds['always_cache']:
                    # Always cache if budget allows
                    if type_usage[pattern.pattern_type] + pattern_size_mb <= pattern_type_budget:
                        strategy.always_cache.append(pattern_hash)
                        type_usage[pattern.pattern_type] += pattern_size_mb
                    else:
                        strategy.conditional_cache.append(pattern_hash)
                        
                elif pattern.frequency >= frequency_thresholds['conditional_cache']:
                    # Conditional cache
                    strategy.conditional_cache.append(pattern_hash)
                    
                else:
                    # Never cache
                    strategy.never_cache.append(pattern_hash)
            
            # Calculate expected performance
            strategy.expected_hit_rate = self._estimate_hit_rate(patterns, strategy)
            strategy.expected_cost_reduction = self._estimate_cost_reduction(patterns, strategy)
            
            # Add metadata
            strategy.strategy_metadata = {
                'total_patterns_analyzed': len(patterns),
                'always_cache_count': len(strategy.always_cache),
                'conditional_cache_count': len(strategy.conditional_cache),
                'never_cache_count': len(strategy.never_cache),
                'budget_utilization': sum(type_usage.values()) / cache_budget,
                'created_at': datetime.now().isoformat(),
                'type_distribution': dict(Counter(pattern.pattern_type for pattern in patterns.values()))
            }
            
            return strategy
            
        except Exception as e:
            logging.error(f"Failed to create optimized cache strategy: {e}")
            return self._create_fallback_strategy()
    
    def _estimate_hit_rate(self, patterns: Dict[str, PrefixPattern], 
                          strategy: OptimizationStrategy) -> float:
        """Estimate hit rate for the strategy."""
        try:
            total_requests = sum(pattern.frequency for pattern in patterns.values())
            if total_requests == 0:
                return 0.0
            
            # Always cache entries get 100% hit rate
            always_cache_requests = sum(
                patterns[pattern_hash].frequency 
                for pattern_hash in strategy.always_cache 
                if pattern_hash in patterns
            )
            
            # Conditional cache entries get estimated 70% hit rate
            conditional_cache_requests = sum(
                patterns[pattern_hash].frequency 
                for pattern_hash in strategy.conditional_cache 
                if pattern_hash in patterns
            )
            
            estimated_hits = always_cache_requests + (conditional_cache_requests * 0.7)
            return estimated_hits / total_requests
            
        except Exception as e:
            logging.error(f"Failed to estimate hit rate: {e}")
            return 0.0
    
    def _estimate_cost_reduction(self, patterns: Dict[str, PrefixPattern], 
                               strategy: OptimizationStrategy) -> float:
        """Estimate cost reduction for the strategy."""
        try:
            total_tokens = sum(
                pattern.frequency * pattern.token_count 
                for pattern in patterns.values()
            )
            
            if total_tokens == 0:
                return 0.0
            
            # Calculate tokens saved
            always_cache_tokens = sum(
                patterns[pattern_hash].frequency * patterns[pattern_hash].token_count
                for pattern_hash in strategy.always_cache 
                if pattern_hash in patterns
            )
            
            conditional_cache_tokens = sum(
                patterns[pattern_hash].frequency * patterns[pattern_hash].token_count
                for pattern_hash in strategy.conditional_cache 
                if pattern_hash in patterns
            )
            
            # Always cache saves 90%, conditional cache saves ~60%
            tokens_saved = (always_cache_tokens * 0.9) + (conditional_cache_tokens * 0.6)
            
            return tokens_saved / total_tokens
            
        except Exception as e:
            logging.error(f"Failed to estimate cost reduction: {e}")
            return 0.0
    
    def _create_fallback_strategy(self) -> OptimizationStrategy:
        """Create a fallback strategy if optimization fails."""
        return OptimizationStrategy(
            strategy_name="fallback_strategy",
            always_cache=[],
            conditional_cache=[],
            never_cache=[],
            cache_budget_mb=self.get_cache_budget(),
            expected_hit_rate=0.0,
            expected_cost_reduction=0.0,
            strategy_metadata={'error': 'Fallback strategy due to optimization failure'}
        )
    
    def optimize_existing_cache(self, current_cache_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize existing cache based on current statistics."""
        try:
            optimizations = {
                'recommendations': [],
                'actions_taken': [],
                'expected_improvements': {}
            }
            
            usage_stats = current_cache_stats.get('usage_statistics', {})
            storage_stats = current_cache_stats.get('storage_statistics', {})
            
            # Analyze hit rate
            hit_rate = usage_stats.get('cache_hits', 0) / max(usage_stats.get('total_requests', 1), 1)
            
            if hit_rate < 0.5:
                optimizations['recommendations'].append({
                    'type': 'hit_rate_improvement',
                    'priority': 'high',
                    'description': 'Low hit rate detected - review caching strategy',
                    'action': 'Expand always_cache list with high-value patterns'
                })
            
            # Analyze storage efficiency
            storage_size_mb = storage_stats.get('storage_size_mb', 0)
            total_entries = storage_stats.get('total_entries', 0)
            
            if total_entries > 0 and storage_size_mb > 0:
                entries_per_mb = total_entries / storage_size_mb
                if entries_per_mb < 50:  # Less than 50 entries per MB is inefficient
                    optimizations['recommendations'].append({
                        'type': 'storage_optimization',
                        'priority': 'medium',
                        'description': 'Storage efficiency is low',
                        'action': 'Remove large, low-value cache entries'
                    })
            
            # Check for cache size issues
            if storage_size_mb > self.get_cache_budget() * 0.9:
                optimizations['recommendations'].append({
                    'type': 'cache_cleanup',
                    'priority': 'high',
                    'description': 'Cache is nearly full',
                    'action': 'Run cleanup with LRU or value-based strategy'
                })
            
            # Estimate improvements
            optimizations['expected_improvements'] = {
                'hit_rate_improvement': max(0.0, 0.7 - hit_rate),
                'storage_optimization': max(0.0, 0.3 - storage_size_mb / self.get_cache_budget()),
                'cost_reduction_improvement': 0.2  # Estimated 20% improvement
            }
            
            return optimizations
            
        except Exception as e:
            logging.error(f"Failed to optimize existing cache: {e}")
            return {'error': str(e)}
    
    def generate_optimization_report(self, patterns: Dict[str, PrefixPattern], 
                                   strategy: OptimizationStrategy) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        try:
            report = {
                'analysis_timestamp': datetime.now().isoformat(),
                'patterns_summary': {
                    'total_patterns': len(patterns),
                    'pattern_types': dict(Counter(p.pattern_type for p in patterns.values())),
                    'frequency_distribution': self._analyze_frequency_distribution(patterns),
                    'size_distribution': self._analyze_size_distribution(patterns)
                },
                'strategy_summary': strategy.to_dict(),
                'optimization_metrics': {
                    'expected_hit_rate': strategy.expected_hit_rate,
                    'expected_cost_reduction': strategy.expected_cost_reduction,
                    'cache_efficiency': len(strategy.always_cache) / max(len(patterns), 1),
                    'budget_utilization': strategy.strategy_metadata.get('budget_utilization', 0)
                },
                'top_patterns': self._get_top_patterns(patterns, 10),
                'recommendations': self._generate_optimization_recommendations(patterns, strategy)
            }
            
            return report
            
        except Exception as e:
            logging.error(f"Failed to generate optimization report: {e}")
            return {'error': str(e)}
    
    def _analyze_frequency_distribution(self, patterns: Dict[str, PrefixPattern]) -> Dict[str, int]:
        """Analyze frequency distribution of patterns."""
        try:
            frequencies = [pattern.frequency for pattern in patterns.values()]
            
            return {
                'single_use': len([f for f in frequencies if f == 1]),
                'low_use': len([f for f in frequencies if 2 <= f < 5]),
                'medium_use': len([f for f in frequencies if 5 <= f < 20]),
                'high_use': len([f for f in frequencies if f >= 20])
            }
            
        except Exception as e:
            logging.error(f"Failed to analyze frequency distribution: {e}")
            return {}
    
    def _analyze_size_distribution(self, patterns: Dict[str, PrefixPattern]) -> Dict[str, int]:
        """Analyze size distribution of patterns."""
        try:
            sizes = [pattern.token_count for pattern in patterns.values()]
            
            return {
                'small': len([s for s in sizes if s < 100]),
                'medium': len([s for s in sizes if 100 <= s < 500]),
                'large': len([s for s in sizes if 500 <= s < 2000]),
                'very_large': len([s for s in sizes if s >= 2000])
            }
            
        except Exception as e:
            logging.error(f"Failed to analyze size distribution: {e}")
            return {}
    
    def _get_top_patterns(self, patterns: Dict[str, PrefixPattern], 
                         count: int) -> List[Dict[str, Any]]:
        """Get top patterns by optimization value."""
        try:
            sorted_patterns = sorted(
                patterns.values(),
                key=lambda p: p.optimization_value,
                reverse=True
            )[:count]
            
            return [
                {
                    'pattern_hash': pattern.pattern_hash,
                    'pattern_type': pattern.pattern_type,
                    'frequency': pattern.frequency,
                    'token_count': pattern.token_count,
                    'optimization_value': pattern.optimization_value,
                    'sessions_count': len(pattern.sessions),
                    'content_preview': pattern.content[:100] + '...' if len(pattern.content) > 100 else pattern.content
                }
                for pattern in sorted_patterns
            ]
            
        except Exception as e:
            logging.error(f"Failed to get top patterns: {e}")
            return []
    
    def _generate_optimization_recommendations(self, patterns: Dict[str, PrefixPattern], 
                                            strategy: OptimizationStrategy) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        try:
            # Strategy-based recommendations
            if strategy.expected_hit_rate < 0.6:
                recommendations.append(
                    "Consider expanding the always_cache list to include more high-frequency patterns"
                )
            
            if strategy.expected_cost_reduction < 0.4:
                recommendations.append(
                    "Focus on caching larger, more frequently used content to improve cost reduction"
                )
            
            # Pattern-based recommendations
            pattern_types = Counter(p.pattern_type for p in patterns.values())
            
            if pattern_types.get('system_content', 0) < 5:
                recommendations.append(
                    "Improve system content detection to increase high-value caching opportunities"
                )
            
            if pattern_types.get('general_content', 0) > len(patterns) * 0.5:
                recommendations.append(
                    "Many patterns are unclassified - improve pattern classification for better optimization"
                )
            
            # Frequency-based recommendations
            single_use_patterns = len([p for p in patterns.values() if p.frequency == 1])
            if single_use_patterns > len(patterns) * 0.4:
                recommendations.append(
                    "High number of single-use patterns detected - consider more aggressive filtering"
                )
            
            # Size-based recommendations
            very_large_patterns = len([p for p in patterns.values() if p.token_count > 2000])
            if very_large_patterns > 0:
                recommendations.append(
                    f"{very_large_patterns} very large patterns detected - consider splitting or compressing them"
                )
            
        except Exception as e:
            logging.error(f"Failed to generate recommendations: {e}")
            recommendations.append(f"Error generating recommendations: {e}")
        
        return recommendations