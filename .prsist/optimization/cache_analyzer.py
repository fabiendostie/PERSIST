#!/usr/bin/env python3
"""
Cache analyzer for KV-Cache optimization system.
Analyzes cache performance, patterns, and provides optimization recommendations.
"""

import json
import logging
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter

import numpy as np

from utils import setup_logging

@dataclass
class CacheAnalysis:
    """Cache analysis results."""
    analysis_timestamp: datetime
    cache_efficiency: Dict[str, float]
    usage_patterns: Dict[str, Any]
    performance_metrics: Dict[str, float]
    optimization_opportunities: List[Dict[str, Any]]
    recommendations: List[str]
    health_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['analysis_timestamp'] = self.analysis_timestamp.isoformat()
        return result

@dataclass
class CachePattern:
    """Represents a cache usage pattern."""
    pattern_type: str
    frequency: int
    average_tokens: float
    hit_rate: float
    cost_impact: float
    recommendation: str

class CacheAnalyzer:
    """Analyzes cache performance and usage patterns."""
    
    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
        self.analysis_history = []
        self.pattern_detector = CachePatternDetector()
        
    def analyze_cache_performance(self) -> CacheAnalysis:
        """Perform comprehensive cache performance analysis."""
        try:
            analysis_timestamp = datetime.now()
            
            # Get cache statistics
            cache_stats = self.cache_manager.get_cache_statistics()
            
            # Analyze efficiency
            efficiency = self._analyze_efficiency(cache_stats)
            
            # Detect usage patterns
            patterns = self._analyze_usage_patterns()
            
            # Calculate performance metrics
            performance = self._calculate_performance_metrics(cache_stats)
            
            # Identify optimization opportunities
            opportunities = self._identify_optimization_opportunities(cache_stats, patterns)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(efficiency, patterns, opportunities)
            
            # Calculate health score
            health_score = self._calculate_health_score(efficiency, performance)
            
            analysis = CacheAnalysis(
                analysis_timestamp=analysis_timestamp,
                cache_efficiency=efficiency,
                usage_patterns=patterns,
                performance_metrics=performance,
                optimization_opportunities=opportunities,
                recommendations=recommendations,
                health_score=health_score
            )
            
            # Store analysis for trending
            self.analysis_history.append(analysis)
            if len(self.analysis_history) > 100:  # Keep last 100 analyses
                self.analysis_history.pop(0)
            
            return analysis
            
        except Exception as e:
            logging.error(f"Failed to analyze cache performance: {e}")
            return self._create_error_analysis(str(e))
    
    def _analyze_efficiency(self, cache_stats: Dict[str, Any]) -> Dict[str, float]:
        """Analyze cache efficiency metrics."""
        try:
            usage_stats = cache_stats.get('usage_statistics', {})
            storage_stats = cache_stats.get('storage_statistics', {})
            
            hit_rate = usage_stats.get('cache_hits', 0) / max(usage_stats.get('total_requests', 1), 1)
            
            # Cost efficiency
            cost_reduction = usage_stats.get('cost_reduction', 0)
            total_tokens = usage_stats.get('total_tokens_cached', 0)
            cost_efficiency = cost_reduction / max(total_tokens, 1) if total_tokens > 0 else 0
            
            # Storage efficiency
            storage_size_mb = storage_stats.get('storage_size_mb', 0)
            total_entries = storage_stats.get('total_entries', 0)
            storage_efficiency = total_entries / max(storage_size_mb, 0.1)
            
            # Token efficiency
            total_tokens_stored = storage_stats.get('total_tokens', 0)
            token_efficiency = total_tokens_stored / max(storage_size_mb, 0.1)
            
            return {
                'hit_rate': hit_rate,
                'cost_efficiency': cost_efficiency,
                'storage_efficiency': storage_efficiency,
                'token_efficiency': token_efficiency,
                'overall_efficiency': (hit_rate + cost_efficiency + 
                                     min(storage_efficiency / 100, 1.0) + 
                                     min(token_efficiency / 1000, 1.0)) / 4
            }
            
        except Exception as e:
            logging.error(f"Failed to analyze efficiency: {e}")
            return {'hit_rate': 0.0, 'cost_efficiency': 0.0, 'storage_efficiency': 0.0, 
                   'token_efficiency': 0.0, 'overall_efficiency': 0.0}
    
    def _analyze_usage_patterns(self) -> Dict[str, Any]:
        """Analyze cache usage patterns."""
        try:
            prefix_store = self.cache_manager.prefix_store
            
            if not prefix_store.index:
                return {
                    'total_patterns': 0,
                    'pattern_types': {},
                    'access_distribution': {},
                    'temporal_patterns': {},
                    'size_distribution': {}
                }
            
            # Analyze access patterns
            access_counts = [meta['access_count'] for meta in prefix_store.index.values()]
            access_distribution = {
                'mean': statistics.mean(access_counts) if access_counts else 0,
                'median': statistics.median(access_counts) if access_counts else 0,
                'std_dev': statistics.stdev(access_counts) if len(access_counts) > 1 else 0,
                'min': min(access_counts) if access_counts else 0,
                'max': max(access_counts) if access_counts else 0
            }
            
            # Analyze temporal patterns
            temporal_patterns = self._analyze_temporal_patterns(prefix_store.index)
            
            # Analyze size distribution
            token_counts = [meta['token_count'] for meta in prefix_store.index.values()]
            size_distribution = {
                'mean_tokens': statistics.mean(token_counts) if token_counts else 0,
                'median_tokens': statistics.median(token_counts) if token_counts else 0,
                'small_entries': len([t for t in token_counts if t < 100]),
                'medium_entries': len([t for t in token_counts if 100 <= t < 500]),
                'large_entries': len([t for t in token_counts if t >= 500])
            }
            
            # Detect content patterns
            content_patterns = self.pattern_detector.detect_content_patterns(prefix_store.index)
            
            return {
                'total_patterns': len(prefix_store.index),
                'pattern_types': content_patterns,
                'access_distribution': access_distribution,
                'temporal_patterns': temporal_patterns,
                'size_distribution': size_distribution
            }
            
        except Exception as e:
            logging.error(f"Failed to analyze usage patterns: {e}")
            return {}
    
    def _analyze_temporal_patterns(self, index: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal access patterns."""
        try:
            if not index:
                return {}
            
            # Parse timestamps
            created_times = []
            accessed_times = []
            
            for meta in index.values():
                try:
                    created_times.append(datetime.fromisoformat(meta['created_at']))
                    accessed_times.append(datetime.fromisoformat(meta['last_accessed']))
                except (ValueError, KeyError):
                    continue
            
            if not created_times:
                return {}
            
            # Analyze creation patterns
            now = datetime.now()
            creation_age_hours = [(now - ct).total_seconds() / 3600 for ct in created_times]
            
            # Analyze access patterns
            access_age_hours = [(now - at).total_seconds() / 3600 for at in accessed_times]
            
            # Find peak hours (simplified)
            access_hours = [at.hour for at in accessed_times]
            peak_hour = Counter(access_hours).most_common(1)[0][0] if access_hours else 0
            
            return {
                'average_age_hours': statistics.mean(creation_age_hours),
                'average_last_access_hours': statistics.mean(access_age_hours),
                'entries_last_24h': len([age for age in creation_age_hours if age <= 24]),
                'entries_last_week': len([age for age in creation_age_hours if age <= 168]),
                'peak_access_hour': peak_hour,
                'recent_access_rate': len([age for age in access_age_hours if age <= 1]) / max(len(access_age_hours), 1)
            }
            
        except Exception as e:
            logging.error(f"Failed to analyze temporal patterns: {e}")
            return {}
    
    def _calculate_performance_metrics(self, cache_stats: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance metrics."""
        try:
            usage_stats = cache_stats.get('usage_statistics', {})
            storage_stats = cache_stats.get('storage_statistics', {})
            
            # Response time metric (estimated)
            hit_rate = usage_stats.get('cache_hits', 0) / max(usage_stats.get('total_requests', 1), 1)
            estimated_response_time_ms = (1 - hit_rate) * 1000 + hit_rate * 50  # Cache hits are 50ms, misses 1000ms
            
            # Throughput metric
            total_requests = usage_stats.get('total_requests', 0)
            # Assume analysis covers last hour for throughput calculation
            estimated_throughput_per_hour = total_requests
            
            # Memory efficiency
            storage_size_mb = storage_stats.get('storage_size_mb', 0)
            total_entries = storage_stats.get('total_entries', 0)
            memory_efficiency = total_entries / max(storage_size_mb, 0.1)
            
            # Cost savings rate
            cost_reduction = usage_stats.get('cost_reduction', 0)
            cost_savings_rate = cost_reduction / max(total_requests, 1)
            
            return {
                'estimated_response_time_ms': estimated_response_time_ms,
                'estimated_throughput_per_hour': estimated_throughput_per_hour,
                'memory_efficiency_entries_per_mb': memory_efficiency,
                'cost_savings_rate': cost_savings_rate,
                'cache_utilization': min(storage_size_mb / self.cache_manager.max_cache_size_mb, 1.0)
            }
            
        except Exception as e:
            logging.error(f"Failed to calculate performance metrics: {e}")
            return {}
    
    def _identify_optimization_opportunities(self, cache_stats: Dict[str, Any], 
                                           patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify optimization opportunities."""
        opportunities = []
        
        try:
            usage_stats = cache_stats.get('usage_statistics', {})
            storage_stats = cache_stats.get('storage_statistics', {})
            
            # Low hit rate opportunity
            hit_rate = usage_stats.get('cache_hits', 0) / max(usage_stats.get('total_requests', 1), 1)
            if hit_rate < 0.5:
                opportunities.append({
                    'type': 'hit_rate_improvement',
                    'impact': 'high',
                    'description': f'Hit rate is {hit_rate:.2%}, below optimal threshold',
                    'recommendation': 'Review prefix extraction strategy and caching criteria',
                    'potential_improvement': f'Could improve hit rate to 70%+',
                    'effort': 'medium'
                })
            
            # Large unused entries opportunity
            access_dist = patterns.get('access_distribution', {})
            if access_dist.get('min', 0) == 1 and storage_stats.get('total_entries', 0) > 50:
                single_access_count = len([meta for meta in self.cache_manager.prefix_store.index.values() 
                                         if meta['access_count'] == 1])
                if single_access_count > storage_stats.get('total_entries', 0) * 0.3:
                    opportunities.append({
                        'type': 'unused_entry_cleanup',
                        'impact': 'medium',
                        'description': f'{single_access_count} entries have only been accessed once',
                        'recommendation': 'Implement aggressive cleanup for single-access entries',
                        'potential_improvement': f'Could free up ~{single_access_count * 0.001:.1f}MB',
                        'effort': 'low'
                    })
            
            # Size optimization opportunity
            size_dist = patterns.get('size_distribution', {})
            large_entries = size_dist.get('large_entries', 0)
            total_entries = storage_stats.get('total_entries', 0)
            if large_entries > total_entries * 0.2:
                opportunities.append({
                    'type': 'size_optimization',
                    'impact': 'medium',
                    'description': f'{large_entries} entries are large (>500 tokens)',
                    'recommendation': 'Consider splitting large entries or improving compression',
                    'potential_improvement': 'Could reduce storage by 20-30%',
                    'effort': 'high'
                })
            
            # Cache size optimization
            cache_utilization = min(storage_stats.get('storage_size_mb', 0) / 
                                  self.cache_manager.max_cache_size_mb, 1.0)
            if cache_utilization > 0.9:
                opportunities.append({
                    'type': 'cache_size_increase',
                    'impact': 'medium',
                    'description': f'Cache is {cache_utilization:.1%} full',
                    'recommendation': 'Consider increasing cache size limit or improving cleanup',
                    'potential_improvement': 'Could improve hit rate and reduce evictions',
                    'effort': 'low'
                })
            elif cache_utilization < 0.3:
                opportunities.append({
                    'type': 'cache_size_decrease',
                    'impact': 'low',
                    'description': f'Cache is only {cache_utilization:.1%} utilized',
                    'recommendation': 'Consider reducing cache size limit to save memory',
                    'potential_improvement': 'Could free up system memory',
                    'effort': 'low'
                })
            
            # Pattern-based opportunities
            content_patterns = patterns.get('pattern_types', {})
            if content_patterns.get('system_prompts', 0) < 5 and total_entries > 20:
                opportunities.append({
                    'type': 'system_prompt_caching',
                    'impact': 'high',
                    'description': 'Low system prompt caching detected',
                    'recommendation': 'Improve system prompt detection and caching',
                    'potential_improvement': 'Could improve hit rate by 15-25%',
                    'effort': 'medium'
                })
            
        except Exception as e:
            logging.error(f"Failed to identify optimization opportunities: {e}")
        
        return opportunities
    
    def _generate_recommendations(self, efficiency: Dict[str, float], 
                                patterns: Dict[str, Any], 
                                opportunities: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        try:
            # Hit rate recommendations
            hit_rate = efficiency.get('hit_rate', 0)
            if hit_rate < 0.3:
                recommendations.append(
                    "CRITICAL: Very low cache hit rate. Review prefix extraction logic "
                    "and ensure consistent content hashing."
                )
            elif hit_rate < 0.5:
                recommendations.append(
                    "WARNING: Below optimal hit rate. Consider expanding prefix candidates "
                    "or improving content similarity detection."
                )
            elif hit_rate > 0.8:
                recommendations.append(
                    "EXCELLENT: High hit rate achieved. Monitor for potential over-caching."
                )
            
            # Storage recommendations
            storage_efficiency = efficiency.get('storage_efficiency', 0)
            if storage_efficiency < 50:  # Less than 50 entries per MB
                recommendations.append(
                    "Storage efficiency is low. Consider compressing cached content "
                    "or removing low-value entries."
                )
            
            # Pattern-based recommendations
            size_dist = patterns.get('size_distribution', {})
            if size_dist.get('small_entries', 0) > size_dist.get('large_entries', 0) * 3:
                recommendations.append(
                    "Many small cache entries detected. Consider merging related small "
                    "entries or increasing minimum cache size threshold."
                )
            
            # Access pattern recommendations
            access_dist = patterns.get('access_distribution', {})
            if access_dist.get('std_dev', 0) > access_dist.get('mean', 1):
                recommendations.append(
                    "High variance in access patterns. Implement tiered caching "
                    "with different retention policies for different access patterns."
                )
            
            # Temporal recommendations
            temporal = patterns.get('temporal_patterns', {})
            recent_access_rate = temporal.get('recent_access_rate', 0)
            if recent_access_rate < 0.3:
                recommendations.append(
                    "Low recent access rate indicates stale cache entries. "
                    "Implement more aggressive cleanup of old entries."
                )
            
            # High-impact opportunity recommendations
            high_impact_opportunities = [opp for opp in opportunities if opp.get('impact') == 'high']
            for opp in high_impact_opportunities:
                recommendations.append(f"HIGH IMPACT: {opp['recommendation']}")
            
            # Add general recommendations if no specific issues found
            if not recommendations:
                if hit_rate > 0.6 and efficiency.get('overall_efficiency', 0) > 0.7:
                    recommendations.append(
                        "Cache is performing well. Consider monitoring trends "
                        "and implementing predictive prefetching."
                    )
                else:
                    recommendations.append(
                        "Cache performance is moderate. Monitor usage patterns "
                        "and consider optimizations based on specific use cases."
                    )
            
        except Exception as e:
            logging.error(f"Failed to generate recommendations: {e}")
            recommendations.append(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def _calculate_health_score(self, efficiency: Dict[str, float], 
                              performance: Dict[str, float]) -> float:
        """Calculate overall cache health score (0.0 to 1.0)."""
        try:
            # Weighted score calculation
            weights = {
                'hit_rate': 0.3,
                'cost_efficiency': 0.25,
                'storage_efficiency_normalized': 0.2,
                'memory_efficiency_normalized': 0.15,
                'cache_utilization': 0.1
            }
            
            scores = {}
            
            # Hit rate score
            scores['hit_rate'] = min(efficiency.get('hit_rate', 0) / 0.8, 1.0)  # Target 80% hit rate
            
            # Cost efficiency score
            scores['cost_efficiency'] = min(efficiency.get('cost_efficiency', 0) / 0.9, 1.0)  # Target 90% cost efficiency
            
            # Storage efficiency score (normalize to 0-1 range)
            storage_eff = efficiency.get('storage_efficiency', 0)
            scores['storage_efficiency_normalized'] = min(storage_eff / 100, 1.0)  # Target 100 entries/MB
            
            # Memory efficiency score (normalize to 0-1 range)
            memory_eff = performance.get('memory_efficiency_entries_per_mb', 0)
            scores['memory_efficiency_normalized'] = min(memory_eff / 100, 1.0)
            
            # Cache utilization score (optimal around 70%)
            utilization = performance.get('cache_utilization', 0)
            if utilization <= 0.7:
                scores['cache_utilization'] = utilization / 0.7
            else:
                scores['cache_utilization'] = max(0.5, 1.0 - (utilization - 0.7) / 0.3)
            
            # Calculate weighted average
            health_score = sum(scores[key] * weights[key] for key in weights if key in scores)
            
            return max(0.0, min(1.0, health_score))
            
        except Exception as e:
            logging.error(f"Failed to calculate health score: {e}")
            return 0.5
    
    def _create_error_analysis(self, error_message: str) -> CacheAnalysis:
        """Create error analysis result."""
        return CacheAnalysis(
            analysis_timestamp=datetime.now(),
            cache_efficiency={'error': error_message},
            usage_patterns={'error': error_message},
            performance_metrics={'error': error_message},
            optimization_opportunities=[],
            recommendations=[f"Error during analysis: {error_message}"],
            health_score=0.0
        )
    
    def get_analysis_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get analysis trends over time."""
        try:
            if not self.analysis_history:
                return {'error': 'No analysis history available'}
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_analyses = [
                analysis for analysis in self.analysis_history
                if analysis.analysis_timestamp >= cutoff_time
            ]
            
            if not recent_analyses:
                return {'error': f'No analyses in the last {hours} hours'}
            
            # Extract trends
            hit_rates = [a.cache_efficiency.get('hit_rate', 0) for a in recent_analyses]
            health_scores = [a.health_score for a in recent_analyses]
            
            trends = {
                'period_hours': hours,
                'analysis_count': len(recent_analyses),
                'hit_rate_trend': {
                    'values': hit_rates,
                    'average': statistics.mean(hit_rates),
                    'min': min(hit_rates),
                    'max': max(hit_rates),
                    'trend': 'improving' if len(hit_rates) > 1 and hit_rates[-1] > hit_rates[0] else 'declining'
                },
                'health_score_trend': {
                    'values': health_scores,
                    'average': statistics.mean(health_scores),
                    'min': min(health_scores),
                    'max': max(health_scores),
                    'trend': 'improving' if len(health_scores) > 1 and health_scores[-1] > health_scores[0] else 'declining'
                },
                'latest_analysis': recent_analyses[-1].to_dict()
            }
            
            return trends
            
        except Exception as e:
            logging.error(f"Failed to get analysis trends: {e}")
            return {'error': str(e)}

class CachePatternDetector:
    """Detects patterns in cache usage."""
    
    def detect_content_patterns(self, index: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
        """Detect content type patterns in cached entries."""
        try:
            patterns = defaultdict(int)
            
            for hash_key, metadata in index.items():
                # Pattern detection based on metadata or content analysis
                # This is simplified - in practice, you'd analyze actual content
                
                if 'system' in str(metadata.get('metadata', {})).lower():
                    patterns['system_prompts'] += 1
                elif 'config' in str(metadata.get('metadata', {})).lower():
                    patterns['configuration'] += 1
                elif 'template' in str(metadata.get('metadata', {})).lower():
                    patterns['templates'] += 1
                elif 'doc' in str(metadata.get('metadata', {})).lower():
                    patterns['documentation'] += 1
                else:
                    patterns['general_content'] += 1
                
                # Size-based patterns
                token_count = metadata.get('token_count', 0)
                if token_count < 100:
                    patterns['small_entries'] += 1
                elif token_count < 500:
                    patterns['medium_entries'] += 1
                else:
                    patterns['large_entries'] += 1
                
                # Access-based patterns
                access_count = metadata.get('access_count', 0)
                if access_count == 1:
                    patterns['single_access'] += 1
                elif access_count < 5:
                    patterns['low_access'] += 1
                elif access_count < 20:
                    patterns['medium_access'] += 1
                else:
                    patterns['high_access'] += 1
            
            return dict(patterns)
            
        except Exception as e:
            logging.error(f"Failed to detect content patterns: {e}")
            return {}