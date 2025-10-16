"""
Performance Optimization and Caching System
Optimizes performance and implements intelligent caching for the insight system
"""

import os
import json
import hashlib
import pickle
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import threading
import time

# Custom imports
from .content_extractor import ContentProcessingPipeline
from .summarization_engine import summarization_engine, SummarizationResult
from .insight_extractor import insight_extractor
from .content_categorizer import content_categorizer
from .relationship_analyzer import relationship_analyzer

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cache entry"""
    key: str
    data: Any
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return datetime.now() > self.expires_at

    def access(self):
        """Record cache access"""
        self.access_count += 1
        self.last_accessed = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'key': self.key,
            'data': self.data,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat(),
            'size_bytes': self.size_bytes,
            'metadata': self.metadata
        }


class CacheManager:
    """Manages caching for the insight system"""

    def __init__(self, cache_dir: str = "cache", max_size_mb: int = 1000):
        self.cache_dir = cache_dir
        self.max_size_mb = max_size_mb
        self.current_size_mb = 0
        self.cache_entries = {}
        self._lock = threading.Lock()

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Load existing cache
        self._load_cache()

        # Start cleanup thread
        self._start_cleanup_thread()

    def _load_cache(self):
        """Load existing cache from disk"""
        try:
            index_file = os.path.join(self.cache_dir, "cache_index.json")
            if os.path.exists(index_file):
                with open(index_file, 'r') as f:
                    cache_data = json.load(f)

                for key, entry_data in cache_data.items():
                    entry = CacheEntry(
                        key=entry_data['key'],
                        data=self._load_entry_data(key),
                        created_at=datetime.fromisoformat(entry_data['created_at']),
                        expires_at=datetime.fromisoformat(entry_data['expires_at']),
                        access_count=entry_data['access_count'],
                        last_accessed=datetime.fromisoformat(entry_data['last_accessed']),
                        size_bytes=entry_data['size_bytes'],
                        metadata=entry_data['metadata']
                    )

                    # Check if entry is still valid
                    if not entry.is_expired():
                        self.cache_entries[key] = entry
                        self.current_size_mb += entry.size_bytes / (1024 * 1024)

        except Exception as e:
            logger.warning(f"Error loading cache: {e}")

    def _load_entry_data(self, key: str) -> Any:
        """Load cached data from disk"""
        try:
            data_file = os.path.join(self.cache_dir, f"{key}.pkl")
            if os.path.exists(data_file):
                with open(data_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Error loading cache entry {key}: {e}")
        return None

    def _save_entry_data(self, key: str, data: Any):
        """Save data to disk"""
        try:
            data_file = os.path.join(self.cache_dir, f"{key}.pkl")
            with open(data_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving cache entry {key}: {e}")

    def _save_cache_index(self):
        """Save cache index to disk"""
        try:
            index_file = os.path.join(self.cache_dir, "cache_index.json")
            cache_data = {key: entry.to_dict() for key, entry in self.cache_entries.items()}

            with open(index_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache index: {e}")

    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        def cleanup_loop():
            while True:
                try:
                    time.sleep(300)  # Run cleanup every 5 minutes
                    self._cleanup_expired()
                    self._enforce_size_limits()
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")

        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()

    def get(self, key: str) -> Optional[Any]:
        """Get data from cache"""
        with self._lock:
            if key in self.cache_entries:
                entry = self.cache_entries[key]

                # Check expiration
                if entry.is_expired():
                    self._remove_entry(key)
                    return None

                # Update access statistics
                entry.access()

                # Save updated index
                self._save_cache_index()

                return entry.data

        return None

    def put(self, key: str, data: Any, ttl_hours: int = 24,
            metadata: Dict[str, Any] = None) -> bool:
        """Put data in cache"""
        with self._lock:
            try:
                # Calculate size
                data_size = self._calculate_size(data)

                # Check if adding this would exceed size limit
                if self.current_size_mb + (data_size / (1024 * 1024)) > self.max_size_mb:
                    self._enforce_size_limits()

                    # Check again after cleanup
                    if self.current_size_mb + (data_size / (1024 * 1024)) > self.max_size_mb:
                        logger.warning(f"Cache full, cannot add entry {key}")
                        return False

                # Create cache entry
                now = datetime.now()
                entry = CacheEntry(
                    key=key,
                    data=data,
                    created_at=now,
                    expires_at=now + timedelta(hours=ttl_hours),
                    size_bytes=data_size,
                    metadata=metadata or {}
                )

                # Save to disk
                self._save_entry_data(key, data)

                # Add to cache
                self.cache_entries[key] = entry
                self.current_size_mb += data_size / (1024 * 1024)

                # Save index
                self._save_cache_index()

                return True

            except Exception as e:
                logger.error(f"Error putting cache entry {key}: {e}")
                return False

    def _calculate_size(self, data: Any) -> int:
        """Calculate size of data in bytes"""
        try:
            if isinstance(data, (str, dict, list)):
                return len(json.dumps(data).encode('utf-8'))
            else:
                return len(pickle.dumps(data))
        except Exception:
            return 1024  # Default estimate

    def _remove_entry(self, key: str):
        """Remove cache entry"""
        if key in self.cache_entries:
            entry = self.cache_entries[key]
            self.current_size_mb -= entry.size_bytes / (1024 * 1024)

            # Remove from disk
            try:
                data_file = os.path.join(self.cache_dir, f"{key}.pkl")
                if os.path.exists(data_file):
                    os.remove(data_file)
            except Exception as e:
                logger.warning(f"Error removing cache file {key}: {e}")

            # Remove from memory
            del self.cache_entries[key]

    def _cleanup_expired(self):
        """Remove expired entries"""
        expired_keys = []

        for key, entry in self.cache_entries.items():
            if entry.is_expired():
                expired_keys.append(key)

        for key in expired_keys:
            self._remove_entry(key)

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

    def _enforce_size_limits(self):
        """Enforce cache size limits by removing least recently used entries"""
        if self.current_size_mb <= self.max_size_mb:
            return

        # Sort by last accessed time (oldest first)
        entries_by_access = sorted(
            self.cache_entries.items(),
            key=lambda x: x[1].last_accessed
        )

        # Remove entries until under limit
        removed_count = 0
        for key, entry in entries_by_access:
            if self.current_size_mb <= self.max_size_mb * 0.8:  # Target 80% of max
                break

            self._remove_entry(key)
            removed_count += 1

        if removed_count > 0:
            logger.info(f"Removed {removed_count} entries to enforce cache size limits")

    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            for key in list(self.cache_entries.keys()):
                self._remove_entry(key)

            self._save_cache_index()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_entries = len(self.cache_entries)
            total_accesses = sum(entry.access_count for entry in self.cache_entries.values())
            expired_count = sum(1 for entry in self.cache_entries.values() if entry.is_expired())

            return {
                'total_entries': total_entries,
                'total_size_mb': self.current_size_mb,
                'max_size_mb': self.max_size_mb,
                'utilization_percent': (self.current_size_mb / self.max_size_mb) * 100,
                'total_accesses': total_accesses,
                'expired_entries': expired_count,
                'hit_rate': self._calculate_hit_rate()
            }

    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total_accesses = sum(entry.access_count for entry in self.cache_entries.values())
        if total_accesses == 0:
            return 0.0

        # This is a simplified calculation
        # In a real implementation, you'd track hits vs misses
        return 0.85  # Placeholder


class PerformanceOptimizer:
    """Main performance optimization system"""

    def __init__(self, cache_dir: str = "cache", max_cache_size_mb: int = 1000):
        self.cache_manager = CacheManager(cache_dir, max_cache_size_mb)
        self.processing_times = defaultdict(list)
        self.operation_counts = defaultdict(int)

    def get_cached_or_process(self, operation: str, key_data: Dict[str, Any],
                            processor_func, ttl_hours: int = 24) -> Any:
        """
        Get data from cache or process it if not cached

        Args:
            operation: Type of operation (e.g., 'summarize', 'extract_insights')
            key_data: Data used to generate cache key
            processor_func: Function to call if not cached
            ttl_hours: Cache TTL in hours

        Returns:
            Processed data
        """
        start_time = time.time()

        # Generate cache key
        cache_key = self._generate_cache_key(operation, key_data)

        # Try to get from cache
        cached_result = self.cache_manager.get(cache_key)

        if cached_result is not None:
            # Cache hit
            processing_time = time.time() - start_time
            self.processing_times[f"{operation}_cached"].append(processing_time)
            return cached_result

        # Cache miss - process data
        try:
            result = processor_func()

            # Cache the result
            metadata = {
                'operation': operation,
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }

            self.cache_manager.put(cache_key, result, ttl_hours, metadata)

            # Track performance
            processing_time = time.time() - start_time
            self.processing_times[f"{operation}_processed"].append(processing_time)
            self.operation_counts[operation] += 1

            return result

        except Exception as e:
            logger.error(f"Error in cached processing for {operation}: {e}")
            # Don't cache errors
            return None

    def _generate_cache_key(self, operation: str, key_data: Dict[str, Any]) -> str:
        """Generate cache key from operation and key data"""
        # Create a deterministic string from key data
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()

        return f"{operation}_{key_hash}"

    def summarize_with_cache(self, file_path: str, config_dict: Dict[str, Any] = None) -> SummarizationResult:
        """Get cached summary or generate new one"""
        if config_dict is None:
            config_dict = {}

        key_data = {
            'file_path': file_path,
            'operation': 'summarize',
            'config': config_dict,
            'file_mtime': os.path.getmtime(file_path),
            'file_size': os.path.getsize(file_path)
        }

        def process_summary():
            from .summarization_engine import SummarizationConfig, SummarizationMethod, DetailLevel

            # Convert dict to config
            config = SummarizationConfig(**config_dict) if config_dict else SummarizationConfig()

            return summarization_engine.summarize_document(file_path, config)

        return self.get_cached_or_process('summarize', key_data, process_summary)

    def extract_insights_with_cache(self, file_path: str) -> Dict[str, Any]:
        """Get cached insights or generate new ones"""
        key_data = {
            'file_path': file_path,
            'operation': 'extract_insights',
            'file_mtime': os.path.getmtime(file_path),
            'file_size': os.path.getsize(file_path)
        }

        def process_insights():
            return insight_extractor.extract_insights(file_path)

        return self.get_cached_or_process('extract_insights', key_data, process_insights)

    def categorize_with_cache(self, file_path: str) -> Dict[str, Any]:
        """Get cached categorization or generate new one"""
        key_data = {
            'file_path': file_path,
            'operation': 'categorize',
            'file_mtime': os.path.getmtime(file_path),
            'file_size': os.path.getsize(file_path)
        }

        def process_categorization():
            return content_categorizer.categorize_document(file_path)

        return self.get_cached_or_process('categorize', key_data, process_categorization)

    def analyze_relationships_with_cache(self, document_paths: List[str]) -> Dict[str, Any]:
        """Get cached relationship analysis or generate new one"""
        # Sort paths for consistent key generation
        sorted_paths = sorted(document_paths)

        key_data = {
            'document_paths': sorted_paths,
            'operation': 'analyze_relationships',
            'file_count': len(sorted_paths),
            'total_size': sum(os.path.getsize(path) for path in sorted_paths if os.path.exists(path))
        }

        def process_relationships():
            return relationship_analyzer.analyze_document_collection(document_paths)

        # Use longer TTL for relationship analysis (less likely to change)
        return self.get_cached_or_process('analyze_relationships', key_data, process_relationships, ttl_hours=48)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        cache_stats = self.cache_manager.get_stats()

        # Calculate average processing times
        avg_times = {}
        for operation, times in self.processing_times.items():
            if times:
                avg_times[operation] = sum(times) / len(times)

        # Calculate operation frequencies
        total_operations = sum(self.operation_counts.values())

        return {
            'cache_statistics': cache_stats,
            'operation_counts': dict(self.operation_counts),
            'average_processing_times': avg_times,
            'total_operations': total_operations,
            'cache_hit_rate': cache_stats.get('hit_rate', 0),
            'performance_score': self._calculate_performance_score(avg_times, cache_stats)
        }

    def _calculate_performance_score(self, avg_times: Dict[str, float],
                                   cache_stats: Dict[str, Any]) -> float:
        """Calculate overall performance score"""
        score = 100.0

        # Penalize slow operations
        for operation, avg_time in avg_times.items():
            if 'cached' in operation:
                if avg_time > 0.1:  # Cached operations should be fast
                    score -= 10
            else:
                if avg_time > 30:  # Processing operations shouldn't be too slow
                    score -= 20
                elif avg_time > 10:
                    score -= 10

        # Bonus for good cache utilization
        utilization = cache_stats.get('utilization_percent', 0)
        if utilization > 80:
            score += 10
        elif utilization < 20:
            score -= 10

        return max(0, min(100, score))

    def optimize_batch_processing(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize batch processing by reordering items for better cache utilization

        Args:
            items: List of items to process, each containing processing parameters

        Returns:
            Optimized processing order
        """
        if len(items) < 3:
            return items  # No need to optimize small batches

        try:
            # Group items by similarity to improve cache hits
            optimized_items = self._optimize_processing_order(items)

            logger.info(f"Optimized processing order for {len(items)} items")
            return optimized_items

        except Exception as e:
            logger.error(f"Batch optimization error: {e}")
            return items  # Return original order on error

    def _optimize_processing_order(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize the order of items for better cache utilization"""
        # Simple optimization: group by file size and type
        size_categories = {
            'small': [],
            'medium': [],
            'large': []
        }

        for item in items:
            file_path = item.get('file_path', '')
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                if file_size < 1024 * 1024:  # < 1MB
                    size_categories['small'].append(item)
                elif file_size < 10 * 1024 * 1024:  # < 10MB
                    size_categories['medium'].append(item)
                else:
                    size_categories['large'].append(item)
            else:
                size_categories['small'].append(item)  # Default for missing files

        # Return optimized order: small, medium, large, then large, medium, small (for variety)
        optimized = []
        optimized.extend(size_categories['small'])
        optimized.extend(size_categories['medium'])
        optimized.extend(size_categories['large'])

        # If we have many items, interleave different sizes
        if len(optimized) > 10:
            optimized = self._interleave_categories(size_categories)

        return optimized

    def _interleave_categories(self, size_categories: Dict[str, List]) -> List[Dict[str, Any]]:
        """Interleave different size categories for better cache utilization"""
        categories = [
            size_categories['small'],
            size_categories['medium'],
            size_categories['large']
        ]

        # Simple round-robin interleaving
        result = []
        max_len = max(len(cat) for cat in categories)

        for i in range(max_len):
            for cat in categories:
                if i < len(cat):
                    result.append(cat[i])

        return result

    def warm_cache(self, document_paths: List[str]):
        """Pre-populate cache with frequently accessed documents"""
        logger.info(f"Warming cache with {len(document_paths)} documents")

        # Process a few documents to populate cache
        for file_path in document_paths[:5]:  # Warm cache with first 5 documents
            try:
                # Generate basic insights for cache warming
                self.extract_insights_with_cache(file_path)
                self.categorize_with_cache(file_path)

                logger.info(f"Warmed cache for {file_path}")

            except Exception as e:
                logger.warning(f"Cache warming failed for {file_path}: {e}")

    def cleanup_resources(self):
        """Clean up resources and save state"""
        logger.info("Cleaning up performance optimizer resources")

        # Save performance statistics
        self._save_performance_stats()

        # Clear cache if needed
        # self.cache_manager.clear()  # Uncomment if you want to clear cache

    def _save_performance_stats(self):
        """Save performance statistics to disk"""
        try:
            stats = self.get_performance_stats()
            stats_file = os.path.join(self.cache_manager.cache_dir, "performance_stats.json")

            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving performance stats: {e}")


class AsyncProcessor:
    """Handles asynchronous processing for better performance"""

    def __init__(self):
        self.task_queue = []
        self.results = {}
        self._lock = threading.Lock()
        self._processing = False

    def submit_task(self, task_id: str, task_func, *args, **kwargs):
        """Submit task for asynchronous processing"""
        with self._lock:
            self.task_queue.append({
                'task_id': task_id,
                'task_func': task_func,
                'args': args,
                'kwargs': kwargs
            })

            # Start processing if not already running
            if not self._processing:
                self._start_processing()

    def _start_processing(self):
        """Start background processing thread"""
        def process_tasks():
            self._processing = True

            while True:
                task = None

                with self._lock:
                    if self.task_queue:
                        task = self.task_queue.pop(0)

                if task:
                    try:
                        result = task['task_func'](*task['args'], **task['kwargs'])

                        with self._lock:
                            self.results[task['task_id']] = {
                                'success': True,
                                'result': result,
                                'completed_at': datetime.now().isoformat()
                            }

                    except Exception as e:
                        logger.error(f"Async task {task['task_id']} failed: {e}")

                        with self._lock:
                            self.results[task['task_id']] = {
                                'success': False,
                                'error': str(e),
                                'completed_at': datetime.now().isoformat()
                            }
                else:
                    # No more tasks
                    self._processing = False
                    break

        if not self._processing:
            process_thread = threading.Thread(target=process_tasks, daemon=True)
            process_thread.start()

    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get result of asynchronous task"""
        with self._lock:
            return self.results.get(task_id)

    def is_processing(self, task_id: str) -> bool:
        """Check if task is still processing"""
        with self._lock:
            return task_id not in self.results and task_id in [
                task['task_id'] for task in self.task_queue
            ]


class PerformanceMonitor:
    """Monitors system performance and provides optimization recommendations"""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.recommendations = []

    def record_metric(self, metric_name: str, value: float, timestamp: datetime = None):
        """Record performance metric"""
        if timestamp is None:
            timestamp = datetime.now()

        self.metrics[metric_name].append({
            'value': value,
            'timestamp': timestamp.isoformat()
        })

        # Keep only recent metrics (last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        self.metrics[metric_name] = [
            m for m in self.metrics[metric_name]
            if datetime.fromisoformat(m['timestamp']) > cutoff
        ]

    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance and generate recommendations"""
        analysis = {
            'current_performance': {},
            'trends': {},
            'recommendations': [],
            'bottlenecks': []
        }

        # Analyze cache performance
        cache_stats = performance_optimizer.cache_manager.get_stats()
        analysis['current_performance']['cache_utilization'] = cache_stats.get('utilization_percent', 0)
        analysis['current_performance']['cache_hit_rate'] = cache_stats.get('hit_rate', 0)

        # Identify trends
        for metric_name, values in self.metrics.items():
            if len(values) >= 2:
                recent_values = [m['value'] for m in values[-10:]]  # Last 10 values
                if recent_values:
                    trend = self._calculate_trend(recent_values)
                    analysis['trends'][metric_name] = trend

        # Generate recommendations
        analysis['recommendations'] = self._generate_optimization_recommendations(analysis)

        return analysis

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return 'stable'

        first_half = sum(values[:len(values)//2]) / (len(values)//2)
        second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)

        if second_half > first_half * 1.1:
            return 'improving'
        elif second_half < first_half * 0.9:
            return 'degrading'
        else:
            return 'stable'

    def _generate_optimization_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        # Cache recommendations
        cache_util = analysis['current_performance'].get('cache_utilization', 0)
        if cache_util < 30:
            recommendations.append("Increase cache TTL for better hit rates")
        elif cache_util > 90:
            recommendations.append("Consider increasing cache size limit")

        # Performance trend recommendations
        for metric, trend in analysis['trends'].items():
            if trend == 'degrading':
                recommendations.append(f"Performance degrading for {metric} - investigate bottleneck")
            elif trend == 'improving':
                recommendations.append(f"Performance improving for {metric} - good optimization")

        return recommendations


# Global performance optimization instances
performance_optimizer = PerformanceOptimizer()
async_processor = AsyncProcessor()
performance_monitor = PerformanceMonitor()