#!/usr/bin/env python3
"""
Memory service orchestrator for Prsist Memory System Phase 3.
Manages and coordinates all memory system services.
"""

import asyncio
import logging
import json
import subprocess
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from database import MemoryDatabase
from memory_manager import MemoryManager
from context_manager import ContextManager
from relevance_scorer import RelevanceScorer
from knowledge_manager import KnowledgeManager
from advanced_change_analyzer import AdvancedChangeImpactAnalyzer
from utils import setup_logging, load_yaml_config

class ServiceStatus:
    """Represents the status of a service."""
    
    def __init__(self, name: str):
        self.name = name
        self.status = "stopped"  # stopped, starting, running, stopping, failed
        self.last_heartbeat = None
        self.error_count = 0
        self.start_time = None
        self.process = None
        self.metadata = {}
    
    def update_heartbeat(self):
        """Update service heartbeat."""
        self.last_heartbeat = datetime.now()
    
    def is_healthy(self, timeout_seconds: int = 60) -> bool:
        """Check if service is healthy."""
        if self.status != "running":
            return False
        
        if self.last_heartbeat is None:
            return False
        
        time_since_heartbeat = (datetime.now() - self.last_heartbeat).total_seconds()
        return time_since_heartbeat < timeout_seconds

class MemoryServiceOrchestrator:
    """Orchestrates all memory system services."""
    
    def __init__(self, config_path: str = None):
        """Initialize service orchestrator."""
        self.config_path = config_path or ".prsist/config/memory-config-v3.yaml"
        self.config = self._load_config()
        self.services = {}
        self.task_queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = False
        self.background_tasks = []
        
        # Initialize core components
        self.memory_db = MemoryDatabase()
        self.memory_manager = MemoryManager()
        self.context_manager = ContextManager(self.config)
        self.relevance_scorer = RelevanceScorer()
        self.knowledge_manager = KnowledgeManager(".prsist/storage")
        self.change_analyzer = AdvancedChangeImpactAnalyzer(self.memory_db)
        
        # Set up context manager with relevance scorer
        self.context_manager.relevance_scorer = self.relevance_scorer
        
        setup_logging(self.config.get("logging", {}).get("level", "INFO"))
        
    def _load_config(self) -> Dict[str, Any]:
        """Load service configuration."""
        try:
            if Path(self.config_path).exists():
                return load_yaml_config(self.config_path)
            else:
                return self._get_default_config()
        except Exception as e:
            logging.warning(f"Failed to load config from {self.config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "memory_system": {
                "version": "3.0",
                "advanced_features": {
                    "file_watching": True,
                    "dynamic_context": True,
                    "relevance_scoring": True,
                    "knowledge_persistence": True,
                    "auto_compression": True
                }
            },
            "services": {
                "file_watcher": {"enabled": True, "auto_restart": True},
                "context_injector": {"enabled": True, "auto_restart": True},
                "knowledge_manager": {"enabled": True, "auto_restart": True},
                "performance_monitor": {"enabled": True, "auto_restart": True}
            },
            "performance": {
                "async_processing": True,
                "thread_pool_size": 4,
                "queue_max_size": 1000,
                "background_sync_interval": 300,
                "cache_size_mb": 100
            },
            "monitoring": {
                "enabled": True,
                "heartbeat_interval": 30,
                "health_check_interval": 60,
                "service_timeout": 120
            }
        }
    
    async def start_services(self):
        """Start all memory system services."""
        try:
            logging.info("Starting Prsist Memory System services...")
            self.running = True
            
            # Start core services
            await self._start_core_services()
            
            # Start optional services based on configuration
            if self.config.get("memory_system", {}).get("advanced_features", {}).get("file_watching"):
                await self.start_file_watcher()
            
            if self.config.get("services", {}).get("context_injector", {}).get("enabled"):
                await self.start_context_service()
            
            if self.config.get("services", {}).get("knowledge_manager", {}).get("enabled"):
                await self.start_knowledge_service()
            
            if self.config.get("services", {}).get("performance_monitor", {}).get("enabled"):
                await self.start_performance_monitor()
            
            # Start background processing
            await self.start_background_processor()
            
            # Start monitoring
            await self.start_monitoring()
            
            logging.info("All memory system services started successfully")
            
        except Exception as e:
            logging.error(f"Failed to start services: {e}")
            await self.stop_services()
            raise
    
    async def _start_core_services(self):
        """Start core memory system services."""
        try:
            # Register core service statuses
            self.services["memory_manager"] = ServiceStatus("memory_manager")
            self.services["context_manager"] = ServiceStatus("context_manager")
            self.services["relevance_scorer"] = ServiceStatus("relevance_scorer")
            self.services["change_analyzer"] = ServiceStatus("change_analyzer")
            
            # Mark as running (these are in-process services)
            for service_name in ["memory_manager", "context_manager", "relevance_scorer", "change_analyzer"]:
                self.services[service_name].status = "running"
                self.services[service_name].start_time = datetime.now()
                self.services[service_name].update_heartbeat()
            
            logging.info("Core services initialized")
            
        except Exception as e:
            logging.error(f"Failed to start core services: {e}")
            raise
    
    async def start_file_watcher(self):
        """Start Node.js file watcher service."""
        try:
            logging.info("Starting file watcher service...")
            
            service_status = ServiceStatus("file_watcher")
            service_status.status = "starting"
            self.services["file_watcher"] = service_status
            
            # Check if Node.js is available
            try:
                subprocess.run(["node", "--version"], check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise RuntimeError("Node.js not found - file watcher requires Node.js")
            
            # Start file watcher process
            watcher_script = Path(".prsist/watchers/file-watcher.js")
            if not watcher_script.exists():
                raise RuntimeError(f"File watcher script not found: {watcher_script}")
            
            process = await asyncio.create_subprocess_exec(
                'node', str(watcher_script),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path.cwd()
            )
            
            service_status.process = process
            service_status.status = "running"
            service_status.start_time = datetime.now()
            service_status.update_heartbeat()
            
            # Monitor process output
            asyncio.create_task(self._monitor_process_output(service_status, "file_watcher"))
            
            logging.info("File watcher service started successfully")
            
        except Exception as e:
            logging.error(f"Failed to start file watcher: {e}")
            if "file_watcher" in self.services:
                self.services["file_watcher"].status = "failed"
                self.services["file_watcher"].error_count += 1
    
    async def start_context_service(self):
        """Start context injection service."""
        try:
            logging.info("Starting context injection service...")
            
            service_status = ServiceStatus("context_injector")
            service_status.status = "running"
            service_status.start_time = datetime.now()
            self.services["context_injector"] = service_status
            
            # Start background context management task
            task = asyncio.create_task(self._context_service_loop())
            self.background_tasks.append(task)
            
            service_status.update_heartbeat()
            logging.info("Context injection service started successfully")
            
        except Exception as e:
            logging.error(f"Failed to start context service: {e}")
            if "context_injector" in self.services:
                self.services["context_injector"].status = "failed"
                self.services["context_injector"].error_count += 1
    
    async def start_knowledge_service(self):
        """Start knowledge persistence service."""
        try:
            logging.info("Starting knowledge persistence service...")
            
            service_status = ServiceStatus("knowledge_manager")
            service_status.status = "running"
            service_status.start_time = datetime.now()
            self.services["knowledge_manager"] = service_status
            
            # Start background knowledge management task
            task = asyncio.create_task(self._knowledge_service_loop())
            self.background_tasks.append(task)
            
            service_status.update_heartbeat()
            logging.info("Knowledge persistence service started successfully")
            
        except Exception as e:
            logging.error(f"Failed to start knowledge service: {e}")
            if "knowledge_manager" in self.services:
                self.services["knowledge_manager"].status = "failed"
                self.services["knowledge_manager"].error_count += 1
    
    async def start_performance_monitor(self):
        """Start performance monitoring service."""
        try:
            logging.info("Starting performance monitoring service...")
            
            service_status = ServiceStatus("performance_monitor")
            service_status.status = "running"
            service_status.start_time = datetime.now()
            self.services["performance_monitor"] = service_status
            
            # Start background performance monitoring task
            task = asyncio.create_task(self._performance_monitor_loop())
            self.background_tasks.append(task)
            
            service_status.update_heartbeat()
            logging.info("Performance monitoring service started successfully")
            
        except Exception as e:
            logging.error(f"Failed to start performance monitor: {e}")
            if "performance_monitor" in self.services:
                self.services["performance_monitor"].status = "failed"
                self.services["performance_monitor"].error_count += 1
    
    async def start_background_processor(self):
        """Start background task processing."""
        try:
            logging.info("Starting background processor...")
            
            service_status = ServiceStatus("background_processor")
            service_status.status = "running"
            service_status.start_time = datetime.now()
            self.services["background_processor"] = service_status
            
            # Start background processing task
            task = asyncio.create_task(self.process_memory_update_queue())
            self.background_tasks.append(task)
            
            service_status.update_heartbeat()
            logging.info("Background processor started successfully")
            
        except Exception as e:
            logging.error(f"Failed to start background processor: {e}")
            if "background_processor" in self.services:
                self.services["background_processor"].status = "failed"
                self.services["background_processor"].error_count += 1
    
    async def start_monitoring(self):
        """Start service monitoring."""
        try:
            logging.info("Starting service monitoring...")
            
            service_status = ServiceStatus("service_monitor")
            service_status.status = "running"
            service_status.start_time = datetime.now()
            self.services["service_monitor"] = service_status
            
            # Start monitoring task
            task = asyncio.create_task(self._service_monitor_loop())
            self.background_tasks.append(task)
            
            service_status.update_heartbeat()
            logging.info("Service monitoring started successfully")
            
        except Exception as e:
            logging.error(f"Failed to start monitoring: {e}")
            if "service_monitor" in self.services:
                self.services["service_monitor"].status = "failed"
                self.services["service_monitor"].error_count += 1
    
    async def _context_service_loop(self):
        """Background context management loop."""
        try:
            while self.running:
                # Update context manager heartbeat
                if "context_injector" in self.services:
                    self.services["context_injector"].update_heartbeat()
                
                # Perform context maintenance tasks
                await self._perform_context_maintenance()
                
                # Sleep for configured interval
                interval = self.config.get("performance", {}).get("background_sync_interval", 300)
                await asyncio.sleep(interval)
                
        except asyncio.CancelledError:
            logging.info("Context service loop cancelled")
        except Exception as e:
            logging.error(f"Context service loop error: {e}")
            if "context_injector" in self.services:
                self.services["context_injector"].status = "failed"
                self.services["context_injector"].error_count += 1
    
    async def _knowledge_service_loop(self):
        """Background knowledge management loop."""
        try:
            while self.running:
                # Update knowledge manager heartbeat
                if "knowledge_manager" in self.services:
                    self.services["knowledge_manager"].update_heartbeat()
                
                # Perform knowledge maintenance tasks
                await self._perform_knowledge_maintenance()
                
                # Sleep for configured interval
                interval = self.config.get("performance", {}).get("background_sync_interval", 300)
                await asyncio.sleep(interval)
                
        except asyncio.CancelledError:
            logging.info("Knowledge service loop cancelled")
        except Exception as e:
            logging.error(f"Knowledge service loop error: {e}")
            if "knowledge_manager" in self.services:
                self.services["knowledge_manager"].status = "failed"
                self.services["knowledge_manager"].error_count += 1
    
    async def _performance_monitor_loop(self):
        """Background performance monitoring loop."""
        try:
            while self.running:
                # Update performance monitor heartbeat
                if "performance_monitor" in self.services:
                    self.services["performance_monitor"].update_heartbeat()
                
                # Collect performance metrics
                await self._collect_performance_metrics()
                
                # Sleep for configured interval
                monitor_interval = self.config.get("monitoring", {}).get("health_check_interval", 60)
                await asyncio.sleep(monitor_interval)
                
        except asyncio.CancelledError:
            logging.info("Performance monitor loop cancelled")
        except Exception as e:
            logging.error(f"Performance monitor loop error: {e}")
            if "performance_monitor" in self.services:
                self.services["performance_monitor"].status = "failed"
                self.services["performance_monitor"].error_count += 1
    
    async def _service_monitor_loop(self):
        """Background service monitoring loop."""
        try:
            while self.running:
                # Update service monitor heartbeat
                if "service_monitor" in self.services:
                    self.services["service_monitor"].update_heartbeat()
                
                # Check service health
                await self._check_service_health()
                
                # Sleep for configured interval
                monitor_interval = self.config.get("monitoring", {}).get("health_check_interval", 60)
                await asyncio.sleep(monitor_interval)
                
        except asyncio.CancelledError:
            logging.info("Service monitor loop cancelled")
        except Exception as e:
            logging.error(f"Service monitor loop error: {e}")
            if "service_monitor" in self.services:
                self.services["service_monitor"].status = "failed"
                self.services["service_monitor"].error_count += 1
    
    async def _monitor_process_output(self, service_status: ServiceStatus, service_name: str):
        """Monitor process output for external services."""
        try:
            process = service_status.process
            
            async for line in process.stdout:
                line_str = line.decode().strip()
                if line_str:
                    logging.debug(f"{service_name}: {line_str}")
                    service_status.update_heartbeat()
            
            # Process has ended
            return_code = await process.wait()
            if return_code != 0:
                service_status.status = "failed"
                service_status.error_count += 1
                logging.error(f"{service_name} process exited with code {return_code}")
            else:
                service_status.status = "stopped"
                logging.info(f"{service_name} process exited normally")
                
        except Exception as e:
            logging.error(f"Error monitoring {service_name} output: {e}")
            service_status.status = "failed"
            service_status.error_count += 1
    
    async def _perform_context_maintenance(self):
        """Perform context maintenance tasks."""
        try:
            # Clean up expired context cache
            self.context_manager.invalidate_cache()
            
            # Compress contexts that are approaching limits
            active_sessions = self.memory_db.get_active_sessions()
            for session in active_sessions:
                session_id = session['id']
                try:
                    # Check if context compression is needed
                    current_context = self.context_manager.get_base_context(session_id)
                    if self.context_manager.is_context_full(current_context):
                        compressed = self.context_manager.compression_engine.auto_compact(current_context)
                        
                        # Store compressed context as snapshot
                        self.memory_db.create_context_snapshot(
                            session_id=session_id,
                            snapshot_type="auto_compression",
                            context_data=compressed,
                            compression_level=1
                        )
                        
                        logging.info(f"Auto-compressed context for session {session_id}")
                except Exception as e:
                    logging.error(f"Failed to maintain context for session {session_id}: {e}")
            
        except Exception as e:
            logging.error(f"Failed to perform context maintenance: {e}")
    
    async def _perform_knowledge_maintenance(self):
        """Perform knowledge management maintenance tasks."""
        try:
            # Clean up old knowledge entries
            retention_days = self.config.get("knowledge_persistence", {}).get("knowledge_decay_days", 90)
            self.knowledge_manager.cleanup_old_knowledge(retention_days)
            
            # Process any pending session knowledge
            active_sessions = self.memory_db.get_active_sessions()
            for session in active_sessions:
                try:
                    # Check if session has accumulated enough data for knowledge extraction
                    session_id = session['id']
                    session_age_hours = (datetime.now() - datetime.fromisoformat(session['created_at'])).total_seconds() / 3600
                    
                    if session_age_hours > 2:  # Process sessions older than 2 hours
                        # Get session data for knowledge extraction
                        session_data = {
                            'session_id': session_id,
                            'context_data': session.get('context_data', {}),
                            'tool_usage': self.memory_db.get_session_tool_usage(session_id),
                            'created_at': session['created_at']
                        }
                        
                        # Extract and persist knowledge
                        self.knowledge_manager.persist_session_knowledge(session_data)
                        
                except Exception as e:
                    logging.error(f"Failed to process knowledge for session {session['id']}: {e}")
            
        except Exception as e:
            logging.error(f"Failed to perform knowledge maintenance: {e}")
    
    async def _collect_performance_metrics(self):
        """Collect system performance metrics."""
        try:
            # Collect memory usage metrics
            import psutil
            import sys
            
            # Process memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            
            self.memory_db.record_performance_metric(
                metric_type="memory_usage_mb",
                metric_value=memory_info.rss / 1024 / 1024,
                measurement_context={"component": "orchestrator"}
            )
            
            # Queue size metrics
            queue_size = self.task_queue.qsize() if hasattr(self.task_queue, 'qsize') else 0
            self.memory_db.record_performance_metric(
                metric_type="queue_size",
                metric_value=queue_size,
                measurement_context={"component": "task_queue"}
            )
            
            # Service count metrics
            running_services = len([s for s in self.services.values() if s.status == "running"])
            self.memory_db.record_performance_metric(
                metric_type="running_services",
                metric_value=running_services,
                measurement_context={"total_services": len(self.services)}
            )
            
            # Context cache metrics
            cache_size = len(self.context_manager.context_cache)
            self.memory_db.record_performance_metric(
                metric_type="context_cache_size",
                metric_value=cache_size,
                measurement_context={"component": "context_manager"}
            )
            
        except Exception as e:
            logging.error(f"Failed to collect performance metrics: {e}")
    
    async def _check_service_health(self):
        """Check health of all services."""
        try:
            timeout_seconds = self.config.get("monitoring", {}).get("service_timeout", 120)
            
            for service_name, service_status in self.services.items():
                if not service_status.is_healthy(timeout_seconds):
                    logging.warning(f"Service {service_name} appears unhealthy")
                    
                    # Attempt to restart if auto-restart is enabled
                    service_config = self.config.get("services", {}).get(service_name, {})
                    if service_config.get("auto_restart", False):
                        logging.info(f"Attempting to restart service {service_name}")
                        await self._restart_service(service_name)
            
        except Exception as e:
            logging.error(f"Failed to check service health: {e}")
    
    async def _restart_service(self, service_name: str):
        """Restart a specific service."""
        try:
            logging.info(f"Restarting service: {service_name}")
            
            # Stop the service if it's running
            await self._stop_service(service_name)
            
            # Wait a moment
            await asyncio.sleep(2)
            
            # Start the service based on its type
            if service_name == "file_watcher":
                await self.start_file_watcher()
            elif service_name == "context_injector":
                await self.start_context_service()
            elif service_name == "knowledge_manager":
                await self.start_knowledge_service()
            elif service_name == "performance_monitor":
                await self.start_performance_monitor()
            else:
                logging.warning(f"Don't know how to restart service: {service_name}")
            
        except Exception as e:
            logging.error(f"Failed to restart service {service_name}: {e}")
    
    async def _stop_service(self, service_name: str):
        """Stop a specific service."""
        try:
            if service_name not in self.services:
                return
            
            service_status = self.services[service_name]
            service_status.status = "stopping"
            
            # Stop external processes
            if service_status.process:
                try:
                    service_status.process.terminate()
                    await asyncio.wait_for(service_status.process.wait(), timeout=10)
                except asyncio.TimeoutError:
                    service_status.process.kill()
                    await service_status.process.wait()
                except Exception as e:
                    logging.error(f"Error stopping process for {service_name}: {e}")
            
            service_status.status = "stopped"
            service_status.process = None
            
        except Exception as e:
            logging.error(f"Failed to stop service {service_name}: {e}")
    
    async def process_memory_update_queue(self):
        """Process queued memory updates."""
        try:
            while self.running:
                try:
                    # Update background processor heartbeat
                    if "background_processor" in self.services:
                        self.services["background_processor"].update_heartbeat()
                    
                    # Wait for a task with timeout
                    task = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=5.0
                    )
                    
                    # Process task in thread pool to avoid blocking
                    await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        self.process_memory_task,
                        task
                    )
                    
                    self.task_queue.task_done()
                    
                except asyncio.TimeoutError:
                    continue  # No tasks available, continue monitoring
                except Exception as e:
                    logging.error(f"Error processing memory task: {e}")
                    
        except asyncio.CancelledError:
            logging.info("Memory update queue processor cancelled")
        except Exception as e:
            logging.error(f"Memory update queue processor error: {e}")
    
    def process_memory_task(self, task: Dict[str, Any]):
        """Process a single memory task (runs in thread pool)."""
        try:
            task_type = task.get("type", "unknown")
            
            if task_type == "file_change":
                self._process_file_change_task(task)
            elif task_type == "context_update":
                self._process_context_update_task(task)
            elif task_type == "relevance_update":
                self._process_relevance_update_task(task)
            elif task_type == "knowledge_extraction":
                self._process_knowledge_extraction_task(task)
            else:
                logging.warning(f"Unknown task type: {task_type}")
                
        except Exception as e:
            logging.error(f"Failed to process memory task: {e}")
    
    def _process_file_change_task(self, task: Dict[str, Any]):
        """Process file change task."""
        try:
            file_path = task.get("file_path")
            change_type = task.get("change_type")
            content_diff = task.get("content_diff")
            
            # Analyze change impact
            impact_analysis = self.change_analyzer.analyze_change_impact(
                file_path, change_type, content_diff
            )
            
            # Update file relevance if needed
            if impact_analysis.get("memory_implications", {}).get("relevance_update_required"):
                relevance_score = impact_analysis.get("overall_impact", 0.5)
                self.memory_db.update_file_relevance(
                    file_path=file_path,
                    relevance_score=relevance_score,
                    relevance_factors=[f"change_impact_{change_type}"],
                    expires_at=datetime.now() + timedelta(hours=24)
                )
            
        except Exception as e:
            logging.error(f"Failed to process file change task: {e}")
    
    def _process_context_update_task(self, task: Dict[str, Any]):
        """Process context update task."""
        try:
            session_id = task.get("session_id")
            update_type = task.get("update_type", "refresh")
            
            if update_type == "refresh":
                # Invalidate context cache for session
                self.context_manager.invalidate_cache(session_id)
            elif update_type == "compress":
                # Trigger context compression
                context = self.context_manager.get_base_context(session_id)
                compressed = self.context_manager.compression_engine.auto_compact(context)
                
                # Store compressed context
                self.memory_db.create_context_snapshot(
                    session_id=session_id,
                    snapshot_type="manual_compression",
                    context_data=compressed,
                    compression_level=1
                )
            
        except Exception as e:
            logging.error(f"Failed to process context update task: {e}")
    
    def _process_relevance_update_task(self, task: Dict[str, Any]):
        """Process relevance update task."""
        try:
            file_path = task.get("file_path")
            boost_score = task.get("boost_score", 0.1)
            duration_hours = task.get("duration_hours", 24)
            
            # Get current relevance
            current_relevances = self.memory_db.get_file_relevance(file_path=file_path)
            current_score = current_relevances[0]["relevance_score"] if current_relevances else 0.5
            
            # Apply boost
            new_score = min(1.0, current_score + boost_score)
            
            # Update relevance
            self.memory_db.update_file_relevance(
                file_path=file_path,
                relevance_score=new_score,
                relevance_factors=["relevance_boost"],
                expires_at=datetime.now() + timedelta(hours=duration_hours)
            )
            
        except Exception as e:
            logging.error(f"Failed to process relevance update task: {e}")
    
    def _process_knowledge_extraction_task(self, task: Dict[str, Any]):
        """Process knowledge extraction task."""
        try:
            session_data = task.get("session_data")
            
            if session_data:
                # Extract and persist knowledge
                self.knowledge_manager.persist_session_knowledge(session_data)
            
        except Exception as e:
            logging.error(f"Failed to process knowledge extraction task: {e}")
    
    async def queue_task(self, task: Dict[str, Any]):
        """Queue a task for background processing."""
        try:
            max_queue_size = self.config.get("performance", {}).get("queue_max_size", 1000)
            
            if self.task_queue.qsize() >= max_queue_size:
                logging.warning("Task queue is full, dropping oldest task")
                try:
                    self.task_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            
            await self.task_queue.put(task)
            
        except Exception as e:
            logging.error(f"Failed to queue task: {e}")
    
    async def stop_services(self):
        """Stop all services gracefully."""
        try:
            logging.info("Stopping Prsist Memory System services...")
            self.running = False
            
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
            
            # Wait for background tasks to complete
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Stop external services
            for service_name, service_status in self.services.items():
                await self._stop_service(service_name)
            
            # Shutdown thread pool
            self.executor.shutdown(wait=True)
            
            logging.info("All services stopped successfully")
            
        except Exception as e:
            logging.error(f"Error stopping services: {e}")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services."""
        try:
            status = {
                "orchestrator_running": self.running,
                "total_services": len(self.services),
                "services": {},
                "queue_size": self.task_queue.qsize() if hasattr(self.task_queue, 'qsize') else 0,
                "timestamp": datetime.now().isoformat()
            }
            
            for service_name, service_status in self.services.items():
                status["services"][service_name] = {
                    "status": service_status.status,
                    "start_time": service_status.start_time.isoformat() if service_status.start_time else None,
                    "last_heartbeat": service_status.last_heartbeat.isoformat() if service_status.last_heartbeat else None,
                    "error_count": service_status.error_count,
                    "healthy": service_status.is_healthy(),
                    "metadata": service_status.metadata
                }
            
            return status
            
        except Exception as e:
            logging.error(f"Failed to get service status: {e}")
            return {"error": str(e)}
    
    def handle_shutdown_signal(self, signum, frame):
        """Handle shutdown signals."""
        logging.info(f"Received shutdown signal {signum}")
        if self.running:
            asyncio.create_task(self.stop_services())


async def main():
    """Main entry point for service orchestrator."""
    orchestrator = MemoryServiceOrchestrator()
    
    # Set up signal handlers
    import signal
    signal.signal(signal.SIGINT, orchestrator.handle_shutdown_signal)
    signal.signal(signal.SIGTERM, orchestrator.handle_shutdown_signal)
    
    try:
        await orchestrator.start_services()
        
        # Keep running until stopped
        while orchestrator.running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt")
    except Exception as e:
        logging.error(f"Orchestrator error: {e}")
    finally:
        await orchestrator.stop_services()


if __name__ == "__main__":
    asyncio.run(main())