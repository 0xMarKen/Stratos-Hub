"""
Performance Monitoring System

Comprehensive monitoring for AI agent execution performance,
resource utilization, and system health metrics.
"""

import asyncio
import time
import psutil
import GPUtil
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from collections import deque, defaultdict
import json
import logging
from decimal import Decimal

import numpy as np
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry
import matplotlib.pyplot as plt
import seaborn as sns

from ..core.config import get_settings
from ..core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    load_average: List[float]
    process_count: int
    thread_count: int
    gpu_metrics: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentMetrics:
    agent_id: str
    timestamp: datetime
    execution_count: int
    success_rate: float
    average_latency: float
    p95_latency: float
    p99_latency: float
    throughput_per_minute: float
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    cost_per_execution: Decimal
    total_revenue: Decimal


@dataclass
class ModelMetrics:
    model_id: str
    model_type: str
    timestamp: datetime
    inference_count: int
    average_inference_time: float
    tokens_per_second: float
    memory_usage_mb: float
    gpu_utilization: float
    batch_size: int
    queue_depth: int
    cache_hit_rate: float
    error_count: int


@dataclass
class PerformanceAlert:
    alert_id: str
    severity: str  # "info", "warning", "critical"
    component: str
    metric_name: str
    current_value: float
    threshold: float
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class MetricsCollector:
    """Collects and aggregates performance metrics"""
    
    def __init__(self):
        self.system_metrics_history: deque = deque(maxlen=1440)  # 24 hours of minute data
        self.agent_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1440))
        self.model_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1440))
        self.alerts: List[PerformanceAlert] = []
        self.is_collecting = False
        self.collection_interval = 60  # seconds
        self.prometheus_registry = CollectorRegistry()
        self.setup_prometheus_metrics()
        
    def setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        self.cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage', registry=self.prometheus_registry)
        self.memory_usage = Gauge('memory_usage_percent', 'Memory usage percentage', registry=self.prometheus_registry)
        self.disk_usage = Gauge('disk_usage_percent', 'Disk usage percentage', registry=self.prometheus_registry)
        
        self.agent_executions = Counter('agent_executions_total', 'Total agent executions', 
                                      ['agent_id', 'status'], registry=self.prometheus_registry)
        self.agent_latency = Histogram('agent_execution_duration_seconds', 'Agent execution duration',
                                     ['agent_id'], registry=self.prometheus_registry)
        self.agent_throughput = Gauge('agent_throughput_per_minute', 'Agent throughput per minute',
                                    ['agent_id'], registry=self.prometheus_registry)
        
        self.model_inference_time = Histogram('model_inference_duration_seconds', 'Model inference duration',
                                            ['model_id', 'model_type'], registry=self.prometheus_registry)
        self.model_tokens_per_second = Gauge('model_tokens_per_second', 'Model tokens per second',
                                           ['model_id'], registry=self.prometheus_registry)
        self.model_gpu_utilization = Gauge('model_gpu_utilization_percent', 'Model GPU utilization',
                                         ['model_id', 'gpu_id'], registry=self.prometheus_registry)
    
    async def start_collection(self):
        """Start metrics collection"""
        if self.is_collecting:
            return
            
        self.is_collecting = True
        logger.info("Starting performance metrics collection")
        
        # Start collection tasks
        asyncio.create_task(self.collect_system_metrics())
        asyncio.create_task(self.monitor_alerts())
        
    async def stop_collection(self):
        """Stop metrics collection"""
        self.is_collecting = False
        logger.info("Stopped performance metrics collection")
    
    async def collect_system_metrics(self):
        """Collect system-level metrics"""
        while self.is_collecting:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
                
                # Memory metrics
                memory = psutil.virtual_memory()
                memory_used_gb = memory.used / (1024**3)
                memory_available_gb = memory.available / (1024**3)
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                disk_usage_percent = (disk.used / disk.total) * 100
                disk_free_gb = disk.free / (1024**3)
                
                # Network metrics
                network = psutil.net_io_counters()
                
                # Process metrics
                process_count = len(psutil.pids())
                thread_count = sum(p.num_threads() for p in psutil.process_iter(['num_threads']))
                
                # GPU metrics
                gpu_metrics = []
                try:
                    gpus = GPUtil.getGPUs()
                    for gpu in gpus:
                        gpu_metrics.append({
                            'id': gpu.id,
                            'name': gpu.name,
                            'load': gpu.load * 100,
                            'memory_util': gpu.memoryUtil * 100,
                            'memory_total': gpu.memoryTotal,
                            'memory_used': gpu.memoryUsed,
                            'temperature': gpu.temperature
                        })
                except Exception as e:
                    logger.debug(f"GPU metrics collection failed: {e}")
                
                # Create metrics object
                metrics = SystemMetrics(
                    timestamp=datetime.now(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_used_gb=memory_used_gb,
                    memory_available_gb=memory_available_gb,
                    disk_usage_percent=disk_usage_percent,
                    disk_free_gb=disk_free_gb,
                    network_bytes_sent=network.bytes_sent,
                    network_bytes_recv=network.bytes_recv,
                    load_average=load_avg,
                    process_count=process_count,
                    thread_count=thread_count,
                    gpu_metrics=gpu_metrics
                )
                
                # Store metrics
                self.system_metrics_history.append(metrics)
                
                # Update Prometheus metrics
                self.cpu_usage.set(cpu_percent)
                self.memory_usage.set(memory.percent)
                self.disk_usage.set(disk_usage_percent)
                
                # Check for alerts
                await self.check_system_alerts(metrics)
                
            except Exception as e:
                logger.error(f"System metrics collection error: {e}")
            
            await asyncio.sleep(self.collection_interval)
    
    def record_agent_execution(self, agent_id: str, duration: float, success: bool, 
                             memory_usage: float, cpu_usage: float, cost: Decimal):
        """Record agent execution metrics"""
        try:
            # Update Prometheus metrics
            status = "success" if success else "failure"
            self.agent_executions.labels(agent_id=agent_id, status=status).inc()
            self.agent_latency.labels(agent_id=agent_id).observe(duration)
            
            # Calculate throughput (simplified - would need more sophisticated calculation)
            # This is a placeholder for demonstration
            throughput = 60.0 / max(duration, 0.1)  # rough estimate
            self.agent_throughput.labels(agent_id=agent_id).set(throughput)
            
            logger.debug(f"Recorded execution for agent {agent_id}: {duration:.3f}s, success={success}")
            
        except Exception as e:
            logger.error(f"Failed to record agent execution metrics: {e}")
    
    def record_model_inference(self, model_id: str, model_type: str, inference_time: float,
                             tokens_per_second: float, memory_usage: float, gpu_utilization: float):
        """Record model inference metrics"""
        try:
            # Update Prometheus metrics
            self.model_inference_time.labels(model_id=model_id, model_type=model_type).observe(inference_time)
            self.model_tokens_per_second.labels(model_id=model_id).set(tokens_per_second)
            
            # GPU utilization (assuming GPU 0 for simplicity)
            self.model_gpu_utilization.labels(model_id=model_id, gpu_id="0").set(gpu_utilization)
            
            logger.debug(f"Recorded inference for model {model_id}: {inference_time:.3f}s, {tokens_per_second:.1f} tok/s")
            
        except Exception as e:
            logger.error(f"Failed to record model inference metrics: {e}")
    
    async def check_system_alerts(self, metrics: SystemMetrics):
        """Check for system performance alerts"""
        alerts = []
        
        # CPU alert
        if metrics.cpu_percent > 90:
            alerts.append(PerformanceAlert(
                alert_id=f"cpu_high_{int(time.time())}",
                severity="critical",
                component="system",
                metric_name="cpu_percent",
                current_value=metrics.cpu_percent,
                threshold=90,
                message=f"High CPU usage: {metrics.cpu_percent:.1f}%",
                timestamp=metrics.timestamp
            ))
        
        # Memory alert
        if metrics.memory_percent > 85:
            alerts.append(PerformanceAlert(
                alert_id=f"memory_high_{int(time.time())}",
                severity="warning" if metrics.memory_percent < 95 else "critical",
                component="system",
                metric_name="memory_percent",
                current_value=metrics.memory_percent,
                threshold=85,
                message=f"High memory usage: {metrics.memory_percent:.1f}%",
                timestamp=metrics.timestamp
            ))
        
        # Disk alert
        if metrics.disk_usage_percent > 90:
            alerts.append(PerformanceAlert(
                alert_id=f"disk_high_{int(time.time())}",
                severity="critical",
                component="system",
                metric_name="disk_usage_percent",
                current_value=metrics.disk_usage_percent,
                threshold=90,
                message=f"High disk usage: {metrics.disk_usage_percent:.1f}%",
                timestamp=metrics.timestamp
            ))
        
        # GPU alerts
        for gpu in metrics.gpu_metrics:
            if gpu['load'] > 95:
                alerts.append(PerformanceAlert(
                    alert_id=f"gpu_{gpu['id']}_high_{int(time.time())}",
                    severity="warning",
                    component="gpu",
                    metric_name="gpu_load",
                    current_value=gpu['load'],
                    threshold=95,
                    message=f"High GPU {gpu['id']} load: {gpu['load']:.1f}%",
                    timestamp=metrics.timestamp
                ))
        
        # Add new alerts
        for alert in alerts:
            self.alerts.append(alert)
            logger.warning(f"Performance Alert: {alert.message}")
    
    async def monitor_alerts(self):
        """Monitor and resolve alerts"""
        while self.is_collecting:
            try:
                current_time = datetime.now()
                
                # Auto-resolve old alerts (after 5 minutes)
                for alert in self.alerts:
                    if not alert.resolved and (current_time - alert.timestamp).total_seconds() > 300:
                        alert.resolved = True
                        alert.resolution_time = current_time
                        logger.info(f"Auto-resolved alert: {alert.alert_id}")
                
                # Clean up old resolved alerts (keep for 1 hour)
                self.alerts = [alert for alert in self.alerts 
                             if not alert.resolved or 
                             (current_time - alert.resolution_time).total_seconds() < 3600]
                
            except Exception as e:
                logger.error(f"Alert monitoring error: {e}")
            
            await asyncio.sleep(60)  # Check every minute
    
    def get_system_metrics_summary(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get system metrics summary for the specified duration"""
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        recent_metrics = [m for m in self.system_metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {}
        
        return {
            "duration_minutes": duration_minutes,
            "sample_count": len(recent_metrics),
            "cpu": {
                "average": np.mean([m.cpu_percent for m in recent_metrics]),
                "max": np.max([m.cpu_percent for m in recent_metrics]),
                "min": np.min([m.cpu_percent for m in recent_metrics]),
                "p95": np.percentile([m.cpu_percent for m in recent_metrics], 95)
            },
            "memory": {
                "average": np.mean([m.memory_percent for m in recent_metrics]),
                "max": np.max([m.memory_percent for m in recent_metrics]),
                "min": np.min([m.memory_percent for m in recent_metrics]),
                "average_used_gb": np.mean([m.memory_used_gb for m in recent_metrics])
            },
            "disk": {
                "average": np.mean([m.disk_usage_percent for m in recent_metrics]),
                "free_gb": recent_metrics[-1].disk_free_gb if recent_metrics else 0
            },
            "network": {
                "bytes_sent": recent_metrics[-1].network_bytes_sent if recent_metrics else 0,
                "bytes_recv": recent_metrics[-1].network_bytes_recv if recent_metrics else 0
            },
            "processes": {
                "count": recent_metrics[-1].process_count if recent_metrics else 0,
                "threads": recent_metrics[-1].thread_count if recent_metrics else 0
            }
        }
    
    def get_agent_performance_report(self, agent_id: str, duration_minutes: int = 60) -> Dict[str, Any]:
        """Generate performance report for a specific agent"""
        # This would aggregate agent-specific metrics
        # For now, return a placeholder structure
        return {
            "agent_id": agent_id,
            "duration_minutes": duration_minutes,
            "total_executions": 0,
            "success_rate": 0.0,
            "average_latency": 0.0,
            "p95_latency": 0.0,
            "throughput": 0.0,
            "error_rate": 0.0,
            "resource_usage": {
                "average_cpu": 0.0,
                "average_memory": 0.0,
                "peak_memory": 0.0
            },
            "cost_metrics": {
                "total_cost": Decimal('0'),
                "cost_per_execution": Decimal('0'),
                "total_revenue": Decimal('0')
            }
        }
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get all active (unresolved) alerts"""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        from prometheus_client import generate_latest
        return generate_latest(self.prometheus_registry).decode('utf-8')
    
    def generate_performance_chart(self, metric_type: str, duration_hours: int = 24) -> str:
        """Generate performance chart and return file path"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=duration_hours)
            recent_metrics = [m for m in self.system_metrics_history if m.timestamp >= cutoff_time]
            
            if not recent_metrics:
                return ""
            
            # Setup the plot
            plt.style.use('seaborn-v0_8')
            fig, ax = plt.subplots(figsize=(12, 6))
            
            timestamps = [m.timestamp for m in recent_metrics]
            
            if metric_type == "cpu":
                values = [m.cpu_percent for m in recent_metrics]
                ax.plot(timestamps, values, label='CPU Usage %', color='#FF6B6B', linewidth=2)
                ax.set_ylabel('CPU Usage (%)')
                ax.set_title(f'CPU Usage - Last {duration_hours} Hours')
                ax.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='Warning (80%)')
                ax.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='Critical (90%)')
                
            elif metric_type == "memory":
                values = [m.memory_percent for m in recent_metrics]
                ax.plot(timestamps, values, label='Memory Usage %', color='#4ECDC4', linewidth=2)
                ax.set_ylabel('Memory Usage (%)')
                ax.set_title(f'Memory Usage - Last {duration_hours} Hours')
                ax.axhline(y=85, color='orange', linestyle='--', alpha=0.7, label='Warning (85%)')
                ax.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='Critical (95%)')
                
            elif metric_type == "disk":
                values = [m.disk_usage_percent for m in recent_metrics]
                ax.plot(timestamps, values, label='Disk Usage %', color='#45B7D1', linewidth=2)
                ax.set_ylabel('Disk Usage (%)')
                ax.set_title(f'Disk Usage - Last {duration_hours} Hours')
                ax.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='Critical (90%)')
            
            # Format the plot
            ax.set_xlabel('Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save the chart
            chart_filename = f"performance_chart_{metric_type}_{int(time.time())}.png"
            chart_path = f"/tmp/{chart_filename}"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Failed to generate performance chart: {e}")
            return ""


class PerformanceProfiler:
    """Profile individual agent/model performance"""
    
    def __init__(self):
        self.active_profiles: Dict[str, Dict[str, Any]] = {}
        self.profile_history: List[Dict[str, Any]] = []
        
    def start_profiling(self, session_id: str, agent_id: str, model_id: Optional[str] = None):
        """Start profiling a session"""
        self.active_profiles[session_id] = {
            "agent_id": agent_id,
            "model_id": model_id,
            "start_time": time.time(),
            "start_memory": psutil.Process().memory_info().rss,
            "checkpoints": []
        }
    
    def add_checkpoint(self, session_id: str, checkpoint_name: str, metadata: Optional[Dict] = None):
        """Add a profiling checkpoint"""
        if session_id not in self.active_profiles:
            return
            
        current_time = time.time()
        current_memory = psutil.Process().memory_info().rss
        
        checkpoint = {
            "name": checkpoint_name,
            "timestamp": current_time,
            "memory": current_memory,
            "metadata": metadata or {}
        }
        
        self.active_profiles[session_id]["checkpoints"].append(checkpoint)
    
    def end_profiling(self, session_id: str, success: bool = True) -> Dict[str, Any]:
        """End profiling and return results"""
        if session_id not in self.active_profiles:
            return {}
            
        profile = self.active_profiles.pop(session_id)
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        # Calculate metrics
        total_duration = end_time - profile["start_time"]
        memory_delta = end_memory - profile["start_memory"]
        
        # Analyze checkpoints
        checkpoint_durations = []
        if profile["checkpoints"]:
            prev_time = profile["start_time"]
            for checkpoint in profile["checkpoints"]:
                duration = checkpoint["timestamp"] - prev_time
                checkpoint_durations.append({
                    "name": checkpoint["name"],
                    "duration": duration,
                    "memory": checkpoint["memory"]
                })
                prev_time = checkpoint["timestamp"]
        
        result = {
            "session_id": session_id,
            "agent_id": profile["agent_id"],
            "model_id": profile.get("model_id"),
            "success": success,
            "total_duration": total_duration,
            "memory_delta_mb": memory_delta / (1024 * 1024),
            "checkpoints": checkpoint_durations,
            "timestamp": datetime.now().isoformat()
        }
        
        self.profile_history.append(result)
        return result


# Global instances
metrics_collector = MetricsCollector()
performance_profiler = PerformanceProfiler() 