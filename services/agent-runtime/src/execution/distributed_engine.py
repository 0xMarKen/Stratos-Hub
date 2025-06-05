"""
Distributed Execution Engine

Handles distributed execution of AI agents across multiple nodes,
with load balancing, fault tolerance, and automatic scaling.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable
from datetime import datetime, timedelta
import pickle
import zlib

import redis.asyncio as redis
import aiohttp
import consul.aio
from kubernetes import client, config
import docker
import psutil

from ..core.config import get_settings
from ..core.logging import get_logger
from ..monitoring.metrics import MetricsCollector
from ..models.model_registry import ModelRegistry
from ..llm.llm_integration import LLMIntegration

settings = get_settings()
logger = get_logger(__name__)


class NodeStatus(Enum):
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


class ExecutionMode(Enum):
    LOCAL = "local"
    DISTRIBUTED = "distributed"
    KUBERNETES = "kubernetes"
    DOCKER_SWARM = "docker_swarm"


@dataclass
class NodeInfo:
    node_id: str
    hostname: str
    ip_address: str
    port: int
    status: NodeStatus
    capabilities: List[str]
    resource_capacity: Dict[str, Any]
    resource_usage: Dict[str, Any]
    active_executions: int
    max_executions: int
    last_heartbeat: datetime
    version: str
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExecutionTask:
    task_id: str
    agent_id: str
    model_type: str
    priority: int
    input_data: Any
    resource_requirements: Dict[str, Any]
    timeout: int
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    assigned_node: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    task_id: str
    success: bool
    result: Any
    execution_time: float
    node_id: str
    resource_usage: Dict[str, Any]
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class NodeManager:
    """Manages distributed nodes and their health"""
    
    def __init__(self, redis_client: redis.Redis, consul_client):
        self.redis = redis_client
        self.consul = consul_client
        self.nodes: Dict[str, NodeInfo] = {}
        self.node_selector = NodeSelector()
        self.health_check_interval = 30
        self.heartbeat_timeout = 60
        
    async def register_node(self, node_info: NodeInfo) -> bool:
        """Register a new execution node"""
        try:
            # Store in local cache
            self.nodes[node_info.node_id] = node_info
            
            # Store in Redis for cluster coordination
            await self.redis.hset(
                "nodes",
                node_info.node_id,
                json.dumps({
                    "hostname": node_info.hostname,
                    "ip_address": node_info.ip_address,
                    "port": node_info.port,
                    "status": node_info.status.value,
                    "capabilities": node_info.capabilities,
                    "resource_capacity": node_info.resource_capacity,
                    "last_heartbeat": node_info.last_heartbeat.isoformat(),
                    "version": node_info.version,
                    "tags": node_info.tags
                })
            )
            
            # Register with Consul for service discovery
            await self.consul.agent.service.register(
                name="stratoshub-agent-node",
                service_id=node_info.node_id,
                address=node_info.ip_address,
                port=node_info.port,
                tags=list(node_info.capabilities) + [f"version:{node_info.version}"],
                check=consul.Check.http(
                    url=f"http://{node_info.ip_address}:{node_info.port}/health",
                    interval="30s",
                    timeout="10s"
                )
            )
            
            logger.info(f"Registered node {node_info.node_id} at {node_info.ip_address}:{node_info.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register node {node_info.node_id}: {e}")
            return False
    
    async def unregister_node(self, node_id: str) -> bool:
        """Unregister a node"""
        try:
            # Remove from local cache
            if node_id in self.nodes:
                del self.nodes[node_id]
            
            # Remove from Redis
            await self.redis.hdel("nodes", node_id)
            
            # Deregister from Consul
            await self.consul.agent.service.deregister(node_id)
            
            logger.info(f"Unregistered node {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister node {node_id}: {e}")
            return False
    
    async def update_node_status(self, node_id: str, status: NodeStatus, 
                               resource_usage: Optional[Dict[str, Any]] = None) -> bool:
        """Update node status and resource usage"""
        try:
            if node_id not in self.nodes:
                return False
            
            node = self.nodes[node_id]
            node.status = status
            node.last_heartbeat = datetime.now()
            
            if resource_usage:
                node.resource_usage = resource_usage
            
            # Update in Redis
            await self.redis.hset(
                "nodes",
                node_id,
                json.dumps({
                    "hostname": node.hostname,
                    "ip_address": node.ip_address,
                    "port": node.port,
                    "status": status.value,
                    "capabilities": node.capabilities,
                    "resource_capacity": node.resource_capacity,
                    "resource_usage": node.resource_usage,
                    "active_executions": node.active_executions,
                    "last_heartbeat": node.last_heartbeat.isoformat(),
                    "version": node.version,
                    "tags": node.tags
                })
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update node status for {node_id}: {e}")
            return False
    
    async def get_healthy_nodes(self, capabilities: Optional[List[str]] = None) -> List[NodeInfo]:
        """Get list of healthy nodes with optional capability filtering"""
        healthy_nodes = []
        
        for node in self.nodes.values():
            # Check if node is healthy
            if node.status not in [NodeStatus.HEALTHY, NodeStatus.BUSY]:
                continue
            
            # Check heartbeat timeout
            if datetime.now() - node.last_heartbeat > timedelta(seconds=self.heartbeat_timeout):
                await self.update_node_status(node.node_id, NodeStatus.UNHEALTHY)
                continue
            
            # Check capabilities
            if capabilities:
                if not all(cap in node.capabilities for cap in capabilities):
                    continue
            
            healthy_nodes.append(node)
        
        return healthy_nodes
    
    async def select_node(self, task: ExecutionTask) -> Optional[NodeInfo]:
        """Select the best node for a task"""
        # Get nodes that can handle this task
        required_capabilities = [task.model_type]
        healthy_nodes = await self.get_healthy_nodes(required_capabilities)
        
        if not healthy_nodes:
            return None
        
        # Use node selector to pick the best node
        return self.node_selector.select_best_node(healthy_nodes, task)
    
    async def start_health_monitor(self):
        """Start background health monitoring"""
        while True:
            try:
                await self.check_node_health()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5)
    
    async def check_node_health(self):
        """Check health of all registered nodes"""
        for node_id, node in self.nodes.items():
            try:
                # Check if node is responsive
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                    async with session.get(f"http://{node.ip_address}:{node.port}/health") as response:
                        if response.status == 200:
                            health_data = await response.json()
                            await self.update_node_status(
                                node_id,
                                NodeStatus(health_data.get('status', 'healthy')),
                                health_data.get('resource_usage')
                            )
                        else:
                            await self.update_node_status(node_id, NodeStatus.UNHEALTHY)
                            
            except Exception as e:
                logger.warning(f"Health check failed for node {node_id}: {e}")
                await self.update_node_status(node_id, NodeStatus.UNHEALTHY)


class NodeSelector:
    """Implements node selection algorithms"""
    
    def select_best_node(self, nodes: List[NodeInfo], task: ExecutionTask) -> Optional[NodeInfo]:
        """Select the best node for a task using weighted scoring"""
        if not nodes:
            return None
        
        scored_nodes = []
        
        for node in nodes:
            score = self.calculate_node_score(node, task)
            scored_nodes.append((score, node))
        
        # Sort by score (highest first)
        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        
        return scored_nodes[0][1]
    
    def calculate_node_score(self, node: NodeInfo, task: ExecutionTask) -> float:
        """Calculate a score for how suitable a node is for a task"""
        score = 100.0
        
        # Penalize based on current load
        load_ratio = node.active_executions / max(node.max_executions, 1)
        score -= load_ratio * 30
        
        # Penalize based on resource usage
        if node.resource_usage:
            cpu_usage = node.resource_usage.get('cpu_percent', 0)
            memory_usage = node.resource_usage.get('memory_percent', 0)
            score -= (cpu_usage + memory_usage) / 2 * 0.2
        
        # Bonus for having required resources available
        required_memory = task.resource_requirements.get('memory_mb', 0)
        available_memory = node.resource_capacity.get('memory_mb', 0) - \
                          node.resource_usage.get('memory_mb', 0)
        
        if required_memory <= available_memory:
            score += 20
        else:
            score -= 50  # Heavy penalty if not enough memory
        
        # Bonus for priority tasks
        if task.priority > 5:
            score += task.priority * 2
        
        # Bonus for node being lightly loaded
        if load_ratio < 0.5:
            score += 10
        
        return max(score, 0)


class TaskQueue:
    """Manages task queuing and distribution"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.pending_queue = "tasks:pending"
        self.processing_queue = "tasks:processing"
        self.completed_queue = "tasks:completed"
        self.failed_queue = "tasks:failed"
        
    async def enqueue_task(self, task: ExecutionTask) -> bool:
        """Add a task to the pending queue"""
        try:
            task_data = {
                "task_id": task.task_id,
                "agent_id": task.agent_id,
                "model_type": task.model_type,
                "priority": task.priority,
                "input_data": self.serialize_data(task.input_data),
                "resource_requirements": task.resource_requirements,
                "timeout": task.timeout,
                "retry_count": task.retry_count,
                "max_retries": task.max_retries,
                "created_at": task.created_at.isoformat(),
                "metadata": task.metadata
            }
            
            # Add to priority queue (Redis sorted set)
            await self.redis.zadd(
                self.pending_queue,
                {json.dumps(task_data): task.priority}
            )
            
            logger.debug(f"Enqueued task {task.task_id} with priority {task.priority}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enqueue task {task.task_id}: {e}")
            return False
    
    async def dequeue_task(self) -> Optional[ExecutionTask]:
        """Get the highest priority task from the queue"""
        try:
            # Get highest priority task (ZREVRANGE gets highest scores first)
            result = await self.redis.zrevrange(
                self.pending_queue, 0, 0, withscores=True
            )
            
            if not result:
                return None
            
            task_json, priority = result[0]
            task_data = json.loads(task_json)
            
            # Remove from pending queue
            await self.redis.zrem(self.pending_queue, task_json)
            
            # Create task object
            task = ExecutionTask(
                task_id=task_data["task_id"],
                agent_id=task_data["agent_id"],
                model_type=task_data["model_type"],
                priority=int(priority),
                input_data=self.deserialize_data(task_data["input_data"]),
                resource_requirements=task_data["resource_requirements"],
                timeout=task_data["timeout"],
                retry_count=task_data["retry_count"],
                max_retries=task_data["max_retries"],
                created_at=datetime.fromisoformat(task_data["created_at"]),
                metadata=task_data.get("metadata", {})
            )
            
            # Add to processing queue
            await self.redis.hset(
                self.processing_queue,
                task.task_id,
                json.dumps(task_data)
            )
            
            return task
            
        except Exception as e:
            logger.error(f"Failed to dequeue task: {e}")
            return None
    
    async def complete_task(self, result: ExecutionResult) -> bool:
        """Mark a task as completed"""
        try:
            # Remove from processing queue
            await self.redis.hdel(self.processing_queue, result.task_id)
            
            # Add to completed queue
            result_data = {
                "task_id": result.task_id,
                "success": result.success,
                "result": self.serialize_data(result.result),
                "execution_time": result.execution_time,
                "node_id": result.node_id,
                "resource_usage": result.resource_usage,
                "error": result.error,
                "completed_at": datetime.now().isoformat(),
                "metadata": result.metadata
            }
            
            queue = self.completed_queue if result.success else self.failed_queue
            await self.redis.hset(queue, result.task_id, json.dumps(result_data))
            
            # Set expiration for completed tasks (7 days)
            await self.redis.expire(result.task_id, 7 * 24 * 3600)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete task {result.task_id}: {e}")
            return False
    
    async def retry_task(self, task: ExecutionTask) -> bool:
        """Retry a failed task"""
        try:
            if task.retry_count >= task.max_retries:
                # Move to failed queue
                await self.complete_task(ExecutionResult(
                    task_id=task.task_id,
                    success=False,
                    result=None,
                    execution_time=0,
                    node_id="",
                    resource_usage={},
                    error=f"Max retries ({task.max_retries}) exceeded"
                ))
                return False
            
            # Increment retry count and re-enqueue
            task.retry_count += 1
            task.assigned_node = None
            task.started_at = None
            
            await self.enqueue_task(task)
            return True
            
        except Exception as e:
            logger.error(f"Failed to retry task {task.task_id}: {e}")
            return False
    
    async def get_queue_stats(self) -> Dict[str, int]:
        """Get queue statistics"""
        try:
            pending = await self.redis.zcard(self.pending_queue)
            processing = await self.redis.hlen(self.processing_queue)
            completed = await self.redis.hlen(self.completed_queue)
            failed = await self.redis.hlen(self.failed_queue)
            
            return {
                "pending": pending,
                "processing": processing,
                "completed": completed,
                "failed": failed
            }
            
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {"pending": 0, "processing": 0, "completed": 0, "failed": 0}
    
    def serialize_data(self, data: Any) -> str:
        """Serialize data for storage"""
        try:
            return zlib.compress(pickle.dumps(data)).hex()
        except Exception as e:
            logger.warning(f"Failed to serialize data: {e}")
            return json.dumps(str(data))
    
    def deserialize_data(self, data: str) -> Any:
        """Deserialize data from storage"""
        try:
            return pickle.loads(zlib.decompress(bytes.fromhex(data)))
        except Exception as e:
            logger.warning(f"Failed to deserialize data: {e}")
            try:
                return json.loads(data)
            except:
                return data


class DistributedExecutionEngine:
    """Main distributed execution engine"""
    
    def __init__(self):
        self.node_manager: Optional[NodeManager] = None
        self.task_queue: Optional[TaskQueue] = None
        self.model_registry: Optional[ModelRegistry] = None
        self.llm_integration: Optional[LLMIntegration] = None
        self.metrics = MetricsCollector()
        self.redis: Optional[redis.Redis] = None
        self.consul = None
        self.execution_mode = ExecutionMode.LOCAL
        self.is_running = False
        self.worker_tasks: List[asyncio.Task] = []
        
    async def initialize(self, execution_mode: ExecutionMode = ExecutionMode.DISTRIBUTED):
        """Initialize the distributed execution engine"""
        try:
            self.execution_mode = execution_mode
            
            # Initialize Redis connection
            self.redis = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password,
                decode_responses=True
            )
            
            # Initialize Consul for service discovery
            self.consul = consul.aio.Consul(
                host=settings.consul_host,
                port=settings.consul_port
            )
            
            # Initialize components
            self.node_manager = NodeManager(self.redis, self.consul)
            self.task_queue = TaskQueue(self.redis)
            self.model_registry = ModelRegistry()
            self.llm_integration = LLMIntegration()
            
            # Register this node if in distributed mode
            if execution_mode != ExecutionMode.LOCAL:
                await self.register_current_node()
            
            logger.info(f"Distributed execution engine initialized in {execution_mode.value} mode")
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed execution engine: {e}")
            raise
    
    async def start(self):
        """Start the execution engine"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background tasks
        if self.execution_mode != ExecutionMode.LOCAL:
            self.worker_tasks.append(
                asyncio.create_task(self.node_manager.start_health_monitor())
            )
        
        # Start task processing workers
        worker_count = min(psutil.cpu_count(), 8)
        for i in range(worker_count):
            self.worker_tasks.append(
                asyncio.create_task(self.task_worker(f"worker-{i}"))
            )
        
        logger.info(f"Started distributed execution engine with {worker_count} workers")
    
    async def stop(self):
        """Stop the execution engine"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()
        
        # Unregister node
        if self.execution_mode != ExecutionMode.LOCAL and self.node_manager:
            current_node_id = self.get_current_node_id()
            await self.node_manager.unregister_node(current_node_id)
        
        # Close connections
        if self.redis:
            await self.redis.aclose()
        
        logger.info("Stopped distributed execution engine")
    
    async def submit_task(self, task: ExecutionTask) -> str:
        """Submit a task for execution"""
        if not self.task_queue:
            raise RuntimeError("Execution engine not initialized")
        
        # Generate task ID if not provided
        if not task.task_id:
            task.task_id = str(uuid.uuid4())
        
        # Enqueue task
        success = await self.task_queue.enqueue_task(task)
        
        if not success:
            raise RuntimeError(f"Failed to enqueue task {task.task_id}")
        
        # Record metrics
        self.metrics.record_task_submitted(task)
        
        return task.task_id
    
    async def get_task_result(self, task_id: str) -> Optional[ExecutionResult]:
        """Get the result of a completed task"""
        if not self.redis:
            return None
        
        # Check completed queue
        result_data = await self.redis.hget("tasks:completed", task_id)
        if result_data:
            data = json.loads(result_data)
            return ExecutionResult(
                task_id=data["task_id"],
                success=data["success"],
                result=self.task_queue.deserialize_data(data["result"]),
                execution_time=data["execution_time"],
                node_id=data["node_id"],
                resource_usage=data["resource_usage"],
                error=data.get("error"),
                metadata=data.get("metadata", {})
            )
        
        # Check failed queue
        result_data = await self.redis.hget("tasks:failed", task_id)
        if result_data:
            data = json.loads(result_data)
            return ExecutionResult(
                task_id=data["task_id"],
                success=False,
                result=None,
                execution_time=data.get("execution_time", 0),
                node_id=data.get("node_id", ""),
                resource_usage=data.get("resource_usage", {}),
                error=data.get("error", "Task failed"),
                metadata=data.get("metadata", {})
            )
        
        return None
    
    async def task_worker(self, worker_id: str):
        """Worker that processes tasks from the queue"""
        logger.info(f"Started task worker {worker_id}")
        
        while self.is_running:
            try:
                # Get next task
                task = await self.task_queue.dequeue_task()
                
                if not task:
                    await asyncio.sleep(1)  # No tasks available
                    continue
                
                logger.info(f"Worker {worker_id} processing task {task.task_id}")
                
                # Execute task
                result = await self.execute_task(task, worker_id)
                
                # Complete task
                await self.task_queue.complete_task(result)
                
                # Record metrics
                self.metrics.record_task_completed(result)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(5)  # Wait before retrying
        
        logger.info(f"Stopped task worker {worker_id}")
    
    async def execute_task(self, task: ExecutionTask, worker_id: str) -> ExecutionResult:
        """Execute a single task"""
        start_time = time.time()
        node_id = self.get_current_node_id()
        
        try:
            task.started_at = datetime.now()
            task.assigned_node = node_id
            
            # Load model if needed
            if not self.model_registry.is_model_loaded(task.agent_id):
                await self.model_registry.load_model(task.agent_id)
            
            # Execute the actual AI task
            if task.model_type in ["text-generation", "chat-completion"]:
                result = await self.execute_llm_task(task)
            else:
                result = await self.execute_ml_task(task)
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                task_id=task.task_id,
                success=True,
                result=result,
                execution_time=execution_time,
                node_id=node_id,
                resource_usage=self.get_current_resource_usage(),
                metadata={"worker_id": worker_id}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task {task.task_id} execution failed: {e}")
            
            return ExecutionResult(
                task_id=task.task_id,
                success=False,
                result=None,
                execution_time=execution_time,
                node_id=node_id,
                resource_usage=self.get_current_resource_usage(),
                error=str(e),
                metadata={"worker_id": worker_id}
            )
    
    async def execute_llm_task(self, task: ExecutionTask) -> Any:
        """Execute LLM-specific task"""
        # This would integrate with the LLM system
        return {"message": "LLM task completed", "input": task.input_data}
    
    async def execute_ml_task(self, task: ExecutionTask) -> Any:
        """Execute general ML task"""
        # This would integrate with various ML models
        return {"message": "ML task completed", "input": task.input_data}
    
    async def register_current_node(self):
        """Register the current node with the cluster"""
        node_info = NodeInfo(
            node_id=self.get_current_node_id(),
            hostname=settings.hostname,
            ip_address=self.get_local_ip(),
            port=settings.port,
            status=NodeStatus.HEALTHY,
            capabilities=["text-generation", "image-generation", "embeddings"],
            resource_capacity=self.get_resource_capacity(),
            resource_usage=self.get_current_resource_usage(),
            active_executions=0,
            max_executions=psutil.cpu_count() * 2,
            last_heartbeat=datetime.now(),
            version="1.0.0"
        )
        
        await self.node_manager.register_node(node_info)
    
    def get_current_node_id(self) -> str:
        """Get the current node ID"""
        import socket
        return f"{socket.gethostname()}-{settings.port}"
    
    def get_local_ip(self) -> str:
        """Get the local IP address"""
        import socket
        try:
            # Connect to a remote address to get local IP
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
            sock.close()
            return ip
        except:
            return "127.0.0.1"
    
    def get_resource_capacity(self) -> Dict[str, Any]:
        """Get system resource capacity"""
        memory = psutil.virtual_memory()
        return {
            "cpu_cores": psutil.cpu_count(),
            "memory_mb": memory.total // (1024 * 1024),
            "disk_gb": psutil.disk_usage('/').total // (1024 * 1024 * 1024)
        }
    
    def get_current_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage"""
        memory = psutil.virtual_memory()
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": memory.percent,
            "memory_mb": memory.used // (1024 * 1024),
            "disk_percent": psutil.disk_usage('/').percent
        }
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get overall cluster status"""
        if not self.node_manager or not self.task_queue:
            return {}
        
        nodes = await self.node_manager.get_healthy_nodes()
        queue_stats = await self.task_queue.get_queue_stats()
        
        return {
            "total_nodes": len(self.node_manager.nodes),
            "healthy_nodes": len(nodes),
            "total_capacity": sum(node.max_executions for node in nodes),
            "active_executions": sum(node.active_executions for node in nodes),
            "queue_stats": queue_stats,
            "execution_mode": self.execution_mode.value
        } 