"""
Test suite for the StratosHub Execution Engine

Comprehensive tests covering agent execution, resource management,
performance monitoring, and error handling scenarios.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal

from src.execution.engine import ExecutionEngine
from src.execution.scheduler import TaskScheduler
from src.models.registry import ModelRegistry
from src.core.metrics import MetricsCollector
from src.core.database import Database
from src.execution.types import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    ResourceAllocation,
)


class TestExecutionEngine:
    """Test suite for ExecutionEngine class"""
    
    @pytest.fixture
    async def engine(self):
        """Create a test execution engine instance"""
        engine = ExecutionEngine()
        await engine.initialize()
        yield engine
        await engine.shutdown()
    
    @pytest.fixture
    def mock_model_registry(self):
        """Mock model registry for testing"""
        registry = Mock(spec=ModelRegistry)
        registry.get_model = AsyncMock()
        registry.load_model = AsyncMock()
        registry.is_model_available = Mock(return_value=True)
        return registry
    
    @pytest.fixture
    def mock_database(self):
        """Mock database for testing"""
        db = Mock(spec=Database)
        db.save_execution = AsyncMock()
        db.update_execution_status = AsyncMock()
        db.get_agent = AsyncMock()
        return db
    
    @pytest.fixture
    def sample_execution_request(self):
        """Sample execution request for testing"""
        return ExecutionRequest(
            execution_id="test-execution-123",
            agent_id="test-agent-456",
            user_public_key="11111111111111111111111111111112",
            input_data={
                "text": "Generate a story about AI agents",
                "parameters": {
                    "max_length": 500,
                    "temperature": 0.7
                }
            },
            resource_requirements=ResourceAllocation(
                memory_mb=1024,
                cpu_cores=2,
                gpu_memory_mb=512,
                max_execution_time=60,
                disk_space_mb=1024
            ),
            payment_amount=Decimal("0.1"),
            timeout_seconds=300
        )
    
    async def test_engine_initialization(self, engine):
        """Test execution engine initialization"""
        assert engine.is_healthy()
        assert engine.max_concurrent > 0
        assert engine.resource_monitor is not None
        assert engine.execution_queue is not None
    
    async def test_submit_execution_request(self, engine, sample_execution_request):
        """Test submitting an execution request"""
        with patch.object(engine, '_validate_request') as mock_validate:
            mock_validate.return_value = True
            
            execution_id = await engine.submit_execution(sample_execution_request)
            
            assert execution_id == sample_execution_request.execution_id
            mock_validate.assert_called_once_with(sample_execution_request)
    
    async def test_execution_validation(self, engine, sample_execution_request):
        """Test execution request validation"""
        # Valid request should pass
        is_valid = engine._validate_request(sample_execution_request)
        assert is_valid
        
        # Invalid agent ID should fail
        invalid_request = sample_execution_request.copy()
        invalid_request.agent_id = ""
        is_valid = engine._validate_request(invalid_request)
        assert not is_valid
        
        # Invalid resource requirements should fail
        invalid_request = sample_execution_request.copy()
        invalid_request.resource_requirements.memory_mb = 0
        is_valid = engine._validate_request(invalid_request)
        assert not is_valid
    
    async def test_resource_allocation(self, engine, sample_execution_request):
        """Test resource allocation for executions"""
        with patch.object(engine.resource_monitor, 'can_allocate') as mock_can_allocate:
            with patch.object(engine.resource_monitor, 'allocate_resources') as mock_allocate:
                mock_can_allocate.return_value = True
                mock_allocate.return_value = True
                
                can_allocate = await engine._check_resource_availability(
                    sample_execution_request.resource_requirements
                )
                
                assert can_allocate
                mock_can_allocate.assert_called_once()
    
    async def test_execution_lifecycle(self, engine, sample_execution_request):
        """Test complete execution lifecycle"""
        with patch.object(engine, '_execute_agent') as mock_execute:
            mock_result = ExecutionResult(
                execution_id=sample_execution_request.execution_id,
                status=ExecutionStatus.COMPLETED,
                output_data={"generated_text": "Test story about AI agents"},
                execution_time=45.2,
                resource_usage={
                    "memory_peak_mb": 512,
                    "cpu_usage_percent": 65.0,
                    "gpu_usage_percent": 30.0
                },
                error=None,
                metadata={"model_version": "1.0.0"}
            )
            mock_execute.return_value = mock_result
            
            # Submit execution
            execution_id = await engine.submit_execution(sample_execution_request)
            
            # Wait for completion
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Get result
            result = await engine.get_execution_result(execution_id)
            
            assert result.status == ExecutionStatus.COMPLETED
            assert result.output_data is not None
            assert result.execution_time > 0
    
    async def test_concurrent_executions(self, engine):
        """Test handling multiple concurrent executions"""
        requests = []
        for i in range(5):
            request = ExecutionRequest(
                execution_id=f"concurrent-{i}",
                agent_id=f"agent-{i}",
                user_public_key="11111111111111111111111111111112",
                input_data={"text": f"Test input {i}"},
                resource_requirements=ResourceAllocation(
                    memory_mb=256,
                    cpu_cores=1,
                    gpu_memory_mb=0,
                    max_execution_time=30,
                    disk_space_mb=512
                ),
                payment_amount=Decimal("0.05"),
                timeout_seconds=60
            )
            requests.append(request)
        
        with patch.object(engine, '_execute_agent') as mock_execute:
            mock_execute.return_value = ExecutionResult(
                execution_id="test",
                status=ExecutionStatus.COMPLETED,
                output_data={"result": "success"},
                execution_time=10.0,
                resource_usage={},
                error=None,
                metadata={}
            )
            
            # Submit all requests concurrently
            tasks = [engine.submit_execution(req) for req in requests]
            execution_ids = await asyncio.gather(*tasks)
            
            assert len(execution_ids) == 5
            assert len(set(execution_ids)) == 5  # All unique
    
    async def test_execution_timeout(self, engine, sample_execution_request):
        """Test execution timeout handling"""
        # Set a very short timeout
        sample_execution_request.timeout_seconds = 1
        
        with patch.object(engine, '_execute_agent') as mock_execute:
            # Simulate a long-running execution
            async def slow_execution(*args, **kwargs):
                await asyncio.sleep(2)
                return ExecutionResult(
                    execution_id=sample_execution_request.execution_id,
                    status=ExecutionStatus.COMPLETED,
                    output_data={},
                    execution_time=2.0,
                    resource_usage={},
                    error=None,
                    metadata={}
                )
            
            mock_execute.side_effect = slow_execution
            
            execution_id = await engine.submit_execution(sample_execution_request)
            
            # Wait for timeout
            await asyncio.sleep(1.5)
            
            result = await engine.get_execution_result(execution_id)
            assert result.status == ExecutionStatus.TIMEOUT
    
    async def test_execution_error_handling(self, engine, sample_execution_request):
        """Test execution error handling"""
        with patch.object(engine, '_execute_agent') as mock_execute:
            # Simulate execution error
            mock_execute.side_effect = Exception("Model execution failed")
            
            execution_id = await engine.submit_execution(sample_execution_request)
            
            # Wait for completion
            await asyncio.sleep(0.1)
            
            result = await engine.get_execution_result(execution_id)
            assert result.status == ExecutionStatus.FAILED
            assert "Model execution failed" in result.error
    
    async def test_resource_monitoring(self, engine):
        """Test resource monitoring during execution"""
        monitor = engine.resource_monitor
        
        # Test current resource usage
        usage = await monitor.get_current_usage()
        assert 'memory_percent' in usage
        assert 'cpu_percent' in usage
        assert 'disk_percent' in usage
        
        # Test resource limits
        assert monitor.memory_limit_mb > 0
        assert monitor.cpu_limit_cores > 0
    
    async def test_execution_queue_management(self, engine):
        """Test execution queue management"""
        queue = engine.execution_queue
        
        # Test queue operations
        assert queue.size() == 0
        
        # Add items to queue
        for i in range(3):
            await queue.put(f"item-{i}")
        
        assert queue.size() == 3
        
        # Remove items from queue
        item = await queue.get()
        assert item == "item-0"
        assert queue.size() == 2
    
    async def test_metrics_collection(self, engine, sample_execution_request):
        """Test metrics collection during execution"""
        with patch.object(engine, '_execute_agent') as mock_execute:
            mock_execute.return_value = ExecutionResult(
                execution_id=sample_execution_request.execution_id,
                status=ExecutionStatus.COMPLETED,
                output_data={},
                execution_time=30.0,
                resource_usage={
                    "memory_peak_mb": 512,
                    "cpu_usage_percent": 45.0
                },
                error=None,
                metadata={}
            )
            
            initial_count = engine.metrics.get_counter("executions_total")
            
            await engine.submit_execution(sample_execution_request)
            await asyncio.sleep(0.1)  # Wait for completion
            
            final_count = engine.metrics.get_counter("executions_total")
            assert final_count > initial_count
    
    async def test_graceful_shutdown(self, engine):
        """Test graceful shutdown of execution engine"""
        # Submit some executions
        for i in range(3):
            request = ExecutionRequest(
                execution_id=f"shutdown-test-{i}",
                agent_id=f"agent-{i}",
                user_public_key="11111111111111111111111111111112",
                input_data={"text": "test"},
                resource_requirements=ResourceAllocation(
                    memory_mb=256,
                    cpu_cores=1,
                    gpu_memory_mb=0,
                    max_execution_time=60,
                    disk_space_mb=512
                ),
                payment_amount=Decimal("0.01"),
                timeout_seconds=120
            )
            await engine.submit_execution(request)
        
        # Shutdown should wait for completions
        start_time = time.time()
        await engine.shutdown(timeout=30)
        shutdown_time = time.time() - start_time
        
        # Should shutdown within reasonable time
        assert shutdown_time < 35
        assert not engine.is_healthy()


@pytest.mark.integration
class TestExecutionEngineIntegration:
    """Integration tests for ExecutionEngine"""
    
    async def test_real_model_execution(self):
        """Test execution with real model (requires setup)"""
        # This test would require actual model setup
        # Skipped in unit tests, run in integration environment
        pytest.skip("Integration test - requires real model setup")
    
    async def test_database_integration(self):
        """Test database integration during execution"""
        # This test would require actual database
        # Skipped in unit tests, run in integration environment
        pytest.skip("Integration test - requires database setup")


# Performance benchmarks
@pytest.mark.benchmark
class TestExecutionEnginePerformance:
    """Performance benchmarks for ExecutionEngine"""
    
    async def test_execution_throughput(self, benchmark):
        """Benchmark execution throughput"""
        def create_and_execute():
            # Benchmark code here
            pass
        
        result = benchmark(create_and_execute)
        assert result is not None
    
    async def test_memory_usage(self):
        """Test memory usage during high load"""
        # Memory profiling code here
        pass 