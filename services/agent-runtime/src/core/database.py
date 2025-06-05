"""
Database layer for StratosHub Agent Runtime

Provides async database operations, connection pooling, and ORM models
for agent execution tracking and performance monitoring.
"""

import asyncio
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from decimal import Decimal
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import (
    Column, String, Integer, BigInteger, Boolean, DateTime, Numeric,
    Text, JSON, ForeignKey, Index, UniqueConstraint, select, update, delete
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

from ..core.config import get_settings
from ..core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Base class for all database models"""
    pass


class Agent(Base):
    """Agent model for database storage"""
    __tablename__ = 'agents'

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_public_key: Mapped[str] = mapped_column(String(44), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    model_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    metadata_uri: Mapped[str] = mapped_column(String(512), nullable=True)
    price_per_execution: Mapped[Decimal] = mapped_column(Numeric(20, 9), nullable=False)
    capabilities: Mapped[List[str]] = mapped_column(JSON, nullable=False, default=list)
    resource_requirements: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    execution_count: Mapped[int] = mapped_column(BigInteger, default=0)
    total_revenue: Mapped[Decimal] = mapped_column(Numeric(20, 9), default=0)
    success_count: Mapped[int] = mapped_column(BigInteger, default=0)
    failure_count: Mapped[int] = mapped_column(BigInteger, default=0)
    average_execution_time: Mapped[float] = mapped_column(Numeric(10, 3), default=0)
    reputation_score: Mapped[int] = mapped_column(Integer, default=1000)
    stake_amount: Mapped[Decimal] = mapped_column(Numeric(20, 9), default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    last_execution_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    executions: Mapped[List["Execution"]] = relationship("Execution", back_populates="agent", cascade="all, delete-orphan")
    performance_metrics: Mapped[List["AgentPerformanceMetric"]] = relationship("AgentPerformanceMetric", back_populates="agent", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index('idx_agent_owner_active', 'owner_public_key', 'is_active'),
        Index('idx_agent_model_type_active', 'model_type', 'is_active'),
        Index('idx_agent_reputation', 'reputation_score'),
        Index('idx_agent_price', 'price_per_execution'),
    )

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        if self.execution_count == 0:
            return 0.0
        return (self.success_count / self.execution_count) * 100

    def update_stats(self, execution_time: float, success: bool, revenue: Decimal):
        """Update agent statistics after execution"""
        self.execution_count += 1
        if success:
            self.success_count += 1
            self.total_revenue += revenue
        else:
            self.failure_count += 1
        
        # Update average execution time
        if self.execution_count == 1:
            self.average_execution_time = execution_time
        else:
            self.average_execution_time = (
                (self.average_execution_time * (self.execution_count - 1) + execution_time) / 
                self.execution_count
            )
        
        self.last_execution_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)


class Execution(Base):
    """Execution model for tracking agent runs"""
    __tablename__ = 'executions'

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    agent_id: Mapped[str] = mapped_column(String(64), ForeignKey('agents.id'), nullable=False, index=True)
    user_public_key: Mapped[str] = mapped_column(String(44), nullable=False, index=True)
    input_data_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    output_data_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default='pending', index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    execution_time: Mapped[Optional[float]] = mapped_column(Numeric(10, 3), nullable=True)
    gas_used: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    price_paid: Mapped[Decimal] = mapped_column(Numeric(20, 9), nullable=False)
    platform_fee: Mapped[Decimal] = mapped_column(Numeric(20, 9), nullable=False)
    resource_usage: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True, default=dict)

    # Relationships
    agent: Mapped["Agent"] = relationship("Agent", back_populates="executions")

    # Indexes
    __table_args__ = (
        Index('idx_execution_agent_status', 'agent_id', 'status'),
        Index('idx_execution_user_status', 'user_public_key', 'status'),
        Index('idx_execution_started_at', 'started_at'),
        Index('idx_execution_completed_at', 'completed_at'),
    )


class AgentPerformanceMetric(Base):
    """Performance metrics for agents over time"""
    __tablename__ = 'agent_performance_metrics'

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id: Mapped[str] = mapped_column(String(64), ForeignKey('agents.id'), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now(timezone.utc), index=True)
    executions_count: Mapped[int] = mapped_column(Integer, default=0)
    success_rate: Mapped[float] = mapped_column(Numeric(5, 2), default=0)
    average_execution_time: Mapped[float] = mapped_column(Numeric(10, 3), default=0)
    revenue_generated: Mapped[Decimal] = mapped_column(Numeric(20, 9), default=0)
    reputation_score: Mapped[int] = mapped_column(Integer, default=1000)
    resource_efficiency: Mapped[float] = mapped_column(Numeric(5, 2), default=0)
    error_rate: Mapped[float] = mapped_column(Numeric(5, 2), default=0)

    # Relationships
    agent: Mapped["Agent"] = relationship("Agent", back_populates="performance_metrics")

    # Indexes
    __table_args__ = (
        Index('idx_performance_agent_timestamp', 'agent_id', 'timestamp'),
        UniqueConstraint('agent_id', 'timestamp', name='uq_agent_performance_timestamp'),
    )


class MarketplaceMetric(Base):
    """Overall marketplace metrics"""
    __tablename__ = 'marketplace_metrics'

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now(timezone.utc), index=True)
    total_agents: Mapped[int] = mapped_column(Integer, default=0)
    active_agents: Mapped[int] = mapped_column(Integer, default=0)
    total_executions: Mapped[int] = mapped_column(BigInteger, default=0)
    total_volume: Mapped[Decimal] = mapped_column(Numeric(20, 9), default=0)
    total_fees: Mapped[Decimal] = mapped_column(Numeric(20, 9), default=0)
    average_gas_price: Mapped[Decimal] = mapped_column(Numeric(20, 9), default=0)
    network_utilization: Mapped[float] = mapped_column(Numeric(5, 2), default=0)

    # Indexes
    __table_args__ = (
        Index('idx_marketplace_timestamp', 'timestamp'),
    )


class Database:
    """Database connection and session management"""
    
    def __init__(self):
        self.engine = None
        self.session_factory = None
        self._connected = False
    
    async def connect(self):
        """Establish database connection"""
        try:
            # Build connection URL
            if settings.database_url:
                connection_url = settings.database_url
            else:
                connection_url = (
                    f"postgresql+asyncpg://{settings.database_user}:{settings.database_password}"
                    f"@{settings.database_host}:{settings.database_port}/{settings.database_name}"
                )
            
            # Create engine with connection pooling
            self.engine = create_async_engine(
                connection_url,
                pool_size=settings.database_pool_size,
                max_overflow=settings.database_max_overflow,
                pool_timeout=settings.database_timeout,
                pool_recycle=3600,  # Recycle connections every hour
                echo=settings.database_echo,
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test connection
            async with self.session_factory() as session:
                await session.execute(select(1))
            
            self._connected = True
            logger.info("Database connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    async def disconnect(self):
        """Close database connection"""
        if self.engine:
            await self.engine.dispose()
            self._connected = False
            logger.info("Database connection closed")
    
    async def health_check(self) -> bool:
        """Check database health"""
        if not self._connected or not self.session_factory:
            return False
        
        try:
            async with self.session_factory() as session:
                await session.execute(select(1))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def get_session(self) -> AsyncSession:
        """Get database session"""
        if not self.session_factory:
            raise RuntimeError("Database not connected")
        return self.session_factory()
    
    # Agent operations
    async def save_agent(self, agent_data: Dict[str, Any]) -> Agent:
        """Save new agent to database"""
        async with self.session_factory() as session:
            agent = Agent(**agent_data)
            session.add(agent)
            await session.commit()
            await session.refresh(agent)
            return agent
    
    async def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get agent by ID"""
        async with self.session_factory() as session:
            result = await session.execute(
                select(Agent).where(Agent.id == agent_id)
            )
            return result.scalar_one_or_none()
    
    async def update_agent(self, agent_id: str, update_data: Dict[str, Any]) -> Optional[Agent]:
        """Update agent data"""
        async with self.session_factory() as session:
            result = await session.execute(
                update(Agent)
                .where(Agent.id == agent_id)
                .values(**update_data, updated_at=datetime.now(timezone.utc))
                .returning(Agent)
            )
            agent = result.scalar_one_or_none()
            await session.commit()
            return agent
    
    async def list_agents(
        self, 
        owner: Optional[str] = None,
        model_type: Optional[str] = None,
        is_active: Optional[bool] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[Agent]:
        """List agents with filters"""
        async with self.session_factory() as session:
            query = select(Agent)
            
            if owner:
                query = query.where(Agent.owner_public_key == owner)
            if model_type:
                query = query.where(Agent.model_type == model_type)
            if is_active is not None:
                query = query.where(Agent.is_active == is_active)
            
            query = query.order_by(Agent.created_at.desc()).limit(limit).offset(offset)
            
            result = await session.execute(query)
            return result.scalars().all()
    
    # Execution operations
    async def save_execution(self, execution_data: Dict[str, Any]) -> Execution:
        """Save execution record"""
        async with self.session_factory() as session:
            execution = Execution(**execution_data)
            session.add(execution)
            await session.commit()
            await session.refresh(execution)
            return execution
    
    async def update_execution_status(
        self, 
        execution_id: str, 
        status: str, 
        **kwargs
    ) -> Optional[Execution]:
        """Update execution status and related fields"""
        update_data = {"status": status, **kwargs}
        if status in ['completed', 'failed', 'timeout']:
            update_data['completed_at'] = datetime.now(timezone.utc)
        
        async with self.session_factory() as session:
            result = await session.execute(
                update(Execution)
                .where(Execution.id == execution_id)
                .values(**update_data)
                .returning(Execution)
            )
            execution = result.scalar_one_or_none()
            await session.commit()
            return execution
    
    async def get_execution_history(
        self,
        agent_id: Optional[str] = None,
        user_public_key: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[Execution]:
        """Get execution history with filters"""
        async with self.session_factory() as session:
            query = select(Execution)
            
            if agent_id:
                query = query.where(Execution.agent_id == agent_id)
            if user_public_key:
                query = query.where(Execution.user_public_key == user_public_key)
            if status:
                query = query.where(Execution.status == status)
            
            query = query.order_by(Execution.started_at.desc()).limit(limit).offset(offset)
            
            result = await session.execute(query)
            return result.scalars().all()
    
    # Metrics operations
    async def save_performance_metric(self, metric_data: Dict[str, Any]) -> AgentPerformanceMetric:
        """Save agent performance metric"""
        async with self.session_factory() as session:
            metric = AgentPerformanceMetric(**metric_data)
            session.add(metric)
            await session.commit()
            await session.refresh(metric)
            return metric
    
    async def get_agent_metrics(
        self,
        agent_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[AgentPerformanceMetric]:
        """Get agent performance metrics for time range"""
        async with self.session_factory() as session:
            result = await session.execute(
                select(AgentPerformanceMetric)
                .where(
                    AgentPerformanceMetric.agent_id == agent_id,
                    AgentPerformanceMetric.timestamp >= start_time,
                    AgentPerformanceMetric.timestamp <= end_time
                )
                .order_by(AgentPerformanceMetric.timestamp)
            )
            return result.scalars().all()
    
    async def save_marketplace_metric(self, metric_data: Dict[str, Any]) -> MarketplaceMetric:
        """Save marketplace metric"""
        async with self.session_factory() as session:
            metric = MarketplaceMetric(**metric_data)
            session.add(metric)
            await session.commit()
            await session.refresh(metric)
            return metric


# Global database instance
_database_instance = None

def get_database() -> Database:
    """Get global database instance"""
    global _database_instance
    if _database_instance is None:
        _database_instance = Database()
    return _database_instance 