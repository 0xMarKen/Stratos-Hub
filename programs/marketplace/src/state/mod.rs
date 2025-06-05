//! State module for StratosHub Marketplace smart contracts
//! 
//! Defines all account structures and state management for the decentralized
//! AI agent marketplace on Solana blockchain.

use anchor_lang::prelude::*;

pub mod agent;
pub mod registry;
pub mod config;
pub mod escrow;
pub mod dispute;
pub mod staking;

pub use agent::*;
pub use registry::*;
pub use config::*;
pub use escrow::*;
pub use dispute::*;
pub use staking::*;

/// Maximum number of agents per owner
pub const MAX_AGENTS_PER_OWNER: u32 = 100;

/// Maximum number of capabilities per agent
pub const MAX_CAPABILITIES: usize = 10;

/// Maximum length for string fields
pub const MAX_STRING_LENGTH: usize = 256;

/// Agent account structure
#[account]
pub struct Agent {
    /// Unique identifier for the agent
    pub id: String,
    
    /// Owner's public key
    pub owner: Pubkey,
    
    /// Agent name
    pub name: String,
    
    /// Agent description
    pub description: String,
    
    /// Model type (text-generation, image-generation, etc.)
    pub model_type: String,
    
    /// IPFS URI for metadata
    pub metadata_uri: String,
    
    /// Price per execution in lamports
    pub price_per_execution: u64,
    
    /// Agent capabilities
    pub capabilities: Vec<String>,
    
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
    
    /// Whether the agent is active
    pub is_active: bool,
    
    /// Total number of executions
    pub execution_count: u64,
    
    /// Total revenue earned
    pub total_revenue: u64,
    
    /// Number of successful executions
    pub success_count: u64,
    
    /// Number of failed executions
    pub failure_count: u64,
    
    /// Average execution time in seconds
    pub average_execution_time: u64,
    
    /// Reputation score (0-10000)
    pub reputation_score: u32,
    
    /// Amount staked by the agent owner
    pub stake_amount: u64,
    
    /// Creation timestamp
    pub created_at: i64,
    
    /// Last update timestamp
    pub updated_at: i64,
    
    /// Last execution timestamp
    pub last_execution_at: i64,
    
    /// Bump seed for PDA derivation
    pub bump: u8,
}

impl Agent {
    pub const SPACE: usize = 8 + // discriminator
        64 + // id
        32 + // owner
        128 + // name
        512 + // description
        64 + // model_type
        256 + // metadata_uri
        8 + // price_per_execution
        (4 + 10 * 64) + // capabilities (max 10 items, 64 chars each)
        ResourceRequirements::SPACE + // resource_requirements
        1 + // is_active
        8 + // execution_count
        8 + // total_revenue
        8 + // success_count
        8 + // failure_count
        8 + // average_execution_time
        4 + // reputation_score
        8 + // stake_amount
        8 + // created_at
        8 + // updated_at
        8 + // last_execution_at
        1; // bump
    
    /// Calculate success rate as percentage
    pub fn success_rate(&self) -> u32 {
        if self.execution_count == 0 {
            return 0;
        }
        ((self.success_count * 100) / self.execution_count) as u32
    }
    
    /// Update reputation score based on execution results
    pub fn update_reputation(&mut self, success: bool, execution_time: u64) {
        let base_change = if success { 10 } else { -20 };
        
        // Adjust based on execution time relative to expected
        let time_factor = if execution_time <= self.resource_requirements.max_execution_time as u64 {
            1.0
        } else {
            0.8 // Penalty for slow execution
        };
        
        let change = (base_change as f64 * time_factor) as i32;
        
        if change > 0 {
            self.reputation_score = std::cmp::min(10000, self.reputation_score + change as u32);
        } else {
            self.reputation_score = self.reputation_score.saturating_sub((-change) as u32);
        }
    }
}

/// Resource requirements for agent execution
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug)]
pub struct ResourceRequirements {
    /// Memory requirement in MB
    pub memory_mb: u32,
    
    /// CPU cores required
    pub cpu_cores: u8,
    
    /// GPU memory requirement in MB (0 if not needed)
    pub gpu_memory_mb: u32,
    
    /// Maximum execution time in seconds
    pub max_execution_time: u32,
    
    /// Disk space requirement in MB
    pub disk_space_mb: u32,
}

impl ResourceRequirements {
    pub const SPACE: usize = 4 + 1 + 4 + 4 + 4;
}

/// Agent registry for tracking all agents
#[account]
pub struct AgentRegistry {
    /// Authority that can update the registry
    pub authority: Pubkey,
    
    /// Total number of registered agents
    pub total_agents: u64,
    
    /// Total number of active agents
    pub active_agents: u64,
    
    /// Total executions across all agents
    pub total_executions: u64,
    
    /// Total volume in lamports
    pub total_volume: u64,
    
    /// Registry creation timestamp
    pub created_at: i64,
    
    /// Last update timestamp
    pub updated_at: i64,
    
    /// Bump seed for PDA derivation
    pub bump: u8,
}

impl AgentRegistry {
    pub const SPACE: usize = 8 + // discriminator
        32 + // authority
        8 + // total_agents
        8 + // active_agents
        8 + // total_executions
        8 + // total_volume
        8 + // created_at
        8 + // updated_at
        1; // bump
    
    /// Add agent to registry
    pub fn add_agent(&mut self) {
        self.total_agents += 1;
        self.active_agents += 1;
        self.updated_at = Clock::get().unwrap().unix_timestamp;
    }
    
    /// Remove agent from registry
    pub fn remove_agent(&mut self) {
        self.active_agents = self.active_agents.saturating_sub(1);
        self.updated_at = Clock::get().unwrap().unix_timestamp;
    }
    
    /// Record execution
    pub fn record_execution(&mut self, volume: u64) {
        self.total_executions += 1;
        self.total_volume += volume;
        self.updated_at = Clock::get().unwrap().unix_timestamp;
    }
}

/// Marketplace configuration
#[account]
pub struct MarketplaceConfig {
    /// Authority that can update the config
    pub authority: Pubkey,
    
    /// Treasury account for fees
    pub treasury: Pubkey,
    
    /// Token mint for payments
    pub payment_mint: Pubkey,
    
    /// Platform fee in basis points (100 = 1%)
    pub platform_fee_bps: u16,
    
    /// Agent registration fee in lamports
    pub agent_registration_fee: u64,
    
    /// Minimum price per execution
    pub min_price_per_execution: u64,
    
    /// Maximum price per execution
    pub max_price_per_execution: u64,
    
    /// Maximum agents per owner
    pub max_agents_per_owner: u32,
    
    /// Minimum stake amount for agents
    pub min_stake_amount: u64,
    
    /// Dispute resolution timeframe in seconds
    pub dispute_window: i64,
    
    /// Whether the marketplace is paused
    pub is_paused: bool,
    
    /// Configuration creation timestamp
    pub created_at: i64,
    
    /// Last update timestamp
    pub updated_at: i64,
    
    /// Bump seed for PDA derivation
    pub bump: u8,
}

impl MarketplaceConfig {
    pub const SPACE: usize = 8 + // discriminator
        32 + // authority
        32 + // treasury
        32 + // payment_mint
        2 + // platform_fee_bps
        8 + // agent_registration_fee
        8 + // min_price_per_execution
        8 + // max_price_per_execution
        4 + // max_agents_per_owner
        8 + // min_stake_amount
        8 + // dispute_window
        1 + // is_paused
        8 + // created_at
        8 + // updated_at
        1; // bump
    
    /// Calculate platform fee for a given amount
    pub fn calculate_platform_fee(&self, amount: u64) -> u64 {
        (amount * self.platform_fee_bps as u64) / 10000
    }
    
    /// Validate price range
    pub fn is_valid_price(&self, price: u64) -> bool {
        price >= self.min_price_per_execution && price <= self.max_price_per_execution
    }
} 