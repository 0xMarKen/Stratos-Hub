use anchor_lang::prelude::*;

#[account]
#[derive(InitSpace)]
pub struct MarketplaceState {
    /// Authority that can update marketplace configuration
    pub authority: Pubkey,
    /// Fee charged on each execution (in basis points, 100 = 1%)
    pub fee_basis_points: u16,
    /// Maximum number of agents a single user can register
    pub max_agents_per_user: u32,
    /// Minimum stake amount required to become a provider
    pub min_stake_amount: u64,
    /// Time window for disputing executions (in seconds)
    pub dispute_window: i64,
    /// Total number of agents registered
    pub total_agents: u64,
    /// Total number of executions processed
    pub total_executions: u64,
    /// Total volume processed (in lamports)
    pub total_volume: u64,
    /// Total fees collected (in lamports)
    pub total_fees: u64,
    /// Whether the marketplace is paused for emergency
    pub is_paused: bool,
    /// Timestamp when marketplace was created
    pub created_at: i64,
    /// Timestamp when last updated
    pub updated_at: i64,
    /// Bump seed for PDA
    pub bump: u8,
}

#[account]
#[derive(InitSpace)]
pub struct AgentAccount {
    /// Unique identifier for the agent
    #[max_len(64)]
    pub agent_id: String,
    /// Human-readable name
    #[max_len(128)]
    pub name: String,
    /// Description of the agent's capabilities
    #[max_len(512)]
    pub description: String,
    /// Owner's public key
    pub owner: Pubkey,
    /// Type of ML model
    pub model_type: ModelType,
    /// Price per execution in lamports
    pub price_per_execution: u64,
    /// URI pointing to agent metadata on IPFS/Arweave
    #[max_len(256)]
    pub metadata_uri: String,
    /// List of agent capabilities
    #[max_len(10, 64)]
    pub capabilities: Vec<String>,
    /// Resource requirements for execution
    pub resource_requirements: ResourceRequirements,
    /// Whether the agent is currently active
    pub is_active: bool,
    /// Whether the agent is verified by the platform
    pub is_verified: bool,
    /// Total number of executions
    pub execution_count: u64,
    /// Total revenue earned (in lamports)
    pub total_revenue: u64,
    /// Average rating (0-100)
    pub rating: u8,
    /// Number of ratings received
    pub rating_count: u32,
    /// Timestamp when agent was registered
    pub created_at: i64,
    /// Timestamp when last updated
    pub updated_at: i64,
    /// PDA bump seed
    pub bump: u8,
}

#[account]
#[derive(InitSpace)]
pub struct ProviderAccount {
    /// Provider's public key
    pub owner: Pubkey,
    /// Amount of tokens staked
    pub staked_amount: u64,
    /// Number of agents owned by this provider
    pub agent_count: u32,
    /// Total executions across all agents
    pub total_executions: u64,
    /// Total revenue earned
    pub total_revenue: u64,
    /// Provider reputation score (0-100)
    pub reputation: u8,
    /// Number of successful executions
    pub successful_executions: u64,
    /// Number of disputed executions
    pub disputed_executions: u32,
    /// Whether provider is currently slashed
    pub is_slashed: bool,
    /// Amount currently slashed
    pub slashed_amount: u64,
    /// Timestamp when staking began
    pub stake_timestamp: i64,
    /// Timestamp when unstaking was initiated (0 if not unstaking)
    pub unstake_timestamp: i64,
    /// PDA bump seed
    pub bump: u8,
}

#[account]
#[derive(InitSpace)]
pub struct ExecutionRecord {
    /// Unique execution identifier
    #[max_len(64)]
    pub execution_id: String,
    /// Agent that was executed
    pub agent: Pubkey,
    /// User who initiated the execution
    pub user: Pubkey,
    /// Agent owner who executed
    pub agent_owner: Pubkey,
    /// Price paid for this execution
    pub price_paid: u64,
    /// Fee paid to marketplace
    pub fee_paid: u64,
    /// Hash of input data
    pub input_data_hash: [u8; 32],
    /// Hash of output data (set when completed)
    pub output_data_hash: Option<[u8; 32]>,
    /// Gas/compute units used
    pub gas_used: u64,
    /// Current status of the execution
    pub status: ExecutionStatus,
    /// Whether execution was successful
    pub success: bool,
    /// Error message if execution failed
    #[max_len(256)]
    pub error_message: Option<String>,
    /// Timestamp when execution started
    pub created_at: i64,
    /// Timestamp when execution completed
    pub completed_at: Option<i64>,
    /// PDA bump seed
    pub bump: u8,
}

#[account]
#[derive(InitSpace)]
pub struct DisputeRecord {
    /// Associated execution ID
    #[max_len(64)]
    pub execution_id: String,
    /// Execution account
    pub execution: Pubkey,
    /// User who initiated the dispute
    pub disputer: Pubkey,
    /// Agent owner being disputed
    pub disputed_party: Pubkey,
    /// Reason for the dispute
    #[max_len(512)]
    pub dispute_reason: String,
    /// URI pointing to evidence
    #[max_len(256)]
    pub evidence_uri: String,
    /// Current status of the dispute
    pub status: DisputeStatus,
    /// Resolution of the dispute
    pub resolution: Option<DisputeResolution>,
    /// Percentage of refund (0-100)
    pub refund_percentage: u8,
    /// Timestamp when dispute was created
    pub created_at: i64,
    /// Timestamp when dispute was resolved
    pub resolved_at: Option<i64>,
    /// PDA bump seed
    pub bump: u8,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Eq, InitSpace)]
pub enum ModelType {
    TextGeneration,
    TextClassification,
    ImageGeneration,
    ImageClassification,
    AudioGeneration,
    AudioTranscription,
    DataAnalysis,
    PredictiveModeling,
    ReinforcementLearning,
    Custom,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, InitSpace)]
pub struct ResourceRequirements {
    /// Memory required in MB
    pub memory_mb: u32,
    /// CPU cores required
    pub cpu_cores: u8,
    /// GPU memory required in MB (0 if no GPU needed)
    pub gpu_memory_mb: u32,
    /// Maximum execution time in seconds
    pub max_execution_time: u32,
    /// Minimum disk space in MB
    pub disk_space_mb: u32,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Eq, InitSpace)]
pub enum ExecutionStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Disputed,
    Cancelled,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Eq, InitSpace)]
pub enum DisputeStatus {
    Open,
    UnderReview,
    Resolved,
    Dismissed,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Eq, InitSpace)]
pub enum DisputeResolution {
    RefundUser,
    FavorProvider,
    PartialRefund,
    Escalate,
}

impl MarketplaceState {
    pub const INIT_SPACE: usize = 
        32 + // authority
        2 + // fee_basis_points
        4 + // max_agents_per_user
        8 + // min_stake_amount
        8 + // dispute_window
        8 + // total_agents
        8 + // total_executions
        8 + // total_volume
        8 + // total_fees
        1 + // is_paused
        8 + // created_at
        8 + // updated_at
        1; // bump

    pub fn update_stats(&mut self, execution_price: u64, fee_amount: u64) {
        self.total_executions = self.total_executions.saturating_add(1);
        self.total_volume = self.total_volume.saturating_add(execution_price);
        self.total_fees = self.total_fees.saturating_add(fee_amount);
        self.updated_at = Clock::get().unwrap().unix_timestamp;
    }
}

impl AgentAccount {
    pub const INIT_SPACE: usize = 
        4 + 64 + // agent_id
        4 + 128 + // name
        4 + 512 + // description
        32 + // owner
        1 + // model_type (largest enum variant)
        8 + // price_per_execution
        4 + 256 + // metadata_uri
        4 + (10 * (4 + 64)) + // capabilities
        4 + 4 + 1 + 4 + 4 + 4 + // resource_requirements
        1 + // is_active
        1 + // is_verified
        8 + // execution_count
        8 + // total_revenue
        1 + // rating
        4 + // rating_count
        8 + // created_at
        8 + // updated_at
        1; // bump

    pub fn update_execution_stats(&mut self, price: u64, success: bool) {
        self.execution_count = self.execution_count.saturating_add(1);
        if success {
            self.total_revenue = self.total_revenue.saturating_add(price);
        }
        self.updated_at = Clock::get().unwrap().unix_timestamp;
    }

    pub fn update_rating(&mut self, new_rating: u8) {
        let total_rating = (self.rating as u64)
            .saturating_mul(self.rating_count as u64)
            .saturating_add(new_rating as u64);
        self.rating_count = self.rating_count.saturating_add(1);
        self.rating = (total_rating / self.rating_count as u64) as u8;
        self.updated_at = Clock::get().unwrap().unix_timestamp;
    }
}

impl ProviderAccount {
    pub const INIT_SPACE: usize = 
        32 + // owner
        8 + // staked_amount
        4 + // agent_count
        8 + // total_executions
        8 + // total_revenue
        1 + // reputation
        8 + // successful_executions
        4 + // disputed_executions
        1 + // is_slashed
        8 + // slashed_amount
        8 + // stake_timestamp
        8 + // unstake_timestamp
        1; // bump

    pub fn add_agent(&mut self) {
        self.agent_count = self.agent_count.saturating_add(1);
    }

    pub fn remove_agent(&mut self) {
        self.agent_count = self.agent_count.saturating_sub(1);
    }

    pub fn update_execution_stats(&mut self, revenue: u64, success: bool) {
        self.total_executions = self.total_executions.saturating_add(1);
        if success {
            self.total_revenue = self.total_revenue.saturating_add(revenue);
            self.successful_executions = self.successful_executions.saturating_add(1);
        }
        self.update_reputation();
    }

    pub fn add_dispute(&mut self) {
        self.disputed_executions = self.disputed_executions.saturating_add(1);
        self.update_reputation();
    }

    fn update_reputation(&mut self) {
        if self.total_executions > 0 {
            let success_rate = (self.successful_executions * 100) / self.total_executions;
            let dispute_penalty = if self.total_executions > 0 {
                (self.disputed_executions as u64 * 10) / self.total_executions
            } else {
                0
            };
            self.reputation = (success_rate.saturating_sub(dispute_penalty)) as u8;
        }
    }
}

impl ExecutionRecord {
    pub const INIT_SPACE: usize = 
        4 + 64 + // execution_id
        32 + // agent
        32 + // user
        32 + // agent_owner
        8 + // price_paid
        8 + // fee_paid
        32 + // input_data_hash
        1 + 32 + // output_data_hash (Option)
        8 + // gas_used
        1 + // status
        1 + // success
        1 + 4 + 256 + // error_message (Option)
        8 + // created_at
        1 + 8 + // completed_at (Option)
        1; // bump
}

impl DisputeRecord {
    pub const INIT_SPACE: usize = 
        4 + 64 + // execution_id
        32 + // execution
        32 + // disputer
        32 + // disputed_party
        4 + 512 + // dispute_reason
        4 + 256 + // evidence_uri
        1 + // status
        1 + 1 + // resolution (Option)
        1 + // refund_percentage
        8 + // created_at
        1 + 8 + // resolved_at (Option)
        1; // bump
} 