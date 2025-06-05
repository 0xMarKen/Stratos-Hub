use anchor_lang::prelude::*;
use anchor_spl::token::{self, Token, TokenAccount, Transfer};

use crate::state::{Agent, AgentRegistry, MarketplaceConfig};
use crate::errors::MarketplaceError;
use crate::events::AgentCreated;

#[derive(Accounts)]
#[instruction(agent_id: String)]
pub struct CreateAgent<'info> {
    #[account(
        init,
        payer = owner,
        space = Agent::SPACE,
        seeds = [b"agent", agent_id.as_bytes()],
        bump
    )]
    pub agent: Account<'info, Agent>,

    #[account(
        mut,
        seeds = [b"registry"],
        bump
    )]
    pub registry: Account<'info, AgentRegistry>,

    #[account(
        seeds = [b"config"],
        bump
    )]
    pub config: Account<'info, MarketplaceConfig>,

    #[account(mut)]
    pub owner: Signer<'info>,

    #[account(
        mut,
        constraint = owner_token_account.owner == owner.key(),
        constraint = owner_token_account.mint == config.payment_mint
    )]
    pub owner_token_account: Account<'info, TokenAccount>,

    #[account(
        mut,
        constraint = treasury_token_account.owner == config.treasury,
        constraint = treasury_token_account.mint == config.payment_mint
    )]
    pub treasury_token_account: Account<'info, TokenAccount>,

    pub token_program: Program<'info, Token>,
    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
}

pub fn create_agent(
    ctx: Context<CreateAgent>,
    agent_id: String,
    name: String,
    description: String,
    model_type: String,
    metadata_uri: String,
    price_per_execution: u64,
    capabilities: Vec<String>,
    resource_requirements: ResourceRequirements,
) -> Result<()> {
    let config = &ctx.accounts.config;
    let registry = &mut ctx.accounts.registry;
    let agent = &mut ctx.accounts.agent;
    let owner = &ctx.accounts.owner;

    // Validate input parameters
    require!(
        agent_id.len() <= 64 && !agent_id.is_empty(),
        MarketplaceError::InvalidAgentId
    );
    require!(
        name.len() <= 128 && !name.is_empty(),
        MarketplaceError::InvalidAgentName
    );
    require!(
        description.len() <= 512,
        MarketplaceError::InvalidAgentDescription
    );
    require!(
        metadata_uri.len() <= 256,
        MarketplaceError::InvalidMetadataUri
    );
    require!(
        price_per_execution >= config.min_price_per_execution,
        MarketplaceError::PriceTooLow
    );
    require!(
        price_per_execution <= config.max_price_per_execution,
        MarketplaceError::PriceTooHigh
    );
    require!(
        capabilities.len() <= 10 && !capabilities.is_empty(),
        MarketplaceError::InvalidCapabilities
    );

    // Validate model type
    let valid_model_types = vec![
        "text-generation",
        "text-classification", 
        "image-generation",
        "image-classification",
        "audio-generation",
        "audio-transcription",
        "data-analysis",
        "predictive-modeling",
        "reinforcement-learning",
        "custom"
    ];
    require!(
        valid_model_types.contains(&model_type.as_str()),
        MarketplaceError::InvalidModelType
    );

    // Validate resource requirements
    require!(
        resource_requirements.memory_mb >= 128 && resource_requirements.memory_mb <= 32768,
        MarketplaceError::InvalidResourceRequirements
    );
    require!(
        resource_requirements.cpu_cores >= 1 && resource_requirements.cpu_cores <= 16,
        MarketplaceError::InvalidResourceRequirements
    );
    require!(
        resource_requirements.max_execution_time >= 1 && resource_requirements.max_execution_time <= 3600,
        MarketplaceError::InvalidResourceRequirements
    );
    require!(
        resource_requirements.disk_space_mb >= 100 && resource_requirements.disk_space_mb <= 10240,
        MarketplaceError::InvalidResourceRequirements
    );

    // Check if owner has reached the maximum number of agents
    require!(
        registry.get_agent_count_by_owner(&owner.key()) < config.max_agents_per_owner,
        MarketplaceError::MaxAgentsReached
    );

    // Calculate and collect registration fee
    let registration_fee = config.agent_registration_fee;
    if registration_fee > 0 {
        let transfer_instruction = Transfer {
            from: ctx.accounts.owner_token_account.to_account_info(),
            to: ctx.accounts.treasury_token_account.to_account_info(),
            authority: owner.to_account_info(),
        };

        let cpi_ctx = CpiContext::new(
            ctx.accounts.token_program.to_account_info(),
            transfer_instruction,
        );

        token::transfer(cpi_ctx, registration_fee)?;
    }

    // Initialize agent account
    let clock = Clock::get()?;
    
    agent.id = agent_id.clone();
    agent.owner = owner.key();
    agent.name = name;
    agent.description = description;
    agent.model_type = model_type;
    agent.metadata_uri = metadata_uri;
    agent.price_per_execution = price_per_execution;
    agent.capabilities = capabilities;
    agent.resource_requirements = resource_requirements;
    agent.is_active = true;
    agent.execution_count = 0;
    agent.total_revenue = 0;
    agent.success_count = 0;
    agent.failure_count = 0;
    agent.average_execution_time = 0;
    agent.reputation_score = 1000; // Starting reputation
    agent.stake_amount = 0;
    agent.created_at = clock.unix_timestamp;
    agent.updated_at = clock.unix_timestamp;
    agent.last_execution_at = 0;
    agent.bump = *ctx.bumps.get("agent").unwrap();

    // Update registry
    registry.total_agents += 1;
    registry.add_agent_to_owner(&owner.key());

    // Emit event
    emit!(AgentCreated {
        agent_id: agent_id,
        owner: owner.key(),
        price_per_execution: price_per_execution,
        model_type: agent.model_type.clone(),
        timestamp: clock.unix_timestamp,
    });

    msg!("Agent created successfully: {}", agent.id);
    Ok(())
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct ResourceRequirements {
    pub memory_mb: u32,
    pub cpu_cores: u8,
    pub gpu_memory_mb: u32,
    pub max_execution_time: u32, // seconds
    pub disk_space_mb: u32,
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            memory_mb: 512,
            cpu_cores: 1,
            gpu_memory_mb: 0,
            max_execution_time: 60,
            disk_space_mb: 1024,
        }
    }
} 