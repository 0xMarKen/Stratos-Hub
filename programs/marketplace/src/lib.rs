use anchor_lang::prelude::*;
use anchor_spl::token::{self, Token, TokenAccount, Transfer};
use anchor_spl::associated_token::AssociatedToken;

declare_id!("StratosHub11111111111111111111111111111111");

pub mod instructions;
pub mod state;
pub mod errors;
pub mod events;
pub mod utils;

use instructions::*;
use state::*;
use errors::*;
use events::*;

#[program]
pub mod marketplace {
    use super::*;

    /// Initialize the marketplace with configuration
    pub fn initialize(
        ctx: Context<Initialize>,
        authority: Pubkey,
        treasury: Pubkey,
        payment_mint: Pubkey,
        platform_fee_bps: u16,
        agent_registration_fee: u64,
        min_price_per_execution: u64,
        max_price_per_execution: u64,
        max_agents_per_owner: u32,
        min_stake_amount: u64,
        dispute_window: i64,
    ) -> Result<()> {
        instructions::initialize::initialize(
            ctx,
            authority,
            treasury,
            payment_mint,
            platform_fee_bps,
            agent_registration_fee,
            min_price_per_execution,
            max_price_per_execution,
            max_agents_per_owner,
            min_stake_amount,
            dispute_window,
        )
    }

    /// Register a new AI agent
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
        instructions::create_agent::create_agent(
            ctx,
            agent_id,
            name,
            description,
            model_type,
            metadata_uri,
            price_per_execution,
            capabilities,
            resource_requirements,
        )
    }

    /// Update existing agent configuration
    pub fn update_agent(
        ctx: Context<UpdateAgent>,
        price_per_execution: Option<u64>,
        metadata_uri: Option<String>,
        is_active: Option<bool>,
        resource_requirements: Option<ResourceRequirements>,
    ) -> Result<()> {
        instructions::update_agent::update_agent(
            ctx,
            price_per_execution,
            metadata_uri,
            is_active,
            resource_requirements,
        )
    }

    /// Execute an AI agent
    pub fn execute_agent(
        ctx: Context<ExecuteAgent>,
        execution_id: String,
        input_data_hash: [u8; 32],
        max_gas: u64,
        timeout: u32,
    ) -> Result<()> {
        instructions::execute_agent::execute_agent(
            ctx,
            execution_id,
            input_data_hash,
            max_gas,
            timeout,
        )
    }

    /// Complete agent execution and record results
    pub fn complete_execution(
        ctx: Context<CompleteExecution>,
        execution_id: String,
        output_data_hash: [u8; 32],
        gas_used: u64,
        execution_time: u64,
        success: bool,
    ) -> Result<()> {
        instructions::complete_execution::complete_execution(
            ctx,
            execution_id,
            output_data_hash,
            gas_used,
            execution_time,
            success,
        )
    }

    /// Stake tokens for an agent
    pub fn stake_agent(
        ctx: Context<StakeAgent>,
        amount: u64,
    ) -> Result<()> {
        instructions::stake_agent::stake_agent(ctx, amount)
    }

    /// Unstake tokens from an agent
    pub fn unstake_agent(
        ctx: Context<UnstakeAgent>,
        amount: u64,
    ) -> Result<()> {
        instructions::unstake_agent::unstake_agent(ctx, amount)
    }

    /// Initialize escrow for agent execution
    pub fn create_escrow(
        ctx: Context<CreateEscrow>,
        execution_id: String,
        amount: u64,
        timeout: i64,
    ) -> Result<()> {
        instructions::create_escrow::create_escrow(
            ctx,
            execution_id,
            amount,
            timeout,
        )
    }

    /// Release escrow funds after successful execution
    pub fn release_escrow(
        ctx: Context<ReleaseEscrow>,
        execution_id: String,
    ) -> Result<()> {
        instructions::release_escrow::release_escrow(ctx, execution_id)
    }

    /// Initiate dispute for failed execution
    pub fn create_dispute(
        ctx: Context<CreateDispute>,
        execution_id: String,
        dispute_reason: String,
        evidence_uri: String,
    ) -> Result<()> {
        instructions::create_dispute::create_dispute(
            ctx,
            execution_id,
            dispute_reason,
            evidence_uri,
        )
    }

    /// Resolve dispute by authority
    pub fn resolve_dispute(
        ctx: Context<ResolveDispute>,
        execution_id: String,
        resolution: DisputeResolution,
        refund_percentage: u8,
    ) -> Result<()> {
        instructions::resolve_dispute::resolve_dispute(
            ctx,
            execution_id,
            resolution,
            refund_percentage,
        )
    }

    /// Update marketplace configuration (admin only)
    pub fn update_config(
        ctx: Context<UpdateConfig>,
        platform_fee_bps: Option<u16>,
        agent_registration_fee: Option<u64>,
        min_price_per_execution: Option<u64>,
        max_price_per_execution: Option<u64>,
        max_agents_per_owner: Option<u32>,
        min_stake_amount: Option<u64>,
        dispute_window: Option<i64>,
        is_paused: Option<bool>,
    ) -> Result<()> {
        instructions::update_config::update_config(
            ctx,
            platform_fee_bps,
            agent_registration_fee,
            min_price_per_execution,
            max_price_per_execution,
            max_agents_per_owner,
            min_stake_amount,
            dispute_window,
            is_paused,
        )
    }

    /// Emergency pause marketplace
    pub fn emergency_pause(ctx: Context<EmergencyPause>) -> Result<()> {
        instructions::emergency_pause::emergency_pause(ctx)
    }

    /// Withdraw platform fees
    pub fn withdraw_fees(
        ctx: Context<WithdrawFees>,
        amount: u64,
    ) -> Result<()> {
        instructions::withdraw_fees::withdraw_fees(ctx, amount)
    }

    /// Slash agent stake for malicious behavior
    pub fn slash_stake(
        ctx: Context<SlashStake>,
        agent_id: String,
        amount: u64,
        reason: String,
    ) -> Result<()> {
        instructions::slash_stake::slash_stake(ctx, agent_id, amount, reason)
    }

    /// Migrate agent to new version
    pub fn migrate_agent(
        ctx: Context<MigrateAgent>,
        agent_id: String,
        new_metadata_uri: String,
        new_resource_requirements: ResourceRequirements,
    ) -> Result<()> {
        instructions::migrate_agent::migrate_agent(
            ctx,
            agent_id,
            new_metadata_uri,
            new_resource_requirements,
        )
    }
}

#[derive(Accounts)]
pub struct InitializeMarketplace<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + MarketplaceState::INIT_SPACE,
        seeds = [b"marketplace"],
        bump
    )]
    pub marketplace: Account<'info, MarketplaceState>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
    
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
#[instruction(agent_id: String)]
pub struct RegisterAgent<'info> {
    #[account(
        init,
        payer = owner,
        space = 8 + AgentAccount::INIT_SPACE,
        seeds = [b"agent", agent_id.as_bytes()],
        bump
    )]
    pub agent: Account<'info, AgentAccount>,
    
    #[account(mut)]
    pub marketplace: Account<'info, MarketplaceState>,
    
    #[account(mut)]
    pub owner: Signer<'info>,
    
    #[account(
        init_if_needed,
        payer = owner,
        space = 8 + ProviderAccount::INIT_SPACE,
        seeds = [b"provider", owner.key().as_ref()],
        bump
    )]
    pub provider: Account<'info, ProviderAccount>,
    
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
#[instruction(execution_id: String)]
pub struct ExecuteAgent<'info> {
    #[account(
        init,
        payer = user,
        space = 8 + ExecutionRecord::INIT_SPACE,
        seeds = [b"execution", execution_id.as_bytes()],
        bump
    )]
    pub execution: Account<'info, ExecutionRecord>,
    
    #[account(mut)]
    pub agent: Account<'info, AgentAccount>,
    
    #[account(mut)]
    pub marketplace: Account<'info, MarketplaceState>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
    #[account(mut)]
    pub user_token_account: Account<'info, TokenAccount>,
    
    #[account(mut)]
    pub agent_owner_token_account: Account<'info, TokenAccount>,
    
    #[account(mut)]
    pub marketplace_fee_account: Account<'info, TokenAccount>,
    
    pub token_program: Program<'info, Token>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
#[instruction(execution_id: String)]
pub struct CompleteExecution<'info> {
    #[account(
        mut,
        seeds = [b"execution", execution_id.as_bytes()],
        bump
    )]
    pub execution: Account<'info, ExecutionRecord>,
    
    #[account(mut)]
    pub agent: Account<'info, AgentAccount>,
    
    pub agent_owner: Signer<'info>,
}

#[derive(Accounts)]
#[instruction(execution_id: String)]
pub struct DisputeExecution<'info> {
    #[account(
        init,
        payer = disputer,
        space = 8 + DisputeRecord::INIT_SPACE,
        seeds = [b"dispute", execution_id.as_bytes()],
        bump
    )]
    pub dispute: Account<'info, DisputeRecord>,
    
    #[account(
        mut,
        seeds = [b"execution", execution_id.as_bytes()],
        bump
    )]
    pub execution: Account<'info, ExecutionRecord>,
    
    #[account(mut)]
    pub disputer: Signer<'info>,
    
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct StakeProvider<'info> {
    #[account(
        mut,
        seeds = [b"provider", provider.key().as_ref()],
        bump
    )]
    pub provider_account: Account<'info, ProviderAccount>,
    
    #[account(mut)]
    pub provider: Signer<'info>,
    
    #[account(mut)]
    pub provider_token_account: Account<'info, TokenAccount>,
    
    #[account(mut)]
    pub stake_vault: Account<'info, TokenAccount>,
    
    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
pub struct UpdateMarketplaceConfig<'info> {
    #[account(
        mut,
        seeds = [b"marketplace"],
        bump,
        has_one = authority
    )]
    pub marketplace: Account<'info, MarketplaceState>,
    
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct EmergencyPause<'info> {
    #[account(
        mut,
        seeds = [b"marketplace"],
        bump,
        has_one = authority
    )]
    pub marketplace: Account<'info, MarketplaceState>,
    
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct ResumeOperations<'info> {
    #[account(
        mut,
        seeds = [b"marketplace"],
        bump,
        has_one = authority
    )]
    pub marketplace: Account<'info, MarketplaceState>,
    
    pub authority: Signer<'info>,
}

// Additional account contexts for other instructions...
#[derive(Accounts)]
pub struct UpdateAgent<'info> {
    #[account(
        mut,
        seeds = [b"agent", agent.agent_id.as_bytes()],
        bump,
        has_one = owner
    )]
    pub agent: Account<'info, AgentAccount>,
    
    pub owner: Signer<'info>,
}

#[derive(Accounts)]
#[instruction(execution_id: String)]
pub struct ResolveDispute<'info> {
    #[account(
        mut,
        seeds = [b"dispute", execution_id.as_bytes()],
        bump
    )]
    pub dispute: Account<'info, DisputeRecord>,
    
    #[account(
        mut,
        seeds = [b"execution", execution_id.as_bytes()],
        bump
    )]
    pub execution: Account<'info, ExecutionRecord>,
    
    #[account(
        mut,
        seeds = [b"marketplace"],
        bump,
        has_one = authority
    )]
    pub marketplace: Account<'info, MarketplaceState>,
    
    pub authority: Signer<'info>,
    
    #[account(mut)]
    pub user_token_account: Account<'info, TokenAccount>,
    
    #[account(mut)]
    pub agent_owner_token_account: Account<'info, TokenAccount>,
    
    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
pub struct UnstakeProvider<'info> {
    #[account(
        mut,
        seeds = [b"provider", provider.key().as_ref()],
        bump
    )]
    pub provider_account: Account<'info, ProviderAccount>,
    
    #[account(mut)]
    pub provider: Signer<'info>,
    
    #[account(mut)]
    pub provider_token_account: Account<'info, TokenAccount>,
    
    #[account(mut)]
    pub stake_vault: Account<'info, TokenAccount>,
    
    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
pub struct SlashProvider<'info> {
    #[account(
        mut,
        seeds = [b"provider", provider_account.owner.as_ref()],
        bump
    )]
    pub provider_account: Account<'info, ProviderAccount>,
    
    #[account(
        mut,
        seeds = [b"marketplace"],
        bump,
        has_one = authority
    )]
    pub marketplace: Account<'info, MarketplaceState>,
    
    pub authority: Signer<'info>,
    
    #[account(mut)]
    pub stake_vault: Account<'info, TokenAccount>,
    
    #[account(mut)]
    pub slash_destination: Account<'info, TokenAccount>,
    
    pub token_program: Program<'info, Token>,
} 