use anchor_lang::prelude::*;
use anchor_spl::token::{self, Token, TokenAccount, Transfer};
use anchor_spl::associated_token::AssociatedToken;

declare_id!("StratosHub11111111111111111111111111111111");

pub mod instructions;
pub mod state;
pub mod errors;
pub mod events;

use instructions::*;
use state::*;
use errors::*;

#[program]
pub mod stratoshub_marketplace {
    use super::*;

    /// Initialize the global marketplace state
    pub fn initialize_marketplace(
        ctx: Context<InitializeMarketplace>,
        fee_basis_points: u16,
        max_agents_per_user: u32,
    ) -> Result<()> {
        instructions::initialize_marketplace(ctx, fee_basis_points, max_agents_per_user)
    }

    /// Register a new AI agent in the marketplace
    pub fn register_agent(
        ctx: Context<RegisterAgent>,
        agent_id: String,
        name: String,
        description: String,
        model_type: ModelType,
        price_per_execution: u64,
        metadata_uri: String,
        capabilities: Vec<String>,
        resource_requirements: ResourceRequirements,
    ) -> Result<()> {
        instructions::register_agent(
            ctx,
            agent_id,
            name,
            description,
            model_type,
            price_per_execution,
            metadata_uri,
            capabilities,
            resource_requirements,
        )
    }

    /// Update agent metadata and pricing
    pub fn update_agent(
        ctx: Context<UpdateAgent>,
        price_per_execution: Option<u64>,
        metadata_uri: Option<String>,
        is_active: Option<bool>,
        resource_requirements: Option<ResourceRequirements>,
    ) -> Result<()> {
        instructions::update_agent(ctx, price_per_execution, metadata_uri, is_active, resource_requirements)
    }

    /// Execute an agent and process payment
    pub fn execute_agent(
        ctx: Context<ExecuteAgent>,
        execution_id: String,
        input_data_hash: [u8; 32],
        expected_output_hash: Option<[u8; 32]>,
    ) -> Result<()> {
        instructions::execute_agent(ctx, execution_id, input_data_hash, expected_output_hash)
    }

    /// Complete agent execution with result verification
    pub fn complete_execution(
        ctx: Context<CompleteExecution>,
        execution_id: String,
        output_data_hash: [u8; 32],
        gas_used: u64,
        success: bool,
    ) -> Result<()> {
        instructions::complete_execution(ctx, execution_id, output_data_hash, gas_used, success)
    }

    /// Dispute an execution result
    pub fn dispute_execution(
        ctx: Context<DisputeExecution>,
        execution_id: String,
        dispute_reason: String,
        evidence_uri: String,
    ) -> Result<()> {
        instructions::dispute_execution(ctx, execution_id, dispute_reason, evidence_uri)
    }

    /// Resolve a disputed execution
    pub fn resolve_dispute(
        ctx: Context<ResolveDispute>,
        execution_id: String,
        resolution: DisputeResolution,
        refund_percentage: u8,
    ) -> Result<()> {
        instructions::resolve_dispute(ctx, execution_id, resolution, refund_percentage)
    }

    /// Stake tokens to become a verified agent provider
    pub fn stake_provider(
        ctx: Context<StakeProvider>,
        amount: u64,
    ) -> Result<()> {
        instructions::stake_provider(ctx, amount)
    }

    /// Withdraw staked tokens (after cooldown period)
    pub fn unstake_provider(
        ctx: Context<UnstakeProvider>,
        amount: u64,
    ) -> Result<()> {
        instructions::unstake_provider(ctx, amount)
    }

    /// Slash staked tokens for malicious behavior
    pub fn slash_provider(
        ctx: Context<SlashProvider>,
        amount: u64,
        reason: String,
    ) -> Result<()> {
        instructions::slash_provider(ctx, amount, reason)
    }

    /// Update marketplace configuration (admin only)
    pub fn update_marketplace_config(
        ctx: Context<UpdateMarketplaceConfig>,
        fee_basis_points: Option<u16>,
        max_agents_per_user: Option<u32>,
        min_stake_amount: Option<u64>,
        dispute_window: Option<i64>,
    ) -> Result<()> {
        instructions::update_marketplace_config(
            ctx,
            fee_basis_points,
            max_agents_per_user,
            min_stake_amount,
            dispute_window,
        )
    }

    /// Emergency pause for security incidents
    pub fn emergency_pause(ctx: Context<EmergencyPause>) -> Result<()> {
        instructions::emergency_pause(ctx)
    }

    /// Resume operations after emergency pause
    pub fn resume_operations(ctx: Context<ResumeOperations>) -> Result<()> {
        instructions::resume_operations(ctx)
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