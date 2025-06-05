/**
 * StratosHub SDK
 * TypeScript SDK for interacting with the StratosHub AI Agent Marketplace
 */

// Core client exports
export { StratosHubClient } from './client/StratosHubClient';
export { WalletAdapter } from './client/WalletAdapter';
export { ProgramClient } from './client/ProgramClient';

// Agent management exports
export { AgentManager } from './managers/AgentManager';
export { ExecutionManager } from './managers/ExecutionManager';
export { MarketplaceManager } from './managers/MarketplaceManager';
export { StakingManager } from './managers/StakingManager';

// Type definitions
export type {
  Agent,
  AgentStatus,
  AgentMetadata,
  AgentExecution,
  ExecutionStatus,
  ExecutionResult,
  ResourceRequirements,
  AgentCapability,
  ModelType,
  PricingModel,
} from './types/agent';

export type {
  MarketplaceConfig,
  MarketplaceStats,
  TradingPair,
  OrderBook,
  PriceHistory,
  VolumeMetrics,
} from './types/marketplace';

export type {
  StakeAccount,
  StakingPool,
  RewardMetrics,
  SlashingEvent,
  UnbondingEntry,
  StakingConfig,
} from './types/staking';

export type {
  Transaction,
  TransactionStatus,
  TransactionReceipt,
  GasEstimate,
  BlockchainConfig,
  NetworkInfo,
} from './types/blockchain';

export type {
  ApiResponse,
  PaginatedResponse,
  ErrorResponse,
  SortOrder,
  FilterOptions,
  SearchOptions,
} from './types/api';

// Utility exports
export { formatSOL, parseSOL } from './utils/currency';
export { validateAddress, validateSignature } from './utils/validation';
export { retryWithBackoff, timeout } from './utils/async';
export { createHash, verifyHash } from './utils/crypto';
export { logger } from './utils/logger';

// Constants
export {
  PROGRAM_IDS,
  NETWORK_CONFIGS,
  DEFAULT_RPC_ENDPOINTS,
  TOKEN_ADDRESSES,
  MARKETPLACE_CONSTANTS,
} from './constants';

// Error classes
export {
  StratosHubError,
  ValidationError,
  NetworkError,
  TransactionError,
  ContractError,
} from './errors';

// Default export
export { StratosHubClient as default } from './client/StratosHubClient'; 