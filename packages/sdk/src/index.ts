/**
 * StratosHub SDK - Official TypeScript SDK for StratosHub AI Agent Platform
 * 
 * Provides comprehensive access to StratosHub's marketplace, agent management,
 * and smart contract functionality on the Solana blockchain.
 * 
 * @packageDocumentation
 */

export { StratosClient } from './client';
export { AgentManager } from './agents';
export { ContractManager } from './contracts';
export { ModelManager } from './models';

// Type exports
export type {
  StratosClientConfig,
  StratosClientOptions,
  Cluster,
  Environment,
} from './types/client';

export type {
  Agent,
  AgentConfig,
  AgentDeployment,
  AgentExecution,
  AgentStatus,
  ModelType,
  ResourceRequirements,
  ExecutionResult,
  ExecutionStatus,
  AgentMetadata,
  AgentCapability,
  PricingModel,
  TriggerConfig,
  WebhookTrigger,
  ScheduleTrigger,
  EventTrigger,
} from './types/agents';

export type {
  MarketplaceState,
  AgentAccount,
  ExecutionRecord,
  ProviderAccount,
  DisputeRecord,
  DisputeStatus,
  DisputeResolution,
} from './types/contracts';

export type {
  Model,
  ModelProvider,
  ModelCategory,
  ModelConfiguration,
  InferenceOptions,
  ModelMetrics,
  ModelVersion,
} from './types/models';

export type {
  TransactionResult,
  InstructionResult,
  AccountInfo,
  ProgramDeployment,
  EventLog,
  ErrorCode,
  StratosError,
} from './types/common';

// Utility exports
export { 
  validateAgentConfig,
  validateResourceRequirements,
  validatePricingModel,
} from './utils/validation';

export {
  formatLamports,
  formatSOL,
  parseLamports,
  parseSOL,
} from './utils/currency';

export {
  generateAgentId,
  generateExecutionId,
  hashData,
  verifySignature,
} from './utils/crypto';

export {
  createAgentMetadata,
  uploadToIPFS,
  downloadFromIPFS,
} from './utils/metadata';

// Constants
export const PROGRAM_IDS = {
  MARKETPLACE: 'StratosHub11111111111111111111111111111111',
  ESCROW: 'StratosEscrow111111111111111111111111111111',
  REGISTRY: 'StratosRegistry11111111111111111111111111111',
  GOVERNANCE: 'StratosDAO1111111111111111111111111111111',
} as const;

export const CLUSTER_URLS = {
  mainnet: 'https://api.mainnet-beta.solana.com',
  devnet: 'https://api.devnet.solana.com',
  testnet: 'https://api.testnet.solana.com',
  localnet: 'http://localhost:8899',
} as const;

export const DEFAULT_CONFIG = {
  commitment: 'confirmed' as const,
  skipPreflight: false,
  maxRetries: 3,
  timeout: 30000,
  confirmTransactionInitialTimeout: 60000,
};

// Version
export const VERSION = '0.1.0'; 