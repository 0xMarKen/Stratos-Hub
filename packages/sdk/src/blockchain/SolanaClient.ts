/**
 * Solana Blockchain Client
 * 
 * Comprehensive Solana integration for StratosHub AI Agent Platform
 * with smart contract interactions, transaction management, and monitoring.
 */

import {
  Connection,
  PublicKey,
  Transaction,
  TransactionInstruction,
  SystemProgram,
  LAMPORTS_PER_SOL,
  SendOptions,
  Commitment,
  TransactionSignature,
  AccountInfo,
  GetProgramAccountsFilter,
  TokenAccountsFilter,
  ParsedAccountData,
  ConfirmedSignatureInfo,
  TransactionResponse,
  VersionedTransaction,
} from '@solana/web3.js';
import {
  TOKEN_PROGRAM_ID,
  ASSOCIATED_TOKEN_PROGRAM_ID,
  getAssociatedTokenAddress,
  createAssociatedTokenAccountInstruction,
  createTransferInstruction,
  getAccount,
  TokenAccountNotFoundError,
  TokenInvalidAccountOwnerError,
} from '@solana/spl-token';
import { AnchorProvider, Program, Wallet, BN, IdlAccounts } from '@coral-xyz/anchor';
import { StratosHubIDL } from '../idl/stratoshub';

export interface SolanaConfig {
  rpcEndpoint: string;
  wsEndpoint?: string;
  commitment?: Commitment;
  confirmTransactionInitialTimeout?: number;
  skipPreflight?: boolean;
  maxRetries?: number;
  retryDelay?: number;
}

export interface TransactionOptions {
  skipPreflight?: boolean;
  preflightCommitment?: Commitment;
  maxRetries?: number;
  timeout?: number;
  priorityFee?: number;
}

export interface AccountFilter {
  memcmp?: {
    offset: number;
    bytes: string;
  };
  dataSize?: number;
}

export interface TokenBalance {
  mint: string;
  owner: string;
  amount: string;
  decimals: number;
  uiAmount: number | null;
}

export interface TransactionMetrics {
  signature: string;
  blockTime: number | null;
  slot: number;
  fee: number;
  computeUnitsConsumed?: number;
  logMessages: string[];
  status: 'success' | 'failed';
  error?: string;
}

export class SolanaClient {
  private connection: Connection;
  private config: SolanaConfig;
  private provider?: AnchorProvider;
  private program?: Program<typeof StratosHubIDL>;
  private retryCount: Map<string, number> = new Map();

  constructor(config: SolanaConfig, wallet?: Wallet) {
    this.config = {
      commitment: 'confirmed',
      confirmTransactionInitialTimeout: 60000,
      skipPreflight: false,
      maxRetries: 3,
      retryDelay: 1000,
      ...config,
    };

    this.connection = new Connection(
      config.rpcEndpoint,
      {
        commitment: this.config.commitment,
        wsEndpoint: config.wsEndpoint,
        confirmTransactionInitialTimeout: this.config.confirmTransactionInitialTimeout,
      }
    );

    if (wallet) {
      this.initializeAnchor(wallet);
    }
  }

  /**
   * Initialize Anchor provider and program
   */
  private initializeAnchor(wallet: Wallet) {
    this.provider = new AnchorProvider(
      this.connection,
      wallet,
      {
        commitment: this.config.commitment,
        skipPreflight: this.config.skipPreflight,
      }
    );

    this.program = new Program(
      StratosHubIDL,
      new PublicKey("StratosHub11111111111111111111111111111111"),
      this.provider
    );
  }

  /**
   * Get connection info and health status
   */
  async getConnectionInfo(): Promise<{
    version: string;
    slot: number;
    blockHeight: number;
    health: string;
    tps: number;
  }> {
    try {
      const [version, slot, blockHeight] = await Promise.all([
        this.connection.getVersion(),
        this.connection.getSlot(),
        this.connection.getBlockHeight(),
      ]);

      // Get recent performance samples for TPS calculation
      const perfSamples = await this.connection.getRecentPerformanceSamples(1);
      const tps = perfSamples.length > 0 ? perfSamples[0].samplePeriodSecs > 0 
        ? perfSamples[0].numTransactions / perfSamples[0].samplePeriodSecs 
        : 0 : 0;

      return {
        version: version.solanaCore,
        slot,
        blockHeight,
        health: 'ok',
        tps: Math.round(tps),
      };
    } catch (error) {
      throw new SolanaError(`Failed to get connection info: ${error.message}`);
    }
  }

  /**
   * Get account balance in SOL
   */
  async getBalance(publicKey: PublicKey): Promise<number> {
    try {
      const balance = await this.connection.getBalance(publicKey);
      return balance / LAMPORTS_PER_SOL;
    } catch (error) {
      throw new SolanaError(`Failed to get balance: ${error.message}`);
    }
  }

  /**
   * Get all token balances for an account
   */
  async getTokenBalances(owner: PublicKey): Promise<TokenBalance[]> {
    try {
      const tokenAccounts = await this.connection.getParsedTokenAccountsByOwner(
        owner,
        { programId: TOKEN_PROGRAM_ID }
      );

      return tokenAccounts.value.map(account => {
        const parsedInfo = account.account.data.parsed.info;
        return {
          mint: parsedInfo.mint,
          owner: parsedInfo.owner,
          amount: parsedInfo.tokenAmount.amount,
          decimals: parsedInfo.tokenAmount.decimals,
          uiAmount: parsedInfo.tokenAmount.uiAmount,
        };
      });
    } catch (error) {
      throw new SolanaError(`Failed to get token balances: ${error.message}`);
    }
  }

  /**
   * Transfer SOL between accounts
   */
  async transferSOL(
    from: PublicKey,
    to: PublicKey,
    amount: number,
    options: TransactionOptions = {}
  ): Promise<string> {
    try {
      const lamports = Math.round(amount * LAMPORTS_PER_SOL);
      
      const instruction = SystemProgram.transfer({
        fromPubkey: from,
        toPubkey: to,
        lamports,
      });

      const transaction = new Transaction().add(instruction);
      
      return await this.sendTransaction(transaction, options);
    } catch (error) {
      throw new SolanaError(`Failed to transfer SOL: ${error.message}`);
    }
  }

  /**
   * Transfer SPL tokens
   */
  async transferToken(
    from: PublicKey,
    to: PublicKey,
    mint: PublicKey,
    amount: number,
    decimals: number,
    options: TransactionOptions = {}
  ): Promise<string> {
    try {
      // Get or create associated token accounts
      const fromTokenAccount = await getAssociatedTokenAddress(mint, from);
      const toTokenAccount = await getAssociatedTokenAddress(mint, to);

      const instructions: TransactionInstruction[] = [];

      // Check if destination token account exists
      try {
        await getAccount(this.connection, toTokenAccount);
      } catch (error) {
        if (error instanceof TokenAccountNotFoundError) {
          // Create associated token account for recipient
          instructions.push(
            createAssociatedTokenAccountInstruction(
              from, // payer
              toTokenAccount,
              to, // owner
              mint
            )
          );
        } else {
          throw error;
        }
      }

      // Create transfer instruction
      const transferAmount = BigInt(amount * Math.pow(10, decimals));
      instructions.push(
        createTransferInstruction(
          fromTokenAccount,
          toTokenAccount,
          from,
          transferAmount
        )
      );

      const transaction = new Transaction().add(...instructions);
      
      return await this.sendTransaction(transaction, options);
    } catch (error) {
      throw new SolanaError(`Failed to transfer token: ${error.message}`);
    }
  }

  /**
   * Send a transaction with retry logic
   */
  async sendTransaction(
    transaction: Transaction | VersionedTransaction,
    options: TransactionOptions = {}
  ): Promise<string> {
    const {
      skipPreflight = this.config.skipPreflight,
      maxRetries = this.config.maxRetries,
      timeout = 30000,
      priorityFee = 0,
    } = options;

    if (!this.provider) {
      throw new SolanaError('Wallet not initialized');
    }

    let lastError: Error;
    const maxAttempts = maxRetries! + 1;

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        // Add priority fee if specified
        if (priorityFee > 0 && transaction instanceof Transaction) {
          const priorityFeeIx = SystemProgram.transfer({
            fromPubkey: this.provider.publicKey,
            toPubkey: this.provider.publicKey,
            lamports: priorityFee,
          });
          transaction.instructions.unshift(priorityFeeIx);
        }

        // Get recent blockhash
        const { blockhash, lastValidBlockHeight } = await this.connection.getLatestBlockhash();
        
        if (transaction instanceof Transaction) {
          transaction.recentBlockhash = blockhash;
          transaction.feePayer = this.provider.publicKey;
        }

        // Sign transaction
        const signedTx = await this.provider.wallet.signTransaction(transaction);

        // Send transaction
        const signature = await this.connection.sendRawTransaction(
          signedTx.serialize(),
          {
            skipPreflight,
            maxRetries: 0, // We handle retries ourselves
          }
        );

        // Confirm transaction
        const confirmation = await this.connection.confirmTransaction(
          {
            signature,
            blockhash,
            lastValidBlockHeight,
          },
          this.config.commitment
        );

        if (confirmation.value.err) {
          throw new SolanaError(`Transaction failed: ${JSON.stringify(confirmation.value.err)}`);
        }

        return signature;

      } catch (error) {
        lastError = error as Error;
        
        if (attempt < maxAttempts - 1) {
          const delay = this.config.retryDelay! * Math.pow(2, attempt);
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }

    throw new SolanaError(`Transaction failed after ${maxAttempts} attempts: ${lastError!.message}`);
  }

  /**
   * Get transaction details
   */
  async getTransaction(signature: string): Promise<TransactionResponse | null> {
    try {
      return await this.connection.getTransaction(signature, {
        commitment: 'confirmed',
        maxSupportedTransactionVersion: 0,
      });
    } catch (error) {
      throw new SolanaError(`Failed to get transaction: ${error.message}`);
    }
  }

  /**
   * Get transaction metrics
   */
  async getTransactionMetrics(signature: string): Promise<TransactionMetrics | null> {
    try {
      const transaction = await this.getTransaction(signature);
      
      if (!transaction) {
        return null;
      }

      return {
        signature,
        blockTime: transaction.blockTime,
        slot: transaction.slot,
        fee: transaction.meta?.fee || 0,
        computeUnitsConsumed: transaction.meta?.computeUnitsConsumed,
        logMessages: transaction.meta?.logMessages || [],
        status: transaction.meta?.err ? 'failed' : 'success',
        error: transaction.meta?.err ? JSON.stringify(transaction.meta.err) : undefined,
      };
    } catch (error) {
      throw new SolanaError(`Failed to get transaction metrics: ${error.message}`);
    }
  }

  /**
   * Get account signatures (transaction history)
   */
  async getAccountSignatures(
    account: PublicKey,
    limit: number = 100,
    before?: string,
    until?: string
  ): Promise<ConfirmedSignatureInfo[]> {
    try {
      return await this.connection.getSignaturesForAddress(account, {
        limit,
        before,
        until,
      });
    } catch (error) {
      throw new SolanaError(`Failed to get account signatures: ${error.message}`);
    }
  }

  /**
   * Get program accounts with filters
   */
  async getProgramAccounts(
    programId: PublicKey,
    filters: AccountFilter[] = []
  ): Promise<Array<{ pubkey: PublicKey; account: AccountInfo<Buffer> }>> {
    try {
      const programFilters: GetProgramAccountsFilter[] = filters.map(filter => {
        if (filter.memcmp) {
          return { memcmp: filter.memcmp };
        }
        if (filter.dataSize) {
          return { dataSize: filter.dataSize };
        }
        throw new Error('Invalid filter');
      });

      return await this.connection.getProgramAccounts(programId, {
        filters: programFilters,
      });
    } catch (error) {
      throw new SolanaError(`Failed to get program accounts: ${error.message}`);
    }
  }

  /**
   * Subscribe to account changes
   */
  subscribeToAccount(
    account: PublicKey,
    callback: (accountInfo: AccountInfo<Buffer>, context: { slot: number }) => void
  ): number {
    return this.connection.onAccountChange(account, callback, this.config.commitment);
  }

  /**
   * Subscribe to program account changes
   */
  subscribeToProgramAccounts(
    programId: PublicKey,
    callback: (keyedAccountInfo: {
      accountId: PublicKey;
      accountInfo: AccountInfo<Buffer>;
    }, context: { slot: number }) => void,
    filters: AccountFilter[] = []
  ): number {
    const programFilters: GetProgramAccountsFilter[] = filters.map(filter => {
      if (filter.memcmp) {
        return { memcmp: filter.memcmp };
      }
      if (filter.dataSize) {
        return { dataSize: filter.dataSize };
      }
      throw new Error('Invalid filter');
    });

    return this.connection.onProgramAccountChange(
      programId,
      callback,
      this.config.commitment,
      programFilters
    );
  }

  /**
   * Unsubscribe from account changes
   */
  async unsubscribe(subscriptionId: number): Promise<void> {
    try {
      await this.connection.removeAccountChangeListener(subscriptionId);
    } catch (error) {
      throw new SolanaError(`Failed to unsubscribe: ${error.message}`);
    }
  }

  /**
   * Estimate transaction fees
   */
  async estimateTransactionFee(
    transaction: Transaction | VersionedTransaction
  ): Promise<number> {
    try {
      if (transaction instanceof VersionedTransaction) {
        const fee = await this.connection.getFeeForMessage(
          transaction.message,
          this.config.commitment
        );
        return fee.value || 0;
      } else {
        // Get recent blockhash for fee calculation
        const { blockhash } = await this.connection.getLatestBlockhash();
        transaction.recentBlockhash = blockhash;
        
        const fee = await this.connection.getFeeForMessage(
          transaction.compileMessage(),
          this.config.commitment
        );
        return fee.value || 0;
      }
    } catch (error) {
      throw new SolanaError(`Failed to estimate transaction fee: ${error.message}`);
    }
  }

  /**
   * Get multiple accounts in a single RPC call
   */
  async getMultipleAccounts(
    publicKeys: PublicKey[]
  ): Promise<(AccountInfo<Buffer> | null)[]> {
    try {
      const result = await this.connection.getMultipleAccountsInfo(publicKeys);
      return result;
    } catch (error) {
      throw new SolanaError(`Failed to get multiple accounts: ${error.message}`);
    }
  }

  /**
   * Wait for transaction confirmation with timeout
   */
  async waitForConfirmation(
    signature: string,
    timeout: number = 30000
  ): Promise<boolean> {
    const start = Date.now();
    
    while (Date.now() - start < timeout) {
      try {
        const status = await this.connection.getSignatureStatus(signature);
        
        if (status.value?.confirmationStatus === 'confirmed' || 
            status.value?.confirmationStatus === 'finalized') {
          return !status.value.err;
        }
        
        // Wait before checking again
        await new Promise(resolve => setTimeout(resolve, 1000));
      } catch (error) {
        // Continue waiting unless it's a critical error
        continue;
      }
    }
    
    throw new SolanaError(`Transaction confirmation timeout after ${timeout}ms`);
  }

  /**
   * Get current slot and block time
   */
  async getCurrentSlotInfo(): Promise<{ slot: number; blockTime: number | null }> {
    try {
      const slot = await this.connection.getSlot();
      const blockTime = await this.connection.getBlockTime(slot);
      return { slot, blockTime };
    } catch (error) {
      throw new SolanaError(`Failed to get slot info: ${error.message}`);
    }
  }

  /**
   * Monitor network performance
   */
  async getNetworkPerformance(): Promise<{
    tps: number;
    avgSlotTime: number;
    epochInfo: any;
  }> {
    try {
      const [perfSamples, epochInfo] = await Promise.all([
        this.connection.getRecentPerformanceSamples(60),
        this.connection.getEpochInfo(),
      ]);

      // Calculate average TPS
      const totalTransactions = perfSamples.reduce((sum, sample) => sum + sample.numTransactions, 0);
      const totalTime = perfSamples.reduce((sum, sample) => sum + sample.samplePeriodSecs, 0);
      const tps = totalTime > 0 ? totalTransactions / totalTime : 0;

      // Calculate average slot time
      const totalSlots = perfSamples.reduce((sum, sample) => sum + sample.numSlots, 0);
      const avgSlotTime = totalSlots > 0 ? totalTime / totalSlots : 0;

      return {
        tps: Math.round(tps),
        avgSlotTime: Math.round(avgSlotTime * 1000), // Convert to milliseconds
        epochInfo,
      };
    } catch (error) {
      throw new SolanaError(`Failed to get network performance: ${error.message}`);
    }
  }

  /**
   * Get Anchor program instance
   */
  getProgram(): Program<typeof StratosHubIDL> {
    if (!this.program) {
      throw new SolanaError('Anchor program not initialized');
    }
    return this.program;
  }

  /**
   * Get connection instance
   */
  getConnection(): Connection {
    return this.connection;
  }

  /**
   * Close connection
   */
  async close(): Promise<void> {
    // Close WebSocket connections
    // Note: The Connection class doesn't have a direct close method
    // This would be implementation-specific based on WebSocket handling
  }
}

/**
 * Custom error class for Solana operations
 */
export class SolanaError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SolanaError';
  }
}

/**
 * Utility functions
 */
export const SolanaUtils = {
  /**
   * Convert lamports to SOL
   */
  lamportsToSOL: (lamports: number): number => {
    return lamports / LAMPORTS_PER_SOL;
  },

  /**
   * Convert SOL to lamports
   */
  solToLamports: (sol: number): number => {
    return Math.round(sol * LAMPORTS_PER_SOL);
  },

  /**
   * Check if a public key is valid
   */
  isValidPublicKey: (publicKey: string): boolean => {
    try {
      new PublicKey(publicKey);
      return true;
    } catch {
      return false;
    }
  },

  /**
   * Shorten a public key for display
   */
  shortenPublicKey: (publicKey: string, chars: number = 4): string => {
    return `${publicKey.slice(0, chars)}...${publicKey.slice(-chars)}`;
  },

  /**
   * Generate a random keypair
   */
  generateKeypair: () => {
    const { Keypair } = require('@solana/web3.js');
    return Keypair.generate();
  },

  /**
   * Calculate rent exemption
   */
  calculateRentExemption: async (connection: Connection, dataLength: number): Promise<number> => {
    return await connection.getMinimumBalanceForRentExemption(dataLength);
  },
};

export default SolanaClient; 