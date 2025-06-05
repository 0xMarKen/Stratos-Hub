import { Router } from 'express';
import { body, param, query, validationResult } from 'express-validator';
import { AgentService } from '../services/AgentService';
import { AuthMiddleware } from '../middleware/AuthMiddleware';
import { RateLimitMiddleware } from '../middleware/RateLimitMiddleware';
import { validateRequest } from '../middleware/ValidationMiddleware';

const router = Router();
const agentService = new AgentService();

// GET /api/v1/agents - List agents with pagination and filtering
router.get(
  '/',
  [
    query('page').optional().isInt({ min: 1 }).toInt(),
    query('limit').optional().isInt({ min: 1, max: 100 }).toInt(),
    query('owner').optional().isString(),
    query('modelType').optional().isString(),
    query('status').optional().isIn(['active', 'inactive', 'error']),
    query('minPrice').optional().isFloat({ min: 0 }),
    query('maxPrice').optional().isFloat({ min: 0 }),
    query('search').optional().isString().trim(),
    validateRequest,
  ],
  async (req, res) => {
    try {
      const {
        page = 1,
        limit = 20,
        owner,
        modelType,
        status,
        minPrice,
        maxPrice,
        search,
      } = req.query;

      const result = await agentService.listAgents({
        page: Number(page),
        limit: Number(limit),
        filters: {
          owner,
          modelType,
          status,
          priceRange: minPrice || maxPrice ? { min: minPrice, max: maxPrice } : undefined,
          search,
        },
      });

      res.json({
        success: true,
        data: result.agents,
        pagination: {
          page: Number(page),
          limit: Number(limit),
          total: result.total,
          totalPages: Math.ceil(result.total / Number(limit)),
        },
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        error: 'Failed to fetch agents',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }
);

// GET /api/v1/agents/:agentId - Get specific agent details
router.get(
  '/:agentId',
  [param('agentId').isString().notEmpty(), validateRequest],
  async (req, res) => {
    try {
      const { agentId } = req.params;
      const agent = await agentService.getAgent(agentId);

      if (!agent) {
        return res.status(404).json({
          success: false,
          error: 'Agent not found',
        });
      }

      res.json({
        success: true,
        data: agent,
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        error: 'Failed to fetch agent',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }
);

// POST /api/v1/agents - Register new agent
router.post(
  '/',
  AuthMiddleware.requireAuth,
  RateLimitMiddleware.agentCreation,
  [
    body('name').isString().isLength({ min: 1, max: 128 }).trim(),
    body('description').isString().isLength({ min: 1, max: 512 }).trim(),
    body('modelType').isIn([
      'text-generation',
      'text-classification',
      'image-generation',
      'image-classification',
      'audio-generation',
      'audio-transcription',
      'data-analysis',
      'predictive-modeling',
      'reinforcement-learning',
      'custom',
    ]),
    body('pricePerExecution').isFloat({ min: 0 }),
    body('metadataUri').isURL(),
    body('capabilities').isArray().isLength({ min: 1, max: 10 }),
    body('capabilities.*').isString().isLength({ min: 1, max: 64 }),
    body('resourceRequirements.memoryMb').isInt({ min: 128, max: 32768 }),
    body('resourceRequirements.cpuCores').isInt({ min: 1, max: 16 }),
    body('resourceRequirements.gpuMemoryMb').optional().isInt({ min: 0 }),
    body('resourceRequirements.maxExecutionTime').isInt({ min: 1, max: 3600 }),
    body('resourceRequirements.diskSpaceMb').isInt({ min: 100, max: 10240 }),
    validateRequest,
  ],
  async (req, res) => {
    try {
      const agentData = req.body;
      const owner = req.user.publicKey;

      const agent = await agentService.registerAgent({
        ...agentData,
        owner,
      });

      res.status(201).json({
        success: true,
        data: agent,
        message: 'Agent registered successfully',
      });
    } catch (error) {
      res.status(400).json({
        success: false,
        error: 'Failed to register agent',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }
);

// PUT /api/v1/agents/:agentId - Update agent
router.put(
  '/:agentId',
  AuthMiddleware.requireAuth,
  [
    param('agentId').isString().notEmpty(),
    body('pricePerExecution').optional().isFloat({ min: 0 }),
    body('metadataUri').optional().isURL(),
    body('isActive').optional().isBoolean(),
    body('resourceRequirements.memoryMb').optional().isInt({ min: 128, max: 32768 }),
    body('resourceRequirements.cpuCores').optional().isInt({ min: 1, max: 16 }),
    body('resourceRequirements.gpuMemoryMb').optional().isInt({ min: 0 }),
    body('resourceRequirements.maxExecutionTime').optional().isInt({ min: 1, max: 3600 }),
    validateRequest,
  ],
  async (req, res) => {
    try {
      const { agentId } = req.params;
      const updateData = req.body;
      const owner = req.user.publicKey;

      const agent = await agentService.updateAgent(agentId, updateData, owner);

      res.json({
        success: true,
        data: agent,
        message: 'Agent updated successfully',
      });
    } catch (error) {
      if (error instanceof Error && error.message.includes('not found')) {
        return res.status(404).json({
          success: false,
          error: 'Agent not found',
        });
      }

      if (error instanceof Error && error.message.includes('unauthorized')) {
        return res.status(403).json({
          success: false,
          error: 'Unauthorized to update this agent',
        });
      }

      res.status(400).json({
        success: false,
        error: 'Failed to update agent',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }
);

// POST /api/v1/agents/:agentId/execute - Execute agent
router.post(
  '/:agentId/execute',
  AuthMiddleware.requireAuth,
  RateLimitMiddleware.agentExecution,
  [
    param('agentId').isString().notEmpty(),
    body('inputData').isObject(),
    body('options.temperature').optional().isFloat({ min: 0, max: 2 }),
    body('options.maxTokens').optional().isInt({ min: 1, max: 4096 }),
    body('options.timeout').optional().isInt({ min: 1, max: 300 }),
    validateRequest,
  ],
  async (req, res) => {
    try {
      const { agentId } = req.params;
      const { inputData, options = {} } = req.body;
      const user = req.user.publicKey;

      const execution = await agentService.executeAgent({
        agentId,
        user,
        inputData,
        options,
      });

      res.status(202).json({
        success: true,
        data: execution,
        message: 'Agent execution initiated',
      });
    } catch (error) {
      res.status(400).json({
        success: false,
        error: 'Failed to execute agent',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }
);

// GET /api/v1/agents/:agentId/executions - Get agent execution history
router.get(
  '/:agentId/executions',
  AuthMiddleware.requireAuth,
  [
    param('agentId').isString().notEmpty(),
    query('page').optional().isInt({ min: 1 }).toInt(),
    query('limit').optional().isInt({ min: 1, max: 100 }).toInt(),
    query('status').optional().isIn(['pending', 'running', 'completed', 'failed']),
    validateRequest,
  ],
  async (req, res) => {
    try {
      const { agentId } = req.params;
      const { page = 1, limit = 20, status } = req.query;
      const user = req.user.publicKey;

      const result = await agentService.getExecutionHistory({
        agentId,
        user,
        page: Number(page),
        limit: Number(limit),
        status,
      });

      res.json({
        success: true,
        data: result.executions,
        pagination: {
          page: Number(page),
          limit: Number(limit),
          total: result.total,
          totalPages: Math.ceil(result.total / Number(limit)),
        },
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        error: 'Failed to fetch execution history',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }
);

// GET /api/v1/agents/:agentId/metrics - Get agent performance metrics
router.get(
  '/:agentId/metrics',
  [
    param('agentId').isString().notEmpty(),
    query('timeframe').optional().isIn(['24h', '7d', '30d', '90d']),
    validateRequest,
  ],
  async (req, res) => {
    try {
      const { agentId } = req.params;
      const { timeframe = '24h' } = req.query;

      const metrics = await agentService.getAgentMetrics(agentId, timeframe);

      res.json({
        success: true,
        data: metrics,
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        error: 'Failed to fetch agent metrics',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }
);

export default router; 