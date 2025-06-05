import React from 'react';
import { Agent, AgentStatus } from '@stratoshub/sdk';
import { formatSOL } from '../utils/currency';
import { Badge } from './ui/Badge';
import { Button } from './ui/Button';

interface AgentCardProps {
  agent: Agent;
  onExecute: (agentId: string) => void;
  onViewDetails: (agentId: string) => void;
}

export const AgentCard: React.FC<AgentCardProps> = ({
  agent,
  onExecute,
  onViewDetails,
}) => {
  const getStatusColor = (status: AgentStatus) => {
    switch (status) {
      case 'active':
        return 'bg-green-100 text-green-800';
      case 'inactive':
        return 'bg-gray-100 text-gray-800';
      case 'error':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-yellow-100 text-yellow-800';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md border border-gray-200 p-6 hover:shadow-lg transition-shadow">
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            {agent.name}
          </h3>
          <p className="text-sm text-gray-600 line-clamp-2">
            {agent.description}
          </p>
        </div>
        <Badge className={getStatusColor(agent.status)}>
          {agent.status}
        </Badge>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <span className="text-xs text-gray-500 uppercase tracking-wide">
            Model Type
          </span>
          <p className="text-sm font-medium text-gray-900">
            {agent.modelType}
          </p>
        </div>
        <div>
          <span className="text-xs text-gray-500 uppercase tracking-wide">
            Price
          </span>
          <p className="text-sm font-medium text-gray-900">
            {formatSOL(agent.pricePerExecution)} SOL
          </p>
        </div>
      </div>

      <div className="mb-4">
        <span className="text-xs text-gray-500 uppercase tracking-wide">
          Capabilities
        </span>
        <div className="flex flex-wrap gap-1 mt-1">
          {agent.capabilities.slice(0, 3).map((capability) => (
            <span
              key={capability}
              className="inline-block bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded"
            >
              {capability}
            </span>
          ))}
          {agent.capabilities.length > 3 && (
            <span className="inline-block bg-gray-100 text-gray-800 text-xs px-2 py-1 rounded">
              +{agent.capabilities.length - 3} more
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4 mb-4 text-center">
        <div>
          <p className="text-sm font-medium text-gray-900">
            {agent.executionCount}
          </p>
          <p className="text-xs text-gray-500">Executions</p>
        </div>
        <div>
          <p className="text-sm font-medium text-gray-900">
            {agent.rating}%
          </p>
          <p className="text-xs text-gray-500">Success Rate</p>
        </div>
        <div>
          <p className="text-sm font-medium text-gray-900">
            {formatSOL(agent.totalRevenue)}
          </p>
          <p className="text-xs text-gray-500">Revenue</p>
        </div>
      </div>

      <div className="flex gap-2">
        <Button
          variant="primary"
          size="sm"
          onClick={() => onExecute(agent.id)}
          disabled={agent.status !== 'active'}
          className="flex-1"
        >
          Execute
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => onViewDetails(agent.id)}
          className="flex-1"
        >
          Details
        </Button>
      </div>
    </div>
  );
}; 