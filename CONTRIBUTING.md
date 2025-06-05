# Contributing to StratosHub

We welcome contributions to StratosHub! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Security Vulnerabilities](#security-vulnerabilities)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

Before contributing, ensure you have the following installed:

- **Node.js** (>=18.0.0)
- **Rust** (>=1.70.0)
- **Solana CLI** (>=1.16.0)
- **Anchor CLI** (>=0.29.0)
- **Docker** (>=20.10.0)
- **Python** (>=3.11)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/stratoshub.git
   cd stratoshub
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/stratoshub/stratoshub.git
   ```

## Development Setup

### Local Environment

1. Install dependencies:
   ```bash
   npm install
   ```

2. Copy environment variables:
   ```bash
   cp .env.example .env.local
   ```

3. Start the development environment:
   ```bash
   docker-compose up -d
   npm run dev
   ```

4. Deploy smart contracts locally:
   ```bash
   anchor build
   anchor deploy --provider.cluster localnet
   ```

### IDE Configuration

#### VS Code

Install recommended extensions:
```bash
code --install-extension rust-lang.rust-analyzer
code --install-extension ms-vscode.vscode-typescript-next
code --install-extension bradlc.vscode-tailwindcss
code --install-extension ms-python.python
```

## Contributing Guidelines

### Branching Strategy

- **main**: Production-ready code
- **develop**: Integration branch for features
- **feature/**: New features (`feature/agent-versioning`)
- **fix/**: Bug fixes (`fix/execution-timeout`)
- **docs/**: Documentation updates (`docs/api-reference`)
- **chore/**: Maintenance tasks (`chore/update-dependencies`)

### Commit Convention

We follow the [Conventional Commits](https://conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation only changes
- **style**: Code style changes (formatting, missing semi-colons, etc)
- **refactor**: Code change that neither fixes a bug nor adds a feature
- **perf**: Performance improvements
- **test**: Adding missing tests or correcting existing tests
- **chore**: Changes to build process or auxiliary tools

Examples:
```bash
feat(marketplace): add agent execution dispute resolution
fix(sdk): handle connection timeout in agent deployment
docs(api): update authentication endpoints documentation
chore(deps): update anchor to v0.29.0
```

### Code Style

#### TypeScript/JavaScript
- Use TypeScript for all new code
- Follow ESLint configuration
- Use Prettier for formatting
- Prefer functional programming patterns
- Include comprehensive JSDoc comments

```typescript
/**
 * Deploys an AI agent to the StratosHub marketplace
 * @param config - Agent configuration including model and pricing
 * @returns Promise resolving to deployment result
 */
async function deployAgent(config: AgentConfig): Promise<AgentDeployment> {
  // Implementation
}
```

#### Rust
- Follow `rustfmt` formatting
- Use `clippy` for linting
- Include comprehensive documentation
- Write idiomatic Rust code

```rust
/// Registers a new agent in the marketplace
/// 
/// # Arguments
/// * `ctx` - The instruction context
/// * `agent_id` - Unique identifier for the agent
/// * `config` - Agent configuration parameters
/// 
/// # Errors
/// Returns `ErrorCode::InvalidAgentId` if agent_id is malformed
pub fn register_agent(
    ctx: Context<RegisterAgent>,
    agent_id: String,
    config: AgentConfig,
) -> Result<()> {
    // Implementation
}
```

#### Python
- Follow PEP 8 style guide
- Use type hints for all functions
- Include docstrings for all public functions
- Use Black for formatting

```python
async def execute_agent(
    agent_id: str,
    input_data: Dict[str, Any],
    options: Optional[ExecutionOptions] = None,
) -> ExecutionResult:
    """
    Execute an AI agent with the provided input data.
    
    Args:
        agent_id: Unique identifier of the agent to execute
        input_data: Input parameters for the agent
        options: Optional execution configuration
        
    Returns:
        ExecutionResult containing output and metadata
        
    Raises:
        AgentNotFoundError: If the specified agent doesn't exist
        ExecutionTimeoutError: If execution exceeds time limit
    """
    # Implementation
```

## Pull Request Process

### Before Submitting

1. **Sync with upstream**:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**:
   - Write tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass
   - Follow code style guidelines

4. **Test thoroughly**:
   ```bash
   npm run test
   npm run test:integration
   npm run test:e2e
   anchor test
   ```

5. **Lint and format**:
   ```bash
   npm run lint:fix
   npm run format
   ```

### Submitting the PR

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create pull request** on GitHub with:
   - Clear title following conventional commits
   - Detailed description of changes
   - Link to related issues
   - Screenshots/videos for UI changes
   - Testing instructions

3. **PR Template**:
   ```markdown
   ## Description
   Brief description of changes made.

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update

   ## How Has This Been Tested?
   Describe the tests you ran and how to reproduce them.

   ## Checklist
   - [ ] My code follows the style guidelines
   - [ ] I have performed a self-review
   - [ ] I have commented my code where necessary
   - [ ] I have made corresponding changes to documentation
   - [ ] My changes generate no new warnings
   - [ ] I have added tests that prove my fix/feature works
   - [ ] New and existing unit tests pass locally
   ```

### Review Process

1. **Automated Checks**: All CI checks must pass
2. **Code Review**: At least one maintainer approval required
3. **Testing**: Reviewer may request additional testing
4. **Documentation**: Ensure documentation is updated
5. **Merge**: Maintainer will merge after approval

## Issue Reporting

### Bug Reports

Include the following information:

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
- OS: [e.g. macOS 14.0]
- Node.js version: [e.g. 18.17.0]
- Solana CLI version: [e.g. 1.16.0]
- Browser: [e.g. Chrome 120.0]

**Additional context**
Any other context about the problem.
```

### Feature Requests

```markdown
**Is your feature request related to a problem?**
A clear description of what the problem is.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Alternative solutions or features you've considered.

**Additional context**
Any other context or screenshots about the feature request.
```

## Security Vulnerabilities

**Do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to security@stratoshub.io. Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fixes (if any)

We will respond within 48 hours and provide a timeline for resolution.

## Development Workflow

### Daily Development

1. **Start development environment**:
   ```bash
   docker-compose up -d
   npm run dev
   ```

2. **Make changes** and test locally

3. **Run tests before committing**:
   ```bash
   npm run test:unit
   npm run lint
   ```

4. **Commit with conventional format**:
   ```bash
   git add .
   git commit -m "feat(component): add new functionality"
   ```

### Smart Contract Development

1. **Write tests first**:
   ```typescript
   // tests/marketplace.ts
   describe("Agent Registration", () => {
     it("should register agent successfully", async () => {
       // Test implementation
     });
   });
   ```

2. **Implement functionality**:
   ```rust
   // programs/marketplace/src/instructions/register_agent.rs
   pub fn register_agent(ctx: Context<RegisterAgent>) -> Result<()> {
     // Implementation
   }
   ```

3. **Test thoroughly**:
   ```bash
   anchor test
   ```

4. **Deploy and verify**:
   ```bash
   anchor build
   anchor deploy --provider.cluster devnet
   ```

### Frontend Development

1. **Component-driven development**:
   - Create components in Storybook first
   - Implement with TypeScript and proper types
   - Add unit tests

2. **Integration testing**:
   ```bash
   npm run test:e2e
   ```

3. **Accessibility testing**:
   - Test with screen readers
   - Ensure keyboard navigation
   - Check color contrast

## Testing

### Test Categories

1. **Unit Tests**: Test individual functions/components
2. **Integration Tests**: Test service interactions
3. **E2E Tests**: Test complete user workflows
4. **Smart Contract Tests**: Test on-chain functionality
5. **Load Tests**: Test performance under load

### Test Requirements

- All new features must include tests
- Aim for >90% code coverage
- Tests should be deterministic and fast
- Use appropriate mocking for external services

### Running Tests

```bash
# All tests
npm run test

# Specific test suites
npm run test:unit
npm run test:integration
npm run test:e2e

# Smart contract tests
anchor test

# Load tests
npm run test:load
```

## Documentation

### Types of Documentation

1. **API Documentation**: Generated from code comments
2. **User Guides**: Step-by-step instructions
3. **Architecture Docs**: System design and patterns
4. **Deployment Guides**: Infrastructure setup

### Documentation Standards

- Use clear, concise language
- Include code examples
- Provide context and motivation
- Keep documentation up-to-date with code changes

### Building Documentation

```bash
# API documentation
npm run docs:generate

# User documentation
npm run docs:build

# Serve locally
npm run docs:serve
```

## Recognition

Contributors will be recognized in:
- GitHub contributors list
- Release notes for significant contributions
- Annual contributor recognition

Thank you for contributing to StratosHub! Your efforts help build the future of decentralized AI infrastructure. 