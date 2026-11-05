# Multi-Agent GPU Optimization System

A hierarchical multi-agent system for optimizing Triton GPU kernels through coordinated analysis and optimization at different architectural levels.

## Architecture Overview

The system consists of four hierarchy levels:

```
┌─────────────────────────────────────────┐
│         COORDINATOR                     │  ← Resolves conflicts, makes final decisions
│   (Global optimization strategy)        │
└────────────────┬────────────────────────┘
                 │
┌────────────────┴────────────────────────┐
│             GPU LEVEL                   │  ← Grid dimensions, CTA allocation
│   (Cross-SM resource allocation)        │
└────────────────┬────────────────────────┘
                 │
┌────────────────┴────────────────────────┐
│             SM LEVEL                    │  ← Shared memory, occupancy, warps
│   (Streaming Multiprocessor resources)  │
└────────────────┬────────────────────────┘
                 │
┌────────────────┴────────────────────────┐
│          THREAD LEVEL                   │  ← Registers, coalescing, block sizes
│   (Thread block optimization)           │
└─────────────────────────────────────────┘
```

## Components

### 1. Kernel Configuration (`kernel_config.py`)

Represents complete kernel configuration with:
- **Thread/Block configuration**: num_warps, block_size, num_ctas
- **Grid dimensions**: x, y, z dimensions for kernel launch
- **Memory patterns**: Loads, stores, shared memory, coalescing
- **Performance metrics**: Execution time, throughput, occupancy
- **Tensor information**: Shapes, dtypes, sizes

```python
from agent_system import KernelConfig, BlockSize, GridDimensions

config = KernelConfig(
    kernel_name="my_kernel",
    num_warps=4,
    block_size=BlockSize(XBLOCK=256, YBLOCK=128),
    grid=GridDimensions(x=1024, y=8),
)
```

### 2. Message Protocol (`messages.py`)

**Message Types:**
- `ResourceRequestMessage`: Lower → Higher level (e.g., "Need more shared memory")
- `ConstraintMessage`: Higher → Lower level (e.g., "Max 48KB shared memory available")
- `OptimizationProposal`: Agent → Coordinator (proposed configuration changes)
- `ConflictResolutionMessage`: Coordinator → Agents (conflict resolution decisions)
- `AnalysisRequestMessage/Response`: Request/provide analysis

**Message Priority:**
- `CRITICAL (0)`: Immediate attention required
- `HIGH (1)`: Process soon
- `NORMAL (2)`: Standard priority
- `LOW (3)`: Can be deferred

```python
from agent_system import ResourceRequestMessage, MessagePriority

message = ResourceRequestMessage(
    sender_id="thread_agent_1",
    recipient_id="sm_agent_1",
    priority=MessagePriority.HIGH,
    requested_shared_memory_kb=64.0,
    reason="Improve data reuse",
    expected_benefit="30% speedup"
)
```

### 3. Message Bus (`message_bus.py`)

Central communication hub that:
- Routes messages between agents based on hierarchy
- Manages priority queues for each agent
- Maintains message history for debugging
- Validates agent-to-agent communication permissions
- Thread-safe operations

```python
from agent_system import MessageBus

bus = MessageBus()
bus.register_agent(my_agent)
bus.send_message(message)

# Get pending messages
messages = bus.get_messages(agent_id="thread_agent_1", max_count=10)

# Statistics
stats = bus.get_statistics()
print(f"Total agents: {stats['total_agents']}")
```

### 4. GPU Architecture (`architecture.py`)

Hardware specifications for optimization:
- **Predefined architectures**: Tesla V100, A100, RTX 3090
- **SM specs**: Warps, threads, registers per SM
- **Memory specs**: Shared memory, L1/L2 cache, bandwidth
- **Compute specs**: FP32/FP16/Tensor Core TFLOPS

```python
from agent_system import TESLA_V100, get_architecture

# Use predefined architecture
arch = TESLA_V100
print(f"Shared memory per SM: {arch.shared_memory_per_sm_kb} KB")

# Get by identifier
arch = get_architecture("cuda:80")  # Returns A100
```

### 5. Base Agent (`base_agent.py`)

Abstract base class for all agents with:
- **Required methods**: `analyze()`, `suggest_optimizations()`
- **Message handling**: `receive_message()`, `send_message()`
- **State management**: Activate/deactivate, status tracking
- **Knowledge base**: Architecture-specific information

```python
from agent_system import BaseAgent, KernelConfig

class MyAgent(BaseAgent):
    def analyze(self, kernel_config: KernelConfig):
        # Perform analysis specific to this agent's level
        return {
            "bottleneck": "memory_bandwidth",
            "confidence": 0.85,
        }

    def suggest_optimizations(self, analysis_results):
        # Generate optimization proposals
        proposals = []
        # ... create proposals ...
        return proposals
```

## Usage Example

```python
from agent_system import (
    MessageBus, BaseAgent, KernelConfig,
    BlockSize, GridDimensions, KernelType,
    OptimizationProposal
)

# 1. Create message bus
bus = MessageBus()

# 2. Create custom agent
class ThreadAgent(BaseAgent):
    def analyze(self, kernel_config):
        return {
            "utilization": kernel_config.total_threads_per_cta /
                          self.knowledge_base.max_threads_per_block,
            "bottleneck": "low_parallelism" if kernel_config.num_warps < 8 else None
        }

    def suggest_optimizations(self, analysis):
        if analysis["utilization"] < 0.5:
            return [OptimizationProposal(
                sender_id=self.agent_id,
                config_changes={"num_warps": 8},
                optimization_strategy="Increase thread parallelism",
                expected_speedup=1.4,
                confidence_score=0.8
            )]
        return []

# 3. Create agent instance
agent = ThreadAgent(
    agent_id="thread_opt_1",
    agent_level="thread",
    target_architecture="cuda:70",
    message_bus=bus
)

# 4. Analyze kernel
config = KernelConfig(
    kernel_name="my_kernel",
    kernel_type=KernelType.ELEMENTWISE_1D,
    num_warps=4,
    block_size=BlockSize(XBLOCK=256),
    grid=GridDimensions(x=1024)
)

analysis = agent.analyze(config)
proposals = agent.suggest_optimizations(analysis)

# 5. Process proposals
for proposal in proposals:
    print(f"Strategy: {proposal.optimization_strategy}")
    print(f"Changes: {proposal.config_changes}")
    print(f"Expected speedup: {proposal.expected_speedup}x")
```

## Running the Example

```bash
cd agent_system
python example.py
```

## Key Features

✅ **Hierarchical organization**: Thread → SM → GPU → Coordinator
✅ **Type-safe**: Full Python typing with dataclasses
✅ **Message-based communication**: Priority queues and routing
✅ **Architecture-aware**: GPU-specific optimization knowledge
✅ **Extensible**: Easy to add new agent types
✅ **Logging**: Comprehensive debug logging
✅ **Thread-safe**: Safe concurrent message passing

## Integration with Existing System

This agent system can be integrated with your existing TTGIR optimizer:

```python
# In ttgir_config_optimizer.py
from agent_system import MessageBus, KernelConfig, ThreadAgent, SMAgent

def optimize_with_agents(ttgir_analysis, metadata):
    # Create message bus
    bus = MessageBus()

    # Create agents
    thread_agent = ThreadAgent(...)
    sm_agent = SMAgent(...)

    # Convert TTGIR analysis to KernelConfig
    config = KernelConfig(
        kernel_name=metadata['name'],
        num_warps=ttgir_analysis.num_warps,
        # ... map other fields ...
    )

    # Analyze and optimize
    thread_analysis = thread_agent.analyze(config)
    sm_analysis = sm_agent.analyze(config)

    # Combine proposals
    all_proposals = (
        thread_agent.suggest_optimizations(thread_analysis) +
        sm_agent.suggest_optimizations(sm_analysis)
    )

    return all_proposals
```

## Future Extensions

- Implement specific agents: `RegisterPressureAgent`, `MemoryCoalescingAgent`, `OccupancyAgent`
- Add machine learning for performance prediction
- Implement conflict resolution strategies in coordinator
- Add support for multi-GPU optimization
- Integrate with empirical benchmarking

## License

Same as parent project (MIT).
