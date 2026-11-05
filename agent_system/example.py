"""
Example Usage of Multi-Agent GPU Optimization System

Demonstrates how to use the agent system to optimize Triton kernels.
"""

import logging
from agent_system import (
    KernelConfig,
    BlockSize,
    GridDimensions,
    MemoryAccessPattern,
    PerformanceMetrics,
    TensorInfo,
    KernelType,
    MessageBus,
    BaseAgent,
    OptimizationProposal,
    TESLA_V100,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# Example: Create a custom thread-level agent
class ThreadLevelAgent(BaseAgent):
    """Agent that optimizes thread-level parallelism"""

    def analyze(self, kernel_config: KernelConfig):
        """Analyze thread-level characteristics"""
        analysis = {
            "agent_level": "thread",
            "num_warps": kernel_config.num_warps,
            "threads_per_warp": kernel_config.threads_per_warp,
            "total_threads": kernel_config.total_threads_per_cta,
        }

        # Check if we have enough parallelism
        max_threads = self.knowledge_base.max_threads_per_block
        current_threads = kernel_config.total_threads_per_cta

        if current_threads < max_threads * 0.5:
            analysis["bottleneck"] = "low_thread_utilization"
            analysis["recommendation"] = "increase_num_warps"
            analysis["confidence"] = 0.8
        else:
            analysis["bottleneck"] = None
            analysis["confidence"] = 0.9

        return analysis

    def suggest_optimizations(self, analysis_results):
        """Suggest optimizations based on analysis"""
        proposals = []

        if analysis_results.get("bottleneck") == "low_thread_utilization":
            # Propose increasing num_warps
            proposal = OptimizationProposal(
                sender_id=self.agent_id,
                recipient_id="coordinator",
                config_changes={"num_warps": analysis_results["num_warps"] * 2},
                analysis_summary="Low thread utilization detected",
                optimization_strategy="Increase num_warps to improve parallelism",
                expected_speedup=1.5,
                confidence_score=0.8,
                risk_level="low",
            )
            proposals.append(proposal)

        return proposals


def main():
    """Example usage of the agent system"""

    print("="*80)
    print("Multi-Agent GPU Optimization System - Example")
    print("="*80)
    print()

    # 1. Create a message bus
    print("Step 1: Creating message bus...")
    bus = MessageBus()
    print(f"✓ Message bus created\n")

    # 2. Create agents at different levels
    print("Step 2: Creating agents...")
    thread_agent = ThreadLevelAgent(
        agent_id="thread_agent_1",
        agent_level="thread",
        target_architecture="cuda:70",
        message_bus=bus
    )
    print(f"✓ Created {thread_agent}")
    print()

    # 3. Create a sample kernel configuration
    print("Step 3: Creating kernel configuration...")
    kernel_config = KernelConfig(
        kernel_name="triton_poi_fused_add_0",
        kernel_type=KernelType.ELEMENTWISE_1D,
        num_warps=4,
        num_ctas=1,
        block_size=BlockSize(XBLOCK=256),
        grid=GridDimensions(x=1024),
        memory_access=MemoryAccessPattern(
            global_loads=2,
            global_stores=1,
            shared_memory_bytes=0,
            coalescing_efficiency="high",
        ),
        current_performance=PerformanceMetrics(
            execution_time_ms=0.523,
            occupancy_percentage=50.0,
        ),
        target_architecture="cuda:70",
    )

    print(kernel_config)
    print()

    # 4. Analyze the kernel
    print("Step 4: Analyzing kernel...")
    analysis = thread_agent.analyze(kernel_config)
    print("Analysis results:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    print()

    # 5. Generate optimization proposals
    print("Step 5: Generating optimization proposals...")
    proposals = thread_agent.suggest_optimizations(analysis)
    print(f"Generated {len(proposals)} proposal(s):")
    for i, proposal in enumerate(proposals, 1):
        print(f"\nProposal {i}:")
        print(f"  Strategy: {proposal.optimization_strategy}")
        print(f"  Config changes: {proposal.config_changes}")
        print(f"  Expected speedup: {proposal.expected_speedup}x")
        print(f"  Confidence: {proposal.confidence_score:.1%}")
        print(f"  Risk: {proposal.risk_level}")
    print()

    # 6. Show message bus statistics
    print("Step 6: Message bus statistics...")
    stats = bus.get_statistics()
    print(f"Total agents: {stats['total_agents']}")
    print(f"Agents by level: {stats['agents_by_level']}")
    print(f"Pending messages: {stats['total_pending_messages']}")
    print()

    # 7. Show agent status
    print("Step 7: Agent status...")
    thread_agent.activate()
    status = thread_agent.get_status()
    print("Thread agent status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    print()

    # 8. Demonstrate GPU architecture knowledge base
    print("Step 8: GPU Architecture Knowledge Base...")
    arch = TESLA_V100
    print(f"Target: {arch.name}")
    print(f"  Compute capability: {arch.compute_capability}")
    print(f"  SMs: {arch.num_sms}")
    print(f"  Max warps/SM: {arch.max_warps_per_sm}")
    print(f"  Shared memory/SM: {arch.shared_memory_per_sm_kb} KB")
    print(f"  L2 cache: {arch.l2_cache_size_mb} MB")
    print(f"  Memory bandwidth: {arch.memory_bandwidth_gb_s} GB/s")
    print(f"  FP32 performance: {arch.fp32_tflops} TFLOPS")
    print()

    print("="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
