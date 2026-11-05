"""
LangGraph-based orchestration for multi-agent GPU optimization system

Uses LangGraph StateGraph to coordinate parallel agent execution
and hierarchical decision making.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Callable, Dict, Any, List, Literal, Optional
from langgraph.graph import StateGraph, END
from .state import OptimizationState, KernelConfig
from .agents import WarpAgent, SMAgent, GPUAgent, CoordinatorAgent
from .agents import DataMovementAgent, ResourceBudgetAgent
from .agents import OrchestratorAgent as OrchestratorAgentV2
from .knowledge_base import compute_grid, detect_kernel_type, get_architecture
from .llm_provider import LLMProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Debug printing helpers
# ---------------------------------------------------------------------------

_SEPARATOR = "=" * 80
_THIN_SEP = "-" * 72


def _fmt_dict(d: dict, indent: int = 4) -> str:
    """Pretty-format a dict for debug printing."""
    prefix = " " * indent
    return json.dumps(d, indent=2, default=str).replace("\n", f"\n{prefix}")


def _print_node_header(name: str, iteration: int = 0) -> None:
    print(f"\n  {'>'*3} [{name}] (iteration {iteration})")


def _print_data_movement_summary(state: OptimizationState) -> None:
    """Print DataMovementAgent output summary."""
    dm = state.get("data_movement_proposal")
    if not dm:
        print("    Result: (no proposal)")
        return
    pref = dm.get("preferred", {})
    alts = dm.get("alternatives", [])
    hard = dm.get("hard_constraints", [])
    soft = dm.get("soft_constraints", [])
    needs = dm.get("what_I_need_from_others", "")
    print(f"    Preferred: block_size={pref.get('block_size', '?')}, "
          f"num_stages={pref.get('num_stages', '?')}, "
          f"num_warps={pref.get('num_warps', '?')}, "
          f"shared_memory={pref.get('shared_memory_bytes', '?')}")
    if alts:
        for i, alt in enumerate(alts):
            print(f"    Alt {i+1}: block_size={alt.get('block_size', '?')}, "
                  f"num_stages={alt.get('num_stages', '?')}, "
                  f"num_warps={alt.get('num_warps', '?')}")
    if hard:
        print(f"    Hard constraints: {hard}")
    if soft:
        print(f"    Soft constraints: {soft}")
    if needs:
        print(f"    Needs from others: {needs}")


def _print_resource_budget_summary(state: OptimizationState) -> None:
    """Print ResourceBudgetAgent output summary."""
    rb = state.get("resource_budget_proposal")
    if not rb:
        print("    Result: (no proposal)")
        return
    pref = rb.get("preferred", {})
    alts = rb.get("alternatives", [])
    hard = rb.get("hard_constraints", [])
    reg = rb.get("register_estimate", {})
    needs = rb.get("what_I_need_from_others", "")
    print(f"    Preferred: num_warps={pref.get('num_warps', '?')}, "
          f"shared_memory={pref.get('shared_memory_bytes', '?')}, "
          f"block_size={pref.get('block_size', '?')}, "
          f"num_stages={pref.get('num_stages', '?')}")
    if alts:
        for i, alt in enumerate(alts):
            print(f"    Alt {i+1}: num_warps={alt.get('num_warps', '?')}, "
                  f"block_size={alt.get('block_size', '?')}, "
                  f"num_stages={alt.get('num_stages', '?')}")
    if hard:
        print(f"    Hard constraints: {hard}")
    if reg:
        print(f"    Register estimate: {reg.get('total', '?')} regs/thread")
    if needs:
        print(f"    Needs from others: {needs}")


def _print_compute_grid_summary(state: OptimizationState) -> None:
    """Print compute_grid() output."""
    grid = state.get("computed_grid")
    metrics = state.get("execution_metrics")
    if not grid:
        print("    Result: (no grid computed)")
        return
    dims = f"({grid.get('grid_x', '?')}, {grid.get('grid_y', 1)}, {grid.get('grid_z', 1)})"
    total = grid.get("total_ctas", "?")
    print(f"    Grid: {dims}  |  Total CTAs: {total}")
    if metrics:
        sat = metrics.get("sm_saturation", "?")
        cps = metrics.get("ctas_per_sm", 0)
        waste = metrics.get("wave_quantization_waste", 0)
        print(f"    SM saturation: {sat} ({cps:.1f} CTAs/SM)  |  Wave waste: {waste:.1%}")


def _print_orchestrator_summary(state: OptimizationState) -> None:
    """Print OrchestratorAgent output."""
    conflicts = state.get("conflict_detection", {})
    candidates = state.get("candidate_configs", [])
    if conflicts:
        conflict_list = conflicts.get("conflicts", [])
        if conflict_list:
            print(f"    Conflicts detected: {len(conflict_list)}")
            for c in conflict_list[:5]:
                desc = c if isinstance(c, str) else c.get("description", str(c))
                print(f"      - {desc}")
        else:
            print("    Conflicts: none")
    print(f"    Candidates emitted: {len(candidates)}")
    for i, cand in enumerate(candidates[:5]):
        cfg = cand.get("config", {})
        conf = cand.get("confidence", 0)
        rationale = cand.get("rationale", "")[:80]
        bs = cfg.get("block_size", {})
        nw = cfg.get("num_warps", "?")
        ns = cfg.get("num_stages", "?")
        print(f"      [{i}] warps={nw} stages={ns} block_size={bs} conf={conf:.2f} — {rationale}")


def _print_benchmark_summary(state: OptimizationState) -> None:
    """Print benchmark results per candidate."""
    results = state.get("benchmark_results_per_candidate", [])
    bench = state.get("benchmark_results", {})
    iters = state.get("benchmark_iterations", 0)
    if not results and not bench:
        print("    Result: (benchmarking not configured)")
        return
    print(f"    Benchmark round: {iters}")
    for r in results:
        idx = r.get("candidate_index", "?")
        err = r.get("error")
        if err:
            tag = "CORRECTNESS ERROR" if "output validation" in err else "ERROR"
            print(f"      [{idx}] {tag}: {err}")
            continue
        speedup = r.get("speedup", 0)
        orig_med = r.get("original_stats", {}).get("median", "?")
        rec_med = r.get("stats", {}).get("median", "?")
        marker = " <-- BEST" if bench and abs(speedup - bench.get("speedup", -1)) < 1e-6 else ""
        validation = r.get("validation")
        val_tag = ""
        if validation is not None:
            n = validation.get("num_tensors_checked", 0)
            val_tag = f" [output OK, {n} tensor(s)]" if validation.get("valid") else " [output MISMATCH]"
        if isinstance(speedup, (int, float)):
            pct = (speedup - 1.0) * 100
            label = "improvement" if speedup >= 1.0 else "REGRESSION"
            print(f"      [{idx}] {speedup:.3f}x ({pct:+.1f}% {label}) — {orig_med}ms -> {rec_med}ms{val_tag}{marker}")
        else:
            print(f"      [{idx}] speedup={speedup}{val_tag}{marker}")
    if state.get("regression_detected"):
        print("    >> ALL candidates regressed — triggering retry")


def _print_final_config(state: OptimizationState) -> None:
    """Print the final recommended configuration."""
    result = state.get("optimization_result")
    if not result:
        print("    (no optimization result)")
        return
    rec = result.get("recommended_config", {})
    orig = result.get("original_config", {})
    recommendation = result.get("recommendation", "")
    bench = state.get("benchmark_results", {})

    if recommendation == "keep_original":
        print("    Decision: KEEP ORIGINAL (all optimizations regressed)")
        print(f"    Config: warps={orig.get('num_warps', '?')}, "
              f"stages={orig.get('num_stages', '?')}, "
              f"block_size={orig.get('block_size', {})}")
        return

    speedup = bench.get("speedup")
    print(f"    num_warps:  {rec.get('num_warps', '?')}")
    print(f"    num_stages: {rec.get('num_stages', '?')}")
    print(f"    block_size: {rec.get('block_size', {})}")
    if rec.get("grid"):
        print(f"    grid:       {rec.get('grid')}")
    if speedup is not None and isinstance(speedup, (int, float)):
        pct = (speedup - 1.0) * 100
        print(f"    Speedup:    {speedup:.3f}x ({pct:+.1f}%)")
    confidence = result.get("confidence_score", 0)
    if confidence:
        print(f"    Confidence: {confidence:.1%}")
    reasoning = result.get("coordinator_reasoning", "")
    if reasoning:
        print(f"    Reasoning:  {reasoning[:200]}")

# Maximum number of benchmark verification retries before falling back to
# the original configuration.
MAX_BENCHMARK_RETRIES = 3

# Valid num_warps values (must be powers of 2)
_VALID_NUM_WARPS = [1, 2, 4, 8, 16, 32]


def _nearest_power_of_2(n: int) -> int:
    """Round to nearest power of 2 (at least 1)."""
    if n <= 1:
        return 1
    import math
    lo = 1 << int(math.log2(n))
    hi = lo << 1
    return lo if (n - lo) <= (hi - n) else hi


def _sanitize_candidate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Fix common LLM mistakes in candidate configs.

    Handles:
    - num_warps must be a power of 2 (Triton constraint). Rounds to nearest
      valid value from ``_VALID_NUM_WARPS``.
    - block_size values must be powers of 2 (Triton's ``tl.arange`` requires
      this). Non-power-of-2 values are rounded to the nearest power of 2.
    """
    config = dict(config)

    nw = config.get("num_warps")
    if nw is not None and nw not in _VALID_NUM_WARPS:
        closest = min(_VALID_NUM_WARPS, key=lambda v: abs(v - nw))
        logger.warning(f"Sanitizing num_warps={nw} -> {closest} (must be power of 2)")
        config["num_warps"] = closest

    bs = config.get("block_size")
    if isinstance(bs, dict):
        sanitized_bs = {}
        for k, v in bs.items():
            if isinstance(v, int) and v > 0 and (v & (v - 1)) != 0:
                fixed = _nearest_power_of_2(v)
                logger.warning(f"Sanitizing block_size {k}={v} -> {fixed} (must be power of 2)")
                sanitized_bs[k] = fixed
            else:
                sanitized_bs[k] = v
        config["block_size"] = sanitized_bs

    return config


def construct_agent_feedback(
    benchmark_results_per_candidate: List[Dict[str, Any]],
    candidate_configs: List[Dict[str, Any]],
    data_movement_proposal: Dict[str, Any],
    resource_budget_proposal: Dict[str, Any],
    profiling_data: Optional[Dict[str, Any]] = None,
    benchmark_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, str]:
    """Construct per-agent scoped feedback from benchmark results.

    This is a **pure function** (no LLM call).  It maps benchmark results back
    to each agent's parameter choices and formats actionable feedback strings.

    Args:
        benchmark_results_per_candidate: Per-candidate results from the most
            recent ``_benchmark_candidates_node`` invocation.  Each dict has
            at least ``candidate_index``, ``config``, and optionally
            ``speedup`` / ``error`` / ``stats``.  Used as a fallback when
            ``benchmark_history`` is not provided.
        candidate_configs: The ``candidate_configs`` list from the
            OrchestratorAgent (each has ``config``, ``rationale``, …).
        data_movement_proposal: The DataMovementAgent's full proposal dict.
        resource_budget_proposal: The ResourceBudgetAgent's full proposal dict.
        profiling_data: Optional ``ncu`` profiling data dict.
        benchmark_history: Optional list of completed benchmark rounds, each
            ``{"round": int, "results": [...]}``.  When provided, prior rounds
            are surfaced to the agents alongside the current round so they
            can avoid re-proposing already-measured configurations.

    Returns:
        ``{"data_movement_feedback": str, "resource_budget_feedback": str}``
    """

    if benchmark_history:
        rounds = [r for r in benchmark_history if r.get("results")]
    elif benchmark_results_per_candidate:
        rounds = [{"round": 1, "results": benchmark_results_per_candidate}]
    else:
        rounds = []

    if not rounds:
        msg = "No benchmark data available from previous round."
        return {"data_movement_feedback": msg, "resource_budget_feedback": msg}

    dm_lines = [
        "BENCHMARK FEEDBACK (your expertise: data movement — but review ALL parameters):",
        "",
    ]
    rb_lines = [
        "BENCHMARK FEEDBACK (your expertise: SM resources — but review ALL parameters):",
        "",
    ]

    def _format_candidate(result: Dict[str, Any]) -> str:
        idx = result.get("candidate_index", "?")
        cfg = result.get("config", {})
        speedup = result.get("speedup")
        error = result.get("error")
        cand_bs = cfg.get("block_size", {})
        cand_nw = cfg.get("num_warps")
        cand_stages = cfg.get("num_stages")
        cand_smem = cfg.get("shared_memory_bytes", "?")
        cached = result.get("cached", False)

        if error:
            if "output validation failed" in error:
                outcome = f"CORRECTNESS ERROR: {error}"
            else:
                outcome = f"ERROR: {error}"
        elif speedup is not None:
            tag = " [cached]" if cached else ""
            if speedup >= 1.0:
                outcome = f"{speedup:.3f}x speedup (IMPROVEMENT){tag}"
            else:
                outcome = (
                    f"{speedup:.3f}x speedup "
                    f"(REGRESSION, {(1.0 - speedup) * 100:.1f}% slower){tag}"
                )
        else:
            outcome = "no speedup data"

        return (
            f"- Candidate {idx}: block_size={cand_bs}, num_warps={cand_nw}, "
            f"num_stages={cand_stages}, shared_memory={cand_smem} "
            f"-> {outcome}"
        )

    multi_round = len(rounds) > 1
    for r_idx, round_entry in enumerate(rounds):
        is_latest = r_idx == len(rounds) - 1
        round_num = round_entry.get("round", r_idx + 1)
        results = round_entry.get("results", [])

        if multi_round:
            if is_latest:
                header = f"CURRENT ROUND {round_num} (most recent):"
            else:
                header = (
                    f"PRIOR ROUND {round_num} (already measured — do NOT repeat these configs):"
                )
            dm_lines.append(header)
            rb_lines.append(header)

        for result in results:
            full_line = _format_candidate(result)
            dm_lines.append(full_line)
            rb_lines.append(full_line)

        if multi_round:
            dm_lines.append("")
            rb_lines.append("")

    # ---- Optional profiling data — ALL metrics go to BOTH agents ----
    if profiling_data:
        all_metrics: List[str] = []
        if "global_memory_throughput_pct" in profiling_data:
            all_metrics.append(
                f"- Global memory throughput: "
                f"{profiling_data['global_memory_throughput_pct']:.1f}% of peak"
            )
        if "l2_hit_rate" in profiling_data:
            all_metrics.append(f"- L2 cache hit rate: {profiling_data['l2_hit_rate']:.1f}%")
        if "l1_hit_rate" in profiling_data:
            all_metrics.append(f"- L1 cache hit rate: {profiling_data['l1_hit_rate']:.1f}%")
        if "coalescing_efficiency" in profiling_data:
            all_metrics.append(
                f"- Coalescing efficiency: {profiling_data['coalescing_efficiency']:.1f}%"
            )
        if "achieved_occupancy_pct" in profiling_data:
            all_metrics.append(
                f"- Achieved occupancy: {profiling_data['achieved_occupancy_pct']:.1f}%"
            )
        if "registers_per_thread" in profiling_data:
            all_metrics.append(
                f"- Registers per thread: {profiling_data['registers_per_thread']}"
            )
        if "warp_stall_reasons" in profiling_data:
            all_metrics.append(
                f"- Warp stall reasons: {profiling_data['warp_stall_reasons']}"
            )
        if "register_spills" in profiling_data:
            all_metrics.append(
                f"- Register spills: {profiling_data['register_spills']} per thread"
            )
        if all_metrics:
            profiling_section = ["", "PROFILING DATA (all metrics):"] + all_metrics
            dm_lines.extend(profiling_section)
            rb_lines.extend(profiling_section)

    # ---- Revision instructions ----
    no_repeat = (
        " All configurations listed above have already been measured — "
        "propose a NEW configuration distinct from every one of them."
        if multi_round else ""
    )

    dm_lines.append("")
    dm_lines.append(
        "INSTRUCTION: Revise your FULL config proposal based on this empirical data. "
        "Your expertise is data movement — focus on block_size and num_stages, "
        "but also consider how num_warps interacted with your choices." + no_repeat
    )

    rb_lines.append("")
    rb_lines.append(
        "INSTRUCTION: Revise your FULL config proposal based on this empirical data. "
        "Your expertise is SM resources — focus on num_warps and shared_memory, "
        "but also consider how block_size interacted with your choices." + no_repeat
    )

    return {
        "data_movement_feedback": "\n".join(dm_lines),
        "resource_budget_feedback": "\n".join(rb_lines),
    }


class GPUOptimizationOrchestrator:
    """
    LangGraph-based orchestrator for multi-agent GPU kernel optimization

    Workflow:
    1. Parse input kernel configuration
    2. Execute parallel analysis by Warp, SM, and GPU agents
    3. Coordinator synthesizes results
    4. Benchmark recommended config against original (closed-loop)
    5. Return optimized configuration (or original on regression)
    """

    def __init__(
        self,
        llm_provider: str = "deepseek",  # "openai", "deepseek", "ollama", "vllm"
        llm_model: Optional[str] = None,
        architecture_code: str = "cuda:70",
        temperature: float = 0.2,
        max_iterations: int = 3,
        num_candidates: int = 3,
        debug_mode: bool = False,
        debug_dir: Optional[str] = None,
        use_cached_prompts: bool = False,
        cached_prompts_dir: Optional[str] = None,
        benchmark_fn: Optional[Callable] = None,
        profiler_fn: Optional[Callable] = None,
        skip_resource_budget: bool = False,
        skip_data_movement: bool = False,
        include_agent_proposals: bool = False,
        disable_negotiation: bool = False,
    ):
        """
        Initialize orchestrator

        Args:
            llm_provider: LLM provider ("openai", "deepseek", "ollama", "vllm")
            llm_model: Model name (default depends on provider)
            architecture_code: Target GPU architecture
            temperature: LLM sampling temperature
            max_iterations: Max refinement iterations
            debug_mode: Enable detailed logging of prompts and responses
            debug_dir: Directory to save debug artifacts (default: ./debug_multiagent/)
            use_cached_prompts: If True, read responses from a previous debug run instead of calling the LLM
            cached_prompts_dir: Path to debug run directory with cached response files
            benchmark_fn: Optional callable for closed-loop benchmark verification.
                Signature: (original_config, recommended_config, shapes_file,
                kernel_module, kernel_name, ...) -> Dict with speedup stats.
                When None, the benchmark node is a pass-through.
            profiler_fn: Optional callable for NCU profiling after benchmarking.
                Signature: (config, shapes_file, kernel_module, kernel_name,
                kernel_class, device) -> Dict with profiling metrics.
                When None, profiling is skipped (default).
            skip_resource_budget: If True, remove ResourceBudgetAgent from the
                graph (ablation A1a: single-agent with DataMovement only).
            skip_data_movement: If True, remove DataMovementAgent from the
                graph (ablation A1b: single-agent with ResourceBudget only).
            include_agent_proposals: If True, inject each agent's preferred
                config as additional benchmark candidates alongside the
                orchestrator's synthesised candidates.  The total candidate
                count (``num_candidates``) is preserved: N-2 orchestrator
                candidates + 1 DM preferred + 1 RB preferred.
            disable_negotiation: If True, skip the negotiation sub-loop
                even when high-severity conflicts are detected.  Agents
                still propose independently; the orchestrator resolves
                conflicts in a single pass.
        """
        self.architecture_code = architecture_code
        self.max_iterations = max_iterations
        self.num_candidates = num_candidates
        self.debug_mode = debug_mode
        self.benchmark_fn = benchmark_fn
        self.profiler_fn = profiler_fn
        self.skip_resource_budget = skip_resource_budget
        self.skip_data_movement = skip_data_movement
        self.include_agent_proposals = include_agent_proposals
        # Negotiation is disabled by default when include_agent_proposals is on,
        # because the phased benchmark resolves conflicts empirically.
        self.disable_negotiation = disable_negotiation or include_agent_proposals
        base_debug_dir = debug_dir or "./debug_multiagent"

        # Generate run-specific folder name using current time
        if self.debug_mode:
            import os
            from datetime import datetime
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.debug_dir = os.path.join(base_debug_dir, run_id)
            os.makedirs(self.debug_dir, exist_ok=True)
            logger.info(f"Debug mode enabled. Artifacts will be saved to: {self.debug_dir}")
        else:
            self.debug_dir = base_debug_dir
            run_id = None

        # Initialize LLM provider
        self.llm = LLMProvider(
            provider=llm_provider,
            model=llm_model,
            temperature=temperature,
            debug_mode=self.debug_mode,
            debug_dir=self.debug_dir,
            run_id=run_id,
            use_cached_prompts=use_cached_prompts,
            cached_prompts_dir=cached_prompts_dir,
        )

        # Initialize agents (old — kept for backward compatibility, tests patch these)
        self.warp_agent = WarpAgent(self.llm, architecture_code)
        self.sm_agent = SMAgent(self.llm, architecture_code)
        self.gpu_agent = GPUAgent(self.llm, architecture_code)
        self.coordinator = CoordinatorAgent(self.llm, architecture_code)

        # Approach 7 agents (new negotiation-based architecture)
        self.data_movement_agent = DataMovementAgent(self.llm, architecture_code)
        self.resource_budget_agent = ResourceBudgetAgent(self.llm, architecture_code)
        self.orchestrator_agent = OrchestratorAgentV2(self.llm, architecture_code)
        self._architecture = get_architecture(architecture_code)

        # Build LangGraph workflow
        self.graph = self._build_graph()

        logger.info(f"Initialized GPUOptimizationOrchestrator with {llm_provider}")

    def _build_graph(self) -> StateGraph:
        """Build LangGraph StateGraph for the Approach 7 optimization workflow.

        Topology (full system):
            data_movement -> (error check) -> resource_budget -> (error check)
                -> compute_grid -> detect_conflicts
                -> conditional:
                    "negotiate": data_movement   (negotiation sub-loop)
                    "proceed":   orchestrator    (consensus or max rounds)
            orchestrator -> benchmark -> construct_feedback -> check_convergence
                -> conditional:
                    "continue": data_movement    (benchmark regression)
                    "end":      END

        Single-agent ablation modes:
            skip_resource_budget: DM -> grid -> conflicts -> orchestrator -> ...
            skip_data_movement:   RB -> grid -> conflicts -> orchestrator -> ...

        Error handling: if any agent sets state["error"], the workflow
        short-circuits to END for this shape.  The outer per-shape loop
        in multiagent_triton_runner.py continues to the next shape.
        """
        workflow = StateGraph(OptimizationState)

        # Shared nodes (always present)
        workflow.add_node("compute_grid", self._compute_grid_node)
        workflow.add_node("detect_conflicts", self._detect_conflicts_node)
        workflow.add_node("orchestrator", self._orchestrator_node_v2)
        workflow.add_node("benchmark", self._benchmark_candidates_node)
        workflow.add_node("construct_feedback", self._construct_feedback_node)
        workflow.add_node("check_convergence", self._check_convergence_node)

        # --- Single-agent ablation: DM only (skip RB) ---
        if self.skip_resource_budget:
            workflow.add_node("data_movement", self._data_movement_node)
            workflow.set_entry_point("data_movement")

            workflow.add_conditional_edges(
                "data_movement",
                self._check_agent_error,
                {"error": END, "ok": "compute_grid"},
            )
            workflow.add_edge("compute_grid", "detect_conflicts")
            workflow.add_conditional_edges(
                "detect_conflicts",
                self._should_negotiate,
                {"negotiate": "data_movement", "proceed": "orchestrator"},
            )

            # Feedback loop back to DM
            loop_back = "data_movement"

        # --- Single-agent ablation: RB only (skip DM) ---
        elif self.skip_data_movement:
            workflow.add_node("resource_budget", self._resource_budget_node)
            workflow.set_entry_point("resource_budget")

            workflow.add_conditional_edges(
                "resource_budget",
                self._check_agent_error,
                {"error": END, "ok": "compute_grid"},
            )
            workflow.add_edge("compute_grid", "detect_conflicts")
            workflow.add_conditional_edges(
                "detect_conflicts",
                self._should_negotiate,
                {"negotiate": "resource_budget", "proceed": "orchestrator"},
            )

            # Feedback loop back to RB
            loop_back = "resource_budget"

        # --- Full system (both agents) ---
        else:
            workflow.add_node("data_movement", self._data_movement_node)
            workflow.add_node("resource_budget", self._resource_budget_node)
            workflow.set_entry_point("data_movement")

            workflow.add_conditional_edges(
                "data_movement",
                self._check_agent_error,
                {"error": END, "ok": "resource_budget"},
            )
            workflow.add_conditional_edges(
                "resource_budget",
                self._check_agent_error,
                {"error": END, "ok": "compute_grid"},
            )
            workflow.add_edge("compute_grid", "detect_conflicts")
            workflow.add_conditional_edges(
                "detect_conflicts",
                self._should_negotiate,
                {"negotiate": "data_movement", "proceed": "orchestrator"},
            )

            # Feedback loop back to DM (which chains to RB)
            loop_back = "data_movement"

        # Post-negotiation: orchestrator -> (check) -> benchmark -> feedback -> convergence
        workflow.add_conditional_edges(
            "orchestrator",
            self._check_agent_error,
            {"error": END, "ok": "benchmark"},
        )
        workflow.add_edge("benchmark", "construct_feedback")
        workflow.add_edge("construct_feedback", "check_convergence")

        # Conditional loop-back with per-agent feedback
        workflow.add_conditional_edges(
            "check_convergence",
            self._should_continue,
            {
                "continue": loop_back,
                "end": END,
            }
        )

        return workflow.compile()

    def _warp_analysis_node(self, state: OptimizationState) -> OptimizationState:
        """Node for warp-level analysis"""
        return self.warp_agent.analyze(state, num_candidates=self.num_candidates)

    def _sm_analysis_node(self, state: OptimizationState) -> OptimizationState:
        """Node for SM-level analysis"""
        return self.sm_agent.analyze(state, num_candidates=self.num_candidates)

    def _gpu_analysis_node(self, state: OptimizationState) -> OptimizationState:
        """Node for GPU-level analysis"""
        return self.gpu_agent.analyze(state, num_candidates=self.num_candidates)

    def _coordinator_node(self, state: OptimizationState) -> OptimizationState:
        """Node for coordination and decision making"""
        return self.coordinator.coordinate(state, num_candidates=self.num_candidates)

    # --- Approach 7 node methods ---

    def _data_movement_node(self, state: OptimizationState) -> OptimizationState:
        """Node for data-movement analysis (Approach 7)."""
        is_negotiation = state.get("negotiation_in_progress", False)
        iteration = state.get("iteration", 0)

        if is_negotiation:
            neg_round = state.get("negotiation_round", 0)
            _print_node_header(
                f"DataMovementAgent (negotiation round {neg_round})", iteration,
            )
        else:
            _print_node_header("DataMovementAgent", iteration)
            # Fresh iteration — reset negotiation state
            state["negotiation_round"] = 0
            state["negotiation_history"] = None
            state["negotiation_in_progress"] = False
            state["consensus_reached"] = False

        state = self.data_movement_agent.analyze(state, num_candidates=self.num_candidates)
        _print_data_movement_summary(state)
        if state.get("error"):
            print(f"    ERROR: {state['error']}")
        return state

    def _resource_budget_node(self, state: OptimizationState) -> OptimizationState:
        """Node for resource-budget analysis (Approach 7).

        When not in a negotiation round, the DM proposal is temporarily masked
        so that RB analyses the TTGIR independently (prevents anchoring bias).
        The proposal is restored afterwards so the orchestrator sees both.
        """
        _print_node_header("ResourceBudgetAgent", state.get("iteration", 0))

        # Always mask DM's proposal during RB analysis to prevent anchoring.
        # RB must propose independently — even during negotiation rounds,
        # exposing DM's values causes RB to capitulate (verified on geglu).
        # The orchestrator already sees both proposals for conflict resolution.
        dm_backup = state.get("data_movement_proposal")
        state["data_movement_proposal"] = None

        state = self.resource_budget_agent.analyze(state, num_candidates=self.num_candidates)

        # Restore DM's proposal so the orchestrator can see both
        state["data_movement_proposal"] = dm_backup

        _print_resource_budget_summary(state)
        if state.get("error"):
            print(f"    ERROR: {state['error']}")
        return state

    def _compute_grid_node(self, state: OptimizationState) -> OptimizationState:
        """Pure-function node: compute grid dimensions from block_size proposals.

        No LLM call.  Uses ``compute_grid()`` to derive grid dimensions and
        execution-quality metrics (wave quantization, SM saturation) which
        the OrchestratorAgent consumes for conflict detection.
        """
        _print_node_header("ComputeGrid", state.get("iteration", 0))
        config = state["kernel_config"]
        dm_proposal = state.get("data_movement_proposal") or {}
        preferred_bs = dm_proposal.get("preferred", {}).get("block_size")

        if not preferred_bs:
            # Fallback to current config's block_size
            preferred_bs = dict(config.block_size) if config.block_size else {"XBLOCK": 128}

        kernel_type = detect_kernel_type(config.ttgir, config.kernel_name)
        grid_info = compute_grid(
            preferred_bs,
            config.tensor_shapes,
            kernel_type,
            self._architecture.num_sms,
        )

        state["computed_grid"] = grid_info
        state["execution_metrics"] = {
            "total_ctas": grid_info.get("total_ctas", 0),
            "ctas_per_sm": grid_info.get("ctas_per_sm", 0),
            "sm_saturation": grid_info.get("sm_saturation", "unknown"),
            "wave_quantization_waste": grid_info.get("wave_quantization_waste", 0),
        }
        _print_compute_grid_summary(state)
        return state

    def _detect_conflicts_node(self, state: OptimizationState) -> OptimizationState:
        """Lightweight programmatic node: detect conflicts between agent proposals.

        No LLM call.  Compares DM and RB proposals parameter-by-parameter,
        decides whether to route back for negotiation or proceed to orchestrator.
        """
        _print_node_header("ConflictDetection", state.get("iteration", 0))

        dm_proposal = state.get("data_movement_proposal") or {}
        rb_proposal = state.get("resource_budget_proposal") or {}
        grid_info = state.get("computed_grid") or {}

        conflicts = self.orchestrator_agent._detect_conflicts(
            dm_proposal, rb_proposal, grid_info,
        )

        state["conflict_detection"] = {"conflicts": conflicts}

        high_severity = [c for c in conflicts if c.get("severity") == "high"]
        neg_round = state.get("negotiation_round", 0)
        max_rounds = state.get("max_negotiation_rounds", 2)

        if high_severity and neg_round < max_rounds:
            # Enter negotiation sub-loop
            state["negotiation_in_progress"] = True
            state["negotiation_round"] = neg_round + 1
            history = state.get("negotiation_history") or []
            history.append({
                "round": neg_round + 1,
                "dm_preferred": dm_proposal.get("preferred", {}),
                "rb_preferred": rb_proposal.get("preferred", {}),
                "conflicts": conflicts,
            })
            state["negotiation_history"] = history
            state["consensus_reached"] = False
            print(f"    {len(high_severity)} high-severity conflicts "
                  f"— negotiation round {neg_round + 1}/{max_rounds}")
        else:
            # Proceed to orchestrator
            state["negotiation_in_progress"] = False
            state["consensus_reached"] = len(high_severity) == 0
            if conflicts:
                print(f"    {len(conflicts)} conflicts (none high-severity "
                      f"or max rounds reached) — proceeding")
            else:
                print("    No conflicts — consensus reached")

        return state

    def _check_agent_error(
        self, state: OptimizationState,
    ) -> Literal["error", "ok"]:
        """Short-circuit to END if any agent set state['error']."""
        if state.get("error"):
            return "error"
        return "ok"

    def _should_negotiate(
        self, state: OptimizationState,
    ) -> Literal["negotiate", "proceed"]:
        """Route to negotiation loop or proceed to orchestrator."""
        if self.disable_negotiation:
            state["negotiation_in_progress"] = False
            return "proceed"
        if state.get("negotiation_in_progress"):
            return "negotiate"
        return "proceed"

    def _orchestrator_node_v2(self, state: OptimizationState) -> OptimizationState:
        """Node for orchestrator coordination (Approach 7).

        Delegates to ``OrchestratorAgent.coordinate()`` which reads both
        agent proposals + computed grid/execution metrics, detects conflicts,
        and emits ``candidate_configs``.

        When ``include_agent_proposals`` is enabled, each agent's preferred
        config is injected as an additional candidate so the benchmark can
        select the best among orchestrator-synthesised *and* raw agent
        proposals.
        """
        _print_node_header("OrchestratorAgent", state.get("iteration", 0))

        if self.include_agent_proposals:
            # Orchestrator produces 1 config; agent preferred+alternatives fill the rest
            orch_n = 1
        else:
            orch_n = self.num_candidates

        state = self.orchestrator_agent.coordinate(state, num_candidates=orch_n)

        if self.include_agent_proposals:
            candidates = state.get("candidate_configs", [])
            dm_prop = state.get("data_movement_proposal") or {}
            rb_prop = state.get("resource_budget_proposal") or {}

            orch_count = len(candidates)
            print(f"    [agent-proposals] Orchestrator synthesized {orch_count} config(s), injecting agent preferred configs...")

            # Force 3 Phase 1 candidates: 1 Orch + 1 DM preferred + 1 RB preferred
            def _cfg_key(c):
                cfg = c.get("config", {})
                bs = cfg.get("block_size", {})
                return (cfg.get("num_warps"), cfg.get("num_stages"),
                        tuple(sorted(bs.items())))

            existing_keys = {_cfg_key(c) for c in candidates}

            # Tag orchestrator candidates
            for c in candidates:
                if "variant_type" not in c:
                    c["variant_type"] = "orchestrator"

            # ---- Diversity safety net (Plan change D) ----
            # If DM's and RB's preferred configs both fall in the scale-up
            # octant (XBLOCK >= 128, num_warps >= 4, num_stages == 1), nudge
            # the orchestrator's blended candidate to contrast in all three
            # axes.  DM's and RB's preferred configs stay untouched and are
            # still injected below; this only modifies the orchestrator's own
            # synthesis so Phase 1 always probes the opposite region when the
            # two lenses accidentally converge.
            def _primary_block_value(cfg):
                bs = cfg.get("block_size") or {}
                if not bs:
                    return None, None
                for pref in ("XBLOCK", "BLOCK_M", "BLOCK_SIZE"):
                    if pref in bs:
                        return pref, bs[pref]
                key = sorted(bs.keys())[0]
                return key, bs[key]

            dm_pref = dm_prop.get("preferred", {}) if isinstance(dm_prop, dict) else {}
            rb_pref = rb_prop.get("preferred", {}) if isinstance(rb_prop, dict) else {}
            _, dm_xb = _primary_block_value(dm_pref)
            _, rb_xb = _primary_block_value(rb_pref)
            dm_nw = dm_pref.get("num_warps")
            rb_nw = rb_pref.get("num_warps")
            dm_ns = dm_pref.get("num_stages")
            rb_ns = rb_pref.get("num_stages")

            # Loosened trigger: fire whenever both agents land on XBLOCK >= 128.
            # The strict "scale-up octant" form (XBLOCK >= 128 AND nw >= 4 AND
            # ns == 1) missed the realistic failure mode where DM proposes
            # XBLOCK=256 with nw=2/ns=2 and RB proposes XBLOCK=1024/nw=4/ns=1 —
            # both still anchor on big tiles but the test reads as "divergent".
            converged = (
                dm_xb is not None and rb_xb is not None
                and dm_xb >= 128 and rb_xb >= 128
            )

            if converged and candidates:
                orch_cand = candidates[0]
                orch_cfg = dict(orch_cand.get("config", {}))
                orch_bs = dict(orch_cfg.get("block_size", {}))
                primary_key, _ = _primary_block_value(orch_cfg)
                if primary_key is not None and primary_key in orch_bs:
                    orch_bs[primary_key] = max(2, orch_bs[primary_key] // 8)
                orch_cfg["block_size"] = orch_bs
                orch_cfg["num_warps"] = 2
                orch_cfg["num_stages"] = 2
                orch_cand["config"] = orch_cfg
                orch_cand["variant_type"] = "orchestrator_contrast"
                orch_cand["rationale"] = (
                    (orch_cand.get("rationale", "") or "")
                    + " [diversity-check: both agents converged on scale-up "
                      f"(DM XBLOCK={dm_xb},nw={dm_nw},ns={dm_ns} | "
                      f"RB XBLOCK={rb_xb},nw={rb_nw},ns={rb_ns}); "
                      "orchestrator candidate nudged to contrasting "
                      "small-XBLOCK/low-warp/deep-pipeline region]"
                )
                existing_keys = {_cfg_key(c) for c in candidates}
                contrasted_xb = orch_bs.get(primary_key) if primary_key else "?"
                print(f"    [agent-proposals] Diversity check FIRED: both agents in "
                      f"scale-up octant; orchestrator candidate contrasted to "
                      f"{primary_key}={contrasted_xb}, nw=2, ns=2.")
            elif dm_xb is not None and rb_xb is not None:
                print(f"    [agent-proposals] Diversity check: DM/RB divergent "
                      f"(DM XBLOCK={dm_xb},nw={dm_nw},ns={dm_ns} | "
                      f"RB XBLOCK={rb_xb},nw={rb_nw},ns={rb_ns}) — no contrast needed.")

            for label, prop, prefix in [
                ("DM", dm_prop, "agent_dm"),
                ("RB", rb_prop, "agent_rb"),
            ]:
                pref = prop.get("preferred", {})
                if not pref:
                    print(f"    [agent-proposals] ~ {label} had no preferred config to inject")
                    continue

                # Try preferred first, then walk through alternatives
                # (ordered by confidence) until we find one not already in the pool.
                candidates_to_try = [("preferred", pref)]
                alts_sorted = sorted(
                    prop.get("alternatives", []),
                    key=lambda a: -(a.get("confidence", 0) if isinstance(a, dict) else 0),
                )
                for j, alt in enumerate(alts_sorted):
                    alt_cfg = alt.get("config", alt) if isinstance(alt, dict) else alt
                    candidates_to_try.append((f"alt[{j}]", alt_cfg))

                injected = False
                for cand_name, cand_cfg in candidates_to_try:
                    if not isinstance(cand_cfg, dict):
                        continue
                    entry = {
                        "config": cand_cfg,
                        "rationale": f"{label} agent {cand_name} (injected)",
                        "confidence": 0.85 if cand_name == "preferred" else 0.80,
                        "variant_type": f"{prefix}_preferred",
                    }
                    key = _cfg_key(entry)
                    bs = cand_cfg.get("block_size", {})
                    if key not in existing_keys:
                        candidates.append(entry)
                        existing_keys.add(key)
                        print(f"    [agent-proposals] + {label} {cand_name}: warps={cand_cfg.get('num_warps','?')} stages={cand_cfg.get('num_stages','?')} block_size={bs}")
                        injected = True
                        break
                    else:
                        print(f"    [agent-proposals] ~ {label} {cand_name} DEDUP: warps={cand_cfg.get('num_warps','?')} stages={cand_cfg.get('num_stages','?')} block_size={bs}")

                if not injected:
                    print(f"    [agent-proposals] ~ {label}: all candidates dedup'd, skipping")

            # Sort candidates by confidence (highest first)
            candidates.sort(key=lambda c: c.get("confidence", 0), reverse=True)

            print(f"    [agent-proposals] Final candidate count: {len(candidates)} (was {orch_count} from orchestrator)")
            state["candidate_configs"] = candidates

        _print_orchestrator_summary(state)
        if state.get("error"):
            print(f"    ERROR: {state['error']}")
        return state

    def _construct_feedback_node(self, state: OptimizationState) -> OptimizationState:
        """Construct per-agent scoped feedback from benchmark results.

        Only fires when ``regression_detected`` is True (all candidates
        regressed).  Calls ``construct_agent_feedback()`` and stores the
        result in ``state["agent_feedback"]`` so the next agent round
        receives targeted feedback.
        """
        if not state.get("regression_detected"):
            return state
        print(f"\n  {'>'*3} [ConstructFeedback] Regression detected — building per-agent feedback")

        feedback = construct_agent_feedback(
            benchmark_results_per_candidate=state.get("benchmark_results_per_candidate", []),
            candidate_configs=state.get("candidate_configs", []),
            data_movement_proposal=state.get("data_movement_proposal") or {},
            resource_budget_proposal=state.get("resource_budget_proposal") or {},
            profiling_data=state.get("profiling_data"),
            benchmark_history=state.get("benchmark_history") or [],
        )
        state["agent_feedback"] = feedback
        return state

    @staticmethod
    def _cfg_key(cfg: dict) -> str:
        """Hashable, JSON-safe key for a config dict, used to dedup benchmarks."""
        bs = cfg.get("block_size", {})
        parts = [
            f"w={cfg.get('num_warps')}",
            f"s={cfg.get('num_stages')}",
            f"sm={cfg.get('shared_memory_bytes')}",
        ]
        for k in sorted(bs):
            parts.append(f"{k}={bs[k]}")
        return "|".join(parts)

    @staticmethod
    def _generate_neighbors(cfg: dict) -> list[dict]:
        """Generate neighbor configs by tweaking one parameter at a time."""
        import copy
        neighbors = []
        # num_warps: half and double (clamped to [1, 32], power of 2)
        nw = cfg.get("num_warps", 4)
        for new_nw in [max(1, nw // 2), min(32, nw * 2)]:
            if new_nw != nw:
                n = copy.deepcopy(cfg)
                n["num_warps"] = new_nw
                neighbors.append(n)
        # num_stages: ±1 (clamped to [1, 5])
        ns = cfg.get("num_stages", 1)
        for new_ns in [max(1, ns - 1), min(5, ns + 1)]:
            if new_ns != ns:
                n = copy.deepcopy(cfg)
                n["num_stages"] = new_ns
                neighbors.append(n)
        # block_size keys: half and double each (clamped to [16, 4096], power of 2)
        for bk, bv in cfg.get("block_size", {}).items():
            for new_bv in [max(16, bv // 2), min(4096, bv * 2)]:
                if new_bv != bv:
                    n = copy.deepcopy(cfg)
                    n["block_size"][bk] = new_bv
                    neighbors.append(n)
        return neighbors

    def _run_benchmark(self, cfg: dict, bc: dict, original_config: dict):
        """Benchmark a single config.

        Returns the result dict from ``benchmark_fn`` on success, or
        ``None`` on error.  When an error occurs the detail is stored in
        ``self._last_benchmark_error`` so callers can include it in
        per-candidate results.
        """
        self._last_benchmark_error = None
        cfg = _sanitize_candidate_config(cfg)
        try:
            result = self.benchmark_fn(
                original_config=original_config,
                recommended_config=cfg,
                shapes_file=Path(bc["shapes_file"]),
                kernel_module=Path(bc["kernel_module"]),
                kernel_name=bc["kernel_name"],
                kernel_class=bc.get("kernel_class"),
                device=bc.get("device", "cuda"),
                warmup=bc.get("warmup", 5),
                rep=bc.get("rep", 20),
            )
        except Exception as e:
            logger.error(f"Benchmark exception: {e}")
            self._last_benchmark_error = str(e)
            return None
        finally:
            # Free GPU memory after each benchmark to prevent OOM on long runs
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        if result.get("error"):
            self._last_benchmark_error = result["error"]
            return None
        return result

    def _benchmark_candidates_node(self, state: OptimizationState) -> OptimizationState:
        """Benchmark candidate configs against original in up to 3 phases.

        Phase 1: Benchmark candidate_configs (orchestrator + agent preferred).
        Phase 2: If an agent preferred config improved, benchmark that agent's
                 alternatives to find a better variant.
        Phase 3: Neighborhood search — tweak each parameter of the best config
                 by ±1 step to find nearby optima.

        A global dedup set (``benchmarked_configs`` in state) tracks all configs
        benchmarked across phases and iterations to avoid redundant work.
        """
        _print_node_header("Benchmark", state.get("iteration", 0))
        # Guard: pass through when benchmarking is not configured
        if self.benchmark_fn is None or not state.get("benchmark_config"):
            print("    (benchmarking not configured — pass-through)")
            return state

        optimization_result = state.get("optimization_result")
        if not optimization_result:
            logger.warning("Benchmark candidates node: no optimization result to benchmark")
            return state

        bc = state["benchmark_config"]
        original_config = optimization_result.get("original_config", {})

        # Global dedup set: persists across iterations to avoid re-benchmarking
        benchmarked = state.get("benchmarked_configs", {})  # key -> speedup

        # Get candidates — fall back to single recommended_config for backward compat
        candidates = state.get("candidate_configs", [])
        if not candidates:
            rec = optimization_result.get("recommended_config", {})
            if rec:
                candidates = [{"config": rec, "rationale": "single recommendation", "confidence": 0.5}]
            else:
                logger.warning("Benchmark candidates node: no candidates to benchmark")
                return state

        # ---- Phase 1: benchmark candidate configs ----
        results_per_candidate = []
        total_benchmarks = 0

        for i, candidate in enumerate(candidates):
            candidate_config = candidate.get("config", {})
            if not candidate_config:
                logger.warning(f"Candidate {i}: empty config, skipping")
                results_per_candidate.append({
                    "candidate_index": i,
                    "config": candidate_config,
                    "error": "empty config",
                })
                continue

            candidate_config = _sanitize_candidate_config(candidate_config)
            key = self._cfg_key(candidate_config)

            # Skip if already benchmarked in a prior iteration
            if key in benchmarked:
                speedup = benchmarked[key]
                logger.info(f"Candidate {i}: cached speedup={speedup:.3f}x (skipped)")
                results_per_candidate.append({
                    "candidate_index": i,
                    "config": candidate_config,
                    "rationale": candidate.get("rationale", ""),
                    "speedup": speedup,
                    "cached": True,
                })
                continue

            result = self._run_benchmark(candidate_config, bc, original_config)
            total_benchmarks += 1

            if result is None:
                results_per_candidate.append({
                    "candidate_index": i,
                    "config": candidate_config,
                    "error": self._last_benchmark_error or "benchmark failed",
                })
                benchmarked[key] = 0.0  # mark as tried
                continue

            speedup = result.get("speedup", 1.0)
            benchmarked[key] = speedup
            validation = result.get("validation")
            entry = {
                "candidate_index": i,
                "config": candidate_config,
                "rationale": candidate.get("rationale", ""),
                "confidence": candidate.get("confidence", 0),
                "stats": result.get("recommended_stats", {}),
                "original_stats": result.get("original_stats", {}),
                "speedup": speedup,
            }
            if validation is not None:
                entry["validation"] = validation
            results_per_candidate.append(entry)

            # Cache baseline stats on first successful benchmark
            if state.get("baseline_benchmark_result") is None and result.get("original_stats"):
                state["baseline_benchmark_result"] = result["original_stats"]

            logger.info(
                f"Candidate {i}: speedup={speedup:.3f}x "
                f"(median={result.get('recommended_stats', {}).get('median', '?')}ms)"
            )

        state["benchmark_results_per_candidate"] = results_per_candidate
        state["benchmark_iterations"] = state.get("benchmark_iterations", 0) + 1

        # Append a snapshot of this round so feedback can show the agents every
        # config they have already tried across all rounds (avoids re-proposing
        # already-measured configs in feedback iterations 2+).
        history = state.get("benchmark_history")
        if history is None:
            history = []
            state["benchmark_history"] = history
        history.append({
            "round": state["benchmark_iterations"],
            "results": list(results_per_candidate),
        })

        # --- Select the best candidate ---
        successful = [r for r in results_per_candidate if "speedup" in r and "error" not in r]

        if not successful:
            logger.warning("All candidate benchmarks failed, triggering retry")
            state["regression_detected"] = True
            state["converged"] = False
            state["benchmark_results"] = {"error": "all candidates failed"}
            state["benchmarked_configs"] = benchmarked
            return state

        successful.sort(key=lambda r: r["speedup"], reverse=True)
        best = successful[0]

        state["benchmark_results"] = {
            "original_stats": best.get("original_stats", {}),
            "recommended_stats": best.get("stats", {}),
            "speedup": best["speedup"],
            "original_config": original_config,
            "recommended_config": best["config"],
        }

        optimization_result["benchmark_selection"] = {
            "num_candidates_tested": len(candidates),
            "num_successful": len(successful),
            "best_candidate_index": best["candidate_index"],
            "best_speedup": best["speedup"],
            "all_results": [
                {
                    "index": r["candidate_index"],
                    "speedup": r.get("speedup"),
                    "error": r.get("error"),
                }
                for r in results_per_candidate
            ],
        }

        if best["speedup"] < 1.0:
            regression_pct = (1.0 - best["speedup"]) * 100
            logger.warning(
                f"ALL candidates regressed. Best (index {best['candidate_index']}): "
                f"{regression_pct:.1f}% slower (speedup={best['speedup']:.3f}x). "
                f"Triggering retry."
            )
            state["regression_detected"] = True
            state["converged"] = False
        else:
            improvement_pct = (best["speedup"] - 1.0) * 100
            logger.info(
                f"BEST candidate {best['candidate_index']}: "
                f"{improvement_pct:.1f}% improvement (speedup={best['speedup']:.3f}x)"
            )
            state["regression_detected"] = False
            optimization_result["recommended_config"] = best["config"]
            state["optimization_result"] = optimization_result

        # ---- Phase 2: benchmark alternatives based on Phase 1 results ----
        phase1_best_speedup = best["speedup"] if best.get("speedup") else 0
        phase2_results = []

        if self.include_agent_proposals:
            dm_prop = state.get("data_movement_proposal") or {}
            rb_prop = state.get("resource_budget_proposal") or {}

            # Phase 2 runs in two modes:
            #  - phase1_failed: every Phase 1 candidate regressed against the
            #    baseline. The breadth alternatives (e.g. small XBLOCK) never
            #    reached Phase 1, so probing them now is exactly when they're
            #    most likely to recover the run.
            #  - all_similar: every Phase 1 result is within 5% of the best,
            #    meaning we're stuck on a flat ridge — broaden by sampling
            #    each agent's alternatives.
            # Both modes route through the same "pick unseen alt from each
            # of Orch/DM/RB" logic below.
            phase1_failed = phase1_best_speedup < 1.0
            phase1_speedups = [r["speedup"] for r in successful if "speedup" in r]
            winning_variant_for_trigger = candidates[best["candidate_index"]].get("variant_type", "")
            phase1_contrast_won = winning_variant_for_trigger == "orchestrator_contrast"
            all_similar = phase1_failed or phase1_contrast_won or (
                len(phase1_speedups) >= 2
                and max(phase1_speedups) > 0
                and (max(phase1_speedups) - min(phase1_speedups)) / max(phase1_speedups) < 0.05
            )

            if all_similar:
                # All Phase 1 configs similar OR all regressed OR contrast won — divide
                # num_candidates across 3 sources and pick the highest-
                # confidence unseen alt from each. Contrast-won case routes here
                # because the contrast candidate (variant_type=orchestrator_contrast)
                # otherwise has no Phase 2 dispatch branch and the agents'
                # alternatives (which often contain the small-XBLOCK regime BO
                # discovers) never reach the bench.
                N = self.num_candidates
                slots_per_source = max(1, N // 3)
                if phase1_failed:
                    print(f"    Phase 2: all Phase 1 candidates regressed (best {phase1_best_speedup:.2f}x) — benchmarking {slots_per_source} alt(s) from each of Orch/DM/RB to escape baseline")
                elif phase1_contrast_won:
                    print(f"    Phase 2: diversity contrast won Phase 1 ({phase1_best_speedup:.2f}x) — benchmarking {slots_per_source} alt(s) from each of Orch/DM/RB")
                else:
                    print(f"    Phase 2: all Phase 1 configs similar ({min(phase1_speedups):.2f}x–{max(phase1_speedups):.2f}x) — benchmarking {slots_per_source} alt(s) from each of Orch/DM/RB")

                # Helper: from a list of alts (sorted by confidence if available),
                # pick `n` unseen ones, advancing past redundant candidates.
                def _pick_unseen(alts, n, label_prefix):
                    picked = []
                    for j, alt in enumerate(alts):
                        if len(picked) >= n:
                            break
                        alt_cfg = alt.get("config", alt) if isinstance(alt, dict) else alt
                        if not isinstance(alt_cfg, dict):
                            continue
                        cfg = {k: v for k, v in alt_cfg.items()
                               if k in ("num_warps", "num_stages", "block_size",
                                        "shared_memory_bytes")}
                        if not cfg:
                            continue
                        cfg = _sanitize_candidate_config(cfg)
                        key = self._cfg_key(cfg)
                        if key in benchmarked:
                            continue  # skip redundant, try next
                        picked.append((f"{label_prefix}[{j}]", cfg, key))
                    return picked

                # Sort alternatives by confidence (if available)
                def _sort_by_conf(alts):
                    return sorted(alts, key=lambda a: -(a.get("confidence", 0) if isinstance(a, dict) else 0))

                dm_alts = _sort_by_conf(dm_prop.get("alternatives", []))
                rb_alts = _sort_by_conf(rb_prop.get("alternatives", []))
                orch_alts_raw = state.get("orchestrator_alternatives", [])

                alt_candidates = []
                alt_candidates.extend(_pick_unseen(orch_alts_raw, slots_per_source, "Orch alt"))
                alt_candidates.extend(_pick_unseen(dm_alts, slots_per_source, "DM alt"))
                alt_candidates.extend(_pick_unseen(rb_alts, slots_per_source, "RB alt"))

                for alt_label, cfg, key in alt_candidates:
                    result = self._run_benchmark(cfg, bc, original_config)
                    total_benchmarks += 1
                    if result is None:
                        benchmarked[key] = 0.0
                        continue
                    speedup = result.get("speedup", 1.0)
                    benchmarked[key] = speedup
                    phase2_results.append({"config": cfg, "speedup": speedup, "source": alt_label})
                    print(f"    Phase 2: {alt_label} = {speedup:.3f}x")

            else:
                # Phase 1 configs differ — benchmark alternatives of the winner
                winning_variant = candidates[best["candidate_index"]].get("variant_type", "")
                winning_agent = None
                if winning_variant == "agent_dm_preferred":
                    winning_agent = ("DM", dm_prop)
                elif winning_variant == "agent_rb_preferred":
                    winning_agent = ("RB", rb_prop)
                elif winning_variant == "orchestrator":
                    # Orchestrator won — benchmark Orch's alternatives; if none,
                    # fall back to the dedup'd agent's alternatives.
                    winning_agent = None
                    N = self.num_candidates
                    orch_alts = state.get("orchestrator_alternatives", [])
                    all_alts = []
                    for i, alt in enumerate(orch_alts):
                        if len(all_alts) >= N:
                            break
                        all_alts.append((f"Orch alt[{i}]", alt))
                    # Fallback: if orchestrator has no alternatives, use the
                    # dedup'd agent's alternatives (since Orch == that agent)
                    if not all_alts:
                        orch_cfg_key = self._cfg_key(_sanitize_candidate_config(
                            candidates[best["candidate_index"]].get("config", {})
                        ))
                        for label, prop in [("DM", dm_prop), ("RB", rb_prop)]:
                            pref_cfg = prop.get("preferred", {})
                            if pref_cfg and self._cfg_key(_sanitize_candidate_config(pref_cfg)) == orch_cfg_key:
                                for i, alt in enumerate(prop.get("alternatives", [])[:N]):
                                    all_alts.append((f"{label} alt[{i}] (Orch fallback)", alt))
                                break
                    print(f"    Phase 2: Orch won ({phase1_best_speedup:.3f}x) — benchmarking {len(all_alts)} alternatives")
                    for alt_label, alt_cfg in all_alts:
                        cfg = {k: v for k, v in alt_cfg.items()
                               if k in ("num_warps", "num_stages", "block_size",
                                        "shared_memory_bytes")}
                        if not cfg:
                            continue
                        cfg = _sanitize_candidate_config(cfg)
                        key = self._cfg_key(cfg)
                        if key in benchmarked:
                            phase2_results.append({"config": cfg, "speedup": benchmarked[key], "source": alt_label})
                            continue
                        result = self._run_benchmark(cfg, bc, original_config)
                        total_benchmarks += 1
                        if result is None:
                            benchmarked[key] = 0.0
                            continue
                        speedup = result.get("speedup", 1.0)
                        benchmarked[key] = speedup
                        phase2_results.append({"config": cfg, "speedup": speedup, "source": alt_label})
                        print(f"    Phase 2: {alt_label} = {speedup:.3f}x")

                if winning_agent:
                    agent_label, agent_prop = winning_agent
                    alternatives = agent_prop.get("alternatives", [])[:self.num_candidates]
                    if alternatives:
                        print(f"    Phase 2: {agent_label} preferred won ({phase1_best_speedup:.3f}x) — benchmarking {len(alternatives)} alternatives")
                        for i, alt in enumerate(alternatives):
                            cfg = {k: v for k, v in alt.items()
                                   if k in ("num_warps", "num_stages", "block_size",
                                            "shared_memory_bytes")}
                            if not cfg:
                                continue
                            cfg = _sanitize_candidate_config(cfg)
                            key = self._cfg_key(cfg)
                            if key in benchmarked:
                                phase2_results.append({"config": cfg, "speedup": benchmarked[key], "source": f"{agent_label} alt[{i}]"})
                                continue
                            result = self._run_benchmark(cfg, bc, original_config)
                            total_benchmarks += 1
                            if result is None:
                                benchmarked[key] = 0.0
                                continue
                            speedup = result.get("speedup", 1.0)
                            benchmarked[key] = speedup
                            phase2_results.append({"config": cfg, "speedup": speedup, "source": f"{agent_label} alt[{i}]"})
                            logger.info(f"Phase 2 {agent_label} alt {i}: speedup={speedup:.3f}x")

            if phase2_results:
                best_phase2 = max(phase2_results, key=lambda r: r["speedup"])
                if best_phase2["speedup"] > phase1_best_speedup:
                    print(f"    Phase 2 found better: {best_phase2['speedup']:.3f}x (was {phase1_best_speedup:.3f}x) from {best_phase2.get('source','?')}")

                optimization_result["benchmark_selection"]["phase2"] = {
                    "num_tested": len(phase2_results),
                    "best_speedup": best_phase2["speedup"],
                    "phase1_best": phase1_best_speedup,
                }

        elif not self.include_agent_proposals and phase1_best_speedup >= 1.0:
            # Original Phase 2 for non-proposal mode (A1a/A1b)
            winning_variant = candidates[best["candidate_index"]].get("variant_type", "")
            dm_prop = state.get("data_movement_proposal") or {}
            rb_prop = state.get("resource_budget_proposal") or {}
            winning_agent = None
            if "dm" in winning_variant:
                winning_agent = ("DM", dm_prop)
            elif "rb" in winning_variant:
                winning_agent = ("RB", rb_prop)

            if winning_agent:
                agent_label, agent_prop = winning_agent
                alternatives = agent_prop.get("alternatives", [])[:self.num_candidates]
                if alternatives:
                    print(f"    Phase 2: {agent_label} preferred won — benchmarking {len(alternatives)} alternatives")
                    for i, alt in enumerate(alternatives):
                        cfg = {k: v for k, v in alt.items()
                               if k in ("num_warps", "num_stages", "block_size",
                                        "shared_memory_bytes")}
                        if not cfg:
                            continue
                        cfg = _sanitize_candidate_config(cfg)
                        key = self._cfg_key(cfg)
                        if key in benchmarked:
                            phase2_results.append({"config": cfg, "speedup": benchmarked[key]})
                            continue
                        result = self._run_benchmark(cfg, bc, original_config)
                        total_benchmarks += 1
                        if result is None:
                            benchmarked[key] = 0.0
                            continue
                        speedup = result.get("speedup", 1.0)
                        benchmarked[key] = speedup
                        phase2_results.append({"config": cfg, "speedup": speedup})

                    if phase2_results:
                        best_phase2 = max(phase2_results, key=lambda r: r["speedup"])
                        if best_phase2["speedup"] > phase1_best_speedup:
                            print(f"    Phase 2 found better: {best_phase2['speedup']:.3f}x (was {phase1_best_speedup:.3f}x)")

        # ---- Find overall best across Phase 1 + Phase 2 ----
        all_tested = [(best["config"], best["speedup"])]
        for r in phase2_results:
            if "speedup" in r:
                all_tested.append((r["config"], r["speedup"]))
        overall_best_cfg, overall_best_speedup = max(all_tested, key=lambda x: x[1])

        # Update state with best so far
        if overall_best_speedup > phase1_best_speedup:
            best = {**best, "config": overall_best_cfg, "speedup": overall_best_speedup}
            optimization_result["recommended_config"] = overall_best_cfg
            state["benchmark_results"]["speedup"] = overall_best_speedup
            state["benchmark_results"]["recommended_config"] = overall_best_cfg
            state["optimization_result"] = optimization_result

        # ---- Phase 3: neighborhood search around overall best config ----
        if overall_best_speedup >= 1.0:
            best_cfg = overall_best_cfg
            neighbors = self._generate_neighbors(best_cfg)
            pre_phase3_speedup = overall_best_speedup
            phase3_results = []

            # Filter to only unseen neighbors
            unseen = []
            for n in neighbors:
                key = self._cfg_key(n)
                if key in benchmarked:
                    phase3_results.append({"config": n, "speedup": benchmarked[key]})
                else:
                    unseen.append(n)

            if unseen:
                logger.info(f"Phase 3: neighborhood search — {len(unseen)} unseen neighbors")
                print(f"    Phase 3: benchmarking {len(unseen)} neighbors of best config ({overall_best_speedup:.3f}x)")

                for n_cfg in unseen:
                    result = self._run_benchmark(n_cfg, bc, original_config)
                    total_benchmarks += 1
                    key = self._cfg_key(n_cfg)
                    if result is None:
                        benchmarked[key] = 0.0
                        continue
                    speedup = result.get("speedup", 1.0)
                    benchmarked[key] = speedup
                    phase3_results.append({"config": n_cfg, "speedup": speedup})
                    logger.info(f"Phase 3 neighbor: {n_cfg} -> {speedup:.3f}x")

            if phase3_results:
                best_neighbor = max(phase3_results, key=lambda r: r["speedup"])
                promoted = best_neighbor["speedup"] > overall_best_speedup
                if promoted:
                    logger.info(
                        f"Phase 3 winner: {best_neighbor['speedup']:.3f}x > "
                        f"previous best {overall_best_speedup:.3f}x"
                    )
                    print(
                        f"    Phase 3 found better: "
                        f"{best_neighbor['speedup']:.3f}x (was {overall_best_speedup:.3f}x)"
                    )
                    best = {**best, **best_neighbor}
                    optimization_result["recommended_config"] = best_neighbor["config"]
                    state["benchmark_results"]["speedup"] = best_neighbor["speedup"]
                    state["benchmark_results"]["recommended_config"] = best_neighbor["config"]
                    state["optimization_result"] = optimization_result

                optimization_result["benchmark_selection"]["phase3"] = {
                    "num_neighbors_tested": len(phase3_results),
                    "best_neighbor_speedup": best_neighbor["speedup"],
                    "pre_phase3_speedup": pre_phase3_speedup,
                    "promoted": promoted,
                }

        # ---- Final: report best config across all phases ----
        final_best_speedup = best.get("speedup", 0)
        final_best_cfg = best.get("config", {})
        total_configs_tested = len(benchmarked)
        print(f"    [Summary] Best: {final_best_speedup:.3f}x | Total configs tested: {total_configs_tested} | Benchmarks run: {total_benchmarks}")

        # Store dedup set and total benchmark count
        state["benchmarked_configs"] = benchmarked
        optimization_result["benchmark_selection"]["total_benchmarks_run"] = total_benchmarks

        # --- Optional NCU profiling on best candidate ---
        profiler_fn = getattr(self, 'profiler_fn', None)
        if profiler_fn is not None and successful:
            bc = state["benchmark_config"]
            try:
                profiling_data = profiler_fn(
                    config=best["config"],
                    shapes_file=Path(bc["shapes_file"]),
                    kernel_module=Path(bc["kernel_module"]),
                    kernel_name=bc["kernel_name"],
                    kernel_class=bc.get("kernel_class"),
                    device=bc.get("device", "cuda"),
                )
                if profiling_data:
                    state["profiling_data"] = profiling_data
                    logger.info(
                        "NCU profiling completed: %d metrics collected",
                        len(profiling_data),
                    )
                else:
                    logger.info("NCU profiling returned no data")
            except Exception as e:
                logger.warning("NCU profiling failed (non-fatal): %s", e)

        _print_benchmark_summary(state)
        return state

    def _check_convergence_node(self, state: OptimizationState) -> OptimizationState:
        """Enhanced convergence check with adaptive refinement and benchmark regression handling"""
        state["iteration"] += 1

        # Save current iteration to history
        if state.get("optimization_result"):
            result = state["optimization_result"]
            state["iteration_history"].append({
                "iteration": state["iteration"],
                "result": result,
                "confidence": result.get("confidence_score", 0),
                "bottleneck": result.get("warp_analysis", {}).get("coalescing_efficiency", {}).get("estimated_percentage", 100),
                "benchmark": state.get("benchmark_results"),
            })

        # --- Benchmark regression handling (must come before other checks) ---
        if state.get("regression_detected"):
            bench = state.get("benchmark_results", {})
            bench_iters = state.get("benchmark_iterations", 0)
            feedback_iters = state.get("feedback_iteration", 0)
            max_feedback = state.get("max_feedback_iterations", 2)

            if bench_iters >= MAX_BENCHMARK_RETRIES or feedback_iters >= max_feedback:
                # All retries exhausted — fall back to original configuration
                kc = state["kernel_config"]
                original_config = {
                    "num_warps": kc.num_warps,
                    "num_stages": kc.num_stages,
                    "block_size": dict(kc.block_size),
                    "grid": dict(kc.grid),
                }
                regression_history = [
                    entry.get("benchmark")
                    for entry in state.get("iteration_history", [])
                    if entry.get("benchmark")
                ]
                orig_ms = bench.get("original_stats", {}).get("median", "N/A")
                rec_ms = bench.get("recommended_stats", {}).get("median", "N/A")
                state["optimization_result"] = {
                    "kernel_name": kc.kernel_name,
                    "original_config": original_config,
                    "recommended_config": dict(original_config),  # Same as original
                    "expected_improvement": "0% (keeping original — all optimizations regressed)",
                    "confidence_score": 1.0,
                    "recommendation": "keep_original",
                    "regression_history": regression_history,
                    "coordinator_reasoning": (
                        f"After {MAX_BENCHMARK_RETRIES} optimization attempts, all recommended "
                        f"configurations were slower than the original. The original configuration "
                        f"is retained. Last attempt: original={orig_ms}ms, "
                        f"recommended={rec_ms}ms."
                    ),
                    "tradeoffs": [],
                    "implementation_instructions": "No changes needed — keep original configuration.",
                    "risk_assessment": "low",
                }
                # Clear stale benchmark data — the timing in benchmark_results
                # belongs to the last *failed* LLM config, not the original
                # config we are falling back to.  Keeping it would cause the
                # summary to pair the original config with a wrong median time.
                state["benchmark_results"] = {}
                state["converged"] = True
                state["regression_detected"] = False
                logger.warning(
                    f"All {MAX_BENCHMARK_RETRIES} benchmark attempts regressed. "
                    f"Falling back to original configuration."
                )
                return state

            # Still have retries: feed back actual timing data and continue
            orig_ms = bench.get("original_stats", {}).get("median", "N/A")
            rec_ms = bench.get("recommended_stats", {}).get("median", "N/A")
            speedup = bench.get("speedup", 0)
            regression_pct = f"{(1.0 - speedup) * 100:.1f}" if isinstance(speedup, (int, float)) else "N/A"
            prev_rec_config = bench.get("recommended_config", {})

            feedback = (
                f"BENCHMARK REGRESSION (attempt {bench_iters}/{MAX_BENCHMARK_RETRIES}): "
                f"Your recommended configuration measured {rec_ms}ms median execution time, "
                f"which is {regression_pct}% SLOWER than the original {orig_ms}ms. "
                f"Recommended config that failed: {prev_rec_config}. "
                f"REVISE your recommendation — any new config MUST be faster than {orig_ms}ms."
            )
            state["previous_coordinator_feedback"] = feedback
            state["refinement_focus"] = "benchmark_regression"
            state["converged"] = False
            state["regression_detected"] = False  # Reset for next iteration
            # Increment feedback iteration counter for Approach 7 feedback loop
            state["feedback_iteration"] = state.get("feedback_iteration", 0) + 1
            logger.info(
                f"Benchmark regression on attempt {bench_iters}/{MAX_BENCHMARK_RETRIES} "
                f"(feedback round {state['feedback_iteration']}), "
                f"retrying with timing feedback..."
            )
            return state

        # --- Standard convergence checks ---
        result = state.get("optimization_result")

        # Converge if high confidence result
        if result and result.get("confidence_score", 0) >= 0.85:
            state["converged"] = True
            logger.info(f"Converged with high confidence: {result['confidence_score']:.2f}")
            return state

        # Converge if max iterations
        if state["iteration"] >= state["max_iterations"]:
            state["converged"] = True
            logger.info(f"Max iterations reached: {state['iteration']}")
            return state

        # Converge if error
        if state.get("error"):
            state["converged"] = True
            return state

        # If continuing, set refinement focus based on bottleneck severity
        if result and not state["converged"]:
            # Determine primary bottleneck from analyses
            warp_analysis = state.get("warp_analysis", {})
            sm_analysis = state.get("sm_analysis", {})
            gpu_analysis = state.get("gpu_analysis", {})

            # Calculate severity scores
            warp_coalescing = warp_analysis.get("coalescing_efficiency", {}).get("estimated_percentage", 100)
            sm_occupancy = sm_analysis.get("occupancy_analysis", {}).get("current_occupancy_pct", 100)
            gpu_saturation = gpu_analysis.get("grid_analysis", {}).get("sm_saturation_pct", 100)

            # Determine primary bottleneck
            bottleneck = "unknown"
            severity = "minor"

            if warp_coalescing < 50:
                bottleneck = "warp"
                severity = "critical"
            elif sm_occupancy < 25:
                bottleneck = "sm"
                severity = "critical"
            elif gpu_saturation < 50:
                bottleneck = "gpu"
                severity = "critical"
            elif warp_coalescing < 70:
                bottleneck = "warp"
                severity = "moderate"
            elif sm_occupancy < 50:
                bottleneck = "sm"
                severity = "moderate"
            elif gpu_saturation < 75:
                bottleneck = "gpu"
                severity = "moderate"

            if severity in ["critical", "moderate"]:
                state["refinement_focus"] = bottleneck
                state["previous_coordinator_feedback"] = (
                    f"Previous iteration identified {bottleneck}-level bottleneck "
                    f"with {severity} severity. Focus refinement on this area. "
                    f"Previous confidence: {result.get('confidence_score', 0):.2f}"
                )
                logger.info(f"Iteration {state['iteration']}: Refining {bottleneck}-level optimizations")

        return state

    def _should_continue(self, state: OptimizationState) -> Literal["continue", "end"]:
        """Decide whether to continue iterating or end"""
        if state["converged"] or state.get("error"):
            print(f"\n  {'>'*3} [Convergence] DONE — iteration {state.get('iteration', 0)}")
            return "end"
        print(f"\n  {'>'*3} [Convergence] NOT converged — looping back (iteration {state.get('iteration', 0)})")
        return "continue"

    def optimize(
        self,
        kernel_config: KernelConfig,
        constraints: Optional[Dict[str, Any]] = None,
        benchmark_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Optimize a Triton kernel configuration

        Args:
            kernel_config: Input kernel configuration
            constraints: Optional constraints (e.g., max_shared_memory)
            benchmark_config: Optional dict with benchmark settings for
                closed-loop verification.  Keys: kernel_module, kernel_name,
                kernel_class, shapes_file, device, warmup, rep.

        Returns:
            Optimization result dictionary
        """
        logger.info(f"Starting optimization for kernel: {kernel_config.kernel_name}")

        print(f"\n{_SEPARATOR}")
        print(f"  MULTI-AGENT OPTIMIZATION: {kernel_config.kernel_name}")
        print(f"  Architecture: {self.architecture_code}  |  Max iterations: {self.max_iterations}  |  Candidates: {self.num_candidates}")
        print(f"{_SEPARATOR}")

        # When benchmarking is enabled, ensure enough iterations for retries
        effective_max_iter = self.max_iterations
        if benchmark_config and self.benchmark_fn:
            effective_max_iter = max(self.max_iterations, MAX_BENCHMARK_RETRIES)

        # Initialize state
        initial_state: OptimizationState = {
            "kernel_config": kernel_config,
            "messages": [],
            "warp_analysis": None,
            "sm_analysis": None,
            "gpu_analysis": None,
            "constraints": constraints or {},
            "warp_proposal": None,
            "sm_proposal": None,
            "gpu_proposal": None,
            "optimization_result": None,
            "iteration": 0,
            "max_iterations": effective_max_iter,
            "converged": False,
            "error": None,
            "iteration_history": [],
            "refinement_focus": None,
            "previous_coordinator_feedback": None,
            # Benchmark-in-the-loop fields
            "benchmark_results": None,
            "baseline_benchmark_result": None,
            "regression_detected": False,
            "benchmark_iterations": 0,
            "benchmark_config": benchmark_config,
            # Multi-candidate search fields (Approach 5)
            "candidate_configs": [],
            "benchmark_results_per_candidate": [],
            "benchmark_history": [],
            # Approach 7: negotiation-based agent architecture
            "data_movement_proposal": None,
            "resource_budget_proposal": None,
            "computed_grid": None,
            "execution_metrics": None,
            "conflict_detection": None,
            "negotiation_history": None,
            "consensus_reached": False,
            "negotiation_round": 0,
            "max_negotiation_rounds": 2,
            "negotiation_in_progress": False,
            "agent_feedback": {},
            "feedback_iteration": 0,
            "max_feedback_iterations": 2,
            "profiling_data": None,
        }

        # Run graph
        try:
            final_state = self.graph.invoke(initial_state)

            # Print final config summary
            print(f"\n  {'>'*3} [Final Configuration]")
            _print_final_config(final_state)
            print(f"{_SEPARATOR}\n")
            sys.stdout.flush()

            if final_state.get("error"):
                logger.error(f"Optimization failed: {final_state['error']}")
                return {
                    "success": False,
                    "error": final_state["error"],
                    "kernel_name": kernel_config.kernel_name,
                }

            result = final_state.get("optimization_result")
            if not result:
                logger.warning("Optimization completed but no result produced")
                return {
                    "success": False,
                    "error": "No optimization result produced",
                    "kernel_name": kernel_config.kernel_name,
                }

            logger.info(f"Optimization complete: {result.get('expected_improvement', 'N/A')} expected improvement")

            response = {
                "success": True,
                "result": result,
                "iterations": final_state["iteration"],
                "messages": [msg.to_dict() for msg in final_state["messages"]],
            }

            # Include benchmark data when available
            if final_state.get("benchmark_results"):
                response["benchmark"] = final_state["benchmark_results"]
                response["benchmark_iterations"] = final_state.get("benchmark_iterations", 0)

            # Include multi-candidate benchmark data (Approach 5)
            if final_state.get("benchmark_results_per_candidate"):
                response["benchmark_candidates"] = final_state["benchmark_results_per_candidate"]

            return response

        except Exception as e:
            logger.error(f"Orchestration error: {e}", exc_info=True)
            print(f"\n  ERROR: Optimization failed: {e}", flush=True)
            return {
                "success": False,
                "error": str(e),
                "kernel_name": kernel_config.kernel_name,
            }

    def optimize_from_ttgir(
        self,
        kernel_name: str,
        ttgir: str,
        current_config: Dict[str, Any],
        tensor_shapes: Optional[Dict[str, Any]] = None,
        tensor_dtypes: Optional[Dict[str, str]] = None,
        ttir: Optional[str] = None,
        source_code: Optional[str] = None,
        optimization_target: str = "throughput",
        benchmark_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Optimize from TTGIR directly (convenience method)

        Args:
            kernel_name: Kernel name
            ttgir: Triton GPU IR
            current_config: Current configuration dict
            tensor_shapes: Tensor shape information (e.g., {"x": [1024, 1024]})
            tensor_dtypes: Tensor dtype information (e.g., {"x": "float32"})
            ttir: Optional high-level Triton IR
            source_code: Optional original Triton source
            optimization_target: "throughput", "latency", or "energy"
            benchmark_config: Optional benchmark settings for closed-loop verification

        Returns:
            Optimization result
        """
        # Build kernel config
        kernel_config = KernelConfig(
            kernel_name=kernel_name,
            ttgir=ttgir,
            ttir=ttir,
            source_code=source_code,
            num_warps=current_config.get("num_warps", 4),
            num_stages=current_config.get("num_stages", 2),
            num_ctas=current_config.get("num_ctas", 1),
            block_size=current_config.get("block_size", {}),
            grid=current_config.get("grid", {"x": 1, "y": 1, "z": 1}),
            grid_formula=current_config.get("grid_formula", ""),
            shared_memory_bytes=current_config.get("shared_memory_bytes", 0),
            global_loads=current_config.get("global_loads", 0),
            global_stores=current_config.get("global_stores", 0),
            tensor_shapes=tensor_shapes or {},
            tensor_dtypes=tensor_dtypes or {},
            constexpr_param_names=current_config.get("constexpr_param_names", []),
            target_architecture=self.architecture_code,
            optimization_target=optimization_target,
        )

        return self.optimize(kernel_config, benchmark_config=benchmark_config)


# Convenience function
def optimize_kernel(
    kernel_name: str,
    ttgir: str,
    current_config: Dict[str, Any],
    architecture: str = "cuda:70",
    llm_provider: str = "deepseek",
    tensor_shapes: Optional[Dict[str, Any]] = None,
    tensor_dtypes: Optional[Dict[str, str]] = None,
    ttir: Optional[str] = None,
    source_code: Optional[str] = None,
    optimization_target: str = "throughput",
    debug_mode: bool = False,
    debug_dir: Optional[str] = None,
    benchmark_fn: Optional[Callable] = None,
    benchmark_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Quick optimization of a kernel

    Args:
        kernel_name: Kernel name
        ttgir: Triton GPU IR
        current_config: Current configuration
        architecture: Target GPU architecture
        llm_provider: LLM provider to use
        tensor_shapes: Input tensor shapes (e.g., {"x": [1024, 1024], "y": [1024]})
        tensor_dtypes: Input tensor dtypes (e.g., {"x": "float32", "y": "float32"})
        ttir: Optional high-level Triton IR
        source_code: Optional original Triton source code
        optimization_target: Optimization target ("throughput", "latency", "energy")
        debug_mode: Enable detailed logging (saves prompts/responses to files)
        debug_dir: Directory for debug artifacts (default: ./debug_multiagent/)
        benchmark_fn: Optional callable for closed-loop benchmark verification
        benchmark_config: Optional benchmark settings dict
        **kwargs: Additional arguments for orchestrator

    Returns:
        Optimization result
    """
    orchestrator = GPUOptimizationOrchestrator(
        llm_provider=llm_provider,
        architecture_code=architecture,
        debug_mode=debug_mode,
        debug_dir=debug_dir,
        benchmark_fn=benchmark_fn,
        **kwargs
    )

    return orchestrator.optimize_from_ttgir(
        kernel_name=kernel_name,
        ttgir=ttgir,
        current_config=current_config,
        tensor_shapes=tensor_shapes,
        ttir=ttir,
        source_code=source_code,
        optimization_target=optimization_target,
        benchmark_config=benchmark_config,
    )
