import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Import your existing LLM components
from LLMclient import LLMClient, OptimizationPipeline

class TritonResultsAnalyzer:
    def __init__(self, results_file: str = "triton_autotune_results.json"):
        self.results_file = Path(results_file)
        self.results_data = self._load_results()
    
    def _load_results(self) -> Dict[str, Any]:
        """Load the autotune results JSON file"""
        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")
        
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def extract_kernel_code(self, file_path: str, kernel_name: str) -> Optional[str]:
        """Extract the kernel source code from the original Triton file"""
        try:
            triton_file = Path(file_path)
            if not triton_file.exists():
                print(f"❌ Triton file not found: {file_path}")
                return None
            
            # Read the Triton source file
            with open(triton_file, 'r') as f:
                source_code = f.read()
            
            # Extract the specific kernel function using regex
            kernel_pattern = rf'@triton\.jit\s*\n\s*def\s+{kernel_name}\s*\([^)]*\)\s*:\s*\n(.*?)(?=\n\s*@triton\.jit|\n\s*def\s+\w+|\Z)'
            match = re.search(kernel_pattern, source_code, re.DOTALL)
            
            if match:
                # Get the full function including the decorator
                full_pattern = rf'@triton\.jit\s*\n\s*def\s+{kernel_name}\(.*?\):\s*\n.*?(?=@triton\.jit\s*\n\s*def\s+\w+|def\s+\w+|\Z)'
                full_match = re.search(full_pattern, source_code, re.DOTALL)
                if full_match:
                    return full_match.group(0).strip()
            
            print(f"❌ Could not extract kernel {kernel_name} from {file_path}")
            return None
            
        except Exception as e:
            print(f"❌ Error extracting kernel code: {e}")
            return None
    
    def get_all_kernels_to_analyze(self) -> List[Dict[str, Any]]:
        """Get all kernels from results that need LLM analysis"""
        kernels_to_analyze = []
        
        for run in self.results_data.get("results", []):
            file_path = run.get("file", "")
            for kernel in run.get("kernels", []):
                kernel_info = {
                    "file_path": file_path,
                    "kernel_name": kernel.get("kernel", ""),
                    "xnumel": kernel.get("xnumel", 0),
                    "best_config": kernel.get("best", {}),
                    "near_opt_count": kernel.get("near_opt_count", 0),
                    "near_opt_configs": kernel.get("near_opt", []),  # Added this line
                    "near_opt_threshold_pct": kernel.get("near_opt_threshold_pct", 10),  # Added this line
                    "attempts": kernel.get("attempts", 0),
                    "latency_ms": kernel.get("best", {}).get("lat_ms", 0)
                }
                kernels_to_analyze.append(kernel_info)
        
        return kernels_to_analyze
    
    def analyze_kernel_with_llm(self, kernel_info: Dict[str, Any], 
                               llm_client: LLMClient, 
                               gpu_model: str = "v100") -> Dict[str, Any]:
        """Analyze a single kernel with LLM"""
        print(f"\n🔍 Analyzing kernel: {kernel_info['kernel_name']}")
        print(f"   File: {Path(kernel_info['file_path']).name}")
        print(f"   XNumel: {kernel_info['xnumel']}")
        print(f"   Best Latency: {kernel_info['latency_ms']:.3f}ms")
        
        # Extract kernel source code
        kernel_code = self.extract_kernel_code(
            kernel_info['file_path'], 
            kernel_info['kernel_name']
        )
        print("=== KERNEL CODE ===")
        print(kernel_code)
        
        if not kernel_code:
            return {
                "error": "Could not extract kernel code",
                **kernel_info
            }
        
        # Create pipeline and run analysis
        pipeline = OptimizationPipeline(llm_client)
        
        try:
            result = pipeline.analyze_and_optimize(
                kernel_source=kernel_code,
                gpu_model=gpu_model,
                temperature=0.1,
                max_tokens=2000
            )
            
            return {
                "success": True,
                "kernel_info": kernel_info,
                "llm_analysis": result,
                "kernel_code_snippet": kernel_code[:500] + "..." if len(kernel_code) > 500 else kernel_code
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "kernel_info": kernel_info
            }
    
    def analyze_all_kernels(self, llm_client: LLMClient, 
                           gpu_model: str = "v100",
                           max_kernels: int = None) -> List[Dict[str, Any]]:
        """Analyze all kernels with LLM"""
        kernels_to_analyze = self.get_all_kernels_to_analyze()
        
        if max_kernels:
            kernels_to_analyze = kernels_to_analyze[:max_kernels]
        
        print(f"🎯 Analyzing {len(kernels_to_analyze)} kernels with LLM...")
        
        results = []
        for i, kernel_info in enumerate(kernels_to_analyze, 1):
            print(f"\n[{i}/{len(kernels_to_analyze)}] ", end="")
            
            result = self.analyze_kernel_with_llm(kernel_info, llm_client, gpu_model)
            results.append(result)
            
            # Add small delay to avoid overwhelming the LLM
            import time
            time.sleep(1)
        
        return results
    
    def evaluate_llm_config_quality(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate whether LLM can select one of the best configurations
        by comparing LLM recommendations with actual near-optimal configs
        """
        evaluation_results = {
            "total_kernels": 0,
            "kernels_with_near_opt_data": 0,
            "llm_recommended_near_optimal": 0,
            "llm_recommended_best": 0,
            "llm_performance_within_threshold": 0,
            "detailed_comparisons": []
        }
        
        for result in analysis_results:
            if not result.get("success", False):
                continue
                
            kernel_info = result["kernel_info"]
            llm_config = result["llm_analysis"]["parsed_configuration"]
            
            evaluation_results["total_kernels"] += 1
            
            # Check if we have near-optimal configuration data
            if kernel_info.get("near_opt_count", 0) > 0:
                evaluation_results["kernels_with_near_opt_data"] += 1
                
                # Compare LLM config with near-optimal configs
                comparison = self._compare_llm_with_near_optimal(llm_config, kernel_info)
                evaluation_results["detailed_comparisons"].append(comparison)
                
                if comparison["is_near_optimal"]:
                    evaluation_results["llm_recommended_near_optimal"] += 1
                
                if comparison["is_best_config"]:
                    evaluation_results["llm_recommended_best"] += 1
                
                if comparison["performance_within_threshold"]:
                    evaluation_results["llm_performance_within_threshold"] += 1
        
        # Calculate success rates
        if evaluation_results["kernels_with_near_opt_data"] > 0:
            evaluation_results["near_optimal_success_rate"] = (
                evaluation_results["llm_recommended_near_optimal"] / 
                evaluation_results["kernels_with_near_opt_data"] * 100
            )
            evaluation_results["best_config_success_rate"] = (
                evaluation_results["llm_recommended_best"] / 
                evaluation_results["kernels_with_near_opt_data"] * 100
            )
            evaluation_results["threshold_success_rate"] = (
                evaluation_results["llm_performance_within_threshold"] / 
                evaluation_results["kernels_with_near_opt_data"] * 100
            )
        
        return evaluation_results
    
    def _compare_llm_with_near_optimal(self, llm_config: Dict[str, Any], 
                                     kernel_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare LLM configuration with near-optimal configurations
        """
        best_config = kernel_info["best_config"]
        near_opt_configs = kernel_info.get("near_opt_configs", [])
        threshold_pct = kernel_info.get("near_opt_threshold_pct", 10)
        
        comparison = {
            "kernel_name": kernel_info["kernel_name"],
            "llm_config": llm_config,
            "best_config": best_config,
            "near_opt_count": len(near_opt_configs),
            "is_near_optimal": False,
            "is_best_config": False,
            "performance_within_threshold": False,
            "matched_config": None,
            "performance_gap_pct": None
        }
        
        # Normalize LLM config for comparison
        llm_norm = self._normalize_config(llm_config)
        best_norm = self._normalize_config(best_config)
        
        # Check if LLM config matches the best config
        if self._configs_match(llm_norm, best_norm):
            comparison["is_best_config"] = True
            comparison["is_near_optimal"] = True
            comparison["performance_within_threshold"] = True
            comparison["matched_config"] = "best"
            comparison["performance_gap_pct"] = 0.0
            return comparison
        
        # Check if LLM config matches any near-optimal config
        for near_opt in near_opt_configs:
            near_opt_config = near_opt.get("config", {})
            near_opt_norm = self._normalize_config(near_opt_config)
            
            if self._configs_match(llm_norm, near_opt_norm):
                comparison["is_near_optimal"] = True
                comparison["matched_config"] = near_opt_config
                
                # Calculate performance gap
                near_opt_latency = near_opt.get("latency", kernel_info["latency_ms"])
                best_latency = kernel_info["latency_ms"]
                performance_gap = (near_opt_latency / best_latency - 1) * 100
                comparison["performance_gap_pct"] = performance_gap
                
                if performance_gap <= threshold_pct:
                    comparison["performance_within_threshold"] = True
                
                break
        
        # If no exact match, check if LLM config parameters are reasonable
        if not comparison["is_near_optimal"]:
            comparison["parameter_analysis"] = self._analyze_llm_parameters(llm_norm, near_opt_configs)
        
        return comparison
    
    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize configuration keys for comparison"""
        normalized = {}
        
        for key, value in config.items():
            norm_key = key.upper().replace('_', '').replace(' ', '')
            
            # Handle common variations
            if norm_key in ['XBLOCK', 'BLOCKSIZE']:
                normalized['XBLOCK'] = int(value)
            elif norm_key in ['NUMWARPS', 'WARPS', 'NUM_WARPS']:
                normalized['NUM_WARPS'] = int(value)
            elif norm_key in ['NUMSTAGES', 'STAGES', 'NUM_STAGES']:
                normalized['NUM_STAGES'] = int(value)
            elif norm_key in ['EVICTIONPOLICY', 'EVICTION', 'EVICT']:
                normalized['EVICTION_POLICY'] = str(value).lower()
            else:
                normalized[norm_key] = value
        
        return normalized
    
    def _configs_match(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> bool:
        """Check if two configurations match"""
        keys_to_compare = ['XBLOCK', 'NUM_WARPS', 'NUM_STAGES']
        
        for key in keys_to_compare:
            if key in config1 and key in config2:
                if config1[key] != config2[key]:
                    return False
            elif (key in config1) != (key in config2):
                return False
        
        return True
    
    def _analyze_llm_parameters(self, llm_config: Dict[str, Any], 
                              near_opt_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if LLM parameters are within reasonable ranges"""
        analysis = {"reasonable_parameters": True, "parameter_checks": {}}
        
        # Extract parameter ranges from near-optimal configs
        xblock_values = []
        warp_values = []
        stage_values = []
        
        for config_data in near_opt_configs:
            config = config_data.get("config", {})
            norm_config = self._normalize_config(config)
            
            if 'XBLOCK' in norm_config:
                xblock_values.append(norm_config['XBLOCK'])
            if 'NUM_WARPS' in norm_config:
                warp_values.append(norm_config['NUM_WARPS'])
            if 'NUM_STAGES' in norm_config:
                stage_values.append(norm_config['NUM_STAGES'])
        
        # Check LLM parameters against ranges
        if xblock_values and 'XBLOCK' in llm_config:
            llm_xblock = llm_config['XBLOCK']
            analysis["parameter_checks"]["xblock"] = {
                "llm_value": llm_xblock,
                "near_opt_range": f"{min(xblock_values)}-{max(xblock_values)}",
                "in_range": min(xblock_values) <= llm_xblock <= max(xblock_values)
            }
            if not analysis["parameter_checks"]["xblock"]["in_range"]:
                analysis["reasonable_parameters"] = False
        
        if warp_values and 'NUM_WARPS' in llm_config:
            llm_warps = llm_config['NUM_WARPS']
            analysis["parameter_checks"]["num_warps"] = {
                "llm_value": llm_warps,
                "near_opt_range": f"{min(warp_values)}-{max(warp_values)}",
                "in_range": min(warp_values) <= llm_warps <= max(warp_values)
            }
            if not analysis["parameter_checks"]["num_warps"]["in_range"]:
                analysis["reasonable_parameters"] = False
        
        return analysis
    
    def generate_analysis_report(self, analysis_results: List[Dict[str, Any]], 
                               output_file: str = "llm_analysis_report.json"):
        """Generate a comprehensive report of LLM analysis results"""
        # Evaluate LLM config quality
        evaluation = self.evaluate_llm_config_quality(analysis_results)
        
        report = {
            "metadata": {
                "total_kernels_analyzed": len(analysis_results),
                "successful_analyses": sum(1 for r in analysis_results if r.get("success", False)),
                "failed_analyses": sum(1 for r in analysis_results if not r.get("success", False)),
                "generated_at": str(Path(output_file).with_suffix('').name)
            },
            "llm_performance_evaluation": evaluation,
            "analysis_results": analysis_results
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Analysis report saved to: {output_file}")
        return report

def demo_analyze_with_ollama(results_file: str = "triton_autotune_results.json", 
                           max_kernels: int = 3):
    """Demo function to analyze autotune results with Ollama"""
    print("=== Analyzing Triton Autotune Results with LLM ===")
    
    # Initialize analyzer
    analyzer = TritonResultsAnalyzer(results_file)
    
    # Initialize Ollama client
    ollama_client = LLMClient(
        provider="ollama",
        model="codellama:13b",  # or your preferred model
    )
    
    # Analyze kernels
    analysis_results = analyzer.analyze_all_kernels(
        llm_client=ollama_client,
        gpu_model="v100",  # or "a100", "rtx_3090", etc.
        max_kernels=max_kernels
    )
    
    # Generate report
    report = analyzer.generate_analysis_report(analysis_results)
    
    # Print summary with evaluation
    print_summary_with_evaluation(analysis_results, report["llm_performance_evaluation"])
    
    return analysis_results

def print_summary_with_evaluation(analysis_results: List[Dict[str, Any]], 
                                evaluation: Dict[str, Any]):
    """Print a summary of the analysis results with LLM performance evaluation"""
    print(f"\n{'='*60}")
    print("📈 LLM ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    successful = [r for r in analysis_results if r.get("success", False)]
    failed = [r for r in analysis_results if not r.get("success", False)]
    
    print(f"✅ Successful analyses: {len(successful)}")
    print(f"❌ Failed analyses: {len(failed)}")
    
    # Print LLM performance evaluation
    if evaluation["kernels_with_near_opt_data"] > 0:
        print(f"\n🎯 LLM CONFIGURATION QUALITY:")
        print(f"   Kernels with near-optimal data: {evaluation['kernels_with_near_opt_data']}")
        print(f"   LLM recommended near-optimal: {evaluation['llm_recommended_near_optimal']} "
              f"({evaluation.get('near_optimal_success_rate', 0):.1f}%)")
        print(f"   LLM recommended best config: {evaluation['llm_recommended_best']} "
              f"({evaluation.get('best_config_success_rate', 0):.1f}%)")
        print(f"   Performance within threshold: {evaluation['llm_performance_within_threshold']} "
              f"({evaluation.get('threshold_success_rate', 0):.1f}%)")
    
    if successful:
        print(f"\n🚀 Top recommendations:")
        for i, result in enumerate(successful[:5], 1):
            kernel_info = result.get("kernel_info", {})
            llm_analysis = result.get("llm_analysis", {})
            parsed_config = llm_analysis.get("parsed_configuration", {})
            
            print(f"  {i}. {kernel_info.get('kernel_name', 'Unknown')}")
            print(f"     Best latency: {kernel_info.get('latency_ms', 0):.3f}ms")
            print(f"     LLM config: {parsed_config}")
    
    if failed:
        print(f"\n⚠️  Failed analyses:")
        for result in failed[:3]:
            kernel_info = result.get("kernel_info", {})
            error = result.get("error", "Unknown error")
            print(f"  - {kernel_info.get('kernel_name', 'Unknown')}: {error}")

# ... (rest of your existing functions remain the same)

# Example usage
if __name__ == "__main__":
    # Analyze all kernels (limit to 3 for demo)
    r = "Data/triton_autotune_results.json"
    results = demo_analyze_with_ollama(r, max_kernels=30)