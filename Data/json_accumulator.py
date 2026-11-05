import json
import os
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

class AutotuneResultsStore:
    def __init__(self, results_file: str = "autotune_results.json"):
        self.results_file = Path(results_file)
        self._create_new_file()  # Always create new file on init
    
    def _create_new_file(self):
        """Remove existing file and create a new empty one"""
        # Remove file if it exists
        if self.results_file.exists():
            self.results_file.unlink()
            print(f"🗑️  Removed existing file: {self.results_file}")
        
        # Create directory if it doesn't exist
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create new file with initial structure
        initial_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
                "description": "Triton kernel autotune results"
            },
            "results": []
        }
        
        with open(self.results_file, 'w') as f:
            json.dump(initial_data, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Created new results file: {self.results_file}")
    
    def _read_data(self) -> Dict[str, Any]:
        """Read current results from JSON file"""
        try:
            with open(self.results_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # If file is corrupted or doesn't exist, create new one
            print("⚠️  Results file corrupted or missing, creating new one...")
            self._create_new_file()
            return self._read_data()
    
    def _write_data(self, data: Dict[str, Any]):
        """Write data to JSON file"""
        with open(self.results_file, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def add_results(self, mod_name: str, file_path: str, results: List[Dict[str, Any]]):
        """Add new autotune results to the store"""
        current_data = self._read_data()
        
        # Create result entry with metadata
        result_entry = {
            "module": mod_name,
            "file": file_path,
            "timestamp": datetime.now().isoformat(),
            "kernel_count": len(results),
            "kernels": results
        }
        
        # Add to results list
        current_data["results"].append(result_entry)
        
        # Update metadata
        current_data["metadata"]["last_updated"] = datetime.now().isoformat()
        current_data["metadata"]["total_runs"] = len(current_data["results"])
        current_data["metadata"]["total_kernels"] = sum(
            len(run["kernels"]) for run in current_data["results"]
        )
        
        # Calculate additional statistics
        all_kernels = []
        for run in current_data["results"]:
            all_kernels.extend(run["kernels"])
        
        if all_kernels:
            current_data["metadata"]["avg_attempts_per_kernel"] = sum(
                k.get("attempts", 0) for k in all_kernels
            ) / len(all_kernels)
            
            current_data["metadata"]["kernels_with_near_optimal"] = sum(
                1 for k in all_kernels if k.get("near_opt_count", 0) > 0
            )
        
        self._write_data(current_data)
        print(f"✅ Added {len(results)} kernel results from {file_path}")
    
    def clear_results(self):
        """Clear all results by creating a new file"""
        print("🧹 Clearing all results...")
        self._create_new_file()
    
    def get_all_results(self) -> Dict[str, Any]:
        """Get all stored results"""
        return self._read_data()
    
    def get_kernel_results(self, kernel_name: str) -> List[Dict[str, Any]]:
        """Get all results for a specific kernel across all runs"""
        data = self._read_data()
        kernel_results = []
        
        for run in data.get("results", []):
            for kernel in run.get("kernels", []):
                if kernel.get("kernel") == kernel_name:
                    kernel_results.append({
                        "file": run["file"],
                        "timestamp": run["timestamp"],
                        **kernel
                    })
        
        return kernel_results
    
    def get_fastest_configs(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get the top K fastest kernel configurations"""
        data = self._read_data()
        all_kernels = []
        
        for run in data.get("results", []):
            for kernel in run.get("kernels", []):
                if "best" in kernel and "lat_ms" in kernel["best"]:
                    all_kernels.append(kernel)
        
        # Sort by latency (ascending - lower is better)
        return sorted(all_kernels, key=lambda x: x["best"]["lat_ms"])[:top_k]
    
    def get_kernels_with_multiple_options(self, min_options: int = 3) -> List[Dict[str, Any]]:
        """Get kernels that have multiple near-optimal configurations"""
        data = self._read_data()
        multi_option_kernels = []
        
        for run in data.get("results", []):
            for kernel in run.get("kernels", []):
                if kernel.get("near_opt_count", 0) >= min_options:
                    multi_option_kernels.append({
                        "file": run["file"],
                        "timestamp": run["timestamp"],
                        **kernel
                    })
        
        return multi_option_kernels
    
    def export_summary_csv(self, csv_path: str = "autotune_summary.csv"):
        """Export a summary of results to CSV"""
        import csv
        
        data = self._read_data()
        csv_file = Path(csv_path)
        
        # Remove existing CSV if it exists
        if csv_file.exists():
            csv_file.unlink()
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow([
                "File", "Kernel", "XNumel", "Best_Latency_ms", 
                "Attempts", "Errors", "Near_Opt_Count", "Threshold_Pct",
                "Num_Warps", "Num_Stages", "XBLOCK"
            ])
            
            # Write data
            for run in data.get("results", []):
                for kernel in run.get("kernels", []):
                    best = kernel.get("best", {})
                    writer.writerow([
                        run["file"],
                        kernel.get("kernel", ""),
                        kernel.get("xnumel", 0),
                        best.get("lat_ms", 0),
                        kernel.get("attempts", 0),
                        kernel.get("errors", 0),
                        kernel.get("near_opt_count", 0),
                        kernel.get("near_opt_threshold_pct", 0),
                        best.get("num_warps", ""),
                        best.get("num_stages", ""),
                        best.get("XBLOCK", "")
                    ])
        
        print(f"📊 Exported summary to {csv_path}")
