# cli.py
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='GPU Kernel Optimizer')
    parser.add_argument('--kernel-file', type=str, required=True, 
                       help='Path to kernel file')
    parser.add_argument('--gpu', type=str, required=True, 
                       help='GPU model (v100, a100, etc.)')
    parser.add_argument('--provider', type=str, default='ollama',
                       choices=['ollama', 'deepseek'], 
                       help='LLM provider')
    parser.add_argument('--model', type=str, 
                       help='Model name (default: provider-specific)')
    
    args = parser.parse_args()
    
    # Read kernel file
    kernel_source = Path(args.kernel_file).read_text()
    
    # Setup client
    client_config = {'provider': args.provider}
    if args.model:
        client_config['model'] = args.model
    
    client = LLMClient(**client_config)
    pipeline = OptimizationPipeline(client)
    
    # Run analysis
    result = pipeline.analyze_and_optimize(
        kernel_source=kernel_source,
        gpu_model=args.gpu
    )
    
    # Output results
    if result['llm_response']['success']:
        print("Optimization successful!")
        print(json.dumps(result['parsed_configuration'], indent=2))
    else:
        print("Optimization failed:", result['llm_response']['error'])
        sys.exit(1)

if __name__ == "__main__":
    main()