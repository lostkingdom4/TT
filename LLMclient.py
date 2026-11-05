import requests
import json
import time
from typing import Dict, Any, List, Optional
import logging
import ollama
from make_prompt import OptimizationPromptGenerator, KernelAnalysis, GPUArchitecture
from dotenv import load_dotenv
import os

class LLMClient:
    """Client for interacting with LLM APIs (Ollama and DeepSeek)"""

    def __init__(self, provider: str = "ollama", **kwargs):
        """
        Initialize LLM client

        Args:
            provider: "ollama" or "deepseek"
            **kwargs: API-specific configuration
        """
        self.provider = provider.lower()
        self.setup_logging()

        if self.provider == "ollama":
            self.base_url = kwargs.get('base_url', 'http://localhost:11434')
            self.model = kwargs.get('model', 'llama2')
            self.headers = {'Content-Type': 'application/json'}
        elif self.provider == "deepseek":
            self.api_key = kwargs.get('api_key')
            if not self.api_key:
                raise ValueError("DeepSeek API key required")
            self.base_url = kwargs.get('base_url', 'https://api.deepseek.com/v1')
            self.model = kwargs.get('model', 'deepseek-coder')
            self.headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def generate_response(self, prompt: str, **generation_params) -> Dict[str, Any]:
        """
        Send prompt to LLM and get response

        Args:
            prompt: The prompt to send
            **generation_params: Generation parameters like temperature, max_tokens, etc.

        Returns:
            Dictionary containing response and metadata
        """
        try:
            if self.provider == "ollama":
                return self._call_ollama(prompt, **generation_params)
            elif self.provider == "deepseek":
                return self._call_deepseek(prompt, **generation_params)
        except Exception as e:
            self.logger.error(f"Error calling {self.provider} API: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": None,
                "usage": {}
            }

    def _call_ollama(self, prompt: str, **params) -> Dict[str, Any]:
        """Call Ollama local API"""
        # url = f"{self.base_url}/api/generate"

        default_params = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": params.get('temperature', 0.1),
                "top_p": params.get('top_p', 0.9),
                "max_tokens": params.get('max_tokens', 4000),
                "stop": params.get('stop', [])
            }
        }

        # Merge with provided params
        if 'temperature' in params:
            default_params["options"]["temperature"] = params['temperature']
        if 'top_p' in params:
            default_params["options"]["top_p"] = params['top_p']
        if 'max_tokens' in params:
            default_params["options"]["max_tokens"] = params['max_tokens']

        self.logger.info(f"Sending request to Ollama (model: {self.model})")
        start_time = time.time()

        response = ollama.generate(**default_params)

        elapsed_time = time.time() - start_time
        result = response.model_dump()

        self.logger.info(f"Ollama response received in {elapsed_time:.2f}s")

        return {
            "success": True,
            "response": result.get("response", ""),
            "model": result.get("model", self.model),
            "total_duration": result.get("total_duration", 0),
            "load_duration": result.get("load_duration", 0),
            "prompt_eval_count": result.get("prompt_eval_count", 0),
            "eval_count": result.get("eval_count", 0),
            "elapsed_time": elapsed_time
        }

    def _call_deepseek(self, prompt: str, **params) -> Dict[str, Any]:
        """Call DeepSeek API"""
        url = f"{self.base_url}/chat/completions"

        messages = [
            {
                "role": "system",
                "content": "You are an expert in GPU kernel optimization and high-performance computing. Analyze the given kernel and GPU architecture to provide optimal configuration recommendations."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": params.get('temperature', 0.1),
            "max_tokens": params.get('max_tokens', 4000),
            "top_p": params.get('top_p', 0.9),
            "stream": False
        }

        self.logger.info(f"Sending request to DeepSeek (model: {self.model})")
        start_time = time.time()

        response = requests.post(url, headers=self.headers,
                                data=json.dumps(payload), timeout=120)
        response.raise_for_status()

        elapsed_time = time.time() - start_time
        result = response.json()

        self.logger.info(f"DeepSeek response received in {elapsed_time:.2f}s")

        # Extract usage information
        usage = result.get("usage", {})
        message_content = result["choices"][0]["message"]["content"]

        return {
            "success": True,
            "response": message_content,
            "model": result.get("model", self.model),
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            },
            "elapsed_time": elapsed_time
        }

class OptimizationPipeline:
    """Complete pipeline from kernel analysis to optimization recommendations"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.prompt_generator = OptimizationPromptGenerator()

    def analyze_and_optimize(self, kernel_source: str, gpu_model: str,
                           tuning_parameters: Optional[Dict] = None,
                           **generation_params) -> Dict[str, Any]:
        """
        Complete analysis and optimization pipeline

        Args:
            kernel_source: Triton kernel source code
            gpu_model: Target GPU model
            tuning_parameters: Optional custom tuning parameters
            **generation_params: LLM generation parameters

        Returns:
            Complete analysis results
        """
        # Generate prompt
        prompt = self.prompt_generator.generate_prompt(kernel_source, gpu_model, tuning_parameters)
        print("=== GENERATED PROMPT ===")
        print(prompt)
        exit()

        # Get LLM response
        llm_response = self.llm_client.generate_response(prompt, **generation_params)

        # Parse response
        parsed_config = self._parse_llm_response(llm_response["response"])

        return {
            "prompt": prompt,
            "llm_response": llm_response,
            "parsed_configuration": parsed_config,
            "kernel_info": self.prompt_generator.kernel_analyzer.analyze_kernel_code(kernel_source),
            "gpu_info": self.prompt_generator.gpu_arch.get_architecture(gpu_model)
        }

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract structured configuration

        This is a basic parser - you might want to enhance it based on your LLM's output format
        """
        print(response)
        try:
            if response.strip().startswith('```json'):
                response = response.strip().strip('```json').strip('```').strip()

            response = json.loads(response)

            if "recommended_configuration" in response:
                return response["recommended_configuration"]
            else:
                # Fallback: extract key-value pairs
                return self._extract_key_values(response)
        except Exception as e:
            self.llm_client.logger.warning(f"Could not parse LLM response as JSON: {e}")
            return {"raw_response": response}

    def _extract_key_values(self, response: str) -> Dict[str, Any]:
        """Extract key-value pairs from text response"""
        config = {}
        lines = response.split('\n')

        for line in lines:
            line = line.strip()
            if ':' in line and any(keyword in line.lower() for keyword in
                                 ['xblock', 'warps', 'stages', 'eviction']):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower().replace(' ', '_')
                    value = parts[1].strip()

                    # Try to convert to appropriate type
                    if value.isdigit():
                        value = int(value)
                    elif value.replace('.', '').isdigit():
                        value = float(value)
                    elif value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'

                    config[key] = value

        return config

# Example usage and demonstration
def demo_ollama():
    """Demonstrate with Ollama"""
    print("=== Testing with Ollama ===")

    # Your kernel source
    kernel_code = """
@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4
    x0 = xindex % 4
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + x2, xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tl.store(out_ptr0 + x2, tmp7, xmask)
"""

    # Initialize Ollama client
    ollama_client = LLMClient(
        provider="ollama",
        model="codellama:13b",  # or "llama2", "mistral", etc.
    )

    # Create pipeline
    pipeline = OptimizationPipeline(ollama_client)

    # Run analysis
    result = pipeline.analyze_and_optimize(
        kernel_source=kernel_code,
        gpu_model="v100",
        temperature=0.1,
        max_tokens=2000
    )
    # print(result)

    # Print results
    print("\n=== ANALYSIS RESULTS ===")
    print(f"LLM Response Success: {result['llm_response']['success']}")
    print(f"Model: {result['llm_response']['model']}")
    print(f"Response Time: {result['llm_response']['elapsed_time']:.2f}s")

    print("\n=== PARSED CONFIGURATION ===")
    print(json.dumps(result['parsed_configuration'], indent=2))

    print("\n=== RAW RESPONSE ===")
    print(result['llm_response']['response'])

def demo_deepseek():
    """Demonstrate with DeepSeek API"""
    print("\n=== Testing with DeepSeek ===")

    # You would need to set your API key
    load_dotenv()
    api_key=os.getenv("DEEPSEEK_API_KEY")

    if api_key == "your_deepseek_api_key_here":
        print("Please set your DeepSeek API key to test")
        return

    kernel_code = """
@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16
    # ... same kernel as above
"""

    # Initialize DeepSeek client
    deepseek_client = LLMClient(
        provider="deepseek",
        api_key=api_key,
        model="deepseek-coder"
    )

    # Create pipeline
    pipeline = OptimizationPipeline(deepseek_client)

    # Run analysis
    result = pipeline.analyze_and_optimize(
        kernel_source=kernel_code,
        gpu_model="a100",
        temperature=0.1,
        max_tokens=3000
    )

    # Print results
    print("\n=== ANALYSIS RESULTS ===")
    print(f"LLM Response Success: {result['llm_response']['success']}")
    print(f"Model: {result['llm_response']['model']}")
    print(f"Response Time: {result['llm_response']['elapsed_time']:.2f}s")

    if 'usage' in result['llm_response']:
        usage = result['llm_response']['usage']
        print(f"Tokens Used: {usage.get('total_tokens', 'N/A')}")

    print("\n=== PARSED CONFIGURATION ===")
    print(json.dumps(result['parsed_configuration'], indent=2))

def batch_analysis(kernels: List[Dict], gpu_models: List[str], client: LLMClient):
    """Run batch analysis on multiple kernels"""
    pipeline = OptimizationPipeline(client)
    results = []

    for kernel_info in kernels:
        for gpu_model in gpu_models:
            print(f"Analyzing kernel on {gpu_model}...")

            result = pipeline.analyze_and_optimize(
                kernel_source=kernel_info['source'],
                gpu_model=gpu_model,
                temperature=0.1
            )

            results.append({
                'kernel_name': kernel_info.get('name', 'unknown'),
                'gpu_model': gpu_model,
                'result': result
            })

            # Be nice to the API
            time.sleep(1)

    return results

if __name__ == "__main__":
    # Test with Ollama (local)
    # demo_ollama()

    # Uncomment to test with DeepSeek (requires API key)
    demo_deepseek()

    # Example of batch processing
    # kernels = [
    #     {'name': 'elementwise_fusion', 'source': 'your_kernel_source_here'},
    #     {'name': 'reduction', 'source': 'another_kernel_source'}
    # ]
    #
    # client = LLMClient(provider="ollama", model="codellama:13b")
    # batch_results = batch_analysis(kernels, ['v100', 'a100'], client)