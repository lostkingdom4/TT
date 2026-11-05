# config.py
import os

# API Configuration
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'your_api_key_here')

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'codellama:13b')

# Generation Parameters
DEFAULT_GENERATION_PARAMS = {
    'temperature': 0.1,
    'max_tokens': 4000,
    'top_p': 0.9
}