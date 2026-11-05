"""
LLM Provider for agent communication
Supports OpenAI GPT-4, DeepSeek, and local models via Ollama and vLLM
"""

import os
import json
import re
from typing import Dict, Any, Optional, List
from openai import OpenAI
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


class LLMProvider:
    """Unified LLM provider supporting multiple backends"""

    def __init__(
        self,
        provider: str = "openai",  # "openai", "deepseek", "ollama", "vllm"
        model: Optional[str] = None,
        temperature: float = 0.2,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        debug_mode: bool = False,
        debug_dir: str = "./debug_multiagent",
        run_id: Optional[str] = None,
        use_cached_prompts: bool = False,
        cached_prompts_dir: Optional[str] = None,
    ):
        """
        Initialize LLM provider

        Args:
            provider: LLM provider type ("openai", "deepseek", "ollama", "vllm")
            model: Model name (defaults based on provider)
            temperature: Sampling temperature (0.1-0.3 for technical tasks)
            api_key: API key (if required)
            base_url: Base URL for API (for custom endpoints)
            debug_mode: Enable detailed logging
            debug_dir: Directory to save debug artifacts (should already be run-specific)
            run_id: Run identifier (used for logging, not for directory creation)
            use_cached_prompts: If True, read responses from cached debug files instead of calling the API
            cached_prompts_dir: Path to a debug run directory containing cached response files
        """
        load_dotenv()

        self.provider = provider
        self.temperature = temperature
        self.debug_mode = debug_mode
        self.debug_dir = debug_dir  # Already run-specific from orchestrator
        self.run_id = run_id
        self.call_counter = 0  # Track number of API calls
        self.use_cached_prompts = use_cached_prompts
        self.cached_prompts_dir = cached_prompts_dir

        # Set default models
        if model is None:
            if provider == "openai":
                model = "gpt-4o"
            elif provider == "deepseek":
                model = "deepseek-chat"
            elif provider == "ollama":
                model = os.getenv("OLLAMA_MODEL", "llama3")
            elif provider == "vllm":
                model = os.getenv("VLLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

        self.model = model

        # Initialize client
        if provider == "openai":
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        elif provider == "deepseek":
            self.client = OpenAI(
                api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com"
            )
        elif provider == "ollama":
            self.client = OpenAI(
                api_key="ollama",  # Dummy key for Ollama
                base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            )
        elif provider == "vllm":
            self.client = OpenAI(
                api_key="vllm",  # Dummy key for vLLM
                base_url=base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        if self.use_cached_prompts and not self.cached_prompts_dir:
            raise ValueError("use_cached_prompts=True requires cached_prompts_dir to be set")
        if self.use_cached_prompts:
            logger.info(f"Cache replay mode: reading responses from {self.cached_prompts_dir}")

        logger.info(f"Initialized LLM provider: {provider} with model: {model}")

    def _read_cached_response(self, call_id: int, agent_name: Optional[str] = None) -> Optional[str]:
        """Read a cached response from a previous debug run.

        Tries the .txt response file and extracts the content after the
        ``RESPONSE CONTENT:`` header.

        Returns the raw response content string, or None if not found.
        """
        if not self.cached_prompts_dir:
            return None

        # Build expected filename
        if agent_name:
            filename = f"call{call_id:03d}_{agent_name}_response.txt"
        else:
            filename = f"call{call_id:03d}_response.txt"

        filepath = os.path.join(self.cached_prompts_dir, filename)
        if not os.path.isfile(filepath):
            logger.warning(f"Cached response file not found: {filepath}")
            return None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()

            # Extract content after the "RESPONSE CONTENT:" marker
            marker = "RESPONSE CONTENT:\n" + "-" * 80 + "\n"
            idx = text.find(marker)
            if idx == -1:
                # Fallback: try just the dashes line
                marker_alt = "RESPONSE CONTENT:\n"
                idx = text.find(marker_alt)
                if idx == -1:
                    logger.warning(f"Could not find content marker in {filepath}")
                    return None
                content = text[idx + len(marker_alt):]
                # Skip the dashes line if present
                if content.startswith("-"):
                    nl = content.find("\n")
                    if nl != -1:
                        content = content[nl + 1:]
            else:
                content = text[idx + len(marker):]

            # Strip trailing whitespace
            content = content.rstrip()
            logger.info(f"Loaded cached response from {filepath} ({len(content)} chars)")
            return content
        except Exception as e:
            logger.warning(f"Failed to read cached response {filepath}: {e}")
            return None

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: str = "json",  # "json" or "text"
        max_tokens: Optional[int] = None,
        agent_name: Optional[str] = None,
    ) -> str:
        """
        Generate response from LLM

        Args:
            system_prompt: System instruction
            user_prompt: User query
            response_format: Expected format ("json" or "text")
            max_tokens: Maximum tokens to generate
            agent_name: Name of the agent making the call (for debug output)

        Returns:
            LLM response as string
        """
        try:
            self.call_counter += 1

            # Debug logging
            if self.debug_mode:
                logger.info(f"\n{'='*80}")
                logger.info(f"LLM Call #{self.call_counter} ({self.provider}/{self.model})")
                logger.info(f"System Prompt ({len(system_prompt)} chars):")
                logger.info(f"{system_prompt[:500]}..." if len(system_prompt) > 500 else system_prompt)
                logger.info(f"\nUser Prompt ({len(user_prompt)} chars):")
                logger.info(f"{user_prompt[:500]}..." if len(user_prompt) > 500 else user_prompt)
                logger.info(f"{'='*80}\n")

                # Save full prompts to file
                self._save_debug_prompt(self.call_counter, system_prompt, user_prompt, agent_name)

            # --- Cache replay mode: read from debug files instead of calling API ---
            if self.use_cached_prompts:
                cached = self._read_cached_response(self.call_counter, agent_name)
                if cached is None:
                    raise RuntimeError(
                        f"Cached response not found for call #{self.call_counter} "
                        f"(agent={agent_name}) in {self.cached_prompts_dir}"
                    )
                logger.info(f"Using cached response for call #{self.call_counter} ({agent_name})")
                if response_format == "json":
                    cached = self._extract_json(cached)
                return cached

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Prepare API call parameters
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            }

            if max_tokens:
                kwargs["max_tokens"] = max_tokens

            # For JSON format, add instruction to system prompt
            if response_format == "json":
                messages[0]["content"] += "\n\nIMPORTANT: Respond with valid JSON only. Do not include markdown code blocks or explanatory text."

            # Call API
            response = self.client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content

            # Debug logging of response
            if self.debug_mode:
                logger.info(f"\n{'='*80}")
                logger.info(f"LLM Response #{self.call_counter}")
                logger.info(f"Response ({len(content)} chars):")
                logger.info(f"{content[:500]}..." if len(content) > 500 else content)
                logger.info(f"{'='*80}\n")

                # Save full response to file
                self._save_debug_response(self.call_counter, content, response, agent_name)

            # Clean up response
            if response_format == "json":
                content = self._extract_json(content)

            return content

        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise

    @staticmethod
    def _escape_newlines_in_strings(text: str) -> str:
        """Escape literal newlines/tabs that appear inside JSON string values."""
        result = []
        in_string = False
        escape = False
        for ch in text:
            if escape:
                result.append(ch)
                escape = False
                continue
            if ch == '\\' and in_string:
                result.append(ch)
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                result.append(ch)
                continue
            if in_string:
                if ch == '\n':
                    result.append('\\n')
                    continue
                if ch == '\r':
                    result.append('\\r')
                    continue
                if ch == '\t':
                    result.append('\\t')
                    continue
            result.append(ch)
        return ''.join(result)

    def _repair_json(self, text: str) -> str:
        """Attempt to fix common LLM JSON errors."""
        # Remove trailing commas before } or ]
        text = re.sub(r',\s*([}\]])', r'\1', text)
        # Remove single-line // comments
        text = re.sub(r'//[^\n]*', '', text)
        # Remove C-style /* */ comments
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        # Replace single quotes around keys/values with double quotes
        # (only when they look like JSON key-value patterns)
        text = re.sub(r"(?<=[\[{,:\s])'([^']*?)'(?=\s*[,:\]}])", r'"\1"', text)
        # Remove control characters except \n \r \t
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        # Escape unescaped newlines/tabs inside JSON string values
        text = self._escape_newlines_in_strings(text)

        # Iteratively fix missing delimiters using error position
        for _ in range(20):
            try:
                json.loads(text)
                return text
            except json.JSONDecodeError as e:
                msg = str(e)
                if "Expecting ',' delimiter" in msg:
                    text = text[:e.pos] + ',' + text[e.pos:]
                elif "Expecting ':' delimiter" in msg:
                    text = text[:e.pos] + ':' + text[e.pos:]
                elif "Expecting property name enclosed in double quotes" in msg:
                    # Could be a trailing comma we missed, or unquoted key
                    # Try removing the character at error pos if it's }
                    ch = text[e.pos] if e.pos < len(text) else ''
                    if ch in '}]':
                        # Likely trailing comma before closing bracket — remove comma
                        before = text[:e.pos].rstrip()
                        if before.endswith(','):
                            text = before[:-1] + text[e.pos:]
                        else:
                            break
                    else:
                        break
                elif "Unterminated string" in msg:
                    # Try to close the unterminated string at next newline
                    nl = text.find('\n', e.pos)
                    if nl != -1:
                        text = text[:nl] + '"' + text[nl:]
                    else:
                        break
                elif "Invalid \\escape" in msg:
                    # Remove the bad escape
                    if e.pos < len(text) - 1:
                        text = text[:e.pos] + text[e.pos + 1:]
                    else:
                        break
                else:
                    break

        # Last resort: try to truncate at the deepest valid point
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        # Try closing unclosed brackets/braces
        bracket_stack = []
        in_string = False
        escape = False
        for ch in text:
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in '{[':
                bracket_stack.append('}' if ch == '{' else ']')
            elif ch in '}]':
                if bracket_stack:
                    bracket_stack.pop()

        if bracket_stack:
            # Remove any trailing comma before closing
            stripped = text.rstrip()
            if stripped.endswith(','):
                stripped = stripped[:-1]
            closing = ''.join(reversed(bracket_stack))
            text = stripped + closing

        return text

    def _extract_json(self, text: str) -> str:
        """Extract JSON from potentially noisy LLM response"""
        # Remove markdown code blocks
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        # Validate JSON
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}. Attempting to fix...")
            # Try to extract JSON object
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                json_str = text[start:end]
                try:
                    json.loads(json_str)
                    return json_str
                except json.JSONDecodeError:
                    pass

                # Try common LLM JSON repairs on the extracted block
                repaired = self._repair_json(json_str)
                try:
                    json.loads(repaired)
                    logger.info("JSON repaired successfully (fixed trailing commas/comments)")
                    return repaired
                except json.JSONDecodeError:
                    pass

            # Give up and return as-is
            logger.error("Could not extract valid JSON from response")
            return text

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM with repair fallback."""
        # Direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try extract + repair pipeline
        cleaned = self._extract_json(response)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON after repair: {e}")
            logger.debug(f"Response was: {response[:500]}...")
            raise ValueError(f"Invalid JSON response: {e}")

    def _save_debug_prompt(self, call_id: int, system_prompt: str, user_prompt: str, agent_name: Optional[str] = None):
        """Save prompts to debug directory"""
        import os

        # Use simple filename without timestamp since directory is already timestamped
        if agent_name:
            filename = f"call{call_id:03d}_{agent_name}_prompt.txt"
        else:
            filename = f"call{call_id:03d}_prompt.txt"
        filepath = os.path.join(self.debug_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"{'='*80}\n")
                f.write(f"LLM Call #{call_id}\n")
                if agent_name:
                    f.write(f"Agent: {agent_name}\n")
                f.write(f"Provider: {self.provider}\n")
                f.write(f"Model: {self.model}\n")
                f.write(f"Temperature: {self.temperature}\n")
                if self.run_id:
                    f.write(f"Run ID: {self.run_id}\n")
                f.write(f"{'='*80}\n\n")

                f.write(f"SYSTEM PROMPT:\n")
                f.write(f"{'-'*80}\n")
                f.write(f"{system_prompt}\n\n")

                f.write(f"USER PROMPT:\n")
                f.write(f"{'-'*80}\n")
                f.write(f"{user_prompt}\n")

            logger.debug(f"Saved prompt to: {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save debug prompt: {e}")

    def _save_debug_response(self, call_id: int, content: str, response_obj, agent_name: Optional[str] = None):
        """Save response to debug directory"""
        import os

        # Use simple filename without timestamp since directory is already timestamped
        if agent_name:
            filename = f"call{call_id:03d}_{agent_name}_response.txt"
        else:
            filename = f"call{call_id:03d}_response.txt"
        filepath = os.path.join(self.debug_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"{'='*80}\n")
                f.write(f"LLM Response #{call_id}\n")
                if agent_name:
                    f.write(f"Agent: {agent_name}\n")
                f.write(f"Provider: {self.provider}\n")
                f.write(f"Model: {self.model}\n")
                if self.run_id:
                    f.write(f"Run ID: {self.run_id}\n")

                # Add token usage if available
                if hasattr(response_obj, 'usage'):
                    usage = response_obj.usage
                    f.write(f"Tokens - Prompt: {usage.prompt_tokens}, "
                           f"Completion: {usage.completion_tokens}, "
                           f"Total: {usage.total_tokens}\n")

                f.write(f"{'='*80}\n\n")

                f.write(f"RESPONSE CONTENT:\n")
                f.write(f"{'-'*80}\n")
                f.write(f"{content}\n")

            # Also save parsed JSON if possible
            try:
                parsed = json.loads(content)
                if agent_name:
                    json_filename = f"call{call_id:03d}_{agent_name}_response.json"
                else:
                    json_filename = f"call{call_id:03d}_response.json"
                json_filepath = os.path.join(self.debug_dir, json_filename)
                with open(json_filepath, 'w', encoding='utf-8') as f:
                    json.dump(parsed, f, indent=2)
                logger.debug(f"Saved response JSON to: {json_filepath}")
            except:
                pass

            logger.debug(f"Saved response to: {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save debug response: {e}")
