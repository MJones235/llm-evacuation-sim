# Function Reference: evacusim/concordia/azure_llm_concordia.py

- Source file: [evacusim/concordia/azure_llm_concordia.py](../../../evacusim/concordia/azure_llm_concordia.py)
- Top-level classes: 1
- Top-level functions: 1

## Module Summary

Azure OpenAI LLM adapter for Concordia.

## Top-Level Functions

### create_concordia_llm_from_config

- Signature: def create_concordia_llm_from_config(config) -> AzureLLMConcordia
- Location: [evacusim/concordia/azure_llm_concordia.py#L376](../../../evacusim/concordia/azure_llm_concordia.py#L376)
- Summary: Create Azure LLM instance from configuration.
- Key calls: load_dotenv, config.get, AzureLLMConcordia.from_env, llm_config.get

## Classes and Methods

### Class AzureLLMConcordia

- Location: [evacusim/concordia/azure_llm_concordia.py#L34](../../../evacusim/concordia/azure_llm_concordia.py#L34)
- Summary: Azure OpenAI LLM adapter for Concordia.
- Method count: 5

#### __init__

- Signature: def __init__(self, endpoint, api_key, model=None, api_version='2024-02-15-preview', temperature=0.7, max_retries=3, max_completion_tokens=8000, timeout=90.0, reasoning_effort=None, response_format='json_object')
- Location: [evacusim/concordia/azure_llm_concordia.py#L47](../../../evacusim/concordia/azure_llm_concordia.py#L47)
- Summary: Initialize Azure OpenAI client for Concordia.
- Key calls: endpoint.rstrip, logger.info, self.endpoint.split, len, split

#### sample_text

- Signature: def sample_text(self, prompt, max_tokens=None, temperature=None, **kwargs) -> str
- Location: [evacusim/concordia/azure_llm_concordia.py#L112](../../../evacusim/concordia/azure_llm_concordia.py#L112)
- Summary: Generate text from a prompt.
- Key calls: range, logger.error, logger.warning, max, requests.post, response.json, strip, choice.get, self._log_prompt_response, result.get

#### get_usage_stats

- Signature: def get_usage_stats(self) -> dict
- Location: [evacusim/concordia/azure_llm_concordia.py#L281](../../../evacusim/concordia/azure_llm_concordia.py#L281)
- Summary: Get cumulative token usage statistics with cost estimates.

#### _log_prompt_response

- Signature: def _log_prompt_response(self, prompt, response, usage, finish_reason, max_completion_tokens) -> None
- Location: [evacusim/concordia/azure_llm_concordia.py#L304](../../../evacusim/concordia/azure_llm_concordia.py#L304)
- Summary: Log the full prompt and response for debugging.
- Key calls: os.getenv, log_path.parent.mkdir, usage.get, Path, llm_current_agent_id.get, llm_current_sim_time.get, log_path.open, f.write, logger.debug, isoformat

#### from_env

- Signature: def from_env(cls, **kwargs) -> 'AzureLLMConcordia'
- Location: [evacusim/concordia/azure_llm_concordia.py#L345](../../../evacusim/concordia/azure_llm_concordia.py#L345)
- Summary: Create instance from environment variables.
- Key calls: os.getenv, cls, ValueError

## Coverage Notes

- Nested helper functions documented in this file: 0
