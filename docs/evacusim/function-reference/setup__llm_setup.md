# Function Reference: evacusim/setup/llm_setup.py

- Source file: [evacusim/setup/llm_setup.py](../../../evacusim/setup/llm_setup.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Language model and embedder setup for Station Concordia simulations.

## Classes and Methods

### Class LLMSetup

- Location: [evacusim/setup/llm_setup.py#L20](../../../evacusim/setup/llm_setup.py#L20)
- Summary: Handles language model and embedder initialization.
- Method count: 1

#### setup_language_model

- Signature: def setup_language_model(config) -> tuple[object, Callable]
- Location: [evacusim/setup/llm_setup.py#L24](../../../evacusim/setup/llm_setup.py#L24)
- Summary: Setup the language model and embedder.
- Key calls: load_dotenv, os.getenv, logger.info, ValueError, config.get, AzureLLMConcordia, llm_config.get, sentence_transformers.SentenceTransformer, st_model.encode, logger.error

## Coverage Notes

- Nested helper functions documented in this file: 0
