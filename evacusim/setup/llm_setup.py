"""
Language model and embedder setup for Station Concordia simulations.

This module is responsible for:
- Initializing Azure LLM for Concordia
- Setting up sentence embedders
- Loading environment variables for LLM credentials
"""

import os
from collections.abc import Callable

from dotenv import load_dotenv

from evacusim.utils.logger import get_logger

logger = get_logger(__name__)


class LLMSetup:
    """Handles language model and embedder initialization."""

    @staticmethod
    def setup_language_model(config: dict) -> tuple[object, Callable]:
        """
        Setup the language model and embedder.

        Args:
            config: Configuration dictionary containing LLM settings

        Returns:
            Tuple of (model, embedder_function)

        Raises:
            ValueError: If no LLM credentials are configured
            ImportError: If required dependencies are not installed
        """
        # Load .env file
        load_dotenv()

        azure_endpoint = os.getenv("AZURE_LLM_ENDPOINT")
        azure_key = os.getenv("AZURE_LLM_API_KEY")
        azure_model = os.getenv("AZURE_LLM_MODEL")

        if not azure_endpoint or not azure_key:
            raise ValueError(
                "No LLM configured. Set Azure credentials in .env "
                "(AZURE_LLM_ENDPOINT, AZURE_LLM_API_KEY)"
            )

        logger.info(f"Using Azure LLM: {azure_model or 'serverless'}")

        try:
            import sentence_transformers

            from evacusim.concordia.azure_llm_concordia import (
                AzureLLMConcordia,
            )

            # Create Azure LLM client designed for Concordia
            # Uses synchronous REST API calls to avoid async/sync conflicts
            llm_config = config.get("llm", {})
            model = AzureLLMConcordia(
                endpoint=azure_endpoint,
                api_key=azure_key,
                model=azure_model,
                temperature=llm_config.get("temperature", 0.7),
                max_retries=llm_config.get("max_retries", 3),
                max_completion_tokens=llm_config.get("max_completion_tokens", 8000),
                timeout=llm_config.get("timeout", 90.0),
            )

            # Setup embedder (force CPU to avoid GPU compatibility issues)
            embedder_name = llm_config.get("embedder", "sentence-transformers/all-mpnet-base-v2")
            logger.info(f"Loading embedder: {embedder_name}...")
            st_model = sentence_transformers.SentenceTransformer(embedder_name, device="cpu")

            def embedder(x):
                return st_model.encode(x, show_progress_bar=False, device="cpu")

            logger.info("Embedder loaded successfully")
            logger.info("Azure LLM for Concordia initialized successfully")
            return model, embedder

        except ImportError as e:
            logger.error(f"Failed to import Azure provider: {e}")
            raise
