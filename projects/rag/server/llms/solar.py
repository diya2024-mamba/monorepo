import os
from typing import Any, Dict, List, Optional

import requests
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT = "3pnhsvccplyz39"


class Solar(LLM):

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.

        Override this method to implement the LLM logic.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The model output as a string. Actual completions SHOULD NOT include the prompt.
        """
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        response = requests.post(
            f"https://api.runpod.ai/v2/{ENDPOINT}/runsync",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "input": {
                    "method_name": "generate",
                    "input": {
                        "prompt": prompt,
                    },
                },
            },
        )
        data = response.json()
        output = data.get("output", {}).get("response")
        if output is None:
            raise ValueError("Failed to generate output")
        return output

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            "model_name": "Solar",
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "solar"
