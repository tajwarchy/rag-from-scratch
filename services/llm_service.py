"""
LLM service — single abstraction layer over the language model backend.

The rest of the codebase calls only generate(prompt) and knows nothing
about which LLM is running underneath. Swapping backends is a one-function change.

Current backend: Ollama (Mistral 7B running locally, no API key required).
"""

import httpx
from core.config_loader import load_config


def generate(prompt: str) -> str:
    """
    Send a prompt to the configured LLM and return the response text.

    This is the ONLY function that knows about the LLM backend.
    query_service.py calls this — it never imports httpx or openai directly.

    ── To swap Ollama for OpenAI, replace this entire function with: ──────────

        import openai
        client = openai.OpenAI()  # reads OPENAI_API_KEY from environment

        def generate(prompt: str) -> str:
            cfg = load_config()
            response = client.chat.completions.create(
                model=cfg["llm"]["model_name"],   # e.g. "gpt-4o"
                messages=[{"role": "user", "content": prompt}],
                temperature=cfg["llm"]["temperature"],
                max_tokens=cfg["llm"]["max_tokens"],
            )
            return response.choices[0].message.content.strip()

    The abstraction makes this a one-function swap — zero changes elsewhere.
    ─────────────────────────────────────────────────────────────────────────────
    """
    cfg = load_config()
    llm_cfg = cfg["llm"]

    url = f"{llm_cfg['base_url']}/api/generate"
    payload = {
        "model":  llm_cfg["model_name"],
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": llm_cfg["temperature"],
            "num_predict": llm_cfg["max_tokens"],
        },
    }

    try:
        response = httpx.post(url, json=payload, timeout=120.0)
        response.raise_for_status()
        return response.json()["response"].strip()
    except httpx.ConnectError:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure it is running: `ollama serve`"
        )
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Ollama returned an error: {e.response.text}")