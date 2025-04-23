from langchain.tools import BaseTool
import os
import requests

class PerplexitySearch(BaseTool):
    name: str = "PerplexitySearch"
    description: str = "Searches the web for chemical or scientific information using the Perplexity API. Input should be a natural language question or query."

    def _run(self, query: str) -> str:
        try:
            api_key = os.getenv("PERPLEXITY_API_KEY")
            url = "https://api.perplexity.ai/chat/completions"

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": "llama-3-sonar-small-32k-online",
                "messages": [
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            }

            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return content

        except Exception as e:
            return f"Error during Perplexity search: {str(e)}"
