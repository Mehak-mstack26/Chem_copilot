import requests
import json
from langchain.tools import Tool

class RetroSynthesis:
    """Tool for querying the retrosynthesis API."""
    
    def __init__(self):
        self.api_url = "http://localhost:8000/retro-synthesis/"
    
    def _run(self, query: str) -> str:
        """Execute the retrosynthesis query.
        
        Args:
            query: The compound name to search for retrosynthesis pathways
            
        Returns:
            A string representation of the retrosynthesis results
        """
        try:
            # Format the request according to the API's requirements
            payload = {
                "material": query.strip(),
                "num_results": 10,
                "alignment": True,
                "expansion": True,
                "filtration": False
            }
            
            response = requests.post(
                self.api_url,
                json=payload
            )
            
            if response.status_code != 200:
                return f"Error: Received status code {response.status_code} from the API. Response: {response.text}"
            
            result = response.json()
            
            if result.get("status") != "success":
                return f"API Error: {result.get('message', 'Unknown error')}"
            
            # Format the response in a readable way
            data = result["data"]
            output = []
            
            output.append(f"Retrosynthesis results for: {query}")
            output.append("\n## Recommended Path")
            output.append(data["reasoning"])
            
            output.append("\n## Reactions")
            for i, reaction in enumerate(data["reactions"]):
                is_recommended = "✓" if reaction["idx"] in data["recommended_indices"] else " "
                output.append(f"\n### Reaction {i+1} {is_recommended}")
                
                reactants = " + ".join(reaction["reactants"])
                products = " + ".join(reaction["products"])
                output.append(f"- {reactants} → {products}")
                output.append(f"- Conditions: {reaction['conditions']}")
                output.append(f"- Source: {reaction['source']}")
                
                if "smiles" in reaction:
                    output.append(f"- SMILES: {reaction['smiles']}")
            
            return "\n".join(output)
        
        except Exception as e:
            return f"Error occurred while querying retrosynthesis API: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Async version of the tool."""
        # For simplicity, we use the sync version
        return self._run(query)
    
