import requests
import json
from langchain_openai import ChatOpenAI

class ReactionClassifier:
    """Tool to classify reaction types based on reaction SMILES and provide detailed information."""
    
    def __init__(self):
        self.api_url = "http://13.201.135.9:9621/reaction_class"
        # Initialize OpenAI for detailed information
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
    def _run(self, reaction_smiles):
        """Run the ReactionClassifier tool.
        
        Args:
            reaction_smiles: A reaction SMILES string
            
        Returns:
            A string with the classified reaction type and educational information
        """
        try:
            # Format the request payload
            payload = {"smiles": [reaction_smiles]}
            
            # Make the API request
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json=payload
            )
            
            # Check if request was successful
            if response.status_code == 200:
                data = response.json()
                
                # Check if we have results
                if data.get("status") == "SUCCESS" and data.get("results") and len(data["results"]) > 0:
                    # Get only the top-ranked reaction
                    top_reaction = data["results"][0]
                    reaction_name = top_reaction.get("reaction_name", "Unknown")
                    reaction_class = top_reaction.get("reaction_classname", "Unknown")
                    reaction_num = top_reaction.get("reaction_num", "Unknown")
                    prediction_certainty = top_reaction.get("prediction_certainty", 0) * 100
                    
                    result = f"## Reaction Classification\n"
                    result += f"- **Type**: {reaction_name} (Reaction #{reaction_num})\n"
                    result += f"- **Class**: {reaction_class}\n"
                    result += f"- **Certainty**: {prediction_certainty:.2f}%\n\n"
                    
                    # Get detailed information about ONLY the top reaction using OpenAI
                    reaction_details = self._get_reaction_info(reaction_name)
                    result += f"## Detailed Information\n{reaction_details}\n"
                    
                    return result
                else:
                    return "No reaction classification results returned by the API."
            else:
                return f"API request failed with status code: {response.status_code}. Response: {response.text}"
        
        except Exception as e:
            return f"Error classifying reaction: {str(e)}"
    
    def _get_reaction_info(self, reaction_name):
        """Get detailed information about a reaction type using OpenAI.
        
        Args:
            reaction_name: The name of the reaction
            
        Returns:
            A string with detailed information about the reaction
        """
        try:
            prompt = f"""Provide comprehensive and detailed information about the '{reaction_name}' reaction in organic chemistry.
            

"""
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            return f"Could not retrieve detailed information: {str(e)}"
        


# Include the following details:
# 1. A description of what the reaction does and its importance
# 2. Typical reagents and conditions used
# 3. Complete mechanism
# 4. Common applications in synthesis
# 5. Any limitations or considerations

# Please give a complete and readable explanation of this reaction.
# Make it informative and complete, suitable for a chemist.