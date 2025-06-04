import os
import openai
from typing import Dict, Any, List, Optional

class DisconnectionSuggester:
    _openai_client: Optional[openai.OpenAI]

    def __init__(self):
        # Initialize OpenAI client (relies on OPENAI_API_KEY env var)
        try:
            if not os.environ.get("OPENAI_API_KEY"):
                print("Warning [DisconnectionSuggester]: OPENAI_API_KEY not set. Disconnections will not use LLM.")
                self._openai_client = None
            else:
                self._openai_client = openai.OpenAI()
                print("[DisconnectionSuggester] OpenAI client initialized successfully.")
        except Exception as e:
            print(f"Warning [DisconnectionSuggester]: Could not initialize OpenAI client: {e}")
            self._openai_client = None

    def _get_disconnections_from_llm(self, smiles: str, functional_groups: List[str]) -> str:
        """
        Uses an LLM to suggest disconnections based on SMILES and functional groups.
        """
        if not self._openai_client:
            return "LLM client not available for suggesting disconnections."
        if not functional_groups:
            return f"No functional groups identified for {smiles} to base disconnections on. Cannot proceed with LLM suggestion."

        fg_string = ", ".join(functional_groups)
        
        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert organic chemist specializing in retrosynthesis. "
                    "Your task is to suggest logical retrosynthetic disconnections for a given molecule based on its identified functional groups. "
                    "For each major functional group present, suggest one or two common and strategic C-X or C-C disconnections that are typically taught in undergraduate or early graduate organic chemistry. "
                    "Clearly state the bond being disconnected (e.g., C-O in an alcohol, C-N in an amide, C=O C-alpha bond for ketones). "
                    "Show the synthon or fragments that would result from the disconnection. "
                    "Then, briefly suggest a common type of forward reaction that could form that bond (e.g., SN2, Grignard addition, Wittig, aldol condensation, esterification). "
                    "Focus on 1-2 key disconnections per relevant functional group. Be concise and clear. If a functional group doesn't offer a standard, simple disconnection (e.g., a simple alkane part), you can state that or focus on more reactive groups."
                    "Structure your output clearly, perhaps using bullet points or numbered lists for each disconnection type related to a functional group."
                )
            },
            {
                "role": "user",
                "content": (
                    f"For the molecule with SMILES: {smiles}\n"
                    f"Identified functional groups are: {fg_string}\n\n"
                    "Please suggest key retrosynthetic disconnections based on these functional groups."
                )
            }
        ]

        try:
            response = self._openai_client.chat.completions.create(
                model="gpt-4o", # Or your preferred model
                messages=prompt_messages,
                temperature=0.5, # Allow for some creativity but still grounded
                max_tokens=1000, # Allow for a reasonably detailed response
                n=1,
                stop=None,
            )
            disconnection_suggestions = response.choices[0].message.content.strip()
            return disconnection_suggestions
        except openai.APIError as api_e:
            print(f"[DisconnectionSuggester] OpenAI API error: {api_e}")
            return f"OpenAI API error during disconnection suggestion: {str(api_e)[:100]}..."
        except Exception as e:
            print(f"[DisconnectionSuggester] LLM error: {e}")
            return f"Error suggesting disconnections via LLM: {str(e)[:100]}..."

    def _run(self, smiles: str, functional_groups: List[str]) -> Dict[str, Any]:
        """
        Main method to get disconnection suggestions.
        This method expects functional_groups to be pre-identified and passed in.
        """
        print(f"[DisconnectionSuggester Class] _run called for SMILES: {smiles} with FGs: {functional_groups}")
        if not smiles:
            return {"error": "Input SMILES string is required."}
        
        # functional_groups are now passed as an argument
        # If not passed or empty, the LLM call will handle it or we can add a check.
        if not functional_groups:
             return {
                "smiles": smiles,
                "functional_groups_identified": [],
                "disconnections": "No functional groups provided to base disconnections on."
            }


        llm_suggestions = self._get_disconnections_from_llm(smiles, functional_groups)

        return {
            "smiles": smiles,
            "functional_groups_identified": functional_groups,
            "disconnection_suggestions": llm_suggestions
        }