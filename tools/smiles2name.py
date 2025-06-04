import os 
from typing import Optional
import requests
import urllib.parse
import openai 

class SMILES2Name:
    _openai_client: Optional[openai.OpenAI]

    def __init__(self):
        print(f"[SMILES2Name __init__] Checking for OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY') is not None}")
        try:
            self._openai_client = openai.OpenAI()
            print("[SMILES2Name Class] OpenAI client initialized successfully.")
        except openai.APIError as e: 
            print(f"Warning: Could not initialize OpenAI client for SMILES2Name due to APIError: {e}. Common name feature might fail.")
            self._openai_client = None
        except Exception as e:
            print(f"Warning: Could not initialize OpenAI client for SMILES2Name (generic exception): {e}. Common name feature might fail.")
            self._openai_client = None

    def _get_common_name_from_llm(self, iupac_name: str, smiles: str) -> str:
        """Gets common name using direct OpenAI API call."""
        if not self._openai_client:
            return "LLM client for common name not available or failed to initialize."

        try:
            prompt_messages = [
                {"role": "system", "content": "You are a helpful chemistry assistant. Your task is to provide the common name for a given chemical compound. If no widely recognized common name exists, state 'No widely recognized common name'. Respond concisely with only the common name or the specified phrase."},
                {"role": "user", "content": f"Given the chemical compound with IUPAC name: '{iupac_name}' (SMILES: {smiles}), what is its widely recognized common name, if any?"}
            ]

            response = self._openai_client.chat.completions.create(
                model="gpt-4o",
                messages=prompt_messages,
                temperature=0,
                max_tokens=60,
                n=1,
                stop=None,
            )
            
            llm_output_content = response.choices[0].message.content.strip()

            if "no widely recognized common name" in llm_output_content.lower() or \
               iupac_name.lower() in llm_output_content.lower().replace('-', ' '):
                return "No widely recognized common name found."
            elif len(llm_output_content) > 0 and len(llm_output_content) < 80:
                return llm_output_content
            else:
                return "Could not be definitively determined by LLM."

        except openai.APIError as api_e: 
            print(f"[SMILES2Name Class] OpenAI API error during common name lookup: {api_e}")
            if hasattr(api_e, 'status_code') and api_e.status_code == 401:
                return "OpenAI API authentication error. Please check your API key."
            return f"OpenAI API error ({str(api_e)[:50]}...)."
        except Exception as llm_e:
            print(f"[SMILES2Name Class] LLM common name lookup error: {llm_e}")
            return f"Error during LLM lookup ({str(llm_e)[:50]}...)."

    def _run(self, smiles: str) -> str:
        print(f"[SMILES2Name Class] _run called with: {smiles}")
        try:
            iupac_name = self._try_cactus(smiles)
            if not iupac_name:
                print(f"[SMILES2Name Class] CACTUS failed for {smiles}, trying PubChem.")
                iupac_name = self._try_pubchem(smiles)

            if iupac_name:
                common_name_str = self._get_common_name_from_llm(iupac_name, smiles)
                if "error" in common_name_str.lower() or \
                   "failed" in common_name_str.lower() or \
                   "not available" in common_name_str.lower() or \
                   "not initialized" in common_name_str.lower():
                    common_name_part = f"Common name: {common_name_str}"
                elif "no widely recognized common name" in common_name_str.lower():
                     common_name_part = f"Common name: {common_name_str}"
                else:
                    common_name_part = f"Common name: {common_name_str}"

                return f"IUPAC name: {iupac_name}\n{common_name_part}"
            else:
                return f"Could not resolve IUPAC name for SMILES: '{smiles}' using CACTUS or PubChem."
        except Exception as e:
            return f"Exception in SMILES2Name _run for '{smiles}': {str(e)}"

    def _try_cactus(self, smiles: str) -> Optional[str]:
        encoded_smiles = urllib.parse.quote(smiles)
        url = f"https://cactus.nci.nih.gov/chemical/structure/{encoded_smiles}/iupac_name"
        try:
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            name_text = res.text.strip()
            if name_text and "Page not found" not in name_text and "<html" not in name_text.lower():
                return name_text
            else:
                print(f"[SMILES2Name Class] CACTUS returned empty or error page for {smiles}: {name_text[:100]}")
        except requests.exceptions.RequestException as e:
            print(f"[SMILES2Name Class] CACTUS request error for {smiles}: {e}")
        except Exception as e:
            print(f"[SMILES2Name Class] CACTUS unexpected error for {smiles}: {e}")
        return None

    def _try_pubchem(self, smiles: str) -> Optional[str]:
        encoded_smiles = urllib.parse.quote(smiles, safe="")
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded_smiles}/property/IUPACName/JSON"
        try:
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            data = res.json()
            if data and 'PropertyTable' in data and 'Properties' in data['PropertyTable'] and \
               len(data['PropertyTable']['Properties']) > 0 and 'IUPACName' in data['PropertyTable']['Properties'][0]:
                return data['PropertyTable']['Properties'][0]['IUPACName']
            else:
                print(f"[SMILES2Name Class] PubChem did not return IUPACName for {smiles}. Response: {data}")
        except requests.exceptions.RequestException as e:
            print(f"[SMILES2Name Class] PubChem request error for {smiles}: {e}")
        except Exception as e:
            print(f"[SMILES2Name Class] PubChem unexpected error for {smiles}: {e}")
        return None