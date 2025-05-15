from langchain.tools import BaseTool
from typing import Optional
import requests
import urllib.parse
from langchain_openai import ChatOpenAI
from api_config import api_key

class SMILES2Name(BaseTool):
    name: str = "SMILES2Name"
    description: str = "Converts SMILES to IUPAC name using CACTUS and finds common name via OpenAI if possible."

    def _run(self, smiles: str) -> str:
        try:
            iupac = self._try_cactus(smiles)
            if not iupac:
                iupac = self._try_pubchem(smiles)

            if iupac:
                llm = ChatOpenAI(model="gpt-4o", temperature=0)
                prompt = f"What is the common name for the following IUPAC compound: {iupac}? If there's no common name, say so clearly."
                response = llm.invoke(prompt)
                content = response.content.strip()

                # Check if GPT explicitly says there's no common name
                if "no common name" in content.lower() or "does not have" in content.lower():
                    return f"IUPAC name: {iupac} (No common name available)"
                else:
                    return f"Common name: {content}\nIUPAC name: {iupac}"
            else:
                return f"Could not resolve IUPAC name for: {smiles} using either CACTUS or PubChem."
        except Exception as e:
            return f"Exception: {str(e)}"

    def _try_cactus(self, smiles: str) -> Optional[str]:
        encoded = urllib.parse.quote(smiles) 
        url = f"https://cactus.nci.nih.gov/chemical/structure/{encoded}/iupac_name"
        try:
            res = requests.get(url, timeout=10)
            text = res.text.strip()
            if res.status_code == 200 and "Page not found" not in text:
                return text
        except Exception as e:
            print(f"CACTUS error: {e}")
        return None

    def _try_pubchem(self, smiles: str) -> Optional[str]:
        # PubChem requires full encoding
        encoded = urllib.parse.quote(smiles, safe="")
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded}/property/IUPACName/JSON"
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                return data['PropertyTable']['Properties'][0]['IUPACName']
        except Exception as e:
            print(f"PubChem error: {e}")
        return None