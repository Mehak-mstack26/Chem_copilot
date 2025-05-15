from langchain.tools import BaseTool
import requests
import re
import urllib.parse

class NameToSMILES(BaseTool):
    name: str = "NameToSMILES"
    description: str = "Convert a compound, molecule, or reaction **name** (e.g., 'aspirin', 'glucose', or 'Friedel–Crafts acylation') to its SMILES representation "
    "using the CAS Common Chemistry API with a fallback to PubChem. "
    "**This tool does NOT accept SMILES as input.** Use only when the input is a chemical name."

    def _run(self, query: str) -> str:
        try:
            if re.search(r"[=#@\\/[\]]", query) or len(query) > 100:
                return f"Error: This looks like a SMILES string, not a name. Please use SMILES2Name instead."
            # First try CAS Common Chemistry
            cas_result = self._try_cas_common_chemistry(query)
            
            # If CAS has SMILES, return it
            if cas_result.startswith("SMILES:"):
                smiles = cas_result.split("SMILES:")[1].split("\n")[0].strip()
                return f"SMILES: {smiles}\nSource: CAS Common Chemistry"
                
            # Otherwise try PubChem as fallback
            pubchem_result = self._try_pubchem(query)
            if pubchem_result.startswith("SMILES:"):
                smiles = pubchem_result.split("SMILES:")[1].split("\n")[0].strip()
                return f"SMILES: {smiles}\nSource: PubChem"
            
            # If both fail, return error message
            return f"No SMILES found for '{query}'"
            
        except Exception as e:
            return f"Exception occurred: {str(e)}"
    
    def _try_cas_common_chemistry(self, query: str) -> str:
        try:
            # Step 1: Search by name
            search_url = f"https://commonchemistry.cas.org/api/search?q={query}"
            search_resp = requests.get(search_url)
            search_resp.raise_for_status()
            
            results = search_resp.json()
            
            if not results or "results" not in results or not results["results"]:
                return f"No results found for '{query}' in CAS Common Chemistry."
            
            cas_rn = results["results"][0].get("rn")
            if not cas_rn:
                return f"CAS RN not found for '{query}'"
            
            # Step 2: Get SMILES from detail endpoint
            detail_url = f"https://commonchemistry.cas.org/api/detail?cas_rn={cas_rn}"
            detail_resp = requests.get(detail_url)
            detail_resp.raise_for_status()
            
            details = detail_resp.json()
            
            # Check for smile and canonicalSmile (singular, not plural)
            smiles = None
            if "smile" in details and details["smile"]:
                smiles = details["smile"]
            elif "canonicalSmile" in details and details["canonicalSmile"]:
                smiles = details["canonicalSmile"]
            
            if smiles:
                return f"SMILES: {smiles}"
            else:
                return f"No SMILES available in CAS Common Chemistry."
                
        except Exception as e:
            return f"CAS Common Chemistry error: {str(e)}"
    
    def _try_pubchem(self, query: str) -> str:
        try:
            # URL encode the query for safety
            encoded_query = urllib.parse.quote(query)
            
            # First try the simpler direct property endpoint
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_query}/property/IsomericSMILES,CanonicalSMILES/JSON"
            response = requests.get(url)
            
            # If that fails, try the search -> property approach
            if response.status_code != 200:
                # Search for the compound to get CID
                search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_query}/cids/JSON"
                search_resp = requests.get(search_url)
                search_resp.raise_for_status()
                
                search_data = search_resp.json()
                
                if "IdentifierList" not in search_data or "CID" not in search_data["IdentifierList"]:
                    return f"No results found for '{query}' in PubChem."
                    
                cid = search_data["IdentifierList"]["CID"][0]
                
                # Get compound properties including SMILES
                prop_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IsomericSMILES,CanonicalSMILES/JSON"
                response = requests.get(prop_url)
                response.raise_for_status()
            
            prop_data = response.json()
            
            if "PropertyTable" not in prop_data or "Properties" not in prop_data["PropertyTable"] or not prop_data["PropertyTable"]["Properties"]:
                return f"Properties not found for '{query}' in PubChem."
                
            properties = prop_data["PropertyTable"]["Properties"][0]
            
            # Prefer IsomericSMILES if available, otherwise use CanonicalSMILES
            smiles = properties.get("IsomericSMILES", properties.get("CanonicalSMILES"))
            
            if smiles:
                return f"SMILES: {smiles}"
            else:
                return f"No SMILES available in PubChem."
                
        except Exception as e:
            return f"PubChem error: {str(e)}"


