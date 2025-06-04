import requests
import re
import urllib.parse
from rdkit import Chem

class NameToSMILES: 

    def __init__(self): 
        pass

    def _run(self, query: str) -> str:
        print(f"[NameToSMILES Class] _run called with: {query}")
        try:
            if ">>" in query or (len(query) > 3 and any(c in query for c in "()[]=#@\\/.")): 
                 if not re.search(r"\s", query) and len(query) > 7: 
                   
                    try:
                        mol_test = Chem.MolFromSmiles(query)
                        if mol_test:
                             return f"Error: Input '{query}' appears to be a SMILES string, not a chemical name. This tool converts names to SMILES."
                    except:
                        pass 
            
            cas_result_str = self._try_cas_common_chemistry(query)
            
            if "SMILES:" in cas_result_str:
                smiles = cas_result_str.split("SMILES:", 1)[1].split("\n")[0].strip()
                if smiles: 
                    return f"SMILES: {smiles}\nSource: CAS Common Chemistry"
                
            print(f"[NameToSMILES Class] CAS failed or no SMILES, trying PubChem for: {query}. CAS_Result: {cas_result_str}")
            pubchem_result_str = self._try_pubchem(query)
            if "SMILES:" in pubchem_result_str:
                smiles = pubchem_result_str.split("SMILES:", 1)[1].split("\n")[0].strip()
                if smiles:
                    return f"SMILES: {smiles}\nSource: PubChem"
            
            return f"No SMILES found for '{query}' from CAS or PubChem. CAS: '{cas_result_str}'. PubChem: '{pubchem_result_str}'."
            
        except Exception as e:
            return f"Exception in NameToSMILES _run for '{query}': {str(e)}"
    
    def _try_cas_common_chemistry(self, query: str) -> str:
        try:
            search_url = f"https://commonchemistry.cas.org/api/search?q={urllib.parse.quote(query)}" 
            search_resp = requests.get(search_url, timeout=10) 
            search_resp.raise_for_status()
            
            results_json = search_resp.json()
            
            if not results_json or "results" not in results_json or not results_json["results"]:
                return f"No results found for '{query}' in CAS Common Chemistry search."
            
            cas_rn = results_json["results"][0].get("rn")
            if not cas_rn:
                return f"CAS RN not found for '{query}' in CAS search results."
            
            detail_url = f"https://commonchemistry.cas.org/api/detail?cas_rn={cas_rn}"
            detail_resp = requests.get(detail_url, timeout=10) 
            detail_resp.raise_for_status()
            
            details_json = detail_resp.json() 
            
            smiles_val = details_json.get("smile") or details_json.get("canonicalSmile") 
            
            if smiles_val:
                return f"SMILES: {smiles_val}"
            else:
                return f"No SMILES available in CAS Common Chemistry detail for RN {cas_rn}."
                
        except requests.exceptions.HTTPError as h_err:
            return f"CAS Common Chemistry HTTP error for '{query}': {str(h_err)} (Status: {h_err.response.status_code})"
        except Exception as e:
            return f"CAS Common Chemistry error for '{query}': {str(e)}"
    
    def _try_pubchem(self, query: str) -> str:
        try:
            encoded_query = urllib.parse.quote(query)
            
            url_direct = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_query}/property/IsomericSMILES,CanonicalSMILES/JSON"
            response_direct = requests.get(url_direct, timeout=10)
            
            prop_data = None
            if response_direct.status_code == 200:
                try:
                    prop_data = response_direct.json()
                except requests.exceptions.JSONDecodeError:
                    print(f"[NameToSMILES Class] PubChem direct property endpoint for '{query}' did not return valid JSON.")
            
            if not prop_data: 
                print(f"[NameToSMILES Class] PubChem direct property failed for '{query}' (Status: {response_direct.status_code}). Trying search->CID->property.")
                search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_query}/cids/JSON"
                search_resp = requests.get(search_url, timeout=10)
                search_resp.raise_for_status()
                search_data = search_resp.json()
                
                if "IdentifierList" not in search_data or not search_data["IdentifierList"].get("CID"):
                    return f"No CID results found for '{query}' in PubChem search."
                cid = search_data["IdentifierList"]["CID"][0]
                
                prop_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IsomericSMILES,CanonicalSMILES/JSON"
                response_prop = requests.get(prop_url, timeout=10)
                response_prop.raise_for_status()
                prop_data = response_prop.json()

            if not prop_data or "PropertyTable" not in prop_data or "Properties" not in prop_data["PropertyTable"] or not prop_data["PropertyTable"]["Properties"]:
                return f"Properties not found for '{query}' in PubChem results."
            properties = prop_data["PropertyTable"]["Properties"][0]
            smiles_val = properties.get("IsomericSMILES") or properties.get("CanonicalSMILES")
            
            if smiles_val:
                return f"SMILES: {smiles_val}" 
            else:
                return f"No SMILES available in PubChem properties for '{query}'."
                
        except requests.exceptions.HTTPError as h_err:
             return f"PubChem HTTP error for '{query}': {str(h_err)} (Status: {h_err.response.status_code if h_err.response else 'N/A'})"
        except Exception as e:
            return f"PubChem error for '{query}': {str(e)}"