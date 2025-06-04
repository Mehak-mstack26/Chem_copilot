import sys
import os
from dotenv import load_dotenv
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, fields, asdict # Added asdict
from enum import Enum
from datetime import datetime
import pubchempy as pcp
import requests # For Perplexity API

load_dotenv()
# --- Project Root Setup ---
# If script and pricing JSONs are all in the same directory:
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = current_script_dir # All relevant files in the script's directory

if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

# --- API KEY DEFINITIONS ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")

# Choose your default LLM provider: "openai" or "perplexity"
DEFAULT_LLM_PROVIDER = os.environ.get("DEFAULT_LLM_PROVIDER", "openai") # Default to openai if not set

if not OPENAI_API_KEY and DEFAULT_LLM_PROVIDER == "openai":
    print("Warning: OPENAI_API_KEY environment variable is not set. OpenAI calls may fail if it's the selected provider.")
if not PERPLEXITY_API_KEY and DEFAULT_LLM_PROVIDER == "perplexity":
    print("Warning: PERPLEXITY_API_KEY environment variable is not set. Perplexity calls may fail if it's the selected provider.")

# --- Import OpenAI after keys are potentially defined ---
if OPENAI_API_KEY and "xxxx" not in OPENAI_API_KEY and "YOUR_OPENAI_API_KEY_HERE" not in OPENAI_API_KEY:
    try:
        import openai
    except ImportError:
        print("OpenAI library not installed. Run: pip install openai")
        openai = None
else:
    if DEFAULT_LLM_PROVIDER == "openai" and OPENAI_API_KEY and ("xxxx" in OPENAI_API_KEY or "YOUR_OPENAI_API_KEY_HERE" in OPENAI_API_KEY):
        print("Warning: OpenAI API key appears to be a placeholder. OpenAI calls might fail.")
    elif DEFAULT_LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        print("Warning: OpenAI is the default provider, but OpenAI API key is not set. OpenAI calls will fail.")
    openai = None


# --- Enums and Dataclasses ---
class HazardLevel(Enum):
    LOW = "Low"; MODERATE = "Moderate"; HIGH = "High"; EXTREME = "Extreme"; UNKNOWN = "Unknown"

class SolubilityType(Enum):
    WATER_SOLUBLE = "Water Soluble"; ORGANIC_SOLUBLE = "Organic Soluble"
    POORLY_SOLUBLE = "Poorly Soluble"; INSOLUBLE = "Insoluble"; UNKNOWN = "Unknown"

@dataclass
class ChemicalProperties:
    name: str
    original_query: Optional[str] = None
    iupac_name: Optional[str] = None
    common_names: List[str] = field(default_factory=list)
    molecular_formula: Optional[str] = None
    molecular_weight: Optional[float] = None
    cas_number: Optional[str] = None
    smiles: Optional[str] = None
    pubchem_cid: Optional[int] = None

    solubility: Dict[str, Any] = field(default_factory=dict)
    solubility_rating: Optional[int] = None

    hazard_level: HazardLevel = HazardLevel.UNKNOWN
    is_corrosive: Optional[bool] = None
    is_flammable: Optional[bool] = None
    is_toxic: Optional[bool] = None
    ghs_hazards: List[Dict[str, str]] = field(default_factory=list)
    hazard_rating: Optional[int] = None
    notes_on_hazards: Optional[str] = None

    green_chemistry_score: Optional[int] = None
    notes_on_green_chemistry: Optional[str] = None

    environmental_impact: Optional[str] = None

    estimated_price_per_kg: Optional[float] = None
    price_currency: Optional[str] = None
    supplier_info: List[Dict[str, Any]] = field(default_factory=list)

    safety_notes: List[str] = field(default_factory=list)

    # --- Physical Properties for Recyclability (LLM will provide these) ---
    boiling_point_celsius: Optional[float] = None
    melting_point_celsius: Optional[float] = None
    density_g_ml: Optional[float] = None
    vapor_pressure_mmhg: Optional[float] = None
    azeotrope_formation_notes: Optional[str] = None
    thermal_stability_notes: Optional[str] = None
    reactivity_stability_notes: Optional[str] = None
    common_recovery_methods: List[str] = field(default_factory=list)

    # --- Recyclability Score & Notes (To be calculated by OUR Python code) ---
    calculated_recyclability_score: Optional[int] = None
    calculated_recyclability_notes: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChemicalProperties':
        # A simple way to create an instance from a dictionary,
        # especially useful when retrieving from cache.
        # This needs to be robust if dict keys don't exactly match field names.
        # For HazardLevel, we need to convert string back to Enum instance
        hazard_level_str = data.get('hazard_level')
        if isinstance(hazard_level_str, str):
            try:
                data['hazard_level'] = HazardLevel[hazard_level_str.upper()]
            except KeyError:
                data['hazard_level'] = HazardLevel.UNKNOWN
        
        # Filter dict to only include keys that are fields of the dataclass
        field_names = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        
        # Handle nested structures like 'solubility' if they are not dicts in the source
        if 'solubility' in filtered_data and not isinstance(filtered_data['solubility'], dict):
            # If it's just a string, try to put it into the expected structure
            if isinstance(filtered_data['solubility'], str):
                 filtered_data['solubility'] = {'water_solubility': filtered_data['solubility']}
            else: # Fallback if it's some other unexpected type
                 filtered_data['solubility'] = {}


        return cls(**filtered_data)


    def to_dict(self):
        data = asdict(self)
        if isinstance(data.get('hazard_level'), HazardLevel):
            data['hazard_level'] = data['hazard_level'].value
        if not isinstance(data.get('solubility'), dict):
            sol_val = data.get('solubility')
            data['solubility'] = {}
            if isinstance(sol_val, str):
                 data['solubility']['water_solubility'] = sol_val
        return data

# --- ChemicalAnalysisAgent Class ---
class ChemicalAnalysisAgent:
    PRICING_FILE_PRIMARY = "pricing_data.json"
    PRICING_FILE_SECONDARY = "second_source.json"
    PRICING_FILE_TERTIARY = "sigma_source.json"
    USD_TO_INR_RATE = 83.0

    def __init__(self,
                 openai_api_key: Optional[str] = OPENAI_API_KEY,
                 perplexity_api_key: Optional[str] = PERPLEXITY_API_KEY,
                 llm_provider: str = DEFAULT_LLM_PROVIDER):

        self.llm_provider = llm_provider
        self.openai_api_key_val = openai_api_key
        self.perplexity_api_key_val = perplexity_api_key
        self.openai_client = None

        if self.llm_provider == "openai":
            if not self.openai_api_key_val or "YOUR_OPENAI_API_KEY_HERE" in self.openai_api_key_val or "xxxx" in self.openai_api_key_val:
                print("Warning: OpenAI API key is required for 'openai' provider but appears to be a placeholder or not fully set.")
            if openai:
                try:
                    self.openai_client = openai.OpenAI(api_key=self.openai_api_key_val)
                    print("[LLM Init] OpenAI client initialized successfully.")
                except Exception as e:
                    print(f"Error initializing OpenAI client: {e}. OpenAI features will be disabled.")
                    self.openai_client = None
            else:
                print("[LLM Init] OpenAI library not available, client not initialized.")
        elif self.llm_provider == "perplexity":
            if not self.perplexity_api_key_val or "YOUR_PERPLEXITY_API_KEY_HERE" in self.perplexity_api_key_val or "xxxx" in self.perplexity_api_key_val:
                print("Warning: Perplexity API key is required for 'perplexity' provider but appears to be a placeholder or not fully set.")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}. Choose 'openai' or 'perplexity'.")

        print(f"[LLM] Attempting to use '{self.llm_provider}' provider for LLM tasks.")
        self.pricing_sources = []
        self._load_all_pricing_data()

    def _load_single_pricing_source(self, filename: str, source_display_name: str) -> None:
        pricing_file_path = os.path.join(project_root_dir, filename)
        try:
            with open(pricing_file_path, 'r', encoding='utf-8') as f: data = json.load(f)
            print(f"[Pricing] Successfully loaded: {source_display_name} ({filename}) from {pricing_file_path}")
            self.pricing_sources.append({"name": source_display_name, "filename": filename, "data": data})
        except FileNotFoundError:
            print(f"[Pricing Warn] Not found: {filename} at {pricing_file_path}. Source unavailable.")
        except json.JSONDecodeError as e:
            print(f"[Pricing Error] Parse error {filename}: {e}. Source unavailable.")
        except Exception as e:
            print(f"[Pricing Error] Load error {filename}: {e}. Source unavailable.")

    def _load_all_pricing_data(self) -> None:
        self.pricing_sources = []
        self._load_single_pricing_source(self.PRICING_FILE_PRIMARY, "Primary Local Data")
        self._load_single_pricing_source(self.PRICING_FILE_SECONDARY, "Secondary Local Data")
        self._load_single_pricing_source(self.PRICING_FILE_TERTIARY, "Tertiary Local Data (Sigma)")

    def _is_cas_number(self, identifier: str) -> bool:
        return bool(re.match(r'^\d{2,7}-\d{2}-\d$', identifier))

    def _is_smiles_like(self, identifier: str) -> bool:
        if not isinstance(identifier, str): return False
        identifier_stripped = identifier.strip()
        if not identifier_stripped or " " in identifier_stripped:
            return False

        # If it contains typical SMILES structural characters, it's highly likely.
        if any(c in identifier_stripped for c in "()[]=#@.\\"): # Added dot for disconnected structures
            return True
        
        # If it mostly contains characters common in SMILES (atomic symbols, numbers for ring closures)
        # This is a loose check.
        # Allowed: A-Z, a-z, 0-9, and specific symbols above.
        # Basic check for atom symbols (one uppercase, optional lowercase) and possibly numbers.
        # Avoids matching things that are clearly just words or random strings.
        if re.fullmatch(r"[A-Za-z0-9+\-]*", identifier_stripped): # Simplified: allows + and - for charges
            # If it contains any lowercase letter (part of two-letter element symbols) or a number
            if re.search(r"[a-z0-9]", identifier_stripped):
                return True
            # For short, all-caps strings like "CCO", "CO", "NO" - these are ambiguous
            # but PubChem can often resolve them correctly as SMILES.
            if len(identifier_stripped) <= 3 and identifier_stripped.isupper() and identifier_stripped.isalpha():
                # Check if it is NOT a common short word/abbreviation that isn't a SMILES
                # This list would be hard to maintain. Better to let PubChem try.
                return True # Let PubChem try for things like "CCO", "PCL" etc.
            # If it's just a single uppercase letter, could be an atom
            if len(identifier_stripped) == 1 and identifier_stripped.isupper():
                return True

        return False


    def _get_pubchem_data(self, chemical_identifier: str) -> Tuple[Optional[pcp.Compound], Optional[Dict[str, Any]]]:
        # (Code from your previous version - unchanged)
        print(f"[PubChem] Attrib. for '{chemical_identifier}'")
        compound: Optional[pcp.Compound] = None
        full_json_data: Optional[Dict[str, Any]] = None
        search_methods = []
        if self._is_cas_number(chemical_identifier):
            search_methods.append({'id': chemical_identifier, 'namespace': 'cas', 'type': 'CAS'})
        if self._is_smiles_like(chemical_identifier):
            search_methods.append({'id': chemical_identifier, 'namespace': 'smiles', 'type': 'SMILES'})
        if not any(m['id'] == chemical_identifier and m['type'] == 'Name' for m in search_methods): # Avoid duplicate name search if identifier itself is name-like
             search_methods.append({'id': chemical_identifier, 'namespace': 'name', 'type': 'Name'})

        for method in search_methods:
            if compound: break # Found compound, no need to try other methods
            print(f"[PubChem] Trying {method['type']} search for '{method['id']}'...")
            try:
                compounds = pcp.get_compounds(method['id'], method['namespace'])
                if compounds:
                    compound = compounds[0]
                    print(f"[PubChem] Found by {method['type']} '{method['id']}': CID {compound.cid}")
                    try:
                        print(f"[PubChem] Fetching full JSON record for CID {compound.cid}...")
                        # Attempt to get PUG View JSON first as it's often more comprehensive for GHS
                        response = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{compound.cid}/JSON")
                        if response.status_code == 404: # Fallback to PUG REST if PUG View is not found
                            print("[PubChem] PUG View JSON not found, trying PUG REST JSON...")
                            response = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{compound.cid}/JSON")
                        response.raise_for_status() # Raise HTTPError for bad responses (4XX or 5XX)
                        full_json_data = response.json()
                        print(f"[PubChem] Successfully fetched full JSON for CID {compound.cid}.")
                    except requests.RequestException as e_json:
                        print(f"[PubChem Error] Failed to fetch full JSON for CID {compound.cid}: {e_json}")
                        full_json_data = None # Ensure it's None on failure
                    except json.JSONDecodeError as e_decode:
                        print(f"[PubChem Error] Failed to parse full JSON for CID {compound.cid}: {e_decode}")
                        full_json_data = None # Ensure it's None on failure
                    break # Exit loop once compound is found and JSON fetched (or fetch attempted)
            except pcp.PubChemHTTPError as e_pcp: # Specific PubChemPy error
                print(f"[PubChem] {method['type']} search failed for '{method['id']}': {e_pcp}")
            except requests.exceptions.RequestException as e_req: # General requests error
                print(f"[PubChem] Network error during {method['type']} search for '{method['id']}': {e_req}")
            except Exception as e_gen: # Other potential errors
                print(f"[PubChem] General error in {method['type']} search for '{method['id']}': {e_gen}")
        
        if not compound:
            print(f"[PubChem] No compound found for '{chemical_identifier}' after all attempts.")
        return compound, full_json_data


    def _extract_ghs_from_pubchem_json(self, pubchem_json: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
        # (Code from your previous version - unchanged)
        ghs_hazards_list = []
        if not pubchem_json: return ghs_hazards_list
        try:
            # Check if it's PUG View JSON structure
            record = pubchem_json.get('Record')
            if not record:
                # Check for PUG REST JSON structure (simpler, might lack detailed GHS)
                if 'PC_Compounds' in pubchem_json:
                    # Limited GHS extraction from PUG REST if needed; for now, we primarily expect PUG View
                    print("[PubChem GHS] GHS extraction from PUG REST JSON is complex and may be limited. PUG View preferred.")
                return ghs_hazards_list # Or attempt basic extraction from PC_Compounds if designed

            sections = record.get('Section', [])
            safety_section = next((s for s in sections if s.get('TOCHeading') == 'Safety and Hazards'), None)
            if not safety_section: return ghs_hazards_list

            haz_id_section = next((s for s in safety_section.get('Section', []) if s.get('TOCHeading') == 'Hazards Identification'), None)
            if not haz_id_section: return ghs_hazards_list

            ghs_class_section = next((s for s in haz_id_section.get('Section', []) if s.get('TOCHeading') == 'GHS Classification'), None)
            if not ghs_class_section: return ghs_hazards_list
            
            information_list = ghs_class_section.get('Information', [])
            
            # First, try to map pictograms to their descriptions from the "Pictogram(s)" entry
            pictograms_map = {} # Stores URL -> Description
            for info_item_pict in information_list:
                if info_item_pict.get('Name') == 'Pictogram(s)':
                    value_pict = info_item_pict.get('Value')
                    if value_pict and isinstance(value_pict.get('StringWithMarkup'), list):
                        for swm_pict in value_pict['StringWithMarkup']:
                            if isinstance(swm_pict.get('Markup'), list):
                                for markup in swm_pict['Markup']:
                                    if markup.get('Type') == 'Icon':
                                        # Use URL as a key, Extra as description.
                                        # Some entries might lack 'Extra' (description), use a placeholder.
                                        pict_url = markup.get('URL', 'N/A_URL_'+str(len(pictograms_map))) # Ensure unique key if URL missing
                                        pict_desc = markup.get('Extra', 'Unknown Pictogram')
                                        pictograms_map[pict_url] = pict_desc
            
            # Then, process GHS Hazard Statements
            for info_item in information_list:
                name = info_item.get('Name')
                value_obj = info_item.get('Value')
                if not value_obj or not isinstance(value_obj, dict): continue

                if name == 'GHS Hazard Statements':
                    statements_data = value_obj.get('StringWithMarkup', [])
                    for swm_item in statements_data:
                        text = swm_item.get('String', '')
                        # Regex to capture H-code (optional), statement text, and ignore percentages/sources
                        match = re.match(r"(H\d{3}[A-Za-z+]*)?\s*(?:\(.*\%\))?[:\s]*(.*?)(?:\s*\[(?:Warning|Danger).*?\].*)?$", text.strip())
                        if match:
                            h_code = match.group(1) or ""  # H-code like H302, H302+H312
                            statement_text = match.group(2).strip() # The actual hazard statement
                            
                            if statement_text: # Ensure there's a statement
                                # Attempt to find a pictogram. This is heuristic as direct mapping is complex.
                                # For simplicity, we'll just pick the first pictogram description found, or "N/A"
                                pictogram_display = next(iter(pictograms_map.values()), "N/A") if pictograms_map else "N/A"
                                
                                full_statement = f"{h_code}: {statement_text}".strip().lstrip(": ")
                                # Avoid duplicates
                                if not any(entry['statement'] == full_statement for entry in ghs_hazards_list):
                                    ghs_hazards_list.append({"pictogram": pictogram_display, "statement": full_statement})
            
            if ghs_hazards_list:
                print(f"[PubChem GHS] Extracted {len(ghs_hazards_list)} unique GHS entries from JSON.")
        except Exception as e:
            print(f"[PubChem GHS Error] Error parsing GHS from JSON: {e}")
        return ghs_hazards_list


    def _search_single_local_source(self, source_data: Dict[str, Any], input_id_norm: str, iupac_norm: Optional[str], commons_norm: List[str], smiles: Optional[str]) -> Optional[Dict[str, Any]]:
        # (Code from your previous version - unchanged)
        if not source_data: return None

        # 1. SMILES based search (more reliable if SMILES is good)
        if smiles:
            # Direct key match for SMILES
            if smiles in source_data:
                v_list = source_data[smiles]
                if isinstance(v_list, (float, int)): # Simplest: "SMILES": price
                    return {"price": float(v_list), "currency": "INR", "location": "Unknown Location", "source_name_in_json": smiles, "match_type": "SMILES Key"}
                elif isinstance(v_list, list) and len(v_list) >= 1 and isinstance(v_list[0], (float, int)): # "SMILES": [price, location_optional]
                    price = v_list[0]
                    loc = v_list[1] if len(v_list) >=2 and isinstance(v_list[1], str) else "Unknown Location"
                    return {"price": float(price), "currency": "INR", "location": loc, "source_name_in_json": smiles, "match_type": "SMILES Key"}

            # Value match for SMILES: "Chemical Name": ["SMILES", price, location_optional]
            for k, v_list in source_data.items():
                if isinstance(v_list, list) and len(v_list) >= 2 and isinstance(v_list[0], str) and v_list[0] == smiles:
                    price = v_list[1] if isinstance(v_list[1], (float, int)) else None
                    loc = v_list[2] if len(v_list) >=3 and isinstance(v_list[2], str) else "Unknown Location"
                    if price is not None:
                        return {"price": float(price), "currency": "INR", "location": loc, "source_name_in_json": k, "match_type": "SMILES Value"}
        
        # 2. Name based search (if SMILES search fails or SMILES not available)
        names_to_check = set(commons_norm) # Start with common names
        if iupac_norm: names_to_check.add(iupac_norm)
        names_to_check.add(input_id_norm) # Add the original query too

        matches = []
        for k, v_list in source_data.items(): # k is chemical name from JSON key
            key_norm = k.lower().strip()
            # Handle cases like "Methanol (Reagent Grade)" -> "methanol"
            base_key_norm = re.match(r"^(.*?)\s*\(", key_norm) 
            base_key_norm = base_key_norm.group(1).strip() if base_key_norm else key_norm

            price_val, loc_val = None, "Unknown Location"
            
            if isinstance(v_list, (float, int)): # "Name": price
                price_val = v_list
            elif isinstance(v_list, list) and len(v_list) >= 1:
                if isinstance(v_list[0], (float, int)): # "Name": [price, loc_opt]
                    price_val = v_list[0]
                    if len(v_list) >= 2 and isinstance(v_list[1], str): loc_val = v_list[1]
                elif len(v_list) >= 2 and isinstance(v_list[1], (float, int)): # "Name": ["SMILES_val", price, loc_opt] - SMILES already handled, but price structure is same
                    price_val = v_list[1]
                    if len(v_list) >= 3 and isinstance(v_list[2], str): loc_val = v_list[2]
            
            if price_val is not None:
                for name_check_norm in names_to_check:
                    if name_check_norm == base_key_norm or name_check_norm == key_norm: # Match against base or full key
                        matches.append({"price": float(price_val), "currency": "INR", "location": loc_val, 
                                        "source_name_in_json": k, "match_type": "Exact Name", "len": len(name_check_norm)})
        
        if matches:
            # Prefer longer, more specific matches
            best_match = sorted(matches, key=lambda x: x["len"], reverse=True)[0]
            del best_match["len"] # Remove temporary len key
            return best_match
            
        return None


    def _get_pricing_from_all_local_sources(self, in_id: str, iupac: Optional[str], commons: List[str], smiles: Optional[str]) -> Optional[Dict[str, Any]]:
        # (Code from your previous version - unchanged)
        in_id_n = in_id.lower().strip()
        iupac_n = iupac.lower().strip() if iupac else None
        commons_n = [c.lower().strip() for c in commons if c] # Filter out empty common names

        for src in self.pricing_sources:
            if not src.get("data"): continue # Skip if source data failed to load

            print(f"[Pricing] Searching in {src['name']} for '{in_id}' (SMILES: {smiles})...")
            match = self._search_single_local_source(src["data"], in_id_n, iupac_n, commons_n, smiles)
            if match:
                print(f"[Pricing] Found '{in_id}' in {src['name']}. Price: {match['price']} {match['currency']}")
                match.update({"source_file_display_name": src['name'], "source_filename": src['filename']})
                return match
        
        print(f"[Pricing] '{in_id}' (SMILES: {smiles}, IUPAC: {iupac}, Names: {commons}) not found in local data.")
        return None


    def _get_perplexity_completion(self, system_prompt_content: str, user_prompt_content: str) -> Optional[str]:
        # (Code from your previous version - unchanged)
        if not self.perplexity_api_key_val or "YOUR_PERPLEXITY_API_KEY_HERE" in self.perplexity_api_key_val or "xxxx" in self.perplexity_api_key_val:
            print("[Perplexity Error] API key not configured or is a placeholder.")
            return None
        payload = {"model": "llama-3-sonar-large-32k-online", "messages": [{"role": "system", "content": system_prompt_content}, {"role": "user", "content": user_prompt_content}]}
        headers = {"Authorization": f"Bearer {self.perplexity_api_key_val}", "Content-Type": "application/json", "Accept": "application/json"}
        try:
            response = requests.post("https://api.perplexity.ai/chat/completions", json=payload, headers=headers, timeout=120) # Increased timeout
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.RequestException as e:
            print(f"[Perplexity API Error] Request failed: {e}")
            if hasattr(e, 'response') and e.response is not None: print(f"Response: {e.response.status_code}, {e.response.text[:200]}")
        except (KeyError, IndexError, TypeError) as e:
            print(f"[Perplexity API Error] Parse response: {e}")
        return None

    def _get_openai_completion(self, system_prompt_content: str, user_prompt_content: str) -> Optional[str]:
        # (Code from your previous version - unchanged, added e.request for APIStatusError)
        if not self.openai_client:
            print("[OpenAI Error] Client not initialized.")
            return None
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "system", "content": system_prompt_content}, {"role": "user", "content": user_prompt_content}],
                temperature=0.2,
                timeout=120.0 
            )
            return response.choices[0].message.content
        except openai.APIConnectionError as e: print(f"[OpenAI API Error] Connection error: {e}")
        except openai.RateLimitError as e: print(f"[OpenAI API Error] Rate limit exceeded: {e}")
        except openai.AuthenticationError as e: print(f"[OpenAI API Error] Authentication error: {e}")
        except openai.APIStatusError as e: print(f"[OpenAI API Error] Status error {e.status_code}: {e.response}. Request: {e.request if hasattr(e, 'request') else 'N/A'}")
        except openai.APITimeoutError as e: print(f"[OpenAI API Error] Request timed out: {e}")
        except Exception as e:
            print(f"[OpenAI API Error or other] {type(e).__name__}: {e}")
        return None

    def _get_llm_completion(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        if self.llm_provider == "openai": return self._get_openai_completion(system_prompt, user_prompt)
        elif self.llm_provider == "perplexity": return self._get_perplexity_completion(system_prompt, user_prompt)
        print(f"[LLM Error] Provider '{self.llm_provider}' not supported or configured.")
        return None

    def _get_llm_derived_pricing(self, chemical_name: str, smiles: Optional[str], formula: Optional[str], cid: Optional[int]) -> Optional[Dict[str, Any]]:
        # (Code from your previous version - unchanged)
        context_parts = [f"Chemical: {chemical_name}"]
        if smiles: context_parts.append(f"SMILES: {smiles}")
        if formula: context_parts.append(f"Formula: {formula}")
        if cid: context_parts.append(f"PubChem CID: {cid}")
        system_prompt = "You are a chemical market analyst. Provide price estimations in JSON format. Respond *only* with the JSON object."
        user_prompt = f"""{", ".join(context_parts)}
Estimate bulk price in INR/kg or USD/kg. For research/small industrial scale. Provide numerical estimate or range (e.g., "10000-15000" INR, "100-150" USD). Prioritize INR.
JSON: {{"estimated_price_per_kg_inr": "float_or_range_or_null", "estimated_price_per_kg_usd": "float_or_range_or_null", "price_confidence": "low/medium/high", "price_basis_notes": "Brief notes..."}}
Respond ONLY with the JSON object.
""" # Note: float_or_range_or_null was string in prompt, actual parsing handles float/range
        llm_response_content = self._get_llm_completion(system_prompt, user_prompt)
        if not llm_response_content:
            print(f"[Pricing LLM] No content from LLM for '{chemical_name}'.")
            return None
        
        json_str_to_parse = llm_response_content # Start with the full response
        try:
            # Attempt to extract JSON from markdown code block first
            match_json_block = re.search(r"```json\s*([\s\S]*?)\s*```", llm_response_content, re.IGNORECASE)
            if match_json_block:
                json_str_to_parse = match_json_block.group(1).strip()
            else:
                # If no code block, try to match a direct JSON object
                match_direct_json = re.search(r"^\s*(\{[\s\S]*?\})\s*$", llm_response_content.strip())
                if match_direct_json:
                    json_str_to_parse = match_direct_json.group(1)
                # else: it's not a clean JSON block, json.loads will likely fail but we let it try

            if not json_str_to_parse.strip(): # Check if string is empty after stripping
                print(f"[Pricing LLM Error] Extracted JSON string is empty for '{chemical_name}'. LLM response: '{llm_response_content[:200]}...'")
                return None

            data = json.loads(json_str_to_parse)
            price_usd_raw = data.get("estimated_price_per_kg_usd")
            price_inr_raw = data.get("estimated_price_per_kg_inr")
            final_price_inr, llm_currency = None, None

            def parse_val(v_raw: Any) -> Optional[float]:
                if isinstance(v_raw, (int, float)): return float(v_raw)
                if isinstance(v_raw, str):
                    v_clean = v_raw.replace(',', '').strip()
                    if not v_clean or v_clean.lower() == 'null' or v_clean.lower() == 'n/a': return None
                    # Try to parse range "100-150" or "100 - 150"
                    m = re.match(r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)", v_clean)
                    if m: return (float(m.group(1)) + float(m.group(2))) / 2.0
                    # Try to parse single float
                    try: return float(v_clean)
                    except ValueError: return None
                return None

            price_inr = parse_val(price_inr_raw)
            price_usd = parse_val(price_usd_raw)

            if price_inr is not None:
                final_price_inr, llm_currency = price_inr, "INR"
            elif price_usd is not None:
                final_price_inr, llm_currency = price_usd * self.USD_TO_INR_RATE, "USD"
            
            if final_price_inr is not None:
                return {"price_inr": final_price_inr, "currency_llm_provided": llm_currency,
                        "raw_llm_price_value": price_inr_raw if llm_currency == "INR" else price_usd_raw,
                        "confidence": data.get("price_confidence"), "basis_notes": data.get("price_basis_notes"),
                        "source_type": f"LLM ({self.llm_provider})"}
            else: # No valid price found from LLM
                return {"price_inr": None, "currency_llm_provided": None, "raw_llm_price_value": None,
                        "confidence": data.get("price_confidence", "low_no_price"),
                        "basis_notes": data.get("price_basis_notes", "LLM provided no price or unrecognized format."),
                        "source_type": f"LLM ({self.llm_provider})"}

        except json.JSONDecodeError as e:
            print(f"[Pricing LLM Error] JSONDecodeError for '{chemical_name}': {e}. String tried: '{json_str_to_parse[:200]}...'")
        except Exception as e: # Catch other potential errors during parsing
            print(f"[Pricing LLM Error] General error parsing price for '{chemical_name}': {e}. String tried: '{json_str_to_parse[:200]}...'")
        return None


    def _parse_int_score(self, val: Any, field_name: str) -> Optional[int]:
        # (Code from your previous version - unchanged)
        if val is None: return None
        if isinstance(val, str) and (val.lower() == 'null' or val.lower() == 'n/a' or not val.strip()):
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            print(f"[LLM Parse Warn] Could not parse integer for {field_name} from value: '{val}'")
            return None


    def _get_llm_derived_properties(self, name: str, formula: Optional[str], smiles: Optional[str],
                                   cid: Optional[int], cas: Optional[str], mw: Optional[float],
                                   iupac: Optional[str]) -> Dict[str, Any]:
        print(f"[LLM Props] Querying LLM ({self.llm_provider}) for derived properties of '{name}'.")
        context_parts = [f"Chemical: {name}."]
        known_info = []
        if cid: known_info.append(f"PubChem CID: {cid}")
        if iupac: known_info.append(f"IUPAC Name: {iupac}")
        if formula: known_info.append(f"Formula: {formula}")
        if mw: known_info.append(f"MW: {mw:.2f}" if isinstance(mw, float) else f"MW: {mw}")
        if smiles: known_info.append(f"SMILES: {smiles}")
        if cas: known_info.append(f"CAS: {cas}")

        if known_info: context_parts.append(f"Known info: {'; '.join(known_info)}.")
        else: context_parts.append("No definitive structural or ID information known from databases yet.")
        full_context = "\n".join(context_parts)
        guess_instr = "" # If you want LLM to guess ID fields, add them here like "iupac_name_llm_guess": "string or null"

        system_prompt = "You are a chemical safety, properties, environmental, and physical properties expert. Provide accurate assessments in JSON format. Respond *only* with the JSON object, no other text or explanations. If guessing core chemical identity, clearly indicate it (e.g., field ends with '_llm_guess')."
        user_prompt = f"""{full_context}
Provide analysis in JSON. Ratings 1-10 (10=high/extreme/excellent) or null. If core ID (IUPAC, formula, MW, CAS, SMILES) unknown, you may guess it if explicitly asked for (e.g. iupac_name_llm_guess).
Respond ONLY with the JSON object.
{{
    {guess_instr}
    "solubility": {{
        "water_solubility": "Water Soluble/Organic Soluble/Poorly Soluble/Insoluble/Unknown",
        "organic_solvents_compatibility": ["list of types like alcohols, ethers, hydrocarbons, or null"],
        "solubility_rating": "integer 1-10 or null",
        "notes_on_solubility": "brief explanation for solubility_rating"
    }},
    "hazards": {{
        "corrosive": true/false/null,
        "flammable": true/false/null,
        "toxic": true/false/null,
        "carcinogenic_suspected": true/false/null,
        "overall_hazard_level": "Low/Moderate/High/Extreme/Unknown",
        "hazard_rating": "integer 1-10 or null",
        "notes_on_hazards": "brief explanation for hazard_rating",
        "environmental_hazard_notes": "brief notes on environmental risks",
        "ghs_info_llm": [
            {{ "pictogram_description": "Name of GHS pictogram", "h_code": "HXXX", "h_statement": "Full GHS hazard statement text"}}
        ]
    }},
    "green_chemistry": {{
        "renewable_feedstock_potential": "yes/no/partially/unknown",
        "atom_economy_typical_reactions": "high/moderate/low/varies/unknown",
        "biodegradability_assessment": "readily/partially/poorly/non-biodegradable/unknown",
        "energy_efficiency_synthesis": "typically high/moderate/low/varies/unknown",
        "waste_generation_typical_reactions": "typically high/moderate/low/varies/unknown",
        "notes_on_green_chemistry": "brief qualitative notes justifying overall_score",
        "overall_score": "integer 1-10 or null"
    }},
    "physical_properties_for_recyclability": {{
        "boiling_point_celsius": "float or null (e.g., 78.3)",
        "melting_point_celsius": "float or null (e.g., -114.1)",
        "density_g_ml": "float or null (e.g., 0.789 at 20C)",
        "vapor_pressure_mmhg_at_20c": "float or null (e.g., 44.0, specify temperature if not 20C in notes if possible)",
        "azeotrope_formation_notes": "string or null (e.g., 'Forms azeotrope with water (95.6% ethanol) boiling at 78.2 C.')",
        "thermal_stability_notes": "string or null (e.g., 'Stable on distillation under atmospheric pressure up to X C.', or 'Decomposes at Y C.')",
        "reactivity_stability_notes": "string or null (e.g., 'Generally stable; avoid strong oxidizing agents. May form peroxides on storage.')",
        "common_recovery_methods": ["list of strings like 'distillation', 'extraction', 'crystallization', 'filtration' or null (These are potential methods, not an assessment of their ease or success yet)"]
    }},
    "safety_precautions": ["list of key safety measures when handling or null"],
    "storage_recommendations": "brief storage condition recommendations or null",
    "disposal_considerations": "brief notes on proper disposal or null",
    "environmental_impact_summary": "brief overall assessment of environmental effects or null"
}}
If 'ghs_info_llm' cannot be determined, use empty list []. For other missing values, use null or "Unknown" as appropriate for the field type.
Ensure overall_hazard_level and water_solubility use specified capitalized options if known, otherwise "Unknown".
Be specific and provide justification for ratings/scores in the notes fields.
"""
        llm_response_content = self._get_llm_completion(system_prompt, user_prompt)

        default_empty_response = {
            "solubility": {"water_solubility": SolubilityType.UNKNOWN.value, "organic_solvents_compatibility": [], "solubility_rating": None, "notes_on_solubility": None},
            "hazards": {"corrosive": None, "flammable": None, "toxic": None, "carcinogenic_suspected": None, "overall_hazard_level": HazardLevel.UNKNOWN.value, "hazard_rating": None, "notes_on_hazards": None, "environmental_hazard_notes": None, "ghs_info_llm": []},
            "green_chemistry": {"renewable_feedstock_potential": "unknown", "atom_economy_typical_reactions": "unknown", "biodegradability_assessment": "unknown", "energy_efficiency_synthesis": "unknown", "waste_generation_typical_reactions": "unknown", "notes_on_green_chemistry": None, "overall_score": None},
            "physical_properties_for_recyclability": { # Changed key name here
                "boiling_point_celsius": None, "melting_point_celsius": None, "density_g_ml": None,
                "vapor_pressure_mmhg_at_20c": None, "azeotrope_formation_notes": None,
                "thermal_stability_notes": None, "reactivity_stability_notes": None,
                "common_recovery_methods": []
            },
            "safety_precautions": [],
            "storage_recommendations": None,
            "disposal_considerations": None,
            "environmental_impact_summary": "Assessment unavailable from LLM."
        }
        # Add guessed fields if you were to ask for them
        # if "iupac_name_llm_guess" not in default_empty_response: default_empty_response["iupac_name_llm_guess"] = None


        if not llm_response_content:
            print(f"[LLM NO RESPONSE for {name}]. Using default structure.")
            return default_empty_response

        print(f"[LLM RAW RESPONSE for {name}]:\n---\n{llm_response_content}\n---")
        
        json_str_to_parse = llm_response_content
        try:
            match_json_block = re.search(r"```json\s*([\s\S]*?)\s*```", llm_response_content, re.IGNORECASE)
            if match_json_block:
                json_str_to_parse = match_json_block.group(1).strip()
            else:
                match_direct_json = re.search(r"^\s*(\{[\s\S]*?\})\s*$", llm_response_content.strip())
                if match_direct_json:
                    json_str_to_parse = match_direct_json.group(1)
            
            if not json_str_to_parse.strip():
                 print(f"[LLM Parse Error] Extracted JSON string is empty for '{name}'. LLM response: '{llm_response_content[:200]}...'")
                 return default_empty_response

            parsed_data = json.loads(json_str_to_parse)
            
            final_data = json.loads(json.dumps(default_empty_response)) # Deep copy default

            for main_key, main_value_from_llm in parsed_data.items():
                if main_key in final_data:
                    if isinstance(main_value_from_llm, dict) and isinstance(final_data[main_key], dict):
                        for sub_key, sub_value_from_llm in main_value_from_llm.items():
                            if sub_key in final_data[main_key]:
                                final_data[main_key][sub_key] = sub_value_from_llm
                            # else: # LLM provided an extra sub-key not in default, optionally add it
                            # final_data[main_key][sub_key] = sub_value_from_llm 
                    else: # Overwrite non-dict field or if types mismatch (e.g. LLM gave string for dict)
                        final_data[main_key] = main_value_from_llm
                # else: # LLM provided an extra main_key not in default, optionally add it
                #    final_data[main_key] = main_value_from_llm
            
            # Ensure environmental_impact_summary isn't stuck on default if LLM provided something else
            if final_data.get("environmental_impact_summary") == "Assessment unavailable from LLM." and \
               parsed_data.get("environmental_impact_summary") and \
               parsed_data.get("environmental_impact_summary") != "Assessment unavailable from LLM.":
                final_data["environmental_impact_summary"] = parsed_data.get("environmental_impact_summary")

            return final_data

        except json.JSONDecodeError as e:
            print(f"[LLM Parse Error] Props for {name}. JSONDecodeError: {e}. Raw string passed to loads (first 500 chars): '{json_str_to_parse[:500]}...'")
            return default_empty_response
        except Exception as e:
            print(f"[LLM Parse Error] Props for {name}. General Exception: {type(e).__name__} - {e}. Raw string passed to loads (first 500 chars): '{json_str_to_parse[:500]}...'")
            return default_empty_response
        
    def _get_cached_property(self, identifier: str, property_name: str, cache: Dict, default: Any = None) -> Any:
        """Safely retrieves a specific property for a chemical from the cache."""
        if identifier in cache and isinstance(cache[identifier], dict):
            return cache[identifier].get(property_name, default)
        # Try original query if identifier might be a resolved name/SMILES
        # This part might need refinement based on how robust your cache keys are
        # For now, assumes identifier is the original_query used for caching.
        return default
        
    def _calculate_reaction_specific_recyclability(
        self,
        chem_props_of_interest: ChemicalProperties, # Object for the agent/solvent
        reaction_details: Dict, # Dict for the current reaction step
        full_chemical_cache: Dict # Full cache to lookup other components
    ) -> Tuple[Optional[int], str]: # Will now return score 0-100
        """
        Calculates a reaction-specific recyclability score (0-100%) and notes.
        """

        min_achievable_points = -10 
        max_achievable_points = 15 
        score_points = 0 
        notes_parts = [f"Reaction-Specific Assessment for '{chem_props_of_interest.name}' (orig: '{chem_props_of_interest.original_query}'):"]

        # Key properties of the chemical of interest (agent/solvent)
        bp_interest = chem_props_of_interest.boiling_point_celsius
        mp_interest = chem_props_of_interest.melting_point_celsius
        thermal_stability_interest = chem_props_of_interest.thermal_stability_notes
        reactivity_stability_interest = chem_props_of_interest.reactivity_stability_notes
        recovery_methods_llm = chem_props_of_interest.common_recovery_methods or []
        
        reaction_temp = reaction_details.get("conditions", {}).get("temperature")
        
        main_product_smiles_key = reaction_details.get("product")
        bp_product = None
        product_name_for_notes = "product"
        if main_product_smiles_key:
            product_data_from_cache = self._get_cached_property(main_product_smiles_key, None, full_chemical_cache) # Get whole dict
            if product_data_from_cache:
                bp_product = product_data_from_cache.get("boiling_point_celsius")
                product_name_for_notes = product_data_from_cache.get("name", main_product_smiles_key)

        # 1. Stability at Reaction Temperature
        if thermal_stability_interest:
            notes_parts.append(f"ThermalNote: '{thermal_stability_interest}'.")
            stable_at_rxn_temp = True 
            if reaction_temp is not None:
                if "decomposes" in thermal_stability_interest.lower():
                    match_decomp_temp = re.search(r"decomposes\s*(?:at|before|around|above)?\s*(\d+\.?\d*)\s*C", thermal_stability_interest, re.IGNORECASE)
                    if match_decomp_temp:
                        decomp_temp = float(match_decomp_temp.group(1))
                        if reaction_temp >= decomp_temp:
                            stable_at_rxn_temp = False
                            notes_parts.append(f"Unstable at reaction temp {reaction_temp}°C (decomposes ~{decomp_temp}°C).")
                            score_points -= 3 
                elif "stable up to" in thermal_stability_interest.lower():
                     match_stable_temp = re.search(r"stable\s*(?:up\s*to)?\s*(\d+\.?\d*)\s*C", thermal_stability_interest, re.IGNORECASE)
                     if match_stable_temp:
                         stable_temp = float(match_stable_temp.group(1))
                         if reaction_temp > stable_temp:
                             stable_at_rxn_temp = False
                             notes_parts.append(f"May be unstable at reaction temp {reaction_temp}°C (stable up to {stable_temp}°C).")
                             score_points -= 2
            if stable_at_rxn_temp and ( "stable under normal conditions" in thermal_stability_interest.lower() or "generally stable" in thermal_stability_interest.lower() or "stable up to" in thermal_stability_interest.lower()): # Added check for positive stability
                notes_parts.append(f"Assumed stable at reaction temperature ({reaction_temp}°C if specified).")
                score_points += 2 # Increased points for stability at rxn temp
        else:
            notes_parts.append("Thermal stability not specified for reaction conditions.")
            score_points -=1 # Penalty for unknown

        # 2. Reactivity in Reaction Context
        if reactivity_stability_interest:
            notes_parts.append(f"ReactivityNote: '{reactivity_stability_interest}'.")
            if any(s in reactivity_stability_interest.lower() for s in ["unstable", "reactive with water", "reactive with air", "peroxides", "polymerizes"]):
                notes_parts.append("Potential reactivity/instability issues in reaction.")
                score_points -= 2
            elif "stable" in reactivity_stability_interest.lower():
                score_points += 1
        else:
            notes_parts.append("General reactivity/stability not specified.")

        # 3. Separation from Main Product (e.g., by distillation)
        can_distill_effectively = False
        stable_for_its_own_distillation = True # Assume true unless proven false

        if bp_interest is not None:
            notes_parts.append(f"Own BP: {bp_interest}°C.")
            if thermal_stability_interest:
                if "decomposes" in thermal_stability_interest.lower():
                    match_decomp = re.search(r"decomposes\s*(?:at|before|around|above)?\s*(\d+\.?\d*)\s*C", thermal_stability_interest, re.IGNORECASE)
                    if match_decomp and bp_interest >= float(match_decomp.group(1)): stable_for_its_own_distillation = False
                    elif "before boiling" in thermal_stability_interest.lower(): stable_for_its_own_distillation = False
            
            if not stable_for_its_own_distillation:
                notes_parts.append("Unstable for its own distillative recovery.")
                score_points -= 3
            else: 
                score_points += 1 
                if "distillation" in [m.lower() for m in recovery_methods_llm]: score_points +=1
                
                if bp_product is not None:
                    notes_parts.append(f"Product ('{product_name_for_notes}') BP: {bp_product}°C.")
                    bp_diff = abs(bp_interest - bp_product)
                    if bp_diff >= 30: 
                        notes_parts.append(f"Good BP difference from product ({bp_diff:.0f}°C).")
                        score_points += 3
                        can_distill_effectively = True
                    elif bp_diff >= 15: 
                        notes_parts.append(f"Moderate BP difference ({bp_diff:.0f}°C).")
                        score_points += 1
                        can_distill_effectively = True 
                    else: 
                        notes_parts.append(f"Poor BP difference ({bp_diff:.0f}°C).")
                        score_points -= 2
                else:
                    notes_parts.append(f"Product ('{product_name_for_notes}') BP unknown for comparison.")
                    if 40 < bp_interest < 200 : # Generally good BP range for a solvent if product BP unknown
                        score_points +=1 
                        can_distill_effectively = True # Assume it might be possible
                    else:
                        score_points -=1 # Less ideal BP if no product to compare against
        else:
            notes_parts.append("Own BP unknown for distillation.")
            score_points -= 2


        # 4. Other recovery if distillation not primary/effective
        if not can_distill_effectively and mp_interest is not None and mp_interest > 20: 
            if "crystallization" in [m.lower() for m in recovery_methods_llm] or \
               "filtration" in [m.lower() for m in recovery_methods_llm]:
                notes_parts.append("Solid, may be recoverable by crystallization/filtration.")
                score_points += 2 
        elif not can_distill_effectively and "extraction" in [m.lower() for m in recovery_methods_llm]:
             notes_parts.append("Extraction suggested if distillation is not primary.")
             score_points +=1
        
        # 5. Azeotropes
        if chem_props_of_interest.azeotrope_formation_notes and \
           chem_props_of_interest.azeotrope_formation_notes.strip().lower() not in ["null", "n/a", "unknown", ""]:
            notes_parts.append(f"AzeotropeNotes: {chem_props_of_interest.azeotrope_formation_notes}.")
            if "forms azeotrope" in chem_props_of_interest.azeotrope_formation_notes.lower():
                score_points -= 1 
                # Further penalty if it's with water and water is relevant (e.g. aqueous workup)
                if "water" in chem_props_of_interest.azeotrope_formation_notes.lower():
                    # Check if reaction involves water explicitly (e.g. as solvent or byproduct)
                    # This requires more context from `reaction_details` if available
                    if reaction_details.get("solvents") and "O" in reaction_details.get("solvents", "").split('.') or \
                       reaction_details.get("agents") and "O" in reaction_details.get("agents", "").split('.'):
                        notes_parts.append("Forms azeotrope with water (present in reaction).")
                        score_points -= 1 # General penalty for azeotrope formation

        # Final score clamping
        percentage_score_val = 50 + (score_points * 5) # Each point is 5% swing from 50%
        
        calculated_score_percent = max(0, min(100, int(round(percentage_score_val))))
        
        final_notes_str = " ".join(notes_parts)
        final_notes_str += f" Reaction-specific recyclability score: {calculated_score_percent}%."
        return calculated_score_percent, final_notes_str
    
    def analyze_chemical(self, chemical_identifier: str) -> ChemicalProperties:
        props = ChemicalProperties(name=chemical_identifier, original_query=chemical_identifier)
        pubchem_compound, pubchem_full_json_data = self._get_pubchem_data(chemical_identifier)
        current_name_for_llm = chemical_identifier

        if pubchem_compound:
            props.name = pubchem_compound.iupac_name or \
                         (pubchem_compound.synonyms[0] if pubchem_compound.synonyms else chemical_identifier)
            current_name_for_llm = props.name
            props.iupac_name = pubchem_compound.iupac_name
            props.smiles = pubchem_compound.canonical_smiles
            props.pubchem_cid = pubchem_compound.cid
            unique_common_names = {name.strip() for name in (pubchem_compound.synonyms or []) if name} 
            if chemical_identifier.strip() not in unique_common_names and \
               (not props.iupac_name or chemical_identifier.strip().lower() != props.iupac_name.strip().lower()):
                unique_common_names.add(chemical_identifier.strip())
            props.common_names = sorted(list(unique_common_names))[:10]
            props.molecular_formula = pubchem_compound.molecular_formula
            props.molecular_weight = float(pubchem_compound.molecular_weight) if pubchem_compound.molecular_weight else None
            cas_from_syns = [s for s in (pubchem_compound.synonyms or []) if self._is_cas_number(s)]
            pubchem_cas_list = getattr(pubchem_compound, 'cas', [])
            if isinstance(pubchem_cas_list, list) and pubchem_cas_list: props.cas_number = pubchem_cas_list[0]
            elif isinstance(pubchem_cas_list, str) and self._is_cas_number(pubchem_cas_list): props.cas_number = pubchem_cas_list
            elif cas_from_syns: props.cas_number = cas_from_syns[0]
            props.ghs_hazards = self._extract_ghs_from_pubchem_json(pubchem_full_json_data)

        llm_derived = self._get_llm_derived_properties(current_name_for_llm, props.molecular_formula,
                                                       props.smiles, props.pubchem_cid, props.cas_number,
                                                       props.molecular_weight, props.iupac_name)
        
        llm_solubility_info = llm_derived.get("solubility", {})
        props.solubility['water_solubility'] = llm_solubility_info.get('water_solubility', SolubilityType.UNKNOWN.value)
        props.solubility['organic_solvents_compatibility'] = llm_solubility_info.get('organic_solvents_compatibility', [])
        props.solubility['notes_on_solubility'] = llm_solubility_info.get('notes_on_solubility')
        props.solubility_rating = self._parse_int_score(llm_solubility_info.get("solubility_rating"), "Solubility rating")

        llm_hazards_info = llm_derived.get("hazards", {})
        haz_lvl_str = str(llm_hazards_info.get("overall_hazard_level", "unknown")).lower()
        try: props.hazard_level = HazardLevel[haz_lvl_str.upper()]
        except KeyError: props.hazard_level = HazardLevel.UNKNOWN
        props.is_corrosive = llm_hazards_info.get("corrosive")
        props.is_flammable = llm_hazards_info.get("flammable")
        props.is_toxic = llm_hazards_info.get("toxic")
        props.hazard_rating = self._parse_int_score(llm_hazards_info.get("hazard_rating"), "Hazard rating")
        props.notes_on_hazards = llm_hazards_info.get("notes_on_hazards")
        if not props.ghs_hazards and "ghs_info_llm" in llm_hazards_info: 
            for item in llm_hazards_info.get("ghs_info_llm", []):
                if isinstance(item, dict) and item.get("h_statement"):
                    new_stmt = f"{item.get('h_code', '')}: {item.get('h_statement', '')}".strip().lstrip(": ")
                    if not any(e['statement'] == new_stmt for e in props.ghs_hazards):
                         props.ghs_hazards.append({"pictogram": item.get("pictogram_description", "N/A"), "statement": new_stmt})

        gc_info = llm_derived.get("green_chemistry", {})
        props.green_chemistry_score = self._parse_int_score(gc_info.get("overall_score"), "GC score")
        props.notes_on_green_chemistry = gc_info.get("notes_on_green_chemistry")
        
        llm_phys_props = llm_derived.get("physical_properties_for_recyclability", {})
        def parse_float_or_none_local(val: Any, field_name: str) -> Optional[float]: # Local helper
            if val is None: return None
            if isinstance(val, (int, float)): return float(val)
            if isinstance(val, str):
                val_clean = val.strip().lower()
                if not val_clean or val_clean == 'null' or val_clean == 'n/a': return None
                try: return float(val)
                except ValueError:
                    print(f"[LLM Parse Warn] Non-float for {field_name} ('{val}') in {props.name if hasattr(props, 'name') else 'Unknown Chemical'}")
                    return None
            return None
            
        props.boiling_point_celsius = parse_float_or_none_local(llm_phys_props.get("boiling_point_celsius"), "BP")
        props.melting_point_celsius = parse_float_or_none_local(llm_phys_props.get("melting_point_celsius"), "MP")
        props.density_g_ml = parse_float_or_none_local(llm_phys_props.get("density_g_ml"), "Density")
        props.vapor_pressure_mmhg = parse_float_or_none_local(llm_phys_props.get("vapor_pressure_mmhg_at_20c"), "VP")
        props.azeotrope_formation_notes = llm_phys_props.get("azeotrope_formation_notes")
        props.thermal_stability_notes = llm_phys_props.get("thermal_stability_notes")
        props.reactivity_stability_notes = llm_phys_props.get("reactivity_stability_notes")
        props.common_recovery_methods = llm_phys_props.get("common_recovery_methods", [])
        if not isinstance(props.common_recovery_methods, list) : props.common_recovery_methods = []

        props.safety_notes = llm_derived.get("safety_precautions", [])
        if not isinstance(props.safety_notes, list): props.safety_notes = []
        props.environmental_impact = llm_derived.get("environmental_impact_summary")
        
        # No generic score calculation here anymore
        props.calculated_recyclability_score = None
        props.calculated_recyclability_notes = None

        # Pricing logic
        local_price = self._get_pricing_from_all_local_sources(chemical_identifier, props.iupac_name, props.common_names, props.smiles)
        # ... (rest of pricing logic as before) ...
        if local_price:
            props.estimated_price_per_kg = local_price.get("price")
            props.price_currency = local_price.get("currency", "INR")
            props.supplier_info = [{"name": f"Local DB: {local_price.get('source_file_display_name','N/A')} ({local_price.get('source_filename','N/A')} - Key: '{local_price.get('source_name_in_json','N/A')}')",
                                    "availability": (f"Price: {props.estimated_price_per_kg:.2f} {props.price_currency}. Match: {local_price.get('match_type','N/A')}."),
                                    "location": local_price.get('location', 'Unknown Location'), "source_type": "Local JSON"}]
        else:
            llm_price_data = self._get_llm_derived_pricing(current_name_for_llm, props.smiles, props.molecular_formula, props.pubchem_cid)
            if llm_price_data and llm_price_data.get("price_inr") is not None:
                props.estimated_price_per_kg = llm_price_data["price_inr"]
                props.price_currency = "INR" 
                raw_val_display = llm_price_data.get("raw_llm_price_value", "N/A")
                llm_curr_display = llm_price_data.get("currency_llm_provided", "N/A")
                avail_details = f"Est. Price: {props.estimated_price_per_kg:.2f} {props.price_currency}/kg. "
                if llm_curr_display and llm_curr_display.upper() != "INR": 
                    avail_details += f"(LLM provided: {raw_val_display} {llm_curr_display}, converted). "
                else: 
                    avail_details += f"(LLM provided: {raw_val_display} {llm_curr_display or 'value'}). "
                avail_details += f"Conf: {llm_price_data.get('confidence','N/A')}. Basis: {llm_price_data.get('basis_notes','N/A')}"
                props.supplier_info = [{"name": f"LLM Estimation ({self.llm_provider})", "availability": avail_details,
                                        "location": "Global Market (Est.)", "source_type": f"LLM ({self.llm_provider})"}]
            else: 
                props.estimated_price_per_kg, props.price_currency = None, None
                avail_note = "Not in local DBs. "
                if llm_price_data: 
                    avail_note += (f"LLM ({self.llm_provider}) consulted: Price estimation difficult. "
                                  f"Conf: {llm_price_data.get('confidence','N/A')}. Basis: {llm_price_data.get('basis_notes','N/A')}")
                else: 
                    avail_note += f"LLM ({self.llm_provider}) call for pricing failed or no parsable pricing."
                props.supplier_info = [{"name": "No Definitive Pricing Data", "availability": avail_note, "location": "N/A", "source_type": "None"}]

        if props.original_query == "CCO": # Debug for CCO
            print(f"--- DEBUG: analyze_chemical PRE-CACHE for original_query '{props.original_query}' ---")
            print(f"  props.name: {props.name}")
            print(f"  props.iupac_name: {props.iupac_name}")
            print(f"  props.smiles: {props.smiles}")
            print(f"  props.pubchem_cid: {props.pubchem_cid}")
            print(f"  props.boiling_point_celsius: {props.boiling_point_celsius}")

        return props


    def generate_report(self, cp: ChemicalProperties) -> str:
        # This report will show the generic physical properties.
        # The reaction-specific scores will be in the final augmented JSON,
        # not directly available to this generic report function for a single chemical.
        report = f"""
CHEMICAL ANALYSIS REPORT ({self.llm_provider.upper()} LLM Used - Generic Properties)
========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

IDENTIFICATION
--------------
Query: {cp.original_query or 'N/A'}, Analyzed As: {cp.name}
IUPAC: {cp.iupac_name or 'N/A'}, CID: {cp.pubchem_cid or 'N/A'}, CAS: {cp.cas_number or 'N/A'}
SMILES: {cp.smiles or 'N/A'}, Formula: {cp.molecular_formula or 'N/A'}
MW: {f'{cp.molecular_weight:.2f} g/mol' if cp.molecular_weight is not None else 'N/A'}
Common Names: {', '.join(cp.common_names) if cp.common_names else 'N/A'}

PHYSICAL PROPERTIES (LLM-derived for recyclability assessment)
-----------------------------------------------------------------
Boiling Point: {f'{cp.boiling_point_celsius:.1f} °C' if cp.boiling_point_celsius is not None else 'N/A'}
Melting Point: {f'{cp.melting_point_celsius:.1f} °C' if cp.melting_point_celsius is not None else 'N/A'}
Density: {f'{cp.density_g_ml:.3f} g/mL' if cp.density_g_ml is not None else 'N/A'}
Vapor Pressure: {f'{cp.vapor_pressure_mmhg:.1f} mmHg' if cp.vapor_pressure_mmhg is not None else 'N/A'}
Azeotrope Notes: {cp.azeotrope_formation_notes or 'N/A'}
Thermal Stability: {cp.thermal_stability_notes or 'Not assessed'}
Reactivity/Stability: {cp.reactivity_stability_notes or 'Not assessed'}
Common Recovery Methods (LLM Suggestion): {', '.join(cp.common_recovery_methods) if cp.common_recovery_methods else 'N/A'}
"""
        # Removed generic score display from here as it's now reaction-specific
        # if cp.calculated_recyclability_score is not None: # This field is no longer on generic ChemicalProperties
        #     report += f"Generic Recyclability Score (0-100%): {cp.calculated_recyclability_score}%\n"
        #     report += f"Generic Recyclability Notes: {cp.calculated_recyclability_notes or 'N/A'}\n"

        report += f"""
SAFETY
------
Hazard Level: {cp.hazard_level.value if isinstance(cp.hazard_level, Enum) else cp.hazard_level} (Rating 1-10: {cp.hazard_rating if cp.hazard_rating is not None else 'N/A'})
Hazard Notes: {cp.notes_on_hazards or 'Not assessed'}
Corrosive: {'Y' if cp.is_corrosive else 'N' if cp.is_corrosive==False else 'Unk'} | Flammable: {'Y' if cp.is_flammable else 'N' if cp.is_flammable==False else 'Unk'} | Toxic: {'Y' if cp.is_toxic else 'N' if cp.is_toxic==False else 'Unk'}
GHS Statements:"""
        if cp.ghs_hazards:
            for gh in cp.ghs_hazards: report += f"\n  - {gh.get('statement', 'N/A')} (Pict: {gh.get('pictogram', 'N/A')})"
        else: report += " None found."
        report += f"""

SOLUBILITY & GREEN CHEM
-----------------------
Water Sol: {cp.solubility.get('water_solubility', 'Unk')} (Rating 1-10: {cp.solubility_rating if cp.solubility_rating is not None else 'N/A'})
Solubility Notes: {cp.solubility.get('notes_on_solubility', 'N/A')}
Organic Solvents: {', '.join(cp.solubility.get('organic_solvents_compatibility', [])) if cp.solubility.get('organic_solvents_compatibility') else 'N/A'}
Green Score (1-10): {cp.green_chemistry_score if cp.green_chemistry_score is not None else 'N/A'}
Green Chemistry Notes: {cp.notes_on_green_chemistry or 'Not assessed'}
Env. Impact Summary: {cp.environmental_impact or 'Not assessed'}

PRICING (INR/kg) & AVAILABILITY
-------------------------------
Price: {f'{cp.estimated_price_per_kg:.2f} {cp.price_currency}/kg' if cp.estimated_price_per_kg is not None and cp.price_currency else 'N/A'}"""
        if cp.supplier_info:
            for s_info in cp.supplier_info:
                report += f"\nSource: {s_info.get('name', 'N/A')}\n  Details: {s_info.get('availability', 'N/A')}"
                if s_info.get('location') and s_info.get('location') not in ["N/A", "Unknown Location"]: report += f"\n  Location: {s_info.get('location')}"
        else: report += "\nNo pricing source info."
        report += "\n\nHANDLING PRECAUTIONS\n--------------------"
        if cp.safety_notes:
            for note in cp.safety_notes: report += f"\n• {note}"
        else: report += "\nStandard lab safety. Consult SDS."
        return report


# --- Standalone Execution Logic ---
def perform_standalone_chemical_analysis(chemical_identifier: str,
                                         provider: str = DEFAULT_LLM_PROVIDER,
                                         api_key_override_openai: Optional[str] = None,
                                         api_key_override_perplexity: Optional[str] = None
                                         ) -> Tuple[Optional[ChemicalProperties], Optional[str]]:
    # (Code from your previous version - unchanged)
    print(f"\n>> Standalone Analysis: '{chemical_identifier}' (Provider: {provider}) <<", flush=True)
    current_openai_key = OPENAI_API_KEY
    current_perplexity_key = PERPLEXITY_API_KEY
    if api_key_override_openai and provider == "openai":
        current_openai_key = api_key_override_openai
    if api_key_override_perplexity and provider == "perplexity":
        current_perplexity_key = api_key_override_perplexity

    key_ok = True
    msg = "" # Initialize msg
    if provider == "openai":
        if not current_openai_key or "YOUR_OPENAI_API_KEY_HERE" in current_openai_key or "xxxx" in current_openai_key:
            key_ok = False; msg = "OpenAI API key invalid/placeholder. Please set your actual key."
        elif openai is None:
            key_ok = False; msg = "OpenAI library not available."
    elif provider == "perplexity":
        if not current_perplexity_key or "YOUR_PERPLEXITY_API_KEY_HERE" in current_perplexity_key or "xxxx" in current_perplexity_key:
            key_ok = False; msg = "Perplexity API key invalid/placeholder. Please set your actual key."
    
    if not key_ok:
        print(f"[Standalone Config Error] {msg}")
        return None, msg

    try:
        agent = ChemicalAnalysisAgent(
            openai_api_key=current_openai_key,
            perplexity_api_key=current_perplexity_key,
            llm_provider=provider
        )
        props = agent.analyze_chemical(chemical_identifier) # This now calculates generic recyclability internally
        if props:
            report = agent.generate_report(props)
            return props, report
        else:
            msg = f"Analysis for '{chemical_identifier}' returned no properties object."
            print(f"[Standalone Warn] {msg}")
            return None, msg
    except Exception as e:
        err_msg = f"Error analyzing '{chemical_identifier}': {e}"
        print(f"[Standalone Error] {err_msg}")
        # import traceback; traceback.print_exc() # Uncomment for full trace during debug
        return None, err_msg


# --- Pathway Processing Functions ---
def extract_chemicals_from_pathway_data(pathway_data: Dict) -> set:
    # (Code from your previous version - unchanged)
    unique_chemicals = set()
    for molecule_smiles, steps_data in pathway_data.items():
        for step_name, step_reactions in steps_data.items():
            if isinstance(step_reactions, list):
                for reaction_details in step_reactions:
                    if isinstance(reaction_details, dict):
                        agents_str = reaction_details.get("agents")
                        if agents_str and isinstance(agents_str, str):
                            for agent in agents_str.split('.'):
                                cleaned_agent = agent.strip()
                                if cleaned_agent: unique_chemicals.add(cleaned_agent)
                        solvents_str = reaction_details.get("solvents")
                        if solvents_str and isinstance(solvents_str, str):
                            for solvent in solvents_str.split('.'):
                                cleaned_solvent = solvent.strip()
                                if cleaned_solvent: unique_chemicals.add(cleaned_solvent)
    return unique_chemicals

def add_properties_to_pathway_data(
    agent_instance: ChemicalAnalysisAgent, 
    pathway_data: Dict, 
    chemical_properties_cache: Dict
) -> Dict:
    """
    Adds chemical properties to the pathway data, including a reaction-specific
    recyclability score and notes for agents and solvents.
    """
    for molecule_smiles_key, steps_data in pathway_data.items():
        for step_name, step_reactions in steps_data.items():
            if isinstance(step_reactions, list):
                for reaction_details in step_reactions: # This is the dict for one reaction step
                    if not isinstance(reaction_details, dict):
                        continue

                    # --- Process Agents ---
                    agents_str = reaction_details.get("agents")
                    if agents_str and isinstance(agents_str, str):
                        new_agents_properties_list = [] # Build a new list
                        for agent_name_original_query in agents_str.split('.'):
                            cleaned_agent_original_query = agent_name_original_query.strip()
                            if not cleaned_agent_original_query:
                                continue

                            # Get the base properties from the cache
                            # The cache stores DICTIONARY versions of ChemicalProperties
                            agent_props_dict_from_cache = chemical_properties_cache.get(cleaned_agent_original_query)
                            
                            # Start with a fresh dictionary for this agent in this reaction
                            # This avoids modifying the cache and ensures only relevant data is included
                            current_agent_output_dict = {}

                            if agent_props_dict_from_cache and "error" not in agent_props_dict_from_cache:
                                # Copy all generic properties from the cache
                                current_agent_output_dict = agent_props_dict_from_cache.copy()
                                
                                # Create a ChemicalProperties object to pass to the calculation method
                                # This ensures the calculation method works with the defined object structure
                                temp_agent_obj = ChemicalProperties.from_dict(agent_props_dict_from_cache)
                                
                                # Calculate reaction-specific score
                                r_score_percent, r_notes = agent_instance._calculate_reaction_specific_recyclability(
                                    temp_agent_obj, 
                                    reaction_details, 
                                    chemical_properties_cache 
                                )
                                
                                # Add/Overwrite with the reaction-specific score and notes
                                # Using the field names as defined in your ChemicalProperties dataclass
                                current_agent_output_dict['calculated_recyclability_score'] = r_score_percent
                                current_agent_output_dict['calculated_recyclability_notes'] = r_notes
                                
                                # Ensure original_query is present, as it's useful for debugging
                                if 'original_query' not in current_agent_output_dict:
                                    current_agent_output_dict['original_query'] = cleaned_agent_original_query

                            else: 
                                # If not in cache or error in cache, create a minimal error entry
                                current_agent_output_dict = {
                                    "error": f"Base props not found or error in cache for agent: {cleaned_agent_original_query}", 
                                    "original_query": cleaned_agent_original_query,
                                    "name": cleaned_agent_original_query, # Default name to original query
                                    "calculated_recyclability_score": None, # Ensure these fields exist even on error
                                    "calculated_recyclability_notes": "Cannot calculate: base properties missing."
                                }
                            new_agents_properties_list.append({cleaned_agent_original_query: current_agent_output_dict})
                        reaction_details["agents_properties"] = new_agents_properties_list
                    
                    # --- Process Solvents ---
                    solvents_str = reaction_details.get("solvents")
                    if solvents_str and isinstance(solvents_str, str):
                        new_solvents_properties_list = [] # Build a new list
                        for solvent_name_original_query in solvents_str.split('.'):
                            cleaned_solvent_original_query = solvent_name_original_query.strip()
                            if not cleaned_solvent_original_query:
                                continue

                            solvent_props_dict_from_cache = chemical_properties_cache.get(cleaned_solvent_original_query)
                            current_solvent_output_dict = {}

                            if solvent_props_dict_from_cache and "error" not in solvent_props_dict_from_cache:
                                current_solvent_output_dict = solvent_props_dict_from_cache.copy()
                                temp_solvent_obj = ChemicalProperties.from_dict(solvent_props_dict_from_cache)
                                
                                r_score_percent, r_notes = agent_instance._calculate_reaction_specific_recyclability(
                                    temp_solvent_obj,
                                    reaction_details,
                                    chemical_properties_cache
                                )
                                current_solvent_output_dict['calculated_recyclability_score'] = r_score_percent
                                current_solvent_output_dict['calculated_recyclability_notes'] = r_notes
                                
                                if 'original_query' not in current_solvent_output_dict:
                                     current_solvent_output_dict['original_query'] = cleaned_solvent_original_query
                            else:
                                current_solvent_output_dict = {
                                    "error": f"Base props not found or error in cache for solvent: {cleaned_solvent_original_query}", 
                                    "original_query": cleaned_solvent_original_query,
                                    "name": cleaned_solvent_original_query,
                                    "calculated_recyclability_score": None,
                                    "calculated_recyclability_notes": "Cannot calculate: base properties missing."
                                }
                            new_solvents_properties_list.append({cleaned_solvent_original_query: current_solvent_output_dict})
                        reaction_details["solvents_properties"] = new_solvents_properties_list
    return pathway_data

# --- Dummy File Creation (for testing local pricing files) ---
def setup_dummy_pricing_files(base_dir: str):
    # (Code from your previous version - unchanged)
    dummy_files_content = {
        "pricing_data.json": {"Water": ["O", 0.01], "Ethanol": ["CCO", 150.0], "Dichloromethane": ["ClCCl", 200.0], "I": 6900.0, "Iodine": 6900.0},
        "second_source.json": {
            "Triethylamine": ["CCN(CC)CC", 1452.0],  # Example price for Triethylamine
            "Hydrochloric acid": ["Cl", 50.0], 
            "Cobalt(II) chloride hexahydrate": 9500.0, # For Cobalt if identified
            "N-Ethyldiisopropylamine": ["CCN(C(C)C)C(C)C", 3300.0], # DIPEA / Hunig's Base
            "Tetrahydrofuran, AR grade": ["C1CCOC1", 1564.0] 
        },
        "sigma_source.json": {
            "TBTU": ["CN(C)C(On1nnc2ccccc21)=[N+](C)C", 25000.0],
            "N,N-Diisopropylethylamine": ["CCN(C(C)C)C(C)C", 5000.0], # DIPEA
            "Methanol": ["CO", 120.0], # For "CO" if it's misidentified as Methanol by SMILES
            "Tetrahydrofuran": ["C1CCOC1", 400.0]
            }
    }
    for fname, content in dummy_files_content.items():
        fpath = os.path.join(base_dir, fname)
        # Create only if not exists to avoid overwriting during tests, 
        # or always create if you want to ensure fresh dummy files.
        # For now, let's always create for consistency during testing.
        # if not os.path.exists(fpath): 
        try:
            with open(fpath, 'w', encoding='utf-8') as f: json.dump(content, f, indent=2)
            print(f"Created/Updated dummy pricing file: {fpath}")
        except IOError as e: print(f"Could not create/update dummy pricing file {fpath}: {e}")


# --- Main Execution Block ---
if __name__ == "__main__":
    print(f"Script running from: {current_script_dir}")
    print(f"Project root (for pricing/output files) set to: {project_root_dir}")

    # Ensure API keys are set before proceeding
    if "YOUR_OPENAI_API_KEY_HERE" in OPENAI_API_KEY and DEFAULT_LLM_PROVIDER == "openai":
        print("CRITICAL: OpenAI API Key is a placeholder. Please set your actual key in the script.")
        sys.exit(1)
    if "YOUR_PERPLEXITY_API_KEY_HERE" in PERPLEXITY_API_KEY and DEFAULT_LLM_PROVIDER == "perplexity":
        print("CRITICAL: Perplexity API Key is a placeholder. Please set your actual key in the script.")
        sys.exit(1)

    # 1. Setup dummy pricing files
    print("\n--- Setting up dummy pricing files (in script directory) ---")
    setup_dummy_pricing_files(project_root_dir)

    # 2. Define YOUR specific input pathway JSON data directly
    print("\n--- Using specific inline pathway input data ---")
    single_pathway_data_input = {
      "CCc1cc(-c2noc(-c3cc(CC(C)C)c(C)s3)n2)cc(C)c1CCC(=O)O": {
    "step1": [
      {
        "product": "CCc1cc(-c2noc(-c3cc(CC(C)C)c(C)s3)n2)cc(C)c1CCC(=O)O",
        "reactants": "Cc1sc(C(=O)O)cc1CC(C)C.CCc1cc(C(=N)NO)cc(C)c1CCC(=O)O",
        "agents": "CN(C)C(On1nnc2ccccc21)=[N+](C)C.F[B-](F)(F)F",
        "solvents": "ClCCl.CCN(C(C)C)C(C)C",
        "reaction_name": "1,2,4-Oxadiazole synthesis",
        "reaction_class": "O-heterocycle synthesis",
        "prediction_certainty": 0.9996649026870728,
        "rxn_string": "[CH2:1]([C:5]1[CH:6]=[C:7]([C:11]([OH:13])=O)[S:8][C:9]=1[CH3:10])[CH:2]([CH3:4])[CH3:3].CCN(C(C)C)C(C)C.CN(C(ON1N=NC2C=CC=CC1=2)=[N+](C)C)C.[B-](F)(F)(F)F.[CH2:45]([C:47]1[CH:52]=[C:51]([C:53](=[NH:56])[NH:54]O)[CH:50]=[C:49]([CH3:57])[C:48]=1[CH2:58][CH2:59][C:60]([OH:62])=[O:61])[CH3:46]>C(Cl)Cl>[CH2:45]([C:47]1[CH:52]=[C:51]([C:53]2[N:54]=[C:11]([C:7]3[S:8][C:9]([CH3:10])=[C:5]([CH2:1][CH:2]([CH3:3])[CH3:4])[CH:6]=3)[O:13][N:56]=2)[CH:50]=[C:49]([CH3:57])[C:48]=1[CH2:58][CH2:59][C:60]([OH:62])=[O:61])[CH3:46]",
        "conditions": {
          "temperature": 25.0,
          "yield": 15.6,
          "rxn_time": 16.0
        },
        "experimental_details": {
          "procedure": "To a solution of 4-isobutyl-5-methyl-thiophene-2-carboxylic acid (126 mg, 637 μmol) in DCM (5 mL), DIPEA (249 mg, 1.93 mmol) is added followed TBTU (202 mg, 628 μmol). The mixture is stirred at rt for 30 min before 3-[2-ethyl-4-(N-hydroxycarbamimidoyl)-6-methyl-phenyl]-propionic acid (159 mg, 637 μm...",
          "date_of_experiment": "",
          "extracted_from_file": "ord_dataset-afd812677c134591a99f46ce28de2524",
          "is_mapped": True
        }
      }
    ],
    "step2": [
      {
        "product": "CCc1cc(C(=N)NO)cc(C)c1CCC(=O)O",
        "reactants": "CCc1cc(C#N)cc(C)c1CCC(=O)O.NO",
        "agents": "Cl",
        "solvents": "CCO.CCN(CC)CC",
        "reaction_name": "Cyano to Hydroxyamidino",
        "reaction_class": "Other functional group interconversion",
        "prediction_certainty": 0.9997430443763732,
        "rxn_string": "[C:1]([C:3]1[CH:8]=[C:7]([CH3:9])[C:6]([CH2:10][CH2:11][C:12]([OH:14])=[O:13])=[C:5]([CH2:15][CH3:16])[CH:4]=1)#[N:2].CCN(CC)CC.Cl.[NH2:25][OH:26]>CCO>[CH2:15]([C:5]1[CH:4]=[C:3]([C:1](=[NH:2])[NH:25][OH:26])[CH:8]=[C:7]([CH3:9])[C:6]=1[CH2:10][CH2:11][C:12]([OH:14])=[O:13])[CH3:16]",
        "conditions": {
          "temperature": 25.0,
          "yield": 101.6,
          "rxn_time": None
        },
        "experimental_details": {
          "procedure": "To a solution of 3-(4-cyano-2-ethyl-6-methyl-phenyl)-propionic acid (10.0 g, 46.0 mmol) in EtOH (80 mL), NEt3 (13.97 g, 138.1 mmol) followed by hydroxylamine hydrochloride (6.40 g, 92.1 mmol) is added. The mixture is refluxed for 7 h before it is cooled to rt. The solvent is removed in vacuo. The re...",
          "date_of_experiment": "",
          "extracted_from_file": "ord_dataset-dc3bb1b1ac4e4229a3fc28fb559a9777",
          "is_mapped": True
        }
      },
      {
        "product": "CCc1cc(C(=N)NO)cc(C)c1CCC(=O)O",
        "reactants": "CCc1cc(C#N)cc(C)c1CCC(=O)O.NO",
        "agents": "CC(C)(C)[O-].Cl.[K+]",
        "solvents": "CO",
        "reaction_name": "Cyano to Hydroxyamidino",
        "reaction_class": "Other functional group interconversion",
        "prediction_certainty": 0.9961703419685364,
        "rxn_string": "CC(C)([O-])C.[K+].Cl.[NH2:8][OH:9].[C:10]([C:12]1[CH:17]=[C:16]([CH3:18])[C:15]([CH2:19][CH2:20][C:21]([OH:23])=[O:22])=[C:14]([CH2:24][CH3:25])[CH:13]=1)#[N:11]>CO>[CH2:24]([C:14]1[CH:13]=[C:12]([C:10](=[NH:11])[NH:8][OH:9])[CH:17]=[C:16]([CH3:18])[C:15]=1[CH2:19][CH2:20][C:21]([OH:23])=[O:22])[CH3:25]",
        "conditions": {
          "temperature": None,
          "yield": 81.1,
          "rxn_time": None
        },
        "experimental_details": {
          "procedure": "To an ice-cooled solution of 5-ethyl-4-hydroxy-3-methylbenzaldehyde (10.0 g, 60.9 mmol) in DCM (50 mL) and pyridine (15 mL), trifluoromethanesulfonic acid anhydride (18.9 g, 67 mmol) is added over a period of 20 min. Upon complete addition, the ice bath is removed and the reaction is stirred for fur...",
          "date_of_experiment": "",
          "extracted_from_file": "ord_dataset-b195433d5c354ddfb6cde0d53c41910f",
          "is_mapped": True
        }
      }
    ],
    "step3": [
      {
        "product": "CCc1cc(C#N)cc(C)c1CCC(=O)O",
        "reactants": "CCOC(=O)CCc1c(C)cc(C#N)cc1CC",
        "agents": "[Na+].[OH-]",
        "solvents": "O.C1CCOC1",
        "reaction_name": "CO2H-Et deprotection",
        "reaction_class": "RCO2H deprotections",
        "prediction_certainty": 0.9999641180038452,
        "rxn_string": "C([O:3][C:4](=[O:18])[CH2:5][CH2:6][C:7]1[C:12]([CH3:13])=[CH:11][C:10]([C:14]#[N:15])=[CH:9][C:8]=1[CH2:16][CH3:17])C>C1COCC1.[OH-].[Na+].O>[C:14]([C:10]1[CH:11]=[C:12]([CH3:13])[C:7]([CH2:6][CH2:5][C:4]([OH:18])=[O:3])=[C:8]([CH2:16][CH3:17])[CH:9]=1)#[N:15]",
        "conditions": {
          "temperature": None,
          "yield": 84.0,
          "rxn_time": None
        },
        "experimental_details": {
          "procedure": "A solution of 3-(4-cyano-2-ethyl-6-methyl-phenyl)-propionic acid ethyl ester (55.0 g, 224 mmol) in THF (220 mL) and 1N aq. NaOH solution (220 mL) is stirred at rt for 2 before it is diluted with water (200 mL) and extracted with DCM (2×200 mL). The aqueous phase is added to 32% aq. HCl solution (50 ...",
          "date_of_experiment": "",
          "extracted_from_file": "ord_dataset-dc3bb1b1ac4e4229a3fc28fb559a9777",
          "is_mapped": True
        }
      }
    ]
  }
    }

    all_unique_chemicals_to_analyze = extract_chemicals_from_pathway_data(single_pathway_data_input)
    print(f"Found {len(all_unique_chemicals_to_analyze)} unique chemicals to analyze: {all_unique_chemicals_to_analyze if all_unique_chemicals_to_analyze else 'None'}")

    main_chemical_agent = ChemicalAnalysisAgent(
        openai_api_key=OPENAI_API_KEY,
        perplexity_api_key=PERPLEXITY_API_KEY,
        llm_provider=DEFAULT_LLM_PROVIDER
    )

    chemical_properties_cache = {}

    if all_unique_chemicals_to_analyze:
        print(f"\n--- Analyzing unique chemicals first (to populate cache) ---")
        for chemical_name in sorted(list(all_unique_chemicals_to_analyze)): # chemical_name is original_query
            print(f"\n>> Requesting analysis for: {chemical_name} (Provider: {DEFAULT_LLM_PROVIDER}) <<", flush=True)
            try:
                # analyze_chemical now returns ChemicalProperties object with generic data
                props_obj = main_chemical_agent.analyze_chemical(chemical_name) 
                if props_obj and isinstance(props_obj, ChemicalProperties):
                    chemical_properties_cache[chemical_name] = props_obj.to_dict()
                    print(f"Successfully analyzed and cached generic properties for: {chemical_name} (Name: {props_obj.name})")
                    
                    if chemical_name == "CCO": # DEBUG CCO specifically after caching
                        print(f"--- DEBUG CACHE content for key '{chemical_name}': ---")
                        print(json.dumps(chemical_properties_cache.get(chemical_name), indent=2))

                else:
                    error_message = f"Analysis failed for {chemical_name}, no properties object returned."
                    chemical_properties_cache[chemical_name] = {"error": error_message, "original_query": chemical_name}
                    print(f"Warning/Error analyzing {chemical_name}: {error_message}")
            except Exception as e:
                error_message = f"Exception analyzing {chemical_name}: {e}"
                chemical_properties_cache[chemical_name] = {"error": error_message, "original_query": chemical_name}
                print(f"CRITICAL Error analyzing {chemical_name}: {error_message}")
                # import traceback; traceback.print_exc()

    # Now call add_properties_to_pathway_data, which will use the cache
    # and the agent_instance to calculate reaction-specific scores.
    augmented_pathway_data = add_properties_to_pathway_data(
        main_chemical_agent, single_pathway_data_input, chemical_properties_cache
    )
    
    output_file_augmented = os.path.join(project_root_dir, "augmented_reaction_specific_recyclability.json") # New output filename

    # ... (API Key Status and file saving as before) ...
    print("\n--- API Key Status ---")
    openai_key_ok = OPENAI_API_KEY and "xxxx" not in OPENAI_API_KEY and "YOUR_OPENAI_API_KEY_HERE" not in OPENAI_API_KEY
    perplexity_key_ok = PERPLEXITY_API_KEY and "xxxx" not in PERPLEXITY_API_KEY and "YOUR_PERPLEXITY_API_KEY_HERE" not in PERPLEXITY_API_KEY
    print(f"OpenAI API Key: {'SET and appears valid' if openai_key_ok else 'NOT SET or PLACEHOLDER'}")
    print(f"Perplexity API Key: {'SET and appears valid' if perplexity_key_ok else 'NOT SET or PLACEHOLDER'}")
    print(f"Default LLM Provider: {DEFAULT_LLM_PROVIDER.upper()}")
    
    provider_key_ok = (DEFAULT_LLM_PROVIDER == "openai" and openai_key_ok and openai is not None) or \
                      (DEFAULT_LLM_PROVIDER == "perplexity" and perplexity_key_ok)
    if not provider_key_ok:
        print(f"WARNING: Default provider ({DEFAULT_LLM_PROVIDER.upper()}) key invalid or library not loaded. LLM calls may fail.")
    if not openai_key_ok and not perplexity_key_ok:
        print("CRITICAL WARNING: NEITHER OpenAI NOR Perplexity API keys are properly set. LLM calls WILL FAIL.")

    if not all_unique_chemicals_to_analyze:
        print("\nNo chemicals to analyze, so no augmented data to save.")
    else:
        try:
            with open(output_file_augmented, 'w', encoding='utf-8') as f:
                json.dump(augmented_pathway_data, f, indent=2, ensure_ascii=False) # ensure_ascii for broader char support
            print(f"\nSuccessfully saved the AUGMENTED pathway data to: {output_file_augmented}")
        except IOError as e:
            print(f"Error: Could not write to output file - {output_file_augmented}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while saving the output file: {e}")

    print("\n--- Script Finished ---")
    if all_unique_chemicals_to_analyze:
        print(f"If successful, check output: {output_file_augmented}")
    print(f"Pricing files should be in: {project_root_dir}")