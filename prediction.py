import sys
import os
from dotenv import load_dotenv
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict # Added asdict
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
    hazard_level: HazardLevel = HazardLevel.UNKNOWN
    is_corrosive: Optional[bool] = None
    is_flammable: Optional[bool] = None
    is_toxic: Optional[bool] = None
    ghs_hazards: List[Dict[str, str]] = field(default_factory=list)
    green_chemistry_score: Optional[int] = None
    # NEW FIELD for green chemistry notes
    notes_on_green_chemistry: Optional[str] = None  # <--- ADD THIS LINE
    estimated_price_per_kg: Optional[float] = None
    price_currency: Optional[str] = None
    supplier_info: List[Dict[str, Any]] = field(default_factory=list)
    safety_notes: List[str] = field(default_factory=list)
    environmental_impact: Optional[str] = None
    hazard_rating: Optional[int] = None
    notes_on_hazards: Optional[str] = None 
    solubility_rating: Optional[int] = None
    # pubchem_full_json is correctly removed

    def to_dict(self):
        data = asdict(self)
        if isinstance(data.get('hazard_level'), HazardLevel):
            data['hazard_level'] = data['hazard_level'].value
        return data

# --- ChemicalAnalysisAgent Class (FULL DEFINITION)---
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
                except Exception as e:
                    print(f"Error initializing OpenAI client: {e}. OpenAI features will be disabled.")
                    self.openai_client = None
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
            self.pricing_sources.append({"name": source_display_name, "filename": filename, "data": {}})
        except json.JSONDecodeError as e:
            print(f"[Pricing Error] Parse error {filename}: {e}. Source unavailable.")
            self.pricing_sources.append({"name": source_display_name, "filename": filename, "data": {}})
        except Exception as e:
            print(f"[Pricing Error] Load error {filename}: {e}. Source unavailable.")
            self.pricing_sources.append({"name": source_display_name, "filename": filename, "data": {}})

    def _load_all_pricing_data(self) -> None:
        self.pricing_sources = []
        self._load_single_pricing_source(self.PRICING_FILE_PRIMARY, "Primary Local Data")
        self._load_single_pricing_source(self.PRICING_FILE_SECONDARY, "Secondary Local Data")
        self._load_single_pricing_source(self.PRICING_FILE_TERTIARY, "Tertiary Local Data (Sigma)")

    def _is_cas_number(self, identifier: str) -> bool:
        return bool(re.match(r'^\d{2,7}-\d{2}-\d$', identifier))

    def _is_smiles_like(self, identifier: str) -> bool:
        if not isinstance(identifier, str): return False
        if " " in identifier.strip() and len(identifier.strip().split()) > 1: return False
        smiles_chars = set("()[]=#@+-.0123456789" + "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        if len(set(identifier) - smiles_chars) < 3 and len(identifier) > 2:
            if any(c in identifier for c in "()[]=#@") or re.search(r"[cnospbrilq]", identifier, re.I):
                 if len(identifier) > 3 or any(c in identifier for c in "()[]=#@"):
                    if not re.fullmatch(r"[A-Za-z]{1,2}\d*", identifier) and not re.fullmatch(r"[A-Za-z]+\d*[A-Za-z]*", identifier):
                        return True
        return False

    def _get_pubchem_data(self, chemical_identifier: str) -> Tuple[Optional[pcp.Compound], Optional[Dict[str, Any]]]:
        print(f"[PubChem] Attrib. for '{chemical_identifier}'")
        compound: Optional[pcp.Compound] = None
        full_json_data: Optional[Dict[str, Any]] = None
        search_methods = []
        if self._is_cas_number(chemical_identifier):
            search_methods.append({'id': chemical_identifier, 'namespace': 'cas', 'type': 'CAS'})
        if self._is_smiles_like(chemical_identifier):
            search_methods.append({'id': chemical_identifier, 'namespace': 'smiles', 'type': 'SMILES'})
        if not any(m['id'] == chemical_identifier and m['type'] == 'Name' for m in search_methods):
             search_methods.append({'id': chemical_identifier, 'namespace': 'name', 'type': 'Name'})

        for method in search_methods:
            if compound: break
            print(f"[PubChem] Trying {method['type']} search for '{method['id']}'...")
            try:
                compounds = pcp.get_compounds(method['id'], method['namespace'])
                if compounds:
                    compound = compounds[0]
                    print(f"[PubChem] Found by {method['type']} '{method['id']}': CID {compound.cid}")
                    try:
                        print(f"[PubChem] Fetching full JSON record for CID {compound.cid}...")
                        response = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{compound.cid}/JSON")
                        if response.status_code == 404:
                            print("[PubChem] PUG View JSON not found, trying PUG REST JSON...")
                            response = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{compound.cid}/JSON")
                        response.raise_for_status()
                        full_json_data = response.json()
                        print(f"[PubChem] Successfully fetched full JSON for CID {compound.cid}.")
                    except requests.RequestException as e_json:
                        print(f"[PubChem Error] Failed to fetch full JSON for CID {compound.cid}: {e_json}")
                        full_json_data = None
                    except json.JSONDecodeError as e_decode:
                        print(f"[PubChem Error] Failed to parse full JSON for CID {compound.cid}: {e_decode}")
                        full_json_data = None
                    break
            except pcp.PubChemHTTPError as e_pcp:
                print(f"[PubChem] {method['type']} search failed for '{method['id']}': {e_pcp}")
            except requests.exceptions.RequestException as e_req:
                print(f"[PubChem] Network error during {method['type']} search for '{method['id']}': {e_req}")
            except Exception as e_gen:
                print(f"[PubChem] General error in {method['type']} search for '{method['id']}': {e_gen}")
        if not compound:
            print(f"[PubChem] No compound found for '{chemical_identifier}' after all attempts.")
        return compound, full_json_data

    def _extract_ghs_from_pubchem_json(self, pubchem_json: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
        ghs_hazards_list = []
        if not pubchem_json: return ghs_hazards_list
        try:
            record = pubchem_json.get('Record')
            if not record:
                if 'PC_Compounds' in pubchem_json:
                    print("[PubChem GHS] GHS extraction from PUG REST JSON is complex. Limited data expected.")
                return ghs_hazards_list
            sections = record.get('Section', [])
            safety_section = next((s for s in sections if s.get('TOCHeading') == 'Safety and Hazards'), None)
            if not safety_section: return ghs_hazards_list
            haz_id_section = next((s for s in safety_section.get('Section', []) if s.get('TOCHeading') == 'Hazards Identification'), None)
            if not haz_id_section: return ghs_hazards_list
            ghs_class_section = next((s for s in haz_id_section.get('Section', []) if s.get('TOCHeading') == 'GHS Classification'), None)
            if not ghs_class_section: return ghs_hazards_list
            information_list = ghs_class_section.get('Information', [])
            pictograms_map = {}
            for info_item_pict in information_list:
                if info_item_pict.get('Name') == 'Pictogram(s)':
                    value_pict = info_item_pict.get('Value')
                    if value_pict and isinstance(value_pict.get('StringWithMarkup'), list):
                        for swm_pict in value_pict['StringWithMarkup']:
                            if isinstance(swm_pict.get('Markup'), list):
                                for markup in swm_pict['Markup']:
                                    if markup.get('Type') == 'Icon':
                                        pict_url = markup.get('URL', 'N/A_URL_'+str(len(pictograms_map)))
                                        pict_desc = markup.get('Extra', 'Unknown Pictogram')
                                        pictograms_map[pict_url] = pict_desc
            for info_item in information_list:
                name, value_obj = info_item.get('Name'), info_item.get('Value')
                if not value_obj or not isinstance(value_obj, dict): continue
                if name == 'GHS Hazard Statements':
                    statements_data = value_obj.get('StringWithMarkup', [])
                    for swm_item in statements_data:
                        text = swm_item.get('String', '')
                        match = re.match(r"(H\d{3}[A-Za-z+]*)?\s*(?:\(.*\%\))?[:\s]*(.*?)(?:\s*\[(?:Warning|Danger).*?\].*)?$", text.strip())
                        if match:
                            h_code = match.group(1) or ""
                            statement_text = match.group(2).strip()
                            if statement_text:
                                pictogram_display = next(iter(pictograms_map.values()), "N/A") if pictograms_map else "N/A"
                                full_statement = f"{h_code}: {statement_text}".strip().lstrip(": ")
                                if not any(entry['statement'] == full_statement for entry in ghs_hazards_list):
                                    ghs_hazards_list.append({"pictogram": pictogram_display, "statement": full_statement})
            if ghs_hazards_list: print(f"[PubChem GHS] Extracted {len(ghs_hazards_list)} unique GHS entries from JSON.")
        except Exception as e: print(f"[PubChem GHS Error] Parsing GHS from JSON: {e}")
        return ghs_hazards_list

    def _search_single_local_source(self, source_data: Dict[str, Any], input_id_norm: str, iupac_norm: Optional[str], commons_norm: List[str], smiles: Optional[str]) -> Optional[Dict[str, Any]]:
        if not source_data: return None
        if smiles:
            if smiles in source_data:
                v_list = source_data[smiles]
                if isinstance(v_list, (float, int)):
                    return {"price": float(v_list), "currency": "INR", "location": "Unknown Location", "source_name_in_json": smiles, "match_type": "SMILES Key"}
                elif isinstance(v_list, list) and len(v_list) >= 1 and isinstance(v_list[0], (float, int)):
                    price = v_list[0]
                    loc = v_list[1] if len(v_list) >=2 and isinstance(v_list[1], str) else "Unknown Location"
                    return {"price": float(price), "currency": "INR", "location": loc, "source_name_in_json": smiles, "match_type": "SMILES Key"}
            for k, v_list in source_data.items():
                if isinstance(v_list, list) and len(v_list) >= 2 and isinstance(v_list[0], str) and v_list[0] == smiles:
                    price = v_list[1] if isinstance(v_list[1], (float, int)) else None
                    loc = v_list[2] if len(v_list) >=3 and isinstance(v_list[2], str) else "Unknown Location"
                    if price is not None:
                        return {"price": float(price), "currency": "INR", "location": loc, "source_name_in_json": k, "match_type": "SMILES Value"}
        names_to_check = set(commons_norm)
        if iupac_norm: names_to_check.add(iupac_norm)
        names_to_check.add(input_id_norm)
        matches = []
        for k, v_list in source_data.items():
            key_norm = k.lower().strip()
            base_key_norm = re.match(r"^(.*?)\s*\(", key_norm).group(1).strip() if re.match(r"^(.*?)\s*\(", key_norm) else key_norm
            price_val, loc_val = None, "Unknown Location"
            if isinstance(v_list, (float, int)): price_val = v_list
            elif isinstance(v_list, list) and len(v_list) >= 1:
                if isinstance(v_list[0], (float, int)):
                    price_val = v_list[0]
                    if len(v_list) >= 2 and isinstance(v_list[1], str): loc_val = v_list[1]
                elif len(v_list) >= 2 and isinstance(v_list[1], (float, int)):
                    price_val = v_list[1]
                    if len(v_list) >= 3 and isinstance(v_list[2], str): loc_val = v_list[2]
            if price_val is not None:
                for name_check_norm in names_to_check:
                    if name_check_norm == base_key_norm or name_check_norm == key_norm:
                        matches.append({"price": float(price_val), "currency": "INR", "location": loc_val,
                                        "source_name_in_json": k, "match_type": "Exact Name", "len": len(name_check_norm)})
        if matches:
            best_match = sorted(matches, key=lambda x: x["len"], reverse=True)[0]
            del best_match["len"]
            return best_match
        return None

    def _get_pricing_from_all_local_sources(self, in_id: str, iupac: Optional[str], commons: List[str], smiles: Optional[str]) -> Optional[Dict[str, Any]]:
        in_id_n = in_id.lower().strip()
        iupac_n = iupac.lower().strip() if iupac else None
        commons_n = [c.lower().strip() for c in commons if c]
        for src in self.pricing_sources:
            if not src.get("data"): continue
            print(f"[Pricing] Searching in {src['name']} for '{in_id}' (SMILES: {smiles})...")
            match = self._search_single_local_source(src["data"], in_id_n, iupac_n, commons_n, smiles)
            if match:
                print(f"[Pricing] Found '{in_id}' in {src['name']}. Price: {match['price']} {match['currency']}")
                match.update({"source_file_display_name": src['name'], "source_filename": src['filename']})
                return match
        print(f"[Pricing] '{in_id}' (SMILES: {smiles}, IUPAC: {iupac}, Names: {commons}) not found in local data.")
        return None

    def _get_perplexity_completion(self, system_prompt_content: str, user_prompt_content: str) -> Optional[str]:
        if not self.perplexity_api_key_val or "YOUR_PERPLEXITY_API_KEY_HERE" in self.perplexity_api_key_val or "xxxx" in self.perplexity_api_key_val:
            print("[Perplexity Error] API key not configured or is a placeholder.")
            return None
        payload = {"model": "llama-3-sonar-large-32k-online", "messages": [{"role": "system", "content": system_prompt_content}, {"role": "user", "content": user_prompt_content}]}
        headers = {"Authorization": f"Bearer {self.perplexity_api_key_val}", "Content-Type": "application/json", "Accept": "application/json"}
        try:
            response = requests.post("https://api.perplexity.ai/chat/completions", json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.RequestException as e:
            print(f"[Perplexity API Error] Request failed: {e}")
            if hasattr(e, 'response') and e.response is not None: print(f"Response: {e.response.status_code}, {e.response.text[:200]}")
        except (KeyError, IndexError, TypeError) as e:
            print(f"[Perplexity API Error] Parse response: {e}")
        return None

    def _get_openai_completion(self, system_prompt_content: str, user_prompt_content: str) -> Optional[str]:
        if not self.openai_client:
            print("[OpenAI Error] Client not initialized.")
            return None
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "system", "content": system_prompt_content}, {"role": "user", "content": user_prompt_content}],
                temperature=0.2
            )
            return response.choices[0].message.content
        except openai.APIConnectionError as e: print(f"[OpenAI API Error] Connection error: {e}")
        except openai.RateLimitError as e: print(f"[OpenAI API Error] Rate limit exceeded: {e}")
        except openai.AuthenticationError as e: print(f"[OpenAI API Error] Authentication error: {e}")
        except openai.APIStatusError as e: print(f"[OpenAI API Error] Status error {e.status_code}: {e.response}")
        except Exception as e:
            print(f"[OpenAI API Error or other] {type(e).__name__}: {e}")
        return None

    def _get_llm_completion(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        if self.llm_provider == "openai": return self._get_openai_completion(system_prompt, user_prompt)
        elif self.llm_provider == "perplexity": return self._get_perplexity_completion(system_prompt, user_prompt)
        print(f"[LLM Error] Provider '{self.llm_provider}' not supported or configured.")
        return None

    def _get_llm_derived_pricing(self, chemical_name: str, smiles: Optional[str], formula: Optional[str], cid: Optional[int]) -> Optional[Dict[str, Any]]:
        context_parts = [f"Chemical: {chemical_name}"]
        if smiles: context_parts.append(f"SMILES: {smiles}")
        if formula: context_parts.append(f"Formula: {formula}")
        if cid: context_parts.append(f"PubChem CID: {cid}")
        system_prompt = "You are a chemical market analyst. Provide price estimations in JSON format. Respond *only* with the JSON object."
        user_prompt = f"""{", ".join(context_parts)}
Estimate bulk price in INR/kg or USD/kg. For research/small industrial scale. Provide numerical estimate or range (e.g., "10000-15000" INR, "100-150" USD). Prioritize INR.
JSON: {{"estimated_price_per_kg_inr": float_or_range_or_null, "estimated_price_per_kg_usd": float_or_range_or_null, "price_confidence": "low/medium/high", "price_basis_notes": "Brief notes..."}}
Respond ONLY with the JSON object.
"""
        llm_response_content = self._get_llm_completion(system_prompt, user_prompt)
        if not llm_response_content:
            print(f"[Pricing LLM] No content from LLM for '{chemical_name}'.")
            return None
        try:
            json_str_to_parse = llm_response_content
            match_json = re.search(r"```json\s*([\s\S]*?)\s*```", llm_response_content, re.IGNORECASE)
            if match_json: json_str_to_parse = match_json.group(1).strip()
            else:
                json_match_direct = re.search(r"^\s*(\{[\s\S]*?\})\s*$", llm_response_content.strip())
                if json_match_direct: json_str_to_parse = json_match_direct.group(1)
            if not json_str_to_parse: return None
            data = json.loads(json_str_to_parse)
            price_usd = data.get("estimated_price_per_kg_usd")
            price_inr = data.get("estimated_price_per_kg_inr")
            final_price_inr, llm_currency = None, None
            def parse_val(v):
                if isinstance(v, (int, float)): return float(v)
                if isinstance(v, str):
                    cv = v.replace(',', '')
                    m = re.match(r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)", cv)
                    if m: return (float(m.group(1)) + float(m.group(2))) / 2
                    try: return float(cv)
                    except ValueError: return None
                return None
            if price_inr is not None:
                pv = parse_val(price_inr)
                if pv is not None: final_price_inr, llm_currency = pv, "INR"
            if final_price_inr is None and price_usd is not None:
                pv = parse_val(price_usd)
                if pv is not None: final_price_inr, llm_currency = pv * self.USD_TO_INR_RATE, "USD"
            if final_price_inr is not None:
                return {"price_inr": final_price_inr, "currency_llm_provided": llm_currency,
                        "raw_llm_price_value": price_inr if llm_currency == "INR" else price_usd,
                        "confidence": data.get("price_confidence"), "basis_notes": data.get("price_basis_notes"),
                        "source_type": f"LLM ({self.llm_provider})"}
            else:
                return {"price_inr": None, "currency_llm_provided": None, "raw_llm_price_value": None,
                        "confidence": data.get("price_confidence", "low_no_price"),
                        "basis_notes": data.get("price_basis_notes", "LLM no price/unrecognized format."),
                        "source_type": f"LLM ({self.llm_provider})"}
        except json.JSONDecodeError as e: print(f"[Pricing LLM Error] JSONDecodeError for '{chemical_name}': {e}. String: '{json_str_to_parse[:200]}...'")
        except Exception as e: print(f"[Pricing LLM Error] General error for '{chemical_name}': {e}.")
        return None

    def _parse_int_score(self, val: Any, field: str) -> Optional[int]:
        if val is None: return None
        try: return int(val)
        except (ValueError, TypeError): print(f"[LLM Parse Warn] Non-int for {field}: '{val}'"); return None

    def _get_llm_derived_properties(self, name: str, formula: Optional[str], smiles: Optional[str],
                                   cid: Optional[int], cas: Optional[str], mw: Optional[float],
                                   iupac: Optional[str]) -> Dict[str, Any]:
        print(f"[LLM Props] Querying LLM ({self.llm_provider}) for derived properties of '{name}'.")
        # ... (context building remains the same)
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

        guess_instr = "" # Keep this as is

        system_prompt = "You are a chemical safety, properties, and environmental expert. Provide accurate assessments in JSON format. Respond *only* with the JSON object, no other text or explanations. If guessing core chemical identity, clearly indicate it (e.g., field ends with '_llm_guess')."
        user_prompt = f"""{full_context}
Provide analysis in JSON. Ratings 1-10 (10=high/extreme/excellent) or null. If core ID (IUPAC, formula, MW, CAS, SMILES) unknown, guess it.
Respond ONLY with the JSON object.
{{
    {guess_instr}
    "solubility": {{
        "water_solubility": "Water Soluble/Organic Soluble/Poorly Soluble/Insoluble/Unknown",
        "organic_solvents_compatibility": ["list of types like alcohols, ethers, hydrocarbons"],
        "solubility_rating": "integer 1-10 or null (e.g., 1 for very poor, 10 for excellent)",
        "notes_on_solubility": "Provide a brief explanation for the solubility_rating, considering factors like polarity, functional groups, and intermolecular forces. If rating is high, explain why. If low, explain why."
    }},
    "hazards": {{
        "corrosive": true/false/null,
        "flammable": true/false/null,
        "toxic": true/false/null,
        "carcinogenic_suspected": true/false/null,
        "overall_hazard_level": "Low/Moderate/High/Extreme/Unknown",
        "hazard_rating": "integer 1-10 or null (e.g., 1 for very low hazard, 10 for extreme hazard)",
        "notes_on_hazards": "Provide a brief explanation for the hazard_rating, considering factors like GHS statements, reactivity, toxicity, and physical hazards. Justify the rating.", // <--- NEWLY ADDED
        "environmental_hazard_notes": "brief notes on environmental risks (distinct from overall hazard notes)",
        "ghs_info_llm": [
            {{ "pictogram_description": "Name of GHS pictogram (e.g., Exclamation Mark)", "h_code": "HXXX (e.g. H302)", "h_statement": "Full GHS hazard statement text"}}
        ]
    }},
    "safety_precautions": ["list of key safety measures when handling"],
    "storage_recommendations": "brief storage condition recommendations",
    "disposal_considerations": "brief notes on proper disposal",
    "green_chemistry": {{
        "renewable_feedstock_potential": "yes/no/partially/unknown",
        "atom_economy_typical_reactions": "high/moderate/low/varies/unknown",
        "biodegradability_assessment": "readily/partially/poorly/non-biodegradable/unknown",
        "energy_efficiency_synthesis": "typically high/moderate/low/varies/unknown",
        "waste_generation_typical_reactions": "typically high/moderate/low/varies/unknown",
        "notes_on_green_chemistry": "Provide brief qualitative notes regarding its green chemistry profile, considering factors like synthesis, use, and disposal. Justify the overall_score.",
        "overall_score": "integer 1-10 or null"
    }},
    "environmental_impact_summary": "brief overall assessment of environmental effects (can summarize environmental_hazard_notes)"
}}
If 'ghs_info_llm' cannot be determined, use empty list []. For other missing values, use null or "Unknown" as appropriate for the field type. Ensure overall_hazard_level and water_solubility use specified capitalized options.
Be specific and provide justification for ratings in the notes fields.
"""
        llm_response_content = self._get_llm_completion(system_prompt, user_prompt)
        default_empty_response = {
            "solubility": {"water_solubility": SolubilityType.UNKNOWN.value, "solubility_rating": None, "notes_on_solubility": None},
            "hazards": {"overall_hazard_level": HazardLevel.UNKNOWN.value, "hazard_rating": None, "notes_on_hazards": None, "environmental_hazard_notes": None, "ghs_info_llm": []},
            "safety_precautions": [],
            "green_chemistry": {"overall_score": None, "notes_on_green_chemistry": None},
            "environmental_impact_summary": "Assessment unavailable due to LLM error."
        }
        if not llm_response_content: return default_empty_response

        try:
            json_str = llm_response_content
            match_json = re.search(r"```json\s*([\s\S]*?)\s*```", llm_response_content, re.IGNORECASE)
            if match_json: json_str = match_json.group(1).strip()
            else:
                json_match_direct = re.search(r"^\s*(\{[\s\S]*?\})\s*$", llm_response_content.strip())
                if json_match_direct: json_str = json_match_direct.group(1)
            
            parsed_data = json.loads(json_str)
            
            # Ensure nested dictionaries and new fields exist with None as default if missing from LLM response
            if "solubility" not in parsed_data: parsed_data["solubility"] = {}
            parsed_data["solubility"].setdefault("notes_on_solubility", None)
            
            if "hazards" not in parsed_data: parsed_data["hazards"] = {}
            parsed_data["hazards"].setdefault("notes_on_hazards", None)
            parsed_data["hazards"].setdefault("environmental_hazard_notes", None) # Already prompted, ensure default
            
            if "green_chemistry" not in parsed_data: parsed_data["green_chemistry"] = {}
            parsed_data["green_chemistry"].setdefault("notes_on_green_chemistry", None)

            return parsed_data
        except json.JSONDecodeError as e:
            print(f"[LLM Parse Error] Props for {name}. Raw (after potential strip): {json_str[:500]}... Error: {e}")
            return default_empty_response

    def analyze_chemical(self, chemical_identifier: str) -> ChemicalProperties:
        props = ChemicalProperties(name=chemical_identifier, original_query=chemical_identifier)
        pubchem_compound, pubchem_full_json_data = self._get_pubchem_data(chemical_identifier)
        current_name_for_llm = chemical_identifier

        if pubchem_compound:
            props.name = pubchem_compound.iupac_name or (pubchem_compound.synonyms[0] if pubchem_compound.synonyms else chemical_identifier)
            current_name_for_llm = props.name
            props.iupac_name = pubchem_compound.iupac_name
            unique_common_names = {name.strip() for name in (pubchem_compound.synonyms or [])}
            if chemical_identifier.strip() not in unique_common_names and \
               (not props.iupac_name or chemical_identifier.strip().lower() != props.iupac_name.strip().lower()):
                unique_common_names.add(chemical_identifier.strip())
            props.common_names = sorted(list(unique_common_names))[:10]
            props.molecular_formula = pubchem_compound.molecular_formula
            props.molecular_weight = float(pubchem_compound.molecular_weight) if pubchem_compound.molecular_weight else None
            cas_from_syns = [s for s in (pubchem_compound.synonyms or []) if self._is_cas_number(s)]
            if hasattr(pubchem_compound, 'cas') and pubchem_compound.cas:
                 props.cas_number = pubchem_compound.cas[0] if isinstance(pubchem_compound.cas, list) else str(pubchem_compound.cas)
            elif cas_from_syns: props.cas_number = cas_from_syns[0]
            props.smiles = pubchem_compound.canonical_smiles
            props.pubchem_cid = pubchem_compound.cid
            props.ghs_hazards = self._extract_ghs_from_pubchem_json(pubchem_full_json_data)

        llm_derived = self._get_llm_derived_properties(current_name_for_llm, props.molecular_formula,
                                                           props.smiles, props.pubchem_cid, props.cas_number,
                                                           props.molecular_weight, props.iupac_name)
        if not pubchem_compound:
            props.iupac_name = props.iupac_name or llm_derived.get("iupac_name_llm_guess")
            if props.name == chemical_identifier and props.iupac_name:
                props.name = props.iupac_name
                current_name_for_llm = props.name
            props.molecular_formula = props.molecular_formula or llm_derived.get("molecular_formula_llm_guess")
            mw_g = llm_derived.get("molecular_weight_llm_guess")
            if mw_g is not None:
                try: props.molecular_weight = props.molecular_weight or float(mw_g)
                except (ValueError, TypeError): print(f"LLM MW guess '{mw_g}' not float.")
            cas_g = llm_derived.get("cas_number_llm_guess")
            if cas_g and self._is_cas_number(cas_g): props.cas_number = props.cas_number or cas_g
            props.smiles = props.smiles or llm_derived.get("smiles_llm_guess")

        # --- Populate from LLM derived info ---

        # Solubility: update existing props.solubility dict with LLM's solubility dict
        llm_solubility_info = llm_derived.get("solubility", {})
        props.solubility.update(llm_solubility_info) # Merges all keys from LLM's solubility dict
        props.solubility_rating = self._parse_int_score(llm_solubility_info.get("solubility_rating"), "Solubility rating") or props.solubility_rating
        # If you want to ensure notes_on_solubility is specifically set if not already by update:
        if "notes_on_solubility" in llm_solubility_info:
            props.solubility["notes_on_solubility"] = llm_solubility_info["notes_on_solubility"]


        # Hazards:
        llm_hazards_info = llm_derived.get("hazards", {})
        haz_lvl_str = llm_hazards_info.get("overall_hazard_level", props.hazard_level.value if isinstance(props.hazard_level, Enum) else "unknown").lower()
        props.hazard_level = HazardLevel[haz_lvl_str.upper()] if haz_lvl_str.upper() in HazardLevel.__members__ else HazardLevel.UNKNOWN
        props.is_corrosive = llm_hazards_info.get("corrosive", props.is_corrosive)
        props.is_flammable = llm_hazards_info.get("flammable", props.is_flammable)
        props.is_toxic = llm_hazards_info.get("toxic", props.is_toxic)
        props.hazard_rating = self._parse_int_score(llm_hazards_info.get("hazard_rating"), "Hazard rating") or props.hazard_rating
        props.notes_on_hazards = llm_hazards_info.get("notes_on_hazards")

        props.environmental_impact = llm_hazards_info.get("environmental_hazard_notes", props.environmental_impact)
        
        # Environmental hazard notes from LLM
        # Check if 'environmental_impact' is already set (e.g. from a previous source or if it's a primary field)
        # If you want environmental_hazard_notes from LLM to be the primary source for props.environmental_impact:
        # if llm_hazards_info.get("environmental_hazard_notes"):
        #     props.environmental_impact = llm_hazards_info.get("environmental_hazard_notes", props.environmental_impact)
        # OR if you want it as a separate field, add 'environmental_hazard_notes' to ChemicalProperties dataclass
        # and assign it here: props.environmental_hazard_notes = llm_hazards_info.get("environmental_hazard_notes")

        if not props.ghs_hazards and "ghs_info_llm" in llm_hazards_info: # GHS assignment
            for item in llm_hazards_info.get("ghs_info_llm", []):
                if isinstance(item, dict) and item.get("h_statement"):
                    new_stmt = f"{item.get('h_code', '')}: {item.get('h_statement', '')}".strip().lstrip(": ")
                    if not any(e['statement'] == new_stmt for e in props.ghs_hazards):
                         props.ghs_hazards.append({"pictogram": item.get("pictogram_description", "N/A"), "statement": new_stmt})
            if props.ghs_hazards: print("[GHS] Used LLM-derived GHS.")

        props.safety_notes = llm_derived.get("safety_precautions", props.safety_notes)
        
        gc_info = llm_derived.get("green_chemistry", {})
        props.green_chemistry_score = self._parse_int_score(gc_info.get("overall_score"), "GC score") or props.green_chemistry_score
        props.notes_on_green_chemistry = gc_info.get("notes_on_green_chemistry")
        
        # Overwrite props.environmental_impact if a specific summary is provided by LLM at top level
        props.environmental_impact = llm_derived.get("environmental_impact_summary", props.environmental_impact)


        # Pricing (remains the same)
        local_price = self._get_pricing_from_all_local_sources(chemical_identifier, props.iupac_name, props.common_names, props.smiles)
        if local_price:
            props.estimated_price_per_kg = local_price.get("price")
            props.price_currency = local_price.get("currency", "INR")
            props.supplier_info = [{"name": f"Local DB: {local_price.get('source_file_display_name','N/A')} ({local_price.get('source_filename','N/A')} - Key: '{local_price.get('source_name_in_json','N/A')}')",
                                    "availability": (f"Price: {props.estimated_price_per_kg:.2f} {props.price_currency}. Match: {local_price.get('match_type','N/A')}."),
                                    "location": local_price.get('location', 'Unknown Location'), "source_type": "Local JSON"}]
        else:
            llm_price = self._get_llm_derived_pricing(current_name_for_llm, props.smiles, props.molecular_formula, props.pubchem_cid)
            if llm_price and llm_price.get("price_inr") is not None:
                props.estimated_price_per_kg = llm_price["price_inr"]
                props.price_currency = "INR"
                raw_val = llm_price.get("raw_llm_price_value", "N/A")
                llm_curr = llm_price.get("currency_llm_provided", "N/A")
                avail_details = f"Est. Price: {props.estimated_price_per_kg:.2f} {props.price_currency}/kg. "
                if llm_curr and llm_curr != "INR": avail_details += f"(LLM: {raw_val} {llm_curr}, converted). "
                else: avail_details += f"(LLM: {raw_val} {llm_curr}). "
                avail_details += f"Conf: {llm_price.get('confidence','N/A')}. Basis: {llm_price.get('basis_notes','N/A')}"
                props.supplier_info = [{"name": f"LLM Estimation ({self.llm_provider})", "availability": avail_details,
                                        "location": "Global Market (Est.)", "source_type": f"LLM ({self.llm_provider})"}]
            else:
                props.estimated_price_per_kg, props.price_currency = None, None
                avail_note = "Not in local DBs. "
                if llm_price: avail_note += (f"LLM ({self.llm_provider}) consulted: Price difficult. "
                                          f"Conf: {llm_price.get('confidence','N/A')}. Basis: {llm_price.get('basis_notes','N/A')}")
                else: avail_note += f"LLM ({self.llm_provider}) call failed/no parsable pricing."
                props.supplier_info = [{"name": "No Definitive Pricing Data", "availability": avail_note, "location": "N/A", "source_type": "None"}]
        return props

    def generate_report(self, cp: ChemicalProperties) -> str:
        report = f"""
CHEMICAL ANALYSIS REPORT ({self.llm_provider.upper()} LLM Used)
========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

IDENTIFICATION
--------------
Query: {cp.original_query or 'N/A'}, Analyzed As: {cp.name}
IUPAC: {cp.iupac_name or 'N/A'}, CID: {cp.pubchem_cid or 'N/A'}, CAS: {cp.cas_number or 'N/A'}
SMILES: {cp.smiles or 'N/A'}, Formula: {cp.molecular_formula or 'N/A'}
MW: {f'{cp.molecular_weight:.2f} g/mol' if cp.molecular_weight is not None else 'N/A'}
Common Names: {', '.join(cp.common_names) if cp.common_names else 'N/A'}

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
Organic Solvents: {', '.join(cp.solubility.get('organic_solvents_compatibility', ['Unk']))}
Green Score (1-10): {cp.green_chemistry_score if cp.green_chemistry_score is not None else 'N/A'}
Green Chemistry Notes: {cp.notes_on_green_chemistry or 'Not assessed'}
Env. Impact: {cp.environmental_impact or 'Not assessed'}

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
    print(f"\n>> Standalone Analysis: '{chemical_identifier}' (Provider: {provider}) <<", flush=True)
    current_openai_key = OPENAI_API_KEY
    current_perplexity_key = PERPLEXITY_API_KEY
    if api_key_override_openai and provider == "openai":
        current_openai_key = api_key_override_openai
    if api_key_override_perplexity and provider == "perplexity":
        current_perplexity_key = api_key_override_perplexity

    key_ok = True
    if provider == "openai":
        if not current_openai_key or "YOUR_OPENAI_API_KEY_HERE" in current_openai_key or "xxxx" in current_openai_key:
            key_ok = False; msg = "OpenAI API key invalid/placeholder."
        elif openai is None:
            key_ok = False; msg = "OpenAI library not available."
    elif provider == "perplexity":
        if not current_perplexity_key or "YOUR_PERPLEXITY_API_KEY_HERE" in current_perplexity_key or "xxxx" in current_perplexity_key:
            key_ok = False; msg = "Perplexity API key invalid/placeholder."
    if not key_ok:
        print(f"[Standalone Config Error] {msg}"); return None, msg

    try:
        agent = ChemicalAnalysisAgent(
            openai_api_key=current_openai_key,
            perplexity_api_key=current_perplexity_key,
            llm_provider=provider
        )
        props = agent.analyze_chemical(chemical_identifier)
        if props:
            report = agent.generate_report(props)
            return props, report
        else:
            msg = f"Analysis for '{chemical_identifier}' returned no properties."
            print(f"[Standalone Warn] {msg}")
            return None, msg
    except Exception as e:
        err_msg = f"Error analyzing '{chemical_identifier}': {e}"
        print(f"[Standalone Error] {err_msg}")
        # import traceback; traceback.print_exc() # Uncomment for full trace during debug
        return None, err_msg

# --- Pathway Processing Functions ---
def extract_chemicals_from_pathway_data(pathway_data: Dict) -> set:
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

def add_properties_to_pathway_data(pathway_data: Dict, chemical_properties_cache: Dict) -> Dict:
    for molecule_smiles, steps_data in pathway_data.items():
        for step_name, step_reactions in steps_data.items():
            if isinstance(step_reactions, list):
                for reaction_details in step_reactions:
                    if isinstance(reaction_details, dict):
                        agents_str = reaction_details.get("agents")
                        if agents_str and isinstance(agents_str, str):
                            reaction_details["agents_properties"] = []
                            for agent_name in agents_str.split('.'):
                                cleaned_agent = agent_name.strip()
                                if cleaned_agent:
                                    props = chemical_properties_cache.get(cleaned_agent, {"error": f"Props not found for agent: {cleaned_agent}"})
                                    reaction_details["agents_properties"].append({cleaned_agent: props})
                        solvents_str = reaction_details.get("solvents")
                        if solvents_str and isinstance(solvents_str, str):
                            reaction_details["solvents_properties"] = []
                            for solvent_name in solvents_str.split('.'):
                                cleaned_solvent = solvent_name.strip()
                                if cleaned_solvent:
                                    props = chemical_properties_cache.get(cleaned_solvent, {"error": f"Props not found for solvent: {cleaned_solvent}"})
                                    reaction_details["solvents_properties"].append({cleaned_solvent: props})
    return pathway_data

# --- Dummy File Creation (for testing local pricing files) ---
def setup_dummy_pricing_files(base_dir: str):
    # Simplified for brevity. Ensure your ChemicalAnalysisAgent uses these filenames.
    dummy_files_content = {
        "pricing_data.json": {"Water": ["O", 0.01], "Ethanol": ["CCO", 150.0], "Dichloromethane": ["ClCCl", 200.0]},
        "second_source.json": {"Triethylamine": ["CCN(CC)CC", 300.0], "Hydrochloric acid": ["Cl", 50.0]},
        "sigma_source.json": {
            "TBTU": ["CN(C)C(On1nnc2ccccc21)=[N+](C)C", 25000.0],
            "N,N-Diisopropylethylamine": ["CCN(C(C)C)C(C)C", 5000.0],
            "Methanol": ["CO", 120.0],
            "Tetrahydrofuran": ["C1CCOC1", 400.0]
            }
    }
    for fname, content in dummy_files_content.items():
        fpath = os.path.join(base_dir, fname)
        if not os.path.exists(fpath):
            try:
                with open(fpath, 'w', encoding='utf-8') as f: json.dump(content, f, indent=2)
                print(f"Created dummy pricing file: {fpath}")
            except IOError as e: print(f"Could not create dummy pricing file {fpath}: {e}")
        # else: print(f"Dummy pricing file already exists: {fpath}") # Optional: uncomment to see existing

# --- Main Execution Block ---
if __name__ == "__main__":
    print(f"Script running from: {current_script_dir}")
    print(f"Project root (for pricing/output files) set to: {project_root_dir}")

    # 1. Setup dummy pricing files
    print("\n--- Setting up dummy pricing files (in script directory) ---")
    setup_dummy_pricing_files(project_root_dir)

    # 2. Define YOUR specific input pathway JSON data directly
    print("\n--- Using specific inline pathway input data ---")
    single_pathway_data_input = {
    "CN(Cc1ccc(OCC(=O)O)c(-c2cccc(C#N)c2)c1)S(=O)(=O)c1ccc(F)cc1": {
    "step1": [
      {
        "product": "CN(Cc1ccc(OCC(=O)O)c(-c2cccc(C#N)c2)c1)S(=O)(=O)c1ccc(F)cc1",
        "reactants": "CN(Cc1ccc(OCC(=O)OC(C)(C)C)c(-c2cccc(C#N)c2)c1)S(=O)(=O)c1ccc(F)cc1",
        "agents": "",
        "solvents": "ClCCl.O=C(O)C(F)(F)F",
        "reaction_name": "CO2H-tBu deprotection",
        "reaction_class": "RCO2H deprotections",
        "prediction_certainty": 0.9998878240585328,
        "rxn_string": "[C:1]([C:3]1[CH:4]=[C:5]([C:9]2[CH:23]=[C:22]([CH2:24][N:25]([CH3:36])[S:26]([C:29]3[CH:34]=[CH:33][C:32]([F:35])=[CH:31][CH:30]=3)(=[O:28])=[O:27])[CH:21]=[CH:20][C:10]=2[O:11][CH2:12][C:13]([O:15]C(C)(C)C)=[O:14])[CH:6]=[CH:7][CH:8]=1)#[N:2]>C(O)(C(F)(F)F)=O.C(Cl)Cl>[C:1]([C:3]1[CH:4]=[C:5]([C:9]2[CH:23]=[C:22]([CH2:24][N:25]([CH3:36])[S:26]([C:29]3[CH:30]=[CH:31][C:32]([F:35])=[CH:33][CH:34]=3)(=[O:28])=[O:27])[CH:21]=[CH:20][C:10]=2[O:11][CH2:12][C:13]([OH:15])=[O:14])[CH:6]=[CH:7][CH:8]=1)#[N:2]",
        "conditions": {
          "temperature": None,
          "yield": 43.6,
          "rxn_time": None
        },
        "experimental_details": {
          "procedure": "t-butyl 2-(2-(3-cyanophenyl)-4-((4-fluoro-N-methylphenylsulfonamido)methyl)phenoxy)acetate (4) (780 mg; 1.53 mmol) was stirred in a mixture of TFA (15 mL) and DCM (30 mL) for 1 h. After concentration in vacuo, the residue was purified via reversed phase semi-preparative HPLC to yield 303 mg (43.6%) ...",
          "date_of_experiment": "",
          "extracted_from_file": "ord_dataset-c3c1091f873b4f40827973a6f1f9b685",
          "is_mapped": True
        }
      }
    ]
  }
}

    # Extract unique chemicals to analyze them first
    all_unique_chemicals_to_analyze = extract_chemicals_from_pathway_data(single_pathway_data_input)
    print(f"Found {len(all_unique_chemicals_to_analyze)} unique chemicals to analyze: {all_unique_chemicals_to_analyze if all_unique_chemicals_to_analyze else 'None'}")

    # Create a cache for analyzed chemical properties
    chemical_properties_cache = {}

    if all_unique_chemicals_to_analyze:
        print(f"\n--- Analyzing unique chemicals first ---")
        for chemical_name in sorted(list(all_unique_chemicals_to_analyze)):
            # print(f"\n--- Requesting analysis for: {chemical_name} ---") # Already printed by perform_standalone
            props_obj, report_str = perform_standalone_chemical_analysis(
                chemical_identifier=chemical_name,
                provider=DEFAULT_LLM_PROVIDER
            )
            if props_obj and isinstance(props_obj, ChemicalProperties):
                chemical_properties_cache[chemical_name] = props_obj.to_dict() # Store dict version
                print(f"Successfully analyzed and cached properties for: {chemical_name}")
            else:
                error_message = report_str or f"Analysis failed for {chemical_name}, no properties object returned."
                chemical_properties_cache[chemical_name] = {"error": error_message, "original_query": chemical_name}
                print(f"Warning/Error analyzing {chemical_name}: {error_message}")
    
    # Add the cached properties back to the original data structure
    # This modifies single_pathway_data_input in-place
    augmented_pathway_data = add_properties_to_pathway_data(single_pathway_data_input, chemical_properties_cache)

    # Define the output file path for the augmented data
    output_file_augmented = os.path.join(project_root_dir, "augmented2_pathway_data_with_properties.json")

    # API Key Status Check
    print("\n--- API Key Status ---")
    openai_key_ok = OPENAI_API_KEY and "xxxx" not in OPENAI_API_KEY and "YOUR_OPENAI_API_KEY_HERE" not in OPENAI_API_KEY
    perplexity_key_ok = PERPLEXITY_API_KEY and "xxxx" not in PERPLEXITY_API_KEY and "YOUR_PERPLEXITY_API_KEY_HERE" not in PERPLEXITY_API_KEY
    print(f"OpenAI API Key: {'SET and appears valid' if openai_key_ok else 'NOT SET or PLACEHOLDER'}")
    print(f"Perplexity API Key: {'SET and appears valid' if perplexity_key_ok else 'NOT SET or PLACEHOLDER'}")
    print(f"Default LLM Provider: {DEFAULT_LLM_PROVIDER.upper()}")
    provider_key_ok = (DEFAULT_LLM_PROVIDER == "openai" and openai_key_ok) or \
                      (DEFAULT_LLM_PROVIDER == "perplexity" and perplexity_key_ok)
    if not provider_key_ok:
        print(f"WARNING: Default provider ({DEFAULT_LLM_PROVIDER.upper()}) key invalid. LLM calls may fail.")
    if not openai_key_ok and not perplexity_key_ok:
        print("CRITICAL WARNING: NEITHER OpenAI NOR Perplexity API keys are properly set. LLM calls WILL FAIL.")

    # Save the augmented data structure
    if not all_unique_chemicals_to_analyze:
        print("\nNo chemicals to analyze, so no augmented data to save.")
    else:
        try:
            with open(output_file_augmented, 'w', encoding='utf-8') as f:
                json.dump(augmented_pathway_data, f, indent=2)
            print(f"\nSuccessfully saved the AUGMENTED pathway data to: {output_file_augmented}")
        except IOError:
            print(f"Error: Could not write to output file - {output_file_augmented}")
        except Exception as e:
            print(f"An unexpected error occurred while saving the output file: {e}")

    print("\n--- Script Finished ---")
    if all_unique_chemicals_to_analyze:
        print(f"If successful, check output: {output_file_augmented}")
    print(f"Pricing files should be in: {project_root_dir}")