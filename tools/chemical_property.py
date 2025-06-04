import sys
import os
from dotenv import load_dotenv
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import pubchempy as pcp
import requests # For Perplexity API

load_dotenv()
# --- Project Root Setup (Optional if not using other project modules) ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_script_dir)
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

# --- API KEY DEFINITIONS ---
# Directly define your API keys here.
# Replace "sk-YOUR_OPENAI_API_KEY_HERE" and "pplx-YOUR_PERPLEXITY_API_KEY_HERE"
# with your actual keys.
# If a key is not available or you don't want to use a provider, set its key to None.

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")

# Choose your default LLM provider: "openai" or "perplexity"
DEFAULT_LLM_PROVIDER = os.environ.get("DEFAULT_LLM_PROVIDER", "openai") # Default to openai if not set

if not OPENAI_API_KEY and DEFAULT_LLM_PROVIDER == "openai":
    print("Warning: OPENAI_API_KEY environment variable is not set. OpenAI calls may fail if it's the selected provider.")
if not PERPLEXITY_API_KEY and DEFAULT_LLM_PROVIDER == "perplexity":
    print("Warning: PERPLEXITY_API_KEY environment variable is not set. Perplexity calls may fail if it's the selected provider.")

# --- Import OpenAI after keys are potentially defined ---
if OPENAI_API_KEY:
    import openai
else:
    if DEFAULT_LLM_PROVIDER == "openai":
        print("Warning: OpenAI API key is not set, but OpenAI is the default provider. OpenAI calls will fail.")
    openai = None # Set to None if key is missing, so client init can check


# --- Enums and Dataclasses (Same as before) ---
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
    estimated_price_per_kg: Optional[float] = None
    price_currency: Optional[str] = None
    supplier_info: List[Dict[str, Any]] = field(default_factory=list)
    safety_notes: List[str] = field(default_factory=list)
    environmental_impact: Optional[str] = None
    hazard_rating: Optional[int] = None
    solubility_rating: Optional[int] = None
    pubchem_full_json: Optional[Dict[str, Any]] = None


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
        self.openai_api_key_val = openai_api_key # Store the passed key
        self.perplexity_api_key_val = perplexity_api_key # Store the passed key
        self.openai_client = None # Initialize client as None

        if self.llm_provider == "openai":
            if not self.openai_api_key_val:
                raise ValueError("OpenAI API key is required for 'openai' provider but not set in the script.")
            if openai: # Check if openai module was imported
                self.openai_client = openai.OpenAI(api_key=self.openai_api_key_val)
            else:
                 raise ImportError("OpenAI library could not be imported, likely because the API key was not set before import.")
        elif self.llm_provider == "perplexity":
            if not self.perplexity_api_key_val:
                raise ValueError("Perplexity API key is required for 'perplexity' provider but not set in the script.")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}. Choose 'openai' or 'perplexity'.")
        
        print(f"[LLM] Using '{self.llm_provider}' provider for LLM tasks.")
        self.pricing_sources = []
        self._load_all_pricing_data()

    # ... (Rest of the ChemicalAnalysisAgent class methods:
    # _load_single_pricing_source, _load_all_pricing_data,
    # _is_cas_number, _is_smiles_like,
    # _get_pubchem_data, _extract_ghs_from_pubchem_json,
    # _search_single_local_source, _get_pricing_from_all_local_sources,
    # _get_perplexity_completion, _get_openai_completion, _get_llm_completion,
    # _get_llm_derived_pricing, _parse_int_score, _get_llm_derived_properties,
    # analyze_chemical, generate_report
    # ... are IDENTICAL to the previous full script you approved with GHS JSON parsing and Perplexity)
    # --- Pricing Data Loading ---
    def _load_single_pricing_source(self, filename: str, source_display_name: str) -> None:
        pricing_file_path = os.path.join(project_root_dir, filename)
        try:
            with open(pricing_file_path, 'r', encoding='utf-8') as f: data = json.load(f)
            print(f"[Pricing] Successfully loaded: {source_display_name} ({filename})")
            self.pricing_sources.append({"name": source_display_name, "filename": filename, "data": data})
        except FileNotFoundError:
            print(f"[Pricing Warn] Not found: {filename}. Source unavailable.")
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

    # --- Identifier Checks ---
    def _is_cas_number(self, identifier: str) -> bool:
        return bool(re.match(r'^\d{2,7}-\d{2}-\d$', identifier))

    def _is_smiles_like(self, identifier: str) -> bool:
        if " " in identifier.strip() and len(identifier.strip().split()) > 1: return False
        smiles_chars = set("()[]=#@+-.0123456789" + "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        if len(set(identifier) - smiles_chars) < 3 and len(identifier) > 2 and any(c in identifier for c in "()[]=#@+-."):
            return True
        return False

    # --- PubChem Data Retrieval ---
    def _get_pubchem_data(self, chemical_identifier: str) -> Tuple[Optional[pcp.Compound], Optional[Dict[str, Any]]]:
        print(f"[PubChem] Attrib. for '{chemical_identifier}'")
        compound: Optional[pcp.Compound] = None
        full_json_data: Optional[Dict[str, Any]] = None

        search_methods = []
        if self._is_cas_number(chemical_identifier): search_methods.append({'id': chemical_identifier, 'namespace': 'cas', 'type': 'CAS'})
        if self._is_smiles_like(chemical_identifier): search_methods.append({'id': chemical_identifier, 'namespace': 'smiles', 'type': 'SMILES'})
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
            except Exception as e_gen:
                print(f"[PubChem] General error in {method['type']} search for '{method['id']}': {e_gen}")
        
        if not compound:
            print(f"[PubChem] No compound found for '{chemical_identifier}' after all attempts.")
        return compound, full_json_data

    def _extract_ghs_from_pubchem_json(self, pubchem_json: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
        ghs_hazards_list = []
        if not pubchem_json: return ghs_hazards_list
        try:
            record = pubchem_json.get('Record', {})
            sections = record.get('Section', [])
            safety_section = next((s for s in sections if s.get('TOCHeading') == 'Safety and Hazards'), None)
            if not safety_section: return ghs_hazards_list
            haz_id_section = next((s for s in safety_section.get('Section', []) if s.get('TOCHeading') == 'Hazards Identification'), None)
            if not haz_id_section: return ghs_hazards_list
            ghs_class_section = next((s for s in haz_id_section.get('Section', []) if s.get('TOCHeading') == 'GHS Classification'), None)
            if not ghs_class_section: return ghs_hazards_list

            information = ghs_class_section.get('Information', [])
            pictograms_desc = {}
            
            for info_item in information:
                name, value = info_item.get('Name'), info_item.get('Value')
                if not value: continue
                if name == 'Pictogram(s)':
                    for swm in value.get('StringWithMarkup', []):
                        for markup in swm.get('Markup', []):
                            if markup.get('Type') == 'Icon': pictograms_desc[markup.get('URL','N/A')] = markup.get('Extra', 'Unknown Pictogram')
                elif name == 'GHS Hazard Statements':
                    for swm_item in value.get('StringWithMarkup', []):
                        text = swm_item.get('String', '')
                        match = re.match(r"(H\d{3}[A-Za-z+]*)?\s*\(?\d*%\)?:?\s*(.*)", text)
                        h_code, statement = (match.group(1) or "", re.sub(r"\s*\[(Warning|Danger)\s+.*?\]", "", match.group(2)).strip()) if match else ("", text)
                        pictogram_display = next(iter(pictograms_desc.values()), "N/A") if pictograms_desc else "N/A"
                        if statement: ghs_hazards_list.append({"pictogram": pictogram_display, "statement": f"{h_code}: {statement}".strip().lstrip(": ")})
            if ghs_hazards_list: print(f"[PubChem GHS] Extracted {len(ghs_hazards_list)} GHS entries from JSON.")
        except Exception as e: print(f"[PubChem GHS Error] Parsing GHS from JSON: {e}")
        return ghs_hazards_list

    # --- Local Pricing Search ---
    def _search_single_local_source(self, source_data: Dict[str, Any], input_id_norm: str, iupac_norm: Optional[str], commons_norm: List[str], smiles: Optional[str]) -> Optional[Dict[str, Any]]:
        if not source_data: return None
        if smiles:
            for k, v_list in source_data.items():
                if isinstance(v_list, list) and len(v_list) >= 2 and isinstance(v_list[0], str) and v_list[0] == smiles:
                    price = v_list[1] if isinstance(v_list[1], (float, int)) else None
                    loc = v_list[2] if len(v_list) >=3 and isinstance(v_list[2], str) else "Unknown Location"
                    if price is not None: return {"price": float(price), "currency": "INR", "location": loc, "source_name_in_json": k, "match_type": "SMILES"}
        names_to_check = set(commons_norm); 
        if iupac_norm: names_to_check.add(iupac_norm)
        names_to_check.add(input_id_norm)
        matches = []
        for k, v_list in source_data.items():
            if not (isinstance(v_list, list) and len(v_list) >= 2): continue
            key_norm = k.lower().strip()
            base_key_norm = re.match(r"^(.*?)\s*\(", key_norm).group(1).strip() if re.match(r"^(.*?)\s*\(", key_norm) else key_norm
            for name_check in names_to_check:
                if name_check == base_key_norm:
                    price = v_list[1] if isinstance(v_list[1], (float, int)) else None
                    loc = v_list[2] if len(v_list) >=3 and isinstance(v_list[2], str) else "Unknown Location"
                    if price is not None: matches.append({"price": float(price), "currency": "INR", "location": loc, "source_name_in_json": k, "match_type": "Exact Name", "len": len(name_check)})
        if matches: 
            best_match = sorted(matches, key=lambda x: x["len"], reverse=True)[0]
            del best_match["len"] # clean up temp key
            return best_match
        return None

    def _get_pricing_from_all_local_sources(self, in_id: str, iupac: Optional[str], commons: List[str], smiles: Optional[str]) -> Optional[Dict[str, Any]]:
        in_id_n, iupac_n, commons_n = in_id.lower().strip(), (iupac.lower().strip() if iupac else None), [c.lower().strip() for c in commons if c]
        for src in self.pricing_sources:
            if not src.get("data"): continue
            print(f"[Pricing] Searching in {src['name']}...")
            match = self._search_single_local_source(src["data"], in_id_n, iupac_n, commons_n, smiles)
            if match:
                print(f"[Pricing] Found '{in_id}' in {src['name']}. Price: {match['price']} {match['currency']}")
                match.update({"source_file_display_name": src['name'], "source_filename": src['filename']})
                return match
        print(f"[Pricing] '{in_id}' (SMILES: {smiles}, Names: {commons}) not found in local data.")
        return None

    # --- LLM Interaction ---
    def _get_perplexity_completion(self, system_prompt_content: str, user_prompt_content: str) -> Optional[str]:
        if not self.perplexity_api_key_val:
            print("[Perplexity Error] API key not configured.")
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
        except (KeyError, IndexError) as e: print(f"[Perplexity API Error] Parse response: {e}")
        return None

    def _get_openai_completion(self, system_prompt_content: str, user_prompt_content: str) -> Optional[str]:
        if not self.openai_client: # Check if client was initialized
            print("[OpenAI Error] Client not initialized (API key issue or OpenAI lib not imported).")
            return None
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo", messages=[{"role": "system", "content": system_prompt_content}, {"role": "user", "content": user_prompt_content}], temperature=0.2)
            return response.choices[0].message.content
        except Exception as e: # Catch generic openai errors
            print(f"[OpenAI API Error] {e}")
        return None

    def _get_llm_completion(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        if self.llm_provider == "openai": return self._get_openai_completion(system_prompt, user_prompt)
        elif self.llm_provider == "perplexity": return self._get_perplexity_completion(system_prompt, user_prompt)
        return None

    def _get_llm_derived_pricing(self, chemical_name: str, smiles: Optional[str], formula: Optional[str], cid: Optional[int]) -> Optional[Dict[str, Any]]:
        # print(f"\n--- Entered _get_llm_derived_pricing for: {chemical_name} ---", flush=True)
        # print(f"    _get_llm_derived_pricing ARGS: chemical_name='{chemical_name}', smiles='{smiles}', formula='{formula}', cid={cid}", flush=True)

        context_parts = [f"Chemical: {chemical_name}"]
        if smiles: context_parts.append(f"SMILES: {smiles}")
        if formula: context_parts.append(f"Formula: {formula}")
        if cid: context_parts.append(f"PubChem CID: {cid}")
        system_prompt = "You are a chemical market analyst. Provide price estimations in JSON format."
        
        user_prompt = f"""{", ".join(context_parts)}
Please provide an estimated bulk price for this chemical in INR per kg.
If INR is not possible, provide it in USD per kg.
Consider typical research grade or small industrial scale pricing.
If an exact price is unknown, please provide your best possible *numerical estimate* or a price range (e.g., "10000-15000" for INR, or "100-150" for USD).
It is important to provide a numerical value if at all possible, even if confidence is very low.
Return JSON with EITHER "estimated_price_per_kg_inr" OR "estimated_price_per_kg_usd": {{
    "estimated_price_per_kg_inr": float_or_string_range_or_null, 
    "estimated_price_per_kg_usd": float_or_string_range_or_null,
    "price_confidence": "very_low/low/medium/high",
    "price_basis_notes": "Brief notes..."
}}
Provide only one of the price fields, setting the other to null if you use one. Prioritize INR if possible.
"""

        llm_response_content = self._get_llm_completion(system_prompt, user_prompt)

        # print(f"\n[DEBUG Pricing LLM RAW RESPONSE for '{chemical_name}'] --- START RAW ---", flush=True)
        # print(llm_response_content, flush=True)
        # print(f"--- END RAW ---\n", flush=True)

        if not llm_response_content:
            print(f"[Pricing LLM] LLM did not return content for '{chemical_name}'.", flush=True)
            return None
        try:
            json_str_to_parse = llm_response_content
            match_json = re.search(r"```json\s*([\s\S]*?)\s*```", llm_response_content, re.IGNORECASE)
            if match_json:
                json_str_to_parse = match_json.group(1).strip()
                # print(f"[Pricing LLM] Stripped markdown. JSON string to parse: <<<START>>>\n{json_str_to_parse}\n<<<END>>>", flush=True)
            else:
                json_str_to_parse = llm_response_content.strip()
                # print(f"[Pricing LLM] No markdown block detected. JSON string to parse: <<<START>>>\n{json_str_to_parse}\n<<<END>>>", flush=True)

            if not json_str_to_parse:
                # print(f"[Pricing LLM Error] JSON string is empty for '{chemical_name}'.", flush=True)
                return None

            data = json.loads(json_str_to_parse)
            
            price_value_usd = data.get("estimated_price_per_kg_usd")
            price_value_inr = data.get("estimated_price_per_kg_inr")
            
            final_price_inr = None # This will be the INR price we pass back
            llm_provided_currency = None # To track what the LLM gave

            # Helper to parse float or string range (e.g., "100-150")
            def parse_price_value(value_str_or_float):
                if isinstance(value_str_or_float, (int, float)):
                    return float(value_str_or_float)
                if isinstance(value_str_or_float, str):
                    # Remove commas from price strings like "25,000"
                    cleaned_value_str = value_str_or_float.replace(',', '')
                    match_range = re.match(r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)", cleaned_value_str)
                    if match_range:
                        low = float(match_range.group(1))
                        high = float(match_range.group(2))
                        # print(f"[Pricing LLM] Parsed price range: {value_str_or_float}, using average.")
                        return (low + high) / 2
                    else: # Try direct float conversion for strings like "25000.0"
                        return float(cleaned_value_str)
                return None 

            if price_value_inr is not None:
                parsed_val = parse_price_value(price_value_inr)
                if parsed_val is not None:
                    final_price_inr = parsed_val
                    llm_provided_currency = "INR"
                    # print(f"[Pricing LLM] Got price directly in INR: {final_price_inr}", flush=True)
            
            if final_price_inr is None and price_value_usd is not None: # Only if INR wasn't found/parsed
                parsed_val = parse_price_value(price_value_usd)
                if parsed_val is not None:
                    final_price_inr = parsed_val * self.USD_TO_INR_RATE # Convert USD to INR
                    llm_provided_currency = "USD"
                    # print(f"[Pricing LLM] Got price in USD: {parsed_val}, converted to INR: {final_price_inr}", flush=True)

            if final_price_inr is not None:
                return {
                    "price_inr": final_price_inr, # This is always in INR now
                    "currency_llm_provided": llm_provided_currency, # What LLM gave (USD or INR)
                    "raw_llm_price_value": price_value_inr if llm_provided_currency == "INR" else price_value_usd,
                    "confidence": data.get("price_confidence"),
                    "basis_notes": data.get("price_basis_notes"),
                    "source_type": f"LLM ({self.llm_provider})"
                }
            else: # Neither USD nor INR price was found or parsable
                # print(f"[Pricing LLM] LLM did not provide a usable 'estimated_price_per_kg_inr' or 'estimated_price_per_kg_usd' for '{chemical_name}'. Basis: {data.get('price_basis_notes')}", flush=True)
                return {
                    "price_inr": None,
                    "currency_llm_provided": None,
                    "raw_llm_price_value": None,
                    "confidence": data.get("price_confidence", "low_no_price_data"),
                    "basis_notes": data.get("price_basis_notes", "LLM indicated no specific price available or format not recognized."),
                    "source_type": f"LLM ({self.llm_provider})"
                }
        except json.JSONDecodeError as e:
            print(f"[Pricing LLM Error] JSONDecodeError for '{chemical_name}': {e}. String tried: '{json_str_to_parse[:300]}...'", flush=True)
        except (ValueError, TypeError) as e:
            price_key_tried = "estimated_price_per_kg_inr or estimated_price_per_kg_usd"
            print(f"[Pricing LLM Error] ValueError/TypeError parsing price for '{chemical_name}': {e}. Data for {price_key_tried} was problematic.", flush=True)
        
        print(f"--- Exiting _get_llm_derived_pricing with None due to error or unhandled case for '{chemical_name}' ---", flush=True)
        return None

    def _parse_int_score(self, val: Any, field: str) -> Optional[int]:
        if val is None: return None
        try: return int(val)
        except (ValueError, TypeError): print(f"[LLM Parse Warn] Non-int score for {field}: '{val}'"); return None

    def _get_llm_derived_properties(self, name: str, formula: Optional[str], smiles: Optional[str], 
                                   cid: Optional[int], cas: Optional[str], mw: Optional[float], 
                                   iupac: Optional[str]) -> Dict[str, Any]:
        
        print(f"[LLM Props] Querying LLM ({self.llm_provider}) for derived properties of '{name}'.")
        context_parts = [f"Chemical: {name}."]
        known_info = []
        if cid: known_info.append(f"PubChem CID: {cid}")
        if iupac: known_info.append(f"IUPAC Name: {iupac}")
        if formula: known_info.append(f"Formula: {formula}")
        if mw: known_info.append(f"MW: {mw:.2f}" if isinstance(mw, float) else f"MW: {mw}") # Format float MW
        if smiles: known_info.append(f"SMILES: {smiles}")
        if cas: known_info.append(f"CAS: {cas}")
        
        if known_info: 
            context_parts.append(f"Known info: {'; '.join(known_info)}.")
        else: 
            context_parts.append("No definitive structural or ID information known from databases yet.")

        full_context = "\n".join(context_parts)
        
        guess_instr = ""
        if not iupac: guess_instr += '"iupac_name_llm_guess": "Guess IUPAC name or null",\n'
        if not formula: guess_instr += '"molecular_formula_llm_guess": "Guess formula or null",\n'
        if not mw: guess_instr += '"molecular_weight_llm_guess": "Guess MW (float) or null",\n'
        if not cas: guess_instr += '"cas_number_llm_guess": "Guess CAS (XXX-XX-X) or null",\n'
        if not smiles: guess_instr += '"smiles_llm_guess": "Guess SMILES or null",\n'

        system_prompt = "You are a chemical safety, properties, and environmental expert. Provide accurate assessments in JSON format. If guessing core chemical identity, clearly indicate it (e.g., field ends with '_llm_guess')."
        
        # Corrected user_prompt construction
        user_prompt = f"""{full_context}
Provide analysis in JSON. Ratings 1-10 (10=high/extreme/excellent) or null. If core ID (IUPAC, formula, MW, CAS, SMILES) unknown, guess it.
{{
    {guess_instr}
    "solubility": {{
        "water_solubility": "Water Soluble/Organic Soluble/Poorly Soluble/Insoluble/Unknown", 
        "organic_solvents_compatibility": ["list of types like alcohols, ethers, hydrocarbons"], 
        "notes_on_solubility": "any additional notes, e.g. pH dependence",
        "solubility_rating": "integer 1-10 or null"
    }},
    "hazards": {{
        "corrosive": true/false/null, 
        "flammable": true/false/null, 
        "toxic": true/false/null,
        "carcinogenic_suspected": true/false/null, 
        "environmental_hazard_notes": "brief notes on environmental risks",
        "overall_hazard_level": "Low/Moderate/High/Extreme/Unknown",
        "hazard_rating": "integer 1-10 or null",
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
        "overall_score": "integer 1-10 or null"
    }},
    "environmental_impact_summary": "brief overall assessment of environmental effects"
}}
If 'ghs_info_llm' cannot be determined, use empty list []. For other missing values, use null or "Unknown" as appropriate for the field type. Ensure overall_hazard_level and water_solubility use specified capitalized options.
"""
        
        llm_response_content = self._get_llm_completion(system_prompt, user_prompt)
        default_empty_response = {
            "solubility": {"water_solubility": SolubilityType.UNKNOWN.value, "solubility_rating": None},
            "hazards": {"overall_hazard_level": HazardLevel.UNKNOWN.value, "hazard_rating": None, "ghs_info_llm": []},
            "safety_precautions": [], "green_chemistry": {"overall_score": None},
            "environmental_impact_summary": "Assessment unavailable due to LLM error."
        }
        if not llm_response_content: return default_empty_response

        try:
            # Attempt to strip potential markdown ```json ... ``` block if present
            match = re.search(r"```json\s*([\s\S]*?)\s*```", llm_response_content, re.IGNORECASE)
            if match:
                json_str = match.group(1)
                print("[LLM Info] Stripped markdown JSON block from LLM response.")
            else:
                json_str = llm_response_content
            
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"[LLM Parse Error] Props for {name}. Raw (after potential strip): {json_str[:500]}... Error: {e}")
            return default_empty_response

    # --- Main Analysis Orchestration ---
    def analyze_chemical(self, chemical_identifier: str) -> ChemicalProperties:
        # print(f"\n\n!!!! ENTERED ChemicalAnalysisAgent.analyze_chemical for: {chemical_identifier} !!!!\n\n", flush=True)
        props = ChemicalProperties(name=chemical_identifier, original_query=chemical_identifier)
        # print("!!!! PROPS initialized !!!!", flush=True)

        # print("!!!! Before _get_pubchem_data !!!!", flush=True)
        pubchem_compound, pubchem_full_json = self._get_pubchem_data(chemical_identifier)
        props.pubchem_full_json = pubchem_full_json
        current_name_for_llm = chemical_identifier # Initialize
        # print(f"!!!! After _get_pubchem_data. Compound found: {bool(pubchem_compound)} !!!!", flush=True)

        if pubchem_compound:
            # print("!!!! Processing PubChem compound data !!!!", flush=True)
            props.name = pubchem_compound.iupac_name or (pubchem_compound.synonyms[0] if pubchem_compound.synonyms else chemical_identifier)
            current_name_for_llm = props.name # Update current_name_for_llm if PubChem provides a better name
            props.iupac_name = pubchem_compound.iupac_name
            props.common_names = list(pubchem_compound.synonyms[:10]) if pubchem_compound.synonyms else []
            orig_low = chemical_identifier.lower()
            # Ensure original query is in common names if it's different and not IUPAC
            if orig_low not in {n.lower() for n in props.common_names} and (not props.iupac_name or orig_low != props.iupac_name.lower()):
                props.common_names.insert(0, chemical_identifier)
            # If common_names is still empty and props.name is different from original query, add original query
            if not props.common_names and props.name and props.name.lower() != orig_low:
                 props.common_names.insert(0, chemical_identifier)

            props.molecular_formula = pubchem_compound.molecular_formula
            props.molecular_weight = float(pubchem_compound.molecular_weight) if pubchem_compound.molecular_weight else None
            if pubchem_compound.synonyms:
                cas_syns = [s for s in pubchem_compound.synonyms if self._is_cas_number(s)]
                if cas_syns: props.cas_number = cas_syns[0]
            props.smiles = pubchem_compound.canonical_smiles
            props.pubchem_cid = pubchem_compound.cid
            props.ghs_hazards = self._extract_ghs_from_pubchem_json(props.pubchem_full_json)
            # print("!!!! Done processing PubChem compound data !!!!", flush=True)
            # current_name_for_llm remains chemical_identifier

        # Ensure current_name_for_llm is set before LLM props call
        if not current_name_for_llm: # Should not happen if initialized, but as a safeguard
            current_name_for_llm = chemical_identifier
            # print(f"!!!! WARNING: current_name_for_llm was empty before LLM props, defaulted to: '{chemical_identifier}' !!!!", flush=True)

        # print(f"!!!! Before _get_llm_derived_properties. current_name_for_llm: '{current_name_for_llm}' !!!!", flush=True)
        llm_derived_info = self._get_llm_derived_properties(current_name_for_llm, props.molecular_formula, props.smiles, props.pubchem_cid, props.cas_number, props.molecular_weight, props.iupac_name)
        # print("!!!! After _get_llm_derived_properties !!!!", flush=True)

        if not pubchem_compound: # Update core props if PubChem failed & LLM guessed them
            # print("!!!! Updating core props from LLM guesses as PubChem failed !!!!", flush=True)
            props.iupac_name = props.iupac_name or llm_derived_info.get("iupac_name_llm_guess")
            # If LLM guessed a name, and props.name is still original_query, update props.name
            if props.name == chemical_identifier and props.iupac_name:
                props.name = props.iupac_name
                current_name_for_llm = props.name # Also update current_name_for_llm for pricing
                # print(f"!!!! Updated props.name and current_name_for_llm from LLM IUPAC guess: '{props.name}' !!!!", flush=True)

            props.molecular_formula = props.molecular_formula or llm_derived_info.get("molecular_formula_llm_guess")
            mw_g = llm_derived_info.get("molecular_weight_llm_guess")
            if mw_g is not None:
                try: props.molecular_weight = props.molecular_weight or float(mw_g)
                except (ValueError,TypeError): print(f"LLM MW guess '{mw_g}' not float.")
            cas_g = llm_derived_info.get("cas_number_llm_guess")
            if cas_g and self._is_cas_number(cas_g): props.cas_number = props.cas_number or cas_g
            props.smiles = props.smiles or llm_derived_info.get("smiles_llm_guess")

        # Assign other properties from LLM
        props.solubility = llm_derived_info.get("solubility", props.solubility)
        llm_hazards = llm_derived_info.get("hazards", {})
        haz_lvl_str = llm_hazards.get("overall_hazard_level", "unknown").lower()
        props.hazard_level = HazardLevel[haz_lvl_str.upper()] if haz_lvl_str.upper() in HazardLevel.__members__ else HazardLevel.UNKNOWN
        props.is_corrosive = llm_hazards.get("corrosive", props.is_corrosive)
        props.is_flammable = llm_hazards.get("flammable", props.is_flammable)
        props.is_toxic = llm_hazards.get("toxic", props.is_toxic)

        if not props.ghs_hazards and "ghs_info_llm" in llm_hazards: # Use LLM GHS if PubChem didn't provide
            for item in llm_hazards.get("ghs_info_llm", []):
                if isinstance(item, dict) and item.get("h_statement"):
                    props.ghs_hazards.append({"pictogram": item.get("pictogram_description", "N/A"), "statement": f"{item.get('h_code', '')}: {item.get('h_statement', '')}".strip().lstrip(": ")})
            if props.ghs_hazards: print("[GHS] Used LLM-derived GHS info.", flush=True)

        props.safety_notes = llm_derived_info.get("safety_precautions", props.safety_notes)
        props.environmental_impact = llm_derived_info.get("environmental_impact_summary", props.environmental_impact)
        gc_score = llm_derived_info.get("green_chemistry", {}).get("overall_score")
        props.green_chemistry_score = self._parse_int_score(gc_score, "GC score") or props.green_chemistry_score
        props.hazard_rating = self._parse_int_score(llm_hazards.get("hazard_rating"), "Hazard rating") or props.hazard_rating
        sol_rating = llm_derived_info.get("solubility", {}).get("solubility_rating")
        props.solubility_rating = self._parse_int_score(sol_rating, "Solubility rating") or props.solubility_rating

        # print("!!!! REACHED PRICING SECTION !!!!", flush=True)

        # Pricing
        # print(f"\n[DEBUG ACA] Attempting local pricing for: '{chemical_identifier}'", flush=True)
        # Use props.name for local search if available and different, but also include original chemical_identifier
        # The _get_pricing_from_all_local_sources already takes `in_id` (original query) and `iupac` (props.iupac_name)
        # So, chemical_identifier (original query) and props.iupac_name are sufficient for names.
        local_price = self._get_pricing_from_all_local_sources(chemical_identifier, props.iupac_name, props.common_names, props.smiles)

        if local_price:
            # print(f"\n[DEBUG ACA] Local price FOUND: {local_price}", flush=True)
            props.estimated_price_per_kg = local_price.get("price")
            props.price_currency = local_price.get("currency", "INR") # Default to INR if not specified
            # print(f"[DEBUG ACA]   Extracted local price: {props.estimated_price_per_kg} {props.price_currency}", flush=True)
            props.supplier_info = [{"name": f"Local DB: {local_price.get('source_file_display_name','N/A')} ({local_price.get('source_filename','N/A')} - Key: '{local_price.get('source_name_in_json','N/A')}')",
                                    "availability": (f"Price: {props.estimated_price_per_kg:.2f} {props.price_currency}. Match: {local_price.get('match_type','N/A')}."),
                                    "location": local_price.get('location', 'Unknown Location'), "source_type": "Local JSON"}]
        else: # Local price NOT found, try LLM
            # print(f"\n[DEBUG ACA] Local price NOT found for '{chemical_identifier}'. Attempting LLM pricing.", flush=True)

            # Ensure current_name_for_llm is sensible for the pricing query
            # It would have been updated if PubChem found a name or if LLM guessed an IUPAC name when PubChem failed.
            if not current_name_for_llm:
                current_name_for_llm = chemical_identifier # Fallback, should have been set earlier
                print(f"!!!! WARNING: current_name_for_llm was empty before LLM pricing, defaulted to: '{chemical_identifier}' !!!!", flush=True)

            # print(f"[DEBUG ACA] Name for LLM pricing query: '{current_name_for_llm}' (SMILES: {props.smiles}, Formula: {props.molecular_formula}, CID: {props.pubchem_cid})", flush=True) # LOG 2

            llm_price_info = self._get_llm_derived_pricing(current_name_for_llm, props.smiles, props.molecular_formula, props.pubchem_cid)
            # print(f"[DEBUG ACA] LLM Price Info object received: {llm_price_info}", flush=True)

            if llm_price_info and llm_price_info.get("price_inr") is not None: # Check for "price_inr"
                props.estimated_price_per_kg = llm_price_info["price_inr"] # Already in INR
                props.price_currency = "INR" # Display currency is INR

                raw_llm_val = llm_price_info.get("raw_llm_price_value", "N/A")
                llm_curr_provided = llm_price_info.get("currency_llm_provided", "N/A")
                
                availability_details = (f"Est. Price: {props.estimated_price_per_kg:.2f} {props.price_currency}/kg. ")
                if llm_curr_provided and llm_curr_provided != "INR": # If LLM gave USD, show original
                    availability_details += f"(LLM provided: {raw_llm_val} {llm_curr_provided}, converted to INR). "
                else: # LLM gave INR or currency unknown
                        availability_details += f"(LLM provided: {raw_llm_val} {llm_curr_provided}). "

                availability_details += (f"Conf: {llm_price_info.get('confidence','N/A')}. Basis: {llm_price_info.get('basis_notes','N/A')}")

                # print(f"[DEBUG ACA] LLM Price (INR): {props.estimated_price_per_kg}", flush=True)
                props.supplier_info = [{
                    "name": f"LLM Estimation ({self.llm_provider})",
                    "availability": availability_details,
                    "location": "Global Market (Est.)", "source_type": f"LLM ({self.llm_provider})"
                }]
            else: # LLM pricing failed or returned a dict with price_inr as None
                # print(f"[DEBUG ACA] LLM pricing failed or returned null/no price (in INR) for '{current_name_for_llm}'.", flush=True)
                props.estimated_price_per_kg = None
                props.price_currency = None
                availability_note = "Not in local DBs. "
                if llm_price_info: 
                    availability_note += (f"LLM ({self.llm_provider}) consulted: Price estimation difficult. "
                                            f"Confidence: {llm_price_info.get('confidence','N/A')}. "
                                            f"Basis: {llm_price_info.get('basis_notes','N/A')}")
                else: 
                    availability_note += f"LLM ({self.llm_provider}) call failed or did not return parsable pricing data."

                props.supplier_info = [{"name": "No Definitive Pricing Data",
                                        "availability": availability_note,
                                        "location": "N/A", "source_type": "None"}]
                
        # print("!!!! END OF PRICING SECTION, before returning props !!!!", flush=True)
        return props

    def generate_report(self, chemical_properties: ChemicalProperties) -> str:
        report = f"""
CHEMICAL ANALYSIS REPORT ({self.llm_provider.upper()} LLM Used)
========================
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

IDENTIFICATION
--------------
Original Query: {chemical_properties.original_query or 'N/A'}
Analyzed As: {chemical_properties.name} 
IUPAC Name: {chemical_properties.iupac_name or 'Not available'}
Common Names: {', '.join(chemical_properties.common_names) if chemical_properties.common_names else 'N/A'}
PubChem CID: {chemical_properties.pubchem_cid or 'Not available'}
CAS Number: {chemical_properties.cas_number or 'Not available'}
SMILES (Canonical): {chemical_properties.smiles or 'Not available'}
Molecular Formula: {chemical_properties.molecular_formula or 'Not available'}
Molecular Weight: {f'{chemical_properties.molecular_weight:.2f} g/mol' if chemical_properties.molecular_weight is not None else 'Not available'}

SAFETY PROFILE
--------------
Overall Hazard Level: {chemical_properties.hazard_level.value}
Hazard Rating (1-10, 10=extreme): {chemical_properties.hazard_rating if chemical_properties.hazard_rating is not None else 'N/A'}
Corrosive: {'Yes' if chemical_properties.is_corrosive else 'No' if chemical_properties.is_corrosive is False else 'Unknown'}
Flammable: {'Yes' if chemical_properties.is_flammable else 'No' if chemical_properties.is_flammable is False else 'Unknown'}
Toxic: {'Yes' if chemical_properties.is_toxic else 'No' if chemical_properties.is_toxic is False else 'Unknown'}
GHS Hazard Statements:""" # Changed title here
        
        # Crude check for GHS source in log (not perfectly reliable but gives a hint)
        # A better way would be to add a 'ghs_source' field to ChemicalProperties
        # For simplicity, we'll determine the hint based on whether pubchem_full_json was populated
        # and if the GHS extraction from it logged success.
        ghs_source_hint = ""
        # This check is difficult to do reliably without passing more state or checking logs precisely.
        # Let's assume that if props.ghs_hazards is populated, it's from one of the sources.
        # For now, the hint logic is removed for simplicity, as it was error-prone.
        # We can add a `ghs_source` attribute to ChemicalProperties later if needed.

        if chemical_properties.ghs_hazards:
            # report += f" {ghs_source_hint}" # Removed hint for now
            for gh in chemical_properties.ghs_hazards:
                # Only print the statement
                report += f"\n  - {gh.get('statement', 'N/A')}"
        else:
            report += " No GHS hazard statements found or provided."

        report += f"""

SOLUBILITY PROPERTIES
--------------------
Water Solubility: {chemical_properties.solubility.get('water_solubility', 'Unknown')}
Solubility Rating (1-10, 10=excellent): {chemical_properties.solubility_rating if chemical_properties.solubility_rating is not None else 'N/A'}
Compatible Organic Solvents: {', '.join(chemical_properties.solubility.get('organic_solvents_compatibility', ['Unknown']))}
Solubility Notes: {chemical_properties.solubility.get('notes_on_solubility', 'N/A')}

GREEN CHEMISTRY & ENVIRONMENT
-----------------------------
Green Chemistry Score (Est. 1-10): {f'{chemical_properties.green_chemistry_score}' if chemical_properties.green_chemistry_score is not None else 'N/A'}
Environmental Impact Summary: {chemical_properties.environmental_impact or 'Not assessed'}

PRICING & AVAILABILITY (INR/kg)
---------------------------------------------
Estimated Price per kg: {f'{chemical_properties.estimated_price_per_kg:.2f} {chemical_properties.price_currency}/kg' if chemical_properties.estimated_price_per_kg is not None and chemical_properties.price_currency else 'Not available'}"""

        if chemical_properties.supplier_info:
            report += "\nSource Information:"
            for s_info in chemical_properties.supplier_info:
                report += f"\n  - Source: {s_info.get('name', 'N/A')}"
                report += f"\n    Details: {s_info.get('availability', 'N/A')}"
                if s_info.get('location') and s_info.get('location') not in ["N/A"]:
                     report += f"\n    Location: {s_info.get('location')}"
        else: report += "\n  No pricing source information available."
        report += """

SAFETY PRECAUTIONS & HANDLING
-----------------------------"""
        if chemical_properties.safety_notes:
            for precaution in chemical_properties.safety_notes: report += f"\n• {precaution}"
        else: report += "\nFollow standard lab safety. Consult SDS for details."
        return report
    
# --- Standalone Execution Logic ---
def perform_standalone_chemical_analysis(chemical_identifier: str,
                                         provider: str = DEFAULT_LLM_PROVIDER,
                                         api_key_override: Optional[str] = None
                                         ) -> Tuple[Optional[ChemicalProperties], Optional[str]]:
    print(f"\n\n>>>>> ENTERED perform_standalone_chemical_analysis for: '{chemical_identifier}' with provider: {provider} <<<<<\n\n", flush=True)
    
    global OPENAI_API_KEY, PERPLEXITY_API_KEY

    current_openai_key_to_use = OPENAI_API_KEY
    current_perplexity_key_to_use = PERPLEXITY_API_KEY

    # Apply override if provided for the selected provider
    if api_key_override:
        if provider == "openai":
            print(f"[Standalone] OpenAI API key override provided.")
            current_openai_key_to_use = api_key_override
        elif provider == "perplexity":
            print(f"[Standalone] Perplexity API key override provided.")
            current_perplexity_key_to_use = api_key_override
        # If you add more providers, you might need to handle the override for them too.

    # Basic check if keys are placeholder (using the keys that will actually be used)
    if provider == "openai" and (not current_openai_key_to_use or "YOUR_OPENAI_API_KEY_HERE" in current_openai_key_to_use):
        msg = "OpenAI API key is not set correctly or is a placeholder. Please replace placeholder."
        if api_key_override and current_openai_key_to_use == api_key_override:
             msg += " (The provided override key might also be invalid/placeholder)."
        print(f"[Standalone Config Error] {msg}"); return None, msg
    if provider == "perplexity" and (not current_perplexity_key_to_use or "YOUR_PERPLEXITY_API_KEY_HERE" in current_perplexity_key_to_use):
        msg = "Perplexity API key is not set correctly or is a placeholder. Please replace placeholder."
        if api_key_override and current_perplexity_key_to_use == api_key_override:
             msg += " (The provided override key might also be invalid/placeholder)."
        print(f"[Standalone Config Error] {msg}"); return None, msg

    try:
        # print(">>>>> perform_standalone: Before ChemicalAnalysisAgent instantiation <<<<<", flush=True)
        # Pass the potentially overridden keys to the agent
        agent = ChemicalAnalysisAgent(openai_api_key=current_openai_key_to_use,
                                      perplexity_api_key=current_perplexity_key_to_use,
                                      llm_provider=provider)
        # print(">>>>> perform_standalone: ChemicalAnalysisAgent INSTANTIATED <<<<<", flush=True)
        # print(f">>>>> perform_standalone: Before calling agent.analyze_chemical for '{chemical_identifier}' <<<<<", flush=True)

        props = agent.analyze_chemical(chemical_identifier)
        # print(f">>>>> perform_standalone: After calling agent.analyze_chemical. Props object is None: {props is None} <<<<<", flush=True)
        if props:
            # print(">>>>> perform_standalone: Props object received. Before agent.generate_report <<<<<", flush=True)
            report = agent.generate_report(props)
            # print(">>>>> perform_standalone: Report generated. Returning props and report. <<<<<", flush=True)
            return props, report
        msg = f"Error: Null properties for {chemical_identifier}."
        # print(f">>>>> perform_standalone: {msg} <<<<<", flush=True)
        return None, msg
    except ValueError as e:
        print(f">>>>> perform_standalone: CONFIG ERROR (ValueError): {e} <<<<<", flush=True)
        return None, str(e)
    except ImportError as e:
        print(f">>>>> perform_standalone: IMPORT ERROR (ImportError): {e} <<<<<", flush=True)
        return None, str(e)
    except Exception as e:
        print(f">>>>> perform_standalone: UNEXPECTED EXCEPTION: {e} <<<<<", flush=True)
        import traceback
        traceback.print_exc() # Print full traceback here
        return None, f"Unexpected error in perform_standalone: {e}"
        # Check for specific API error messages if possible
        # if openai and isinstance(e, openai.APIError): # Check if openai was imported and e is an APIError
        #      error_msg = f"[Standalone OpenAI API Error] {e}"
        # elif isinstance(e, requests.exceptions.HTTPError): # Common for Perplexity
        #      if e.response.status_code == 401: error_msg = f"[Standalone Perplexity Auth Error] {e}"
        #      else: error_msg = f"[Standalone Perplexity HTTP Error] {e}"
        # print(error_msg); import traceback; traceback.print_exc(); return None, error_msg

def main():
    global project_root_dir 
    if 'project_root_dir' not in globals() or not project_root_dir:
        csd_main = os.path.dirname(os.path.abspath(__file__))
        project_root_dir = os.path.dirname(csd_main)
        if project_root_dir not in sys.path: sys.path.insert(0, project_root_dir)

    dummy_files = {
        "pricing_data.json": {"Benzene (CAS 71-43-2)": ["C1=CC=CC=C1", 100.0, "Pune, India"], "Water": ["O", 1.0, "Global"]}, 
        "second_source.json": {"Ethanol (CAS 64-17-5)": ["CCO", 150.0, "Mumbai, India"]},
        "sigma_source.json": {"Methanol": ["CO", 120.0, "Bangalore, India"]} 
    }
    for fname, content in dummy_files.items():
        fpath = os.path.join(project_root_dir, fname)
        if not os.path.exists(fpath):
            try:
                with open(fpath, 'w', encoding='utf-8') as f: json.dump(content, f, indent=2)
                print(f"Created/Replaced dummy file: {fpath} for testing")
            except IOError as e: print(f"Could not create dummy file {fpath}: {e}")
    
    print(f"OpenAI Key Status: {'SET (check if placeholder)' if OPENAI_API_KEY else 'NOT SET'}")
    print(f"Perplexity Key Status: {'SET (check if placeholder)' if PERPLEXITY_API_KEY else 'NOT SET'}")
    print(f"Default LLM Provider for this run: {DEFAULT_LLM_PROVIDER.upper()}")
    
    test_chemicals = [
        "2-Amino-5-cyano-3-methylbenzoic acid"
    ]
    
    provider_to_run = DEFAULT_LLM_PROVIDER
    # Check if chosen provider's key is valid (not None and not placeholder)
    if provider_to_run == "openai" and (not OPENAI_API_KEY or "YOUR_OPENAI_API_KEY_HERE" in OPENAI_API_KEY):
        print("Warning: OpenAI key is placeholder or missing. LLM calls for OpenAI will fail.")
        if PERPLEXITY_API_KEY and "YOUR_PERPLEXITY_API_KEY_HERE" not in PERPLEXITY_API_KEY :
            print("Switching to Perplexity for test run as OpenAI key is invalid.")
            provider_to_run = "perplexity"
        else:
            print("CRITICAL: Cannot run LLM tests. Both OpenAI and Perplexity keys are invalid or placeholders.")
            return
    elif provider_to_run == "perplexity" and (not PERPLEXITY_API_KEY or "YOUR_PERPLEXITY_API_KEY_HERE" in PERPLEXITY_API_KEY):
        print("Warning: Perplexity key is placeholder or missing. LLM calls for Perplexity will fail.")
        if OPENAI_API_KEY and "YOUR_OPENAI_API_KEY_HERE" not in OPENAI_API_KEY:
            print("Switching to OpenAI for test run as Perplexity key is invalid.")
            provider_to_run = "openai"
        else:
            print("CRITICAL: Cannot run LLM tests. Both Perplexity and OpenAI keys are invalid or placeholders.")
            return


    print(f"--- Chemical Analysis Demo (Attempting with {provider_to_run.upper()}) ---"); print("=" * 70)
    
    for chemical_id in test_chemicals:
        print(f"\n\n>>> Analyzing: {chemical_id}"); print("-" * 50)
        # Capture stdout to check for GHS source hint later if needed (crude)
        # This part is for the `generate_report` GHS source hint. A cleaner way is to store ghs_source in props.
        # from io import StringIO
        # old_stdout = sys.stdout
        # sys.stdout = captured_output = StringIO()

        _props_obj, report_str = perform_standalone_chemical_analysis(chemical_id, provider=provider_to_run)
        
        # sys.stdout = old_stdout # Restore stdout
        # print(captured_output.getvalue()) # Print captured logs for debugging if needed
        
        if report_str: print(report_str)
        else: print(f"Failed report for {chemical_id} or error occurred.")
        print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
