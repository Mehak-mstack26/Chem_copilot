import os
import re
import time
import pandas as pd
from rdkit import Chem, RDLogger
import traceback
from functools import lru_cache
from typing import Optional, List, Dict, Any, Tuple
import autogen
import openai # For direct OpenAI calls in handle_full_info etc.
import ast
import sys
from dotenv import load_dotenv
import json
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import pubchempy as pcp
import requests # For Perplexity API

# --- Load Environment Variables ---
load_dotenv()

# --- Project Root Setup ---
try:
    PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError: # Fallback for environments where __file__ is not defined
    PROJECT_ROOT_DIR = os.getcwd()

if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR)

# --- API KEY DEFINITIONS ---
try:
    import api_config
    if not os.getenv("OPENAI_API_KEY"):
        if hasattr(api_config, 'api_key') and api_config.api_key:
            os.environ["OPENAI_API_KEY"] = api_config.api_key
except ImportError:
    pass # print("Warning: api_config.py not found.")
except Exception:
    pass # print(f"Warning: Unexpected error during api_config import.")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")
DEFAULT_LLM_PROVIDER = os.environ.get("DEFAULT_LLM_PROVIDER", "openai")

RDLogger.DisableLog('rdApp.*')

# --- ChemicalAnalysisAgent Enums and Dataclasses ---
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
    ghs_hazards: List[Dict[str, str]] = field(default_factory=list) # List of {"pictogram": "desc", "statement": "Hxxx: text"}
    green_chemistry_score: Optional[int] = None
    estimated_price_per_kg: Optional[float] = None
    price_currency: Optional[str] = None
    supplier_info: List[Dict[str, Any]] = field(default_factory=list)
    safety_notes: List[str] = field(default_factory=list) # Specific precautions from LLM
    environmental_impact: Optional[str] = None
    hazard_rating: Optional[int] = None # 1-10
    solubility_rating: Optional[int] = None # 1-10
    pubchem_full_json: Optional[Dict[str, Any]] = None


# --- ChemicalAnalysisAgent Class ---
class ChemicalAnalysisAgent:
    PRICING_FILE_PRIMARY = "pricing_data.json"
    PRICING_FILE_SECONDARY = "second_source.json"
    PRICING_FILE_TERTIARY = "sigma_source.json"
    USD_TO_INR_RATE = 83.0

    def __init__(self,
                 openai_api_key_val: Optional[str] = OPENAI_API_KEY,
                 perplexity_api_key_val: Optional[str] = PERPLEXITY_API_KEY,
                 llm_provider_val: str = DEFAULT_LLM_PROVIDER):

        self.llm_provider = llm_provider_val
        self.openai_api_key_val = openai_api_key_val
        self.perplexity_api_key_val = perplexity_api_key_val
        self.openai_client = None

        if self.llm_provider == "openai":
            if not self.openai_api_key_val:
                print("Warning: OpenAI API key is not available. OpenAI calls by ChemicalAnalysisAgent will fail.")
            elif 'openai' in sys.modules: # Check if openai module was imported
                try:
                    self.openai_client = openai.OpenAI(api_key=self.openai_api_key_val)
                except Exception as e:
                    print(f"Error initializing OpenAI client for ChemicalAnalysisAgent: {e}")
                    self.openai_client = None
            else:
                 print("Warning: OpenAI library could not be imported for ChemicalAnalysisAgent.")
        elif self.llm_provider == "perplexity":
            if not self.perplexity_api_key_val:
                print("Warning: Perplexity API key is not available. Perplexity calls by ChemicalAnalysisAgent will fail.")
        else:
            print(f"Warning: Unsupported LLM provider '{self.llm_provider}' for ChemicalAnalysisAgent. Defaulting to no LLM.")
            self.llm_provider = None
        self.pricing_sources = []
        self._load_all_pricing_data()

    def _load_single_pricing_source(self, filename: str, source_display_name: str) -> None:
        pricing_file_path = os.path.join(PROJECT_ROOT_DIR, filename)
        try:
            with open(pricing_file_path, 'r', encoding='utf-8') as f: data = json.load(f)
            self.pricing_sources.append({"name": source_display_name, "filename": filename, "data": data})
        except FileNotFoundError: self.pricing_sources.append({"name": source_display_name, "filename": filename, "data": {}})
        except json.JSONDecodeError: self.pricing_sources.append({"name": source_display_name, "filename": filename, "data": {}})
        except Exception: self.pricing_sources.append({"name": source_display_name, "filename": filename, "data": {}})

    def _load_all_pricing_data(self) -> None:
        self.pricing_sources = []
        self._load_single_pricing_source(self.PRICING_FILE_PRIMARY, "Primary Local Data")
        self._load_single_pricing_source(self.PRICING_FILE_SECONDARY, "Secondary Local Data")
        self._load_single_pricing_source(self.PRICING_FILE_TERTIARY, "Tertiary Local Data (Sigma)")

    def _is_cas_number(self, identifier: str) -> bool: return bool(re.match(r'^\d{2,7}-\d{2}-\d$', identifier))
    def _is_smiles_like(self, identifier: str) -> bool:
        if " " in identifier.strip() and len(identifier.strip().split()) > 1: return False
        smiles_chars = set("()[]=#@+-.0123456789" + "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        return len(set(identifier) - smiles_chars) < 3 and len(identifier) > 2 and any(c in identifier for c in "()[]=#@+-.")

    def _get_pubchem_data(self, chemical_identifier: str) -> Tuple[Optional[pcp.Compound], Optional[Dict[str, Any]]]:
        compound: Optional[pcp.Compound] = None; full_json_data: Optional[Dict[str, Any]] = None
        search_methods = []
        if self._is_cas_number(chemical_identifier): search_methods.append({'id': chemical_identifier, 'namespace': 'cas', 'type': 'CAS'})
        if self._is_smiles_like(chemical_identifier): search_methods.append({'id': chemical_identifier, 'namespace': 'smiles', 'type': 'SMILES'})
        search_methods.append({'id': chemical_identifier, 'namespace': 'name', 'type': 'Name'})
        for method in search_methods:
            if compound: break
            try:
                compounds = pcp.get_compounds(method['id'], method['namespace'])
                if compounds:
                    compound = compounds[0]
                    try:
                        response = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{compound.cid}/JSON")
                        response.raise_for_status(); full_json_data = response.json()
                    except (requests.RequestException, json.JSONDecodeError): pass # Error logged by caller if needed
                    break
            except (pcp.PubChemHTTPError, Exception): pass
        return compound, full_json_data

    def _extract_ghs_from_pubchem_json(self, pubchem_json: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
        ghs_hazards_list = []
        if not pubchem_json: return ghs_hazards_list
        try:
            record = pubchem_json.get('Record', {}); sections = record.get('Section', [])
            safety_section = next((s for s in sections if s.get('TOCHeading') == 'Safety and Hazards'), None)
            if not safety_section: return ghs_hazards_list
            haz_id_section = next((s for s in safety_section.get('Section', []) if s.get('TOCHeading') == 'Hazards Identification'), None)
            if not haz_id_section: return ghs_hazards_list
            ghs_class_section = next((s for s in haz_id_section.get('Section', []) if s.get('TOCHeading') == 'GHS Classification'), None)
            if not ghs_class_section: return ghs_hazards_list
            information = ghs_class_section.get('Information', []); pictograms_desc = {}
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
                        if statement: ghs_hazards_list.append({"pictogram_description": pictogram_display, "statement": f"{h_code}: {statement}".strip().lstrip(": ")}) # Changed key
        except Exception: pass # Error logged by caller if needed
        return ghs_hazards_list

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
            del best_match["len"]; return best_match
        return None

    def _get_pricing_from_all_local_sources(self, in_id: str, iupac: Optional[str], commons: List[str], smiles: Optional[str]) -> Optional[Dict[str, Any]]:
        in_id_n, iupac_n, commons_n = in_id.lower().strip(), (iupac.lower().strip() if iupac else None), [c.lower().strip() for c in commons if c]
        for src in self.pricing_sources:
            if not src.get("data"): continue
            match = self._search_single_local_source(src["data"], in_id_n, iupac_n, commons_n, smiles)
            if match:
                match.update({"source_file_display_name": src['name'], "source_filename": src['filename']})
                return match
        return None

    def _get_llm_completion_with_fallback(self, system_prompt: str, user_prompt: str, model_openai="gpt-4o", model_perplexity="llama-3-sonar-large-32k-online", max_tokens=2000) -> Optional[str]:
        """Tries preferred provider, then falls back if key is missing or call fails."""
        providers_to_try = []
        if self.llm_provider == "openai": providers_to_try = ["openai", "perplexity"]
        elif self.llm_provider == "perplexity": providers_to_try = ["perplexity", "openai"]
        else: providers_to_try = ["openai", "perplexity"] # Default order if main provider is None

        for provider_name in providers_to_try:
            if provider_name == "openai" and self.openai_client:
                try:
                    response = self.openai_client.chat.completions.create(
                        model=model_openai, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                        temperature=0.1, max_tokens=max_tokens)
                    return response.choices[0].message.content
                except Exception as e: print(f"[OpenAI LLM Error] {e}. Trying fallback.")
            elif provider_name == "perplexity" and self.perplexity_api_key_val:
                payload = {"model": model_perplexity, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]}
                headers = {"Authorization": f"Bearer {self.perplexity_api_key_val}", "Content-Type": "application/json", "Accept": "application/json"}
                try:
                    response = requests.post("https://api.perplexity.ai/chat/completions", json=payload, headers=headers, timeout=120)
                    response.raise_for_status()
                    return response.json()['choices'][0]['message']['content']
                except Exception as e: print(f"[Perplexity LLM Error] {e}. Trying fallback.")
        print("[LLM Error] All LLM providers failed or are not configured.")
        return None

    def _parse_llm_json_response(self, llm_response_content: Optional[str], default_on_error: Dict = None) -> Optional[Dict]:
        if default_on_error is None:
            default_on_error = {}
        if not llm_response_content:
            return default_on_error
        try:
            json_str_to_parse = llm_response_content
            match_json = re.search(r"```json\s*([\s\S]*?)\s*```", llm_response_content, re.IGNORECASE)
            if match_json:
                json_str_to_parse = match_json.group(1).strip()
            else: # If no markdown code block, try to find JSON object directly
                first_brace = json_str_to_parse.find('{')
                last_brace = json_str_to_parse.rfind('}')
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    json_str_to_parse = json_str_to_parse[first_brace : last_brace+1]
                else: # Could not find valid JSON structure
                    print(f"[LLM JSON Parse WARN] Could not reliably find JSON object in: {llm_response_content[:100]}...")
                    return default_on_error
            return json.loads(json_str_to_parse)
        except json.JSONDecodeError as e:
            print(f"[LLM JSON Parse Error] Failed to decode JSON: {e}. Response: {llm_response_content[:200]}...")
            return default_on_error

    def _get_llm_derived_pricing(self, chemical_name: str, smiles: Optional[str], formula: Optional[str], cid: Optional[int]) -> Optional[Dict[str, Any]]:
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
        llm_response_content = self._get_llm_completion_with_fallback(system_prompt, user_prompt, max_tokens=500)
        parsed_data = self._parse_llm_json_response(llm_response_content)
        if not parsed_data: return None
        
        try:
            price_value_usd = parsed_data.get("estimated_price_per_kg_usd"); price_value_inr = parsed_data.get("estimated_price_per_kg_inr")
            final_price_inr = None; llm_provided_currency = None
            def parse_price_value(val):
                if isinstance(val, (int, float)): return float(val)
                if isinstance(val, str):
                    cleaned_val = val.replace(',', '') # Remove commas from numbers like "1,000"
                    match_r = re.match(r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)", cleaned_val) # Range "100-150"
                    if match_r: return (float(match_r.group(1)) + float(match_r.group(2))) / 2
                    match_s = re.match(r"(\d+\.?\d*)", cleaned_val) # Single number "100" or "100.50"
                    if match_s: return float(match_s.group(1))
                return None

            if price_value_inr is not None:
                parsed_val = parse_price_value(price_value_inr)
                if parsed_val is not None: final_price_inr, llm_provided_currency = parsed_val, "INR"
            
            if final_price_inr is None and price_value_usd is not None:
                parsed_val = parse_price_value(price_value_usd)
                if parsed_val is not None: final_price_inr, llm_provided_currency = parsed_val * self.USD_TO_INR_RATE, "USD"
            
            if final_price_inr is not None:
                return {"price_inr": final_price_inr, "currency_llm_provided": llm_provided_currency,
                        "raw_llm_price_value": price_value_inr if llm_provided_currency == "INR" else price_value_usd,
                        "confidence": parsed_data.get("price_confidence"), "basis_notes": parsed_data.get("price_basis_notes"),
                        "source_type": f"LLM ({self.llm_provider or 'fallback'})"}
            
            return {"price_inr": None, "currency_llm_provided": None, "raw_llm_price_value": None,
                    "confidence": parsed_data.get("price_confidence", "low_no_price_data"),
                    "basis_notes": parsed_data.get("price_basis_notes", "LLM no price or format issue."),
                    "source_type": f"LLM ({self.llm_provider or 'fallback'})"}

        except (ValueError, TypeError) as e:
            print(f"[LLM Pricing Parse Error] Error processing parsed LLM price data: {e}. Data: {parsed_data}")
            return None


    def _parse_int_score(self, val: Any, field: str) -> Optional[int]:
        if val is None: return None
        try: return int(val)
        except (ValueError, TypeError): return None

    def _get_llm_derived_properties(self, name: str, formula: Optional[str], smiles: Optional[str],
                                   cid: Optional[int], cas: Optional[str], mw: Optional[float],
                                   iupac: Optional[str]) -> Dict[str, Any]:
        context_parts = [f"Chemical: {name}."]; known_info = []
        if cid: known_info.append(f"PubChem CID: {cid}")
        if iupac: known_info.append(f"IUPAC Name: {iupac}") 
        if formula: known_info.append(f"Formula: {formula}")
        if mw: known_info.append(f"MW: {mw:.2f}" if isinstance(mw, float) else f"MW: {mw}")
        if smiles: known_info.append(f"SMILES: {smiles}")
        if cas: known_info.append(f"CAS: {cas}")

        if known_info: context_parts.append(f"Known info: {'; '.join(known_info)}.")
        else: context_parts.append("No definitive structural or ID information known from databases yet.")
        full_context = "\n".join(context_parts); guess_instr = ""
        if not iupac: guess_instr += '"iupac_name_llm_guess": "Guess IUPAC name or null",\n'
        if not formula: guess_instr += '"molecular_formula_llm_guess": "Guess formula or null",\n'
        if not mw: guess_instr += '"molecular_weight_llm_guess": "Guess MW (float) or null",\n'
        if not cas: guess_instr += '"cas_number_llm_guess": "Guess CAS (XXX-XX-X) or null",\n'
        if not smiles: guess_instr += '"smiles_llm_guess": "Guess SMILES or null",\n'

        system_prompt = "You are a chemical safety, properties, and environmental expert. Provide accurate assessments in JSON format. If guessing core chemical identity, clearly indicate it (e.g., field ends with '_llm_guess')."
        user_prompt = f"""{full_context}
Provide analysis in JSON. Ratings 1-10 (10=high/extreme/excellent) or null. If core ID (IUPAC, formula, MW, CAS, SMILES) unknown, guess it.
Output JSON format:
{{{{  # Start of main JSON object
    {guess_instr}
    "solubility": {{{{ "water_solubility": "Water Soluble/Organic Soluble/Poorly Soluble/Insoluble/Unknown", "organic_solvents_compatibility": ["alcohols", "ethers", "..."], "notes_on_solubility": "notes", "solubility_rating": "integer 1-10 or null" }}}},
    "hazards": {{{{ "corrosive": true/false/null, "flammable": true/false/null, "toxic": true/false/null, "carcinogenic_suspected": true/false/null, "environmental_hazard_notes": "notes", "overall_hazard_level": "Low/Moderate/High/Extreme/Unknown", "hazard_rating": "integer 1-10 or null", "ghs_info_llm": [ {{{{ "pictogram_description": "Pictogram Name", "h_code": "HXXX", "h_statement": "Full statement" }}}}] }}}},
    "safety_precautions": ["list of key safety measures"],
    "storage_recommendations": "storage conditions", "disposal_considerations": "disposal notes",
    "green_chemistry": {{{{ "renewable_feedstock_potential": "yes/no/unknown", "atom_economy_typical_reactions": "high/low/unknown", "biodegradability_assessment": "readily/poorly/unknown", "energy_efficiency_synthesis": "high/low/unknown", "waste_generation_typical_reactions": "high/low/unknown", "overall_score": "integer 1-10 or null" }}}},
    "environmental_impact_summary": "overall assessment"
}}}} # End of main JSON object
Use empty list [] for 'ghs_info_llm' if none. For others, use null or "Unknown".
"""
        llm_response_content = self._get_llm_completion_with_fallback(system_prompt, user_prompt, max_tokens=2500)
        default_empty_response = {"solubility": {"water_solubility": SolubilityType.UNKNOWN.value, "solubility_rating": None},
                                  "hazards": {"overall_hazard_level": HazardLevel.UNKNOWN.value, "hazard_rating": None, "ghs_info_llm": []},
                                  "safety_precautions": [], "green_chemistry": {"overall_score": None},
                                  "environmental_impact_summary": "Assessment unavailable due to LLM error."}
        
        parsed_data = self._parse_llm_json_response(llm_response_content, default_empty_response)
        return parsed_data


    def analyze_chemical(self, chemical_identifier: str) -> ChemicalProperties:
        props = ChemicalProperties(name=chemical_identifier, original_query=chemical_identifier)
        pubchem_compound, pubchem_full_json = self._get_pubchem_data(chemical_identifier)
        props.pubchem_full_json = pubchem_full_json
        current_name_for_llm = chemical_identifier
        if pubchem_compound:
            props.name = pubchem_compound.iupac_name or (pubchem_compound.synonyms[0] if pubchem_compound.synonyms else chemical_identifier)
            current_name_for_llm = props.name
            props.iupac_name = pubchem_compound.iupac_name
            props.common_names = list(pubchem_compound.synonyms[:10]) if pubchem_compound.synonyms else []
            if chemical_identifier not in props.common_names and chemical_identifier != props.name and chemical_identifier != props.iupac_name:
                props.common_names.insert(0, chemical_identifier)

            props.molecular_formula = pubchem_compound.molecular_formula
            props.molecular_weight = float(pubchem_compound.molecular_weight) if pubchem_compound.molecular_weight else None
            if pubchem_compound.synonyms:
                cas_syns = [s for s in pubchem_compound.synonyms if self._is_cas_number(s)]
                if cas_syns: props.cas_number = cas_syns[0]
            props.smiles = pubchem_compound.canonical_smiles
            props.pubchem_cid = pubchem_compound.cid
            props.ghs_hazards = self._extract_ghs_from_pubchem_json(props.pubchem_full_json)

        if not current_name_for_llm: current_name_for_llm = chemical_identifier 
        llm_derived_info = self._get_llm_derived_properties(current_name_for_llm, props.molecular_formula, props.smiles, props.pubchem_cid, props.cas_number, props.molecular_weight, props.iupac_name)

        if not pubchem_compound: 
            props.iupac_name = props.iupac_name or llm_derived_info.get("iupac_name_llm_guess")
            if props.name == chemical_identifier and props.iupac_name:
                props.name = props.iupac_name; current_name_for_llm = props.name
            props.molecular_formula = props.molecular_formula or llm_derived_info.get("molecular_formula_llm_guess")
            mw_g = llm_derived_info.get("molecular_weight_llm_guess")
            if mw_g is not None and props.molecular_weight is None:
                try: props.molecular_weight = float(mw_g)
                except (ValueError,TypeError): pass
            cas_g = llm_derived_info.get("cas_number_llm_guess")
            if cas_g and self._is_cas_number(cas_g) and props.cas_number is None: props.cas_number = cas_g
            if props.smiles is None: props.smiles = llm_derived_info.get("smiles_llm_guess")


        props.solubility = llm_derived_info.get("solubility", props.solubility)
        llm_hazards = llm_derived_info.get("hazards", {})
        haz_lvl_str = llm_hazards.get("overall_hazard_level", "unknown").lower()
        props.hazard_level = HazardLevel[haz_lvl_str.upper()] if haz_lvl_str.upper() in HazardLevel.__members__ else HazardLevel.UNKNOWN
        props.is_corrosive = llm_hazards.get("corrosive", props.is_corrosive)
        props.is_flammable = llm_hazards.get("flammable", props.is_flammable)
        props.is_toxic = llm_hazards.get("toxic", props.is_toxic)

        if not props.ghs_hazards and "ghs_info_llm" in llm_hazards: 
            for item in llm_hazards.get("ghs_info_llm", []):
                if isinstance(item, dict) and item.get("h_statement"): 
                    props.ghs_hazards.append({
                        "pictogram_description": item.get("pictogram_description", "N/A"),
                        "statement": f"{item.get('h_code', '')}: {item.get('h_statement', '')}".strip().lstrip(": ")
                    })

        props.safety_notes = llm_derived_info.get("safety_precautions", props.safety_notes) 
        props.environmental_impact = llm_derived_info.get("environmental_impact_summary", props.environmental_impact)
        gc_score = llm_derived_info.get("green_chemistry", {}).get("overall_score")
        props.green_chemistry_score = self._parse_int_score(gc_score, "GC score") or props.green_chemistry_score
        props.hazard_rating = self._parse_int_score(llm_hazards.get("hazard_rating"), "Hazard rating") or props.hazard_rating
        sol_rating = llm_derived_info.get("solubility", {}).get("solubility_rating")
        props.solubility_rating = self._parse_int_score(sol_rating, "Solubility rating") or props.solubility_rating

        local_price = self._get_pricing_from_all_local_sources(chemical_identifier, props.iupac_name, props.common_names, props.smiles)
        if local_price:
            props.estimated_price_per_kg = local_price.get("price")
            props.price_currency = local_price.get("currency", "INR")
            props.supplier_info = [{"name": f"Local DB: {local_price.get('source_file_display_name','N/A')} (...)",
                                    "availability": (f"Price: {props.estimated_price_per_kg:.2f} {props.price_currency}. Match: {local_price.get('match_type','N/A')}."),
                                    "location": local_price.get('location', 'Unknown Location'), "source_type": "Local JSON"}]
        else:
            if not current_name_for_llm: current_name_for_llm = chemical_identifier 
            llm_price_info = self._get_llm_derived_pricing(current_name_for_llm, props.smiles, props.molecular_formula, props.pubchem_cid)
            if llm_price_info and llm_price_info.get("price_inr") is not None:
                props.estimated_price_per_kg = llm_price_info["price_inr"]; props.price_currency = "INR"
                availability_details = f"Est. Price: {props.estimated_price_per_kg:.2f} INR/kg. (LLM: {llm_price_info.get('raw_llm_price_value')} {llm_price_info.get('currency_llm_provided')}, Conf: {llm_price_info.get('confidence')})"
                props.supplier_info = [{"name": f"LLM Estimation ({self.llm_provider or 'fallback'})", "availability": availability_details,
                                        "location": "Global Market (Est.)", "source_type": f"LLM ({self.llm_provider or 'fallback'})"}]
            else:
                props.estimated_price_per_kg = None; props.price_currency = None
                availability_note = "Not in local DBs. LLM pricing inconclusive."
                if llm_price_info: availability_note += f" (Conf: {llm_price_info.get('confidence','N/A')}, Basis: {llm_price_info.get('basis_notes','N/A')})"
                props.supplier_info = [{"name": "No Definitive Pricing Data", "availability": availability_note,
                                        "location": "N/A", "source_type": "None"}]
        return props

    def generate_report(self, props: ChemicalProperties) -> str:
        """Generates a comprehensive text report from ChemicalProperties."""
        report_parts = [f"Chemical Report: {props.name}\n" + "=" * (17 + len(props.name))]
        if props.original_query and props.original_query != props.name:
            report_parts.append(f"Original Query: {props.original_query}")
        
        core_info = [
            ("IUPAC Name", props.iupac_name),
            ("Common Names", ", ".join(props.common_names) if props.common_names else "N/A"),
            ("Molecular Formula", props.molecular_formula),
            ("Molecular Weight", f"{props.molecular_weight:.2f} g/mol" if props.molecular_weight else None),
            ("CAS Number", props.cas_number),
            ("SMILES", props.smiles),
            ("PubChem CID", props.pubchem_cid)
        ]
        report_parts.append("\n--- Core Identification ---")
        for label, value in core_info:
            if value is not None: report_parts.append(f"{label}: {value}")

        report_parts.append("\n--- Physicochemical Properties ---")
        report_parts.append(f"Solubility in Water: {props.solubility.get('water_solubility', 'Unknown')}")
        if props.solubility.get('organic_solvents_compatibility'):
            report_parts.append(f"Organic Solvents Compatibility: {', '.join(props.solubility['organic_solvents_compatibility'])}")
        if props.solubility.get('notes_on_solubility'):
             report_parts.append(f"Solubility Notes: {props.solubility['notes_on_solubility']}")
        if props.solubility_rating is not None:
             report_parts.append(f"Solubility Rating (1-10): {props.solubility_rating}/10")


        report_parts.append("\n--- Hazard Information ---")
        report_parts.append(f"Overall Hazard Level: {props.hazard_level.value}")
        if props.hazard_rating is not None:
            report_parts.append(f"Hazard Rating (1-10): {props.hazard_rating}/10")
        if props.is_corrosive is not None: report_parts.append(f"Corrosive: {'Yes' if props.is_corrosive else 'No'}")
        if props.is_flammable is not None: report_parts.append(f"Flammable: {'Yes' if props.is_flammable else 'No'}")
        if props.is_toxic is not None: report_parts.append(f"Toxic: {'Yes' if props.is_toxic else 'No'}")
        
        if props.ghs_hazards:
            report_parts.append("GHS Hazard Statements:")
            for ghs in props.ghs_hazards[:5]: # Max 5 for brevity in text report
                report_parts.append(f"  - Pictogram: {ghs.get('pictogram_description', 'N/A')}, Statement: {ghs.get('statement', 'N/A')}")
        else:
            report_parts.append("GHS Hazard Statements: Not available or not found.")

        if props.safety_notes:
            report_parts.append("Key Safety Precautions (LLM Suggested):")
            for note in props.safety_notes: report_parts.append(f"  - {note}")
        
        report_parts.append("\n--- Economic Information ---")
        if props.estimated_price_per_kg is not None and props.price_currency:
            report_parts.append(f"Estimated Price: {props.estimated_price_per_kg:.2f} {props.price_currency}/kg")
            if props.supplier_info:
                source_desc = props.supplier_info[0].get('name', 'N/A')
                availability = props.supplier_info[0].get('availability', 'N/A')
                report_parts.append(f"Price Source: {source_desc} ({availability})")
        else:
            report_parts.append("Estimated Price: Not available")
            if props.supplier_info: # Even if no price, supplier might have notes
                 report_parts.append(f"Pricing/Supplier Notes: {props.supplier_info[0].get('availability', 'No specific notes.')}")


        report_parts.append("\n--- Environmental & Green Chemistry ---")
        if props.environmental_impact:
            report_parts.append(f"Environmental Impact Summary: {props.environmental_impact}")
        if props.green_chemistry_score is not None:
            report_parts.append(f"Green Chemistry Score (1-10): {props.green_chemistry_score}/10")
        
        # Add more details from llm_derived_info.get("green_chemistry", {}) if needed.
        # e.g. props.llm_derived_info.get("green_chemistry", {}).get("biodegradability_assessment")
        
        # report_parts.append("\n--- Full PubChem JSON (Abbreviated) ---")
        # if props.pubchem_full_json:
        #     report_parts.append(json.dumps(props.pubchem_full_json, indent=2)[:1000] + "\n...(truncated)")
        # else:
        #     report_parts.append("Not available.")
            
        return "\n".join(report_parts)

# --- Autogen LLM Configuration & Global Setup ---
config_list_gpt4o = [{"model": "gpt-4o", "api_key": OPENAI_API_KEY}]
config_list_gpt3_5_turbo = [{"model": "gpt-3.5-turbo-0125", "api_key": OPENAI_API_KEY}]
llm_config_chatbot_agent = {"config_list": config_list_gpt3_5_turbo, "temperature": 0.1, "timeout": 90}
REACTION_ANALYSIS_OUTPUT_DIR = os.path.join(PROJECT_ROOT_DIR, "reaction_analysis_outputs")
os.makedirs(REACTION_ANALYSIS_OUTPUT_DIR, exist_ok=True)
dataset_path1 = os.environ.get('REACTION_DATASET_PATH1', None)
dataset_path2 = os.environ.get('REACTION_DATASET_PATH2', None)

try:
    from tools import autogen_tool_functions as ag_tools
    from tools.asckos import ReactionClassifier
except ImportError: ag_tools = None; ReactionClassifier = None # Handled in main
reaction_classifier_core_logic = ReactionClassifier(dataset_path1, dataset_path2) if ReactionClassifier and dataset_path1 else None
reaction_cache = {}; compound_cache = {}

# --- Utility, Tool Wrappers, Data Extraction ---
def sanitize_filename(name):
    if not isinstance(name, str): name = str(name)
    return re.sub(r'[^\w\.\-]+', '_', name)[:100]

def save_analysis_to_file(entity_identifier, analysis_data, query_context_type="analysis", original_name=None):
    if not analysis_data: return
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    id_part = sanitize_filename(entity_identifier if entity_identifier else "no_id")
    is_reaction = ">>" in entity_identifier if entity_identifier else False
    prefix = "rxn_" if is_reaction else "cmpd_"
    filename_parts = []
    if original_name and original_name != entity_identifier: filename_parts.append(sanitize_filename(original_name))
    filename_parts.extend([f"{prefix}{id_part}", sanitize_filename(query_context_type), timestamp])
    
    if isinstance(analysis_data, dict): 
        filename = "_".join(filter(None, filename_parts)) + ".json"
        filepath = os.path.join(REACTION_ANALYSIS_OUTPUT_DIR, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(analysis_data, f, indent=2)
        except Exception as e: print(f"[SAVE_JSON_ERROR] Error saving {filepath}: {e}")
    elif isinstance(analysis_data, str): 
        filename = "_".join(filter(None, filename_parts)) + ".txt"
        filepath = os.path.join(REACTION_ANALYSIS_OUTPUT_DIR, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"Entity: {entity_identifier}\nContext: {query_context_type}\nTimestamp: {timestamp}\n\n{analysis_data}")
        except Exception as e: print(f"[SAVE_TXT_ERROR] Error saving {filepath}: {e}")
    else:
        print(f"[SAVE_ANALYSIS_ERROR] Unknown data type for analysis: {type(analysis_data)}")


@lru_cache(maxsize=100)
def query_reaction_dataset(reaction_smiles):
    if not reaction_smiles: return None
    if reaction_smiles in reaction_cache and 'dataset_info' in reaction_cache[reaction_smiles]:
        return reaction_cache[reaction_smiles]['dataset_info']
    
    current_classifier = reaction_classifier_core_logic # This is your ReactionClassifier instance
    if not current_classifier: return None

    # Determine which dataset is likely being used or prioritize one
    # This logic might need refinement based on how ReactionClassifier loads/accesses its data
    df = None
    dataset_type = None # 'orderly' or 'classification'

    if hasattr(current_classifier, 'dataset1') and current_classifier.dataset1 is not None and not current_classifier.dataset1.empty:
        df = current_classifier.dataset1
        # Check for a distinctive column to guess dataset type
        if 'date_of_experiment' in df.columns or 'agent_000' in df.columns:
            dataset_type = 'orderly'
        elif 'reaction_classnum' in df.columns:
            dataset_type = 'classification'
        else: # Fallback if no distinctive column, assume 'orderly' if it's the primary one
            dataset_type = 'orderly' 
            
    elif hasattr(current_classifier, 'dataset2') and current_classifier.dataset2 is not None and not current_classifier.dataset2.empty:
        df = current_classifier.dataset2
        if 'reaction_classnum' in df.columns:
            dataset_type = 'classification'
        elif 'date_of_experiment' in df.columns or 'agent_000' in df.columns: # Less likely but for completeness
            dataset_type = 'orderly'
        else:
            dataset_type = 'classification' # Fallback for dataset2
            
    if df is None or df.empty: return None

    smiles_columns_to_check = ['rxn_str', 'reaction_smiles', 'smiles', 'rxn_smiles'] # General SMILES columns
    
    exact_match_row = None
    for col_name in smiles_columns_to_check:
        if col_name in df.columns and df[col_name].dtype == 'object':
            # Try exact match
            temp_match_df = df[df[col_name] == reaction_smiles]
            if not temp_match_df.empty:
                exact_match_row = temp_match_df.iloc[0]
                break
            # Try matching core reaction (Reactants>>Products) if query has agents
            if ">" in reaction_smiles and reaction_smiles.count('>') == 2: # R>A>>P format
                 core_rxn_parts = reaction_smiles.split('>')
                 core_rxn_smiles = f"{core_rxn_parts[0]}>>{core_rxn_parts[2]}"
                 temp_match_core_df = df[df[col_name] == core_rxn_smiles]
                 if not temp_match_core_df.empty:
                     exact_match_row = temp_match_core_df.iloc[0]
                     break
            elif ">>" in reaction_smiles and ">" not in reaction_smiles.split(">>")[0] and ">" not in reaction_smiles.split(">>")[1]: # R>>P format
                # Try matching if dataset has R>A>>P by checking just R and P parts
                # This is more complex and might lead to false positives if not careful
                pass # Skipping this more complex partial match for now to avoid ambiguity


    result = {}
    if exact_match_row is not None:
        # Fields common or potentially common
        if 'procedure_details' in exact_match_row.index and pd.notna(exact_match_row['procedure_details']):
            result['procedure_details'] = str(exact_match_row['procedure_details'])
        if 'rxn_time' in exact_match_row.index and pd.notna(exact_match_row['rxn_time']):
            result['rxn_time'] = str(exact_match_row['rxn_time'])
        if 'temperature' in exact_match_row.index and pd.notna(exact_match_row['temperature']):
            result['temperature'] = str(exact_match_row['temperature'])
        if 'yield_000' in exact_match_row.index and pd.notna(exact_match_row['yield_000']):
             result['yield_000'] = str(exact_match_row['yield_000'])
        if 'atmosphere' in exact_match_row.index and pd.notna(exact_match_row['atmosphere']): # Orderly dataset has 'atmosphere'
            result['atmosphere'] = str(exact_match_row['atmosphere'])

        # Orderly specific agent/solvent extraction (agent_000, solvent_000)
        if dataset_type == 'orderly':
            agents = []
            for i in range(16): # agent_000 to agent_015
                agent_col = f'agent_{i:03d}'
                if agent_col in exact_match_row.index and pd.notna(exact_match_row[agent_col]):
                    agents.append(str(exact_match_row[agent_col]))
            if agents: result['agents_list'] = agents

            solvents = []
            for i in range(11): # solvent_000 to solvent_010
                solvent_col = f'solvent_{i:03d}'
                if solvent_col in exact_match_row.index and pd.notna(exact_match_row[solvent_col]):
                    solvents.append(str(exact_match_row[solvent_col]))
            if solvents: result['solvents_list'] = solvents
        
        # Classification specific fields
        if dataset_type == 'classification':
            if 'reaction_name' in exact_match_row.index and pd.notna(exact_match_row['reaction_name']):
                result['reaction_name'] = str(exact_match_row['reaction_name'])
            if 'reaction_classname' in exact_match_row.index and pd.notna(exact_match_row['reaction_classname']):
                result['reaction_classname'] = str(exact_match_row['reaction_classname'])
        
        # Fallback for agents/solvents if not picked up by orderly specific logic but columns exist (e.g. from dataset2 if it has them)
        if 'agents_list' not in result and 'agents_list' in exact_match_row.index and pd.notna(exact_match_row['agents_list']):
            val = exact_match_row['agents_list']
            if isinstance(val, str): result['agents_list'] = ast.literal_eval(val) if val.startswith('[') else [val]
            elif isinstance(val, list): result['agents_list'] = val
        if 'solvents_list' not in result and 'solvents_list' in exact_match_row.index and pd.notna(exact_match_row['solvents_list']):
            val = exact_match_row['solvents_list']
            if isinstance(val, str): result['solvents_list'] = ast.literal_eval(val) if val.startswith('[') else [val]
            elif isinstance(val, list): result['solvents_list'] = val


    reaction_cache.setdefault(reaction_smiles, {})['dataset_info'] = result if result else None
    return result if result else None


def extract_reaction_smiles(query: str) -> Optional[str]: 
    smi_core_chars = r"[\w@\[\]\(\)\.\+\-\=\#\:\$\%\~\<\>]" 
    explicit_pattern = rf"(?i:\b(?:reaction\s+smiles|rxn)\s*[:=]?\s*)({smi_core_chars}+(?:>>{smi_core_chars}+)+)"
    match = re.search(explicit_pattern, query)
    if match:
        smiles = match.group(1).strip(); parts = smiles.split(">>")
        if len(parts) >= 2 and all(p.strip() for p in parts): return smiles
    standalone_pattern_strict_double_arrow = rf"(?<![\w\/])({smi_core_chars}+(?:>>{smi_core_chars}+)+)(?![\w\/])"
    potential_matches = re.findall(standalone_pattern_strict_double_arrow, query)
    for smi_candidate in potential_matches:
        smi_candidate = smi_candidate.strip(); parts = smi_candidate.split(">>")
        if len(parts) >= 2 and all(p.strip() for p in parts):
            try:
                if Chem.MolFromSmiles(parts[0].split('.')[0]) and Chem.MolFromSmiles(parts[-1].split('.')[0]): return smi_candidate
            except: pass
    gt_pattern_general = rf"(?<![\w\/])({smi_core_chars}+(?:>{smi_core_chars}+)+)(?![\w\/])"
    match_gt = re.search(gt_pattern_general, query)
    if match_gt:
        temp_smiles = match_gt.group(1).strip()
        if ">>" not in temp_smiles:
            gt_parts = temp_smiles.split('>'); cleaned_gt_parts = [p.strip() for p in gt_parts if p.strip()]
            if len(cleaned_gt_parts) >= 2:
                products_gt = cleaned_gt_parts[-1]; reactants_gt = ".".join(cleaned_gt_parts[:-1])
                if reactants_gt and products_gt:
                    try:
                        if Chem.MolFromSmiles(reactants_gt.split('.')[0]) and Chem.MolFromSmiles(products_gt.split('.')[0]):
                            return f"{reactants_gt}>>{products_gt}"
                    except: pass
    return None

def extract_single_compound_smiles(query: str) -> Optional[str]: 
    words = query.split(); regex_candidates = re.findall(r"[A-Za-z0-9@\[\]\(\)\+\-\=\#\:\.\$\%\/\\\{\}]{3,}", query)
    combined_candidates = list(set(words + regex_candidates)); combined_candidates.sort(key=lambda x: (len(x), sum(1 for c in x if c in '()[]=#')), reverse=True)
    for s_cand in combined_candidates:
        s_cand = s_cand.strip('.,;:)?!\'"')
        if not s_cand or '>>' in s_cand or '>' in s_cand or '<' in s_cand: continue
        if s_cand.isnumeric() and not ('[' in s_cand and ']' in s_cand) : continue
        try:
            mol = Chem.MolFromSmiles(s_cand, sanitize=True)
            if mol:
                num_atoms = mol.GetNumAtoms()
                if num_atoms >= 1:
                    if num_atoms <= 2 and s_cand.isalpha() and s_cand.lower() in ['as', 'in', 'is', 'at', 'or', 'to', 'be', 'of', 'on', 'no', 'do', 'go', 'so', 'if', 'it', 'me', 'my', 'he', 'we', 'by', 'up', 'us', 'an', 'am', 'are']:
                        if not any(c in s_cand for c in '()[]=#.-+@:/\\%{}') and len(s_cand) <=2 : continue
                    if any(c in s_cand for c in '()[]=#.-+@:/\\%{}') or num_atoms > 2 or len(s_cand) > 3: return s_cand
        except Exception: pass
    return None

# --- Helper for Reaction Component Pricing ---
def _get_price_source_confidence_for_reaction_component(
    supplier_info_list: List[Dict[str, Any]],
    llm_provider_for_chem_agent: Optional[str]
) -> Tuple[str, str, Optional[str]]:
    if not supplier_info_list:
        return "N/A", "N/A", None

    first_source_info = supplier_info_list[0]
    source_type = first_source_info.get("source_type", "Unknown")
    source_name_detail = first_source_info.get("name", "Unknown Source")
    raw_llm_confidence = None

    if source_type == "Local JSON":
        display_name_match = re.search(r"Local DB: (.*?)( \(\.\.\.\))?$", source_name_detail)
        confidence_str = "High"
        source_str = "Local Database" # Default
        if display_name_match:
            db_name = display_name_match.group(1)
            if "Sigma" in db_name: source_str = "Sigma-Aldrich Catalog (Local Cache)"
            elif "Primary" in db_name: source_str = "Primary Local Data (Local Cache)"
            elif "Secondary" in db_name: source_str = "Secondary Local Data (Local Cache)"
            else: source_str = f"{db_name} (Local Cache)"
        return source_str, confidence_str, None
        
    elif source_type.startswith("LLM"):
        availability_str = first_source_info.get("availability", "")
        conf_match = re.search(r"Conf:\s*([\w-]+)", availability_str, re.IGNORECASE)
        
        confidence_str_from_llm = "medium" 
        if conf_match:
            raw_llm_confidence = conf_match.group(1).strip()
            confidence_str_from_llm = raw_llm_confidence.lower()

        if confidence_str_from_llm in ["high", "very_high"]: final_confidence_display = "High"
        elif confidence_str_from_llm == "medium": final_confidence_display = "Medium"
        else: final_confidence_display = "Low"
        
        source_str = f"AI Price Estimation ({llm_provider_for_chem_agent or 'LLM'})"
        return source_str, final_confidence_display, raw_llm_confidence

    return source_name_detail, "Unknown", None


# --- Core analysis functions (handle_full_info MODIFIED) ---
def handle_full_info(query_text_for_llm_summary, reaction_smiles_clean, original_compound_name=None, callbacks=None):
    if not reaction_smiles_clean or ">>" not in reaction_smiles_clean:
        return {'visualization_path': None, 'analysis': {"error": f"Invalid reaction SMILES: Must contain '>>'. Input: '{reaction_smiles_clean}'"},
                'analysis_context': "invalid_smiles_input_no_double_arrow", 'processed_smiles_for_tools': reaction_smiles_clean}

    chem_agent = ChemicalAnalysisAgent()
    reactants_smi_list, agents_smi_list, products_smi_list = [], [], []

    # --- Corrected SMILES Parsing Logic (from previous good version) ---
    try:
        if ">>" not in reaction_smiles_clean:
            raise ValueError("Reaction SMILES must contain '>>'")
        main_parts = reaction_smiles_clean.split(">>", 1) 
        if len(main_parts) != 2:
            raise ValueError("Reaction SMILES format error after splitting by '>>'")
        reactants_agents_str = main_parts[0].strip()
        products_str = main_parts[1].strip()
        if ">" in reactants_agents_str and reactants_agents_str.count('>') == 1:
            r_str, a_str = reactants_agents_str.split(">", 1)
            reactants_smi_list = [s.strip() for s in r_str.split('.') if s.strip()]
            agents_smi_list = [s.strip() for s in a_str.split('.') if s.strip()]
        else:
            reactants_smi_list = [s.strip() for s in reactants_agents_str.split('.') if s.strip()]
        products_smi_list = [s.strip() for s in products_str.split('.') if s.strip()]
        if not reactants_smi_list or not products_smi_list:
             raise ValueError("Parsed SMILES resulted in empty reactants or products.")
    except ValueError as e:
        return {'visualization_path': None, 'analysis': {"error": f"SMILES Parsing Error: {e}. Input: '{reaction_smiles_clean}'"},
                'analysis_context': "smiles_parsing_error_detailed", 'processed_smiles_for_tools': reaction_smiles_clean}
    # --- End of Corrected SMILES Parsing Logic ---

    final_json_output = {
        "reaction_smiles_interpreted": reaction_smiles_clean,
        "reaction_details": {
            "reactants_identified": [],  # Will still populate for context, but not in safety_and_notes
            "products_identified": [],   # Will still populate for context, but not in safety_and_notes
            "reaction_name": "N/A",      # Changed from reaction_name_from_classification_dataset
            # "yield_from_dataset" REMOVED from this section
            # "reaction_class_from_classification_dataset" REMOVED
        },
        "reagents_and_solvents": [],
        "reaction_conditions": {
            "temperature_from_dataset": "N/A",
            "time_from_dataset": "N/A",
            "yield_from_dataset": "N/A", # Moved yield here as it's a condition/result
            "atmosphere_llm_or_dataset": "N/A"
        },
        "safety_and_notes": {
            "safety": "N/A", # Changed from overall_reaction_safety_assessment_llm
            "notes": "N/A"   # Changed from notes_llm
            # "reactants_detailed_safety" REMOVED
            # "products_detailed_safety" REMOVED
        },
        "procedure_steps": ["Procedure details not available from dataset."], 
        "visualization_path": None
    }

    # 2. Dataset Query
    dataset_info = query_reaction_dataset(reaction_smiles_clean) 

    raw_procedure_text = None 
    if dataset_info:
        # Populate yield here under reaction_conditions
        final_json_output["reaction_conditions"]["yield_from_dataset"] = f"{dataset_info.get('yield_000')}%" if dataset_info.get('yield_000') and dataset_info.get('yield_000','nan').lower() != 'nan' else "N/A"
        
        final_json_output["reaction_conditions"]["temperature_from_dataset"] = dataset_info.get('temperature') or "N/A"
        final_json_output["reaction_conditions"]["time_from_dataset"] = dataset_info.get('rxn_time') or "N/A"
        if dataset_info.get('atmosphere') and dataset_info.get('atmosphere','nan').lower() != 'nan':
            final_json_output["reaction_conditions"]["atmosphere_llm_or_dataset"] = dataset_info.get('atmosphere')
        
        # Populate reaction_name (formerly reaction_name_from_classification_dataset)
        if dataset_info.get('reaction_name'): 
            final_json_output["reaction_details"]["reaction_name"] = dataset_info.get('reaction_name')
        # reaction_classname is no longer stored as per request

        raw_procedure_text = dataset_info.get('procedure_details') 

    # 3. Visualization (no change)
    if ag_tools:
        try:
            viz_path_result = ag_tools.visualize_chemical_structure(reaction_smiles_clean)
            if viz_path_result and not str(viz_path_result).lower().startswith('error') and ".png" in viz_path_result:
                final_json_output["visualization_path"] = viz_path_result
        except Exception: pass

    # 4. Identify Reactant/Product Names (for LLM context and reaction_details)
    # Detailed safety for individual reactants/products is NO LONGER STORED in final_json_output
    reactant_names_for_llm_context = []
    for r_smi in reactants_smi_list:
        try:
            props = chem_agent.analyze_chemical(r_smi) # Analyze for name resolution
            name_to_add = props.name or props.iupac_name or (props.common_names[0] if props.common_names else r_smi)
            final_json_output["reaction_details"]["reactants_identified"].append(name_to_add)
            reactant_names_for_llm_context.append(name_to_add)
            # No longer appending to safety_and_notes.reactants_detailed_safety
        except Exception as e:
            final_json_output["reaction_details"]["reactants_identified"].append(r_smi)
            reactant_names_for_llm_context.append(r_smi) 

    product_names_for_llm_context = []
    for p_smi in products_smi_list:
        try:
            props = chem_agent.analyze_chemical(p_smi) # Analyze for name resolution
            name_to_add = props.name or props.iupac_name or (props.common_names[0] if props.common_names else p_smi)
            final_json_output["reaction_details"]["products_identified"].append(name_to_add)
            product_names_for_llm_context.append(name_to_add)
            # No longer appending to safety_and_notes.products_detailed_safety
        except Exception as e:
            final_json_output["reaction_details"]["products_identified"].append(p_smi)
            product_names_for_llm_context.append(p_smi)


    # 5. Reagents and Solvents Information (Logic from previous good version - no change needed here for this request)
    all_lhs_identifiers = set() 
    all_lhs_identifiers.update(reactants_smi_list) 
    all_lhs_identifiers.update(agents_smi_list)    
    dataset_agents = dataset_info.get('agents_list', []) if dataset_info else []
    dataset_solvents = dataset_info.get('solvents_list', []) if dataset_info else []
    for da in dataset_agents:
        if isinstance(da, str) and da.strip(): all_lhs_identifiers.add(da.strip())
    for ds in dataset_solvents:
        if isinstance(ds, str) and ds.strip(): all_lhs_identifiers.add(ds.strip())
    llm_context_auxiliary_names_set = set()
    added_to_reagents_output_set = set()

    # For determining primary reactants (those not considered mere auxiliaries)
    # We use the names already put into reactants_identified for this check.
    resolved_primary_reactant_names_from_step4_set = set(final_json_output["reaction_details"]["reactants_identified"])


    COMMON_AUXILIARIES_BY_NAME_LOWER = {
        "thionyl chloride", "hydrochloric acid", "sulfuric acid", "nitric acid", "socl2",
        "sodium hydroxide", "potassium hydroxide", "water", "dichloromethane", "hcl",
        "chloroform", "methanol", "ethanol", "diethyl ether", "tetrahydrofuran", 
        "toluene", "benzene", "dmso", "dmf", "acetonitrile", "acetic acid",
        "sodium bicarbonate", "sodium carbonate", "potassium carbonate", "magnesium sulfate",
        "sodium sulfate", "sodium chloride", "triethylamine", "pyridine", "diisopropylethylamine",
        "hydrogen peroxide", "sodium borohydride", "lithium aluminum hydride"}
    COMMON_AUXILIARIES_BY_SMILES = {
        "O=S(Cl)Cl", "Cl", "OS(=O)(=O)O", "[N+](=O)([O-])O", "O", "[Na+].[OH-]", "[K+].[OH-]",
        "ClCCl", "ClC(Cl)Cl", "CO", "CCO", "CCOCC", "C1COCC1", "Cc1ccccc1", "c1ccccc1",
        "CS(=O)C", "CN(C)C=O", "CC#N"}

    for item_identifier in sorted(list(all_lhs_identifiers)): 
        if item_identifier in products_smi_list: continue
        try:
            props = chem_agent.analyze_chemical(item_identifier)
            name_for_component = props.name or props.iupac_name or (props.common_names[0] if props.common_names else item_identifier)
            smiles_for_component = props.smiles or (item_identifier if chem_agent._is_smiles_like(item_identifier) else None)

            if name_for_component in added_to_reagents_output_set or item_identifier in added_to_reagents_output_set:
                if name_for_component not in resolved_primary_reactant_names_from_step4_set : # If not a primary reactant
                    llm_context_auxiliary_names_set.add(name_for_component)
                continue
            role = "Reagent/Solvent"; add_to_reagents_output_list = False
            is_explicit_smiles_agent = item_identifier in agents_smi_list
            is_dataset_agent_match = item_identifier in dataset_agents or name_for_component in dataset_agents
            is_dataset_solvent_match = item_identifier in dataset_solvents or name_for_component in dataset_solvents
            is_common_aux = (name_for_component.lower() in COMMON_AUXILIARIES_BY_NAME_LOWER or 
                             (smiles_for_component and smiles_for_component in COMMON_AUXILIARIES_BY_SMILES))
            
            # Is it one of the main reactants identified in step 4?
            is_main_reactant_from_step4 = name_for_component in resolved_primary_reactant_names_from_step4_set

            if is_explicit_smiles_agent: role = "Reagent (from SMILES)"; add_to_reagents_output_list = True
            elif is_dataset_agent_match: role = "Reagent (from Dataset)"; add_to_reagents_output_list = True
            elif is_dataset_solvent_match: role = "Solvent (from Dataset)"; add_to_reagents_output_list = True
            elif is_common_aux: # If it's a common auxiliary, it should be in reagents/solvents list
                role = f"Reagent/Solvent (Common Auxiliary)"
                add_to_reagents_output_list = True
            
            if add_to_reagents_output_list:
                price_source, price_confidence, _ = _get_price_source_confidence_for_reaction_component(
                    props.supplier_info, chem_agent.llm_provider)
                final_json_output["reagents_and_solvents"].append({
                    "name": name_for_component, "role": role,
                    "price_per_unit": props.estimated_price_per_kg if props.estimated_price_per_kg is not None else "N/A",
                    "currency": props.price_currency if props.estimated_price_per_kg is not None else "N/A",
                    "unit_basis": "kg" if props.estimated_price_per_kg is not None else "N/A",
                    "price_source": price_source, "price_confidence": price_confidence})
                added_to_reagents_output_set.add(name_for_component)
                added_to_reagents_output_set.add(item_identifier) 
                llm_context_auxiliary_names_set.add(name_for_component) 
            # If not added to reagents list but also not a main reactant from step 4, add to aux context
            elif not is_main_reactant_from_step4:
                 llm_context_auxiliary_names_set.add(name_for_component)
        except Exception as e:
            is_main_reactant_name_check = item_identifier in resolved_primary_reactant_names_from_step4_set
            is_product_name_check = item_identifier in product_names_for_llm_context
            if not is_main_reactant_name_check and not is_product_name_check :
                 llm_context_auxiliary_names_set.add(item_identifier)
            pass
    unique_agent_solvent_names_for_llm_context = sorted(list(llm_context_auxiliary_names_set))
    # --- End of Corrected Step 5 ---


    # 6. LLM for Procedure Formatting & Atmosphere
    if raw_procedure_text: 
        proc_format_prompt_system = "You are a helpful assistant. Reformat chemical procedures into steps and identify atmosphere. Return JSON."
        proc_format_prompt_user = f"""Given the raw chemical procedure text:
---
{raw_procedure_text[:2000]} 
---
Please provide:
1.  The procedure reformatted into a numbered list of clear, concise steps.
2.  The reaction atmosphere (e.g., 'under nitrogen', 'argon blanket', 'air', or "Not specified").

Return the response as a JSON object with two keys: "procedure_steps" (a list of strings) and "atmosphere" (a string).
If the text is not a procedure or cannot be reliably formatted into steps, set "procedure_steps" to an empty list [] or ["Not a procedure."]. Set "atmosphere" to "Not applicable" in such cases.

JSON Output:
"""
        llm_proc_response_content = chem_agent._get_llm_completion_with_fallback(proc_format_prompt_system, proc_format_prompt_user, max_tokens=1000)
        parsed_llm_data = chem_agent._parse_llm_json_response(llm_proc_response_content, 
                                                               default_on_error={"procedure_steps": [], "atmosphere": "LLM Error"})
        llm_steps = parsed_llm_data.get("procedure_steps", [])
        if llm_steps and not (len(llm_steps) == 1 and llm_steps[0].lower() in ["not a procedure.", ""]):
            final_json_output["procedure_steps"] = llm_steps
        elif raw_procedure_text: 
            final_json_output["procedure_steps"] = [f"Could not format procedure from dataset. Raw snippet: {raw_procedure_text[:300]}..."]
        
        if final_json_output["reaction_conditions"]["atmosphere_llm_or_dataset"] == "N/A": 
            llm_atmosphere = parsed_llm_data.get("atmosphere", "Not specified") if parsed_llm_data else "Not specified"
            if llm_atmosphere and llm_atmosphere.lower() not in ["not specified", "llm error", "not applicable"]:
                final_json_output["reaction_conditions"]["atmosphere_llm_or_dataset"] = llm_atmosphere
    
    if final_json_output["reaction_conditions"]["atmosphere_llm_or_dataset"] == "N/A":
        final_json_output["reaction_conditions"]["atmosphere_llm_or_dataset"] = "Not specified"

    # 7. LLM for Overall Reaction Safety Assessment & Operational Notes
    llm_context_for_safety = f"""Reaction SMILES: {reaction_smiles_clean}
Reactants identified: {', '.join(reactant_names_for_llm_context)}
Products identified: {', '.join(product_names_for_llm_context)}
Key reagents/solvents involved (if known): {', '.join(unique_agent_solvent_names_for_llm_context)}"""
    # Note: Detailed safety for individual components is NO LONGER directly passed in this prompt text
    # as it was removed from the JSON structure's safety_and_notes. The LLM should infer based on names.

    safety_notes_prompt_system = "You are a chemical safety expert. Provide a concise reaction-specific safety assessment and key operational notes in JSON format."
    safety_notes_prompt_user = f"""For the reaction context:
{llm_context_for_safety}

Based on this information (and considering general hazards of these compound classes and the overall transformation), please provide:
1. A brief (1-2 sentences) overall safety assessment for conducting *this specific reaction*. Focus on risks arising from the reaction itself or the combination of substances. This will be the value for the "safety" key.
2. Key *operational notes or precautions* relevant to carrying out this reaction. This will be the value for the "notes" key.

Return JSON in the format:
{{
    "safety": "Reaction-specific safety assessment summary...",
    "notes": "Operational notes and precautions for this reaction..."
}}
"""
    llm_safety_notes_response = chem_agent._get_llm_completion_with_fallback(safety_notes_prompt_system, safety_notes_prompt_user, max_tokens=500)
    parsed_safety_notes_data = chem_agent._parse_llm_json_response(llm_safety_notes_response)
    if parsed_safety_notes_data:
        final_json_output["safety_and_notes"]["safety"] = parsed_safety_notes_data.get("safety", "N/A") # Changed key
        final_json_output["safety_and_notes"]["notes"] = parsed_safety_notes_data.get("notes", "N/A")   # Changed key

    return {
        'visualization_path': final_json_output.pop("visualization_path"), 
        'analysis': final_json_output, 
        'analysis_context': "full_reaction_json_recipe_card_v6_key_changes", 
        'processed_smiles_for_tools': reaction_smiles_clean
    }

# --- handle_compound_full_info, handle_followup_question, etc. ---
def handle_compound_full_info(query_text_for_summary_context, compound_smiles, original_compound_name_context=None, callbacks=None):
    global _current_moi_context
    if not compound_smiles:
        return {'visualization_path': None, 'analysis': {"error": "No valid compound SMILES"},
                'analysis_context': "invalid_compound_smiles", 'processed_smiles_for_tools': None}

    chem_agent = ChemicalAnalysisAgent()
    try:
        props = chem_agent.analyze_chemical(compound_smiles)
        _current_moi_context["name"] = props.name or original_compound_name_context or compound_smiles
        _current_moi_context["smiles"] = props.smiles or compound_smiles
        
        # Using the generate_report method from ChemicalAnalysisAgent for text output
        report_text = chem_agent.generate_report(props) 

        viz_path = None
        if ag_tools and (props.smiles or compound_smiles):
            try:
                viz_path_result = ag_tools.visualize_chemical_structure(props.smiles or compound_smiles)
                if viz_path_result and not str(viz_path_result).lower().startswith('error') and ".png" in viz_path_result:
                    viz_path = viz_path_result
            except Exception: pass

        return {
            'visualization_path': viz_path,
            'analysis': report_text, 
            'analysis_context': "compound_full_text_report_agent",
            'processed_smiles_for_tools': props.smiles or compound_smiles
        }
    except Exception as e:
        return {'visualization_path': None, 'analysis': {"error": f"Error analyzing compound {compound_smiles}: {e}"},
                'analysis_context': "compound_analysis_error", 'processed_smiles_for_tools': compound_smiles}


def handle_followup_question(query_text, reaction_smiles, original_compound_name=None, callbacks=None):
    cached_reaction_data_full = reaction_cache.get(reaction_smiles, {}).get('full_info', {})
    if not cached_reaction_data_full or not isinstance(cached_reaction_data_full.get('analysis'), dict):
        return { "analysis": None, "analysis_context": "followup_no_structured_cache" }

    structured_analysis = cached_reaction_data_full['analysis']
    query_lower = query_text.lower()
    response_text = None

    # Try to answer based on the new "recipe card" structure
    if any(k in query_lower for k in ['procedure', 'steps', 'method']):
        proc_steps = structured_analysis.get('procedure_steps', [])
        if proc_steps and proc_steps[0] != "Procedure details not available.": # Check default message
            response_text = "Procedure Steps:\n" + "\n".join([f"{i+1}. {step}" for i, step in enumerate(proc_steps)])
        else: response_text = "Procedure details are not available in the structured summary."
    
    elif any(k in query_lower for k in ['condition', 'temperature', 'time', 'atmosphere']):
        cond = structured_analysis.get('reaction_conditions', {})
        details = []
        if 'temperature' in query_lower: details.append(f"Temp: {cond.get('temperature_from_dataset', 'N/A')}")
        if 'time' in query_lower: details.append(f"Time: {cond.get('time_from_dataset', 'N/A')}")
        if 'atmosphere' in query_lower: details.append(f"Atmosphere: {cond.get('atmosphere_llm_or_dataset', 'N/A')}")
        response_text = "Reaction Conditions: " + "; ".join(details) if details else "Requested condition not found."

    elif any(k in query_lower for k in ['reagent', 'solvent', 'catalyst', 'component price', 'cost']):
        components = structured_analysis.get('reagents_and_solvents', [])
        found_components = []
        for comp in components:
            comp_name_lower = comp.get('name', '').lower()
            comp_role_lower = comp.get('role', '').lower()
            if any(keyword in query_lower for keyword in [comp_name_lower, comp_role_lower] if keyword) or \
               ('reagent' in query_lower and 'reagent' in comp_role_lower) or \
               ('solvent' in query_lower and 'solvent' in comp_role_lower) or \
               ('catalyst' in query_lower and 'reagent' in comp_role_lower): # Simple catalyst check
                
                price_info = f"{comp.get('price_per_unit', 'N/A')} {comp.get('currency', '')}/{comp.get('unit_basis', '')}" if comp.get('price_per_unit') != "N/A" else "Price N/A"
                found_components.append(f"{comp.get('name')} ({comp.get('role')}): {price_info}, Source: {comp.get('price_source','N/A')}, Conf: {comp.get('price_confidence','N/A')}")
        if found_components:
            response_text = "Reagent/Solvent Information:\n" + "\n".join(found_components)
        else: response_text = "Requested reagent/solvent information not found or not detailed in summary."
        
    elif 'yield' in query_lower:
        response_text = f"Yield from Dataset: {structured_analysis.get('reaction_details',{}).get('yield_from_dataset', 'N/A')}"
    
    elif any(k in query_lower for k in ['safety', 'hazard', 'precaution', 'note']):
        safety_notes = structured_analysis.get('safety_and_notes', {})
        overall_safety = safety_notes.get('overall_reaction_safety_assessment_llm', 'N/A')
        op_notes = safety_notes.get('notes_llm', 'N/A')
        response_text = f"Overall Safety: {overall_safety}\nOperational Notes: {op_notes}"
        # Could add reactant/product specific safety if query is more specific.

    elif any(k in query_lower for k in ['mechanism', 'literature', 'reference', 'publication']):
        details = structured_analysis.get('reaction_details', {})
        mech = details.get('mechanism_llm_guess', 'N/A')
        lit = details.get('literature_reference_llm_guess', {})
        lit_str = f"Title: {lit.get('title','N/A')}, Authors: {lit.get('authors','N/A')}, Journal: {lit.get('journal','N/A')}, DOI: {lit.get('doi','N/A')}"
        response_text = f"Mechanism Guess: {mech}\nLiterature Guess: {lit_str}"

    if response_text:
        return {"visualization_path": None, "analysis": response_text, "analysis_context": "followup_structured_answer_recipe", "processed_smiles_for_tools": reaction_smiles}
    return { "analysis": None, "analysis_context": "followup_property_unmatched_structured_recipe" }


# --- Autogen Agent setups ---
_assistant_tool_agent = None; _user_proxy_tool_agent = None
_assistant_chatbot_agent = None; _user_proxy_chatbot_agent = None
_current_moi_context: Dict[str, Optional[str]] = {"name": None, "smiles": None}
CHATBOT_TOOL_PY_FUNCTIONS_BASE = []
if ag_tools:
    CHATBOT_TOOL_PY_FUNCTIONS_BASE = [
        ag_tools.get_functional_groups, ag_tools.convert_name_to_smiles,
        ag_tools.suggest_disconnections, ag_tools.convert_smiles_to_name,
        ag_tools.visualize_chemical_structure, ag_tools.get_full_chemical_report
    ]

TOOL_AGENT_SYSTEM_MESSAGE = """You are Chem Copilot, an expert chemistry assistant. You are tasked with executing a specific chemical analysis tool based on the user's query.
You have access to the following tools:
- `get_functional_groups(smiles_or_reaction_smiles: str)`: Identifies functional groups.
- `convert_name_to_smiles(chemical_name: str)`: Converts name to SMILES.
- `convert_smiles_to_name(smiles_string: str)`: Converts SMILES to name.
- `analyze_reaction_bond_changes(reaction_smiles: str)`: Analyzes bond changes in a reaction.
- `visualize_chemical_structure(smiles_or_reaction_smiles: str)`: Generates a visualization.
- `classify_reaction_and_get_details(reaction_smiles: str)`: Classifies reaction and gets details.
- `query_specific_property_for_reaction(reaction_smiles: str, property_to_query: str)`: Queries a specific property for a reaction.
- `suggest_disconnections(smiles: str)`: Suggests retrosynthetic disconnections for a compound SMILES.
- `get_full_chemical_report(chemical_identifier: str)`: Provides a comprehensive analysis report for a chemical (name, SMILES, or CAS). This uses the ChemicalAnalysisAgent and returns a text report for single compounds.

Your goal is to:
1. Understand the user's request.
2. Select THE MOST APPROPRIATE tool.
3. Execute the tool with the correct input extracted from the query.
4. Return the raw output from the tool directly as your final answer. Do not add any conversational fluff.
5. If the query is general knowledge or no tool is appropriate, respond with:
   "I can only perform specific chemical analyses using my tools if you provide a SMILES string or a chemical name for tool-based processing. I cannot answer general knowledge questions. Please provide a specific chemical entity or task for my tools.TERMINATE"
Your response should be ONLY the tool's output or the refusal message.
"""

def get_tool_agents():
    global _assistant_tool_agent, _user_proxy_tool_agent, llm_config_chatbot_agent, ag_tools
    if not ag_tools:
        print("Critical: ag_tools not loaded. Cannot create tool agents.")
        return None, None

    if _assistant_tool_agent is None:
        tool_functions_for_tool_agent = [
            ag_tools.get_functional_groups, ag_tools.convert_name_to_smiles,
            ag_tools.suggest_disconnections, ag_tools.convert_smiles_to_name,
            ag_tools.analyze_reaction_bond_changes, ag_tools.visualize_chemical_structure,
            ag_tools.classify_reaction_and_get_details, ag_tools.query_specific_property_for_reaction,
            ag_tools.get_full_chemical_report
        ]
        assistant_llm_tools_definition = []
        for func in tool_functions_for_tool_agent:
            func_name = func.__name__
            param_name = "chemical_identifier" 
            param_desc = "Input for the tool."

            if func_name == "get_full_chemical_report": param_name, param_desc = "chemical_identifier", "The name, SMILES string, or CAS number of the chemical."
            elif func_name == "convert_name_to_smiles": param_name, param_desc = "chemical_name", "The chemical name."
            elif func_name == "convert_smiles_to_name": param_name, param_desc = "smiles_string", "The SMILES string."
            elif func_name == "suggest_disconnections": param_name, param_desc = "smiles", "The SMILES string of the molecule."
            elif func_name == "query_specific_property_for_reaction":
                assistant_llm_tools_definition.append({
                    "type": "function", "function": { "name": func_name, "description": func.__doc__ or f"Executes {func_name} tool.",
                        "parameters": {"type": "object", "properties": {
                                "reaction_smiles": {"type": "string", "description": "The Reaction SMILES string."},
                                "property_to_query": {"type": "string", "description": "The specific property like 'yield' or 'temperature'."}},
                            "required": ["reaction_smiles", "property_to_query"]}}})
                continue
            else: param_name, param_desc = "smiles_or_reaction_smiles", "The SMILES string of the compound or reaction."

            assistant_llm_tools_definition.append({
                "type": "function", "function": { "name": func_name, "description": (func.__doc__ or f"Executes {func_name} tool.").splitlines()[0].strip(),
                    "parameters": {"type": "object", "properties": {param_name: {"type": "string", "description": param_desc}}, "required": [param_name]}}})

        tool_agent_llm_config = llm_config_chatbot_agent.copy()
        if assistant_llm_tools_definition: tool_agent_llm_config["tools"] = assistant_llm_tools_definition
        
        _assistant_tool_agent = autogen.AssistantAgent(name="ChemistryToolAgent_v5_Recipe", llm_config=tool_agent_llm_config, system_message=TOOL_AGENT_SYSTEM_MESSAGE,
            is_termination_msg=lambda x: isinstance(x.get("content"), str) and x.get("content", "").rstrip().endswith("TERMINATE") or \
                                        (isinstance(x.get("content"), str) and "I can only perform specific chemical analyses" in x.get("content","") and not x.get("tool_calls")))
        _user_proxy_tool_agent = autogen.UserProxyAgent(name="UserProxyToolExecutor_v5_Recipe", human_input_mode="NEVER", max_consecutive_auto_reply=2, code_execution_config=False, 
            function_map={func.__name__: func for func in tool_functions_for_tool_agent},
            is_termination_msg=lambda x: (isinstance(x.get("content"), str) and x.get("content", "").rstrip().endswith("TERMINATE")) or \
                                        (isinstance(x.get("content"), str) and "I can only perform specific chemical analyses" in x.get("content","")))
    return _assistant_tool_agent, _user_proxy_tool_agent

def run_autogen_tool_agent_query(user_input: str, callbacks=None):
    assistant, user_proxy = get_tool_agents()
    if not assistant or not user_proxy:
        return {"analysis": "Tool agents not initialized (ag_tools might be missing).", "visualization_path": None}
    user_proxy.reset(); assistant.reset()
    ai_response_text = "Tool agent did not provide a clear answer (default)."
    try:
        user_proxy.initiate_chat(recipient=assistant, message=user_input, max_turns=3, request_timeout=llm_config_chatbot_agent.get("timeout", 60) + 10)
        messages = user_proxy.chat_messages.get(assistant, [])
        if messages:
            last_msg_obj = messages[-1]
            if last_msg_obj.get("role") == "assistant" and last_msg_obj.get("content"): ai_response_text = last_msg_obj["content"].strip()
            elif last_msg_obj.get("content"): ai_response_text = last_msg_obj.get("content").strip()
        if ai_response_text.upper() == "TERMINATE" or ai_response_text == "Tool agent did not provide a clear answer (default).":
            if messages and len(messages) > 1:
                potential_reply = messages[-2].get("content", "").strip()
                if potential_reply and potential_reply.upper() != "TERMINATE": ai_response_text = potential_reply
        if ai_response_text.upper().endswith("TERMINATE"): ai_response_text = ai_response_text[:-len("TERMINATE")].strip()
        if not ai_response_text.strip() and ai_response_text != "Tool agent did not provide a clear answer (default).": ai_response_text = "Tool agent processing complete."
        
        viz_path_agent = None
        if "static/autogen_visualizations/" in ai_response_text:
            match_viz = re.search(r"(static/autogen_visualizations/[\w\-\.\_]+\.png)", ai_response_text)
            if match_viz: viz_path_agent = match_viz.group(1)
        return {"visualization_path": viz_path_agent, "analysis": ai_response_text }
    except Exception as e:
        print(f"Error in run_autogen_tool_agent_query: {e}\n{traceback.format_exc()}")
        return {"visualization_path": None, "analysis": f"An error occurred in the tool agent: {str(e)}"}

CHATBOT_TOOL_PARAM_INFO = {
    "get_functional_groups": {"param_name": "smiles_or_reaction_smiles", "description": "The SMILES or reaction SMILES string."},
    "convert_name_to_smiles": {"param_name": "chemical_name", "description": "The chemical name."},
    "suggest_disconnections": {"param_name": "smiles", "description": "The SMILES string of the molecule for disconnection suggestions."},
    "convert_smiles_to_name": {"param_name": "smiles_string", "description": "The SMILES string to convert to a name."},
    "visualize_chemical_structure": {"param_name": "smiles_or_reaction_smiles", "description": "The SMILES string for visualization."},
    "get_full_chemical_report": {"param_name": "chemical_identifier", "description": "The name, SMILES, or CAS number for a full chemical report."}
}

def _update_chatbot_system_message_with_moi():
    global _assistant_chatbot_agent, _current_moi_context, CHATBOT_TOOL_PY_FUNCTIONS_BASE, CHATBOT_TOOL_PARAM_INFO
    if not _assistant_chatbot_agent: return

    moi_name = _current_moi_context.get("name", "Not Set"); moi_smiles = _current_moi_context.get("smiles", "Not Set")
    tools_desc = [f"- `{f.__name__}({(CHATBOT_TOOL_PARAM_INFO.get(f.__name__,{}).get('param_name','input'))}: str)`: {(f.__doc__ or '').splitlines()[0].strip()}" for f in CHATBOT_TOOL_PY_FUNCTIONS_BASE]
    tools_list_str = "\n".join(tools_desc) if tools_desc else "No tools specified."

    system_message = f"""You are ChemCopilot, a specialized AI assistant for chemistry.
Current Molecule of Interest (MOI): Name: {moi_name}, SMILES: {moi_smiles}
Available tools:
{tools_list_str}
Tasks:
1. Use MOI context. If asked for MOI's SMILES: state "The SMILES for {moi_name} is {moi_smiles}. TERMINATE".
2. For other MOI properties, use MOI's SMILES ('{moi_smiles}') for tool calls.
3. If MOI not set/relevant & user provides new entity, use that for tool calls.
4. General queries not needing tools/MOI: respond with "I am a specialized chemistry assistant... Could you please clarify or provide a specific chemical entity? TERMINATE"
5. Response Format: Concise. ALWAYS append " TERMINATE".
"""
    if _assistant_chatbot_agent: _assistant_chatbot_agent.update_system_message(system_message)

def get_chatbot_agents():
    global _assistant_chatbot_agent, _user_proxy_chatbot_agent, llm_config_chatbot_agent, CHATBOT_TOOL_PY_FUNCTIONS_BASE, CHATBOT_TOOL_PARAM_INFO
    if not CHATBOT_TOOL_PY_FUNCTIONS_BASE:
        print("Critical: CHATBOT_TOOL_PY_FUNCTIONS_BASE is empty. Cannot create chatbot agents.")
        return None, None
    if _assistant_chatbot_agent is None:
        tools_cfg = []
        for func in CHATBOT_TOOL_PY_FUNCTIONS_BASE:
            p_info = CHATBOT_TOOL_PARAM_INFO.get(func.__name__, {"param_name": "input", "description": "Input."})
            tools_cfg.append({"type": "function", "function": {"name": func.__name__, "description": (func.__doc__ or '').splitlines()[0].strip(),
                             "parameters": {"type": "object", "properties": {p_info["param_name"]: {"type": "string", "description": p_info["description"]}}, "required": [p_info["param_name"]]}}})
        
        cfg = llm_config_chatbot_agent.copy()
        if tools_cfg: cfg["tools"] = tools_cfg
        
        _assistant_chatbot_agent = autogen.AssistantAgent(name="ChemistryChatbotAgent_MOI_v5_Recipe", llm_config=cfg, system_message="Initializing MOI-aware ChemCopilot...",
            is_termination_msg=lambda x: isinstance(x.get("content"), str) and x.get("content", "").rstrip().endswith("TERMINATE"))
        _user_proxy_chatbot_agent = autogen.UserProxyAgent(name="UserProxyChatConversational_MOI_v5_Recipe", human_input_mode="NEVER", max_consecutive_auto_reply=3, code_execution_config=False,
            function_map={tool.__name__: tool for tool in CHATBOT_TOOL_PY_FUNCTIONS_BASE},
            is_termination_msg=lambda x: (isinstance(x.get("content"), str) and x.get("content", "").rstrip().endswith("TERMINATE")) or \
                                        (isinstance(x.get("content"), str) and "I am a specialized chemistry assistant" in x.get("content", "")))
        _update_chatbot_system_message_with_moi()
    return _assistant_chatbot_agent, _user_proxy_chatbot_agent

def clear_chatbot_memory_autogen():
    global _current_moi_context, _assistant_chatbot_agent, _user_proxy_chatbot_agent
    _current_moi_context = {"name": None, "smiles": None}
    if _assistant_chatbot_agent: _assistant_chatbot_agent.reset()
    if _user_proxy_chatbot_agent: _user_proxy_chatbot_agent.reset()
    if _assistant_chatbot_agent: _update_chatbot_system_message_with_moi()

MAX_CHATBOT_TURNS = 5
def run_autogen_chatbot_query(user_input: str, callbacks=None):
    global _current_moi_context, _assistant_chatbot_agent, _user_proxy_chatbot_agent
    priming_match = re.match(r"Let's discuss the molecule of interest: (.*?) with SMILES (.*)\. Please acknowledge\.", user_input, re.IGNORECASE)
    if priming_match:
        moi_name, moi_smiles = priming_match.groups()
        _current_moi_context["name"] = moi_name.strip(); _current_moi_context["smiles"] = moi_smiles.strip()
        if not _assistant_chatbot_agent or not _user_proxy_chatbot_agent: get_chatbot_agents()
        if not _assistant_chatbot_agent: return {"analysis": "Chatbot agents could not be initialized.", "visualization_path": None, "error": "Agent init failed"}
        _update_chatbot_system_message_with_moi()
        return {"analysis": f"Acknowledged. We are now discussing {_current_moi_context['name']} ({_current_moi_context['smiles']}). How can I help?", "visualization_path": None, "error": None}

    if not _assistant_chatbot_agent or not _user_proxy_chatbot_agent:
        get_chatbot_agents()
        if not _assistant_chatbot_agent: return {"analysis": "Chatbot agents failed to initialize.", "visualization_path": None, "error": "Agent init failed"}
    else: _update_chatbot_system_message_with_moi()
    if _assistant_chatbot_agent: _assistant_chatbot_agent.reset()
    if _user_proxy_chatbot_agent: _user_proxy_chatbot_agent.reset()

    ai_response_text = "Chatbot did not provide a clear answer (default)."; viz_path = None
    try:
        _user_proxy_chatbot_agent.initiate_chat(recipient=_assistant_chatbot_agent, message=user_input, max_turns=MAX_CHATBOT_TURNS, request_timeout=llm_config_chatbot_agent.get("timeout", 90) + 30, clear_history=True)
        conv_history = _assistant_chatbot_agent.chat_messages.get(_user_proxy_chatbot_agent, [])
        if conv_history:
            for msg_obj in reversed(conv_history):
                if msg_obj.get("role") == "assistant":
                    if msg_obj.get("tool_calls"): continue 
                    if msg_obj.get("content"): ai_response_text = msg_obj.get("content", "").strip(); break
            if ai_response_text == "Chatbot did not provide a clear answer (default)." and conv_history[-1].get("role") == "assistant" and conv_history[-1].get("content"):
                ai_response_text = conv_history[-1].get("content").strip()
        
        if ai_response_text.upper().endswith("TERMINATE"): ai_response_text = ai_response_text[:-len("TERMINATE")].strip()
        if not ai_response_text.strip() and "Chatbot did not provide a clear answer" not in ai_response_text: ai_response_text = "Chatbot processing complete."

        user_proxy_sent = _user_proxy_chatbot_agent.chat_messages.get(_assistant_chatbot_agent, [])
        for msg_item in user_proxy_sent:
            if msg_item.get("role") == "tool" and "static/autogen_visualizations/" in msg_item.get("content", ""):
                match_viz = re.search(r"(static/autogen_visualizations/[\w\-\.\_]+\.png)", msg_item.get("content", ""))
                if match_viz: viz_path = match_viz.group(1); break
        if not viz_path and "static/autogen_visualizations/" in ai_response_text:
             match_viz = re.search(r"(static/autogen_visualizations/[\w\-\.\_]+\.png)", ai_response_text)
             if match_viz: viz_path = match_viz.group(1)
        return {"visualization_path": viz_path, "analysis": ai_response_text, "error": None }
    except openai.APITimeoutError as e: return { "visualization_path": None, "analysis": f"OpenAI API timed out: {str(e)}", "error": str(e)}
    except Exception as e: return { "visualization_path": None, "analysis": f"An error occurred in the MOI chatbot: {str(e)}", "error": str(e) }

# --- Main Query Routing Logic ---
def enhanced_query(full_query: str, callbacks=None, original_compound_name: str = None):
    global _current_moi_context
    final_result = {}
    query_context_for_filename = "unknown_query_type"
    reaction_smiles_for_tools = extract_reaction_smiles(full_query)
    compound_smiles_for_tools = None
    if not reaction_smiles_for_tools: compound_smiles_for_tools = extract_single_compound_smiles(full_query)
    query_lower = full_query.lower()
    full_info_keywords = ["full info", "full data", "complete analysis", "details about", "tell me about", "explain this", "analyze this reaction", "analyze this compound"]
    try:
        if compound_smiles_for_tools and any(keyword in query_lower for keyword in full_info_keywords):
            _current_moi_context["name"] = original_compound_name or compound_smiles_for_tools; _current_moi_context["smiles"] = compound_smiles_for_tools
            final_result = handle_compound_full_info(full_query, compound_smiles_for_tools, original_compound_name or compound_smiles_for_tools, callbacks=callbacks)
            query_context_for_filename = final_result.get('analysis_context', 'compound_full_report_from_smiles')

        elif reaction_smiles_for_tools and any(keyword in query_lower for keyword in full_info_keywords):
            _current_moi_context["name"] = original_compound_name or reaction_smiles_for_tools; _current_moi_context["smiles"] = reaction_smiles_for_tools
            final_result = handle_full_info(full_query, reaction_smiles_for_tools, original_compound_name or reaction_smiles_for_tools, callbacks=callbacks) 
            query_context_for_filename = final_result.get('analysis_context', 'reaction_full_recipe_card') 

        elif not compound_smiles_for_tools and not reaction_smiles_for_tools and any(keyword in query_lower for keyword in full_info_keywords) and ag_tools:
            name_entity = None
            # Prioritize "named '...'" then general "of ..."
            match_named = re.search(r"full info(?: about| for| of)?\s+(?:named|called)\s*['\"]?([\w\s\-(),]+?)['\"]?(?:\s+with smiles|\s+smiles|\s+cas|\.|$)", query_lower, re.IGNORECASE)
            if match_named: name_entity = match_named.group(1).strip()
            else:
                match_general = re.search(r"full info(?: about| for| of)?\s+([\w\s\-(),]+?)(?:\s+with smiles|\s+smiles|\s+cas|\.|$)", query_lower, re.IGNORECASE)
                if match_general: name_entity = match_general.group(1).strip()
            
            if name_entity and not extract_single_compound_smiles(name_entity) and not extract_reaction_smiles(name_entity):
                n2s_output = ag_tools.convert_name_to_smiles(name_entity)
                smiles_from_name = None
                if isinstance(n2s_output, str):
                    s_match = re.search(r"SMILES:\s*([^\s\n]+)", n2s_output)
                    if s_match: smiles_from_name = s_match.group(1).strip()
                
                if smiles_from_name:
                    _current_moi_context["name"] = name_entity; _current_moi_context["smiles"] = smiles_from_name
                    if ">>" in smiles_from_name: final_result = handle_full_info(f"Full info for {smiles_from_name}", smiles_from_name, name_entity)
                    else: final_result = handle_compound_full_info(f"Full info for {smiles_from_name}", smiles_from_name, name_entity)
                    query_context_for_filename = final_result.get('analysis_context', 'full_info_from_name_conversion')
                else: 
                    final_result = {"analysis": {"error":f"Could not find SMILES for name '{name_entity}' via tool."}, "processed_smiles_for_tools": None}
                    query_context_for_filename = "full_info_name_conversion_failed_tool"
            elif not name_entity : # No name found, but full info keyword present
                 final_result = run_autogen_chatbot_query(full_query) # Try chatbot for general full info queries
                 query_context_for_filename = "full_info_no_entity_to_chatbot"


        if not final_result: 
            if reaction_smiles_for_tools:
                _current_moi_context["name"] = original_compound_name or reaction_smiles_for_tools; _current_moi_context["smiles"] = reaction_smiles_for_tools
                followup_res = handle_followup_question(full_query, reaction_smiles_for_tools)
                if followup_res and followup_res.get('analysis'): final_result = followup_res
                else: final_result = run_autogen_chatbot_query(full_query)
                query_context_for_filename = final_result.get('analysis_context', 'chatbot_or_followup_reaction_recipe')
            elif compound_smiles_for_tools:
                 _current_moi_context["name"] = original_compound_name or compound_smiles_for_tools; _current_moi_context["smiles"] = compound_smiles_for_tools
                 tool_res = run_autogen_tool_agent_query(full_query)
                 if tool_res and tool_res.get("analysis") and "I can only perform specific" not in tool_res["analysis"]: final_result = tool_res
                 else: final_result = run_autogen_chatbot_query(full_query)
                 query_context_for_filename = final_result.get('analysis_context', 'tool_or_chatbot_compound_recipe')
            else: 
                final_result = run_autogen_chatbot_query(full_query)
                query_context_for_filename = "chatbot_general_no_smiles_recipe"
        
        if not final_result: # Ultimate fallback
            final_result = run_autogen_chatbot_query(full_query)
            query_context_for_filename = "chatbot_fallback_routing_failed_recipe"

        final_result["current_moi_name"] = _current_moi_context.get("name")
        final_result["current_moi_smiles"] =  _current_moi_context.get("smiles")
        analysis_content = final_result.get("analysis")
        smiles_to_save = final_result.get('processed_smiles_for_tools', reaction_smiles_for_tools or compound_smiles_for_tools)
        if query_context_for_filename == "unknown_query_type" and final_result.get('analysis_context'): query_context_for_filename = final_result.get('analysis_context')

        should_save = False
        if isinstance(analysis_content, dict) and not analysis_content.get("error"): should_save = True
        elif isinstance(analysis_content, str) and len(analysis_content.strip()) > 50 and \
             not any(phrase in analysis_content for phrase in ["I am a specialized chemistry assistant", "I can only perform specific chemical analyses"]): should_save = True
        
        if smiles_to_save and should_save and not query_context_for_filename.startswith("visualization_"):
            save_analysis_to_file(smiles_to_save, analysis_content, query_context_for_filename, original_compound_name)

        if 'processed_smiles_for_tools' not in final_result: final_result['processed_smiles_for_tools'] = smiles_to_save
        final_result['analysis_context'] = query_context_for_filename
        return final_result

    except Exception as e:
        tb_str = traceback.format_exc(); print(f"CRITICAL Error in enhanced_query: {str(e)}\n{tb_str}")
        err_dict = {"error": f"Error processing query: {str(e)}."} 
        smiles_err_ctx = compound_smiles_for_tools or reaction_smiles_for_tools or "no_entity"
        save_analysis_to_file(smiles_err_ctx, f"Query: {full_query}\n{str(err_dict)}\n{tb_str}", "enhanced_query_CRITICAL_error_recipe", original_compound_name)
        return {"visualization_path": None, "analysis": err_dict, "error": str(e), "processed_smiles_for_tools": smiles_err_ctx, 
                "analysis_context": "enhanced_query_exception_recipe", "current_moi_name": _current_moi_context.get("name"), "current_moi_smiles":  _current_moi_context.get("smiles")}


# --- display_analysis_result MODIFIED ---
def display_analysis_result(title: str, analysis_result: dict, is_chatbot: bool = False):
    print(f"\n--- {title} ---")
    if not analysis_result or not isinstance(analysis_result, dict):
        print(f"Invalid analysis result format. Raw: {analysis_result}")
        print(f"--- End of {title} ---\n"); return

    analysis_content = analysis_result.get("analysis")

    if isinstance(analysis_content, dict): 
        if "error" in analysis_content: print(f"Error in analysis: {analysis_content['error']}")
        else:
            print("Reaction Analysis (JSON Structure):")
            try: print(json.dumps(analysis_content, indent=2))
            except TypeError: 
                print("Could not serialize full JSON, printing parts:")
                for key, value in analysis_content.items():
                    if isinstance(value, dict) or isinstance(value, list):
                        print(f"  {key}:")
                        try: print(f"    {json.dumps(value, indent=2)}")
                        except: print(f"    {str(value)[:300]}...")
                    else: print(f"  {key}: {str(value)[:300] + '...' if len(str(value)) > 300 else value}")
    elif isinstance(analysis_content, str): 
        cleaned_text = re.sub(r"!\[.*?\]\((.*?)\)", r"[Image: \1]", analysis_content)
        if is_chatbot: print(f"Chem Copilot: {cleaned_text}")
        else: print(cleaned_text)
    elif analysis_result.get("error"): print(f"Error: {analysis_result.get('error')}")
    else: print(f"Could not retrieve/generate response. Content: {analysis_content}")

    if analysis_result.get("visualization_path"): print(f"Visualization: {analysis_result['visualization_path']}")
    print(f"--- End of {title} ---\n")


# --- Main execution block for testing (as before) ---
if __name__ == "__main__":
    if not OPENAI_API_KEY and DEFAULT_LLM_PROVIDER == "openai":
        print("CRITICAL: OPENAI_API_KEY is not set, but OpenAI is the default provider. Exiting.")
        exit(1)
    if not PERPLEXITY_API_KEY and DEFAULT_LLM_PROVIDER == "perplexity":
        if DEFAULT_LLM_PROVIDER == "perplexity" and (not OPENAI_API_KEY or OPENAI_API_KEY == "sk-YOUR_OPENAI_API_KEY_HERE"):
             print("CRITICAL: PERPLEXITY_API_KEY not set (default), and OpenAI key also missing/placeholder. Exiting.")
             exit(1)
        elif DEFAULT_LLM_PROVIDER == "perplexity":
             print("Warning: PERPLEXITY_API_KEY not set (default). Will try OpenAI if available.")

    if not ag_tools or not ReactionClassifier:
        print("CRITICAL: Core tools (autogen_tool_functions or ReactionClassifier) failed to load. Exiting.")
        exit(1)

    dummy_files = {
        "pricing_data.json": {"Benzene": ["C1=CC=CC=C1", 100.0, "Pune"], "Water": ["O", 1.0, "Global"], "Thionyl chloride": ["ClS(=O)Cl", 1800.0, "Mumbai"]},
        "second_source.json": {"Ethanol": ["CCO", 150.0, "Mumbai"], "Dichloromethane": ["ClCCl", 440.0, "Delhi"]},
        "sigma_source.json": {"Methanol (CAS 67-56-1)": ["CO", 120.0, "Bangalore"], "SOCl2": ["ClS(=O)Cl", 1850.0, "Sigma Catalog"]},
        "benzoic_acid_deriv_prices.json": {"6-methylbenzoic acid": ["Cc1ccccc1C(=O)O", 2500.0, "Specialty Supplier"]}
    }
    for fname, content in dummy_files.items():
        fpath = os.path.join(PROJECT_ROOT_DIR, fname)
        if not os.path.exists(fpath):
            try:
                with open(fpath, 'w', encoding='utf-8') as f: json.dump(content, f, indent=2)
            except IOError: pass
            
    # Add the new dummy file to ChemicalAnalysisAgent's load logic if it's meant to be a general source
    # For this test, it's fine as `_get_pricing_from_all_local_sources` will check the defined ones.


    print("\nInteractive Chem Copilot Session")
    print(f"Default LLM Provider for Summaries/Properties: {DEFAULT_LLM_PROVIDER.upper()}")
    print("Example reaction query: 'full info for Cc1ccccc1C(=O)O.ClS(=O)Cl>>Cc1ccccc1C(=O)Cl.OS(=O).Cl'") # 6-methylbenzoic acid + SOCl2 -> 6-methylbenzoyl chloride
    print("Example compound query: 'full info for Aspirin'")
    print("Type 'exit' or 'quit'. Type 'clear chat' to reset MOI.")

    clear_chatbot_memory_autogen()
    last_processed_entity_name_for_saving: Optional[str] = None

    while True:
        try: user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt): print("\nExiting."); break
        if not user_input: continue
        if user_input.lower() in ['exit', 'quit', 'done']: print("Exiting."); break
        if user_input.lower().startswith("clear chat"):
            clear_chatbot_memory_autogen(); last_processed_entity_name_for_saving = None
            print("Chem Copilot: Chat MOI context and agent states cleared."); continue

        current_original_name_for_saving = None
        # Try to capture a name provided with "full info for NAME" type queries
        name_match_explicit = re.search(r"full\s+info\s+(?:for|of|about)\s+(?:named|called)\s*['\"]?([^'\"\n\.,>]+?)['\"]?(?:\s+with smiles|\s+smiles|\.|$)", user_input, re.IGNORECASE)
        if name_match_explicit:
            potential_name = name_match_explicit.group(1).strip()
            if not (">>" in potential_name or extract_single_compound_smiles(potential_name)): # Ensure it's not a SMILES
                 current_original_name_for_saving = potential_name
        else:
            name_match_general = re.search(r"full\s+info\s+(?:for|of|about)\s+([^'\"\n\.,>]+?)(?:\s+with smiles|\s+smiles|\s+cas|\.|$)", user_input, re.IGNORECASE)
            if name_match_general:
                potential_name = name_match_general.group(1).strip()
                # Avoid capturing parts of SMILES or keywords like "reaction"
                if not (">>" in potential_name or extract_single_compound_smiles(potential_name) or potential_name.lower() in ["reaction", "compound"]):
                     current_original_name_for_saving = potential_name
        
        priming_match_loop = re.match(r"Let's discuss the molecule of interest: (.*?) with SMILES .*\. Please acknowledge\.", user_input, re.IGNORECASE)
        if priming_match_loop:
            last_processed_entity_name_for_saving = priming_match_loop.group(1).strip()


        query_result = enhanced_query(user_input, original_compound_name=current_original_name_for_saving or last_processed_entity_name_for_saving)

        is_chatbot_response = "chatbot" in query_result.get('analysis_context', '') or \
                              "tool_agent" in query_result.get('analysis_context', '') or \
                              _current_moi_context.get("name") is not None 

        display_analysis_result(f"Chem Copilot Response", query_result, is_chatbot=is_chatbot_response)

        analysis_context = query_result.get('analysis_context', '')
        if query_result and isinstance(query_result, dict) and \
           ('full_reaction_json_recipe_card' in analysis_context or \
            'compound_full_text_report_agent' in analysis_context or \
            'full_info_from_name_conversion' in analysis_context or \
            'full_info_from_name' in analysis_context) and \
           query_result.get('processed_smiles_for_tools'):
            
            if current_original_name_for_saving: # Prioritize explicitly captured name for saving
                 last_processed_entity_name_for_saving = current_original_name_for_saving
            elif not last_processed_entity_name_for_saving: # If no explicit name, use MOI name if context matches
                if _current_moi_context.get("smiles") == query_result.get('processed_smiles_for_tools') and _current_moi_context.get("name"):
                    last_processed_entity_name_for_saving = _current_moi_context.get("name")
                # else, last_processed_entity_name_for_saving remains as it was (or None)

    print("\nInteractive session ended.")