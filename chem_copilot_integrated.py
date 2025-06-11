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
import requests # For Perplexity API and ASKCOS API

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
    pass
except Exception:
    pass

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")
DEFAULT_LLM_PROVIDER = os.environ.get("DEFAULT_LLM_PROVIDER", "openai")
ASKCOS_API_URL = "http://13.201.135.9:9621/reaction_class" # ASKCOS API URL

RDLogger.DisableLog('rdApp.*')


# --- GLOBAL HELPER FUNCTIONS (Moved from inside ChemicalAnalysisAgent) ---

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

def extract_reaction_smiles(query: str) -> Optional[str]: 
    smi_core_chars = r"[\w@\[\]\(\)\.\+\-\=\#\:\$\%\~\<\>]" 
    explicit_pattern = rf"(?i:\b(?:reaction\s+smiles|rxn)\s*[:=]?\s*)({smi_core_chars}+(?:>>{smi_core_chars}+)+)"
    match = re.search(explicit_pattern, query)
    if match:
        smiles = match.group(1).strip(); parts = smiles.split(">>")
        if len(parts) >= 2 and all(p.strip() for p in parts): return smiles
    
    standalone_pattern_strict_double_arrow = rf"(?<![\w\/])({smi_core_chars}+(?:>>{smi_core_chars}+)+)(?![\w\/])"
    potential_matches_double_arrow = re.findall(standalone_pattern_strict_double_arrow, query)
    for smi_candidate_match in potential_matches_double_arrow:
        smi_candidate = smi_candidate_match if isinstance(smi_candidate_match, str) else smi_candidate_match[0]
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
    combined_candidates = list(set(words + regex_candidates)); 
    combined_candidates.sort(key=lambda x: (len(x), sum(1 for c in x if c in '()[]=#')), reverse=True)
    
    for s_cand_orig in combined_candidates:
        s_cand = s_cand_orig.strip('.,;:)?!\'"')
        if not s_cand or '>>' in s_cand or '>' in s_cand or '<' in s_cand: continue
        if s_cand.isnumeric() and not ('[' in s_cand and ']' in s_cand) : continue
        
        try:
            mol = Chem.MolFromSmiles(s_cand, sanitize=True)
            if mol:
                num_atoms = mol.GetNumAtoms()
                if num_atoms >= 1:
                    if num_atoms <= 2 and s_cand.isalpha() and s_cand.lower() in ['as', 'in', 'is', 'at', 'or', 'to', 'be', 'of', 'on', 'no', 'do', 'go', 'so', 'if', 'it', 'me', 'my', 'he', 'we', 'by', 'up', 'us', 'an', 'am', 'are']:
                        if not any(c in s_cand for c in '()[]=#.-+@:/\\%{}1234567890'):
                            continue 
                    if any(c in s_cand for c in '()[]=#.-+@:/\\%{}') or num_atoms > 2 or len(s_cand) > 3 or s_cand.islower():
                        return s_cand
        except Exception: 
            pass
    return None


# --- ChemicalAnalysisAgent Enums and Dataclasses ---
class HazardLevel(Enum):
    LOW = "Low"; MODERATE = "Moderate"; HIGH = "High"; EXTREME = "Extreme"; UNKNOWN = "Unknown"

class SolubilityType(Enum):
    WATER_SOLUBLE = "Water Soluble"; ORGANIC_SOLUBLE = "Organic Soluble"
    POORLY_SOLUBLE = "Poorly Soluble"; INSOLUBLE = "Insoluble"; UNKNOWN = "Unknown"

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

@dataclass
class ChemicalProperties:
    name: str
    original_query: Optional[str] = None
    iupac_name: Optional[str] = None
    # ... (rest of ChemicalProperties dataclass as before) ...
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
            elif 'openai' in sys.modules: 
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

    # Note: _get_price_source_confidence_for_reaction_component is now a global function
    # and should be called directly, not via self.

    def _load_single_pricing_source(self, filename: str, source_display_name: str) -> None:
        # ... (rest of ChemicalAnalysisAgent methods from _load_single_pricing_source to generate_report)
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
                    except (requests.RequestException, json.JSONDecodeError): pass 
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
                        if statement: ghs_hazards_list.append({"pictogram_description": pictogram_display, "statement": f"{h_code}: {statement}".strip().lstrip(": ")})
        except Exception: pass 
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
        providers_to_try = []
        if self.llm_provider == "openai": providers_to_try = ["openai", "perplexity"]
        elif self.llm_provider == "perplexity": providers_to_try = ["perplexity", "openai"]
        else: providers_to_try = ["openai", "perplexity"]

        for provider_name in providers_to_try:
            if provider_name == "openai" and self.openai_client:
                try:
                    response = self.openai_client.chat.completions.create(
                        model=model_openai, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                        temperature=0.1, max_tokens=max_tokens)
                    return response.choices[0].message.content
                except Exception as e: print(f"[OpenAI LLM Error] {provider_name} call failed: {e}. Trying fallback.")
            elif provider_name == "perplexity" and self.perplexity_api_key_val:
                payload = {"model": model_perplexity, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]}
                headers = {"Authorization": f"Bearer {self.perplexity_api_key_val}", "Content-Type": "application/json", "Accept": "application/json"}
                try:
                    response = requests.post("https://api.perplexity.ai/chat/completions", json=payload, headers=headers, timeout=120)
                    response.raise_for_status()
                    return response.json()['choices'][0]['message']['content']
                except Exception as e: print(f"[Perplexity LLM Error] {provider_name} call failed: {e}. Trying fallback.")
        print("[LLM Error] All LLM providers failed or are not configured.")
        return None

    def _parse_llm_json_response(self, llm_response_content: Optional[str], default_on_error: Dict = None) -> Optional[Dict]:
        if default_on_error is None: default_on_error = {}
        if not llm_response_content: return default_on_error
        try:
            json_str_to_parse = llm_response_content
            match_json = re.search(r"```json\s*([\s\S]*?)\s*```", llm_response_content, re.IGNORECASE)
            if match_json: json_str_to_parse = match_json.group(1).strip()
            else:
                first_brace = json_str_to_parse.find('{'); last_brace = json_str_to_parse.rfind('}')
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    json_str_to_parse = json_str_to_parse[first_brace : last_brace+1]
                else:
                    print(f"[LLM JSON Parse WARN] Could not reliably find JSON object in: {llm_response_content[:100]}..."); return default_on_error
            return json.loads(json_str_to_parse)
        except json.JSONDecodeError as e: print(f"[LLM JSON Parse Error] Failed to decode JSON: {e}. Response: {llm_response_content[:200]}..."); return default_on_error

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
                    cleaned_val = val.replace(',', '') 
                    match_r = re.match(r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)", cleaned_val) 
                    if match_r: return (float(match_r.group(1)) + float(match_r.group(2))) / 2
                    match_s = re.match(r"(\d+\.?\d*)", cleaned_val) 
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
{{{{
    {guess_instr}
    "solubility": {{{{ "water_solubility": "Water Soluble/Organic Soluble/Poorly Soluble/Insoluble/Unknown", "organic_solvents_compatibility": ["alcohols", "ethers", "..."], "notes_on_solubility": "notes", "solubility_rating": "integer 1-10 or null" }}}},
    "hazards": {{{{ "corrosive": true/false/null, "flammable": true/false/null, "toxic": true/false/null, "carcinogenic_suspected": true/false/null, "environmental_hazard_notes": "notes", "overall_hazard_level": "Low/Moderate/High/Extreme/Unknown", "hazard_rating": "integer 1-10 or null", "ghs_info_llm": [ {{{{ "pictogram_description": "Pictogram Name", "h_code": "HXXX", "h_statement": "Full statement" }}}}] }}}},
    "safety_precautions": ["list of key safety measures"],
    "storage_recommendations": "storage conditions", "disposal_considerations": "disposal notes",
    "green_chemistry": {{{{ "renewable_feedstock_potential": "yes/no/unknown", "atom_economy_typical_reactions": "high/low/unknown", "biodegradability_assessment": "readily/poorly/unknown", "energy_efficiency_synthesis": "high/low/unknown", "waste_generation_typical_reactions": "high/low/unknown", "overall_score": "integer 1-10 or null" }}}},
    "environmental_impact_summary": "overall assessment"
}}}}
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
            for ghs in props.ghs_hazards[:5]: 
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
            if props.supplier_info: 
                 report_parts.append(f"Pricing/Supplier Notes: {props.supplier_info[0].get('availability', 'No specific notes.')}")

        report_parts.append("\n--- Environmental & Green Chemistry ---")
        if props.environmental_impact:
            report_parts.append(f"Environmental Impact Summary: {props.environmental_impact}")
        if props.green_chemistry_score is not None:
            report_parts.append(f"Green Chemistry Score (1-10): {props.green_chemistry_score}/10")
        
        return "\n".join(report_parts)

# --- Autogen LLM Configuration & Global Setup ---
config_list_gpt4o = [{"model": "gpt-4o", "api_key": OPENAI_API_KEY}]
config_list_gpt3_5_turbo = [{"model": "gpt-3.5-turbo-0125", "api_key": OPENAI_API_KEY}]
llm_config_chatbot_agent = {"config_list": config_list_gpt3_5_turbo, "temperature": 0.1, "timeout": 90}
REACTION_ANALYSIS_OUTPUT_DIR = os.path.join(PROJECT_ROOT_DIR, "reaction_analysis_outputs")
os.makedirs(REACTION_ANALYSIS_OUTPUT_DIR, exist_ok=True)
dataset_path1 = os.environ.get('REACTION_DATASET_PATH1', None) # Expected to be Orderly
dataset_path2 = os.environ.get('REACTION_DATASET_PATH2', None) # Expected to be Classification

try:
    from tools import autogen_tool_functions as ag_tools
    from tools.asckos import ReactionClassifier 
except ImportError:
    print("Warning: Failed to import 'tools.autogen_tool_functions' or 'tools.asckos.ReactionClassifier'. Some functionalities might be limited.")
    ag_tools = None
    ReactionClassifier = None

reaction_classifier_core_logic = None
if ReactionClassifier:
    if dataset_path1 or dataset_path2: 
        try:
            reaction_classifier_core_logic = ReactionClassifier(dataset_path1, dataset_path2)
            print(f"ReactionClassifier initialized with dataset1: '{dataset_path1}', dataset2: '{dataset_path2}'")
        except Exception as e_rc:
            print(f"Error initializing ReactionClassifier: {e_rc}")
            reaction_classifier_core_logic = None
    else:
        print("Warning: Neither REACTION_DATASET_PATH1 nor REACTION_DATASET_PATH2 are set. Local dataset querying will be unavailable.")
else:
    print("Warning: ReactionClassifier class not available. Local dataset querying will be unavailable.")

reaction_cache = {}; compound_cache = {}

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
            with open(filepath, "w", encoding="utf-8") as f: json.dump(analysis_data, f, indent=2)
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

@lru_cache(maxsize=50)
def _call_askcos_api(reaction_smiles: str) -> Optional[List[Dict[str, Any]]]:
    if not reaction_smiles:
        return None
    headers = {"Content-Type": "application/json"}
    payload = {"smiles": [reaction_smiles]}
    try:
        response = requests.post(ASKCOS_API_URL, headers=headers, json=payload, timeout=20) 
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "SUCCESS" and data.get("results"):
            if isinstance(data["results"], list) and len(data["results"]) > 0:
                if isinstance(data["results"][0], list) and all(isinstance(item, dict) for item in data["results"][0]):
                    return data["results"][0] 
                elif isinstance(data["results"][0], dict) and "rank" in data["results"][0]: 
                    return [data["results"][0]] 
            print(f"[ASKCOS API Info] Unexpected 'results' format for {reaction_smiles}: {data['results']}")
            return None
        else:
            print(f"[ASKCOS API Warning] API call for {reaction_smiles} did not return SUCCESS or no results. Status: {data.get('status')}, Error: {data.get('error')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"[ASKCOS API Error] Request failed for {reaction_smiles}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"[ASKCOS API Error] Failed to decode JSON response for {reaction_smiles}: {e}")
        return None
    except Exception as e:
        print(f"[ASKCOS API Error] Unexpected error for {reaction_smiles}: {e}")
        return None

@lru_cache(maxsize=100)
def query_reaction_dataset(reaction_smiles: str) -> Dict[str, Any]:
    results = {
        "orderly_data": None,
        "classification_data": None,
        "orderly_source_found": False,
        "classification_source_found": False
    }
    if not reaction_smiles or not reaction_classifier_core_logic:
        return results

    if hasattr(reaction_classifier_core_logic, 'dataset1') and \
       reaction_classifier_core_logic.dataset1 is not None and \
       not reaction_classifier_core_logic.dataset1.empty:
        df1 = reaction_classifier_core_logic.dataset1
        orderly_smiles_cols = ['rxn_str', 'reaction_smiles', 'smiles']
        orderly_match_row = None
        for col_name in orderly_smiles_cols:
            if col_name in df1.columns:
                try:
                    match_df = df1[df1[col_name] == reaction_smiles]
                    if not match_df.empty:
                        orderly_match_row = match_df.iloc[0]
                        results["orderly_source_found"] = True
                        break
                except Exception as e_match: 
                    print(f"[Dataset Query WARN] Error matching in Orderly dataset column '{col_name}': {e_match}")
                    continue
        
        if orderly_match_row is not None:
            orderly_data = {}
            if 'procedure_details' in orderly_match_row.index and pd.notna(orderly_match_row['procedure_details']):
                orderly_data['procedure_details'] = str(orderly_match_row['procedure_details'])
            if 'rxn_time' in orderly_match_row.index and pd.notna(orderly_match_row['rxn_time']):
                orderly_data['rxn_time'] = str(orderly_match_row['rxn_time'])
            if 'temperature' in orderly_match_row.index and pd.notna(orderly_match_row['temperature']):
                orderly_data['temperature'] = str(orderly_match_row['temperature'])
            if 'yield_000' in orderly_match_row.index and pd.notna(orderly_match_row['yield_000']):
                 orderly_data['yield_000'] = str(orderly_match_row['yield_000'])

            agents = []
            for i in range(17): 
                agent_col = f'agent_{i:03d}'
                if agent_col in orderly_match_row.index and pd.notna(orderly_match_row[agent_col]):
                    agents.append(str(orderly_match_row[agent_col]))
            if agents: orderly_data['agents_list'] = agents

            solvents = []
            for i in range(11): 
                solvent_col = f'solvent_{i:03d}'
                if solvent_col in orderly_match_row.index and pd.notna(orderly_match_row[solvent_col]):
                    solvents.append(str(orderly_match_row[solvent_col]))
            if solvents: orderly_data['solvents_list'] = solvents
            results["orderly_data"] = orderly_data

    if hasattr(reaction_classifier_core_logic, 'dataset2') and \
       reaction_classifier_core_logic.dataset2 is not None and \
       not reaction_classifier_core_logic.dataset2.empty:
        df2 = reaction_classifier_core_logic.dataset2
        class_smiles_cols = ['rxn_str', 'reaction_smiles', 'smiles']
        class_match_row = None
        for col_name in class_smiles_cols:
            if col_name in df2.columns:
                try:
                    match_df = df2[df2[col_name] == reaction_smiles]
                    if not match_df.empty:
                        class_match_row = match_df.iloc[0]
                        results["classification_source_found"] = True
                        break
                except Exception as e_match_cls:
                    print(f"[Dataset Query WARN] Error matching in Classification dataset column '{col_name}': {e_match_cls}")
                    continue

        if class_match_row is not None:
            class_data = {}
            if 'reaction_name' in class_match_row.index and pd.notna(class_match_row['reaction_name']):
                class_data['reaction_name'] = str(class_match_row['reaction_name'])
            results["classification_data"] = class_data
            
    reaction_cache.setdefault(reaction_smiles, {})['dataset_info_combined'] = results
    return results

def _get_llm_predicted_reaction_details(
    reaction_smiles: str,
    reactant_names: List[str],
    product_names: List[str],
    askcos_classification_context: Optional[List[Dict[str, Any]]],
    chem_agent_instance: ChemicalAnalysisAgent 
) -> Dict[str, Any]:
    
    askcos_top_class_name = "Unknown Classification"
    askcos_context_str = "ASKCOS Classification: Not available or failed."
    if askcos_classification_context and isinstance(askcos_classification_context, list) and askcos_classification_context:
        top_entry = askcos_classification_context[0]
        askcos_top_class_name = top_entry.get("reaction_name", top_entry.get("reaction_classname", "Unknown Classification"))
        askcos_context_str = f"ASKCOS Top Classification: {askcos_top_class_name} (Certainty: {top_entry.get('prediction_certainty', 'N/A'):.2f})"

    system_prompt = "You are an expert chemist. Based on the reaction and its classification, predict typical experimental details. Return JSON."
    user_prompt = f"""
Reaction SMILES: {reaction_smiles}
Identified Reactants: {', '.join(reactant_names) if reactant_names else 'N/A'}
Identified Products: {', '.join(product_names) if product_names else 'N/A'}
{askcos_context_str}

Given this information, please predict or suggest typical experimental details for this type of reaction.
Focus on:
1.  **Reagents/Catalysts**: List key reagents or catalysts commonly used.
2.  **Solvents**: Suggest suitable solvents.
3.  **Reaction Conditions**: Typical temperature (e.g., "Room Temperature", "50-60 °C", "Reflux"), reaction time (e.g., "2-4 hours", "Overnight"), and atmosphere (e.g., "Nitrogen", "Air", "Argon").
4.  **Yield**: A plausible yield or yield range for this transformation (e.g., "60-70%", "Typically high yielding").
5.  **Procedure Outline**: A brief, generalized step-by-step outline of how this reaction might be performed.

Return this information in a JSON object with the following keys:
"predicted_reagents_catalysts": ["list of strings"],
"predicted_solvents": ["list of strings"],
"predicted_temperature": "string",
"predicted_time": "string",
"predicted_atmosphere": "string",
"predicted_yield_range": "string",
"predicted_procedure_outline": ["list of strings, each a step"]

If information for a key is highly speculative or unknown, use "Not specified" or an empty list.
Example for procedure: ["Combine reactants in solvent.", "Add catalyst and stir at specified temperature.", "Monitor reaction by TLC.", "Work-up procedure (e.g., quench, extract, purify)."]
"""
    llm_response = chem_agent_instance._get_llm_completion_with_fallback(system_prompt, user_prompt, max_tokens=1500)
    parsed_data = chem_agent_instance._parse_llm_json_response(llm_response, default_on_error={
        "predicted_reagents_catalysts": [], "predicted_solvents": [],
        "predicted_temperature": "Not specified by LLM", "predicted_time": "Not specified by LLM",
        "predicted_atmosphere": "Not specified by LLM", "predicted_yield_range": "Not specified by LLM",
        "predicted_procedure_outline": ["LLM could not predict procedure steps."]
    })
    return parsed_data

def handle_full_info(query_text_for_llm_summary, reaction_smiles_clean, original_compound_name=None, callbacks=None):
    if not reaction_smiles_clean or ">>" not in reaction_smiles_clean:
        return {'visualization_path': None, 'analysis': {"error": f"Invalid reaction SMILES: Must contain '>>'. Input: '{reaction_smiles_clean}'"},
                'analysis_context': "invalid_smiles_input_no_double_arrow_v9", 'processed_smiles_for_tools': reaction_smiles_clean}

    chem_agent = ChemicalAnalysisAgent() 
    reactants_smi_list, agents_smi_list, products_smi_list = [], [], []

    try: 
        main_parts = reaction_smiles_clean.split(">>", 1)
        reactants_agents_str, products_str = main_parts[0].strip(), main_parts[1].strip()
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
                'analysis_context': "smiles_parsing_error_detailed_v9", 'processed_smiles_for_tools': reaction_smiles_clean}

    # Initialize with the target structure
    final_json_output = {
        "reaction_smiles_interpreted": reaction_smiles_clean,
        "reaction_details": {
            "reactants_identified": [],
            "products_identified": [],
            "reaction_name": "N/A" # Will be filled from ASKCOS/Dataset
        },
        "reagents_and_solvents": [],
        "reaction_conditions": {
            "temperature_from_dataset": "N/A", # Target key
            "time_from_dataset": "N/A",        # Target key
            "yield_from_dataset": "N/A",       # Target key
            "atmosphere_llm_or_dataset": "N/A" # Target key
        },
        "safety_and_notes": { # Target key
            "safety": "N/A", # Target key
            "notes": "N/A"   # Target key
        },
        "procedure_steps": ["Procedure details not available from dataset."] # Target key
        # We'll handle visualization_path separately before returning
    }
    
    # Internal variable to store the source of experimental data
    _experimental_data_source = "Not Found" 

    # 1. Call ASKCOS API for reaction classification
    askcos_class_data_internal = _call_askcos_api(reaction_smiles_clean)
    
    # Determine primary_reaction_name from ASKCOS first
    if askcos_class_data_internal and isinstance(askcos_class_data_internal, list) and askcos_class_data_internal:
        top_askcos_entry = askcos_class_data_internal[0]
        if top_askcos_entry.get("reaction_name"):
            final_json_output["reaction_details"]["reaction_name"] = top_askcos_entry["reaction_name"]
        elif top_askcos_entry.get("reaction_classname"):
            final_json_output["reaction_details"]["reaction_name"] = top_askcos_entry["reaction_classname"]

    # 2. Query Local Datasets
    local_dataset_info = query_reaction_dataset(reaction_smiles_clean)
    orderly_data = local_dataset_info.get("orderly_data")
    classification_data_from_dataset = local_dataset_info.get("classification_data")

    # If primary reaction name is still N/A, try from local classification dataset
    if final_json_output["reaction_details"]["reaction_name"] == "N/A" and \
       classification_data_from_dataset and classification_data_from_dataset.get("reaction_name"):
        final_json_output["reaction_details"]["reaction_name"] = classification_data_from_dataset["reaction_name"]

    # 3. Populate from Orderly Dataset OR use LLM Fallback
    if orderly_data:
        _experimental_data_source = "Orderly Dataset"
        final_json_output["reaction_conditions"]["yield_from_dataset"] = f"{orderly_data.get('yield_000')}%" if orderly_data.get('yield_000', 'nan').lower() != 'nan' else "N/A"
        final_json_output["reaction_conditions"]["temperature_from_dataset"] = orderly_data.get('temperature', "N/A")
        final_json_output["reaction_conditions"]["time_from_dataset"] = orderly_data.get('rxn_time', "N/A")
        final_json_output["reaction_conditions"]["atmosphere_llm_or_dataset"] = orderly_data.get('atmosphere', "N/A") # if 'atmosphere' is in your orderly_data
        
        if orderly_data.get('procedure_details'):
            raw_proc = orderly_data['procedure_details']
            # Using LLM to reformat procedure steps from dataset
            llm_proc_data = chem_agent._parse_llm_json_response(
                chem_agent._get_llm_completion_with_fallback(
                    "Reformat this chemical procedure into a list of steps. Return JSON: {\"steps\": []}",
                    f"Procedure: {raw_proc[:2000]}", max_tokens=1000
                ), default_on_error={"steps": [f"Raw from dataset: {raw_proc[:300]}..."] }
            )
            final_json_output["procedure_steps"] = llm_proc_data.get("steps", [f"Raw: {raw_proc[:300]}..."])
        
        # Initial population for reagents/solvents (will be refined with pricing later)
        initial_reagents_solvents = []
        orderly_reagents = orderly_data.get('agents_list', [])
        orderly_solvents = orderly_data.get('solvents_list', [])
        for agent_name in orderly_reagents:
            initial_reagents_solvents.append({"name": agent_name, "role": "Reagent (from Orderly Dataset)"})
        for solvent_name in orderly_solvents:
            initial_reagents_solvents.append({"name": solvent_name, "role": "Solvent (from Orderly Dataset)"})
        # We'll process this list later to add pricing

    else: # LLM Fallback for experimental details
        _experimental_data_source = "LLM Prediction"
        reactant_names_for_llm = [chem_agent.analyze_chemical(r).name or r for r in reactants_smi_list]
        product_names_for_llm = [chem_agent.analyze_chemical(p).name or p for p in products_smi_list]

        llm_predicted_details = _get_llm_predicted_reaction_details(
            reaction_smiles_clean, reactant_names_for_llm, product_names_for_llm,
            askcos_class_data_internal, 
            chem_agent 
        )
        final_json_output["reaction_conditions"]["temperature_from_dataset"] = llm_predicted_details.get("predicted_temperature", "N/A")
        final_json_output["reaction_conditions"]["time_from_dataset"] = llm_predicted_details.get("predicted_time", "N/A")
        final_json_output["reaction_conditions"]["atmosphere_llm_or_dataset"] = llm_predicted_details.get("predicted_atmosphere", "N/A")
        final_json_output["reaction_conditions"]["yield_from_dataset"] = llm_predicted_details.get("predicted_yield_range", "N/A")
        final_json_output["procedure_steps"] = llm_predicted_details.get("predicted_procedure_outline", ["LLM could not predict procedure."])
        
        initial_reagents_solvents = []
        for r_name in llm_predicted_details.get("predicted_reagents_catalysts", []):
            initial_reagents_solvents.append({"name": r_name, "role": "Reagent/Catalyst (LLM Predicted)"})
        for s_name in llm_predicted_details.get("predicted_solvents", []):
            initial_reagents_solvents.append({"name": s_name, "role": "Solvent (LLM Predicted)"})
    
    # You might want to include _experimental_data_source in the output if useful, e.g.
    # final_json_output["experimental_data_source_note"] = _experimental_data_source


    # 4. Visualization
    visualization_path_internal = None # Store viz path internally
    if ag_tools:
        try:
            viz_path_result = ag_tools.visualize_chemical_structure(reaction_smiles_clean)
            if viz_path_result and not str(viz_path_result).lower().startswith('error') and ".png" in viz_path_result:
                visualization_path_internal = viz_path_result
        except Exception: pass

    # 5. Populate Reactant/Product Names
    for r_smi in reactants_smi_list:
        props = chem_agent.analyze_chemical(r_smi)
        final_json_output["reaction_details"]["reactants_identified"].append(props.name or props.iupac_name or r_smi)
    for p_smi in products_smi_list:
        props = chem_agent.analyze_chemical(p_smi)
        final_json_output["reaction_details"]["products_identified"].append(props.name or props.iupac_name or p_smi)
    
    # 6. Refine Reagents/Solvents with Pricing
    processed_reagent_solvent_names_lower = set()
    
    # First, process items already gathered from Orderly or LLM prediction
    for item in initial_reagents_solvents: # Use the temporary list
        comp_name = item["name"]
        props = chem_agent.analyze_chemical(comp_name)
        price_source_str, price_confidence_str, _ = _get_price_source_confidence_for_reaction_component(
            props.supplier_info, chem_agent.llm_provider
        )
        # Add to final_json_output["reagents_and_solvents"]
        final_json_output["reagents_and_solvents"].append({
            "name": props.name or comp_name, # Use resolved name
            "role": item["role"],
            "price_per_unit": props.estimated_price_per_kg,
            "currency": props.price_currency,
            "unit_basis": "kg" if props.estimated_price_per_kg is not None else None,
            "price_source": price_source_str,
            "price_confidence": price_confidence_str
            # "smiles_for_component": props.smiles # Optional: if you want SMILES in this list
        })
        processed_reagent_solvent_names_lower.add((props.name or comp_name).lower())
        if props.smiles: processed_reagent_solvent_names_lower.add(props.smiles.lower())

    # Add common auxiliaries if not already present
    all_lhs_smi_plus_agents = set(reactants_smi_list) | set(agents_smi_list)
    for smi_cand in all_lhs_smi_plus_agents:
        # Check if SMILES or its typical name is already processed
        props_aux = chem_agent.analyze_chemical(smi_cand)
        name_for_aux_component = props_aux.name or props_aux.iupac_name or smi_cand
        
        if name_for_aux_component.lower() in processed_reagent_solvent_names_lower or \
           (props_aux.smiles and props_aux.smiles.lower() in processed_reagent_solvent_names_lower):
            continue
        if smi_cand in products_smi_list: continue

        is_common_aux = (name_for_aux_component.lower() in COMMON_AUXILIARIES_BY_NAME_LOWER or 
                         (props_aux.smiles and props_aux.smiles in COMMON_AUXILIARIES_BY_SMILES))

        if is_common_aux:
            price_source_str, price_confidence_str, _ = _get_price_source_confidence_for_reaction_component(
                props_aux.supplier_info, chem_agent.llm_provider)
            final_json_output["reagents_and_solvents"].append({
                "name": name_for_aux_component,
                "role": "Reagent/Solvent (Common Auxiliary, Auto-detected)",
                "price_per_unit": props_aux.estimated_price_per_kg,
                "currency": props_aux.price_currency,
                "unit_basis": "kg" if props_aux.estimated_price_per_kg is not None else None,
                "price_source": price_source_str,
                "price_confidence": price_confidence_str
                # "smiles_for_component": props_aux.smiles # Optional
            })
            processed_reagent_solvent_names_lower.add(name_for_aux_component.lower())
            if props_aux.smiles: processed_reagent_solvent_names_lower.add(props_aux.smiles.lower())


    # 7. LLM for Overall Reaction Safety & Operational Notes
    safety_llm_context = f"""Reaction SMILES: {reaction_smiles_clean}
Reactants: {', '.join(final_json_output['reaction_details']['reactants_identified'])}
Products: {', '.join(final_json_output['reaction_details']['products_identified'])}
Primary Reaction Name: {final_json_output['reaction_details'].get('reaction_name', 'N/A')} 
Identified Reagents/Solvents: {', '.join([item['name'] for item in final_json_output['reagents_and_solvents']])}"""

    safety_notes_prompt_system = "You are a chemical safety expert. Provide concise reaction safety assessment and operational notes in JSON."
    safety_notes_prompt_user = f"""For the reaction context:
{safety_llm_context}
Provide:
1. Overall safety assessment for *this specific reaction* (1-2 sentences).
2. Key *operational notes or precautions*.
Return JSON: {{"safety": "...", "notes": "..."}}""" # Match target keys

    llm_safety_notes_response = chem_agent._get_llm_completion_with_fallback(safety_notes_prompt_system, safety_notes_prompt_user, max_tokens=600)
    parsed_safety_data = chem_agent._parse_llm_json_response(llm_safety_notes_response)
    if parsed_safety_data:
        final_json_output["safety_and_notes"]["safety"] = parsed_safety_data.get("safety", "N/A")
        final_json_output["safety_and_notes"]["notes"] = parsed_safety_data.get("notes", "N/A")
    
    # Ensure all keys from your target structure are present, even if with default values
    # Most are covered by initialization, but double-check if any could be missing after logic.
    # For example, if initial_reagents_solvents was empty and no common auxiliaries were added, 
    # final_json_output["reagents_and_solvents"] would remain an empty list, which is fine.

    return {
        'visualization_path': visualization_path_internal, # Use the internally stored path
        'analysis': final_json_output, # This is now the structured JSON
        'analysis_context': "full_reaction_recipe_card_v9_target_keys", 
        'processed_smiles_for_tools': reaction_smiles_clean
    }

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
            'analysis_context': "compound_full_text_report_agent_v7", 
            'processed_smiles_for_tools': props.smiles or compound_smiles
        }
    except Exception as e:
        return {'visualization_path': None, 'analysis': {"error": f"Error analyzing compound {compound_smiles}: {e}"},
                'analysis_context': "compound_analysis_error_v7", 'processed_smiles_for_tools': compound_smiles}


def handle_followup_question(query_text, reaction_smiles, original_compound_name=None, callbacks=None):
    cached_data_wrapper = reaction_cache.get(reaction_smiles, {})
    structured_analysis = cached_data_wrapper.get('analysis') 

    if not structured_analysis or not isinstance(structured_analysis, dict):
        return { "analysis": None, "analysis_context": "followup_no_structured_cache_v7" }

    query_lower = query_text.lower()
    response_text = None

    if any(k in query_lower for k in ['classification', 'reaction type', 'askcos']):
        askcos_info = structured_analysis.get('reaction_summary', {}).get('askcos_classification')
        if askcos_info and isinstance(askcos_info, list) and askcos_info:
            top_class = askcos_info[0]
            response_text = (f"ASKCOS Top Classification: {top_class.get('reaction_name', top_class.get('reaction_classname', 'N/A'))} "
                             f"(Certainty: {top_class.get('prediction_certainty', 'N/A'):.2f}).")
        else: response_text = "ASKCOS classification not available in summary."

    elif any(k in query_lower for k in ['procedure', 'steps', 'method']):
        proc_steps = structured_analysis.get('procedure_outline', [])
        source = structured_analysis.get('reaction_summary', {}).get('experimental_data_source', 'Unknown source')
        if proc_steps and proc_steps[0].lower() != "procedure details not available.":
            response_text = f"Procedure Steps (Source: {source}):\n" + "\n".join([f"{i+1}. {step}" for i, step in enumerate(proc_steps)])
        else: response_text = f"Procedure details not available in summary (Source: {source})."
    
    elif any(k in query_lower for k in ['condition', 'temperature', 'time', 'atmosphere', 'yield']):
        cond = structured_analysis.get('reaction_conditions', {})
        source = structured_analysis.get('reaction_summary', {}).get('experimental_data_source', 'Unknown source')
        details = [f"Source: {source}"]
        if 'temperature' in query_lower or 'condition' in query_lower: details.append(f"Temp: {cond.get('temperature', 'N/A')}")
        if 'time' in query_lower or 'condition' in query_lower: details.append(f"Time: {cond.get('time', 'N/A')}")
        if 'atmosphere' in query_lower or 'condition' in query_lower: details.append(f"Atmosphere: {cond.get('atmosphere', 'N/A')}")
        if 'yield' in query_lower or 'condition' in query_lower : details.append(f"Yield: {cond.get('yield', 'N/A')}")
        response_text = "Reaction Conditions: " + "; ".join(details) if len(details) > 1 else "Requested condition not found."


    elif any(k in query_lower for k in ['reagent', 'solvent', 'catalyst', 'component price', 'cost']):
        components = structured_analysis.get('reagents_and_solvents', [])
        source = structured_analysis.get('reaction_summary', {}).get('experimental_data_source', 'Unknown source for reagents/solvents')
        found_components = [f"Data Source for Reagents/Solvents: {source}"]
        for comp in components:
            comp_name_lower = comp.get('name', '').lower()
            comp_role_lower = comp.get('role', '').lower()
            if any(keyword in query_lower for keyword in [comp_name_lower, comp_role_lower] if keyword and keyword !="n/a") or \
               ('reagent' in query_lower and 'reagent' in comp_role_lower) or \
               ('solvent' in query_lower and 'solvent' in comp_role_lower) or \
               ('catalyst' in query_lower and ('catalyst' in comp_role_lower or 'reagent' in comp_role_lower) ):
                
                price_str = comp.get('price_info', 'Price N/A')
                found_components.append(f"{comp.get('name')} ({comp.get('role')}): {price_str}")
        if len(found_components) > 1: 
            response_text = "Reagent/Solvent Information:\n" + "\n".join(found_components)
        else: response_text = "Requested reagent/solvent information not found or not detailed in summary."
        

    elif any(k in query_lower for k in ['safety', 'hazard', 'precaution', 'note']):
        safety_notes = structured_analysis.get('safety_and_operational_notes', {})
        overall_safety = safety_notes.get('overall_reaction_safety', 'N/A')
        op_notes = safety_notes.get('key_operational_notes', 'N/A')
        response_text = f"Overall Safety: {overall_safety}\nOperational Notes: {op_notes}"

    if response_text:
        return {"visualization_path": None, "analysis": response_text, "analysis_context": "followup_structured_answer_recipe_v7", "processed_smiles_for_tools": reaction_smiles}
    return { "analysis": None, "analysis_context": "followup_property_unmatched_structured_recipe_v7" }

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
            func_name = func.__name__; param_name = "chemical_identifier"; param_desc = "Input for the tool."
            if func_name == "get_full_chemical_report": param_name, param_desc = "chemical_identifier", "The name, SMILES string, or CAS number of the chemical."
            elif func_name == "convert_name_to_smiles": param_name, param_desc = "chemical_name", "The chemical name."
            elif func_name == "convert_smiles_to_name": param_name, param_desc = "smiles_string", "The SMILES string."
            elif func_name == "suggest_disconnections": param_name, param_desc = "smiles", "The SMILES string of the molecule."
            elif func_name == "query_specific_property_for_reaction":
                assistant_llm_tools_definition.append({"type": "function", "function": { "name": func_name, "description": func.__doc__ or f"Executes {func_name} tool.", "parameters": {"type": "object", "properties": {"reaction_smiles": {"type": "string", "description": "The Reaction SMILES string."},"property_to_query": {"type": "string", "description": "The specific property like 'yield' or 'temperature'."}}, "required": ["reaction_smiles", "property_to_query"]}}}); continue
            else: param_name, param_desc = "smiles_or_reaction_smiles", "The SMILES string of the compound or reaction."
            assistant_llm_tools_definition.append({"type": "function", "function": { "name": func_name, "description": (func.__doc__ or f"Executes {func_name} tool.").splitlines()[0].strip(), "parameters": {"type": "object", "properties": {param_name: {"type": "string", "description": param_desc}}, "required": [param_name]}}})

        tool_agent_llm_config = llm_config_chatbot_agent.copy()
        if assistant_llm_tools_definition: tool_agent_llm_config["tools"] = assistant_llm_tools_definition
        
        _assistant_tool_agent = autogen.AssistantAgent(name="ChemistryToolAgent_v7", llm_config=tool_agent_llm_config, system_message=TOOL_AGENT_SYSTEM_MESSAGE, is_termination_msg=lambda x: isinstance(x.get("content"), str) and x.get("content", "").rstrip().endswith("TERMINATE") or (isinstance(x.get("content"), str) and "I can only perform specific chemical analyses" in x.get("content","") and not x.get("tool_calls")))
        _user_proxy_tool_agent = autogen.UserProxyAgent(name="UserProxyToolExecutor_v7", human_input_mode="NEVER", max_consecutive_auto_reply=2, code_execution_config=False, function_map={func.__name__: func for func in tool_functions_for_tool_agent}, is_termination_msg=lambda x: (isinstance(x.get("content"), str) and x.get("content", "").rstrip().endswith("TERMINATE")) or (isinstance(x.get("content"), str) and "I can only perform specific chemical analyses" in x.get("content","")))
    return _assistant_tool_agent, _user_proxy_tool_agent

def run_autogen_tool_agent_query(user_input: str, callbacks=None):
    assistant, user_proxy = get_tool_agents()
    if not assistant or not user_proxy: return {"analysis": "Tool agents not initialized (ag_tools might be missing).", "visualization_path": None}
    user_proxy.reset(); assistant.reset(); ai_response_text = "Tool agent did not provide a clear answer (default)."
    try:
        user_proxy.initiate_chat(recipient=assistant, message=user_input, max_turns=3, request_timeout=llm_config_chatbot_agent.get("timeout", 60) + 10)
        messages = user_proxy.chat_messages.get(assistant, [])
        if messages:
            last_msg_obj = messages[-1]
            if last_msg_obj.get("role") == "assistant" and last_msg_obj.get("content"): ai_response_text = last_msg_obj["content"].strip()
            elif last_msg_obj.get("content"): ai_response_text = last_msg_obj.get("content").strip()
        if ai_response_text.upper() == "TERMINATE" or ai_response_text == "Tool agent did not provide a clear answer (default).":
            if messages and len(messages) > 1: potential_reply = messages[-2].get("content", "").strip();
            if potential_reply and potential_reply.upper() != "TERMINATE": ai_response_text = potential_reply
        if ai_response_text.upper().endswith("TERMINATE"): ai_response_text = ai_response_text[:-len("TERMINATE")].strip()
        if not ai_response_text.strip() and ai_response_text != "Tool agent did not provide a clear answer (default).": ai_response_text = "Tool agent processing complete."
        
        viz_path_agent = None
        if "static/autogen_visualizations/" in ai_response_text: match_viz = re.search(r"(static/autogen_visualizations/[\w\-\.\_]+\.png)", ai_response_text);
        if match_viz: viz_path_agent = match_viz.group(1)
        return {"visualization_path": viz_path_agent, "analysis": ai_response_text }
    except Exception as e: print(f"Error in run_autogen_tool_agent_query: {e}\n{traceback.format_exc()}"); return {"visualization_path": None, "analysis": f"An error occurred in the tool agent: {str(e)}"}

CHATBOT_TOOL_PARAM_INFO = {
    "get_functional_groups": {"param_name": "smiles_or_reaction_smiles", "description": "The SMILES or reaction SMILES string."},
    "convert_name_to_smiles": {"param_name": "chemical_name", "description": "The chemical name."},
    "suggest_disconnections": {"param_name": "smiles", "description": "The SMILES string of the molecule for disconnection suggestions."},
    "convert_smiles_to_name": {"param_name": "smiles_string", "description": "The SMILES string to convert to a name."},
    "visualize_chemical_structure": {"param_name": "smiles_or_reaction_smiles", "description": "The SMILES string for visualization."},
    "get_full_chemical_report": {"param_name": "chemical_identifier", "description": "The name, SMILES, or CAS number for a full chemical report."}
}

def _update_chatbot_system_message_with_moi():
    # ... (remains the same) ...
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
    # ... (remains the same) ...
    global _assistant_chatbot_agent, _user_proxy_chatbot_agent, llm_config_chatbot_agent, CHATBOT_TOOL_PY_FUNCTIONS_BASE, CHATBOT_TOOL_PARAM_INFO
    if not CHATBOT_TOOL_PY_FUNCTIONS_BASE: print("Critical: CHATBOT_TOOL_PY_FUNCTIONS_BASE is empty. Cannot create chatbot agents."); return None, None
    if _assistant_chatbot_agent is None:
        tools_cfg = []
        for func in CHATBOT_TOOL_PY_FUNCTIONS_BASE:
            p_info = CHATBOT_TOOL_PARAM_INFO.get(func.__name__, {"param_name": "input", "description": "Input."})
            tools_cfg.append({"type": "function", "function": {"name": func.__name__, "description": (func.__doc__ or '').splitlines()[0].strip(), "parameters": {"type": "object", "properties": {p_info["param_name"]: {"type": "string", "description": p_info["description"]}}, "required": [p_info["param_name"]]}}})
        
        cfg = llm_config_chatbot_agent.copy()
        if tools_cfg: cfg["tools"] = tools_cfg
        
        _assistant_chatbot_agent = autogen.AssistantAgent(name="ChemistryChatbotAgent_MOI_v7", llm_config=cfg, system_message="Initializing MOI-aware ChemCopilot...", is_termination_msg=lambda x: isinstance(x.get("content"), str) and x.get("content", "").rstrip().endswith("TERMINATE"))
        _user_proxy_chatbot_agent = autogen.UserProxyAgent(name="UserProxyChatConversational_MOI_v7", human_input_mode="NEVER", max_consecutive_auto_reply=3, code_execution_config=False, function_map={tool.__name__: tool for tool in CHATBOT_TOOL_PY_FUNCTIONS_BASE}, is_termination_msg=lambda x: (isinstance(x.get("content"), str) and x.get("content", "").rstrip().endswith("TERMINATE")) or (isinstance(x.get("content"), str) and "I am a specialized chemistry assistant" in x.get("content", "")))
        _update_chatbot_system_message_with_moi()
    return _assistant_chatbot_agent, _user_proxy_chatbot_agent

def clear_chatbot_memory_autogen():
    # ... (remains the same) ...
    global _current_moi_context, _assistant_chatbot_agent, _user_proxy_chatbot_agent
    _current_moi_context = {"name": None, "smiles": None}
    if _assistant_chatbot_agent: _assistant_chatbot_agent.reset()
    if _user_proxy_chatbot_agent: _user_proxy_chatbot_agent.reset()
    if _assistant_chatbot_agent: _update_chatbot_system_message_with_moi()

MAX_CHATBOT_TURNS = 5
def run_autogen_chatbot_query(user_input: str, callbacks=None):
    # ... (remains largely the same, ensure it handles "error" key for consistency) ...
    global _current_moi_context, _assistant_chatbot_agent, _user_proxy_chatbot_agent
    priming_match = re.match(r"Let's discuss the molecule of interest: (.*?) with SMILES (.*)\. Please acknowledge\.", user_input, re.IGNORECASE)
    if priming_match:
        moi_name, moi_smiles = priming_match.groups(); _current_moi_context["name"] = moi_name.strip(); _current_moi_context["smiles"] = moi_smiles.strip()
        if not _assistant_chatbot_agent or not _user_proxy_chatbot_agent: get_chatbot_agents()
        if not _assistant_chatbot_agent: return {"analysis": "Chatbot agents could not be initialized.", "visualization_path": None, "error": "Agent init failed"}
        _update_chatbot_system_message_with_moi(); return {"analysis": f"Acknowledged. We are now discussing {_current_moi_context['name']} ({_current_moi_context['smiles']}). How can I help?", "visualization_path": None, "error": None}

    if not _assistant_chatbot_agent or not _user_proxy_chatbot_agent: get_chatbot_agents()
    if not _assistant_chatbot_agent: return {"analysis": "Chatbot agents failed to initialize.", "visualization_path": None, "error": "Agent init failed"}
    else: _update_chatbot_system_message_with_moi() # Ensure MOI is current
    if _assistant_chatbot_agent: _assistant_chatbot_agent.reset() # Reset before each new conversational query
    if _user_proxy_chatbot_agent: _user_proxy_chatbot_agent.reset()

    ai_response_text = "Chatbot did not provide a clear answer (default)."; viz_path = None
    try:
        _user_proxy_chatbot_agent.initiate_chat(recipient=_assistant_chatbot_agent, message=user_input, max_turns=MAX_CHATBOT_TURNS, request_timeout=llm_config_chatbot_agent.get("timeout", 90) + 30, clear_history=True) # clear_history=True for stateless turn
        conv_history = _assistant_chatbot_agent.chat_messages.get(_user_proxy_chatbot_agent, []) # From assistant's perspective
        if conv_history: # Get the last textual response from the assistant
            for msg_obj in reversed(conv_history):
                if msg_obj.get("role") == "assistant":
                    if msg_obj.get("tool_calls"): continue # Skip tool call messages
                    if msg_obj.get("content"): ai_response_text = msg_obj.get("content", "").strip(); break
            if ai_response_text == "Chatbot did not provide a clear answer (default)." and conv_history[-1].get("role") == "assistant" and conv_history[-1].get("content"):
                 ai_response_text = conv_history[-1].get("content").strip() # Last chance if loop didn't find specific one
        
        if ai_response_text.upper().endswith("TERMINATE"): ai_response_text = ai_response_text[:-len("TERMINATE")].strip()
        if not ai_response_text.strip() and "Chatbot did not provide a clear answer" not in ai_response_text: ai_response_text = "Chatbot processing complete."

        # Check for visualization in tool responses if any
        user_proxy_tool_calls_history = _user_proxy_chatbot_agent.chat_messages.get(_assistant_chatbot_agent, [])
        for msg_item in user_proxy_tool_calls_history:
            if msg_item.get("role") == "tool" and "static/autogen_visualizations/" in msg_item.get("content", ""):
                match_viz = re.search(r"(static/autogen_visualizations/[\w\-\.\_]+\.png)", msg_item.get("content", ""))
                if match_viz: viz_path = match_viz.group(1); break
        # Also check final AI text if viz path was embedded there
        if not viz_path and "static/autogen_visualizations/" in ai_response_text:
             match_viz = re.search(r"(static/autogen_visualizations/[\w\-\.\_]+\.png)", ai_response_text)
             if match_viz: viz_path = match_viz.group(1)
        return {"visualization_path": viz_path, "analysis": ai_response_text, "error": None }
    except openai.APITimeoutError as e: return { "visualization_path": None, "analysis": f"OpenAI API timed out: {str(e)}", "error": str(e)}
    except Exception as e: return { "visualization_path": None, "analysis": f"An error occurred in the MOI chatbot: {str(e)}", "error": str(e) }

def enhanced_query(full_query: str, callbacks=None, original_compound_name: str = None):
    # ... (Main routing logic - this should largely remain the same, but ensure it caches the full JSON from handle_full_info) ...
    # Key change: When `handle_full_info` is called, its full JSON result should be cached if needed by `handle_followup_question`.
    # `reaction_cache[reaction_smiles_for_tools]['full_info'] = final_result`
    global _current_moi_context
    final_result = {}
    query_context_for_filename = "unknown_query_type_v7"
    reaction_smiles_for_tools = extract_reaction_smiles(full_query)
    compound_smiles_for_tools = None
    if not reaction_smiles_for_tools: compound_smiles_for_tools = extract_single_compound_smiles(full_query)
    query_lower = full_query.lower()
    full_info_keywords = ["full info", "full data", "complete analysis", "details about", "tell me about", "explain this", "analyze this reaction", "analyze this compound", "recipe card"]
    
    try:
        if compound_smiles_for_tools and any(keyword in query_lower for keyword in full_info_keywords):
            _current_moi_context["name"] = original_compound_name or compound_smiles_for_tools; _current_moi_context["smiles"] = compound_smiles_for_tools
            final_result = handle_compound_full_info(full_query, compound_smiles_for_tools, original_compound_name or compound_smiles_for_tools, callbacks=callbacks)
            query_context_for_filename = final_result.get('analysis_context', 'compound_full_report_from_smiles_v7')
            if final_result.get('analysis'): # Cache compound report if it's a string
                compound_cache.setdefault(compound_smiles_for_tools, {})['full_report_text'] = final_result['analysis']


        elif reaction_smiles_for_tools and any(keyword in query_lower for keyword in full_info_keywords):
            _current_moi_context["name"] = original_compound_name or reaction_smiles_for_tools; _current_moi_context["smiles"] = reaction_smiles_for_tools
            final_result = handle_full_info(full_query, reaction_smiles_for_tools, original_compound_name or reaction_smiles_for_tools, callbacks=callbacks) 
            query_context_for_filename = final_result.get('analysis_context', 'reaction_full_recipe_card_v7')
            # Cache the structured JSON from handle_full_info
            if final_result.get('analysis') and isinstance(final_result.get('analysis'), dict):
                reaction_cache.setdefault(reaction_smiles_for_tools, {})['full_info'] = final_result # This stores the whole dict with 'analysis' as the JSON card

        elif not compound_smiles_for_tools and not reaction_smiles_for_tools and any(keyword in query_lower for keyword in full_info_keywords) and ag_tools:
            name_entity = None
            match_named = re.search(r"full info(?: about| for| of)?\s+(?:named|called)\s*['\"]?([\w\s\-(),]+?)['\"]?(?:\s+with smiles|\s+smiles|\s+cas|\.|$)", query_lower, re.IGNORECASE)
            if match_named: name_entity = match_named.group(1).strip()
            else:
                match_general = re.search(r"full info(?: about| for| of)?\s+([\w\s\-(),]+?)(?:\s+with smiles|\s+smiles|\s+cas|\.|$)", query_lower, re.IGNORECASE)
                if match_general: name_entity = match_general.group(1).strip()
            
            if name_entity and not extract_single_compound_smiles(name_entity) and not extract_reaction_smiles(name_entity):
                n2s_output = ag_tools.convert_name_to_smiles(name_entity)
                smiles_from_name = None
                if isinstance(n2s_output, str): s_match = re.search(r"SMILES:\s*([^\s\n]+)", n2s_output);
                if s_match: smiles_from_name = s_match.group(1).strip()
                
                if smiles_from_name:
                    _current_moi_context["name"] = name_entity; _current_moi_context["smiles"] = smiles_from_name
                    if ">>" in smiles_from_name: final_result = handle_full_info(f"Full info for {smiles_from_name}", smiles_from_name, name_entity)
                    else: final_result = handle_compound_full_info(f"Full info for {smiles_from_name}", smiles_from_name, name_entity)
                    query_context_for_filename = final_result.get('analysis_context', 'full_info_from_name_conversion_v7')
                    # Cache result similar to above
                    if ">>" in smiles_from_name and final_result.get('analysis') and isinstance(final_result.get('analysis'), dict): reaction_cache.setdefault(smiles_from_name, {})['full_info'] = final_result
                    elif final_result.get('analysis'): compound_cache.setdefault(smiles_from_name, {})['full_report_text'] = final_result['analysis']
                else: 
                    final_result = {"analysis": {"error":f"Could not find SMILES for name '{name_entity}' via tool."}, "processed_smiles_for_tools": None}
                    query_context_for_filename = "full_info_name_conversion_failed_tool_v7"
            elif not name_entity : 
                 final_result = run_autogen_chatbot_query(full_query)
                 query_context_for_filename = "full_info_no_entity_to_chatbot_v7"

        if not final_result: # Fallback to tool agent or chatbot if no "full info" keyword matched or above failed
            if reaction_smiles_for_tools:
                _current_moi_context["name"] = original_compound_name or reaction_smiles_for_tools; _current_moi_context["smiles"] = reaction_smiles_for_tools
                followup_res = handle_followup_question(full_query, reaction_smiles_for_tools) # Follow-up needs cached 'full_info'
                if followup_res and followup_res.get('analysis'): final_result = followup_res
                else: final_result = run_autogen_chatbot_query(full_query) # General chatbot if no specific followup
                query_context_for_filename = final_result.get('analysis_context', 'chatbot_or_followup_reaction_v7')
            elif compound_smiles_for_tools:
                 _current_moi_context["name"] = original_compound_name or compound_smiles_for_tools; _current_moi_context["smiles"] = compound_smiles_for_tools
                 tool_res = run_autogen_tool_agent_query(full_query) # Try direct tool first
                 if tool_res and tool_res.get("analysis") and "I can only perform specific" not in tool_res["analysis"]: final_result = tool_res
                 else: final_result = run_autogen_chatbot_query(full_query) # Fallback to general chatbot
                 query_context_for_filename = final_result.get('analysis_context', 'tool_or_chatbot_compound_v7')
            else: # No SMILES extracted, must be general chatbot
                final_result = run_autogen_chatbot_query(full_query)
                query_context_for_filename = "chatbot_general_no_smiles_v7"
        
        if not final_result: # Ultimate fallback if all above resulted in empty final_result
            final_result = run_autogen_chatbot_query(full_query)
            query_context_for_filename = "chatbot_fallback_routing_failed_v7"

        # Standardize output structure
        final_result["current_moi_name"] = _current_moi_context.get("name")
        final_result["current_moi_smiles"] =  _current_moi_context.get("smiles")
        analysis_content = final_result.get("analysis")
        smiles_to_save = final_result.get('processed_smiles_for_tools', reaction_smiles_for_tools or compound_smiles_for_tools)
        if query_context_for_filename == "unknown_query_type_v7" and final_result.get('analysis_context'): query_context_for_filename = final_result.get('analysis_context')

        # Save to file logic
        should_save = False
        if isinstance(analysis_content, dict) and not analysis_content.get("error"): should_save = True
        elif isinstance(analysis_content, str) and len(analysis_content.strip()) > 50 and \
             not any(phrase in analysis_content for phrase in ["I am a specialized chemistry assistant", "I can only perform specific chemical analyses"]): should_save = True
        
        if smiles_to_save and should_save and not query_context_for_filename.startswith("visualization_"):
            save_analysis_to_file(smiles_to_save, analysis_content, query_context_for_filename, original_compound_name)

        if 'processed_smiles_for_tools' not in final_result: final_result['processed_smiles_for_tools'] = smiles_to_save
        final_result['analysis_context'] = query_context_for_filename # Ensure context is set
        return final_result

    except Exception as e:
        tb_str = traceback.format_exc(); print(f"CRITICAL Error in enhanced_query: {str(e)}\n{tb_str}")
        err_dict = {"error": f"Error processing query: {str(e)}."} 
        smiles_err_ctx = compound_smiles_for_tools or reaction_smiles_for_tools or "no_entity"
        save_analysis_to_file(smiles_err_ctx, f"Query: {full_query}\n{str(err_dict)}\n{tb_str}", "enhanced_query_CRITICAL_error_v7", original_compound_name)
        return {"visualization_path": None, "analysis": err_dict, "error": str(e), "processed_smiles_for_tools": smiles_err_ctx, 
                "analysis_context": "enhanced_query_exception_v7", "current_moi_name": _current_moi_context.get("name"), "current_moi_smiles":  _current_moi_context.get("smiles")}

def display_analysis_result(title: str, analysis_result: dict, is_chatbot: bool = False):
    print(f"\n--- {title} ---")
    if not analysis_result or not isinstance(analysis_result, dict):
        print(f"Invalid analysis result format. Raw: {analysis_result}")
        print(f"--- End of {title} ---\n"); return

    analysis_content = analysis_result.get("analysis")

    if isinstance(analysis_content, dict): 
        if "error" in analysis_content: 
            print(f"Error in analysis: {analysis_content['error']}")
        else:
            # analysis_content is expected to be the structured JSON matching your target
            print("Structured Recipe Card Analysis:")
            try: 
                print(json.dumps(analysis_content, indent=2))
            except TypeError: 
                print("Error: Could not serialize the analysis content to JSON for display.")
                print(f"Raw analysis content (dict): {str(analysis_content)[:1000]}...")
    elif isinstance(analysis_content, str): 
        # This case is more for compound reports or direct tool/chatbot text outputs
        cleaned_text = re.sub(r"!\[.*?\]\((.*?)\)", r"[Image: \1]", analysis_content)
        if is_chatbot: 
            print(f"Chem Copilot (Text Response): {cleaned_text}")
        else: 
            print(f"Analysis (Text Report):\n{cleaned_text}")
    elif analysis_result.get("error"): 
        print(f"Error: {analysis_result.get('error')}")
    else: 
        print(f"Could not retrieve/generate response. Raw content: {analysis_content}")

    viz_path = analysis_result.get("visualization_path")
    if viz_path: 
        print(f"Visualization: {viz_path}") # This should be the web-accessible path if API is used
    
    print(f"--- End of {title} ---\n")

if __name__ == "__main__":
    # ... (remains the same as your provided script) ...
    if not OPENAI_API_KEY and DEFAULT_LLM_PROVIDER == "openai": print("CRITICAL: OPENAI_API_KEY not set. Exiting."); exit(1)
    if not PERPLEXITY_API_KEY and DEFAULT_LLM_PROVIDER == "perplexity":
        if DEFAULT_LLM_PROVIDER == "perplexity" and (not OPENAI_API_KEY or OPENAI_API_KEY == "sk-YOUR_OPENAI_API_KEY_HERE"): print("CRITICAL: PERPLEXITY_API_KEY not set, OpenAI key missing. Exiting."); exit(1)
        elif DEFAULT_LLM_PROVIDER == "perplexity": print("Warning: PERPLEXITY_API_KEY not set. Will try OpenAI if available.")
    if not ag_tools or not ReactionClassifier: print("CRITICAL: Core tools (autogen_tool_functions or ReactionClassifier) failed to load. Exiting."); exit(1)

    dummy_files = {
        "pricing_data.json": {"Benzene": ["C1=CC=CC=C1", 100.0, "Pune"], "Water": ["O", 1.0, "Global"], "Thionyl chloride": ["ClS(=O)Cl", 1800.0, "Mumbai"]},
        "second_source.json": {"Ethanol": ["CCO", 150.0, "Mumbai"], "Dichloromethane": ["ClCCl", 440.0, "Delhi"]},
        "sigma_source.json": {"Methanol (CAS 67-56-1)": ["CO", 120.0, "Bangalore"], "SOCl2": ["ClS(=O)Cl", 1850.0, "Sigma Catalog"]},
        # "benzoic_acid_deriv_prices.json" was in your original example but not used by default loaders
    }
    for fname, content in dummy_files.items():
        fpath = os.path.join(PROJECT_ROOT_DIR, fname)
        if not os.path.exists(fpath):
            try: 
                with open(fpath, 'w', encoding='utf-8') as f: json.dump(content, f, indent=2)
            except IOError: pass
            
    print("\nInteractive Chem Copilot Session (v7 - ASKCOS + LLM Fallback)")
    print(f"Default LLM Provider for Summaries/Properties: {DEFAULT_LLM_PROVIDER.upper()}")
    print("Example reaction query: 'full info for CC(O)C>>CC(=O)C'")
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
            clear_chatbot_memory_autogen()
            last_processed_entity_name_for_saving = None # Reset this too
            print("Chem Copilot: Chat MOI context and agent states cleared."); 
            continue

        current_original_name_for_saving = None
        potential_name = None # Initialize potential_name

        # Try to capture a name provided with "full info for NAME" type queries
        name_match_explicit = re.search(r"full\s+info\s+(?:for|of|about)\s+(?:named|called)\s*['\"]?([^'\"\n\.,>]+?)['\"]?(?:\s+with smiles|\s+smiles|\.|$)", user_input, re.IGNORECASE)
        if name_match_explicit:
            potential_name = name_match_explicit.group(1).strip()
            # Check if the extracted potential_name is not a SMILES itself
            if not (">>" in potential_name or extract_single_compound_smiles(potential_name)):
                 current_original_name_for_saving = potential_name
            else: # It was a SMILES, so don't use it as original_name_for_saving
                potential_name = None # Reset if it was a SMILES
        
        # If not found with "named/called" or if potential_name was reset
        if not current_original_name_for_saving:
            name_match_general = re.search(r"full\s+info\s+(?:for|of|about)\s+([^'\"\n\.,>]+?)(?:\s+with smiles|\s+smiles|\s+cas|\.|$)", user_input, re.IGNORECASE)
            if name_match_general:
                potential_name = name_match_general.group(1).strip()
                # Avoid capturing parts of SMILES or keywords like "reaction"
                if not (">>" in potential_name or extract_single_compound_smiles(potential_name) or potential_name.lower() in ["reaction", "compound"]):
                     current_original_name_for_saving = potential_name
                else:
                    potential_name = None # Reset if it was SMILES or keyword

        # This was the original problematic line, now potential_name is guaranteed to be at least None
        # The logic for setting current_original_name_for_saving is now handled within the if/else blocks above.
        # The purpose of the original line was to set current_original_name_for_saving if potential_name was NOT a SMILES.
        # That logic is now incorporated above.

        priming_match_loop = re.match(r"Let's discuss the molecule of interest: (.*?) with SMILES .*\. Please acknowledge\.", user_input, re.IGNORECASE)
        if priming_match_loop: 
            last_processed_entity_name_for_saving = priming_match_loop.group(1).strip()

        # Use the more specific current_original_name_for_saving if found, otherwise fallback to last_processed if any
        name_to_pass_to_enhanced_query = current_original_name_for_saving or last_processed_entity_name_for_saving
        
        query_result = enhanced_query(user_input, original_compound_name=name_to_pass_to_enhanced_query)

        is_chatbot_response = "chatbot" in query_result.get('analysis_context', '') or \
                              "tool_agent" in query_result.get('analysis_context', '') or \
                              (query_result.get("current_moi_name") is not None) # Check if MOI was set by the query

        display_analysis_result(f"Chem Copilot Response", query_result, is_chatbot=is_chatbot_response)

        analysis_context = query_result.get('analysis_context', '')
        # Update last_processed_entity_name_for_saving if a full info query successfully processed an entity with a name
        if query_result and isinstance(query_result, dict) and \
           ('full_reaction_json_recipe_card' in analysis_context or \
            'compound_full_text_report_agent' in analysis_context or \
            'full_info_from_name_conversion' in analysis_context) and \
           query_result.get('processed_smiles_for_tools'):
            
            # If current_original_name_for_saving was determined for THIS query, it's the most relevant
            if current_original_name_for_saving: 
                 last_processed_entity_name_for_saving = current_original_name_for_saving
            # Else, if the query updated the MOI name and it matches the processed SMILES
            elif query_result.get("current_moi_name") and \
                 query_result.get("current_moi_smiles") == query_result.get('processed_smiles_for_tools'):
                last_processed_entity_name_for_saving = query_result.get("current_moi_name")
            # last_processed_entity_name_for_saving will otherwise retain its previous value (or None)

    print("\nInteractive session ended.")