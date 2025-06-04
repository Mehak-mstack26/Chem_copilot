import os
import re
import time
import pandas as pd
from rdkit import Chem, RDLogger
import traceback
from functools import lru_cache
from typing import Optional, List, Dict, Any
import autogen
import openai # For direct OpenAI calls in handle_full_info etc.
import ast

# For API key loading
try:
    print("[MainScript] Attempting to import api_config...")
    import api_config # This will execute api_config.py
    print(f"[MainScript] api_config imported. OPENAI_API_KEY from env: {os.getenv('OPENAI_API_KEY') is not None}")
    if not os.getenv("OPENAI_API_KEY"):
        print("[MainScript ERROR] api_config was imported but OPENAI_API_KEY is still not set in environment!")
        if hasattr(api_config, 'api_key') and api_config.api_key:
            os.environ["OPENAI_API_KEY"] = api_config.api_key
            print(f"[MainScript] Manually set OPENAI_API_KEY from api_config.api_key. Now set: {os.getenv('OPENAI_API_KEY') is not None}")
        else:
            raise ValueError("api_config.api_key not available after import.")
except ImportError:
    print("Warning: api_config.py not found. Ensure OPENAI_API_KEY is set in your environment directly.")
    if not os.getenv("OPENAI_API_KEY"):
        print("CRITICAL: OPENAI_API_KEY is not set directly either. Exiting.")
        exit(1)
except ValueError as e:
    print(f"CRITICAL Error during API key setup via api_config.py: {e}")
    exit(1)
except Exception as e_cfg:
    print(f"CRITICAL Unexpected error during api_config import or setup: {e_cfg}")
    exit(1)

# Import functions from autogen_tool_functions.py
from tools import autogen_tool_functions as ag_tools
from tools.asckos import ReactionClassifier

RDLogger.DisableLog('rdApp.*')

# --- Autogen LLM Configuration ---
config_list_gpt4o = [
    {
        "model": "gpt-4o",
        "api_key": os.environ.get("OPENAI_API_KEY"),
    }
]
config_list_gpt3_5_turbo = [ # For faster chatbot responses
    {
        "model": "gpt-3.5-turbo-0125", # Or your preferred gpt-3.5 model
        "api_key": os.environ.get("OPENAI_API_KEY"),
    }
]

llm_config_chatbot_agent = { # Configuration for the main chatbot assistant
    "config_list": config_list_gpt3_5_turbo, # Use faster model for chat
    "temperature": 0.1, # Slightly more deterministic
    "timeout": 90, # Timeout for individual LLM calls by the chatbot agent
    # 'tools' will be added dynamically in get_chatbot_agents based on tools_to_register
}

# --- Setup for Saving Analysis Files ---
try:
    PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    PROJECT_ROOT_DIR = os.getcwd()
REACTION_ANALYSIS_OUTPUT_DIR = os.path.join(PROJECT_ROOT_DIR, "reaction_analysis_outputs")
os.makedirs(REACTION_ANALYSIS_OUTPUT_DIR, exist_ok=True)

# --- Utility functions (sanitize_filename, save_analysis_to_file, extract_final_answer - UNCHANGED) ---
def sanitize_filename(name):
    if not isinstance(name, str): name = str(name)
    name = re.sub(r'[^\w\.\-]+', '_', name); return name[:100]

def save_analysis_to_file(reaction_smiles, analysis_text, query_context_type="analysis", original_compound_name=None):
    if not analysis_text or not isinstance(analysis_text, str) or not analysis_text.strip():
        print(f"[SAVE_ANALYSIS] Skipping save: No analysis text provided for '{reaction_smiles}'.")
        return
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    smiles_part = sanitize_filename(reaction_smiles if reaction_smiles else "no_smiles")
    filename_parts = []
    if original_compound_name and original_compound_name != "DirectReactionAnalysis" and original_compound_name != reaction_smiles:
        filename_parts.append(sanitize_filename(original_compound_name))
    smiles_prefix = "cmpd_" if reaction_smiles and ">>" not in reaction_smiles else "rxn_"
    filename_parts.append(f"{smiles_prefix}{smiles_part}")
    filename_parts.append(sanitize_filename(query_context_type))
    filename_parts.append(timestamp)
    filename = "_".join(filter(None, filename_parts)) + ".txt"
    filepath = os.path.join(REACTION_ANALYSIS_OUTPUT_DIR, filename)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            entity_type = "Compound SMILES" if "cmpd_" in smiles_prefix else "Reaction SMILES"
            f.write(f"{entity_type}: {reaction_smiles}\n")
            if original_compound_name and original_compound_name != reaction_smiles:
                 f.write(f"Original Target Context: {original_compound_name}\n")
            f.write(f"Analysis Type: {query_context_type}\n"); f.write(f"Timestamp: {timestamp}\n")
            f.write("="*50 + "\n\n"); f.write(analysis_text)
        print(f"[SAVE_ANALYSIS] Saved analysis to: {filepath}")
    except Exception as e: print(f"[SAVE_ANALYSIS_ERROR] Error saving {filepath}: {e}")

def extract_final_answer(full_output: str):
    match = re.search(r"Final Answer:\s*(.*)", full_output, re.DOTALL)
    return match.group(1).strip() if match else full_output.strip()

# --- Caches and Reaction Classifier ---
dataset_path1 = os.environ.get('REACTION_DATASET_PATH1', None)
dataset_path2 = os.environ.get('REACTION_DATASET_PATH2', None)
try:
    reaction_classifier_core_logic = ReactionClassifier(dataset_path1, dataset_path2)
    print("[MainScript] ReactionClassifier for core logic initialized.")
except Exception as e:
    print(f"Warning: Could not initialize ReactionClassifier for core logic: {e}. Some features in full analysis might be unavailable.")
    reaction_classifier_core_logic = None

reaction_cache = {}
compound_cache = {}

# --- Tool Wrapper for Core Logic Compatibility (AutogenToolWrapper, get_tools_for_core_logic - UNCHANGED) ---
class AutogenToolWrapper:
    def __init__(self, name: str, func: callable, description: str):
        self.name = name
        self.description = description
        self.tool_func = func
    def run(self, args_str: str) -> str:
        try:
            return self.tool_func(args_str)
        except Exception as e:
            return f"Error running tool {self.name}: {str(e)}"

_tools_for_core_logic_instance_list = None
def get_tools_for_core_logic() -> List[AutogenToolWrapper]:
    global _tools_for_core_logic_instance_list
    if _tools_for_core_logic_instance_list is None:
        _tools_for_core_logic_instance_list = [
            AutogenToolWrapper(name="smiles2name", func=ag_tools.convert_smiles_to_name, description=ag_tools.convert_smiles_to_name.__doc__),
            AutogenToolWrapper(name="funcgroups", func=ag_tools.get_functional_groups, description=ag_tools.get_functional_groups.__doc__),
            AutogenToolWrapper(name="bondchangeanalyzer", func=ag_tools.analyze_reaction_bond_changes, description=ag_tools.analyze_reaction_bond_changes.__doc__),
            AutogenToolWrapper(name="chemvisualizer", func=ag_tools.visualize_chemical_structure, description=ag_tools.visualize_chemical_structure.__doc__),
            AutogenToolWrapper(name="nametosmiles", func=ag_tools.convert_name_to_smiles, description=ag_tools.convert_name_to_smiles.__doc__),
            AutogenToolWrapper(name="disconnection", func=ag_tools.suggest_disconnections, description=ag_tools.suggest_disconnections.__doc__),
        ]
    return _tools_for_core_logic_instance_list

# --- Data extraction and processing functions (query_reaction_dataset, extract_reaction_smiles, extract_single_compound_smiles - UNCHANGED) ---
@lru_cache(maxsize=100)
def query_reaction_dataset(reaction_smiles):
    if not reaction_smiles: return None
    if reaction_smiles in reaction_cache and 'dataset_info' in reaction_cache[reaction_smiles]:
        return reaction_cache[reaction_smiles]['dataset_info']
    current_classifier = reaction_classifier_core_logic
    if not current_classifier or \
       (not hasattr(current_classifier, 'dataset1') or current_classifier.dataset1 is None or current_classifier.dataset1.empty) and \
       (not hasattr(current_classifier, 'dataset2') or current_classifier.dataset2 is None or current_classifier.dataset2.empty):
        print("[query_reaction_dataset] Core logic's ReactionClassifier not available or has no data.")
        return None
    try:
        df = None
        if hasattr(current_classifier, 'dataset1') and current_classifier.dataset1 is not None and not current_classifier.dataset1.empty:
            df = current_classifier.dataset1
        elif hasattr(current_classifier, 'dataset2') and current_classifier.dataset2 is not None and not current_classifier.dataset2.empty:
            df = current_classifier.dataset2
        if df is None or df.empty: return None
        fields_to_extract = ['procedure_details', 'rxn_time', 'temperature', 'yield_000', 'reaction_name', 'reaction_classname', 'prediction_certainty']
        smiles_columns = ['rxn_str', 'reaction_smiles', 'smiles', 'rxn_smiles']
        exact_match = None
        for col in smiles_columns:
            if col in df.columns and df[col].dtype == 'object':
                if reaction_smiles is not None:
                    temp_match = df[df[col] == reaction_smiles]
                    if not temp_match.empty:
                        exact_match = temp_match
                        break
        result = {}
        if exact_match is not None and not exact_match.empty:
            row = exact_match.iloc[0]
            for field in fields_to_extract:
                if field in row.index and pd.notna(row[field]) and (not isinstance(row[field], str) or str(row[field]).strip().lower() != "nan"):
                    result[field] = str(row[field])
            for i in range(1, 11):
                key = f'solvent_{i:03d}'
                if key in row.index and pd.notna(row[key]) and (not isinstance(row[key], str) or str(row[key]).strip().lower() != "nan"):
                    result.setdefault('solvents_list', []).append(str(row[key]))
                    if len(result.get('solvents_list', [])) >= 3: break
            for i in range(1, 16):
                key = f'agent_{i:03d}'
                if key in row.index and pd.notna(row[key]) and (not isinstance(row[key], str) or str(row[key]).strip().lower() != "nan"):
                    result.setdefault('agents_list', []).append(str(row[key]))
                    if len(result.get('agents_list', [])) >= 3: break
        reaction_cache.setdefault(reaction_smiles, {})['dataset_info'] = result if result else None
        return result if result else None
    except Exception as e:
        print(f"Error querying dataset for '{reaction_smiles}': {e}")
        return None

def extract_reaction_smiles(query: str) -> Optional[str]:
    smi_core_chars = r"[\w@\[\]\(\)\.\+\-\=\#\:\$\%\~]"
    explicit_pattern = rf"(?i:\b(?:reaction\s+smiles|rxn)\s*[:=]?\s*)({smi_core_chars}+(?:>>{smi_core_chars}+)+)"
    match = re.search(explicit_pattern, query)
    if match:
        smiles = match.group(1).strip()
        parts = smiles.split(">>")
        if len(parts) >= 2 and all(p.strip() for p in parts):
            print(f"[EXTRACT_SMILES] Found by explicit keyword: '{smiles}'")
            return smiles
    standalone_pattern = rf"(?<![\w\/])({smi_core_chars}+(?:>>{smi_core_chars}+)+)(?![\w\/])"
    potential_matches = re.findall(standalone_pattern, query)
    for smi_candidate in potential_matches:
        smi_candidate = smi_candidate.strip()
        parts = smi_candidate.split(">>")
        if len(parts) >= 2 and all(p.strip() for p in parts):
            try:
                if Chem.MolFromSmiles(parts[0].split('.')[0]) and Chem.MolFromSmiles(parts[-1].split('.')[0]):
                    print(f"[EXTRACT_SMILES] Found and validated standalone: '{smi_candidate}'")
                    return smi_candidate
            except: pass
    gt_pattern = rf"(?<![\w\/])({smi_core_chars}+(?:>{smi_core_chars}+)+)(?![\w\/])"
    match_gt = re.search(gt_pattern, query)
    if match_gt:
        temp_smiles = match_gt.group(1).strip()
        if ">>" not in temp_smiles:
            gt_parts = temp_smiles.split('>')
            cleaned_gt_parts = [p.strip() for p in gt_parts if p.strip()]
            if len(cleaned_gt_parts) >= 2:
                products_gt = cleaned_gt_parts[-1]
                reactants_gt = ".".join(cleaned_gt_parts[:-1])
                if reactants_gt and products_gt:
                    try:
                        if Chem.MolFromSmiles(reactants_gt.split('.')[0]) and Chem.MolFromSmiles(products_gt.split('.')[0]):
                            converted_smiles = f"{reactants_gt}>>{products_gt}"
                            print(f"[EXTRACT_SMILES] Converted '>' pattern: '{temp_smiles}' to '>>': '{converted_smiles}'")
                            return converted_smiles
                    except: pass
    print(f"[EXTRACT_SMILES] No valid reaction SMILES found in: '{query[:100]}...'")
    return None

def extract_single_compound_smiles(query: str) -> Optional[str]:
    words = query.split()
    regex_candidates = re.findall(r"[A-Za-z0-9@\[\]\(\)\+\-\=\#\:\.\$\%\/\\\{\}]{3,}", query)
    combined_candidates = list(set(words + regex_candidates))
    combined_candidates.sort(key=lambda x: (len(x), sum(1 for c in x if c in '()[]=#')), reverse=True)
    for s_cand in combined_candidates:
        s_cand = s_cand.strip('.,;:)?!\'"')
        if not s_cand: continue
        if '>>' in s_cand or '>' in s_cand or '<' in s_cand: continue
        if s_cand.isnumeric() and not ('[' in s_cand and ']' in s_cand) : continue
        try:
            mol = Chem.MolFromSmiles(s_cand, sanitize=True)
            if mol:
                num_atoms = mol.GetNumAtoms()
                if num_atoms >= 1:
                    if num_atoms <= 2 and s_cand.isalpha() and s_cand.lower() in [
                        'as', 'in', 'is', 'at', 'or', 'to', 'be', 'of', 'on', 'no', 'do', 'go',
                        'so', 'if', 'it', 'me', 'my', 'he', 'we', 'by', 'up', 'us', 'an', 'am', 'are'
                    ]:
                        if not any(c in s_cand for c in '()[]=#.-+@:/\\%{}') and len(s_cand) <=2 :
                             continue
                    if any(c in s_cand for c in '()[]=#.-+@:/\\%{}') or num_atoms > 2 or len(s_cand) > 3:
                        print(f"[EXTRACT_SINGLE_COMPOUND_SMILES] Validated candidate: {s_cand} (Atoms: {num_atoms})")
                        return s_cand
        except Exception: pass
    print(f"[EXTRACT_SINGLE_COMPOUND_SMILES] No suitable compound SMILES found in query: '{query[:50]}...'")
    return None

# --- Core analysis functions (handle_full_info, handle_compound_full_info, handle_followup_question - UNCHANGED) ---
# These functions (handle_full_info, handle_compound_full_info, handle_followup_question, format_value, create_response)
# are assumed to be correct and are kept as they were in your provided code.
# For brevity, I am omitting their full code here, but they should remain in your actual file.
# ... (Your full implementations of handle_full_info, handle_compound_full_info, handle_followup_question, format_value, create_response)
def handle_full_info(query_text_for_llm_summary, reaction_smiles_clean, original_compound_name=None, callbacks=None):
    tools = get_tools_for_core_logic() # Critical: make `tools` available

    print(f"\n--- [HANDLE_FULL_INFO_ENTRY for '{reaction_smiles_clean}'] ---")
    print(f"Query text for LLM summary: '{query_text_for_llm_summary[:100]}...'")
    if not reaction_smiles_clean or ">>" not in reaction_smiles_clean:
        print(f"[HANDLE_FULL_INFO_ERROR] Invalid or missing reaction_smiles_clean: '{reaction_smiles_clean}'")
        return {'visualization_path': None, 'analysis': f"Error: Invalid reaction SMILES provided for analysis: '{reaction_smiles_clean}'", 'analysis_context': "invalid_smiles_input", 'processed_smiles_for_tools': reaction_smiles_clean}

    if reaction_smiles_clean in reaction_cache and 'full_info' in reaction_cache[reaction_smiles_clean]:
        cached_data = reaction_cache[reaction_smiles_clean]['full_info']
        if isinstance(cached_data, dict) and \
           'analysis' in cached_data and \
           not cached_data.get('analysis_context', '').endswith(("_exception", "_error")):
            print(f"Using CACHED valid full_info data for reaction: {reaction_smiles_clean}")
            if 'processed_smiles_for_tools' not in cached_data:
                 cached_data['processed_smiles_for_tools'] = reaction_smiles_clean
            return cached_data
        else:
            print(f"Cached 'full_info' for {reaction_smiles_clean} found but is an error placeholder or invalid. Regenerating.")

    reaction_cache.setdefault(reaction_smiles_clean, {})
    full_info_results = {}
    tool_dict = {tool.name.lower(): tool for tool in tools}

    try:
        # Visualization
        visualizer_tool = tool_dict.get("chemvisualizer")
        if visualizer_tool:
            try:
                visualization_path = visualizer_tool.run(reaction_smiles_clean)
                if visualization_path and not str(visualization_path).lower().startswith('error') and str(visualization_path).endswith(".png"):
                    full_info_results['Visualization'] = visualization_path
                    reaction_cache[reaction_smiles_clean]['visualization_path'] = visualization_path
                else:
                    full_info_results['Visualization'] = f"Visualization tool message: {visualization_path}"
                    reaction_cache[reaction_smiles_clean]['visualization_path'] = None
            except Exception as e:
                full_info_results['Visualization'] = f"Error visualizing reaction: {str(e)}"
                reaction_cache[reaction_smiles_clean]['visualization_path'] = None
        else:
            full_info_results['Visualization'] = "ChemVisualizer tool not found"
            reaction_cache[reaction_smiles_clean]['visualization_path'] = None

        # Other tools: SMILES2Name, FuncGroups, BondChangeAnalyzer
        for tool_name_lower, data_key, cache_key_for_full_output in [
            ("smiles2name", "Names", "name_info"),
            ("funcgroups", "Functional Groups", "fg_info"),
            ("bondchangeanalyzer", "Bond Changes", "bond_info")
        ]:
            tool_instance = tool_dict.get(tool_name_lower)
            if tool_instance:
                try:
                    tool_result_full = tool_instance.run(reaction_smiles_clean)
                    if isinstance(tool_result_full, str):
                        display_result_for_llm_prompt = tool_result_full[:300] + ("..." if len(tool_result_full) > 300 else "")
                    elif isinstance(tool_result_full, dict) and 'Final Answer' in tool_result_full:
                        display_result_for_llm_prompt = str(tool_result_full['Final Answer'])[:300] + "..."
                    else:
                        display_result_for_llm_prompt = str(tool_result_full)[:300] + "..."

                    full_info_results[data_key] = display_result_for_llm_prompt
                    reaction_cache[reaction_smiles_clean][cache_key_for_full_output] = tool_result_full
                except Exception as e:
                    err_msg = f"Error running {tool_name_lower}: {str(e)}"
                    full_info_results[data_key] = err_msg
                    reaction_cache[reaction_smiles_clean][cache_key_for_full_output] = err_msg
            else:
                msg = f"{tool_name_lower.capitalize()} tool not found"
                full_info_results[data_key] = msg
                reaction_cache[reaction_smiles_clean][cache_key_for_full_output] = msg

        # Reaction Classifier (using reaction_classifier_core_logic instance)
        if reaction_classifier_core_logic:
            try:
                classifier_result_raw = reaction_classifier_core_logic._run(reaction_smiles_clean)
                if isinstance(classifier_result_raw, str):
                    summary_match = re.search(r'## Summary\n(.*?)(?=\n##|$)', classifier_result_raw, re.DOTALL | re.IGNORECASE)
                    classifier_summary = summary_match.group(1).strip() if summary_match else (classifier_result_raw.splitlines()[0] if classifier_result_raw.splitlines() else "No summary found")
                    full_info_results['Reaction Classification'] = classifier_summary[:300] + "..." if len(classifier_summary) > 300 else classifier_summary
                    reaction_cache[reaction_smiles_clean]['classification_info'] = classifier_summary
                else:
                    full_info_results['Reaction Classification'] = "Classifier result not a string"
                    reaction_cache[reaction_smiles_clean]['classification_info'] = "Classifier result not a string"
            except Exception as e:
                full_info_results['Reaction Classification'] = f"Error classifying: {str(e)}"
                reaction_cache[reaction_smiles_clean]['classification_info'] = f"Error classifying: {str(e)}"
        else:
            full_info_results['Reaction Classification'] = "ReactionClassifier (core logic) not available"
            reaction_cache[reaction_smiles_clean]['classification_info'] = "ReactionClassifier (core logic) not available"

        # Dataset Query
        dataset_data = query_reaction_dataset(reaction_smiles_clean)
        procedure_details, rxn_time, temperature, yield_val_from_dataset, solvents, agents_catalysts = None, None, None, None, None, None
        if dataset_data:
            procedure_details = dataset_data.get('procedure_details')
            rxn_time = dataset_data.get('rxn_time')
            temperature = dataset_data.get('temperature')
            yield_val_from_dataset = dataset_data.get('yield_000')
            solvents = dataset_data.get('solvents_list')
            agents_catalysts = dataset_data.get('agents_list')

        reaction_cache[reaction_smiles_clean].update({
            'procedure_details': procedure_details,
            'reaction_time': rxn_time,
            'temperature': temperature,
            'yield': yield_val_from_dataset,
            'solvents': solvents,
            'agents_catalysts': agents_catalysts
        })

        # --- LLM Summary Generation ---
        final_prompt_parts = [
            f"You are a chemistry expert. Synthesize this reaction analysis into a clear explanation:",
            f"Reaction SMILES (Processed): {reaction_smiles_clean}",
            f"NAMES: {full_info_results.get('Names', 'Not available')}",
            f"BOND CHANGES: {full_info_results.get('Bond Changes', 'Not available')}",
            f"FUNCTIONAL GROUPS: {full_info_results.get('Functional Groups', 'Not available')}",
            f"REACTION TYPE (from classifier): {full_info_results.get('Reaction Classification', 'Not available')}"
        ]
        if procedure_details: final_prompt_parts.append(f"PROCEDURE DETAILS: {procedure_details[:500] + '...' if procedure_details and len(procedure_details) > 500 else procedure_details}")

        conditions_parts = []
        if temperature and str(temperature).strip() and str(temperature).lower() != 'nan': conditions_parts.append(f"Temperature: {temperature}")
        if rxn_time and str(rxn_time).strip() and str(rxn_time).lower() != 'nan': conditions_parts.append(f"Time: {rxn_time}")
        if yield_val_from_dataset and str(yield_val_from_dataset).strip() and str(yield_val_from_dataset).lower() != 'nan': conditions_parts.append(f"Yield: {yield_val_from_dataset}%")
        if conditions_parts: final_prompt_parts.append(f"EXPERIMENTAL CONDITIONS: {', '.join(conditions_parts)}")

        materials_parts = []
        if solvents and isinstance(solvents, list) and any(str(s).strip() and str(s).strip().lower() != 'nan' for s in solvents):
            materials_parts.append(f"Solvents: {', '.join(filter(None, [str(s) for s in solvents if str(s).strip() and str(s).strip().lower() != 'nan']))}")
        if agents_catalysts and isinstance(agents_catalysts, list) and any(str(a).strip() and str(a).strip().lower() != 'nan' for a in agents_catalysts):
            materials_parts.append(f"Catalysts/Reagents: {', '.join(filter(None, [str(a) for a in agents_catalysts if str(a).strip() and str(a).strip().lower() != 'nan']))}")
        if materials_parts: final_prompt_parts.append(f"KEY MATERIALS: {'; '.join(materials_parts)}")

        final_prompt_parts.append(
            "\nProvide a thorough, well-structured explanation covering the following aspects if information is available:"
            "\n1. Begins with a high-level summary of what type of reaction this is (e.g., 'This is an esterification reaction...')."
            "\n2. Explains what happens at the molecular level (key bonds broken/formed, atom transfers if evident)."
            "\n3. Discusses the functional group transformations (e.g., 'an alcohol is converted to an ether')."
            "\n4. Includes specific experimental conditions (temperature, time, yield) and key materials (solvents, catalysts/reagents) if provided."
            "\n5. Procedure summary (if known): Briefly describe the experimental steps or context."
            "\n6. Mentions common applications or importance of this reaction type if readily inferable from the classification or functional groups and widely known (be cautious not to overstate or invent)."
            "\nPresent the information clearly and logically for a chemist. Focus ONLY on the provided data. Do not invent information not present in the input."
        )
        final_prompt_for_llm = "\n\n".join(final_prompt_parts)

        analysis_text_summary = f"LLM Summarizer (direct OpenAI) not available or failed. Raw data: {str(full_info_results)}" # Default
        try:
            if os.environ.get("OPENAI_API_KEY"): # Check if API key is set (by api_config.py)
                client = openai.OpenAI() # Initialize direct OpenAI client
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a chemistry expert providing a detailed reaction analysis summary."},
                        {"role": "user", "content": final_prompt_for_llm}
                    ],
                    temperature=0,
                    max_tokens=2000
                )
                analysis_text_summary = response.choices[0].message.content.strip()
                print("[HANDLE_FULL_INFO] LLM summary generated using direct OpenAI call.")
            else:
                print("Warning: OPENAI_API_KEY not found for direct OpenAI call in handle_full_info. LLM summary will be basic.")
        except openai.APIError as api_e:
            print(f"OpenAI API Error during summarization in handle_full_info: {api_e}")
            analysis_text_summary = f"OpenAI API Error during summary generation. Raw data: {str(full_info_results)}"
        except Exception as llm_e:
            print(f"Error during LLM summarization in handle_full_info: {llm_e}")
            analysis_text_summary = f"Error generating LLM summary. Raw data: {str(full_info_results)}"

        structured_result_for_full_info_cache = {
            'visualization_path': reaction_cache[reaction_smiles_clean].get('visualization_path'),
            'analysis': analysis_text_summary,
            'reaction_classification_summary': reaction_cache[reaction_smiles_clean].get('classification_info'),
            'procedure_details': procedure_details,
            'reaction_time': rxn_time,
            'temperature': temperature,
            'yield': yield_val_from_dataset,
            'solvents': solvents or None,
            'agents_catalysts': agents_catalysts or None,
            'analysis_context': "full_direct_openai_summary_generated",
            'processed_smiles_for_tools': reaction_smiles_clean
        }

        reaction_cache[reaction_smiles_clean]['full_info'] = structured_result_for_full_info_cache
        print(f"\n--- [CACHE_STATE_AFTER_FULL_INFO_STORE for '{reaction_smiles_clean}'] ---")
        return structured_result_for_full_info_cache

    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"CRITICAL ERROR in handle_full_info for {reaction_smiles_clean}: {e}\n{tb_str}")
        error_result = {
            'visualization_path': None,
            'analysis': f"An internal error occurred during the full analysis of '{reaction_smiles_clean}'. Details: {str(e)}",
            'analysis_context': "full_analysis_exception",
            'processed_smiles_for_tools': reaction_smiles_clean
        }
        reaction_cache.setdefault(reaction_smiles_clean, {})['full_info'] = error_result
        return error_result

def handle_compound_full_info(query_text_for_summary_context, compound_smiles, original_compound_name_context=None, callbacks=None):
    global _current_moi_context
    tools = get_tools_for_core_logic() # Wrappers for S2N, FG, Visualizer
    print(f"\n--- [HANDLE_COMPOUND_FULL_INFO for '{compound_smiles}'] ---")

    if not compound_smiles:
        return {'visualization_path': None, 'analysis': "Error: No valid compound SMILES provided.",
                'analysis_context': "invalid_compound_smiles", 'processed_smiles_for_tools': None}

    cache_key_full_compound_info = f"{compound_smiles}_full_with_disconnections"
    if compound_smiles in compound_cache and cache_key_full_compound_info in compound_cache[compound_smiles]:
        cached_data = compound_cache[compound_smiles][cache_key_full_compound_info]
        if isinstance(cached_data, dict) and 'analysis' in cached_data and \
           not cached_data.get('analysis_context', '').endswith(("_error", "_exception")):
            print(f"Using CACHED full_compound_info (with disconnections) for: {compound_smiles}")
            return cached_data
        else:
            print(f"Cached 'full_compound_info' for {compound_smiles} (with disconnections) found but invalid/error. Regenerating.")

    compound_cache.setdefault(compound_smiles, {})
    info_results = {}
    tool_dict = {tool.name.lower(): tool for tool in tools}

    # --- 1. Visualization ---
    visualizer_tool = tool_dict.get("chemvisualizer")
    viz_path = None
    if visualizer_tool:
        try:
            viz_path_result = visualizer_tool.run(compound_smiles)
            if viz_path_result and not str(viz_path_result).lower().startswith('error') and str(viz_path_result).endswith(".png"):
                viz_path = viz_path_result
                info_results['Visualization Path'] = viz_path
            else:
                info_results['Visualization Info'] = f"Tool message: {viz_path_result}"
        except Exception as e:
            info_results['Visualization Info'] = f"Error visualizing: {str(e)}"
    else:
        info_results['Visualization Info'] = "ChemVisualizer tool not found"
    compound_cache[compound_smiles]['visualization_path'] = viz_path

    # --- 2. SMILES to Name ---
    s2n_tool = tool_dict.get("smiles2name")
    tool_derived_name_for_moi = None
    if s2n_tool:
        try:
            name_result_str = s2n_tool.run(compound_smiles)
            info_results['Name'] = name_result_str # For the analysis text display
            compound_cache[compound_smiles]['name_info_raw'] = name_result_str

            # Try to parse a clean name from the tool's output
            iupac_match = re.search(r"(?:IUPAC Name|Name):\s*(.+)", name_result_str, re.IGNORECASE)
            temp_name_from_tool = None
            if iupac_match:
                temp_name_from_tool = iupac_match.group(1).strip()
            elif name_result_str and all(err_msg.lower() not in name_result_str.lower() for err_msg in ["error", "not found", "could not convert"]):
                # If no specific "Name:" prefix, but tool didn't error, use the whole string
                temp_name_from_tool = name_result_str.strip()

            # Check if the derived name is valid, not just the SMILES, and not empty
            if temp_name_from_tool and \
               temp_name_from_tool.lower() != compound_smiles.lower() and \
               len(temp_name_from_tool) > 0:
                tool_derived_name_for_moi = temp_name_from_tool

        except Exception as e:
            info_results['Name'] = f"Error getting name: {str(e)}"
            print(f"[handle_compound_full_info] Error running smiles2name: {e}")
    else:
        info_results['Name'] = "SMILES2Name tool not found"

    if tool_derived_name_for_moi:
        _current_moi_context["name"] = tool_derived_name_for_moi
        print(f"[handle_compound_full_info] Updated MOI name from smiles2name tool: '{tool_derived_name_for_moi}' for SMILES: '{compound_smiles}'")
    elif original_compound_name_context and original_compound_name_context.lower() != compound_smiles.lower():
        # If tool didn't give a name, but we had a context name (e.g., user typed "Aspirin"), keep that.
        # enhanced_query would have set this initially.
        # No change needed here as _current_moi_context["name"] would already be original_compound_name_context
        print(f"[handle_compound_full_info] MOI name remains as context-provided: '{original_compound_name_context}' (tool did not provide a better one or failed)")
    else:
        # If no tool-derived name and no better context name, it might be just the SMILES.
        # This ensures _current_moi_context["name"] is at least the SMILES if it was somehow None.
        _current_moi_context["name"] = compound_smiles
        print(f"[handle_compound_full_info] MOI name set/remains as SMILES: '{compound_smiles}' (no other specific name found/provided)")

    # Ensure the SMILES is also correctly set in MOI context (though enhanced_query should have done this)
    _current_moi_context["smiles"] = compound_smiles

    # --- 3. Functional Groups ---
    fg_tool = tool_dict.get("funcgroups")
    functional_groups_list_for_disconnections: List[str] = []
    if fg_tool:
        try:
            fg_result_str = fg_tool.run(compound_smiles)
            info_results['Functional Groups String'] = fg_result_str
            if "Functional Group Analysis:" in fg_result_str:
                try:
                    dict_str = fg_result_str.split("Functional Group Analysis:", 1)[1].strip()
                    fg_dict_parsed = ast.literal_eval(dict_str)
                    if isinstance(fg_dict_parsed, dict) and "functional_groups" in fg_dict_parsed:
                        functional_groups_list_for_disconnections = fg_dict_parsed["functional_groups"]
                        if functional_groups_list_for_disconnections:
                             info_results['Functional Groups'] = ", ".join(functional_groups_list_for_disconnections)
                        else:
                             info_results['Functional Groups'] = "None identified"
                    else:
                        info_results['Functional Groups'] = "Could not parse FG list from tool output."
                except:
                    info_results['Functional Groups'] = "Error parsing functional groups from tool output."
                    print(f"[handle_compound_full_info] Error parsing FGs from: {fg_result_str}")
            else:
                 info_results['Functional Groups'] = fg_result_str
            compound_cache[compound_smiles]['fg_info_raw'] = fg_result_str
            compound_cache[compound_smiles]['fg_list_parsed'] = functional_groups_list_for_disconnections
        except Exception as e:
            info_results['Functional Groups'] = f"Error getting functional groups: {str(e)}"
    else:
        info_results['Functional Groups'] = "FuncGroups tool not found"

    # --- 4. Suggest Disconnections ---
    raw_disconnection_suggestions = ""
    if compound_smiles:
        print(f"[handle_compound_full_info] Calling suggest_disconnections for {compound_smiles}")
        try:
            disconnection_result_full_string = ag_tools.suggest_disconnections(compound_smiles)
            raw_disconnection_suggestions = disconnection_result_full_string
            if "Suggested Disconnections (from LLM):" in disconnection_result_full_string:
                disconnection_info_for_prompt = disconnection_result_full_string.split("Suggested Disconnections (from LLM):", 1)[1].strip()
            elif "Error" in disconnection_result_full_string:
                disconnection_info_for_prompt = disconnection_result_full_string
            else:
                disconnection_info_for_prompt = "Disconnection suggestions not clearly parsed."
            info_results['Disconnections'] = disconnection_info_for_prompt[:700] + ("..." if len(disconnection_info_for_prompt) > 700 else "")
        except Exception as e_disc:
            info_results['Disconnections'] = f"Error suggesting disconnections: {str(e_disc)}"
            raw_disconnection_suggestions = f"Error suggesting disconnections: {str(e_disc)}"
    compound_cache[compound_smiles]['disconnection_info_raw'] = raw_disconnection_suggestions

    # --- 5. LLM Summary Generation ---
    llm_prompt_parts = [
        f"Provide a comprehensive overview of the compound with SMILES: {compound_smiles}.",
        f"User query context (ignore if not relevant to summarization): '{query_text_for_summary_context}'",
        "Information gathered from tools:",
        f"- Names: {info_results.get('Name', 'N/A')}",
        f"- Identified Functional Groups: {info_results.get('Functional Groups', 'N/A')}",
        f"- Potential Retrosynthetic Disconnections (based on FGs): {info_results.get('Disconnections', 'N/A')}"
    ]
    if viz_path:
        llm_prompt_parts.append(f"- A 2D structure image has been generated and is available.")
    else:
        llm_prompt_parts.append(f"- Structure visualization: {info_results.get('Visualization Info', 'Not attempted or failed.')}")
    llm_prompt_parts.extend([
        "\nBased ONLY on the information gathered from the tools above, please provide a well-structured summary including:",
        "1. Chemical names (IUPAC and common, if available from tools).",
        "2. A list of identified functional groups.",
        "3. A brief interpretation of the compound based on these functional groups (e.g., potential reactivity, properties).",
        "4. A summary of key retrosynthetic disconnection ideas suggested, if available.",
        "5. Mention if a visualization is available.",
        "Present this as a chemist would expect, clearly and concisely. Focus strictly on the provided tool outputs. Do not add external knowledge or invent details."
    ])
    final_llm_prompt = "\n".join(llm_prompt_parts)
    final_analysis_text = f"LLM Summarizer (direct OpenAI) failed. Raw data available."
    raw_data_for_fallback = (
        f"Raw data:\n"
        f"Name Info: {info_results.get('Name', 'N/A')}\n"
        f"Functional Groups: {info_results.get('Functional Groups', 'N/A')}\n"
        f"Disconnections: {info_results.get('Disconnections', 'N/A')}"
    )
    final_analysis_text += "\n" + raw_data_for_fallback
    try:
        if os.environ.get("OPENAI_API_KEY"):
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a chemistry expert. Your task is to synthesize the provided chemical information about a compound into a comprehensive, well-structured summary. Focus only on the data given."},
                    {"role": "user", "content": final_llm_prompt}
                ],
                temperature=0,
                max_tokens=2000
            )
            final_analysis_text = response.choices[0].message.content.strip()
            print("[HANDLE_COMPOUND_FULL_INFO] LLM summary (with disconnections) generated using direct OpenAI call.")
        else:
            print("Warning: OPENAI_API_KEY not found for direct OpenAI call in handle_compound_full_info. LLM summary will be basic (raw data).")
    except openai.APIError as api_e:
        print(f"OpenAI API Error during summarization in handle_compound_full_info: {api_e}")
        final_analysis_text = f"OpenAI API Error during compound summary.\n{raw_data_for_fallback}"
    except Exception as e:
        print(f"LLM summary generation for compound failed in handle_compound_full_info: {e}")
        final_analysis_text = f"Error generating LLM summary for compound.\n{raw_data_for_fallback}"
    result_struct = {
        'visualization_path': viz_path,
        'analysis': final_analysis_text, 
        'name_info_raw': compound_cache[compound_smiles].get('name_info_raw'), 
        'fg_info_raw': compound_cache[compound_smiles].get('fg_info_raw'),
        'fg_list_parsed': compound_cache[compound_smiles].get('fg_list_parsed'),
        'disconnection_info_raw': compound_cache[compound_smiles].get('disconnection_info_raw'),
        'analysis_context': "compound_openai_summary_with_disconnections",
        'processed_smiles_for_tools': compound_smiles
    }
    compound_cache[compound_smiles][cache_key_full_compound_info] = result_struct 
    print(f"--- [END HANDLE_COMPOUND_FULL_INFO for '{compound_smiles}'] ---")
    return result_struct

def handle_followup_question(query_text, reaction_smiles, original_compound_name=None, callbacks=None):
    cached_reaction_data = reaction_cache.get(reaction_smiles, {})
    full_info = cached_reaction_data.get('full_info', {})
    llm_summary_analysis = full_info.get('analysis', '')
    property_map = {
        'solvent': {'keywords': ['solvent', 'solution', 'medium', 'dissolve'], 'cache_key': 'solvents', 'text_patterns': [r'solvents?:\s*([^\.\n]+)', r'carried out in\s*([^\.\n]+)', r'using\s*([^\(\n]+)\s*as solvent']},
        'temperature': {'keywords': ['temperature', 'temp', '°c', '°f', 'kelvin', 'heat', 'cool'], 'cache_key': 'temperature', 'text_patterns': [r'temperature:\s*([^\.\n]+)', r'temp\.?:\s*([^\.\n]+)', r'at\s*([\d\.\-]+)\s*°', r'heated to\s*([^\.\n]+)']},
        'yield': {'keywords': ['yield', '% yield', 'percentage', 'efficiency', 'obtained'], 'cache_key': 'yield', 'text_patterns': [r'yield:\s*([^\.\n]+%?)', r'([\d\.\s]+%)\s*yield', r'obtained in\s*([^\.\n]+)\s*yield']},
        'time': {'keywords': ['time', 'duration', 'hour', 'minute', 'day'], 'cache_key': 'reaction_time', 'text_patterns': [r'time:\s*([^\.\n]+)', r'duration:\s*([^\.\n]+)', r'for\s*([\d\.]+\s*(?:h|hr|hrs|hours|minutes|mins|days?))', r'stirred for\s*([^\.\n]+)']},
        'catalyst': {'keywords': ['catalyst', 'reagent', 'agent', 'promoter', 'additive', 'initiator'], 'cache_key': 'agents_catalysts', 'text_patterns': [r'catalysts?:\s*([^\.\n]+)', r'reagents?:\s*([^\.\n]+)', r'using\s*([^\(\n]+)\s*as (?:catalyst|reagent)', r'(?:catalyzed|initiated|promoted) by\s*([^\.\n]+)', r'in the presence of\s*([^\.\n]+)']},
        'pressure': {'keywords': ['pressure', 'psi', 'bar', 'atm', 'atmosphere'], 'cache_key': 'pressure', 'text_patterns': [r'pressure:\s*([^\.\n]+)', r'under\s*([^\.\n]+)\s*pressure', r'at\s*([\d\.]+)\s*(psi|bar|atm)']},
        'ph': {'keywords': ['ph', 'acidic', 'basic', 'neutral'], 'cache_key': 'ph', 'text_patterns': [r'ph:\s*([^\.\n]+)', r'at\s*ph\s*([\d\.]+)', r'under\s*(acidic|basic|neutral)\s*conditions']},
        'procedure': {'keywords': ['procedure', 'protocol', 'steps', 'method', 'preparation', 'synthesis steps'], 'cache_key': 'procedure_details', 'text_patterns': [r'procedure details:?\s*(.*?)(?:\n\n|\Z)', r'experimental steps:?\s*(.*?)(?:\n\n|\Z)', r'method:?\s*(.*?)(?:\n\n|\Z)']},
        'classification': {'keywords': ['classification', 'type of reaction', 'reaction type', 'class'], 'cache_key': 'reaction_classification_summary', 'text_patterns': [r'reaction type \(from classifier\):\s*([^\n]+)', r'classification:\s*([^\n]+)', r'this is an?\s*([\w\s]+reaction)']},
        'name': {'keywords': ['name of reaction', 'reaction name'], 'cache_key': 'name_info', 'text_patterns': [ r'names:\s*([^\n]+)' ]}
    }
    query_lower = query_text.lower()
    for prop_name, prop_details in property_map.items():
        if any(keyword in query_lower for keyword in prop_details['keywords']):
            cache_key_for_prop = prop_details.get('cache_key')
            if cache_key_for_prop and cache_key_for_prop in full_info and full_info[cache_key_for_prop] is not None:
                value = format_value(full_info[cache_key_for_prop])
                if value and value.lower() != "not specified":
                    return create_response(prop_name, value, reaction_smiles)
            if llm_summary_analysis:
                for pattern in prop_details['text_patterns']:
                    flags = re.IGNORECASE | re.DOTALL if prop_name == 'procedure' else re.IGNORECASE
                    match = re.search(pattern, llm_summary_analysis, flags)
                    if match:
                        extracted_value = (match.group(1) if len(match.groups()) > 0 and match.group(1) else match.group(0)).strip(" .,:")
                        if extracted_value and extracted_value.lower() not in ["not available", "n/a", "not specified", "none"]:
                            return create_response(prop_name, extracted_value, reaction_smiles)
            return {
                "visualization_path": None,
                "analysis": f"Specific information about '{prop_name}' was not found in the cached reaction data or its summary. You can ask the chatbot for a new search if this is a general query.",
                "analysis_context": f"followup_{prop_name}_not_readily_found",
                "processed_smiles_for_tools": reaction_smiles
            }
    print(f"[FOLLOWUP_UNMATCHED] Query '{query_text}' did not match specific property keywords for cached data lookup.")
    return { "visualization_path": None, "analysis": None, "analysis_context": "followup_property_unmatched_in_cache", "processed_smiles_for_tools": reaction_smiles }

def format_value(value):
    if isinstance(value, list):
        valid_items = [str(v) for v in value if v is not None and str(v).strip().lower() not in ['nan', 'none', '']]
        return ", ".join(valid_items) if valid_items else "not specified"
    if value is None or (isinstance(value, str) and value.strip().lower() in ['nan', 'none', '']):
        return "not specified"
    return str(value).strip()

def create_response(prop, value, reaction_smiles):
    prop_display_name = prop.replace('_', ' ').capitalize()
    return {
        "visualization_path": None,
        "analysis": f"Regarding the {prop_display_name} for reaction {reaction_smiles}: {value}.",
        "analysis_context": f"followup_{prop}_direct_answer",
        "processed_smiles_for_tools": reaction_smiles
    }

# --- Autogen Agent Definitions and Execution (Tool Agent - UNCHANGED) ---
# Your get_tool_agents() and run_autogen_tool_agent_query() seem largely fine for their purpose.
# ... (Your full implementations of TOOL_AGENT_SYSTEM_MESSAGE, get_tool_agents, run_autogen_tool_agent_query)
TOOL_AGENT_SYSTEM_MESSAGE = """You are Chem Copilot, an expert chemistry assistant. You are tasked with executing a specific chemical analysis tool based on the user's query.
You have access to the following tools:
- `get_functional_groups(smiles_or_reaction_smiles: str)`: Identifies functional groups.
- `convert_name_to_smiles(chemical_name: str)`: Converts name to SMILES.
- `convert_smiles_to_name(smiles_string: str)`: Converts SMILES to name.
- `analyze_reaction_bond_changes(reaction_smiles: str)`: Analyzes bond changes in a reaction.
- `visualize_chemical_structure(smiles_or_reaction_smiles: str)`: Generates a visualization.
- `classify_reaction_and_get_details(reaction_smiles: str)`: Classifies reaction and gets details.
- `query_specific_property_for_reaction(reaction_smiles: str, property_to_query: str)`: Queries a specific property for a reaction.
- `suggest_disconnections(compound_smiles: str)`: Suggests retrosynthetic disconnections for a compound.
- `get_full_chemical_report(chemical_identifier: str)`: Provides a comprehensive analysis report for a chemical (name, SMILES, or CAS).

Your goal is to:
1. Understand the user's request.
2. Select THE MOST APPROPRIATE tool. If multiple tools seem relevant for a general query, pick the one that seems primary or state that the query is too broad for a single tool call.
3. Execute the tool with the correct input extracted from the query.
4. Return the raw output from the tool directly as your final answer. Do not add any conversational fluff or explanations unless the tool's output itself is an explanation.
5. If the query is a general knowledge question (e.g., "What is an element?", "Explain SN2 reactions"), or if no tool is appropriate, you MUST respond with:
   "I can only perform specific chemical analyses using my tools if you provide a SMILES string or a chemical name for tool-based processing. I cannot answer general knowledge questions. Please provide a specific chemical entity or task for my tools.TERMINATE"
Your response should be ONLY the tool's output or the refusal message.
"""
_assistant_tool_agent = None
_user_proxy_tool_agent = None

def get_tool_agents():
    global _assistant_tool_agent, _user_proxy_tool_agent, llm_config_chatbot_agent # Using chatbot's base llm config

    if _assistant_tool_agent is None:
        tool_functions_for_tool_agent = [
            ag_tools.get_functional_groups, ag_tools.convert_name_to_smiles,
            ag_tools.suggest_disconnections, ag_tools.convert_smiles_to_name,
            ag_tools.analyze_reaction_bond_changes, ag_tools.visualize_chemical_structure,
            ag_tools.classify_reaction_and_get_details, ag_tools.query_specific_property_for_reaction,
            ag_tools.get_full_chemical_report
        ]
        
        assistant_llm_tools_definition = [] # For OpenAI 'tools' format
        for func in tool_functions_for_tool_agent:
            func_name = func.__name__
            param_name = "chemical_identifier" # Default, will be overridden
            param_desc = "Input for the tool."

            if func_name == "get_full_chemical_report":
                param_name = "chemical_identifier"
                param_desc = "The name, SMILES string, or CAS number of the chemical."
            elif func_name == "convert_name_to_smiles":
                param_name = "chemical_name"
                param_desc = "The chemical name."
            elif func_name == "convert_smiles_to_name":
                param_name = "smiles_string"
                param_desc = "The SMILES string."
            elif func_name == "suggest_disconnections":
                param_name = "smiles" # Corrected to match your tool function
                param_desc = "The SMILES string of the molecule."
            elif func_name == "query_specific_property_for_reaction":
                # Special handling for multi-parameter tool
                assistant_llm_tools_definition.append({
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "description": func.__doc__ or f"Executes the {func_name} tool.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "reaction_smiles": {"type": "string", "description": "The Reaction SMILES string."},
                                "property_to_query": {"type": "string", "description": "The specific property like 'yield' or 'temperature'."}
                            },
                            "required": ["reaction_smiles", "property_to_query"]
                        }
                    }
                })
                continue # Skip default single-parameter schema for this tool
            else: # Default for get_functional_groups, visualize_chemical_structure, etc.
                param_name = "smiles_or_reaction_smiles"
                param_desc = "The SMILES string of the compound or reaction."

            assistant_llm_tools_definition.append({
                "type": "function",
                "function": {
                    "name": func_name,
                    "description": (func.__doc__ or f"Executes the {func_name} tool.").splitlines()[0].strip(),
                    "parameters": {
                        "type": "object",
                        "properties": {param_name: {"type": "string", "description": param_desc}},
                        "required": [param_name]
                    }
                }
            })

        # Create the llm_config FOR THE ASSISTANT AGENT with tools included
        tool_agent_llm_config = llm_config_chatbot_agent.copy() # Start with a base config
        if assistant_llm_tools_definition:
            tool_agent_llm_config["tools"] = assistant_llm_tools_definition
        
        _assistant_tool_agent = autogen.AssistantAgent(
            name="ChemistryToolAgent_v3", # Consider versioning names when making changes
            llm_config=tool_agent_llm_config, # <<< PASS THE CONFIG WITH TOOLS HERE
            system_message=TOOL_AGENT_SYSTEM_MESSAGE,
            is_termination_msg=lambda x: isinstance(x.get("content"), str) and x.get("content", "").rstrip().endswith("TERMINATE") or \
                                        (isinstance(x.get("content"), str) and "I can only perform specific chemical analyses" in x.get("content","") and not x.get("tool_calls"))
        )
        
        _user_proxy_tool_agent = autogen.UserProxyAgent(
            name="UserProxyToolExecutor_v3",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2, 
            code_execution_config=False, 
            function_map={func.__name__: func for func in tool_functions_for_tool_agent}, # UserProxy needs this to execute
            is_termination_msg=lambda x: (isinstance(x.get("content"), str) and x.get("content", "").rstrip().endswith("TERMINATE")) or \
                                        (isinstance(x.get("content"), str) and "I can only perform specific chemical analyses" in x.get("content",""))
        )
        print(f"[ToolAgent Init] Tools configured for Assistant: {len(tool_agent_llm_config.get('tools', []))}")
    return _assistant_tool_agent, _user_proxy_tool_agent

def run_autogen_tool_agent_query(user_input: str, callbacks=None):
    print(f"[AUTOGEN_TOOL_AGENT_QUERY] Processing: '{user_input[:100]}...'")
    assistant, user_proxy = get_tool_agents()
    user_proxy.reset()
    assistant.reset() # Also reset assistant for clean state
    ai_response_text = "Tool agent did not provide a clear answer (default)."
    try:
        user_proxy.initiate_chat(
            recipient=assistant,
            message=user_input,
            max_turns=3,
            request_timeout=llm_config_chatbot_agent.get("timeout", 60) + 10, # Slightly more for overall chat
        )
        messages_from_assistant_to_proxy = user_proxy.chat_messages.get(assistant, [])
        print(f"[DEBUG run_autogen_tool_agent_query] Messages from Assistant to Proxy: {messages_from_assistant_to_proxy}")
        if messages_from_assistant_to_proxy:
            last_msg_obj = messages_from_assistant_to_proxy[-1]
            if last_msg_obj.get("role") == "assistant" and last_msg_obj.get("content"):
                ai_response_text = last_msg_obj["content"].strip()
            elif last_msg_obj.get("content"):
                 ai_response_text = last_msg_obj.get("content").strip()
        if ai_response_text.upper() == "TERMINATE" or ai_response_text == "Tool agent did not provide a clear answer (default).":
            if messages_from_assistant_to_proxy and len(messages_from_assistant_to_proxy) > 1:
                potential_reply = messages_from_assistant_to_proxy[-2].get("content", "").strip()
                if potential_reply and potential_reply.upper() != "TERMINATE":
                    ai_response_text = potential_reply
            elif "I can only perform specific chemical analyses" in ai_response_text:
                pass
            elif ai_response_text == "Tool agent did not provide a clear answer (default).":
                print(f"[Warning] Tool agent interaction ended without a clear textual reply. Last message object: {messages_from_assistant_to_proxy[-1] if messages_from_assistant_to_proxy else 'None'}")
        if ai_response_text.upper().endswith("TERMINATE"):
            ai_response_text = ai_response_text[:-len("TERMINATE")].strip()
        if not ai_response_text.strip() and ai_response_text != "Tool agent did not provide a clear answer (default).":
            ai_response_text = "Tool agent processing complete."
        print(f"[DEBUG run_autogen_tool_agent_query] Final extracted response: '{ai_response_text}'")
        viz_path_agent = None
        if "static/autogen_visualizations/" in ai_response_text:
            match_viz = re.search(r"(static/autogen_visualizations/[\w\-\.\_]+\.png)", ai_response_text)
            if match_viz: viz_path_agent = match_viz.group(1)
        return {
            "visualization_path": viz_path_agent,
            "analysis": ai_response_text
        }
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Error in run_autogen_tool_agent_query: {e}\n{tb_str}")
        return {
            "visualization_path": None,
            "analysis": f"An error occurred in the tool agent: {str(e)}"
        }

# --- Chatbot Agent Definitions and Execution (MODIFIED FOR MOI CONTEXT) ---
_assistant_chatbot_agent = None
_user_proxy_chatbot_agent = None
_current_moi_context: Dict[str, Optional[str]] = {"name": None, "smiles": None} # Global MOI context
# _chatbot_history_store is not used; context comes from MOI in system message and current query

# Define which tools the chatbot can use
CHATBOT_TOOL_PY_FUNCTIONS = [
    ag_tools.get_functional_groups,
    ag_tools.convert_name_to_smiles, # If user gives a name in chat
    ag_tools.suggest_disconnections, # Add if you want chatbot to use this
    ag_tools.convert_smiles_to_name, # If MOI is SMILES, can get name
    ag_tools.visualize_chemical_structure, # Chatbot can visualize MOI
    ag_tools.get_full_chemical_report
]

def _update_chatbot_system_message_with_moi():
    """Updates the chatbot agent's system message with the current MOI."""
    global _assistant_chatbot_agent, _current_moi_context
    if not _assistant_chatbot_agent:
        print("[Chatbot ERROR] _assistant_chatbot_agent is None, cannot update system message.")
        return

    moi_name = _current_moi_context.get("name", "Not Set")
    moi_smiles = _current_moi_context.get("smiles", "Not Set")

    # Tool list for the system message
    available_tools_desc_chat = []
    for func in CHATBOT_TOOL_PY_FUNCTIONS:
        desc = (func.__doc__ or f"Executes {func.__name__}.").splitlines()[0].strip()
        # Simplistic description, enhance if needed
        param_info = CHATBOT_TOOL_PARAM_INFO.get(func.__name__)
        if param_info:
            param_desc_name = param_info["param_name"]
        else:
            # Fallback if not in CHATBOT_TOOL_PARAM_INFO (should ensure all are covered)
            param_desc_name = "input_string" 
            print(f"[WARN] No param_info for {func.__name__} in _update_chatbot_system_message_with_moi, using default.")
        available_tools_desc_chat.append(f"- `{func.__name__}({param_desc_name}: str)`: {desc}")
    tools_list_str_chat = "\n".join(available_tools_desc_chat) if available_tools_desc_chat else "No tools specified for direct call."


    system_message = f"""You are ChemCopilot, a specialized AI assistant for chemistry.
You MUST use the conversation context, especially the Current Molecule of Interest (MOI), to answer questions.

Current Molecule of Interest (MOI):
Name: {moi_name}
SMILES: {moi_smiles}

Available tools:
{tools_list_str_chat}

Your tasks:
1.  **Acknowledge MOI Context (Priming):**
    *   If the user's message is a specific priming request like "Let's discuss the molecule of interest: [NAME] with SMILES [SMILES]. Please acknowledge.", your response should be managed by the system directly. You will not see this exact query.
2.  **Answer questions about the MOI using tools:**
   *   If a question refers to 'it', 'that molecule', or clearly implies the current MOI:
        *   **If asked for its SMILES string:** Directly state "The SMILES for {moi_name} is {moi_smiles}. TERMINATE". Do NOT use any tools for this.
        *   For other properties (like functional groups), use the MOI's SMILES ('{moi_smiles}') for any relevant tool calls (e.g., `get_functional_groups`).
    *   If the MOI is not set or not relevant, and the user provides a new chemical entity in their query, use that for the tool call.
3.  **Tool Usage Rules:**
    *   When you decide to use a tool, make sure to provide the correct arguments. For example, for `get_functional_groups`, provide the SMILES string.
    *   If the MOI's SMILES is "{moi_smiles}" and the user asks "What are its functional groups?", you should aim to call `get_functional_groups` with `smiles_or_reaction_smiles="{moi_smiles}"`.
4.  **General Queries:**
    *   If a question is general chemistry knowledge NOT requiring a specific tool or the MOI, OR if no clear chemical entity is available for tool use, you MUST respond with:
        "I am a specialized chemistry assistant that uses tools to analyze specific chemical entities based on our conversation. I cannot answer general knowledge questions if they don't relate to a specific molecule we are discussing or a tool I have. Could you please clarify or provide a specific chemical entity? TERMINATE"
5.  **Response Format:** Be concise and helpful. After providing your answer (based on tool output, direct MOI info, or the refusal message), ALWAYS append " TERMINATE" to your response.
"""
    if _assistant_chatbot_agent: 
         _assistant_chatbot_agent.update_system_message(system_message)
         print(f"[Chatbot SystemMsg] Updated. MOI: {moi_name} ({moi_smiles}). Tools for chatbot: {len(CHATBOT_TOOL_PY_FUNCTIONS)}")
    else:
         print("[Chatbot SystemMsg WARN] Attempted to update system message but _assistant_chatbot_agent is still None.")

CHATBOT_TOOL_PARAM_INFO = {
    "get_functional_groups": {"param_name": "smiles_or_reaction_smiles", "description": "The SMILES or reaction SMILES string."},
    "convert_name_to_smiles": {"param_name": "chemical_name", "description": "The chemical name."},
    "suggest_disconnections": {"param_name": "smiles", "description": "The SMILES string of the molecule for disconnection suggestions."},
    "convert_smiles_to_name": {"param_name": "smiles_string", "description": "The SMILES string to convert to a name."},
    "visualize_chemical_structure": {"param_name": "smiles_or_reaction_smiles", "description": "The SMILES string for visualization."},
    "get_full_chemical_report": {"param_name": "chemical_identifier", "description": "The name, SMILES, or CAS number for a full chemical report."}
    # Add other tools if they are in CHATBOT_TOOL_PY_FUNCTIONS
}

def get_chatbot_agents():
    global _assistant_chatbot_agent, _user_proxy_chatbot_agent, llm_config_chatbot_agent, CHATBOT_TOOL_PY_FUNCTIONS, CHATBOT_TOOL_PARAM_INFO
    if _assistant_chatbot_agent is None:
        print("[DEBUG get_chatbot_agents] Initializing Chatbot Agents...")

        assistant_llm_tools_config_chat = []
        for func in CHATBOT_TOOL_PY_FUNCTIONS:
            func_name = func.__name__
            
            param_info = CHATBOT_TOOL_PARAM_INFO.get(func_name)
            if not param_info:
                print(f"[WARN] No parameter info defined in CHATBOT_TOOL_PARAM_INFO for: {func_name}. Using default 'input_string'.")
                param_name = "input_string"
                param_desc = "Input for the tool."
            else:
                param_name = param_info["param_name"]
                param_desc = param_info["description"] # Use the description from your mapping

            tool_schema = {
                "type": "function",
                "function": {
                    "name": func_name,
                    "description": (func.__doc__ or f"Executes {func_name} tool.").splitlines()[0].strip(),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            param_name: {"type": "string", "description": param_desc}
                        },
                        "required": [param_name]
                    }
                }
            }
            assistant_llm_tools_config_chat.append(tool_schema)

        # Create a copy of the base llm_config and add tools to it
        current_llm_config_for_chat_assistant = llm_config_chatbot_agent.copy()
        if assistant_llm_tools_config_chat:
            current_llm_config_for_chat_assistant["tools"] = assistant_llm_tools_config_chat
        
        _assistant_chatbot_agent = autogen.AssistantAgent(
            name="ChemistryChatbotAgent_MOI_v2", # New name
            llm_config=current_llm_config_for_chat_assistant,
            system_message="Initializing MOI-aware ChemCopilot...", # Will be immediately updated
            is_termination_msg=lambda x: isinstance(x.get("content"), str) and x.get("content", "").rstrip().endswith("TERMINATE")
        )
        _user_proxy_chatbot_agent = autogen.UserProxyAgent(
            name="UserProxyChatConversational_MOI_v2", # New name
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3, 
            code_execution_config=False,
            function_map={tool.__name__: tool for tool in CHATBOT_TOOL_PY_FUNCTIONS}, # Crucial
            is_termination_msg=lambda x: (isinstance(x.get("content"), str) and x.get("content", "").rstrip().endswith("TERMINATE")) or \
                                        (isinstance(x.get("content"), str) and "I am a specialized chemistry assistant" in x.get("content", ""))
        )
        _update_chatbot_system_message_with_moi() # Set the initial system message
        print(f"[DEBUG get_chatbot_agents] MOI Chatbot Agents Initialized. Assistant LLM config has {len(current_llm_config_for_chat_assistant.get('tools',[]))} tools.")
    return _assistant_chatbot_agent, _user_proxy_chatbot_agent

def clear_chatbot_memory_autogen():
    global _current_moi_context
    print("[Chatbot Memory] Clearing MOI context and resetting agents...")
    _current_moi_context = {"name": None, "smiles": None}
    if _assistant_chatbot_agent:
        _assistant_chatbot_agent.reset()
    if _user_proxy_chatbot_agent:
        _user_proxy_chatbot_agent.reset()
    # Update system message to reflect cleared MOI
    if _assistant_chatbot_agent: # Ensure agent exists before trying to update
        _update_chatbot_system_message_with_moi()
    print("[Chatbot Memory] MOI context and agent states cleared.")


MAX_CHATBOT_TURNS = 5 # Max agent interactions for one user query

def run_autogen_chatbot_query(user_input: str, callbacks=None): # Added callbacks for consistency
    global _current_moi_context, _assistant_chatbot_agent, _user_proxy_chatbot_agent
    print(f"--- [RUN_AUTOGEN_CHATBOT_QUERY_MOI ENTRY for '{user_input}'] ---")

    # ** CRITICAL: Handle priming message directly to avoid timeout **
    priming_pattern = r"Let's discuss the molecule of interest: (.*?) with SMILES (.*)\. Please acknowledge\."
    priming_match = re.match(priming_pattern, user_input, re.IGNORECASE)

    if priming_match:
        moi_name, moi_smiles = priming_match.groups()
        _current_moi_context["name"] = moi_name.strip()
        _current_moi_context["smiles"] = moi_smiles.strip()
        print(f"[Chatbot Priming] Detected priming message. MOI Name: '{_current_moi_context['name']}', SMILES: '{_current_moi_context['smiles']}'")

        if not _assistant_chatbot_agent or not _user_proxy_chatbot_agent:
            get_chatbot_agents()
        _update_chatbot_system_message_with_moi()
        ack_reply = f"Acknowledged. We are now discussing {_current_moi_context['name']} ({_current_moi_context['smiles']}). How can I help?"
        print(f"[Chatbot Priming] Returning direct acknowledgment: '{ack_reply}'")
        return {"analysis": ack_reply, "visualization_path": None, "error": None}

    if not _assistant_chatbot_agent or not _user_proxy_chatbot_agent:
        print("[Chatbot Query] Agents not yet initialized, calling get_chatbot_agents().")
        get_chatbot_agents()
    else:
        _update_chatbot_system_message_with_moi()

    if _assistant_chatbot_agent: _assistant_chatbot_agent.reset()
    if _user_proxy_chatbot_agent: _user_proxy_chatbot_agent.reset()

    ai_response_text = "Chatbot did not provide a clear answer (default)."
    final_tool_output_for_viz = None

    try:
        print(f"[Chatbot MOI Query] Initiating chat for query: '{user_input}' with MOI: {_current_moi_context}")
        
        _user_proxy_chatbot_agent.initiate_chat(
            recipient=_assistant_chatbot_agent,
            message=user_input,
            max_turns=MAX_CHATBOT_TURNS,
            request_timeout=llm_config_chatbot_agent.get("timeout", 90) + 30,
            clear_history=True # Assistant starts fresh based on its system prompt
        )
        print(f"[Chatbot MOI Query] initiate_chat call COMPLETED.")
        
        # --- MODIFIED RESPONSE EXTRACTION ---
        # We want the messages that the _assistant_chatbot_agent SENT to the _user_proxy_chatbot_agent.
        # These are stored in the _assistant_chatbot_agent's message history for the _user_proxy_chatbot_agent.
        
        # Get the history of messages the ASSISTANT agent has for its conversation with the USER PROXY.
        # The User Proxy acts as the "user" from the Assistant's perspective in this initiated chat.
        conversation_history_from_assistant_perspective = _assistant_chatbot_agent.chat_messages.get(_user_proxy_chatbot_agent, [])
        
        print(f"[DEBUG Chatbot MOI Query] Conversation history from Assistant's perspective (with UserProxy): {conversation_history_from_assistant_perspective}")

        if conversation_history_from_assistant_perspective:
            # Iterate backwards to find the last message FROM the assistant that is NOT a tool call.
            for msg_obj in reversed(conversation_history_from_assistant_perspective):
                # We are interested in messages where the assistant itself was speaking (role: 'assistant')
                # The 'name' field for messages sent by the assistant agent should be its own name.
                # However, 'role' is often more reliable here for its own speech.
                if msg_obj.get("role") == "assistant": # This is the assistant's own speaking turn
                    if msg_obj.get("tool_calls"):
                        # This was a turn where the assistant decided to call a tool.
                        # This is not its final textual answer.
                        print(f"[DEBUG Chatbot MOI Query] Assistant message was a tool_call: {msg_obj.get('tool_calls')}")
                        continue 
                    if msg_obj.get("content"):
                        ai_response_text = msg_obj.get("content", "").strip()
                        print(f"[DEBUG Chatbot MOI Query] Extracted assistant content: '{ai_response_text}'")
                        break # Found the last relevant textual response from assistant
            
            # Fallback if the loop didn't find a suitable message (e.g., only tool calls)
            # or if the last message was empty.
            if ai_response_text == "Chatbot did not provide a clear answer (default)." and conversation_history_from_assistant_perspective:
                # Try the absolute last message from the assistant's perspective, even if it doesn't perfectly fit
                last_msg_from_assistant = conversation_history_from_assistant_perspective[-1]
                if last_msg_from_assistant.get("role") == "assistant" and last_msg_from_assistant.get("content"):
                    ai_response_text = last_msg_from_assistant.get("content").strip()
                    print(f"[DEBUG Chatbot MOI Query] Fallback to last assistant message content: '{ai_response_text}'")
        else:
            print("[DEBUG Chatbot MOI Query] No conversation history found from Assistant's perspective.")

        # --- END OF MODIFIED RESPONSE EXTRACTION ---
        
        if ai_response_text.upper().endswith("TERMINATE"):
            ai_response_text = ai_response_text[:-len("TERMINATE")].strip()
        if not ai_response_text.strip() and ai_response_text != "Chatbot did not provide a clear answer (default).":
            # If after stripping TERMINATE, the response is empty, it might mean the agent only said TERMINATE
            # or the LLM returned an empty string.
            if "Chatbot did not provide a clear answer" not in ai_response_text : # Avoid overwriting actual error
                ai_response_text = "Chatbot processing complete." # Or keep it as is if you prefer to see empty

        # Check for visualization path from tool execution results
        # Tool execution results are typically messages with role="tool" sent from UserProxy to Assistant
        # So, we look for them in the assistant's received messages (which are from the UserProxy).
        # Alternatively, if the tool function itself returns a string that includes the path,
        # and the assistant includes that string in its final textual response, it might be there too.

        # Let's check the messages the USER PROXY sent (which would include tool responses)
        user_proxy_sent_messages = _user_proxy_chatbot_agent.chat_messages.get(_assistant_chatbot_agent, [])
        print(f"[DEBUG Chatbot MOI Query] Messages sent by UserProxy to Assistant (may include tool results): {user_proxy_sent_messages}")
        for msg_item in user_proxy_sent_messages:
            if msg_item.get("role") == "tool": # This is a message from UserProxy reporting a tool's output
                tool_content = msg_item.get("content", "")
                if "static/autogen_visualizations/" in tool_content:
                    match_viz = re.search(r"(static/autogen_visualizations/[\w\-\.\_]+\.png)", tool_content)
                    if match_viz:
                        final_tool_output_for_viz = match_viz.group(1)
                        print(f"[DEBUG Chatbot MOI Query] Found viz path from 'tool' role message: {final_tool_output_for_viz}")
                        break
        
        if not final_tool_output_for_viz and "static/autogen_visualizations/" in ai_response_text:
             match_viz = re.search(r"(static/autogen_visualizations/[\w\-\.\_]+\.png)", ai_response_text)
             if match_viz:
                 final_tool_output_for_viz = match_viz.group(1)
                 print(f"[DEBUG Chatbot MOI Query] Found viz path in final AI text response: {final_tool_output_for_viz}")


        print(f"[DEBUG Chatbot MOI Query] Final extracted ai_response_text: '{ai_response_text}', Viz: {final_tool_output_for_viz}")
        return {
            "visualization_path": final_tool_output_for_viz,
            "analysis": ai_response_text,
            "error": None
        }

    except openai.APITimeoutError as e_timeout: # Make sure openai is imported for this
        tb_str = traceback.format_exc()
        print(f"!!!!!!!! OpenAI APITimeoutError IN MOI Chatbot Query: {e_timeout}\n{tb_str} !!!!!!!!")
        return { "visualization_path": None, "analysis": f"OpenAI API timed out: {str(e_timeout)}", "error": str(e_timeout)}
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"!!!!!!!! ERROR IN MOI Chatbot Query: {e}\n{tb_str} !!!!!!!!")
        return { "visualization_path": None, "analysis": f"An error occurred in the MOI chatbot: {str(e)}", "error": str(e) }
    
# --- MODIFIED Main Query Routing Logic (enhanced_query - UNCHANGED from your last version) ---
# Your enhanced_query logic looks reasonable for routing. It will now call the modified run_autogen_chatbot_query.
# ... (Your full implementation of enhanced_query)
def enhanced_query(full_query: str, callbacks=None, original_compound_name: str = None):
    global _current_moi_context
    final_result = {}
    query_context_for_filename = "unknown_query_type"
    # Extract SMILES first, as they are more definitive
    reaction_smiles_for_tools = extract_reaction_smiles(full_query)
    compound_smiles_for_tools = None
    if not reaction_smiles_for_tools: # Only try compound if no reaction SMILES
        compound_smiles_for_tools = extract_single_compound_smiles(full_query)

    print(f"[ENHANCED_QUERY_AUTOGEN] Original query: '{full_query[:100]}...'")
    print(f"[ENHANCED_QUERY_AUTOGEN] Initial Extracted Reaction SMILES: {reaction_smiles_for_tools}")
    print(f"[ENHANCED_QUERY_AUTOGEN] Initial Extracted Compound SMILES: {compound_smiles_for_tools}")

    query_lower = full_query.lower()
    full_info_keywords = ["full info", "full information", "all info", "details about", "tell me about", "explain this", "give the full", "everything about", "full analysis"] # Broadened list

    try:
        # --- Priority 1: Update MOI + Full info for an explicitly provided COMPOUND SMILES ---
        if compound_smiles_for_tools and any(keyword in query_lower for keyword in full_info_keywords):
            # Determine name for compound
            compound_name_for_moi = original_compound_name or compound_smiles_for_tools

            # Update Current Molecule of Interest (MOI)
            _current_moi_context["name"] = compound_name_for_moi # If you want to use a provided name
            _current_moi_context["smiles"] = compound_smiles_for_tools  # Set SMILES
            print(f"[ENHANCED_QUERY_AUTOGEN] Updated MOI Context (comp SMILES): Name='{_current_moi_context['name']}', SMILES='{_current_moi_context['smiles']}'")
            # If "full info" keywords are present AND a compound SMILES was extracted,
            # prioritize this path for compounds.
            print(f"[ENHANCED_QUERY_AUTOGEN] Full info requested directly with COMPOUND SMILES: {compound_smiles_for_tools}")
            final_result = handle_compound_full_info(full_query, compound_smiles_for_tools, original_compound_name or compound_smiles_for_tools, callbacks=callbacks)
            query_context_for_filename = final_result.get('analysis_context', 'compound_full_info_direct_smiles')
            if 'processed_smiles_for_tools' not in final_result:
                final_result['processed_smiles_for_tools'] = compound_smiles_for_tools

        # --- Priority 2: Update MOI + Full info for an explicitly provided REACTION SMILES ---
        elif reaction_smiles_for_tools and any(keyword in query_lower for keyword in full_info_keywords):

            # Update Current Molecule of Interest (MOI) (for reactions use reaction SMILES as name)
            _current_moi_context["name"] = original_compound_name or reaction_smiles_for_tools # Store provided name
            _current_moi_context["smiles"] = reaction_smiles_for_tools # Set reaction SMILES
            print(f"[ENHANCED_QUERY_AUTOGEN] Updated MOI Context (rxn SMILES): Name='{_current_moi_context['name']}', SMILES='{_current_moi_context['smiles']}'")

            print(f"[ENHANCED_QUERY_AUTOGEN] Full analysis requested directly with REACTION SMILES: {reaction_smiles_for_tools}")
            final_result = handle_full_info(full_query, reaction_smiles_for_tools, original_compound_name or reaction_smiles_for_tools, callbacks=callbacks)
            query_context_for_filename = final_result.get('analysis_context', 'reaction_full_info_direct_smiles')
            if 'processed_smiles_for_tools' not in final_result:
                final_result['processed_smiles_for_tools'] = reaction_smiles_for_tools

        # --- Priority 3: Update MOI + Full info for a NAMED entity (if no SMILES was directly used above) ---
        # This block should only be entered if the above SMILES + "full info" checks didn't trigger.
        elif not compound_smiles_for_tools and not reaction_smiles_for_tools and any(keyword in query_lower for keyword in full_info_keywords):
            name_entity_for_full_info = None
            # Try to extract a name if "full info" is present but no SMILES was found yet
            # Regex needs to be careful not to match SMILES as names
            name_pattern_match = re.search(
                r"(?:full info|tell me everything|full analysis|give the full info)(?: about| for| of)? named?\s*['\"]?([\w\s\-]+?)['\"]?(?:\s|$|\.|\?)", # More specific name pattern
                query_lower
            ) or re.search(
                 r"(?:full info|tell me everything|full analysis|give the full info)(?: about| for| of)?\s+([\w\s\-]+?)(?:\s|$|\.|\?)(?!.*(?:smiles|reaction|>>| sguardo))", # Avoid common SMILES patterns/keywords
                 query_lower
            )

            if name_pattern_match:
                potential_name = name_pattern_match.group(1).strip()
                # Further check if this "potential_name" is not actually a SMILES
                if potential_name and not extract_single_compound_smiles(potential_name) and not extract_reaction_smiles(potential_name):
                    name_entity_for_full_info = potential_name
                    print(f"[ENHANCED_QUERY_AUTOGEN] Detected 'full info' for NAMED entity: '{name_entity_for_full_info}' (no direct SMILES found earlier for full info)")
            
            if name_entity_for_full_info:
                print(f"[ENHANCED_QUERY_AUTOGEN] Processing 'full info' for NAMED entity: '{name_entity_for_full_info}'")
                n2s_tool_output_str = ag_tools.convert_name_to_smiles(name_entity_for_full_info)
                smiles_from_name = None
                if isinstance(n2s_tool_output_str, str):
                    match_smiles = re.search(r"SMILES:\s*([^\s\n]+)", n2s_tool_output_str)
                    if match_smiles: smiles_from_name = match_smiles.group(1).strip()
                
                if smiles_from_name:
                    print(f"[ENHANCED_QUERY_AUTOGEN] SMILES for '{name_entity_for_full_info}' is '{smiles_from_name}'. Proceeding.")

                    # Update Current Molecule of Interest (MOI)
                    _current_moi_context["name"] = name_entity_for_full_info # Use the provided name
                    _current_moi_context["smiles"] = smiles_from_name # Set SMILES that we've extracted
                    print(f"[ENHANCED_QUERY_AUTOGEN] Updated MOI Context (with full info + extracted name): Name='{_current_moi_context['name']}', SMILES='{_current_moi_context['smiles']}'")

                    if ">>" in smiles_from_name:
                        final_result = handle_full_info(f"Full info for {smiles_from_name} (from name '{name_entity_for_full_info}')", smiles_from_name, name_entity_for_full_info)
                    else:
                        final_result = handle_compound_full_info(f"Full info for {smiles_from_name} (from name '{name_entity_for_full_info}')", smiles_from_name, name_entity_for_full_info)
                    query_context_for_filename = final_result.get('analysis_context', 'full_info_from_name')
                    final_result['processed_smiles_for_tools'] = smiles_from_name
                else:
                    final_result = {"analysis": f"Could not find SMILES for name '{name_entity_for_full_info}'.", "processed_smiles_for_tools": None}
                    query_context_for_filename = "full_info_name_conversion_failed"
            # If no name was extracted here but "full info" keywords were present, it will fall through to general chatbot.
        
        # --- Remaining Cases (Specific tool calls for SMILES, or general chat) ---
        # This block is reached if "full info" was NOT the primary intent for an extracted SMILES,
        # OR if no SMILES + "full info" was found, OR no name + "full info" was found.
        if not final_result: # If no result determined by "full info" logic above
            if reaction_smiles_for_tools:
                # Reaction SMILES present, but not a "full info" request for it.
                # Could be visualization, or a specific follow-up question.
                is_visualization_request = any(term in query_lower for term in ["visual", "picture", "image", "show", "draw"])
                if is_visualization_request:
                    print(f"[ENHANCED_QUERY_AUTOGEN] Visualization for REACTION: {reaction_smiles_for_tools}")
                    # ... (visualization logic as before) ...
                    viz_path_cached = reaction_cache.get(reaction_smiles_for_tools, {}).get('visualization_path')
                    if viz_path_cached and not str(viz_path_cached).lower().startswith("error"):
                        final_result = {"visualization_path": viz_path_cached, "analysis": f"Cached visual for: {reaction_smiles_for_tools}"}
                    else:
                        viz_result = ag_tools.visualize_chemical_structure(reaction_smiles_for_tools)
                        if viz_result and not viz_result.lower().startswith("error") and ".png" in viz_result:
                            final_result = {"visualization_path": viz_result, "analysis": f"Visual generated for: {reaction_smiles_for_tools}"}
                            reaction_cache.setdefault(reaction_smiles_for_tools, {})['visualization_path'] = viz_result
                        else:
                            final_result = {"visualization_path": None, "analysis": viz_result}
                    query_context_for_filename = "visualization_reaction"

                else: # Follow-up or general query about the reaction
                    print(f"[ENHANCED_QUERY_AUTOGEN] Follow-up/general for REACTION: {reaction_smiles_for_tools}")
                    #Update MOI before follow up.
                    _current_moi_context["name"] = original_compound_name or reaction_smiles_for_tools  # Store provided name
                    _current_moi_context["smiles"] = reaction_smiles_for_tools # Update SMILES

                    followup_result = handle_followup_question(full_query, reaction_smiles_for_tools, original_compound_name)
                    if followup_result.get('analysis'):
                        final_result = followup_result
                    else:
                        final_result = run_autogen_chatbot_query(full_query, callbacks=callbacks)
                    query_context_for_filename = followup_result.get('analysis_context', 'chatbot_reaction_followup')
                if 'processed_smiles_for_tools' not in final_result:
                    final_result['processed_smiles_for_tools'] = reaction_smiles_for_tools

            elif compound_smiles_for_tools:
                # Compound SMILES present, but not a "full info" request for it.
                # Likely a specific tool query or general chat.
                print(f"[ENHANCED_QUERY_AUTOGEN] Specific tool/chat for COMPOUND: {compound_smiles_for_tools}")

                _current_moi_context["name"] = original_compound_name or compound_smiles_for_tools  # Store provided name
                _current_moi_context["smiles"] = compound_smiles_for_tools # Update SMILES
                # Route to tool agent first. If it refuses, then chatbot.
                tool_agent_result = run_autogen_tool_agent_query(full_query, callbacks=callbacks)
                analysis_from_tool = tool_agent_result.get("analysis", "")
                if "I can only perform specific chemical analyses" in analysis_from_tool or \
                   "Tool agent did not provide a clear answer" in analysis_from_tool or \
                   analysis_from_tool.strip() == "Tool agent processing complete.": # If tool agent had nothing useful
                    print(f"[ENHANCED_QUERY_AUTOGEN] Tool agent ineffective for compound query. Routing to CHATBOT.")
                    final_result = run_autogen_chatbot_query(full_query, callbacks=callbacks)
                    query_context_for_filename = "chatbot_compound_specific_fallback"
                else:
                    final_result = tool_agent_result
                    query_context_for_filename = "tool_agent_compound_specific"
                if 'processed_smiles_for_tools' not in final_result:
                    final_result['processed_smiles_for_tools'] = compound_smiles_for_tools
            
            else: # No SMILES extracted, and no "full info for name" triggered
                print("[ENHANCED_QUERY_AUTOGEN] No SMILES & not 'full info for name'. Routing to CHATBOT.")
                final_result = run_autogen_chatbot_query(full_query, callbacks=callbacks)
                query_context_for_filename = "chatbot_general_no_smiles"
                if 'processed_smiles_for_tools' not in final_result:
                    final_result['processed_smiles_for_tools'] = None
        
        # Fallback if final_result is still empty for some reason
        if not final_result:
            print("[ENHANCED_QUERY_AUTOGEN] Fallback: All routing failed, defaulting to chatbot.")
            final_result = run_autogen_chatbot_query(full_query, callbacks=callbacks)
            query_context_for_filename = "chatbot_fallback_routing_failed"

         # We always send what's current before this enhanced query.
        final_result["current_moi_name"] = _current_moi_context.get("name")
        final_result["current_moi_smiles"] =  _current_moi_context.get("smiles")

        # --- Final processing and saving ---
        analysis_text_to_save = final_result.get("analysis")
        smiles_for_saving = final_result.get('processed_smiles_for_tools')

        # Update query_context_for_filename if it wasn't set by specific logic paths
        if query_context_for_filename == "unknown_query_type" and final_result.get('analysis_context'):
            query_context_for_filename = final_result.get('analysis_context')

        if smiles_for_saving and analysis_text_to_save and isinstance(analysis_text_to_save, str) and \
           not query_context_for_filename.startswith("visualization_") and \
           not any(err_ctx in query_context_for_filename.lower() for err_ctx in ["error", "fail", "not_found", "invalid", "refusal", "no_smiles"]) and \
           len(analysis_text_to_save.strip()) > 50 :
            is_refusal = "I am a specialized chemistry assistant" in analysis_text_to_save or \
                         "I can only perform specific chemical analyses" in analysis_text_to_save
            if not is_refusal: # Don't save generic refusals
                save_analysis_to_file(smiles_for_saving, analysis_text_to_save, query_context_for_filename, original_compound_name)

        if 'processed_smiles_for_tools' not in final_result: # Final safety net
             final_result['processed_smiles_for_tools'] = reaction_smiles_for_tools or compound_smiles_for_tools or (name_entity_for_full_info if 'name_entity_for_full_info' in locals() and name_entity_for_full_info else None)

        # Ensure query_context_for_filename is set in the result for the API
        final_result['analysis_context'] = query_context_for_filename
        return final_result

    except Exception as e:
        # ... (existing exception handling) ...
        tb_str = traceback.format_exc()
        print(f"CRITICAL Error in enhanced_query_autogen for query '{full_query}': {str(e)}\n{tb_str}")
        error_text = f"Error processing your query: {str(e)}."
        # Determine a sensible smiles_ctx_for_error_log
        smiles_ctx_for_error_log = compound_smiles_for_tools or reaction_smiles_for_tools or \
                                   (name_entity_for_full_info if 'name_entity_for_full_info' in locals() and name_entity_for_full_info else None) or \
                                   "no_entity_extracted_in_error"
        save_analysis_to_file(smiles_ctx_for_error_log, f"Query: {full_query}\n{error_text}\n{tb_str}", "enhanced_query_CRITICAL_error", original_compound_name)

        # Ensure we return MOI info even in case of error
        return {
            "visualization_path": None,
            "analysis": error_text,
            "error": str(e), # Add error field
            "processed_smiles_for_tools": smiles_ctx_for_error_log,
            "analysis_context": "enhanced_query_exception",
            "current_moi_name": _current_moi_context.get("name"),
            "current_moi_smiles":  _current_moi_context.get("smiles") # Always return MOI
        }

# --- display_analysis_result - UNCHANGED ---
def display_analysis_result(title: str, analysis_result: dict, is_chatbot: bool = False):
    print(f"\n--- {title} ---")
    if analysis_result and isinstance(analysis_result, dict) and analysis_result.get("analysis"):
        analysis_text = analysis_result["analysis"]
        analysis_text = re.sub(r"!\[.*?\]\((.*?)\)", r"[Image available at: \1]", analysis_text)
        if is_chatbot:
            print(f"Chem Copilot: {analysis_text}")
        else:
            print(analysis_text)
        if analysis_result.get("visualization_path") and \
           (not isinstance(analysis_text, str) or "Image available at" not in analysis_text):
            print(f"Visualization available at: {analysis_result['visualization_path']}")
        if not is_chatbot and analysis_result.get("processed_smiles_for_tools"):
            print(f"Processed SMILES: {analysis_result.get('processed_smiles_for_tools')}")
    elif analysis_result and isinstance(analysis_result, dict) and analysis_result.get("error"):
        print(f"Error: {analysis_result.get('error')}")
    else:
        print("Could not retrieve or generate a response. Result was empty, malformed, or not a dictionary.")
        print(f"Raw result received for '{title}': {analysis_result}")
    print(f"--- End of {title} ---\n")

# --- Main execution block for testing (UNCHANGED from your last version) ---
# This will test the new chatbot logic when non-"full info" queries are made.
if __name__ == "__main__":
    print("\nTesting direct OpenAI API call...")
    try:
        if os.environ.get("OPENAI_API_KEY"):
            client = openai.OpenAI()
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello!"}],
                max_tokens=5,
                timeout=30
            )
            print(f"[Direct OpenAI Test] Success: {completion.choices[0].message.content}")
        else:
            print("[Direct OpenAI Test] SKIPPED - OPENAI_API_KEY not set.")
    except Exception as e_direct_openai:
        print(f"[Direct OpenAI Test] FAILED: {e_direct_openai}")
        print("This indicates a problem with your OpenAI API key, network, or the OpenAI service.")
    print("Continuing with interactive mode...\n")
    print("You can ask for 'full info' on a compound/reaction (by SMILES or name), or chat generally.")
    print("Priming MOI example: 'Let's discuss the molecule of interest: Aspirin with SMILES CC(=O)OC1=CC=CC=C1C(=O)O. Please acknowledge.'")
    print("Then ask: 'What are its functional groups?' or 'Show me the structure.'")
    print("Other examples:")
    print("  'full info for CCO'")
    print("  'tell me everything about the reaction CCO>>CC=O'")
    print("  'give me the full analysis of Aspirin'")
    print("  'What is the name of C1=CC=CS1?' (Tool Agent)")
    print("  'Visualize C1=CC=CS1' (Tool Agent or Chatbot with MOI)")
    print("Type 'exit' or 'quit' to end. Type 'clear chat' to reset MOI context.")

    clear_chatbot_memory_autogen() 
    last_processed_entity_name_for_saving: Optional[str] = None 

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive session.")
            break
        if not user_input: continue
        if user_input.lower() in ['exit', 'quit', 'done']:
            print("Exiting interactive session.")
            break
        
        if user_input.lower() in ["clear chat history", "clear context", "reset chat", "clear chat"]:
            clear_chatbot_memory_autogen()
            last_processed_entity_name_for_saving = None
            print("Chem Copilot: Chat MOI context and agent states have been cleared.")
            continue
        
        # Extract potential original name if "full info for [name]" pattern is used, for saving purposes
        current_original_name_for_saving = None
        name_match_for_saving = re.search(
            r"full\s+(?:info|information|analysis|detail)\s+(?:for|of|about)\s+named?\s*['\"]?([^'\"\n\.,]+?)['\"]?(?:\s|$|\.|\?)",
            user_input, re.IGNORECASE
        ) or re.search(
             r"full\s+(?:info|information|analysis|detail)\s+(?:for|of|about)\s+([^'\"\n\.,\s]+)(?:\s|$|\.|\?)(?!.*smiles)",
             user_input, re.IGNORECASE
        )
        if name_match_for_saving:
            potential_name_for_saving = name_match_for_saving.group(1).strip()
            # Basic check to avoid using SMILES as original name here
            if not (">>" in potential_name_for_saving or extract_single_compound_smiles(potential_name_for_saving)):
                 current_original_name_for_saving = potential_name_for_saving
        
        # If a priming message sets MOI, use that name for saving subsequent full analyses if any
        priming_match_loop = re.match(r"Let's discuss the molecule of interest: (.*?) with SMILES .*\. Please acknowledge\.", user_input, re.IGNORECASE)
        if priming_match_loop:
            last_processed_entity_name_for_saving = priming_match_loop.group(1).strip()
        
        query_result = enhanced_query(user_input, original_compound_name=current_original_name_for_saving or last_processed_entity_name_for_saving)
        
        is_chatbot_response = False
        analysis_context = query_result.get('analysis_context', '')
        if 'chatbot' in analysis_context or 'tool_agent' in analysis_context or \
           "followup" in analysis_context or "refusal" in analysis_context or \
           _current_moi_context.get("name") is not None : # If MOI is set, it's likely a chat interaction
            is_chatbot_response = True 
            
        display_analysis_result(f"Chem Copilot Response", query_result, is_chatbot=is_chatbot_response)

        # Update last_processed_entity_name_for_saving if a "full info" was successful for a new entity
        if query_result and isinstance(query_result, dict) and \
           ('full_direct_openai_summary_generated' in analysis_context or \
            'compound_openai_summary_with_disconnections' in analysis_context or \
            'full_info_from_name' in analysis_context) and \
           query_result.get('processed_smiles_for_tools'):
            
            # Prefer original_compound_name used for the query if available and it was a name-based full info
            if current_original_name_for_saving:
                 last_processed_entity_name_for_saving = current_original_name_for_saving
            elif not last_processed_entity_name_for_saving : # Only update if not already set by priming or a more specific name
                # Fallback: try to get a name from the analysis if possible (e.g., from smiles2name)
                # This part is tricky as 'analysis' is a summary.
                # For now, simply acknowledge that a full analysis was done.
                # The MOI context in the chatbot handles the "current" entity.
                print(f"[Interactive Loop Note] Full analysis successful for SMILES: {query_result.get('processed_smiles_for_tools')}")
                # If MOI was set by this full analysis, reflect it for saving name context
                if _current_moi_context.get("smiles") == query_result.get('processed_smiles_for_tools') and _current_moi_context.get("name"):
                    last_processed_entity_name_for_saving = _current_moi_context.get("name")


    print("\nAll tests completed (or interactive session ended).")