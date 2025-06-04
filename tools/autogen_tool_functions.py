import os
import re
import time
import pandas as pd # ReactionClassifier uses pandas
from rdkit import Chem # Used by several tools
from rdkit.Chem import AllChem # Often needed with RDKit drawing/reactions
from rdkit.Chem.Draw import ReactionToImage, MolToImage # CORRECTED: Added this import

# --- Import the actual classes/logic from your existing tool files ---
from .funcgroups import FuncGroups
from .name2smiles import NameToSMILES
from .smiles2name import SMILES2Name
from .bond import BondChangeAnalyzer
from .visualizer import ChemVisualizer # Your ChemVisualizer class
from .asckos import ReactionClassifier
from .disconnection import DisconnectionSuggester
from .chemical_property import ChemicalAnalysisAgent, ChemicalProperties

import api_config

# --- Global Path for Visualizations ---
PROJECT_ROOT_DIR_AG = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # CHEM_COPILOT directory
AUTOGEN_VISUALIZATION_DIR = os.path.join(PROJECT_ROOT_DIR_AG, "static", "autogen_visualizations")
os.makedirs(AUTOGEN_VISUALIZATION_DIR, exist_ok=True) 

# --- Tool Instances (can be global if they are stateless or manage their own state well) ---
_funcgroups_tool_instance = FuncGroups()
_disconnection_suggester_instance = DisconnectionSuggester()
_name2smiles_tool_instance = NameToSMILES()
_smiles2name_tool_instance = SMILES2Name()
_bond_analyzer_tool_instance = BondChangeAnalyzer()
_chemical_analyzer_instance = None
# _visualizer_tool_instance will be instantiated within its wrapper function to ensure it uses the correct paths
# or if its __init__ becomes path-dependent in the future.

_reaction_classifier_instance_ag = None # Instantiated on first use via helper

def get_reaction_classifier_instance_ag():
    """Gets or initializes the ReactionClassifier instance for AutoGen tools."""
    global _reaction_classifier_instance_ag
    if _reaction_classifier_instance_ag is None:
        dataset_path1 = os.environ.get('REACTION_DATASET_PATH1', None)
        dataset_path2 = os.environ.get('REACTION_DATASET_PATH2', None)
        print(f"[AutoGenTool] Initializing ReactionClassifier. Paths: D1='{dataset_path1}', D2='{dataset_path2}'")
        try:
            _reaction_classifier_instance_ag = ReactionClassifier(dataset_path1, dataset_path2)
            print("ReactionClassifier instance for AutoGen created successfully.")
        except Exception as e:
            print(f"Error initializing ReactionClassifier for AutoGen: {e}")
    return _reaction_classifier_instance_ag

# --- Utility for sanitizing filenames (if needed by visualizer or other tools) ---
def sanitize_filename_ag(name: str) -> str:
    if not isinstance(name, str): name = str(name)
    name = re.sub(r'[^\w\.\-]+', '_', name); return name[:100]

# --- AutoGen Callable Functions ---

def get_functional_groups(smiles_or_reaction_smiles: str) -> str:
    """
    Identifies functional groups in a molecule or reaction.
    Provide a valid SMILES or reaction SMILES string as input.
    Returns a string detailing functional groups, transformations, or an error.

    Args:
        smiles_or_reaction_smiles (str): The SMILES or reaction SMILES string.

    Returns:
        str: Analysis of functional groups or an error message.
    """
    print(f"[AutoGenTool] get_functional_groups for: {smiles_or_reaction_smiles}")
    try:
        result_dict = _funcgroups_tool_instance._run(smiles_or_reaction_smiles)
        if isinstance(result_dict, dict) and "error" in result_dict:
            return f"Error from FuncGroups tool: {result_dict['error']}"
        return f"Functional Group Analysis: {str(result_dict)}" # Ensure good string conversion
    except Exception as e:
        return f"Error in get_functional_groups tool wrapper: {str(e)}"

def convert_name_to_smiles(chemical_name: str) -> str:
    """
    Converts a compound, molecule, or reaction name to its SMILES representation.
    Uses CAS Common Chemistry API with a fallback to PubChem.
    This tool does NOT accept SMILES as input. Use only for chemical names.

    Args:
        chemical_name (str): The chemical name (e.g., 'aspirin', 'glucose').

    Returns:
        str: A string containing the SMILES and source, or an error message.
    """
    print(f"[AutoGenTool] convert_name_to_smiles for: {chemical_name}")
    try:
        return _name2smiles_tool_instance._run(chemical_name)
    except Exception as e:
        return f"Error in convert_name_to_smiles tool wrapper: {str(e)}"

def convert_smiles_to_name(smiles_string: str) -> str:
    """
    Converts a SMILES string to its chemical name(s) (IUPAC and common if found).
    Uses CACTUS, PubChem, and an LLM for common name retrieval.

    Args:
        smiles_string (str): The SMILES string.

    Returns:
        str: A string containing common and IUPAC names, or an error message.
    """
    print(f"[AutoGenTool] convert_smiles_to_name for: {smiles_string}")
    try:
        return _smiles2name_tool_instance._run(smiles_string)
    except Exception as e:
        return f"Error in convert_smiles_to_name tool wrapper: {str(e)}"

def analyze_reaction_bond_changes(reaction_smiles: str) -> str:
    """
    Identifies bonds broken, formed, and changed in a chemical reaction.
    Works with mapped or unmapped reaction SMILES (will attempt to map unmapped ones).

    Args:
        reaction_smiles (str): The reaction SMILES string.

    Returns:
        str: A string detailing bond changes, mapped reaction, or an error message.
    """
    print(f"[AutoGenTool] analyze_reaction_bond_changes for: {reaction_smiles}")
    try:
        result_dict = _bond_analyzer_tool_instance._run(reaction_smiles)
        if isinstance(result_dict, dict) and "error" in result_dict:
            return f"Error from BondChangeAnalyzer: {result_dict['error']}"
        return f"Bond Change Analysis: {str(result_dict)}" # Ensure good string conversion
    except Exception as e:
        return f"Error in analyze_reaction_bond_changes tool wrapper: {str(e)}"

def visualize_chemical_structure(smiles_or_reaction_smiles: str) -> str:
    """
    Visualizes a chemical molecule or reaction from its SMILES string.
    Saves the image to 'static/autogen_visualizations/' and returns the relative path 
    (e.g., 'static/autogen_visualizations/viz_CCO_timestamp.png').
    This path can be used in Markdown for display.

    Args:
        smiles_or_reaction_smiles (str): The SMILES or reaction SMILES string.

    Returns:
        str: Relative path to the generated image or an error message.
    """
    print(f"[AutoGenTool] visualize_chemical_structure for: {smiles_or_reaction_smiles}")
    try:
        local_visualizer_instance = ChemVisualizer() 
        input_type = local_visualizer_instance.detect_input_type(smiles_or_reaction_smiles)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        sanitized_smiles_for_filename = sanitize_filename_ag(smiles_or_reaction_smiles)
        
        filename = f"{input_type}_{sanitized_smiles_for_filename}_{timestamp}.png"
        absolute_output_filepath = os.path.join(AUTOGEN_VISUALIZATION_DIR, filename)
        relative_path_for_app = os.path.join("static/autogen_visualizations", filename)

        if input_type == 'reaction':
            result_path_or_error = local_visualizer_instance.visualize_reaction(
                smiles_or_reaction_smiles,
                output_file=absolute_output_filepath 
            )
        else: 
            result_path_or_error = local_visualizer_instance.visualize_molecule(
                smiles_or_reaction_smiles,
                output_file=absolute_output_filepath
            )

        if result_path_or_error == absolute_output_filepath: 
            return relative_path_for_app
        else:
            return result_path_or_error 

    except Exception as e:
        return f"Error in visualize_chemical_structure tool wrapper: {str(e)}"


def classify_reaction_and_get_details(reaction_smiles: str) -> str:
    """
    Classifies a chemical reaction using an external API and retrieves detailed
    information from local datasets based on the reaction SMILES.

    Args:
        reaction_smiles (str): The reaction SMILES string.

    Returns:
        str: A comprehensive report including API classification, certainty, and
             detailed information from datasets, or an error message.
    """
    print(f"[AutoGenTool] classify_reaction_and_get_details for: {reaction_smiles}")
    classifier = get_reaction_classifier_instance_ag()
    if classifier:
        try:
            return classifier._run(reaction_smiles=reaction_smiles, query=None) 
        except Exception as e:
            return f"Error during reaction classification tool execution: {str(e)}"
    else:
        return "ReactionClassifier tool is not available (failed to initialize)."

def query_specific_property_for_reaction(reaction_smiles: str, property_to_query: str) -> str:
    """
    Queries a specific property (e.g., 'temperature', 'yield', 'solvent', 'catalyst', 'time')
    for a given reaction SMILES. It uses the ReactionClassifier tool which first
    classifies the reaction via API and then searches local datasets for the property.

    Args:
        reaction_smiles (str): The Reaction SMILES string.
        property_to_query (str): The specific property to query (e.g., 'temperature', 'yield').

    Returns:
        str: Information about the specified property for the reaction, or an error/not found message.
    """
    print(f"[AutoGenTool] query_specific_property_for_reaction for: {reaction_smiles}, Property: {property_to_query}")
    classifier = get_reaction_classifier_instance_ag()
    if classifier:
        try:
            return classifier._run(reaction_smiles=reaction_smiles, query=property_to_query)
        except Exception as e:
            return f"Error during specific reaction property query: {str(e)}"
    else:
        return "ReactionClassifier tool is not available for property query (failed to initialize)."
    
def suggest_disconnections(smiles: str) -> str:
    """
    Identifies functional groups in a molecule and then suggests retrosynthetic
    disconnections based on these groups using an LLM.
    Provide a valid SMILES string for a single molecule as input.
    Returns a string detailing the identified functional groups and disconnection suggestions, or an error.

    Args:
        smiles (str): The SMILES string of the molecule.

    Returns:
        str: Analysis of functional groups, disconnection suggestions, or an error message.
    """
    print(f"[AutoGenTool] suggest_disconnections for: {smiles}")
    if ">>" in smiles:
        return "Error: This tool is for single molecules, not reactions. Please provide a molecule SMILES."

    functional_groups = []
    # Step 1: Get functional groups
    try:
        # Call the _run method of your FuncGroups instance or its wrapper
        # Assuming _funcgroups_tool_instance._run returns a dict like {'smiles': '...', 'functional_groups': [...]}
        # or {'error': '...'}
        fg_result_dict = _funcgroups_tool_instance._run(smiles) # Direct call to the instance's method
        
        if isinstance(fg_result_dict, dict) and "error" in fg_result_dict:
            return f"Error from FuncGroups sub-tool: {fg_result_dict['error']}"
        elif isinstance(fg_result_dict, dict) and "functional_groups" in fg_result_dict:
            functional_groups = fg_result_dict["functional_groups"]
            if not functional_groups:
                 print(f"[suggest_disconnections] No functional groups identified by FuncGroups for {smiles}.")
                 # We can still proceed and let the LLM in DisconnectionSuggester handle it, or return early.
                 # Let's proceed for now.
        else:
            # This case should ideally not happen if _funcgroups_tool_instance._run is consistent
            print(f"[suggest_disconnections] Unexpected result from FuncGroups: {fg_result_dict}")
            return f"Error: Could not reliably identify functional groups for {smiles} before suggesting disconnections."

    except Exception as e_fg:
        return f"Error in suggest_disconnections (during FG identification): {str(e_fg)}"

    # Step 2: Get disconnection suggestions
    try:
        # Call the _run method of your DisconnectionSuggester instance
        disconnection_result_dict = _disconnection_suggester_instance._run(smiles, functional_groups)
        
        if isinstance(disconnection_result_dict, dict) and "error" in disconnection_result_dict:
            return f"Error from DisconnectionSuggester sub-tool: {disconnection_result_dict['error']}"
        
        # Format the output nicely
        output_str = f"Disconnection Analysis for SMILES: {smiles}\n"
        output_str += f"Identified Functional Groups: {', '.join(functional_groups) if functional_groups else 'None identified'}\n\n"
        output_str += "Suggested Disconnections (from LLM):\n"
        output_str += disconnection_result_dict.get("disconnection_suggestions", "No suggestions available or LLM error.")
        
        return output_str

    except Exception as e_disc:
        return f"Error in suggest_disconnections (during disconnection suggestion): {str(e_disc)}"

def get_chemical_analyzer_instance(): # Renamed for clarity if needed
    global _chemical_analyzer_instance
    if _chemical_analyzer_instance is None:
        print("[AutoGenTool] Initializing ChemicalAnalysisAgent instance...")
        try:
            _chemical_analyzer_instance = ChemicalAnalysisAgent(api_config.api_key)
            print("[AutoGenTool] ChemicalAnalysisAgent instance created successfully.")
        except Exception as e:
            print(f"[AutoGenTool ERROR] Failed to initialize ChemicalAnalysisAgent: {e}")
    return _chemical_analyzer_instance

def get_full_chemical_report(chemical_identifier: str) -> str:
    """
    Provides a comprehensive analysis report for a chemical compound, using its name,
    SMILES string, or CAS number as input. Fetches data primarily from PubChem and uses
    an LLM for interpretation of solubility, hazards, safety, environmental impact,
    green chemistry score, and estimated pricing.
    Returns a detailed textual report.
    Example: get_full_chemical_report(chemical_identifier="benzene")
    Example: get_full_chemical_report(chemical_identifier="CCO")
    Example: get_full_chemical_report(chemical_identifier="7647-14-5")
    """
    print(f"[AutoGenTool] get_full_chemical_report for: {chemical_identifier}")
    analyzer = get_chemical_analyzer_instance()
    if analyzer:
        try:
            properties: ChemicalProperties = analyzer.analyze_chemical(chemical_identifier)
            report_str = analyzer.generate_report(properties)
            return report_str
        except Exception as e:
            # import traceback # Already imported at top of chem_copilot_autogen_main.py
            # traceback.print_exc() # Can be noisy, good for deep debugging
            print(f"Error within get_full_chemical_report for '{chemical_identifier}': {str(e)}")
            return f"Error generating full chemical report for '{chemical_identifier}': {str(e)}"
    else:
        return "ChemicalAnalysisAgent tool is not available (failed to initialize)."