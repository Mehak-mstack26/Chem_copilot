from langchain.tools import Tool
from tools.retrosynthesis import run_retrosynthesis
from tools.funcgroups import FuncGroups
from tools.name2smiles import NameToSMILES
from tools.smiles2name import SMILES2Name
from tools.visualizer import ChemVisualizer
from tools.bond import BondChangeAnalyzer  
from tools.asckos import ReactionClassifier

def get_retrosynthesis_tool():
    return Tool(
        name="RetroSynthesis",
        description=(
            "Use this tool to search for retrosynthesis pathways for a chemical compound. "
            "Provide a compound name (IUPAC or common name) as input. "
            "Returns a list of reaction steps, conditions, and SMILES notation."
        ),
        func=run_retrosynthesis,  # Pass the function directly, not calling it
    )


def get_funcgroups_tool():
    tool = FuncGroups()
    return Tool(
        name="FuncGroups",
        description=(
            "Use this tool to identify functional groups in a molecule or reaction. "
            "Provide a valid SMILES or reaction SMILES string as input. "
            "Returns functional group names and handles reactants/products."
        ),
        func=tool._run,
    )

def get_name2smiles_tool():
    tool = NameToSMILES()
    return Tool(
        name="NameToSMILES",
        description=(
            "Use this tool to convert a compound/molecule/reaction name to a SMILES string. "
            "Provide a compound name such as 'aspirin', 'benzene', or 'acetaminophen'."
        ),
        func=tool._run,
    )

def get_smiles2name_tool():
    tool = SMILES2Name()
    return Tool(
        name="SMILES2Name",
        description=(
            "Use this tool to convert a SMILES string to a chemical name. "
            "It first finds the IUPAC name using CACTUS, then uses GPT to return the common/trivial name. "
        ),
        func=tool._run,
    )

def get_bond_analyzer_tool():
    tool = BondChangeAnalyzer()
    return Tool(
        name="BondChangeAnalyzer",
        description=(
            "Use this tool to identify bonds broken, formed, and changed in a chemical reaction. "
            "Provide a reaction SMILES string as input. "
            "Returns lists of broken bonds, formed bonds, and bonds that changed type (e.g., single to double)."
        ),
        func=tool._run,
    )

def get_visualizer_tool():
    tool = ChemVisualizer()
    return Tool(
        name="ChemVisualizer",
        description=(
            "Use this tool to visualize molecules or reactions from SMILES strings. "
            "Provide a SMILES or reaction SMILES string as input. "
            "Returns the path to the generated image file."
        ),
        func=tool._run,
    )

def get_reaction_classifier_tool():
    tool = ReactionClassifier()
    return Tool(
        name="ReactionClassifier",
        description=(
            "Use this tool to classify chemical reaction types based on reaction SMILES and get detailed information. "
            "Provide a valid reaction SMILES string as input. "
            "Returns the most likely reaction type, confidence level, and educational information about the reaction."
        ),
        func=tool._run,
    )
