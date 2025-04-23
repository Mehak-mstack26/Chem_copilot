from tools.wrapper import (
    get_funcgroups_tool,
    get_name2smiles_tool,
    get_smiles2name_tool,
    get_bond_analyzer_tool, 
    get_retrosynthesis_tool,
    get_visualizer_tool,
    get_reaction_classifier_tool
)

def make_tools(llm=None, api_keys=None, local_rxn=False, verbose=False):
    tools = []
    
    # Add RetroSynthesis tool
    tools.append(get_retrosynthesis_tool())

    # Add FuncGroups tool
    tools.append(get_funcgroups_tool())
    
    # Add NameToSMILES tool
    tools.append(get_name2smiles_tool())
    
    # Add SMILES2Name tool
    tools.append(get_smiles2name_tool())
    
    # Add BondChangeAnalyzer tool
    tools.append(get_bond_analyzer_tool())
    
    # Add ChemVisualizer tool
    tools.append(get_visualizer_tool())

    # Add ReactionClassifier tool
    tools.append(get_reaction_classifier_tool())
    
    return tools