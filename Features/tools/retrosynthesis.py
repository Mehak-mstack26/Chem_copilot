# from langchain.tools import BaseTool
# import requests
# import re
# import os
# import subprocess
# import urllib.parse
# from tools.name2smiles import NameToSMILES  # Update this with the correct import path

# # Initialize NameToSMILES tool for chemical name conversion
# name_to_smiles_tool = NameToSMILES()

# def get_smiles_for_compound(compound_name):
#     """Get SMILES notation for a compound name using the NameToSMILES tool"""
#     try:
#         result = name_to_smiles_tool._run(compound_name)
#         if "SMILES:" in result:
#             # Extract SMILES from the result
#             smiles = result.split("SMILES:")[1].split("\n")[0].strip()
#             return smiles
#         return None
#     except Exception:
#         return None

# def run_retrosynthesis(compound_name):
#     """
#     Run the RetroSynthesis Agent for a given compound and process the results
#     to ensure all reactions have proper SMILES
#     """
#     try:
#         # Get the path to the RetroSynthesisAgent directory
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         parent_dir = os.path.dirname(current_dir)  # Get out of 'tools'
#         project_root = os.path.dirname(parent_dir)  # Get out of 'Features'
#         retro_agent_dir = os.path.join(project_root, "RetroSynthesisAgent")
        
#         # Save current directory
#         original_dir = os.getcwd()
        
#         # Change to RetroSynthesisAgent directory
#         os.chdir(retro_agent_dir)
        
#         # Run the shell script
#         result = subprocess.run(
#             ["/bin/sh", "runRetroSynAgent.sh", compound_name],
#             capture_output=True,
#             text=True
#         )
        
#         # Change back to original directory
#         os.chdir(original_dir)
        
#         # Parse the output
#         output = result.stdout
        
#         # Extract the recommended pathway section
#         recommended_match = re.search(r"Recommended Reaction Pathway:(.*?)Reasons:", output, re.DOTALL)
#         if not recommended_match:
#             return {"status": "error", "message": "Could not find recommended pathway in output"}
        
#         pathway_text = recommended_match.group(1).strip()
#         indices_match = re.search(r"Recommended Reaction Pathway: (.*?)$", output, re.MULTILINE)
#         indices = indices_match.group(1).strip() if indices_match else ""
#         recommended_indices = re.findall(r"idx(\d+)", indices)
        
#         # Extract all reactions
#         reactions = []
#         reaction_blocks = re.findall(r"Reaction idx: (\d+)\s+Reactants: (.*?)\s+Products: (.*?)\s+Reaction SMILES: (.*?)\s+Conditions: (.*?)\s+Source: (.*?)(?:\s+SourceLink: (.*?))?(?=\s+\s+Reaction|\Z)", output, re.DOTALL)
        
#         for block in reaction_blocks:
#             idx, reactants_str, products_str, smiles, conditions, source = block[:6]
#             source_link = block[6] if len(block) > 6 else None
            
#             # Parse reactants and products
#             reactants = [r.strip() for r in reactants_str.split(",")]
#             products = [p.strip() for p in products_str.split(",")]
            
#             # Clean up the reaction SMILES - remove placeholders if present
#             original_smiles = smiles.strip()
#             cleaned_smiles = original_smiles.replace("[reactant_SMILES]", "").replace("[product_SMILES]", "")
#             cleaned_smiles = re.sub(r'\[[a-zA-Z_]+_SMILES\]', "", cleaned_smiles)
            
#             # Check if SMILES is valid or needs generation
#             valid_smiles = cleaned_smiles if ">>" in cleaned_smiles and not re.search(r'\[.*?\]', cleaned_smiles) else None
            
#             # If no valid SMILES, generate from reactants and products
#             if not valid_smiles:
#                 # First try to get SMILES for each reactant and product
#                 reactant_smiles = []
#                 for reactant in reactants:
#                     smiles = get_smiles_for_compound(reactant)
#                     if smiles:
#                         reactant_smiles.append(smiles)
#                     else:
#                         # Keep original name if conversion fails
#                         reactant_smiles.append(reactant)
                
#                 product_smiles = []
#                 for product in products:
#                     smiles = get_smiles_for_compound(product)
#                     if smiles:
#                         product_smiles.append(smiles)
#                     else:
#                         # Keep original name if conversion fails
#                         product_smiles.append(product)
                
#                 # Create reaction SMILES from reactants and products
#                 reactants_str = '.'.join(reactant_smiles)
#                 products_str = '.'.join(product_smiles)
#                 generated_smiles = f"{reactants_str}>>{products_str}"
                
#                 # Use the generated SMILES if it looks reasonable
#                 if ">>" in generated_smiles:
#                     valid_smiles = generated_smiles
            
#             reactions.append({
#                 "idx": idx,
#                 "reactants": reactants,
#                 "products": products,
#                 "reaction_smiles": original_smiles,
#                 "cleaned_reaction_smiles": valid_smiles,
#                 "conditions": conditions.strip(),
#                 "source": source.strip(),
#                 "source_link": source_link.strip() if source_link else None
#             })
        
#         # Extract reasoning
#         reasoning_match = re.search(r"Reasons:\s+(.*?)(?:={10,}|\Z)", output, re.DOTALL)
#         reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        
#         return {
#             "status": "success",
#             "data": {
#                 "reactions": reactions,
#                 "reasoning": reasoning,
#                 "recommended_indices": recommended_indices
#             }
#         }
    
#     except Exception as e:
#         import traceback
#         return {
#             "status": "error", 
#             "message": str(e),
#             "traceback": traceback.format_exc()
#         }




# from langchain.tools import BaseTool
# import requests
# import re
# import os
# import subprocess
# import urllib.parse
# from tools.name2smiles import NameToSMILES  # Update this with the correct import path

# # Initialize NameToSMILES tool for chemical name conversion
# name_to_smiles_tool = NameToSMILES()

# def get_smiles_for_compound(compound_name):
#     """Get SMILES notation for a compound name using the NameToSMILES tool"""
#     try:
#         result = name_to_smiles_tool._run(compound_name)
#         if "SMILES:" in result:
#             # Extract SMILES from the result
#             smiles = result.split("SMILES:")[1].split("\n")[0].strip()
#             return smiles
#         return None
#     except Exception:
#         return None

# def run_retrosynthesis(compound_name):
#     """
#     Run the RetroSynthesis Agent for a given compound and process the results
#     to ensure all reactions have proper SMILES
#     """
#     try:
#         # Get the path to the RetroSynthesisAgent directory
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         parent_dir = os.path.dirname(current_dir)  # Get out of 'tools'
#         project_root = os.path.dirname(parent_dir)  # Get out of 'Features'
#         retro_agent_dir = os.path.join(project_root, "RetroSynthesisAgent")
        
#         # Save current directory
#         original_dir = os.getcwd()
        
#         # Change to RetroSynthesisAgent directory
#         os.chdir(retro_agent_dir)
        
#         # Run the shell script
#         result = subprocess.run(
#             ["/bin/sh", "runRetroSynAgent.sh", compound_name],
#             capture_output=True,
#             text=True
#         )
        
#         # Change back to original directory
#         os.chdir(original_dir)
        
#         # Parse the output
#         output = result.stdout
        
#         # Extract the recommended pathway section
#         recommended_match = re.search(r"Recommended Reaction Pathway:(.*?)Reasons:", output, re.DOTALL)
#         if not recommended_match:
#             return {"status": "error", "message": "Could not find recommended pathway in output"}
        
#         pathway_text = recommended_match.group(1).strip()
#         indices_match = re.search(r"Recommended Reaction Pathway: (.*?)$", output, re.MULTILINE)
#         indices = indices_match.group(1).strip() if indices_match else ""
#         recommended_indices = re.findall(r"idx(\d+)", indices)
        
#         # Extract all reactions
#         reactions = []
#         reaction_blocks = re.findall(r"Reaction idx: (\d+)\s+Reactants: (.*?)\s+Products: (.*?)\s+Reaction SMILES: (.*?)\s+Conditions: (.*?)\s+Source: (.*?)(?:\s+SourceLink: (.*?))?(?=\s+\s+Reaction|\Z)", output, re.DOTALL)
        
#         for block in reaction_blocks:
#             idx, reactants_str, products_str, smiles, conditions, source = block[:6]
#             source_link = block[6] if len(block) > 6 else None
            
#             # Parse reactants and products
#             reactants = [r.strip() for r in reactants_str.split(",")]
#             products = [p.strip() for p in products_str.split(",")]
            
#             # Clean up the reaction SMILES - remove placeholders if present
#             original_smiles = smiles.strip()
#             cleaned_smiles = original_smiles.replace("[reactant_SMILES]", "").replace("[product_SMILES]", "")
#             cleaned_smiles = re.sub(r'\[[a-zA-Z_]+_SMILES\]', "", cleaned_smiles)
            
#             # Check if SMILES is valid or needs generation
#             valid_smiles = cleaned_smiles if ">>" in cleaned_smiles and not re.search(r'\[.*?\]', cleaned_smiles) else None
            
#             # If no valid SMILES, generate from reactants and products
#             if not valid_smiles:
#                 # First try to get SMILES for each reactant and product
#                 reactant_smiles = []
#                 for reactant in reactants:
#                     smiles = get_smiles_for_compound(reactant)
#                     if smiles:
#                         reactant_smiles.append(smiles)
#                     else:
#                         # Keep original name if conversion fails
#                         reactant_smiles.append(reactant)
                
#                 product_smiles = []
#                 for product in products:
#                     smiles = get_smiles_for_compound(product)
#                     if smiles:
#                         product_smiles.append(smiles)
#                     else:
#                         # Keep original name if conversion fails
#                         product_smiles.append(product)
                
#                 # Create reaction SMILES from reactants and products
#                 reactants_str = '.'.join(reactant_smiles)
#                 products_str = '.'.join(product_smiles)
#                 generated_smiles = f"{reactants_str}>>{products_str}"
                
#                 # Use the generated SMILES if it looks reasonable
#                 if ">>" in generated_smiles:
#                     valid_smiles = generated_smiles
            
#             reactions.append({
#                 "idx": idx,
#                 "reactants": reactants,
#                 "products": products,
#                 "reaction_smiles": original_smiles,
#                 "cleaned_reaction_smiles": valid_smiles,
#                 "conditions": conditions.strip(),
#                 "source": source.strip(),
#                 "source_link": source_link.strip() if source_link else None
#             })
        
#         # Extract reasoning
#         reasoning_match = re.search(r"Reasons:\s+(.*?)(?:={10,}|\Z)", output, re.DOTALL)
#         reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        
#         return {
#             "status": "success",
#             "data": {
#                 "reactions": reactions,
#                 "reasoning": reasoning,
#                 "recommended_indices": recommended_indices
#             }
#         }
    
#     except Exception as e:
#         import traceback
#         return {
#             "status": "error", 
#             "message": str(e),
#             "traceback": traceback.format_exc()
#         }



from langchain.tools import BaseTool
from typing import Optional, Dict, Any, List, Union
import sys
import os
import traceback
import re
import requests
import json

# Add RetroSynthesisAgent's parent directory to sys.path to resolve RetroSynAgent imports
retrosyn_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_root = os.path.dirname(retrosyn_dir)
retrosyn_agent_dir = os.path.join(project_root, 'RetroSynthesisAgent')
sys.path.append(retrosyn_agent_dir)
sys.path.append(project_root)  # Add project root too

# Import name2smiles tool
from tools.name2smiles import NameToSMILES

# Import necessary modules from RetroSynAgent directly
# We'll define our own main function instead of importing main
from RetroSynAgent.treeBuilder import Tree, TreeLoader, CommonSubstanceDB
from RetroSynAgent.pdfProcessor import PDFProcessor
from RetroSynAgent.knowledgeGraph import KnowledgeGraph
from RetroSynAgent import prompts
from RetroSynAgent.GPTAPI import GPTAPI
from RetroSynAgent.pdfDownloader import PDFDownloader
from RetroSynAgent.entityAlignment import EntityAlignment
from RetroSynAgent.treeExpansion import TreeExpansion
from RetroSynAgent.reactionsFiltration import ReactionsFiltration

# Save original method to restore later if needed
original_read_data_from_json = CommonSubstanceDB.read_data_from_json

# Define a patched version of the method
def patched_read_data_from_json(self, filename):
    """Patched method to read data from JSON with correct path resolution"""
    # Map common filenames to their absolute paths
    file_map = {
        'RetroSynAgent/emol.json': os.path.join(retrosyn_agent_dir, 'emol.json'),
        'RetroSynAgent/common_chemicals.json': os.path.join(retrosyn_agent_dir, 'common_chemicals.json'),
        'emol.json': os.path.join(retrosyn_agent_dir, 'emol.json'),
        'common_chemicals.json': os.path.join(retrosyn_agent_dir, 'common_chemicals.json'),
        # Add other JSON files as needed
    }
    
    # Use the mapped path if available, otherwise use the original path
    actual_path = file_map.get(filename, filename)
    
    try:
        print(f"Attempting to read JSON from: {actual_path}")
        with open(actual_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File not found: {actual_path}")
        # Try alternate locations
        alt_paths = [
            os.path.join(retrosyn_agent_dir, os.path.basename(filename)),
            os.path.join(retrosyn_agent_dir, 'RetroSynAgent', os.path.basename(filename)),
            os.path.join(project_root, 'RetroSynthesisAgent', 'RetroSynAgent', os.path.basename(filename)),
            os.path.join(project_root, os.path.basename(filename))
        ]
        
        for alt_path in alt_paths:
            print(f"Trying alternate path: {alt_path}")
            try:
                with open(alt_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                print(f"Successfully loaded from: {alt_path}")
                return data
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"Error reading {alt_path}: {str(e)}")
                continue
        
        print(f"Could not find file {filename} in any location")
        return {}
    except Exception as e:
        print(f"Error reading {actual_path}: {str(e)}")
        return {}

# Apply the monkey patch
CommonSubstanceDB.read_data_from_json = patched_read_data_from_json

# Initialize NameToSMILES tool for chemical name conversion
name_to_smiles_tool = NameToSMILES()

def get_smiles_for_compound(compound_name):
    """Get SMILES notation for a compound name using the NameToSMILES tool"""
    try:
        result = name_to_smiles_tool._run(compound_name)
        if "SMILES:" in result:
            # Extract SMILES from the result
            smiles = result.split("SMILES:")[1].split("\n")[0].strip()
            return smiles
        return None
    except Exception:
        return None

def parse_reaction_data(raw_text):
    """Parse the reaction data from the raw text output"""
    # 1. Extract recommended pathway
    rec_match = re.search(r"Recommended Reaction Pathway:\s*([^\n]+)", raw_text)
    recommended = [idx.strip() for idx in rec_match.group(1).split(",")] if rec_match else []

    # 2. Extract the Reasons block (everything after "Reasons:")
    reasons = ""
    reasons_match = re.search(r"Reasons:\s*((?:.|\n)*)", raw_text)
    if reasons_match:
        reasons = reasons_match.group(1).strip()

    # 3. Split into individual reaction blocks
    blocks = re.split(r"(?=Reaction idx:)", raw_text)
    reactions = []
    for blk in blocks:
        if not blk.strip().startswith("Reaction idx:"):
            continue

        idx_match    = re.search(r"Reaction idx:\s*(\S+)", blk)
        react_match  = re.search(r"Reactants:\s*(.+)", blk)
        prod_match   = re.search(r"Products:\s*(.+)", blk)
        smile_match  = re.search(r"Reaction SMILES:\s*(\S+)", blk)
        cond_match   = re.search(r"Conditions:\s*(.+)", blk)
        source_match = re.search(r"Source:\s*(.+)", blk)
        link_match   = re.search(r"SourceLink:\s*\[?(.+?)\]?(?:\s|$)", blk)

        reaction = {
            "idx": idx_match.group(1) if idx_match else None,
            "reactants": [r.strip() for r in react_match.group(1).split(",")] if react_match else [],
            "products": [p.strip() for p in prod_match.group(1).split(",")]  if prod_match else [],
            "reaction_smiles": smile_match.group(1) if smile_match else None,
            "conditions": {},
            "source": source_match.group(1).strip() if source_match else None,
            "source_link": link_match.group(1).strip() if link_match else None
        }

        # parse conditions into key/value pairs
        if cond_match:
            for part in cond_match.group(1).split(","):
                if ":" in part:
                    key, val = part.split(":", 1)
                    reaction["conditions"][key.strip().lower()] = val.strip()

        reactions.append(reaction)

    return {
        "recommended_pathway": recommended,
        "reactions": reactions,
        "reasons": reasons
    }

# Function to run retrosynthesis with parameters (copied from main.py)
def retrosyn_main(material=None, num_results=10, alignment=True, expansion=True, filtration=False):
    """
    Main function to run the retrosynthesis process.
    Can be called directly with parameters or through command-line arguments.
    Returns the recommendation text that can be parsed by the UI.
    """
    # Define folder names with proper paths
    pdf_folder_name = os.path.join(retrosyn_agent_dir, 'pdf_pi')
    result_folder_name = os.path.join(retrosyn_agent_dir, 'res_pi')
    result_json_name = 'llm_res'
    tree_folder_name = os.path.join(retrosyn_agent_dir, 'tree_pi')
    
    # Create directories if they don't exist
    os.makedirs(pdf_folder_name, exist_ok=True)
    os.makedirs(result_folder_name, exist_ok=True)
    os.makedirs(tree_folder_name, exist_ok=True)
    
    # Initialize required objects
    entityalignment = EntityAlignment()
    treeloader = TreeLoader()
    tree_expansion = TreeExpansion()
    reactions_filtration = ReactionsFiltration()

    ### extractInfos

    # 1  query literatures & download
    downloader = PDFDownloader(material, pdf_folder_name=pdf_folder_name, num_results=num_results, n_thread=3)
    pdf_name_list = downloader.main()
    print(f'successfully downloaded {len(pdf_name_list)} pdfs for {material}')

    # 2 Extract infos from PDF about reactions
    pdf_processor = PDFProcessor(pdf_folder_name=pdf_folder_name, result_folder_name=result_folder_name,
                                 result_json_name=result_json_name)
    pdf_processor.load_existing_results()
    pdf_processor.process_pdfs_txt(save_batch_size=2)

    ### treeBuildWOExapnsion
    results_dict = entityalignment.alignRootNode(result_folder_name, result_json_name, material)

    # 4 construct kg & tree
    tree_name_wo_exp = os.path.join(tree_folder_name, f'{material}_wo_exp.pkl')
    if not os.path.exists(tree_name_wo_exp):
        tree_wo_exp = Tree(material.lower(), result_dict=results_dict)
        print('Starting to construct RetroSynthetic Tree...')
        tree_wo_exp.construct_tree()
        treeloader.save_tree(tree_wo_exp, tree_name_wo_exp)
    else:
        tree_wo_exp = treeloader.load_tree(tree_name_wo_exp)
        print('RetroSynthetic Tree wo expansion already loaded.')
    node_count_wo_exp = countNodes(tree_wo_exp)
    all_path_wo_exp = searchPathways(tree_wo_exp)
    print(f'The tree contains {node_count_wo_exp} nodes and {len(all_path_wo_exp)} pathways before expansion.')

    if alignment:
        print('Starting to align the nodes of RetroSynthetic Tree...')

        ### WO Expansion
        tree_name_wo_exp_alg = os.path.join(tree_folder_name, f'{material}_wo_exp_alg.pkl')
        if not os.path.exists(tree_name_wo_exp_alg):
            reactions_wo_exp = tree_wo_exp.reactions
            reactions_wo_exp_alg_1 = entityalignment.entityAlignment_1(reactions_dict=reactions_wo_exp)
            reactions_wo_exp_alg_all = entityalignment.entityAlignment_2(reactions_dict=reactions_wo_exp_alg_1)
            tree_wo_exp_alg = Tree(material.lower(), reactions=reactions_wo_exp_alg_all)
            tree_wo_exp_alg.construct_tree()
            treeloader.save_tree(tree_wo_exp_alg, tree_name_wo_exp_alg)
        else:
            tree_wo_exp_alg = treeloader.load_tree(tree_name_wo_exp_alg)
            print('aligned RetroSynthetic Tree wo expansion already loaded.')
        node_count_wo_exp_alg = countNodes(tree_wo_exp_alg)
        all_path_wo_exp_alg = searchPathways(tree_wo_exp_alg)
        print(f'The aligned tree contains {node_count_wo_exp_alg} nodes and {len(all_path_wo_exp_alg)} pathways before expansion.')

    ## treeExpansion
    # 5 kg & tree expansion
    results_dict_additional = tree_expansion.treeExpansion(result_folder_name, result_json_name,
                                                           results_dict, material, expansion=expansion, max_iter=5)
    if results_dict_additional:
        results_dict = tree_expansion.update_dict(results_dict, results_dict_additional)

    tree_name_exp = os.path.join(tree_folder_name, f'{material}_w_exp.pkl')
    if not os.path.exists(tree_name_exp):
        tree_exp = Tree(material.lower(), result_dict=results_dict)
        print('Starting to construct Expanded RetroSynthetic Tree...')
        tree_exp.construct_tree()
        treeloader.save_tree(tree_exp, tree_name_exp)
    else:
        tree_exp = treeloader.load_tree(tree_name_exp)
        print('RetroSynthetic Tree w expansion already loaded.')

    # nodes & pathway count (tree w exp)
    node_count_exp = countNodes(tree_exp)
    all_path_exp = searchPathways(tree_exp)
    print(f'The tree contains {node_count_exp} nodes and {len(all_path_exp)} pathways after expansion.')

    if alignment:
        ### Expansion
        tree_name_exp_alg = os.path.join(tree_folder_name, f'{material}_w_exp_alg.pkl')
        if not os.path.exists(tree_name_exp_alg):
            reactions_exp = tree_exp.reactions
            reactions_exp_alg_1 = entityalignment.entityAlignment_1(reactions_dict=reactions_exp)
            reactions_exp_alg_all = entityalignment.entityAlignment_2(reactions_dict=reactions_exp_alg_1)
            tree_exp_alg = Tree(material.lower(), reactions=reactions_exp_alg_all)
            tree_exp_alg.construct_tree()
            treeloader.save_tree(tree_exp_alg, tree_name_exp_alg)
        else:
            tree_exp_alg = treeloader.load_tree(tree_name_exp_alg)
            print('aligned RetroSynthetic Tree wo expansion already loaded.')
        node_count_exp_alg = countNodes(tree_exp_alg)
        all_path_exp_alg = searchPathways(tree_exp_alg)
        print(f'The aligned tree contains {node_count_exp_alg} nodes and {len(all_path_exp_alg)} pathways after expansion.')
        tree_exp = tree_exp_alg

    all_pathways_w_reactions = reactions_filtration.getFullReactionPathways(tree_exp)

    ## Filtration
    if filtration:
        # filter reactions based on conditions
        reactions_txt_filtered = reactions_filtration.filterReactions(tree_exp)
        # build & save tree
        tree_name_filtered = os.path.join(tree_folder_name, f'{material}_filtered.pkl')
        if not os.path.exists(tree_name_filtered):
            print('Starting to construct Filtered RetroSynthetic Tree...')
            tree_filtered = Tree(material.lower(), reactions_txt=reactions_txt_filtered)
            tree_filtered.construct_tree()
            treeloader.save_tree(tree_filtered, tree_name_filtered)
        else:
            tree_filtered = treeloader.load_tree(tree_name_filtered)
            print('Filtered RetroSynthetic Tree already loaded.')
        node_count_filtered = countNodes(tree_filtered)
        all_path_filtered = searchPathways(tree_filtered)
        print(f'The tree contains {node_count_filtered} nodes and {len(all_path_filtered)} pathways after filtration.')

        # filter invalid pathways
        filtered_pathways = reactions_filtration.filterPathways(tree_filtered)
        all_pathways_w_reactions = filtered_pathways

    ### Recommendation
    # recommend based on specific criterion
    prompt_recommend1 = prompts.recommend_prompt_commercial.format(all_pathways = all_pathways_w_reactions)
    recommend1_reactions_txt = recommendReactions(prompt_recommend1, result_folder_name, response_name='recommend_pathway1')
    
    return recommend1_reactions_txt 

# Helper functions from main.py
def countNodes(tree):
    node_count = tree.get_node_count()
    return node_count

def searchPathways(tree):
    all_path = tree.find_all_paths()
    return all_path

def recommendReactions(prompt, result_folder_name, response_name):
    res = GPTAPI().answer_wo_vision(prompt)
    result_file_path = os.path.join(result_folder_name, f'{response_name}.txt')
    os.makedirs(os.path.dirname(result_file_path), exist_ok=True)
    with open(result_file_path, 'w') as f:
        f.write(res)
    start_idx = res.find("Recommended Reaction Pathway:")
    recommend_reactions_txt = res[start_idx:] if start_idx >= 0 else res
    print(f'\n=================================================='
          f'==========\n{recommend_reactions_txt}\n====================='
          f'=======================================\n')
    return recommend_reactions_txt

# Use with parameters
def run_with_params(material, num_results=10, alignment=True, expansion=True, filtration=False):
    """Run the retrosynthesis with the given parameters and return the result"""
    try:
        # Call our local implementation of main()
        output = retrosyn_main(
            material=material,
            num_results=num_results,
            alignment=alignment,
            expansion=expansion,
            filtration=filtration
        )
        return output
    except Exception as e:
        error_msg = f"Error running retrosynthesis: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise Exception(f"Error running retrosynthesis: {str(e)}")

class RetroSynthesisTool(BaseTool):
    name: str = "retrosynthesis"
    description: str = "Performs retrosynthesis analysis on a chemical compound"
    
    def _run(self, compound_name: str) -> Dict[str, Any]:
        """Run retrosynthesis analysis on the given compound"""
        try:
            # Call the main function directly with parameters
            raw_output = run_with_params(
                material=compound_name,
                num_results=10,
                alignment=True,
                expansion=True,
                filtration=False
            )
            
            # Parse the output
            parsed_data = parse_reaction_data(raw_output)
            
            # Format for better readability
            reactions = parsed_data["reactions"]
            recommended_indices = parsed_data["recommended_pathway"]
            reasoning = parsed_data["reasons"]
            
            # Process each reaction to ensure proper SMILES
            for reaction in reactions:
                # Clean up the reaction SMILES - remove placeholders if present
                original_smiles = reaction["reaction_smiles"]
                if original_smiles:
                    cleaned_smiles = original_smiles.replace("[reactant_SMILES]", "").replace("[product_SMILES]", "")
                    cleaned_smiles = re.sub(r'\[[a-zA-Z_]+_SMILES\]', "", cleaned_smiles)
                    
                    # Check if SMILES is valid or needs generation
                    valid_smiles = cleaned_smiles if ">>" in cleaned_smiles and not re.search(r'\[.*?\]', cleaned_smiles) else None
                    
                    # If no valid SMILES, generate from reactants and products
                    if not valid_smiles:
                        # Get SMILES for each reactant and product
                        reactant_smiles = []
                        for reactant in reaction["reactants"]:
                            smiles = get_smiles_for_compound(reactant)
                            if smiles:
                                reactant_smiles.append(smiles)
                            else:
                                # Keep original name if conversion fails
                                reactant_smiles.append(reactant)
                        
                        product_smiles = []
                        for product in reaction["products"]:
                            smiles = get_smiles_for_compound(product)
                            if smiles:
                                product_smiles.append(smiles)
                            else:
                                # Keep original name if conversion fails
                                product_smiles.append(product)
                        
                        # Create reaction SMILES from reactants and products
                        reactants_str = '.'.join(reactant_smiles)
                        products_str = '.'.join(product_smiles)
                        generated_smiles = f"{reactants_str}>>{products_str}"
                        
                        # Use the generated SMILES if it looks reasonable
                        if ">>" in generated_smiles:
                            valid_smiles = generated_smiles
                    
                    reaction["cleaned_reaction_smiles"] = valid_smiles
            
            return {
                "status": "success",
                "data": {
                    "reactions": reactions,
                    "reasoning": reasoning,
                    "recommended_indices": recommended_indices
                }
            }
        
        except Exception as e:
            error_msg = f"Error in RetroSynthesisTool: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return {
                "status": "error", 
                "message": str(e),
                "traceback": traceback.format_exc()
            }

# Function to use for direct calling (without using as a LangChain tool)
def run_retrosynthesis(compound_name):
    """
    Run the RetroSynthesis Agent for a given compound and process the results
    """
    tool = RetroSynthesisTool()
    return tool._run(compound_name)