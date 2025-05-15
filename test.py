# import os
# import re
# import requests # Keep if api_config or other direct calls use it
# try:
#     import api_config
# except ImportError:
#     print("Warning: api_config.py not found. API keys might not be loaded if they are set there.")

# from tools.make_tools import make_tools
# from langchain.agents import AgentExecutor, ZeroShotAgent
# from langchain_openai import ChatOpenAI
# from tools.asckos import ReactionClassifier # Assuming this is correctly set up
# from functools import lru_cache
# import time
# import traceback
# import pandas as pd # Make sure pandas is imported for query_reaction_dataset

# # --- Setup for Saving Analysis Files ---
# try:
#     PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# except NameError:
#     PROJECT_ROOT_DIR = os.getcwd()
# REACTION_ANALYSIS_OUTPUT_DIR = os.path.join(PROJECT_ROOT_DIR, "reaction_analysis_outputs")
# os.makedirs(REACTION_ANALYSIS_OUTPUT_DIR, exist_ok=True)

# def sanitize_filename(name):
#     if not isinstance(name, str): name = str(name)
#     name = re.sub(r'[^\w\.\-]+', '_', name); return name[:100]

# def save_analysis_to_file(reaction_smiles, analysis_text, query_context_type="analysis", original_compound_name=None):
#     if not analysis_text or not isinstance(analysis_text, str) or not analysis_text.strip():
#         print(f"[SAVE_ANALYSIS] Skipping save: No analysis text provided for '{reaction_smiles}'.")
#         return
#     timestamp = time.strftime("%Y%m%d_%H%M%S")
#     smiles_part = sanitize_filename(reaction_smiles if reaction_smiles else "no_smiles")
#     filename_parts = []
#     if original_compound_name and original_compound_name != "DirectReactionAnalysis" and original_compound_name != reaction_smiles:
#         filename_parts.append(sanitize_filename(original_compound_name))
#     filename_parts.append(f"rxn_{smiles_part}")
#     filename_parts.append(sanitize_filename(query_context_type))
#     filename_parts.append(timestamp)
#     filename = "_".join(filter(None, filename_parts)) + ".txt"
#     filepath = os.path.join(REACTION_ANALYSIS_OUTPUT_DIR, filename)
#     try:
#         with open(filepath, "w", encoding="utf-8") as f:
#             f.write(f"Reaction SMILES: {reaction_smiles}\n")
#             if original_compound_name and original_compound_name != reaction_smiles:
#                  f.write(f"Original Target Context: {original_compound_name}\n")
#             f.write(f"Analysis Type: {query_context_type}\n"); f.write(f"Timestamp: {timestamp}\n")
#             f.write("="*50 + "\n\n"); f.write(analysis_text)
#         print(f"[SAVE_ANALYSIS] Saved analysis to: {filepath}")
#     except Exception as e: print(f"[SAVE_ANALYSIS_ERROR] Error saving {filepath}: {e}")

# llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=4000) # General LLM
# tools = make_tools(llm=llm)
# dataset_path1 = os.environ.get('REACTION_DATASET_PATH1', None)
# dataset_path2 = os.environ.get('REACTION_DATASET_PATH2', None)
# # Ensure ReactionClassifier is initialized. If it fails, handle gracefully or ensure paths are valid.
# try:
#     reaction_classifier = ReactionClassifier(dataset_path1, dataset_path2)
# except Exception as e:
#     print(f"Warning: Could not initialize ReactionClassifier: {e}. Classification features will be unavailable.")
#     reaction_classifier = None # Set to None if initialization fails

# reaction_cache = {}

# PREFIX = """
# You are Chem Copilot, an expert chemistry assistant. You have access to the following tools to analyze molecules and chemical reactions.
# Always begin by understanding the user's intent — what kind of information are they asking for?
# Here is how to choose tools:
# - If the user gives a SMILES or reaction SMILES and asks for the name, you MUST use SMILES2Name. Do NOT analyze bonds or functional groups for this task.
# - Use NameToSMILES: when the user gives a compound/reaction name and wants the SMILES or structure.
# - Use FuncGroups: when the user wants to analyze functional groups in a molecule or a reaction (input is SMILES or reaction SMILES).
# - Use BondChangeAnalyzer: when the user asks for which bonds are broken, formed, or changed in a chemical reaction.
# If the user wants all of the above (full analysis of a reaction SMILES), respond with "This requires full analysis." (This will be handled by a separate function.)
# Always return your answer in this format:
# Final Answer: <your answer here>
# For FuncGroups results:
# - Always list the functional groups identified in each reactant and product separately
# - Include the transformation summary showing disappeared groups, appeared groups, and unchanged groups
# - Provide a clear conclusion about what transformation occurred in the reaction
# For BondChangeAnalyzer results:
# - Always list the specific bonds that were broken, formed, or changed with their bond types
# - Include the atom types involved in each bond (e.g., C-O, N-H)
# - Provide a clear conclusion summarizing the key bond changes in the reaction
# """
# FORMAT_INSTRUCTIONS = """
# You can only respond with a single complete
# "Thought, Action, Action Input" format
# OR a single "Final Answer" format
# Complete format:
# Thought: (reflect on your progress and decide what to do next)
# Action: (the action name, should be one of [{tool_names}])
# Action Input: (the input string to the action)
# OR
# Final Answer: (the final answer to the original input question)
# """
# SUFFIX = """
# Question: {input}
# {agent_scratchpad}
# """
# prompt = ZeroShotAgent.create_prompt(
#     tools=tools, prefix=PREFIX, suffix=SUFFIX,
#     format_instructions=FORMAT_INSTRUCTIONS, input_variables=["input", "agent_scratchpad"]
# )
# agent_chain = ZeroShotAgent.from_llm_and_tools(llm=llm, tools=tools, prompt=prompt)
# agent = AgentExecutor(agent=agent_chain, tools=tools, verbose=True, handle_parsing_errors=True)

# def extract_final_answer(full_output: str):
#     match = re.search(r"Final Answer:\s*(.*)", full_output, re.DOTALL)
#     return match.group(1).strip() if match else full_output.strip()

# @lru_cache(maxsize=100)
# def query_reaction_dataset(reaction_smiles):
#     if not reaction_smiles: return None
#     if reaction_smiles in reaction_cache and 'dataset_info' in reaction_cache[reaction_smiles]:
#         return reaction_cache[reaction_smiles]['dataset_info']
    
#     # Ensure reaction_classifier and its datasets are initialized
#     if not reaction_classifier or \
#        (not hasattr(reaction_classifier, 'dataset1') or reaction_classifier.dataset1 is None or reaction_classifier.dataset1.empty) and \
#        (not hasattr(reaction_classifier, 'dataset2') or reaction_classifier.dataset2 is None or reaction_classifier.dataset2.empty):
#         # print(f"[QUERY_DATASET] ReactionClassifier or its datasets not available for {reaction_smiles}")
#         return None # Cannot query if classifier/datasets are not ready

#     try:
#         df = None
#         if hasattr(reaction_classifier, 'dataset1') and reaction_classifier.dataset1 is not None and not reaction_classifier.dataset1.empty:
#             df = reaction_classifier.dataset1
#         elif hasattr(reaction_classifier, 'dataset2') and reaction_classifier.dataset2 is not None and not reaction_classifier.dataset2.empty:
#             df = reaction_classifier.dataset2
        
#         if df is None or df.empty:
#             # print(f"[QUERY_DATASET] No dataset loaded or dataset is empty for {reaction_smiles}")
#             return None

#         fields_to_extract = ['procedure_details', 'rxn_time', 'temperature', 'yield_000', 'reaction_name', 'reaction_classname', 'prediction_certainty']
#         smiles_columns = ['rxn_str', 'reaction_smiles', 'smiles', 'rxn_smiles']
#         exact_match = None
#         for col in smiles_columns:
#             if col in df.columns:
#                 if df[col].dtype == 'object':
#                     temp_match = df[df[col] == reaction_smiles]
#                     if not temp_match.empty:
#                         exact_match = temp_match
#                         break
        
#         result = {}
#         if exact_match is not None and not exact_match.empty:
#             row = exact_match.iloc[0]
#             for field in fields_to_extract:
#                 # Check for pd.notna and ensure value is not just 'nan' string if it's a string
#                 if field in row.index and pd.notna(row[field]) and (not isinstance(row[field], str) or str(row[field]).strip().lower() != "nan"):
#                     result[field] = str(row[field]) # Convert to string for consistency
#             for i in range(1, 11):
#                 key = f'solvent_{i:03d}'
#                 if key in row.index and pd.notna(row[key]) and (not isinstance(row[key], str) or str(row[key]).strip().lower() != "nan"):
#                     result.setdefault('solvents_list', []).append(str(row[key]))
#                     if len(result.get('solvents_list', [])) >= 3: break
#             for i in range(1, 16):
#                 key = f'agent_{i:03d}'
#                 if key in row.index and pd.notna(row[key]) and (not isinstance(row[key], str) or str(row[key]).strip().lower() != "nan"):
#                     result.setdefault('agents_list', []).append(str(row[key]))
#                     if len(result.get('agents_list', [])) >= 3: break
        
#         reaction_cache.setdefault(reaction_smiles, {})['dataset_info'] = result if result else None
#         return result if result else None
#     except Exception as e: print(f"Error querying dataset for '{reaction_smiles}': {e}"); return None

# def extract_reaction_smiles(query: str):
#     explicit_pattern_gg = r"(?:reaction\s+step|rxn|reaction|SMILES)\s*[:=]?\s*([\w@\[\]\.\+\-\=\#\:\(\)\\\/;\$\%\|\{\}\s~]+>>[\w@\[\]\.\+\-\=\#\:\(\)\\\/;\$\%\|\{\}\s~]*)"
#     match = re.search(explicit_pattern_gg, query, re.IGNORECASE)
#     if match:
#         smiles = match.group(1).strip()
#         if ">>" in smiles:
#             parts = smiles.split(">>")
#             if len(parts) == 2 and parts[0].strip() and parts[1].strip():
#                 print(f"[EXTRACT_SMILES] Found by explicit '>>' pattern: '{smiles}'")
#                 return smiles

#     standalone_pattern_gg = r"(?:^|\s|[:=\(\-])([\w@\[\]\.\+\-\=\#\:\(\)\\\/;\$\%\|\{\}\s~]+>>[\w@\[\]\.\+\-\=\#\:\(\)\\\/;\$\%\|\{\}\s~]*)(?:\s|[\.\,\)]|$)"
#     match = re.search(standalone_pattern_gg, query)
#     if match:
#         smiles = match.group(1).strip()
#         if ">>" in smiles and len(smiles) > 3:
#             parts = smiles.split(">>")
#             if len(parts) == 2 and parts[0].strip() and parts[1].strip():
#                 print(f"[EXTRACT_SMILES] Found by standalone '>>' pattern: '{smiles}'")
#                 return smiles
    
#     smi_part_chars = r"[\w@\[\]\+\-\=\#\:\(\)\\\/;\$\%\|\{\}\.]"
#     explicit_pattern_gt = rf"(?:reaction\s+step|rxn|reaction|SMILES)\s*[:=]?\s*({smi_part_chars}+(?:>{smi_part_chars}*)+)"
#     match_gt_explicit = re.search(explicit_pattern_gt, query, re.IGNORECASE)
    
#     extracted_gt_smiles = None
#     if match_gt_explicit:
#         temp_smiles = match_gt_explicit.group(1).strip()
#         if ">>" not in temp_smiles and ">" in temp_smiles :
#             extracted_gt_smiles = temp_smiles
#             print(f"[EXTRACT_SMILES] Found by explicit '>' pattern (pre-conversion): '{extracted_gt_smiles}'")

#     if not extracted_gt_smiles:
#         standalone_pattern_gt = rf"(?:^|\s|[:=\(\-])({smi_part_chars}+(?:>{smi_part_chars}*)+)(?:\s|[\.\,\)]|$)"
#         match_gt_standalone = re.search(standalone_pattern_gt, query)
#         if match_gt_standalone:
#             temp_smiles = match_gt_standalone.group(1).strip()
#             if ">>" not in temp_smiles and ">" in temp_smiles:
#                 extracted_gt_smiles = temp_smiles
#                 print(f"[EXTRACT_SMILES] Found by standalone '>' pattern (pre-conversion): '{extracted_gt_smiles}'")
    
#     if extracted_gt_smiles:
#         parts = extracted_gt_smiles.split('>')
#         cleaned_parts = [p.strip() for p in parts if p.strip()]
#         if len(cleaned_parts) >= 2:
#             products = cleaned_parts[-1]
#             reactants_and_agents_str = ".".join(cleaned_parts[:-1])
#             reactants_and_agents_str = re.sub(r'\.+', '.', reactants_and_agents_str).strip('.')
#             if reactants_and_agents_str and products:
#                 converted_smiles = f"{reactants_and_agents_str}>>{products}"
#                 final_parts_check = converted_smiles.split(">>")
#                 if len(final_parts_check) == 2 and final_parts_check[0].strip() and final_parts_check[1].strip():
#                     print(f"[EXTRACT_SMILES] Converted '>' pattern to '>>': '{converted_smiles}'")
#                     return converted_smiles
#                 else:
#                     print(f"[EXTRACT_SMILES] Conversion of '{extracted_gt_smiles}' to '>>' resulted in invalid structure: '{converted_smiles}'")
#             else:
#                 print(f"[EXTRACT_SMILES] Processing '{extracted_gt_smiles}' after splitting and cleaning led to empty reactants/products part.")
#         else:
#             print(f"[EXTRACT_SMILES] Splitting '{extracted_gt_smiles}' by '>' resulted in less than 2 valid non-empty parts after stripping.")
#     return None

# def handle_full_info(query_text_for_llm_summary, reaction_smiles_clean, original_compound_name=None, callbacks=None):
#     print(f"Running full analysis for CLEAN reaction: {reaction_smiles_clean} (Original query context: '{query_text_for_llm_summary[:100]}...', Saving context: {original_compound_name})\n")

#     if not reaction_smiles_clean or ">>" not in reaction_smiles_clean:
#         print(f"[HANDLE_FULL_INFO_ERROR] Invalid or missing reaction_smiles_clean: '{reaction_smiles_clean}'")
#         return {'visualization_path': None, 'analysis': f"Error: Invalid reaction SMILES provided for analysis: '{reaction_smiles_clean}'", 'analysis_context': "invalid_smiles_input", 'processed_smiles_for_tools': reaction_smiles_clean}

#     if reaction_smiles_clean in reaction_cache and 'full_info' in reaction_cache[reaction_smiles_clean]:
#         cached_data = reaction_cache[reaction_smiles_clean]['full_info']
#         if 'analysis_context' not in cached_data:
#             cached_data['analysis_context'] = "full_llm_summary_cached"
#         cached_data['processed_smiles_for_tools'] = reaction_smiles_clean # Ensure this is part of cached data
#         print(f"Using cached full_info data for reaction: {reaction_smiles_clean}")
#         return cached_data

#     reaction_cache.setdefault(reaction_smiles_clean, {})
#     full_info_results = {}
#     tool_dict = {tool.name.lower(): tool for tool in tools}

#     try:
#         # Visualization
#         visualizer_tool = tool_dict.get("chemvisualizer")
#         if visualizer_tool:
#             try:
#                 visualization_path = visualizer_tool.run(reaction_smiles_clean)
#                 if visualization_path and not str(visualization_path).lower().startswith('error') and str(visualization_path).endswith(".png"):
#                     full_info_results['Visualization'] = visualization_path
#                     reaction_cache[reaction_smiles_clean]['visualization_path'] = visualization_path
#                 else:
#                     full_info_results['Visualization'] = f"Visualization tool message: {visualization_path}"
#                     reaction_cache[reaction_smiles_clean]['visualization_path'] = None
#             except Exception as e:
#                 full_info_results['Visualization'] = f"Error visualizing reaction: {str(e)}"
#                 reaction_cache[reaction_smiles_clean]['visualization_path'] = None
#         else:
#             full_info_results['Visualization'] = "ChemVisualizer tool not found"
#             reaction_cache[reaction_smiles_clean]['visualization_path'] = None

#         for tool_name_lower, data_key, cache_key in [
#             ("smiles2name", "Names", "name_info"),
#             ("funcgroups", "Functional Groups", "fg_info"),
#             ("bondchangeanalyzer", "Bond Changes", "bond_info")
#         ]:
#             tool_instance = tool_dict.get(tool_name_lower)
#             if tool_instance:
#                 try:
#                     tool_result = tool_instance.run(reaction_smiles_clean)
#                     display_result = str(tool_result)[:500] + ("..." if len(str(tool_result)) > 500 else "")
#                     full_info_results[data_key] = display_result
#                     reaction_cache[reaction_smiles_clean][cache_key] = tool_result # Cache full tool output
#                 except Exception as e:
#                     err_msg = f"Error running {tool_name_lower}: {str(e)}"
#                     full_info_results[data_key] = err_msg
#                     reaction_cache[reaction_smiles_clean][cache_key] = err_msg
#             else:
#                 msg = f"{tool_name_lower.capitalize()} tool not found"
#                 full_info_results[data_key] = msg
#                 reaction_cache[reaction_smiles_clean][cache_key] = msg
        
#         if reaction_classifier: # Check if classifier was initialized
#             try:
#                 # Assuming _run is the correct method for ReactionClassifier
#                 classifier_result_raw = reaction_classifier._run(reaction_smiles_clean) 
#                 if isinstance(classifier_result_raw, str):
#                     summary_match = re.search(r'## Summary\n(.*?)(?=\n##|$)', classifier_result_raw, re.DOTALL | re.IGNORECASE)
#                     classifier_summary = summary_match.group(1).strip() if summary_match else (classifier_result_raw.splitlines()[0] if classifier_result_raw.splitlines() else "No summary found")
#                     if len(classifier_summary) > 500: classifier_summary = classifier_summary[:497] + "..."
#                     full_info_results['Reaction Classification'] = classifier_summary
#                     reaction_cache[reaction_smiles_clean]['classification_info'] = classifier_summary # Cache summary for prompt
#                 else: # Should be string, but handle other cases
#                     msg = "Classifier result was not a string as expected."
#                     full_info_results['Reaction Classification'] = msg
#                     reaction_cache[reaction_smiles_clean]['classification_info'] = msg
#             except Exception as e:
#                 err_msg = f"Error classifying reaction: {str(e)}"
#                 full_info_results['Reaction Classification'] = err_msg
#                 reaction_cache[reaction_smiles_clean]['classification_info'] = err_msg
#                 print(f"[HANDLE_FULL_INFO] ReactionClassifier exception: {e}")
#         else:
#             msg = "ReactionClassifier tool not available or not initialized"
#             full_info_results['Reaction Classification'] = msg
#             reaction_cache[reaction_smiles_clean]['classification_info'] = msg
#             print(f"[HANDLE_FULL_INFO] {msg}")


#         dataset_data = query_reaction_dataset(reaction_smiles_clean) 
#         procedure_details, rxn_time, temperature, yield_info, solvents, agents_catalysts = None, None, None, None, None, None # Initialize to None
#         if dataset_data:
#             procedure_details = dataset_data.get('procedure_details')
#             if procedure_details: procedure_details = procedure_details[:500] + "...[truncated]" if len(procedure_details) > 500 else procedure_details
#             rxn_time = dataset_data.get('rxn_time')
#             temperature = dataset_data.get('temperature')
#             yield_info = dataset_data.get('yield_000')
#             solvents = dataset_data.get('solvents_list') # Expect list or None
#             agents_catalysts = dataset_data.get('agents_list') # Expect list or None
        
#         # Update specific cache keys that are part of 'full_info' structure
#         reaction_cache[reaction_smiles_clean].update({
#             'procedure_details': procedure_details, 'reaction_time': rxn_time, 'temperature': temperature,
#             'yield_info': yield_info, 'solvents': solvents, 'agents_catalysts': agents_catalysts
#             # Note: name_info, fg_info, bond_info, classification_info are already in reaction_cache[reaction_smiles_clean] at top level
#         })

#         final_prompt_parts = [
#             f"You are a chemistry expert. Synthesize this reaction analysis into a clear explanation:",
#             f"Reaction SMILES (Processed): {reaction_smiles_clean}",
#             f"NAMES: {full_info_results.get('Names', 'Not available')}", # From tool output summary
#             f"BOND CHANGES: {full_info_results.get('Bond Changes', 'Not available')}", # From tool output summary
#             f"FUNCTIONAL GROUPS: {full_info_results.get('Functional Groups', 'Not available')}", # From tool output summary
#             f"REACTION TYPE (from classifier): {full_info_results.get('Reaction Classification', 'Not available')}" # From classifier summary
#         ]
#         if procedure_details: final_prompt_parts.append(f"PROCEDURE DETAILS: {procedure_details}")
        
#         conditions_parts = []
#         if temperature: conditions_parts.append(f"Temperature: {temperature}")
#         if rxn_time: conditions_parts.append(f"Time: {rxn_time}")
#         # Ensure yield_info is not None and not an empty string before formatting
#         if yield_info and str(yield_info).strip(): conditions_parts.append(f"Yield: {yield_info}%")
#         if conditions_parts: final_prompt_parts.append(f"EXPERIMENTAL CONDITIONS: {', '.join(conditions_parts)}")

#         materials_parts = []
#         if solvents and isinstance(solvents, list) and any(solvents): materials_parts.append(f"Solvents: {', '.join(solvents)}")
#         if agents_catalysts and isinstance(agents_catalysts, list) and any(agents_catalysts): materials_parts.append(f"Catalysts/Reagents: {', '.join(agents_catalysts)}")
#         if materials_parts: final_prompt_parts.append(f"KEY MATERIALS: {'; '.join(materials_parts)}")
        
#         final_prompt_parts.append(
#             "\nProvide a thorough, well-structured explanation covering the following aspects if information is available:"
#             "\n1. Begins with a high-level summary of what type of reaction this is"
#             "\n2. Explains what happens at the molecular level (bonds broken/formed)"
#             "\n3. Discusses the functional group transformations"
#             "\n4. Includes specific experimental conditions (temperature, time, yield, solvents, catalysts)"
#             "\n5. Procedure summary (if known): Briefly describe the experimental steps."
#             "\n6. Mentions common applications or importance of this reaction type"
#             "\nPresent the information clearly and logically for a chemist."
#         )
#         final_prompt_for_llm = "\n\n".join(final_prompt_parts)
        
#         focused_llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=2000)
#         response = focused_llm.invoke(final_prompt_for_llm, {"callbacks": callbacks} if callbacks else None)
#         analysis_text_summary = response.content.strip()

#         structured_result = {
#             'visualization_path': full_info_results.get('Visualization') if full_info_results.get('Visualization') and not str(full_info_results.get('Visualization')).lower().startswith('error') else None,
#             'analysis': analysis_text_summary, # The LLM's comprehensive summary
#             # Store individual pieces of info for potential direct use or structured display from dataset
#             'reaction_classification_summary': full_info_results.get('Reaction Classification', "N/A"), # This is summary from classifier
#             'procedure_details': procedure_details, 
#             'reaction_time': rxn_time, 
#             'temperature': temperature,
#             'yield': yield_info, # Store as 'yield' for consistency from dataset
#             'solvents': solvents or None, 
#             'agents_catalysts': agents_catalysts or None,
#             'analysis_context': "full_llm_summary_generated",
#             'processed_smiles_for_tools': reaction_smiles_clean # Ensure this is part of the result
#         }
        
#         reaction_cache[reaction_smiles_clean]['full_info'] = structured_result
#         return structured_result

#     except Exception as e:
#         tb_str = traceback.format_exc()
#         print(f"CRITICAL ERROR in handle_full_info for {reaction_smiles_clean}: {e}\n{tb_str}")
#         error_result = {
#             'visualization_path': None, 
#             'analysis': f"An internal error occurred during the full analysis of '{reaction_smiles_clean}'. Details: {str(e)}",
#             'analysis_context': "full_analysis_exception",
#             'processed_smiles_for_tools': reaction_smiles_clean
#         }
#         reaction_cache.setdefault(reaction_smiles_clean, {})['full_info'] = error_result
#         return error_result

# def handle_followup_question(query_text_for_llm, reaction_smiles_clean, original_compound_name=None, callbacks=None):
#     # --- Enhanced Debugging at Entry ---
#     print(f"\n--- [HANDLE_FOLLOWUP_ENTRY] ---")
#     print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
#     print(f"Query text for LLM: '{query_text_for_llm}'")
#     print(f"Reaction SMILES clean for cache lookup: '{reaction_smiles_clean}'")
#     print(f"Original compound name (if any): '{original_compound_name}'")
#     print(f"Is reaction_smiles_clean None or empty? {'Yes' if not reaction_smiles_clean else 'No'}")
#     if reaction_smiles_clean:
#         print(f"Does reaction_smiles_clean contain '>>'? {'Yes' if '>>' in reaction_smiles_clean else 'No'}")

#     if not reaction_smiles_clean or ">>" not in reaction_smiles_clean:
#         print(f"[HANDLE_FOLLOWUP_ERROR] Invalid or missing reaction_smiles_clean for followup. Returning error.")
#         return {'visualization_path': None, 'analysis': f"Error: Invalid reaction SMILES context for followup: '{reaction_smiles_clean}'", 'analysis_context': "invalid_smiles_context_followup", 'processed_smiles_for_tools': reaction_smiles_clean}

#     print(f"\n--- [CACHE_STATE_IN_FOLLOWUP FOR SMILES: '{reaction_smiles_clean}'] ---")
#     if reaction_smiles_clean in reaction_cache:
#         print(f"  SUCCESS: Key '{reaction_smiles_clean}' IS PRESENT in reaction_cache.")
#         current_smiles_cache_keys = list(reaction_cache[reaction_smiles_clean].keys())
#         print(f"  Keys under reaction_cache['{reaction_smiles_clean}']: {current_smiles_cache_keys}")
        
#         if 'full_info' in reaction_cache[reaction_smiles_clean]:
#             print(f"  SUCCESS: 'full_info' key IS PRESENT in reaction_cache['{reaction_smiles_clean}'].")
#             # To ensure it's not an error placeholder that looks like full_info:
#             if isinstance(reaction_cache[reaction_smiles_clean]['full_info'], dict) and \
#                'analysis' in reaction_cache[reaction_smiles_clean]['full_info'] and \
#                not reaction_cache[reaction_smiles_clean]['full_info']['analysis'].startswith("An internal error occurred"):
#                 print(f"  VALID 'full_info' seems to be present. Proceeding with specific follow-up logic.")
#             else:
#                 print(f"  WARNING: 'full_info' is present but might be an error placeholder or malformed. Content: {str(reaction_cache[reaction_smiles_clean]['full_info'])[:200]}...")
#         else:
#             print(f"  !!! CRITICAL CACHE MISS: 'full_info' key IS NOT PRESENT in reaction_cache['{reaction_smiles_clean}']. Expect fallback to full_info regeneration. !!!")
#     else:
#         print(f"  !!! CRITICAL CACHE MISS: Key '{reaction_smiles_clean}' IS NOT PRESENT in reaction_cache at all. Expect fallback to full_info regeneration. !!!")
#     print(f"--- [END_CACHE_STATE_IN_FOLLOWUP] ---\n")

#     # THE CRITICAL CHECK THAT LEADS TO REGENERATING FULL_INFO:
#     # Check if full_info is missing or if it's present but is an error message (indicating previous full_info failed)
#     full_info_missing_or_is_error = True # Assume missing initially
#     if reaction_smiles_clean in reaction_cache and 'full_info' in reaction_cache[reaction_smiles_clean]:
#         # Check if it's a dict and not an error string
#         if isinstance(reaction_cache[reaction_smiles_clean]['full_info'], dict) and \
#            'analysis' in reaction_cache[reaction_smiles_clean]['full_info'] and \
#            not reaction_cache[reaction_smiles_clean]['full_info'].get('analysis_context', '').endswith("_exception"): # Check context too
#             full_info_missing_or_is_error = False # It's present and seems valid

#     if full_info_missing_or_is_error:
#         print(f"[HANDLE_FOLLOWUP] Cache miss condition for valid 'full_info' MET. Running handle_full_info for '{reaction_smiles_clean}'.")
#         print(f"  This is because 'full_info' was either not found, or was an error placeholder from a previous attempt.")
#         print(f"  Original follow-up query text that will be passed to handle_full_info: '{query_text_for_llm}'")
#         # This call is the one producing the undesired full output because handle_full_info is designed to be comprehensive.
#         return handle_full_info(query_text_for_llm, reaction_smiles_clean, original_compound_name, callbacks=callbacks)
    
#     # --- If cache for full_info is HIT and VALID, proceed with specific follow-up logic ---
#     print(f"[HANDLE_FOLLOWUP] Cache for 'full_info' HIT and is valid. Proceeding with specific/general follow-up logic for: '{reaction_smiles_clean}'")
#     query_lower = query_text_for_llm.lower()
#     cached_full_info_data = reaction_cache[reaction_smiles_clean]['full_info'] 
#     cached_smiles_level_data = reaction_cache[reaction_smiles_clean] # Contains direct tool outputs like 'name_info', 'fg_info' etc.

#     # Visualization handling (same as before)
#     if any(term in query_lower for term in ["visual", "picture", "image", "show", "draw", "representation", "diagram"]):
#         # ... (visualization handling logic from your previous correct version)
#         viz_path_from_cache = cached_full_info_data.get('visualization_path')
#         if viz_path_from_cache and not str(viz_path_from_cache).lower().startswith("error"):
#             return {"visualization_path": viz_path_from_cache, "analysis": f"Visual representation of reaction: {reaction_smiles_clean}", 'analysis_context': "followup_visualization_cached", 'processed_smiles_for_tools': reaction_smiles_clean}
#         else:
#             tool_dict = {tool.name.lower(): tool for tool in tools}
#             visualizer_tool = tool_dict.get("chemvisualizer")
#             if visualizer_tool:
#                 try:
#                     viz_path_new = visualizer_tool.run(reaction_smiles_clean)
#                     if viz_path_new and not str(viz_path_new).lower().startswith("error") and str(viz_path_new).endswith(".png"):
#                         cached_smiles_level_data['visualization_path'] = viz_path_new # Update this level too
#                         cached_full_info_data['visualization_path'] = viz_path_new # Update full_info's copy
#                         return {"visualization_path": viz_path_new, "analysis": f"Visual representation (newly generated): {reaction_smiles_clean}", 'analysis_context': "followup_visualization_generated", 'processed_smiles_for_tools': reaction_smiles_clean}
#                     else:
#                         return {"visualization_path": None, "analysis": f"Visualization tool message: {viz_path_new}", 'analysis_context': "followup_visualization_tool_error", 'processed_smiles_for_tools': reaction_smiles_clean}
#                 except Exception as e:
#                     return {"visualization_path": None, "analysis": f"Error during follow-up visualization: {str(e)}", 'analysis_context': "followup_visualization_exception", 'processed_smiles_for_tools': reaction_smiles_clean}
#             else:
#                 return {"visualization_path": None, "analysis": "ChemVisualizer tool not found for follow-up.", 'analysis_context': "followup_visualization_no_tool", 'processed_smiles_for_tools': reaction_smiles_clean}

#     # Property keyword mapping (same as before)
#     property_keywords_map = {
#         'temperature': (['temperature', 'temp', 'heat', 'degrees', '°c', '°f', '°k'], 'temperature'),
#         'yield': (['yield', 'yields', '%', 'efficiency', 'conversion'], 'yield'), # Mapped to 'yield' in cached_full_info_data
#         'solvent': (['solvent', 'medium', 'solution'], 'solvents'),
#         'catalyst_reagent': (['catalyst', 'catalytic', 'agent', 'reagent', 'promoter'], 'agents_catalysts'),
#         'time': (['time', 'duration', 'long', 'minute', 'hour'], 'reaction_time'),
#         'procedure': (['procedure', 'protocol', 'steps', 'method', 'synthesis route'], 'procedure_details'),
#         'name': (['name', 'identity', 'what is this reaction called'], 'name_info'), # From cached_smiles_level_data
#         'functional_groups': (['functional group', 'fg', 'moieties', 'substituent'], 'fg_info'), # From cached_smiles_level_data
#         'bonds': (['bond', 'bonding', 'formed', 'broken', 'bond changes'], 'bond_info'), # From cached_smiles_level_data
#         'classification': (['type of reaction', 'class', 'category', 'kind of reaction', 'reaction class'], 'classification_info') # From cached_smiles_level_data
#     }
    
#     matched_specific_keywords_in_query = False
#     specific_data_snippets_for_llm = []
#     unavailable_queried_properties = []

#     # Logic for populating specific_data_snippets_for_llm and unavailable_queried_properties (same as before)
#     for prop_key_in_map, (keywords, cache_key_in_data) in property_keywords_map.items():
#         if any(keyword in query_lower for keyword in keywords):
#             matched_specific_keywords_in_query = True
#             property_display_name = prop_key_in_map.replace('_',' ').capitalize()
            
#             info_value = None
#             if cache_key_in_data in ['temperature', 'yield', 'solvents', 'agents_catalysts', 'reaction_time', 'procedure_details']:
#                 info_value = cached_full_info_data.get(cache_key_in_data) # e.g. cached_full_info_data['yield']
#             else: 
#                 info_value = cached_smiles_level_data.get(cache_key_in_data) # e.g. cached_smiles_level_data['name_info']

#             is_valid_info = False
#             current_info_str_segment = ""
#             if info_value is not None:
#                 if isinstance(info_value, list):
#                     valid_list_items = [str(i) for i in info_value if str(i).strip() and str(i).strip().lower() not in ['nan', 'n/a', 'none', 'not available', '']]
#                     if valid_list_items:
#                         is_valid_info = True
#                         current_info_str_segment = ", ".join(valid_list_items)
#                 else:
#                     temp_str = str(info_value).strip()
#                     # Ensure temp_str is not just an empty string after stripping or common NA values
#                     if temp_str and temp_str.lower() not in ['nan', 'n/a', 'none', 'not available', '']:
#                         is_valid_info = True
#                         current_info_str_segment = temp_str
            
#             if is_valid_info and current_info_str_segment: # Ensure it's not empty string
#                 specific_data_snippets_for_llm.append(f"- {property_display_name}: {current_info_str_segment}")
#                 print(f"[FOLLOWUP_DEBUG_DATA] Found data for '{property_display_name}': {current_info_str_segment[:100]}...")
#             else:
#                 unavailable_queried_properties.append(property_display_name)
#                 print(f"[FOLLOWUP_DEBUG_DATA] Data for '{property_display_name}' (key: {cache_key_in_data}) not found or invalid. Value was: '{info_value}'")

#     # Constructing LLM prompt and instructions (same as your last correct version based on matched_specific_keywords_in_query)
#     final_llm_prompt_parts = [
#         f"User follow-up question about reaction {reaction_smiles_clean}: \"{query_text_for_llm}\""
#     ]
#     context_for_llm = ""
#     instructions_for_llm = ""

#     if matched_specific_keywords_in_query:
#         # ... (same logic as your previous version for this block)
#         print(f"[FOLLOWUP_DEBUG_MODE] Specific property query mode. Snippets: {len(specific_data_snippets_for_llm)}, Unavailable: {unavailable_queried_properties}")
#         if specific_data_snippets_for_llm:
#             context_for_llm = "\nRelevant specific information from previous analysis (use ONLY this):\n" + "\n".join(specific_data_snippets_for_llm)
#             if unavailable_queried_properties: # Add note about unavailable queried properties
#                 context_for_llm += "\n\nNote: Information for the following specifically queried properties was not available from the previous analysis or dataset: " + ", ".join(unavailable_queried_properties) + "."
            
#             instructions_for_llm = (
#                 "\nInstructions: Based STRICTLY on the 'Relevant specific information' provided above, "
#                 "provide a CONCISE answer ONLY to the user's question. "
#                 "If specific information was noted as 'not available' for a queried property, state that clearly if relevant to the question. "
#                 "For any other part of the user's question that CANNOT be answered using ONLY the provided snippets, "
#                 "explicitly state that 'information for that part is not available.' "
#                 "Do not infer, add external knowledge, or provide information not directly asked for in the user's question."
#             )
#         else: # All specifically queried properties were unavailable
#             unavailable_list_str = ", ".join(unavailable_queried_properties) if unavailable_queried_properties else "the requested properties"
#             direct_answer = f"Information for the specifically requested properties ({unavailable_list_str}) is not available from the previous analysis or dataset."
#             print(f"[FOLLOWUP] Direct answer (no LLM call needed): {direct_answer}")
#             return {
#                 "visualization_path": None, "analysis": direct_answer,
#                 "analysis_context": "followup_specific_all_unavailable",
#                 'processed_smiles_for_tools': reaction_smiles_clean
#             }
#     else: # General follow-up query
#         # ... (same logic as your previous version for this block)
#         print(f"[FOLLOWUP_DEBUG_MODE] General follow-up query mode.")
#         general_context_parts = []
#         main_analysis_summary = cached_full_info_data.get('analysis', '') 
#         if main_analysis_summary: # From the LLM summary in full_info
#             summary_intro_match = re.search( 
#                 r"(?:1\.\s*High-Level Summary|Summary of Reaction Type|Overall Summary)[:\s\n]+(.*?)(?=\n\s*(?:2\.|\b[A-Z]|$))",
#                 main_analysis_summary, re.IGNORECASE | re.DOTALL
#             )
#             if summary_intro_match:
#                 general_context_parts.append(f"Overall Reaction Summary Snippet: {summary_intro_match.group(1).strip()[:300]}...")
#             elif len(main_analysis_summary) > 0: # Fallback if specific snippet regex fails
#                 general_context_parts.append(f"Overall Reaction Summary Snippet: {main_analysis_summary[:300]}...")
        
#         class_info = cached_smiles_level_data.get('classification_info') # Direct from classifier tool output cache
#         if class_info and str(class_info).strip() and str(class_info).lower() not in ['n/a', 'not available', 'none', '']:
#             general_context_parts.append(f"Reaction Classification: {str(class_info)[:200]}...")

#         if general_context_parts:
#             context_for_llm = "\nGeneral Context from Previous Analysis (use this to answer if relevant):\n" + "\n".join(general_context_parts)
#             instructions_for_llm = (
#                 "\nInstructions: Answer the user's question CONCISELY using ONLY the 'General Context from Previous Analysis' provided above. "
#                 "If this general context does not contain the specific information to answer the user's question, "
#                 f"respond ONLY with: 'The specific information for your query \"{query_text_for_llm}\" is not available in the summarized context from the previous analysis.' "
#                 "Do not infer or use external knowledge. Avoid re-summarizing the entire context unless explicitly asked by the user."
#             )
#         else: 
#              instructions_for_llm = (
#                 "\nInstructions: There is no specific or general context available from the previous analysis to answer this question. "
#                 f"Respond ONLY with: 'The information required to answer your query \"{query_text_for_llm}\" is not available from the previous analysis.' "
#             )

#     final_llm_prompt_parts.append(context_for_llm)
#     final_llm_prompt_parts.append(instructions_for_llm)
#     final_llm_prompt = "\n".join(filter(None, final_llm_prompt_parts)) # Filter out empty strings

#     print(f"[FOLLOWUP] Final prompt for LLM (length {len(final_llm_prompt)}):\n{final_llm_prompt[:1000]}...\n")

#     focused_llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=250) 
#     response = focused_llm.invoke(final_llm_prompt, {"callbacks": callbacks} if callbacks else None)
#     followup_analysis_text = response.content.strip()
    
#     print(f"[FOLLOWUP] LLM response generated. Length: {len(followup_analysis_text)}. Text: {followup_analysis_text[:200]}...")
    
#     return {
#         "visualization_path": None,
#         "analysis": followup_analysis_text,
#         "analysis_context": "followup_llm_answer_from_context",
#         'processed_smiles_for_tools': reaction_smiles_clean
#     }

# def enhanced_query(full_query: str, callbacks=None, original_compound_name: str = None):
#     final_result = {}
#     query_context_for_filename = "unknown_query_type"
#     # Extract clean reaction SMILES (>> format) from the potentially larger query string
#     reaction_smiles_for_tools = extract_reaction_smiles(full_query) 
    
#     print(f"[ENHANCED_QUERY] Original query: '{full_query[:100]}...'")
#     if reaction_smiles_for_tools:
#         print(f"[ENHANCED_QUERY] Extracted/Converted SMILES for tools: '{reaction_smiles_for_tools}'")
#     else:
#         print(f"[ENHANCED_QUERY] No reaction SMILES extracted or converted from query.")

#     try:
#         # Case 1: No reaction SMILES extracted, use the general agent with the full query.
#         if not reaction_smiles_for_tools:
#             print("[ENHANCED_QUERY] No reaction SMILES identified. Using general agent with the full query.")
#             agent_output = agent.invoke({"input": full_query}, {"callbacks": callbacks} if callbacks else {})
#             final_result = {
#                 "visualization_path": None,
#                 "analysis": extract_final_answer(agent_output.get("output", "Agent did not provide a final answer.")),
#             }
#             query_context_for_filename = "general_agent_no_smiles"
        
#         # Case 2: Reaction SMILES IS available. Now, check query intent.
#         elif "full" in full_query.lower() and any(term in full_query.lower() for term in ["information", "analysis", "detail", "explain", "tell me about", "give me all", "everything about"]):
#             print(f"[ENHANCED_QUERY] Full analysis explicitly requested for extracted SMILES: {reaction_smiles_for_tools}")
#             final_result = handle_full_info(full_query, reaction_smiles_for_tools, original_compound_name, callbacks=callbacks)
#             query_context_for_filename = final_result.get('analysis_context', 'full_summary_fallback')
        
#         # Case 3: Visualization specific requests for the extracted reaction SMILES
#         elif any(term in full_query.lower() for term in ["visual", "picture", "image", "show", "draw", "representation", "diagram"]):
#             print(f"[ENHANCED_QUERY] Visualization requested for extracted SMILES: {reaction_smiles_for_tools}")
#             # Check cache first for this specific SMILES (visualization_path is at top level of reaction_cache[smiles])
#             viz_path_cached = reaction_cache.get(reaction_smiles_for_tools, {}).get('visualization_path')
#             if viz_path_cached and not str(viz_path_cached).lower().startswith("error"):
#                 final_result = {"visualization_path": viz_path_cached, "analysis": f"Cached visual representation for: {reaction_smiles_for_tools}"}
#                 query_context_for_filename = "visualization_cached"
#             else: # Not cached or was error, try to generate
#                 tool_dict = {tool.name.lower(): tool for tool in tools}
#                 visualizer_tool = tool_dict.get("chemvisualizer")
#                 if visualizer_tool:
#                     try:
#                         viz_path = visualizer_tool.run(reaction_smiles_for_tools)
#                         if viz_path and not str(viz_path).lower().startswith("error") and str(viz_path).endswith(".png"):
#                             reaction_cache.setdefault(reaction_smiles_for_tools, {})['visualization_path'] = viz_path
#                             # If full_info was already cached for this SMILES, update its viz_path too
#                             if 'full_info' in reaction_cache.get(reaction_smiles_for_tools, {}): # Ensure full_info exists
#                                 reaction_cache[reaction_smiles_for_tools]['full_info']['visualization_path'] = viz_path
#                             final_result = {"visualization_path": viz_path, "analysis": f"Generated visual representation for: {reaction_smiles_for_tools}"}
#                             query_context_for_filename = "visualization_generated"
#                         else:
#                             final_result = {"visualization_path": None, "analysis": f"Visualization tool message: {viz_path}"}
#                             query_context_for_filename = "visualization_tool_error"
#                     except Exception as e:
#                         final_result = {"visualization_path": None, "analysis": f"Error visualizing reaction '{reaction_smiles_for_tools}': {str(e)}"}
#                         query_context_for_filename = "visualization_exception"
#                 else:
#                     final_result = {"visualization_path": None, "analysis": "ChemVisualizer tool not found."}
#                     query_context_for_filename = "visualization_no_tool"

#         # Case 4: Follow-up or specific property questions IF SMILES was extracted AND context exists or keywords suggest it
#         elif reaction_smiles_for_tools in reaction_cache or \
#              any(keyword in full_query.lower() for keyword in [
#                  "temperature", "yield", "solvent", "catalyst", "reagent", "time", "mechanism", "applications", 
#                  "procedure", "protocol", "steps", "bonds", "functional group", "classification", "name", "identity",
#                  "what is", "how does", "explain further", "why", "details on", "more about", "tell me more", "can you elaborate"
#              ]):
#             print(f"[ENHANCED_QUERY] Specific/Follow-up style question for extracted SMILES: {reaction_smiles_for_tools}")
#             final_result = handle_followup_question(full_query, reaction_smiles_for_tools, original_compound_name, callbacks=callbacks)
#             query_context_for_filename = final_result.get('analysis_context', 'followup_answer_fallback')
        
#         # Case 5: Default - If SMILES was extracted but no specific handler matched above,
#         # treat as a request for full analysis for this (potentially new) SMILES.
#         else:
#             print(f"[ENHANCED_QUERY] Defaulting to full analysis for newly identified/extracted reaction: {reaction_smiles_for_tools}")
#             final_result = handle_full_info(full_query, reaction_smiles_for_tools, original_compound_name, callbacks=callbacks)
#             query_context_for_filename = final_result.get('analysis_context', 'full_summary_default_new_smiles')

#         # --- Save the analysis text if applicable ---
#         analysis_text_to_save = final_result.get("analysis")
#         smiles_for_saving = reaction_smiles_for_tools
        
#         if smiles_for_saving and analysis_text_to_save and isinstance(analysis_text_to_save, str) and \
#            not query_context_for_filename.startswith("visualization_") and \
#            "error" not in query_context_for_filename.lower() and \
#            len(analysis_text_to_save.strip()) > 20 : # Reduced length threshold for saving small valid answers
#             save_analysis_to_file(smiles_for_saving, analysis_text_to_save, query_context_for_filename, original_compound_name)
        
#         # Ensure 'processed_smiles_for_tools' is in the result from all paths
#         final_result['processed_smiles_for_tools'] = reaction_smiles_for_tools
#         return final_result

#     except Exception as e:
#         tb_str = traceback.format_exc()
#         print(f"CRITICAL Error in enhanced_query for query '{full_query}': {str(e)}\n{tb_str}")
#         error_text = f"Error processing your query: {str(e)}. Please check the logs for details."
#         smiles_ctx_for_error_log = reaction_smiles_for_tools if reaction_smiles_for_tools else "no_smiles_extracted"
#         save_analysis_to_file(smiles_ctx_for_error_log, f"Query: {full_query}\n{error_text}\n{tb_str}", "enhanced_query_CRITICAL_error", original_compound_name)
        
#         return {
#             "visualization_path": None, 
#             "analysis": error_text,
#             "processed_smiles_for_tools": reaction_smiles_for_tools # Still return if extracted before error
#         }

# # Ensure pandas is available for query_reaction_dataset
# # (already handled by import at the top)




import os
import re
import requests # Keep if api_config or other direct calls use it
try:
    import api_config
except ImportError:
    print("Warning: api_config.py not found. API keys might not be loaded if they are set there.")

from tools.make_tools import make_tools
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain_openai import ChatOpenAI
from tools.asckos import ReactionClassifier # Assuming this is correctly set up
from functools import lru_cache
import time
import traceback
import pandas as pd # Make sure pandas is imported for query_reaction_dataset

# --- Setup for Saving Analysis Files ---
try:
    PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    PROJECT_ROOT_DIR = os.getcwd()
REACTION_ANALYSIS_OUTPUT_DIR = os.path.join(PROJECT_ROOT_DIR, "reaction_analysis_outputs")
os.makedirs(REACTION_ANALYSIS_OUTPUT_DIR, exist_ok=True)

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
    filename_parts.append(f"rxn_{smiles_part}")
    filename_parts.append(sanitize_filename(query_context_type))
    filename_parts.append(timestamp)
    filename = "_".join(filter(None, filename_parts)) + ".txt"
    filepath = os.path.join(REACTION_ANALYSIS_OUTPUT_DIR, filename)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Reaction SMILES: {reaction_smiles}\n")
            if original_compound_name and original_compound_name != reaction_smiles:
                 f.write(f"Original Target Context: {original_compound_name}\n")
            f.write(f"Analysis Type: {query_context_type}\n"); f.write(f"Timestamp: {timestamp}\n")
            f.write("="*50 + "\n\n"); f.write(analysis_text)
        print(f"[SAVE_ANALYSIS] Saved analysis to: {filepath}")
    except Exception as e: print(f"[SAVE_ANALYSIS_ERROR] Error saving {filepath}: {e}")

llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=4000) # General LLM
tools = make_tools(llm=llm)
dataset_path1 = os.environ.get('REACTION_DATASET_PATH1', None)
dataset_path2 = os.environ.get('REACTION_DATASET_PATH2', None)
try:
    reaction_classifier = ReactionClassifier(dataset_path1, dataset_path2)
except Exception as e:
    print(f"Warning: Could not initialize ReactionClassifier: {e}. Classification features will be unavailable.")
    reaction_classifier = None

reaction_cache = {}

PREFIX = """
You are Chem Copilot, an expert chemistry assistant. You have access to the following tools to analyze molecules and chemical reactions.
Always begin by understanding the user's intent — what kind of information are they asking for?
Here is how to choose tools:
- If the user gives a SMILES or reaction SMILES and asks for the name, you MUST use SMILES2Name. Do NOT analyze bonds or functional groups for this task.
- Use NameToSMILES: when the user gives a compound/reaction name and wants the SMILES or structure.
- Use FuncGroups: when the user wants to analyze functional groups in a molecule or a reaction (input is SMILES or reaction SMILES).
- Use BondChangeAnalyzer: when the user asks for which bonds are broken, formed, or changed in a chemical reaction.
If the user wants all of the above (full analysis of a reaction SMILES), respond with "This requires full analysis." (This will be handled by a separate function.)
Always return your answer in this format:
Final Answer: <your answer here>
For FuncGroups results:
- Always list the functional groups identified in each reactant and product separately
- Include the transformation summary showing disappeared groups, appeared groups, and unchanged groups
- Provide a clear conclusion about what transformation occurred in the reaction
For BondChangeAnalyzer results:
- Always list the specific bonds that were broken, formed, or changed with their bond types
- Include the atom types involved in each bond (e.g., C-O, N-H)
- Provide a clear conclusion summarizing the key bond changes in the reaction
"""
FORMAT_INSTRUCTIONS = """
You can only respond with a single complete
"Thought, Action, Action Input" format
OR a single "Final Answer" format
Complete format:
Thought: (reflect on your progress and decide what to do next)
Action: (the action name, should be one of [{tool_names}])
Action Input: (the input string to the action)
OR
Final Answer: (the final answer to the original input question)
"""
SUFFIX = """
Question: {input}
{agent_scratchpad}
"""
prompt = ZeroShotAgent.create_prompt(
    tools=tools, prefix=PREFIX, suffix=SUFFIX,
    format_instructions=FORMAT_INSTRUCTIONS, input_variables=["input", "agent_scratchpad"]
)
agent_chain = ZeroShotAgent.from_llm_and_tools(llm=llm, tools=tools, prompt=prompt)
agent = AgentExecutor(agent=agent_chain, tools=tools, verbose=True, handle_parsing_errors=True)

def extract_final_answer(full_output: str):
    match = re.search(r"Final Answer:\s*(.*)", full_output, re.DOTALL)
    return match.group(1).strip() if match else full_output.strip()

@lru_cache(maxsize=100)
def query_reaction_dataset(reaction_smiles):
    if not reaction_smiles: return None
    # ... (rest of query_reaction_dataset, ensure it handles reaction_classifier being None)
    if reaction_smiles in reaction_cache and 'dataset_info' in reaction_cache[reaction_smiles]:
        return reaction_cache[reaction_smiles]['dataset_info']
    
    if not reaction_classifier or \
       (not hasattr(reaction_classifier, 'dataset1') or reaction_classifier.dataset1 is None or reaction_classifier.dataset1.empty) and \
       (not hasattr(reaction_classifier, 'dataset2') or reaction_classifier.dataset2 is None or reaction_classifier.dataset2.empty):
        return None

    try:
        df = None
        if hasattr(reaction_classifier, 'dataset1') and reaction_classifier.dataset1 is not None and not reaction_classifier.dataset1.empty:
            df = reaction_classifier.dataset1
        elif hasattr(reaction_classifier, 'dataset2') and reaction_classifier.dataset2 is not None and not reaction_classifier.dataset2.empty:
            df = reaction_classifier.dataset2
        
        if df is None or df.empty: return None

        fields_to_extract = ['procedure_details', 'rxn_time', 'temperature', 'yield_000', 'reaction_name', 'reaction_classname', 'prediction_certainty']
        smiles_columns = ['rxn_str', 'reaction_smiles', 'smiles', 'rxn_smiles']
        exact_match = None
        for col in smiles_columns:
            if col in df.columns and df[col].dtype == 'object':
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
    except Exception as e: print(f"Error querying dataset for '{reaction_smiles}': {e}"); return None

def extract_reaction_smiles(query: str):
    # ... (extract_reaction_smiles function as before) ...
    explicit_pattern_gg = r"(?:reaction\s+step|rxn|reaction|SMILES)\s*[:=]?\s*([\w@\[\]\.\+\-\=\#\:\(\)\\\/;\$\%\|\{\}\s~]+>>[\w@\[\]\.\+\-\=\#\:\(\)\\\/;\$\%\|\{\}\s~]*)"
    match = re.search(explicit_pattern_gg, query, re.IGNORECASE)
    if match:
        smiles = match.group(1).strip()
        if ">>" in smiles:
            parts = smiles.split(">>")
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                print(f"[EXTRACT_SMILES] Found by explicit '>>' pattern: '{smiles}'")
                return smiles

    standalone_pattern_gg = r"(?:^|\s|[:=\(\-])([\w@\[\]\.\+\-\=\#\:\(\)\\\/;\$\%\|\{\}\s~]+>>[\w@\[\]\.\+\-\=\#\:\(\)\\\/;\$\%\|\{\}\s~]*)(?:\s|[\.\,\)]|$)"
    match = re.search(standalone_pattern_gg, query)
    if match:
        smiles = match.group(1).strip()
        if ">>" in smiles and len(smiles) > 3:
            parts = smiles.split(">>")
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                print(f"[EXTRACT_SMILES] Found by standalone '>>' pattern: '{smiles}'")
                return smiles
    
    smi_part_chars = r"[\w@\[\]\+\-\=\#\:\(\)\\\/;\$\%\|\{\}\.]"
    explicit_pattern_gt = rf"(?:reaction\s+step|rxn|reaction|SMILES)\s*[:=]?\s*({smi_part_chars}+(?:>{smi_part_chars}*)+)"
    match_gt_explicit = re.search(explicit_pattern_gt, query, re.IGNORECASE)
    
    extracted_gt_smiles = None
    if match_gt_explicit:
        temp_smiles = match_gt_explicit.group(1).strip()
        if ">>" not in temp_smiles and ">" in temp_smiles :
            extracted_gt_smiles = temp_smiles
            print(f"[EXTRACT_SMILES] Found by explicit '>' pattern (pre-conversion): '{extracted_gt_smiles}'")

    if not extracted_gt_smiles:
        standalone_pattern_gt = rf"(?:^|\s|[:=\(\-])({smi_part_chars}+(?:>{smi_part_chars}*)+)(?:\s|[\.\,\)]|$)"
        match_gt_standalone = re.search(standalone_pattern_gt, query)
        if match_gt_standalone:
            temp_smiles = match_gt_standalone.group(1).strip()
            if ">>" not in temp_smiles and ">" in temp_smiles:
                extracted_gt_smiles = temp_smiles
                print(f"[EXTRACT_SMILES] Found by standalone '>' pattern (pre-conversion): '{extracted_gt_smiles}'")
    
    if extracted_gt_smiles:
        parts = extracted_gt_smiles.split('>')
        cleaned_parts = [p.strip() for p in parts if p.strip()]
        if len(cleaned_parts) >= 2:
            products = cleaned_parts[-1]
            reactants_and_agents_str = ".".join(cleaned_parts[:-1])
            reactants_and_agents_str = re.sub(r'\.+', '.', reactants_and_agents_str).strip('.')
            if reactants_and_agents_str and products:
                converted_smiles = f"{reactants_and_agents_str}>>{products}"
                final_parts_check = converted_smiles.split(">>")
                if len(final_parts_check) == 2 and final_parts_check[0].strip() and final_parts_check[1].strip():
                    print(f"[EXTRACT_SMILES] Converted '>' pattern to '>>': '{converted_smiles}'")
                    return converted_smiles
                else:
                    print(f"[EXTRACT_SMILES] Conversion of '{extracted_gt_smiles}' to '>>' resulted in invalid structure: '{converted_smiles}'")
            else:
                print(f"[EXTRACT_SMILES] Processing '{extracted_gt_smiles}' after splitting and cleaning led to empty reactants/products part.")
        else:
            print(f"[EXTRACT_SMILES] Splitting '{extracted_gt_smiles}' by '>' resulted in less than 2 valid non-empty parts after stripping.")
    return None

def handle_full_info(query_text_for_llm_summary, reaction_smiles_clean, original_compound_name=None, callbacks=None):
    print(f"\n--- [HANDLE_FULL_INFO_ENTRY for '{reaction_smiles_clean}'] ---")
    print(f"Query text for LLM summary: '{query_text_for_llm_summary[:100]}...'")
    if not reaction_smiles_clean or ">>" not in reaction_smiles_clean:
        # ... (error handling)
        print(f"[HANDLE_FULL_INFO_ERROR] Invalid or missing reaction_smiles_clean: '{reaction_smiles_clean}'")
        return {'visualization_path': None, 'analysis': f"Error: Invalid reaction SMILES provided for analysis: '{reaction_smiles_clean}'", 'analysis_context': "invalid_smiles_input", 'processed_smiles_for_tools': reaction_smiles_clean}


    # Check cache for valid, non-error full_info
    if reaction_smiles_clean in reaction_cache and 'full_info' in reaction_cache[reaction_smiles_clean]:
        cached_data = reaction_cache[reaction_smiles_clean]['full_info']
        if isinstance(cached_data, dict) and \
           'analysis' in cached_data and \
           not cached_data.get('analysis_context', '').endswith(("_exception", "_error")): # More robust check for non-error cache
            print(f"Using CACHED valid full_info data for reaction: {reaction_smiles_clean}")
            if 'processed_smiles_for_tools' not in cached_data: # Ensure it's there
                 cached_data['processed_smiles_for_tools'] = reaction_smiles_clean
            return cached_data
        else:
            print(f"Cached 'full_info' for {reaction_smiles_clean} found but is an error placeholder or invalid. Regenerating.")

    # Initialize cache for this SMILES if it's new
    reaction_cache.setdefault(reaction_smiles_clean, {})
    full_info_results = {} # Stores summaries of tool outputs for the LLM prompt
    tool_dict = {tool.name.lower(): tool for tool in tools}

    try:
        # ... (tool calls: visualization, smiles2name, funcgroups, bondchangeanalyzer, reaction_classifier) ...
        # Ensure full tool outputs are cached at reaction_cache[reaction_smiles_clean][TOOL_CACHE_KEY]
        # And summaries/relevant parts are put into full_info_results for the LLM prompt
        # Visualization
        visualizer_tool = tool_dict.get("chemvisualizer")
        if visualizer_tool:
            try:
                visualization_path = visualizer_tool.run(reaction_smiles_clean)
                if visualization_path and not str(visualization_path).lower().startswith('error') and str(visualization_path).endswith(".png"):
                    full_info_results['Visualization'] = visualization_path # For LLM prompt if needed, mostly for direct display
                    reaction_cache[reaction_smiles_clean]['visualization_path'] = visualization_path # For direct access
                else: # Tool ran but error or no path
                    full_info_results['Visualization'] = f"Visualization tool message: {visualization_path}"
                    reaction_cache[reaction_smiles_clean]['visualization_path'] = None
            except Exception as e:
                full_info_results['Visualization'] = f"Error visualizing reaction: {str(e)}"
                reaction_cache[reaction_smiles_clean]['visualization_path'] = None
        else:
            full_info_results['Visualization'] = "ChemVisualizer tool not found"
            reaction_cache[reaction_smiles_clean]['visualization_path'] = None
        
        # Other tools
        for tool_name_lower, data_key, cache_key_for_full_output in [
            ("smiles2name", "Names", "name_info"),
            ("funcgroups", "Functional Groups", "fg_info"),
            ("bondchangeanalyzer", "Bond Changes", "bond_info")
        ]:
            tool_instance = tool_dict.get(tool_name_lower)
            if tool_instance:
                try:
                    tool_result_full = tool_instance.run(reaction_smiles_clean)
                    # For LLM prompt, use a summary or key part. Cache the full result.
                    # This summary logic might need to be tool-specific for best results.
                    if isinstance(tool_result_full, str):
                        display_result_for_llm_prompt = tool_result_full[:300] + ("..." if len(tool_result_full) > 300 else "")
                    elif isinstance(tool_result_full, dict) and 'Final Answer' in tool_result_full: # If tool returns agent-like output
                        display_result_for_llm_prompt = str(tool_result_full['Final Answer'])[:300] + "..."
                    else:
                        display_result_for_llm_prompt = str(tool_result_full)[:300] + "..."
                    
                    full_info_results[data_key] = display_result_for_llm_prompt # Summary for LLM
                    reaction_cache[reaction_smiles_clean][cache_key_for_full_output] = tool_result_full # Full output for direct use
                except Exception as e:
                    err_msg = f"Error running {tool_name_lower}: {str(e)}"
                    full_info_results[data_key] = err_msg
                    reaction_cache[reaction_smiles_clean][cache_key_for_full_output] = err_msg # Cache error
            else: # Tool not found
                msg = f"{tool_name_lower.capitalize()} tool not found"
                full_info_results[data_key] = msg
                reaction_cache[reaction_smiles_clean][cache_key_for_full_output] = msg

        # Reaction Classifier
        if reaction_classifier:
            try:
                classifier_result_raw = reaction_classifier._run(reaction_smiles_clean)
                if isinstance(classifier_result_raw, str):
                    summary_match = re.search(r'## Summary\n(.*?)(?=\n##|$)', classifier_result_raw, re.DOTALL | re.IGNORECASE)
                    classifier_summary = summary_match.group(1).strip() if summary_match else (classifier_result_raw.splitlines()[0] if classifier_result_raw.splitlines() else "No summary found")
                    full_info_results['Reaction Classification'] = classifier_summary[:300] + "..." if len(classifier_summary) > 300 else classifier_summary
                    reaction_cache[reaction_smiles_clean]['classification_info'] = classifier_summary # Cache the (potentially longer) summary
                else:
                    full_info_results['Reaction Classification'] = "Classifier result not a string"
                    reaction_cache[reaction_smiles_clean]['classification_info'] = "Classifier result not a string"
            except Exception as e:
                full_info_results['Reaction Classification'] = f"Error classifying: {str(e)}"
                reaction_cache[reaction_smiles_clean]['classification_info'] = f"Error classifying: {str(e)}"
        else:
            full_info_results['Reaction Classification'] = "ReactionClassifier not available"
            reaction_cache[reaction_smiles_clean]['classification_info'] = "ReactionClassifier not available"


        # Dataset Query
        dataset_data = query_reaction_dataset(reaction_smiles_clean) 
        procedure_details, rxn_time, temperature, yield_val_from_dataset, solvents, agents_catalysts = None, None, None, None, None, None
        if dataset_data:
            procedure_details = dataset_data.get('procedure_details')
            rxn_time = dataset_data.get('rxn_time')
            temperature = dataset_data.get('temperature')
            yield_val_from_dataset = dataset_data.get('yield_000') # This is the value for 'yield' from dataset
            solvents = dataset_data.get('solvents_list')
            agents_catalysts = dataset_data.get('agents_list')
        
        # Store these directly in reaction_cache[reaction_smiles_clean] for easy access by follow-up
        # These will also be part of the structured_result for 'full_info'
        reaction_cache[reaction_smiles_clean].update({
            'procedure_details': procedure_details, 
            'reaction_time': rxn_time, 
            'temperature': temperature,
            'yield': yield_val_from_dataset, # Store under the key 'yield' for consistency
            'solvents': solvents, 
            'agents_catalysts': agents_catalysts
        })

        # Construct final_prompt_for_llm using items from full_info_results and dataset derived values
        # ... (final_prompt_for_llm construction using items from full_info_results and the dataset values directly) ...
        # ... (LLM call to get analysis_text_summary) ...
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
            "\n1. Begins with a high-level summary of what type of reaction this is"
            "\n2. Explains what happens at the molecular level (bonds broken/formed)"
            "\n3. Discusses the functional group transformations"
            "\n4. Includes specific experimental conditions (temperature, time, yield, solvents, catalysts)"
            "\n5. Procedure summary (if known): Briefly describe the experimental steps."
            "\n6. Mentions common applications or importance of this reaction type"
            "\nPresent the information clearly and logically for a chemist."
        )
        final_prompt_for_llm = "\n\n".join(final_prompt_parts)
        
        focused_llm_full_summary = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=2000)
        response = focused_llm_full_summary.invoke(final_prompt_for_llm, {"callbacks": callbacks} if callbacks else None)
        analysis_text_summary = response.content.strip()


        structured_result_for_full_info_cache = {
            'visualization_path': reaction_cache[reaction_smiles_clean].get('visualization_path'), # Get from direct cache
            'analysis': analysis_text_summary, # The LLM's comprehensive summary
            
            # Data that was used to generate the summary, also for direct access
            'reaction_classification_summary': reaction_cache[reaction_smiles_clean].get('classification_info'),
            'procedure_details': procedure_details, 
            'reaction_time': rxn_time, 
            'temperature': temperature,
            'yield': yield_val_from_dataset, # Stored under 'yield'
            'solvents': solvents or None, 
            'agents_catalysts': agents_catalysts or None,
            
            'analysis_context': "full_llm_summary_generated",
            'processed_smiles_for_tools': reaction_smiles_clean
        }
        
        reaction_cache[reaction_smiles_clean]['full_info'] = structured_result_for_full_info_cache
        
        print(f"\n--- [CACHE_STATE_AFTER_FULL_INFO_STORE for '{reaction_smiles_clean}'] ---")
        # ... (debug print for successful store) ...
        if reaction_smiles_clean in reaction_cache and 'full_info' in reaction_cache.get(reaction_smiles_clean, {}):
            stored_item = reaction_cache[reaction_smiles_clean]['full_info']
            if isinstance(stored_item, dict) and 'analysis' in stored_item:
                 print(f"  SUCCESS: 'full_info' (dict with 'analysis') stored for '{reaction_smiles_clean}'. Context: {stored_item.get('analysis_context')}")
            else:
                 print(f"  WARNING: 'full_info' stored for '{reaction_smiles_clean}', but it's not the expected dict. Type: {type(stored_item)}")
        else:
            print(f"  !!! FAILURE: 'full_info' FAILED TO BE STORED or key '{reaction_smiles_clean}' missing after attempt. !!!")
        print(f"--- [END_CACHE_STATE_AFTER_FULL_INFO_STORE] ---\n")
        
        return structured_result_for_full_info_cache

    except Exception as e:
        # ... (exception handling, store error in full_info) ...
        tb_str = traceback.format_exc()
        print(f"CRITICAL ERROR in handle_full_info for {reaction_smiles_clean}: {e}\n{tb_str}")
        error_result = {
            'visualization_path': None, 
            'analysis': f"An internal error occurred during the full analysis of '{reaction_smiles_clean}'. Details: {str(e)}",
            'analysis_context': "full_analysis_exception", # Mark as exception
            'processed_smiles_for_tools': reaction_smiles_clean
        }
        reaction_cache.setdefault(reaction_smiles_clean, {})['full_info'] = error_result
        
        print(f"\n--- [CACHE_STATE_AFTER_FULL_INFO_ERROR_STORE for '{reaction_smiles_clean}'] ---")
        # ... (debug print for error store) ...
        if reaction_smiles_clean in reaction_cache and 'full_info' in reaction_cache.get(reaction_smiles_clean, {}):
            stored_item = reaction_cache[reaction_smiles_clean]['full_info']
            if isinstance(stored_item, dict) and stored_item.get('analysis_context') == "full_analysis_exception":
                 print(f"  SUCCESS: 'full_info' (error placeholder with context 'full_analysis_exception') stored for '{reaction_smiles_clean}'.")
            else:
                 print(f"  WARNING: 'full_info' (error placeholder) stored for '{reaction_smiles_clean}', but context mismatch or wrong type. Stored: {str(stored_item)[:100]}")
        else:
            print(f"  !!! FAILURE: 'full_info' (error placeholder) FAILED TO BE STORED or key '{reaction_smiles_clean}' missing after attempt. !!!")
        print(f"--- [END_CACHE_STATE_AFTER_FULL_INFO_ERROR_STORE] ---\n")
        return error_result


def handle_followup_question(query_text, reaction_smiles, original_compound_name=None, callbacks=None):
    cached = reaction_cache.get(reaction_smiles, {})
    full_info = cached.get('full_info', {})
    original_analysis = cached.get('analysis', '')
    
    property_map = {
        'solvent': {
            'keywords': ['solvent', 'solution', 'medium', 'dissolve'],
            'cache_fields': ['solvent', 'solvents', 'reaction_solvent', 'solvent_system'],
            'text_patterns': [
                r'solvent:?\s*([^\.]+)',
                r'solvents:?\s*([^\.]+)',
                r'carried out in\s*([^\.]+)',
                r'using\s*([^\(]+)\s*as solvent'
            ]
        },
        'temperature': {
            'keywords': ['temperature', 'temp', '°c', '°f', 'kelvin'],
            'cache_fields': ['temperature', 'temp', 'reaction_temp', 'conditions.temp'],
            'text_patterns': [
                r'temperature:?\s*([^\.]+)',
                r'temp:?\s*([^\.]+)',
                r'at\s*([\d\-]+)\s*°',
                r'heated to\s*([^\.]+)'
            ]
        },
        'yield': {
            'keywords': ['yield', '% yield', 'percentage', 'efficiency'],
            'cache_fields': ['yield', 'reaction_yield', 'percentage_yield'],
            'text_patterns': [
                r'yield:?\s*([^\.]+)',
                r'([\d\.]+%) yield',
                r'obtained in\s*([^\.]+)\s*yield'
            ]
        },
        'time': {
            'keywords': ['time', 'duration', 'hours', 'minutes', 'h'],
            'cache_fields': ['time', 'reaction_time', 'duration', 'time_total'],
            'text_patterns': [
                r'time:?\s*([^\.]+)',
                r'duration:?\s*([^\.]+)',
                r'for\s*([\d\.]+\s*(h|hours|minutes))',
                r'stirred for\s*([^\.]+)'
            ]
        },
        'catalyst': {
            'keywords': ['catalyst', 'reagent', 'agent', 'promoter'],
            'cache_fields': ['catalyst', 'catalysts', 'reagent', 'agents_catalysts'],
            'text_patterns': [
                r'catalyst:?\s*([^\.]+)',
                r'using\s*([^\(]+)\s*as catalyst',
                r'catalyzed by\s*([^\.]+)',
                r'in the presence of\s*([^\.]+)'
            ]
        },
        'pressure': {
            'keywords': ['pressure', 'psi', 'bar', 'atmosphere'],
            'cache_fields': ['pressure', 'reaction_pressure'],
            'text_patterns': [
                r'pressure:?\s*([^\.]+)',
                r'under\s*([^\.]+)\s*pressure',
                r'at\s*([\d\.]+)\s*(psi|bar)'
            ]
        },
        'ph': {
            'keywords': ['ph', 'acidic', 'basic', 'neutral'],
            'cache_fields': ['ph', 'reaction_ph'],
            'text_patterns': [
                r'ph:?\s*([^\.]+)',
                r'at\s*ph\s*([\d\.]+)',
                r'under\s*([^\.]+)\s*conditions'
            ]
        },
        'procedure': {
            'keywords': ['procedure', 'protocol', 'steps', 'method'],
            'cache_fields': ['procedure', 'experimental_procedure'],
            'text_patterns': [
                r'procedure:?\s*([^\.]+)',
                r'steps:?\s*([^\.]+)',
                r'method:?\s*([^\.]+)'
            ]
        }
    }

    query_lower = query_text.lower()
    
    for prop, prop_info in property_map.items():
        if any(keyword in query_lower for keyword in prop_info['keywords']):
            # Try structured fields first
            for field in prop_info['cache_fields']:
                if field in full_info and full_info[field]:
                    value = format_value(full_info[field])
                    return create_response(prop, value, reaction_smiles)

            # Try text extraction if structured data not found
            if original_analysis:
                import re
                for pattern in prop_info['text_patterns']:
                    match = re.search(pattern, original_analysis, re.IGNORECASE)
                    if match:
                        value = match.group(1).strip()
                        return create_response(prop, value, reaction_smiles)

            # If no data found in structured fields or text
            return {
                "visualization_path": None,
                "analysis": f"Specific {prop} information is not available in the recorded data.",
                "analysis_context": f"followup_{prop}_not_found",
                "processed_smiles_for_tools": reaction_smiles
            }

    # Fallback for unrecognized queries
    return {
        "visualization_path": None,
        "analysis": "I couldn't find specific information about that in the previous analysis. Would you like me to perform a new search?",
        "analysis_context": "followup_generic_not_found",
        "processed_smiles_for_tools": reaction_smiles
    }

def format_value(value):
    """Clean and format extracted values"""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value if str(v).strip())
    if isinstance(value, str) and value.lower() in ['nan', 'none', '']:
        return "not specified"
    return str(value).strip()

def create_response(prop, value, reaction_smiles):
    """Standardized response format"""
    return {
        "visualization_path": None,
        "analysis": f"The {prop} for this reaction is: {value}.",
        "analysis_context": f"followup_{prop}_direct_answer",
        "processed_smiles_for_tools": reaction_smiles
    }

def enhanced_query(full_query: str, callbacks=None, original_compound_name: str = None):
    # ... (enhanced_query function as before, ensure it has the debug print after calling handle_full_info) ...
    final_result = {}
    query_context_for_filename = "unknown_query_type"
    reaction_smiles_for_tools = extract_reaction_smiles(full_query) 
    
    print(f"[ENHANCED_QUERY] Original query: '{full_query[:100]}...'")
    if reaction_smiles_for_tools:
        print(f"[ENHANCED_QUERY] Extracted/Converted SMILES for tools: '{reaction_smiles_for_tools}'")
    else:
        print(f"[ENHANCED_QUERY] No reaction SMILES extracted or converted from query.")

    try:
        if not reaction_smiles_for_tools:
            # ... (general agent call) ...
            print("[ENHANCED_QUERY] No reaction SMILES identified. Using general agent with the full query.")
            agent_output = agent.invoke({"input": full_query}, {"callbacks": callbacks} if callbacks else {})
            final_result = {
                "visualization_path": None,
                "analysis": extract_final_answer(agent_output.get("output", "Agent did not provide a final answer.")),
            }
            query_context_for_filename = "general_agent_no_smiles"
        
        elif "full" in full_query.lower() and any(term in full_query.lower() for term in ["information", "analysis", "detail", "explain", "tell me about", "give me all", "everything about"]):
            print(f"[ENHANCED_QUERY] Full analysis explicitly requested for extracted SMILES: {reaction_smiles_for_tools}")
            final_result = handle_full_info(full_query, reaction_smiles_for_tools, original_compound_name, callbacks=callbacks)
            # --- Add this block ---
            print(f"\n--- [POST_HANDLE_FULL_INFO_CACHE_CHECK in enhanced_query] ---")
            print(f"SMILES for check: '{reaction_smiles_for_tools}'")
            if reaction_smiles_for_tools in reaction_cache and 'full_info' in reaction_cache.get(reaction_smiles_for_tools, {}):
                cached_item_after_full_info = reaction_cache[reaction_smiles_for_tools]['full_info']
                if isinstance(cached_item_after_full_info, dict) and 'analysis' in cached_item_after_full_info and \
                   not cached_item_after_full_info.get('analysis_context','').endswith(("_exception", "_error")):
                    print(f"  SUCCESS: VALID 'full_info' IS present in cache for '{reaction_smiles_for_tools}' immediately after handle_full_info call. Context: {cached_item_after_full_info.get('analysis_context')}")
                else:
                    print(f"  WARNING: 'full_info' IS present but seems INVALID/ERROR after handle_full_info call for '{reaction_smiles_for_tools}'. Context: {cached_item_after_full_info.get('analysis_context') if isinstance(cached_item_after_full_info, dict) else 'Not a dict'}")

            else:
                print(f"  !!! CRITICAL FAILURE: 'full_info' IS MISSING in cache for '{reaction_smiles_for_tools}' immediately after handle_full_info call. !!!")
            print(f"--- [END_POST_HANDLE_FULL_INFO_CACHE_CHECK] ---\n")
            # --- End of added block ---
            query_context_for_filename = final_result.get('analysis_context', 'full_summary_fallback')
        
        elif any(term in full_query.lower() for term in ["visual", "picture", "image", "show", "draw", "representation", "diagram"]):
            # ... (visualization logic) ...
            print(f"[ENHANCED_QUERY] Visualization requested for extracted SMILES: {reaction_smiles_for_tools}")
            viz_path_cached = reaction_cache.get(reaction_smiles_for_tools, {}).get('visualization_path')
            # Attempt to get from full_info if not at top level
            if not viz_path_cached and 'full_info' in reaction_cache.get(reaction_smiles_for_tools, {}):
                viz_path_cached = reaction_cache[reaction_smiles_for_tools]['full_info'].get('visualization_path')

            if viz_path_cached and not str(viz_path_cached).lower().startswith("error"):
                final_result = {"visualization_path": viz_path_cached, "analysis": f"Cached visual representation for: {reaction_smiles_for_tools}"}
                query_context_for_filename = "visualization_cached"
            else: 
                tool_dict = {tool.name.lower(): tool for tool in tools}
                visualizer_tool = tool_dict.get("chemvisualizer")
                if visualizer_tool:
                    try:
                        viz_path = visualizer_tool.run(reaction_smiles_for_tools)
                        if viz_path and not str(viz_path).lower().startswith("error") and str(viz_path).endswith(".png"):
                            reaction_cache.setdefault(reaction_smiles_for_tools, {})['visualization_path'] = viz_path
                            if 'full_info' in reaction_cache.get(reaction_smiles_for_tools, {}):
                                reaction_cache[reaction_smiles_for_tools]['full_info']['visualization_path'] = viz_path
                            final_result = {"visualization_path": viz_path, "analysis": f"Generated visual representation for: {reaction_smiles_for_tools}"}
                            query_context_for_filename = "visualization_generated"
                        else:
                            final_result = {"visualization_path": None, "analysis": f"Visualization tool message: {viz_path}"}
                            query_context_for_filename = "visualization_tool_error"
                    except Exception as e:
                        final_result = {"visualization_path": None, "analysis": f"Error visualizing reaction '{reaction_smiles_for_tools}': {str(e)}"}
                        query_context_for_filename = "visualization_exception"
                else:
                    final_result = {"visualization_path": None, "analysis": "ChemVisualizer tool not found."}
                    query_context_for_filename = "visualization_no_tool"


        elif reaction_smiles_for_tools in reaction_cache or \
             any(keyword in full_query.lower() for keyword in [ # Broad keywords for routing to follow-up
                 "temperature", "yield", "solvent", "catalyst", "reagent", "time", "mechanism", "applications", 
                 "procedure", "protocol", "steps", "bonds", "functional group", "classification", "name", "identity",
                 "what is", "how does", "explain further", "why", "details on", "more about", "tell me more", "can you elaborate"
             ]):
            print(f"[ENHANCED_QUERY] Specific/Follow-up style question for extracted SMILES: {reaction_smiles_for_tools}")
            final_result = handle_followup_question(full_query, reaction_smiles_for_tools, original_compound_name, callbacks=callbacks)
            query_context_for_filename = final_result.get('analysis_context', 'followup_answer_fallback')
        
        else: # Default to full analysis if SMILES is new and query is not clearly a follow-up
            print(f"[ENHANCED_QUERY] Defaulting to full analysis for newly identified/extracted reaction: {reaction_smiles_for_tools}")
            final_result = handle_full_info(full_query, reaction_smiles_for_tools, original_compound_name, callbacks=callbacks)
            # --- Add this block for debugging default full_info calls too ---
            print(f"\n--- [POST_HANDLE_FULL_INFO_CACHE_CHECK (Default Path) in enhanced_query] ---")
            print(f"SMILES for check: '{reaction_smiles_for_tools}'")
            if reaction_smiles_for_tools in reaction_cache and 'full_info' in reaction_cache.get(reaction_smiles_for_tools, {}):
                cached_item_after_default_full_info = reaction_cache[reaction_smiles_for_tools]['full_info']
                if isinstance(cached_item_after_default_full_info, dict) and 'analysis' in cached_item_after_default_full_info and \
                   not cached_item_after_default_full_info.get('analysis_context','').endswith(("_exception", "_error")):
                    print(f"  SUCCESS: VALID 'full_info' IS present in cache for '{reaction_smiles_for_tools}' after default handle_full_info call. Context: {cached_item_after_default_full_info.get('analysis_context')}")
                else:
                    print(f"  WARNING: 'full_info' IS present but seems INVALID/ERROR after default handle_full_info call for '{reaction_smiles_for_tools}'. Context: {cached_item_after_default_full_info.get('analysis_context') if isinstance(cached_item_after_default_full_info, dict) else 'Not a dict'}")
            else:
                print(f"  !!! CRITICAL FAILURE: 'full_info' IS MISSING in cache for '{reaction_smiles_for_tools}' after default handle_full_info call. !!!")
            print(f"--- [END_POST_HANDLE_FULL_INFO_CACHE_CHECK (Default Path)] ---\n")
            # --- End of added block ---
            query_context_for_filename = final_result.get('analysis_context', 'full_summary_default_new_smiles')

        analysis_text_to_save = final_result.get("analysis")
        smiles_for_saving = reaction_smiles_for_tools
        
        if smiles_for_saving and analysis_text_to_save and isinstance(analysis_text_to_save, str) and \
           not query_context_for_filename.startswith("visualization_") and \
           "error" not in query_context_for_filename.lower() and \
           len(analysis_text_to_save.strip()) > 20 :
            save_analysis_to_file(smiles_for_saving, analysis_text_to_save, query_context_for_filename, original_compound_name)
        
        final_result['processed_smiles_for_tools'] = reaction_smiles_for_tools
        return final_result

    except Exception as e:
        # ... (exception handling) ...
        tb_str = traceback.format_exc()
        print(f"CRITICAL Error in enhanced_query for query '{full_query}': {str(e)}\n{tb_str}")
        error_text = f"Error processing your query: {str(e)}. Please check the logs for details."
        smiles_ctx_for_error_log = reaction_smiles_for_tools if reaction_smiles_for_tools else "no_smiles_extracted"
        save_analysis_to_file(smiles_ctx_for_error_log, f"Query: {full_query}\n{error_text}\n{tb_str}", "enhanced_query_CRITICAL_error", original_compound_name)
        
        return {
            "visualization_path": None, 
            "analysis": error_text,
            "processed_smiles_for_tools": reaction_smiles_for_tools
        }
