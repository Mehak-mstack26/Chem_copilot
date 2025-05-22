import os
import re
import requests
try:
    import api_config
except ImportError:
    print("Warning: api_config.py not found. API keys might not be loaded if they are set there.")

from tools.make_tools import make_tools
# Existing Agent imports
from langchain.agents import AgentExecutor, ZeroShotAgent
# NEW: Imports for Conversational Agent and Memory
from langchain.memory import ConversationBufferMemory
from langchain.agents import ConversationalChatAgent # Or OpenAIFunctionsAgent for newer patterns with GPT-3.5/4
from langchain.schema import HumanMessage, AIMessage # For manually adding to memory

from langchain_openai import ChatOpenAI
from tools.asckos import ReactionClassifier
from functools import lru_cache
import time
import traceback
import pandas as pd

from rdkit import Chem, RDLogger
from typing import Optional
RDLogger.DisableLog('rdApp.*')

# --- Setup for Saving Analysis Files (remains the same) ---
try:
    PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    PROJECT_ROOT_DIR = os.getcwd()
REACTION_ANALYSIS_OUTPUT_DIR = os.path.join(PROJECT_ROOT_DIR, "reaction_analysis_outputs")
os.makedirs(REACTION_ANALYSIS_OUTPUT_DIR, exist_ok=True)

# --- Utility functions (sanitize_filename, save_analysis_to_file, extract_final_answer - remain the same) ---
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

# --- LLM, Tools, Caches, Reaction Classifier (remains the same) ---
llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=4000)
tools = make_tools(llm=llm) # tools are already defined
dataset_path1 = os.environ.get('REACTION_DATASET_PATH1', None)
dataset_path2 = os.environ.get('REACTION_DATASET_PATH2', None)
try:
    reaction_classifier = ReactionClassifier(dataset_path1, dataset_path2)
except Exception as e:
    print(f"Warning: Could not initialize ReactionClassifier: {e}. Classification features will be unavailable.")
    reaction_classifier = None

reaction_cache = {}
compound_cache = {}

# --- MODIFIED ZeroShotAgent setup ---
PREFIX = """
You are Chem Copilot, an expert chemistry assistant. You have access to the following tools to analyze molecules and chemical reactions.
Your primary function is to USE THE PROVIDED TOOLS. You MUST NOT answer questions using your general knowledge if a tool is not applicable.

Here is how to choose tools:
- If the user gives a SMILES or reaction SMILES and asks for the name, you MUST use SMILES2Name. Do NOT analyze bonds or functional groups for this task.
- Use NameToSMILES: when the user gives a compound/reaction name and wants the SMILES or structure.
- Use FuncGroups: when the user wants to analyze functional groups in a molecule or a reaction (input is SMILES or reaction SMILES).
- Use BondChangeAnalyzer: when the user asks for which bonds are broken, formed, or changed in a chemical reaction.

If the user wants all of the above (full analysis of a reaction SMILES), respond with "This requires full analysis." (This will be handled by a separate function.)

IMPORTANT: If the user's query does not clearly map to one of these tool uses, OR if it's a general knowledge question (e.g., "What is an element?", "Explain SN2 reactions in general"), you MUST respond with:
"Final Answer: I can only perform specific chemical analyses using my tools if you provide a SMILES string or a chemical name for tool-based processing (e.g., 'functional groups in ethanol', 'SMILES of water', 'visualize CCO'). I cannot answer general knowledge questions. Please provide a specific chemical entity or task for my tools."

Always return your answer in this format (unless following the IMPORTANT rule above):
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
OR a single "Final Answer" format.

To use a tool, use the following format:
Thought: (reflect on your progress and decide what to do next. This thought should explain WHY you are choosing an action and what you expect from it.)
Action: (the action name, should be one of [{tool_names}])
Action Input: (the input string to the action)

If you have gathered all the information needed to answer the user's original question, or if you must refuse to answer based on the IMPORTANT rule in your instructions, you MUST respond in the "Final Answer" format.
Thought: (A brief summary of why you believe you have the final answer, or why you are refusing to answer.)
Final Answer: (The comprehensive final answer to the original input question, or the refusal message.)
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
general_task_agent = AgentExecutor(agent=agent_chain, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=10)

# --- MODIFIED Conversational Agent Setup ---
CONVERSATIONAL_SYSTEM_MESSAGE = """
You are Chem Copilot, a chemistry assistant. You have access to a specific set of tools to answer questions about molecules and chemical reactions.
Your SOLE PURPOSE is to utilize these tools to answer chemical queries. You MUST NOT answer questions using general knowledge if a tool is not applicable.

Here's how you MUST operate:

1.  **Tool-Based Responses Only:**
    *   If the user provides a SMILES string (e.g., "CCO", "CC(=O)O>>CCC(=O)O"), you MUST use the most appropriate tool(s) (e.g., SMILES2Name, FuncGroups, ChemVisualizer, BondChangeAnalyzer) to answer their question about that SMILES.
    *   If the user asks for the SMILES of a chemical name (e.g., "What is the SMILES of ethanol?"), you MUST use the 'NameToSMILES' tool.
    *   If the user asks for a specific property or analysis of a named chemical that your tools can provide (e.g., "What are the functional groups in ethanol?", "Show me the structure of benzene"), you MUST:
        a. First, use 'NameToSMILES' to get the SMILES. If it fails, inform the user the name was not recognized and you cannot proceed with that part of the query. Do not try to guess.
        b. Then, use the relevant analysis tool with the obtained SMILES.
    *   If a tool provides an error or cannot find information, report that error or lack of information. Do not supplement with general knowledge.

2.  **Handling Queries Not Suited for Tools:**
    *   If the user's question does NOT provide a SMILES string, does NOT provide a chemical name that can be converted by NameToSMILES for further tool use, AND does NOT clearly ask for a task your tools can perform, you MUST respond with:
        "I am a specialized chemistry assistant that uses tools to analyze specific chemical entities (SMILES or recognized names) or perform defined tasks (like conversions). I cannot answer general knowledge questions (e.g., 'What is an element?', 'Explain quantum mechanics'). Could you please provide a SMILES, a chemical name for analysis, or ask for a specific tool-based operation?"
    *   ABSOLUTELY DO NOT attempt to answer general chemistry knowledge questions, definitions, or explanations that fall outside the direct output of your tools. If unsure, err on the side of stating you cannot answer with your tools.

Always prioritize using your tools. Base your answer strictly on the tool's output if a tool is used.
Maintain conversation context, especially regarding established chemical entities, using the provided chat history.
"""

conversational_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

conversational_chat_agent_runnable = ConversationalChatAgent.from_llm_and_tools(
    llm=llm,
    tools=tools,
    system_message=CONVERSATIONAL_SYSTEM_MESSAGE,
    verbose=True # Keep verbose True for debugging, can be False for production
)

chatbot_agent_executor = AgentExecutor(
    agent=conversational_chat_agent_runnable,
    tools=tools,
    memory=conversational_memory,
    verbose=False, # Overall executor verbosity
    handle_parsing_errors=True,
    max_iterations=10,
    return_intermediate_steps=False
)

# --- NEW Memory Management Functions ---
def clear_chatbot_memory():
    """Clears the global conversational_memory."""
    global conversational_memory
    if conversational_memory:
        conversational_memory.clear()
        print("[MEMORY_MANAGEMENT] Chatbot memory cleared.")

def add_to_chatbot_memory(human_message: str, ai_message: str):
    """Manually adds a human/AI exchange to the chatbot's memory."""
    global conversational_memory
    if conversational_memory and human_message and ai_message:
        try:
            conversational_memory.chat_memory.add_user_message(human_message)
            conversational_memory.chat_memory.add_ai_message(ai_message)
            print(f"[MEMORY_MANAGEMENT] Added to chatbot memory: H: '{human_message[:50]}...', A: '{ai_message[:50]}...'")
        except Exception as e:
            print(f"[MEMORY_MANAGEMENT_ERROR] Could not add to memory: {e}")


def run_chatbot_query(user_input: str, callbacks=None):
    print(f"[CHATBOT_QUERY] Processing: '{user_input[:100]}...'")
    try:
        # Ensure chat history is correctly formatted if we were to inspect it
        # print(f"Current chat history before invoke: {conversational_memory.load_memory_variables({})}")

        response = chatbot_agent_executor.invoke(
            {"input": user_input},
            config={"callbacks": callbacks} if callbacks else None
        )

        ai_response_text = response.get("output", "Chatbot did not provide a clear answer.")

        viz_path_agent = None
        if "static/visualizations/" in ai_response_text:
            match_viz = re.search(r"(static/visualizations/[\w\-\.\_]+\.png)", ai_response_text)
            if match_viz: viz_path_agent = match_viz.group(1)
        
        # print(f"[CHATBOT_QUERY] Raw response from executor: {response}")
        # print(f"Current chat history after invoke: {conversational_memory.load_memory_variables({})}")


        return {
            "visualization_path": viz_path_agent,
            "analysis": ai_response_text
        }

    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Error in run_chatbot_query: {e}\n{tb_str}")
        return {
            "visualization_path": None,
            "analysis": f"An error occurred in the chatbot: {str(e)}"
        }

# --- Data extraction and processing functions (remain the same) ---
@lru_cache(maxsize=100)
def query_reaction_dataset(reaction_smiles):
    if not reaction_smiles: return None
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

def extract_single_compound_smiles(query: str) -> Optional[str]:
    words = query.split()
    regex_candidates = re.findall(r"[A-Za-z0-9@\[\]\(\)\+\-\=\#\:\.\$\%\/\\\{\}]{3,}", query)
    combined_candidates = list(set(words + regex_candidates))
    combined_candidates.sort(key=lambda x: (len(x), sum(1 for c in x if c in '()[]=#')), reverse=True)

    for s_cand in combined_candidates:
        s_cand = s_cand.strip('.,;:)?!\'"')
        if not s_cand: continue
        if '>>' in s_cand or '>' in s_cand or '<' in s_cand:
            continue
        if s_cand.isnumeric() and not ('[' in s_cand and ']' in s_cand) :
            continue
        try:
            mol = Chem.MolFromSmiles(s_cand, sanitize=True)
            if mol:
                num_atoms = mol.GetNumAtoms()
                if num_atoms >= 1:
                    if num_atoms <= 2 and s_cand.isalpha() and s_cand.lower() in [
                        'as', 'in', 'is', 'at', 'or', 'to', 'be', 'of', 'on', 'no', 'do', 'go',
                        'so', 'if', 'it', 'me', 'my', 'he', 'we', 'by', 'up', 'us', 'an', 'am', 'are'
                    ]:
                        continue
                    if any(c in s_cand for c in '()[]=#.-+@:/\\%{}') or num_atoms > 2 or len(s_cand) > 3:
                        print(f"[EXTRACT_SINGLE_COMPOUND_SMILES] Validated candidate: {s_cand} (Atoms: {num_atoms})")
                        return s_cand
        except Exception:
            pass
    print(f"[EXTRACT_SINGLE_COMPOUND_SMILES] No suitable compound SMILES found in query: '{query[:50]}...'")
    return None

# --- Core analysis functions (handle_full_info, handle_compound_full_info, handle_followup_question - remain the same) ---
def handle_full_info(query_text_for_llm_summary, reaction_smiles_clean, original_compound_name=None, callbacks=None):
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

        if reaction_classifier:
            try:
                classifier_result_raw = reaction_classifier._run(reaction_smiles_clean)
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
            full_info_results['Reaction Classification'] = "ReactionClassifier not available"
            reaction_cache[reaction_smiles_clean]['classification_info'] = "ReactionClassifier not available"

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
            "\nPresent the information clearly and logically for a chemist. Focus ONLY on the provided data." # Added focus instruction
        )
        final_prompt_for_llm = "\n\n".join(final_prompt_parts)
        
        focused_llm_full_summary = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=2000)
        response = focused_llm_full_summary.invoke(final_prompt_for_llm, {"callbacks": callbacks} if callbacks else None)
        analysis_text_summary = response.content.strip()

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
            'analysis_context': "full_llm_summary_generated",
            'processed_smiles_for_tools': reaction_smiles_clean
        }
        
        reaction_cache[reaction_smiles_clean]['full_info'] = structured_result_for_full_info_cache
        
        print(f"\n--- [CACHE_STATE_AFTER_FULL_INFO_STORE for '{reaction_smiles_clean}'] ---")
        # ... (rest of print statements remain the same)
        
        return structured_result_for_full_info_cache

    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"CRITICAL ERROR in handle_full_info for {reaction_smiles_clean}: {e}\n{tb_str}")
        # ... (rest of error handling remains the same)
        error_result = {
            'visualization_path': None, 
            'analysis': f"An internal error occurred during the full analysis of '{reaction_smiles_clean}'. Details: {str(e)}",
            'analysis_context': "full_analysis_exception", 
            'processed_smiles_for_tools': reaction_smiles_clean
        }
        reaction_cache.setdefault(reaction_smiles_clean, {})['full_info'] = error_result
        return error_result


def handle_compound_full_info(query_text_for_summary_context, compound_smiles, original_compound_name=None, callbacks=None):
    print(f"\n--- [HANDLE_COMPOUND_FULL_INFO for '{compound_smiles}'] ---")
    
    if not compound_smiles:
        return {'visualization_path': None, 'analysis': "Error: No valid compound SMILES provided.",
                'analysis_context': "invalid_compound_smiles", 'processed_smiles_for_tools': None}

    if compound_smiles in compound_cache and 'full_compound_info' in compound_cache[compound_smiles]:
        cached_data = compound_cache[compound_smiles]['full_compound_info']
        if isinstance(cached_data, dict) and 'analysis' in cached_data and \
           not cached_data.get('analysis_context', '').endswith(("_error", "_exception")):
            print(f"Using CACHED full_compound_info for: {compound_smiles}")
            return cached_data
        else:
            print(f"Cached 'full_compound_info' for {compound_smiles} found but invalid/error. Regenerating.")

    compound_cache.setdefault(compound_smiles, {})
    info_results = {}
    tool_dict = {tool.name.lower(): tool for tool in tools}

    # ... (visualization, smiles2name, funcgroups tool calls remain the same)
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

    s2n_tool = tool_dict.get("smiles2name")
    name_info_str = "Not available"
    if s2n_tool:
        try:
            name_result = s2n_tool.run(compound_smiles)
            if isinstance(name_result, dict): 
                iupac = name_result.get('iupac_name', '')
                common = name_result.get('common_name', '')
                name_info_str = f"IUPAC: {iupac}" if iupac else "IUPAC name not found"
                if common and common.lower() != "no widely recognized common name": 
                    name_info_str += f", Common: {common}"
            elif isinstance(name_result, str): 
                iupac_match = re.search(r"IUPAC name:\s*(.+)", name_result, re.IGNORECASE)
                common_match = re.search(r"Common name:\s*(.+)", name_result, re.IGNORECASE)
                i_name = iupac_match.group(1).strip() if iupac_match else "Not found"
                c_name = common_match.group(1).strip() if common_match else ""
                name_info_str = f"IUPAC: {i_name}"
                if c_name and "no widely recognized common name" not in c_name.lower() and c_name.lower() != "none":
                     name_info_str += f", Common: {c_name}"
            else: 
                name_info_str = str(name_result)
            info_results['Name'] = name_info_str
        except Exception as e:
            info_results['Name'] = f"Error getting name: {str(e)}"
    else:
        info_results['Name'] = "SMILES2Name tool not found"
    compound_cache[compound_smiles]['name_info'] = info_results['Name']

    fg_tool = tool_dict.get("funcgroups")
    fg_info_str = "Not available"
    if fg_tool:
        try:
            fg_result = fg_tool.run(compound_smiles)
            if isinstance(fg_result, dict) and 'functional_groups' in fg_result:
                fg_list = fg_result['functional_groups']
                fg_info_str = ", ".join(fg_list) if fg_list else "None identified"
            elif isinstance(fg_result, str):
                fg_info_str = fg_result 
            else: 
                fg_info_str = str(fg_result)
            info_results['Functional Groups'] = fg_info_str
        except Exception as e:
            info_results['Functional Groups'] = f"Error getting functional groups: {str(e)}"
    else:
        info_results['Functional Groups'] = "FuncGroups tool not found"
    compound_cache[compound_smiles]['fg_info'] = info_results['Functional Groups']
    
    llm_prompt_parts = [
        f"Provide a comprehensive overview of the compound with SMILES: {compound_smiles}.",
        f"User query context: '{query_text_for_summary_context}'",
        "Information gathered from tools:", # Emphasize tool origin
        f"- Names: {info_results.get('Name', 'N/A')}",
        f"- Functional Groups: {info_results.get('Functional Groups', 'N/A')}",
    ]
    if viz_path:
        llm_prompt_parts.append(f"- A 2D structure image has been generated.")
    else:
        llm_prompt_parts.append(f"- Structure visualization: {info_results.get('Visualization Info', 'Not attempted or failed.')}")

    llm_prompt_parts.extend([
        "\nBased ONLY on the information gathered from the tools above, please provide a well-structured summary including:", # Stricter instruction
        "1. Chemical names (IUPAC and common, if available from tools).",
        "2. A list of identified functional groups (from tools).",
        "3. A brief interpretation strictly based on these features (e.g., potential properties, reactivity, common uses or class of compounds that can be inferred SOLELY from the identified FGs and structure).",
        "4. Mention that a visualization is available if one was generated.",
        "Present this as a chemist would expect, clearly and concisely. Do not add any information not derivable from the provided tool outputs." # Stricter instruction
    ])
    final_llm_prompt = "\n".join(llm_prompt_parts)

    final_analysis_text = "Error generating summary."
    try:
        # This LLM call is for summarization, not general Q&A
        llm_summarizer = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=1500) # Can be a different model/config
        llm_response = llm_summarizer.invoke(final_llm_prompt, {"callbacks": callbacks} if callbacks else None)
        final_analysis_text = llm_response.content.strip()
    except Exception as e:
        final_analysis_text = f"Error generating LLM summary for compound: {str(e)}\n\nRaw data:\nName: {info_results.get('Name')}\nFGs: {info_results.get('Functional Groups')}"
        print(f"LLM summary generation for compound failed: {e}")

    result_struct = {
        'visualization_path': viz_path,
        'analysis': final_analysis_text,
        'name_info': compound_cache[compound_smiles].get('name_info'),
        'fg_info': compound_cache[compound_smiles].get('fg_info'),
        'analysis_context': "compound_full_info_generated",
        'processed_smiles_for_tools': compound_smiles
    }
    compound_cache[compound_smiles]['full_compound_info'] = result_struct
    
    print(f"--- [END HANDLE_COMPOUND_FULL_INFO for '{compound_smiles}'] ---")
    return result_struct

def handle_followup_question(query_text, reaction_smiles, original_compound_name=None, callbacks=None):
    # This function tries to answer follow-ups from cached data or the LLM summary.
    # If the follow-up is general and not found here, enhanced_query will route it to the chatbot.
    cached_reaction_data = reaction_cache.get(reaction_smiles, {})
    full_info = cached_reaction_data.get('full_info', {})
    llm_summary_analysis = full_info.get('analysis', '')

    property_map = {
        'solvent': {
            'keywords': ['solvent', 'solution', 'medium', 'dissolve'],
            'cache_key': 'solvents',
            'text_patterns': [
                r'solvents?:\s*([^\.]+)',
                r'carried out in\s*([^\.]+)',
                r'using\s*([^\(]+)\s*as solvent'
            ]
        },
        'temperature': {
            'keywords': ['temperature', 'temp', '°c', '°f', 'kelvin', 'heat', 'cool'],
            'cache_key': 'temperature',
            'text_patterns': [
                r'temperature:\s*([^\.]+)',
                r'temp\.?:\s*([^\.]+)',
                r'at\s*([\d\-]+)\s*°',
                r'heated to\s*([^\.]+)'
            ]
        },
        'yield': {
            'keywords': ['yield', '% yield', 'percentage', 'efficiency', 'obtained'],
            'cache_key': 'yield',
            'text_patterns': [
                r'yield:\s*([^\.]+)',
                r'([\d\.]+%) yield',
                r'obtained in\s*([^\.]+)\s*yield'
            ]
        },
        'time': {
            'keywords': ['time', 'duration', 'hour', 'minute', 'day'],
            'cache_key': 'reaction_time',
            'text_patterns': [
                r'time:\s*([^\.]+)',
                r'duration:\s*([^\.]+)',
                r'for\s*([\d\.]+\s*(?:h|hr|hrs|hours|minutes|mins|days?))',
                r'stirred for\s*([^\.]+)'
            ]
        },
        'catalyst': {
            'keywords': ['catalyst', 'reagent', 'agent', 'promoter', 'additive', 'initiator'],
            'cache_key': 'agents_catalysts',
            'text_patterns': [
                r'catalysts?:\s*([^\.]+)',
                r'reagents?:\s*([^\.]+)',
                r'using\s*([^\(]+)\s*as (?:catalyst|reagent)',
                r'(?:catalyzed|initiated|promoted) by\s*([^\.]+)',
                r'in the presence of\s*([^\.]+)'
            ]
        },
        'pressure': {
            'keywords': ['pressure', 'psi', 'bar', 'atm', 'atmosphere'],
            'cache_key': 'pressure', # Note: 'pressure' not typically in current dataset query
            'text_patterns': [
                r'pressure:\s*([^\.]+)',
                r'under\s*([^\.]+)\s*pressure',
                r'at\s*([\d\.]+)\s*(psi|bar|atm)'
            ]
        },
        'ph': {
            'keywords': ['ph', 'acidic', 'basic', 'neutral'],
            'cache_key': 'ph', # Note: 'ph' not typically in current dataset query
            'text_patterns': [
                r'ph:\s*([^\.]+)',
                r'at\s*ph\s*([\d\.]+)',
                r'under\s*(acidic|basic|neutral)\s*conditions'
            ]
        },
        'procedure': {
            'keywords': ['procedure', 'protocol', 'steps', 'method', 'preparation', 'synthesis steps'],
            'cache_key': 'procedure_details',
            'text_patterns': [
                r'procedure details:?\s*(.*?)(?:\n\n|\Z)',
                r'experimental steps:?\s*(.*?)(?:\n\n|\Z)',
                r'method:?\s*(.*?)(?:\n\n|\Z)'
            ]
        }
    }

    query_lower = query_text.lower()
    
    for prop_name, prop_details in property_map.items():
        if any(keyword in query_lower for keyword in prop_details['keywords']):
            cache_key_for_prop = prop_details.get('cache_key')
            # Check cache first
            if cache_key_for_prop and cache_key_for_prop in full_info and full_info[cache_key_for_prop] is not None:
                value = format_value(full_info[cache_key_for_prop])
                if value and value.lower() != "not specified" and value.lower() != "nan":
                    return create_response(prop_name, value, reaction_smiles)

            # Then check LLM summary text
            if llm_summary_analysis:
                for pattern in prop_details['text_patterns']:
                    match = re.search(pattern, llm_summary_analysis, re.IGNORECASE | re.DOTALL)
                    if match:
                        extracted_value = (match.group(1) if len(match.groups()) > 0 and match.group(1) else match.group(0)).strip(" .,:")
                        if extracted_value and extracted_value.lower() != "not available" and extracted_value.lower() != "n/a":
                            return create_response(prop_name, extracted_value, reaction_smiles)
            
            # If not found in either specific cache or parsed from summary
            return {
                "visualization_path": None,
                "analysis": f"Specific information about '{prop_name}' was not found in the cached reaction data or its summary. You can ask the chatbot for a new search.",
                "analysis_context": f"followup_{prop_name}_not_readily_found",
                "processed_smiles_for_tools": reaction_smiles
            }

    print(f"[FOLLOWUP_UNMATCHED] Query '{query_text}' did not match specific property keywords for cached data lookup.")
    # For unmatched follow-ups, enhanced_query will route to the chatbot.
    # This function returning a "not found" style message signals to enhanced_query
    # that this specific lookup failed, and it might proceed to chatbot.
    return {
        "visualization_path": None,
        "analysis": None, # Set to None to indicate this handler didn't provide a final answer
        "analysis_context": "followup_property_unmatched_in_cache", # Signal to enhanced_query
        "processed_smiles_for_tools": reaction_smiles
    }


def format_value(value):
    if isinstance(value, list):
        valid_items = [str(v) for v in value if v is not None and str(v).strip().lower() not in ['nan', 'none', '']]
        return ", ".join(valid_items) if valid_items else "not specified"
    if value is None or (isinstance(value, str) and value.strip().lower() in ['nan', 'none', '']):
        return "not specified"
    return str(value).strip()

def create_response(prop, value, reaction_smiles):
    prop_display_name = prop.replace('_', ' ')
    return {
        "visualization_path": None,
        "analysis": f"Regarding the {prop_display_name} for reaction {reaction_smiles}: {value}.",
        "analysis_context": f"followup_{prop}_direct_answer",
        "processed_smiles_for_tools": reaction_smiles
    }

def handle_general_query(full_query: str, callbacks=None):
    print("[HANDLE_GENERAL_QUERY_SINGLE_TURN] Processing query with general_task_agent (ZeroShot).")
    try:
        agent_output = general_task_agent.invoke({"input": full_query}, {"callbacks": callbacks} if callbacks else {})
        analysis_text = extract_final_answer(agent_output.get("output", "Agent did not provide a final answer."))
        
        viz_path_agent = None
        if "static/visualizations/" in analysis_text:
            match_viz = re.search(r"(static/visualizations/[\w\-\.\_]+\.png)", analysis_text)
            if match_viz: viz_path_agent = match_viz.group(1)

        return {
            "visualization_path": viz_path_agent,
            "analysis": analysis_text
        }
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Error in handle_general_query for query '{full_query[:100]}...': {e}\n{tb_str}")
        return {
            "visualization_path": None,
            "analysis": f"An error occurred while processing your general query with the ZeroShot agent. Details: {str(e)}"
        }

# --- MODIFIED Main Query Routing Logic ---
def enhanced_query(full_query: str, callbacks=None, original_compound_name: str = None):
    final_result = {}
    query_context_for_filename = "unknown_query_type"
    
    reaction_smiles_for_tools = extract_reaction_smiles(full_query)
    compound_smiles_for_tools = None

    if not reaction_smiles_for_tools:
        compound_smiles_for_tools = extract_single_compound_smiles(full_query)
        # ... (logging remains the same)

    print(f"[ENHANCED_QUERY] Original query: '{full_query[:100]}...'")
    # ... (logging remains the same)

    try:
        query_lower = full_query.lower()

        if reaction_smiles_for_tools:
            # ... (logic for full_info_request, visualization_request remains same)
            is_full_info_request = "full" in query_lower and any(term in query_lower for term in ["information", "analysis", "detail", "explain", "tell me about", "give me all", "everything about"])
            is_visualization_request = any(term in query_lower for term in ["visual", "picture", "image", "show", "draw", "representation", "diagram"])
            
            if is_full_info_request:
                print(f"[ENHANCED_QUERY] Full analysis explicitly requested for REACTION SMILES: {reaction_smiles_for_tools}")
                final_result = handle_full_info(full_query, reaction_smiles_for_tools, original_compound_name, callbacks=callbacks)
                query_context_for_filename = final_result.get('analysis_context', 'full_summary_fallback')
            elif is_visualization_request:
                # ... (visualization logic for reactions)
                print(f"[ENHANCED_QUERY] Visualization requested for REACTION SMILES: {reaction_smiles_for_tools}")
                # (Code for visualization handling remains largely the same)
                viz_path_cached = reaction_cache.get(reaction_smiles_for_tools, {}).get('visualization_path')
                # ... (rest of visualization handling)
                if viz_path_cached and not str(viz_path_cached).lower().startswith("error"):
                    final_result = {"visualization_path": viz_path_cached, "analysis": f"Cached visual representation for: {reaction_smiles_for_tools}"}
                    query_context_for_filename = "visualization_cached"
                else: 
                    tool_dict = {tool.name.lower(): tool for tool in tools}
                    visualizer_tool = tool_dict.get("chemvisualizer")
                    if visualizer_tool:
                        # ... (tool run and result handling)
                         pass # Placeholder for brevity
                    else:
                        final_result = {"visualization_path": None, "analysis": "ChemVisualizer tool not found."}
                        query_context_for_filename = "visualization_no_tool"
            # Attempt specific follow-up before defaulting to full analysis or chatbot
            elif reaction_smiles_for_tools in reaction_cache or any(keyword in query_lower for keyword in [
                 "temperature", "yield", "solvent", "catalyst", "time", "procedure", "bonds", "functional group",
                 "classification", "name", "what is", "how does", "explain further", "details on", "more about"
             ]):
                print(f"[ENHANCED_QUERY] Attempting specific/follow-up style question for REACTION SMILES: {reaction_smiles_for_tools}")
                followup_result = handle_followup_question(full_query, reaction_smiles_for_tools, original_compound_name, callbacks=callbacks)
                
                # If handle_followup_question provided a direct answer
                if followup_result.get('analysis'):
                    final_result = followup_result
                    query_context_for_filename = followup_result.get('analysis_context', 'followup_answer_fallback')
                else:
                    # If handle_followup_question couldn't find specific info, it's a more general query.
                    # Route to chatbot instead of full re-analysis if context is "followup_property_unmatched_in_cache".
                    print(f"[ENHANCED_QUERY] Specific follow-up for reaction failed or was too general. Defaulting to chatbot for: {full_query}")
                    chatbot_response = run_chatbot_query(full_query, callbacks=callbacks)
                    final_result = chatbot_response # Use the chatbot's response directly
                    query_context_for_filename = "chatbot_reaction_followup"
            else:
                print(f"[ENHANCED_QUERY] Defaulting to full analysis for newly identified REACTION: {reaction_smiles_for_tools} (or general query about it).")
                # If it's a new reaction SMILES and not a specific property query, do full info.
                # If it's a general question about a known reaction SMILES not caught by followup, could also do full info or chatbot.
                # For now, let's lean towards full_info for a new reaction.
                # If it was a general question about an existing reaction, handle_followup_question failing would lead to chatbot.
                # This branch is for *new* reactions not matching specific follow-up keywords.
                final_result = handle_full_info(full_query, reaction_smiles_for_tools, original_compound_name, callbacks=callbacks)
                query_context_for_filename = final_result.get('analysis_context', 'full_summary_default_new_smiles')
            
            final_result['processed_smiles_for_tools'] = reaction_smiles_for_tools

        elif compound_smiles_for_tools:
            full_info_keywords = ["full info", "full information", "all info", "details about", "tell me about", "explain this compound", "give the full info"]
            is_full_compound_info_request = any(keyword in query_lower for keyword in full_info_keywords) and \
                                            ("compound" in query_lower or "molecule" in query_lower or compound_smiles_for_tools in full_query)
            
            if is_full_compound_info_request:
                print(f"[ENHANCED_QUERY] Full info requested for COMPOUND SMILES: {compound_smiles_for_tools}")
                final_result = handle_compound_full_info(full_query, compound_smiles_for_tools, original_compound_name, callbacks=callbacks)
                query_context_for_filename = final_result.get('analysis_context', 'compound_full_info_generated')
            else:
                print(f"[ENHANCED_QUERY] COMPOUND SMILES '{compound_smiles_for_tools}' found. Query not 'full info' pattern. Using ZeroShot general agent for this specific compound query.")
                agent_result = handle_general_query(full_query, callbacks=callbacks)
                final_result = {
                    "visualization_path": agent_result.get("visualization_path"),
                    "analysis": agent_result.get("analysis")
                }
                query_context_for_filename = "general_agent_compound_context_zeroshot"
            final_result['processed_smiles_for_tools'] = compound_smiles_for_tools
            
        else: # No SMILES identified, use the conversational agent
            print("[ENHANCED_QUERY] No specific SMILES identified. Routing to conversational agent (chatbot).")
            chatbot_result = run_chatbot_query(full_query, callbacks=callbacks)
            final_result = {
                "visualization_path": chatbot_result.get("visualization_path"),
                "analysis": chatbot_result.get("analysis"),
                "processed_smiles_for_tools": None # Chatbot might not always yield a processed SMILES
            }
            query_context_for_filename = "chatbot_agent_no_smiles"
            # `processed_smiles_for_tools` is already set to None above for this branch

        # ... (rest of the function, saving logic, error handling remains the same)
        if reaction_smiles_for_tools and 'full_info' in reaction_cache.get(reaction_smiles_for_tools, {}):
            # ... cache check logging
            pass

        analysis_text_to_save = final_result.get("analysis")
        smiles_for_saving = final_result.get('processed_smiles_for_tools')
        
        if smiles_for_saving and analysis_text_to_save and isinstance(analysis_text_to_save, str) and \
           not query_context_for_filename.startswith("visualization_") and \
           "error" not in query_context_for_filename.lower() and \
           "not_found" not in query_context_for_filename.lower() and \
           ("chatbot_agent_no_smiles" not in query_context_for_filename or "I can only perform specific chemical analyses" not in analysis_text_to_save) and \
           len(analysis_text_to_save.strip()) > 20 : # Avoid saving short refusal messages from chatbot unless it's a specific analysis
            save_analysis_to_file(smiles_for_saving, analysis_text_to_save, query_context_for_filename, original_compound_name)
        
        if 'processed_smiles_for_tools' not in final_result:
             final_result['processed_smiles_for_tools'] = reaction_smiles_for_tools or compound_smiles_for_tools or None

        # Prime memory if a detailed analysis was done by non-chatbot handlers
        # and this is the first significant interaction that the chatbot should know about.
        # This logic can be refined. For now, relying on app.py to manage chat flow.
        # If `app.py` uses `enhanced_query` for initial and `run_chatbot_query` for follow-ups,
        # manual priming might be needed.
        # However, if `enhanced_query` now routes general queries to `run_chatbot_query`,
        # the memory will be built naturally by the chatbot.

        # If a detailed analysis (non-chatbot) happened, and the next turn might be a chatbot follow-up,
        # we could consider adding the detailed analysis to the chatbot's memory here.
        # Example:
        # current_query_is_initial = ... (needs a flag from app.py or infer)
        # if (query_context_for_filename.startswith("full_summary") or query_context_for_filename.startswith("compound_full_info")) and \
        #    final_result.get("analysis") and current_query_is_initial:
        #    if not conversational_memory.load_memory_variables({}).get("chat_history"): # If memory is empty
        #        add_to_chatbot_memory(full_query, final_result.get("analysis"))


        return final_result

    except Exception as e:
        tb_str = traceback.format_exc()
        # ... (error handling remains same)
        print(f"CRITICAL Error in enhanced_query for query '{full_query}': {str(e)}\n{tb_str}")
        error_text = f"Error processing your query: {str(e)}. Please check the logs for details."
        smiles_ctx_for_error_log = reaction_smiles_for_tools or compound_smiles_for_tools or "no_smiles_extracted"
        save_analysis_to_file(smiles_ctx_for_error_log, f"Query: {full_query}\n{error_text}\n{tb_str}", "enhanced_query_CRITICAL_error", original_compound_name)
        
        return {
            "visualization_path": None, 
            "analysis": error_text,
            "processed_smiles_for_tools": smiles_ctx_for_error_log 
        }


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY environment variable not set.")
        try:
            if api_config.OPENAI_API_KEY:
                os.environ["OPENAI_API_KEY"] = api_config.OPENAI_API_KEY
                print("Loaded OPENAI_API_KEY from api_config.py")
            else:
                print("OPENAI_API_KEY not found in api_config.py either. Exiting.")
                exit(1)
        except (AttributeError, NameError, ImportError): # Added ImportError
             print("api_config.py not found or OPENAI_API_KEY not defined within it. Exiting.")
             exit(1)
    
    # Test Case 0: General knowledge question (should be refused by chatbot)
    test_general_knowledge = "What is the difference between an element and a compound?"
    print(f"\n--- Running enhanced_query with (General Knowledge Test): ---")
    print(f"Query: {test_general_knowledge}")
    print(f"--- Starting execution ---\n")
    clear_chatbot_memory() # Clear memory before this test
    result_general = enhanced_query(full_query=test_general_knowledge)
    print(f"\n--- Final Result from enhanced_query (General Knowledge Test) ---")
    print(f"Analysis: {result_general.get('analysis')}")
    print(f"Processed SMILES for tools: {result_general.get('processed_smiles_for_tools')}")
    print(f"--- End of general knowledge test ---\n")

    # Test Case 1: Original failing query for compound (should go to handle_compound_full_info)
    test_query_compound = "Give the full info about this compound - C(O)(=O)C1=C(N)C(C)=CC(C#N)=C1"
    print(f"\n--- Running enhanced_query with (Compound Full Info Test): ---")
    print(f"Query: {test_query_compound}")
    print(f"--- Starting execution ---\n")
    clear_chatbot_memory() # Clear memory
    result_compound = enhanced_query(full_query=test_query_compound)
    print(f"\n--- Final Result from enhanced_query (Compound Full Info Test) ---")
    if result_compound.get("visualization_path"):
        print(f"Visualization Path: {result_compound['visualization_path']}")
    print(f"Analysis: {result_compound.get('analysis')}")
    print(f"Processed SMILES for tools: {result_compound.get('processed_smiles_for_tools')}")
    print(f"--- End of compound full info test ---\n")

    # Test Case 2: Compound query, not full info (should go to handle_general_query with ZeroShot)
    test_query_compound_specific = "What are the functional groups in CCO and what is water?"
    print(f"\n--- Running enhanced_query with (Compound Specific + General Knowledge Test): ---")
    print(f"Query: {test_query_compound_specific}")
    print(f"--- Starting execution ---\n")
    clear_chatbot_memory() # Clear memory
    result_compound_specific = enhanced_query(full_query=test_query_compound_specific)
    print(f"\n--- Final Result from enhanced_query (Compound Specific + General Test) ---")
    print(f"Analysis: {result_compound_specific.get('analysis')}")
    print(f"Processed SMILES for tools: {result_compound_specific.get('processed_smiles_for_tools')}")
    print(f"--- End of compound specific + general test ---\n")


    # Test Case 3: Chat-like interaction
    print(f"\n--- Running Chatbot Interaction Test ---")
    clear_chatbot_memory()
    q1 = "Hello, can you tell me the SMILES for Aspirin?"
    print(f"User: {q1}")
    a1 = run_chatbot_query(q1)
    print(f"ChemCopilot: {a1.get('analysis')}")

    q2 = "Thanks. Now, what are its functional groups?" # Follow-up, chatbot should use memory
    print(f"User: {q2}")
    a2 = run_chatbot_query(q2) # Aspirin SMILES should be in context for the chatbot
    print(f"ChemCopilot: {a2.get('analysis')}")
    
    q3 = "What is an element?" # General knowledge, should be refused by chatbot
    print(f"User: {q3}")
    a3 = run_chatbot_query(q3)
    print(f"ChemCopilot: {a3.get('analysis')}")
    print(f"--- End of Chatbot Interaction Test ---")