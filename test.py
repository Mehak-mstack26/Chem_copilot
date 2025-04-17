# Final part 
# import os
# import re
# from dotenv import load_dotenv
# load_dotenv()

# from tools.make_tools import make_tools 
# from langchain.agents import AgentExecutor, ZeroShotAgent
# from langchain.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI

# # Setup LLM and tools
# llm = ChatOpenAI(model="gpt-4o", temperature=0)
# tools = make_tools(llm=llm)

# # Prompt parts
# PREFIX = """
# You are Chem Copilot, an expert chemistry assistant. You have access to the following tools to analyze molecules and chemical reactions.

# Always begin by understanding the user's **intent** — what kind of information are they asking for?

# Here is how to choose tools:

# - - If the user gives a SMILES or reaction SMILES and asks for the name, you MUST use **SMILES2Name**. Do NOT analyze bonds or functional groups for this task.
# - Use **NameToSMILES**: when the user gives a compound/reaction name and wants the SMILES or structure.
# - Use **FuncGroups**: when the user wants to analyze functional groups in a molecule or a reaction (input is SMILES or reaction SMILES).
# - Use **BondChangeAnalyzer**: when the user asks for which bonds are broken, formed, or changed in a chemical reaction.

# If the user wants all of the above (full analysis), respond with "This requires full analysis." (This will be handled by a separate function.)

# Always return your answer in this format:
# Final Answer: <your answer here>

# For **FuncGroups** results:
# - Always list the functional groups identified in each reactant and product separately
# - Include the transformation summary showing disappeared groups, appeared groups, and unchanged groups
# - Provide a clear conclusion about what transformation occurred in the reaction

# For **BondChangeAnalyzer** results:
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
#     tools=tools,
#     prefix=PREFIX,
#     suffix=SUFFIX,
#     format_instructions=FORMAT_INSTRUCTIONS,
#     input_variables=["input", "agent_scratchpad"]
# )

# agent_chain = ZeroShotAgent.from_llm_and_tools(llm=llm, tools=tools, prompt=prompt)
# agent = AgentExecutor(agent=agent_chain, tools=tools, verbose=True)

# def extract_final_answer(full_output: str):
#     match = re.search(r"Final Answer:\s*(.*)", full_output, re.DOTALL)
#     return match.group(1).strip() if match else full_output.strip()

# # 🧠 Full Info Handler
# def handle_full_info(query, reaction_smiles):
#     print("Running full analysis using all tools...\n")
#     full_info = {}

#     # Map tool names to classes
#     tool_dict = {tool.name.lower(): tool for tool in tools}

#     # Run smiles2name
#     name_tool = tool_dict.get("smiles2name")
#     full_info['Names'] = name_tool.run(reaction_smiles)

#     # Run funcgroups
#     fg_tool = tool_dict.get("funcgroups")
#     full_info['Functional Groups'] = fg_tool.run(reaction_smiles)

#     # Run bond.py
#     bond_tool = tool_dict.get("bondchangeanalyzer")
#     full_info['Bond Changes'] = bond_tool.run(reaction_smiles)

#     # Optional: name2smiles
#     # n2s_tool = tool_dict.get("nametosmiles")

#     # Combine and ask GPT for final answer
#     final_prompt = f"""You are a chemistry expert. Here is a full reaction analysis:
    
# Reaction SMILES: {reaction_smiles}

# Step-by-step tool outputs:
# - Compound/Reaction Names: {full_info['Names']}
# - Functional Groups: {full_info['Functional Groups']}
# - Bond Changes: {full_info['Bond Changes']}

# Please give a complete and readable explanation of this reaction.

# Answer:"""

#     response = llm.invoke(final_prompt)
#     return response.content.strip()

# # 🔍 Main query function
# def enhanced_query(query, callbacks=None):
#     try:
#         # Check if it's a full analysis request
#         if "full information" in query.lower():
#             # Extract SMILES from query using regex
#             match = re.search(r"([A-Za-z0-9@\[\]\.\+\-\=\#\(\)]+>>[A-Za-z0-9@\[\]\.\+\-\=\#\(\)\.]*)", query)
#             if match:
#                 reaction_smiles = match.group(1)
#                 return handle_full_info(query, reaction_smiles)
#             else:
#                 return "Could not extract reaction SMILES from the query."

#         # Otherwise, use normal agent
#         result = agent.invoke({"input": query}, {"callbacks": callbacks} if callbacks else {})
#         return extract_final_answer(result.get("output", ""))

#     except Exception as e:
#         return f"Error occurred: {str(e)}"

# # Run
# if __name__ == "__main__":
#     # query = "Give full information about this rxn CCCl.CC[O-].[Na+]>>CCOCC.[Na+].[Cl-]"
#     query = "Give full information about this rxn CBr.[Na+].[I-]>>CI.[Na+].[Br-]"
#     # query = "Give full information about this rxn C(O)(=O)C1=CC=CC=C1.[N+](=[N-])=N>>C(N)(=O)C1=CC=CC=C1.N#N"
#     # query = "What is the reaction name having this smiles CCCl.CC[O-].[Na+] >> CCOCC.[Na+].[Cl-]"
#     print(enhanced_query(query))





import json
import os
import re
from dotenv import load_dotenv
load_dotenv()

from tools.make_tools import make_tools
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# --- Helper function to format FuncGroups JSON ---
# (Keep your existing format_funcgroups_json function - it looks good)
def format_funcgroups_json(json_input):
    """Convert the functional groups JSON (dict or string) to readable sentences."""
    try:
        # Ensure input is a dictionary
        if isinstance(json_input, str):
            try:
                data = json.loads(json_input)
            except json.JSONDecodeError:
                 # Handle cases where the tool might return plain error string
                 return f"Functional group data could not be parsed:\n{json_input}"
        elif isinstance(json_input, dict):
            data = json_input
        else:
            return f"Invalid format for functional group data: {type(json_input)}"

        sentences = []

        # Reactants section
        if 'reactants' in data and data['reactants']:
            sentences.append("**Reactants Analysis:**")
            for i, reactant in enumerate(data['reactants']):
                if isinstance(reactant, dict): # Check if reactant is a dict
                    if "error" in reactant:
                        sentences.append(f"- Reactant {i+1}: Processing Error - {reactant['error']}")
                        continue
                    smiles = reactant.get('smiles', 'Unknown SMILES')
                    groups = reactant.get('functional_groups', [])
                    if groups:
                        sentences.append(f"- Reactant {i+1} (`{smiles}`) contains: {', '.join(groups)}.")
                    else:
                        sentences.append(f"- Reactant {i+1} (`{smiles}`) has no identifiable functional groups.")
                else:
                     sentences.append(f"- Reactant {i+1}: Invalid format - {reactant}") # Handle unexpected reactant format

        # Products section
        if 'products' in data and data['products']:
            sentences.append("\n**Products Analysis:**")
            for i, product in enumerate(data['products']):
                 if isinstance(product, dict): # Check if product is a dict
                    if "error" in product:
                        sentences.append(f"- Product {i+1}: Processing Error - {product['error']}")
                        continue
                    smiles = product.get('smiles', 'Unknown SMILES')
                    groups = product.get('functional_groups', [])
                    if groups:
                        sentences.append(f"- Product {i+1} (`{smiles}`) contains: {', '.join(groups)}.")
                    else:
                        sentences.append(f"- Product {i+1} (`{smiles}`) has no identifiable functional groups.")
                 else:
                     sentences.append(f"- Product {i+1}: Invalid format - {product}") # Handle unexpected product format


        # Transformation summary
        if 'transformation_summary' in data:
            summary = data.get('transformation_summary', {}) # Use .get for safety
            if isinstance(summary, dict): # Check if summary is a dict
                sentences.append("\n**Transformation Summary:**")

                disappeared = summary.get('disappeared_groups', [])
                sentences.append(f"- Disappeared groups: {', '.join(disappeared) if disappeared else 'None'}.")

                appeared = summary.get('appeared_groups', [])
                sentences.append(f"- Appeared groups: {', '.join(appeared) if appeared else 'None'}.")

                unchanged = summary.get('unchanged_groups', [])
                sentences.append(f"- Unchanged groups: {', '.join(unchanged) if unchanged else 'None'}.")
            else:
                sentences.append("\n**Transformation Summary:** Invalid format")

        return "\n".join(sentences)

    except Exception as e:
        # Fallback for any unexpected error during formatting
        error_info = f"Error formatting functional groups: {str(e)}"
        # Try to include raw data if possible
        raw_data_str = str(json_input)
        if len(raw_data_str) < 200: # Avoid printing huge raw data
             error_info += f"\nRaw data: {raw_data_str}"
        return error_info

# Setup LLM and tools
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = make_tools(llm=llm)

# --- Prompt parts (Keep PREFIX, FORMAT_INSTRUCTIONS, SUFFIX as before) ---
PREFIX = """
You are Chem Copilot, an expert chemistry assistant. You have access to the following tools to analyze molecules and chemical reactions.

Always begin by understanding the user's **intent** — what kind of information are they asking for?

Here is how to choose tools:

- Use **SMILES2Name**: If the user gives a SMILES or reaction SMILES and asks for the name. Do NOT analyze bonds or functional groups for this task.
- Use **NameToSMILES**: when the user gives a compound/reaction name and wants the SMILES or structure.
- Use **FuncGroups**: when the user wants to analyze functional groups in a molecule or a reaction (input is SMILES or reaction SMILES).
- Use **BondChangeAnalyzer**: when the user asks for which bonds are broken, formed, or changed in a chemical reaction.

If the user wants all of the above (full analysis), respond with "This requires full analysis." (This will be handled by a separate function.)

**Formatting Tool Outputs in your Final Answer (Agent Only):**

When you use the **FuncGroups** tool and receive its output, format it in your Final Answer like this (if the query asks for functional groups):

**Functional Group Analysis:**

**Reactants:**
- Molecule (SMILES: <reactant_smiles>): Identified Groups: <list_of_groups>
  (Repeat for each reactant. If there's an error for a reactant, state the error.)

**Products:**
- Molecule (SMILES: <product_smiles>): Identified Groups: <list_of_groups>
  (Repeat for each product.)

**Transformation Summary:**
- Disappeared Functional Groups: <list_of_disappeared_groups>
- Appeared Functional Groups: <list_of_appeared_groups>
- Unchanged Functional Groups: <list_of_unchanged_groups>

**Conclusion:** <Provide a clear conclusion about the functional group transformation that occurred.>

When you use the **BondChangeAnalyzer** tool and receive its output, format it in your Final Answer like this (if the query asks for bond changes):

**Bond Change Analysis:**

- **Broken Bonds:** <List specific bonds broken, e.g., C-Cl (single)>
- **Formed Bonds:** <List specific bonds formed, e.g., C-O (single)>
- **Changed Bonds:** <List specific bonds changed, e.g., C=C (double) to C-C (single)>

**Conclusion:** <Provide a clear conclusion summarizing the key bond changes.>

Always return your final response in this format:
Final Answer: <your formatted answer here>
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
    tools=tools,
    prefix=PREFIX,
    suffix=SUFFIX,
    format_instructions=FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad"]
)

agent_chain = ZeroShotAgent.from_llm_and_tools(llm=llm, tools=tools, prompt=prompt)
# Added handle_parsing_errors for robustness
agent = AgentExecutor(agent=agent_chain, tools=tools, verbose=True, handle_parsing_errors=True)

def extract_final_answer(full_output: str):
    # More robust regex
    match = re.search(r"Final Answer:\s*(.*)", full_output, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else full_output.strip()

# --- MODIFIED: handle_full_info now generates the final combined answer ---
def handle_full_info(query, reaction_smiles):
    print("Running full analysis using all tools...\n")
    full_info = {}
    raw_data = {} # To store raw tool outputs

    tool_dict = {tool.name.lower(): tool for tool in tools}

    try:
        # Run smiles2name
        name_tool = tool_dict.get("smiles2name")
        if name_tool:
             full_info['Names'] = name_tool.run(reaction_smiles)
        else:
             full_info['Names'] = "SMILES2Name tool not available."

        # Run funcgroups
        fg_tool = tool_dict.get("funcgroups")
        if fg_tool:
            raw_fg_result = fg_tool.run(reaction_smiles)
            raw_data['funcgroups'] = raw_fg_result # Store raw data
            # Format the result for the final prompt
            full_info['Functional Groups Formatted'] = format_funcgroups_json(raw_fg_result)
        else:
             full_info['Functional Groups Formatted'] = "FuncGroups tool not available."
             raw_data['funcgroups'] = None

        # Run bondchangeanalyzer
        bond_tool = tool_dict.get("bondchangeanalyzer")
        if bond_tool:
            # Assuming BondChangeAnalyzer returns a readable string based on agent prompt
            raw_bond_result = bond_tool.run(reaction_smiles)
            raw_data['bondchanges'] = raw_bond_result # Store raw data
            full_info['Bond Changes'] = raw_bond_result # Use directly in prompt
        else:
             full_info['Bond Changes'] = "BondChangeAnalyzer tool not available."
             raw_data['bondchanges'] = None

    except Exception as e:
        print(f"Error during tool execution in handle_full_info: {e}")
        # Return error and None for raw data
        return f"An error occurred during the full analysis: {e}", None, None

    # --- Combine results using LLM ---
    final_prompt_template = PromptTemplate(
        input_variables=["reaction_smiles", "names", "functional_groups", "bond_changes"],
        template="""You are a chemistry expert. Based on the following analysis of a chemical reaction, provide a comprehensive and easy-to-understand explanation. Combine the information smoothly into a final report.

Reaction SMILES: {reaction_smiles}

Analysis Details Provided:
---
**Names:**
{names}
---
**Functional Groups Analysis:**
{functional_groups}
---
**Bond Changes Analysis:**
{bond_changes}
---

**Comprehensive Explanation:**
(Combine the above information into a clear, readable summary of the reaction)"""
    )

    final_prompt_str = final_prompt_template.format(
        reaction_smiles=reaction_smiles,
        names=full_info.get('Names', 'Name analysis not available.'),
        functional_groups=full_info.get('Functional Groups Formatted', 'Functional group analysis not available.'),
        bond_changes=full_info.get('Bond Changes', 'Bond change analysis not available.')
    )

    print("\n--- Sending Final Prompt to LLM for Full Analysis Synthesis ---")
    # print(final_prompt_str) # Optional: print for debugging
    print("--- End of Final Prompt ---")

    try:
        response = llm.invoke(final_prompt_str)
        final_combined_answer = response.content.strip()
        # Return the combined answer AND the raw data for separate display
        return final_combined_answer, raw_data.get('funcgroups'), raw_data.get('bondchanges')
    except Exception as e:
        print(f"Error during final LLM synthesis in handle_full_info: {e}")
        error_msg = f"An error occurred while generating the final summary: {e}"
        # Return error message and the raw data collected so far
        return error_msg, raw_data.get('funcgroups'), raw_data.get('bondchanges')


# --- MODIFIED: enhanced_query structure ---
def enhanced_query(query, callbacks=None):
    """
    Determines the query type and calls the appropriate handler.

    Returns:
        tuple: (main_answer, raw_funcgroups_data, raw_bondchange_data)
               - main_answer: Either agent's final answer or the combined full analysis.
               - raw_funcgroups_data: Raw output from FuncGroups tool (or None).
               - raw_bondchange_data: Raw output from BondChangeAnalyzer (or None).
    """
    try:
        # Use a more robust regex for "full information"
        if re.search(r'\b(full|complete)\s+(info|information|analysis)\b', query, re.IGNORECASE):
            # Extract SMILES using regex (allow more chars like %)
            match = re.search(r"([A-Za-z0-9@\[\]\.\+\-\=\#\(\)\%\_]+>>[A-Za-z0-9@\[\]\.\+\-\=\#\(\)\.\%\_]+)", query)
            if match:
                reaction_smiles = match.group(1)
                print(f"Detected 'full analysis' request for SMILES: {reaction_smiles}")
                # Directly call handle_full_info which now returns the tuple
                return handle_full_info(query, reaction_smiles)
            else:
                 # If "full info" requested but no SMILES found, let agent try or return error
                 print("Detected 'full analysis' request but couldn't extract SMILES.")
                 # Option 1: Let agent try (might fail or ask for SMILES)
                 # result = agent.invoke({"input": query}, {"callbacks": callbacks} if callbacks else {})
                 # return extract_final_answer(result.get("output", "")), None, None
                 # Option 2: Return specific error
                 return "Full analysis requested, but could not extract reaction SMILES from the query.", None, None

        # --- Standard Agent Execution ---
        else:
            print("Passing query to standard agent...")
            result = agent.invoke({"input": query}, {"callbacks": callbacks} if callbacks else {})
            final_answer = extract_final_answer(result.get("output", ""))
            # Agent path doesn't return separate raw data (it's incorporated in final_answer)
            return final_answer, None, None

    except Exception as e:
        error_message = f"Error occurred during query processing: {str(e)}"
        print(error_message) # Log server-side
        # Return error message in the expected tuple format
        return f"An error occurred: {str(e)}", None, None


if __name__ == "__main__":
    # query = "Give full information about this rxn CCCl.CC[O-].[Na+]>>CCOCC.[Na+].[Cl-]"
    query = "Give full information about this rxn CBr.[Na+].[I-]>>CI.[Na+].[Br-]"
    # query = "Give full information about this rxn C(O)(=O)C1=CC=CC=C1.[N+](=[N-])=N>>C(N)(=O)C1=CC=CC=C1.N#N"
    # query = "What is the reaction name having this smiles CCCl.CC[O-].[Na+] >> CCOCC.[Na+].[Cl-]"
    print(enhanced_query(query))