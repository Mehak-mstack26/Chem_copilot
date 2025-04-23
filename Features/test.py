# # Final part 
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
#     query = "Give full information about this rxn O=C(NC(=O)c1c(F)cccc1F)c2ccc(Cl)c([N+](=O)[O-])c2"
#     # query = "Give full information about this rxn C(O)(=O)C1=CC=CC=C1.[N+](=[N-])=N>>C(N)(=O)C1=CC=CC=C1.N#N"
#     # query = "What is the reaction name having this smiles CCCl.CC[O-].[Na+] >> CCOCC.[Na+].[Cl-]"
#     print(enhanced_query(query))






# NOW 

import os
import re
import requests
from dotenv import load_dotenv
load_dotenv()

from tools.make_tools import make_tools 
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Setup LLM and tools
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = make_tools(llm=llm)

# Prompt parts
PREFIX = """
You are Chem Copilot, an expert chemistry assistant. You have access to the following tools to analyze molecules and chemical reactions.

Always begin by understanding the user's **intent** — what kind of information are they asking for?

Here is how to choose tools:

- If the user gives a SMILES or reaction SMILES and asks for the name, you MUST use **SMILES2Name**. Do NOT analyze bonds or functional groups for this task.
- Use **NameToSMILES**: when the user gives a compound/reaction name and wants the SMILES or structure.
- Use **FuncGroups**: when the user wants to analyze functional groups in a molecule or a reaction (input is SMILES or reaction SMILES).
- Use **BondChangeAnalyzer**: when the user asks for which bonds are broken, formed, or changed in a chemical reaction.

If the user wants all of the above (full analysis), respond with "This requires full analysis." (This will be handled by a separate function.)

Always return your answer in this format:
Final Answer: <your answer here>

For **FuncGroups** results:
- Always list the functional groups identified in each reactant and product separately
- Include the transformation summary showing disappeared groups, appeared groups, and unchanged groups
- Provide a clear conclusion about what transformation occurred in the reaction

For **BondChangeAnalyzer** results:
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
    tools=tools,
    prefix=PREFIX,
    suffix=SUFFIX,
    format_instructions=FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad"]
)

agent_chain = ZeroShotAgent.from_llm_and_tools(llm=llm, tools=tools, prompt=prompt)
agent = AgentExecutor(agent=agent_chain, tools=tools, verbose=True)

def extract_final_answer(full_output: str):
    match = re.search(r"Final Answer:\s*(.*)", full_output, re.DOTALL)
    return match.group(1).strip() if match else full_output.strip()

# 🧠 Full Info Handler
def handle_full_info(query, reaction_smiles):
    print("Running full analysis using all tools...\n")
    full_info = {}

    # Map tool names to classes
    tool_dict = {tool.name.lower(): tool for tool in tools}

    try:
        # First, visualize the reaction
        visualizer_tool = tool_dict.get("chemvisualizer")
        if visualizer_tool:
            try:
                full_info['Visualization'] = visualizer_tool.run(reaction_smiles)
            except Exception as e:
                full_info['Visualization'] = f"Error visualizing reaction: {str(e)}"
        else:
            full_info['Visualization'] = "ChemVisualizer tool not found"

        # Run smiles2name
        name_tool = tool_dict.get("smiles2name")
        if name_tool:
            try:
                full_info['Names'] = name_tool.run(reaction_smiles)
            except Exception as e:
                full_info['Names'] = f"Error analyzing names: {str(e)}"
        else:
            full_info['Names'] = "SMILES2Name tool not found"

        # Run funcgroups
        fg_tool = tool_dict.get("funcgroups")
        if fg_tool:
            try:
                full_info['Functional Groups'] = fg_tool.run(reaction_smiles)
            except Exception as e:
                full_info['Functional Groups'] = f"Error analyzing functional groups: {str(e)}"
        else:
            full_info['Functional Groups'] = "FuncGroups tool not found"

        # Run bond.py
        bond_tool = tool_dict.get("bondchangeanalyzer")
        if bond_tool:
            try:
                full_info['Bond Changes'] = bond_tool.run(reaction_smiles)
            except Exception as e:
                full_info['Bond Changes'] = f"Error analyzing bond changes: {str(e)}"
        else:
            full_info['Bond Changes'] = "BondChangeAnalyzer tool not found"
            
        # Run reaction classifier - new addition
        reaction_classifier_tool = tool_dict.get("reactionclassifier")
        if reaction_classifier_tool:
            try:
                full_info['Reaction Classification'] = reaction_classifier_tool.run(reaction_smiles)
            except Exception as e:
                full_info['Reaction Classification'] = f"Error classifying reaction: {str(e)}"
        else:
            full_info['Reaction Classification'] = "ReactionClassifier tool not found"

        # Combine all information and ask LLM for final comprehensive answer
        final_prompt = f"""You are a chemistry expert. Synthesize this comprehensive reaction analysis into a clear, educational explanation:
        
Reaction SMILES: {reaction_smiles}

The following tool outputs provide different perspectives on this reaction:

VISUALIZATION:
{full_info['Visualization']}

COMPOUND/REACTION NAMES:
{full_info['Names']}

FUNCTIONAL GROUPS ANALYSIS:
{full_info['Functional Groups']}

BOND CHANGES ANALYSIS:
{full_info['Bond Changes']}

REACTION CLASSIFICATION AND INFORMATION:
{full_info['Reaction Classification']}

Provide a thorough, well-structured explanation of this reaction that:
1. Begins with a high-level summary of what type of reaction this is
2. Explains what happens at the molecular level (bonds broken/formed)
3. Discusses the functional group transformations
4. Includes relevant details about the reaction mechanism
5. Mentions common applications or importance of this reaction type

Please give a complete and readable explanation of this reaction.
"""

        response = llm.invoke(final_prompt)
        return {
            'visualization_path': full_info['Visualization'] if 'Visualization' in full_info and not full_info['Visualization'].startswith('Error') else None,
            'analysis': response.content.strip(),
            'reaction_classification': full_info.get('Reaction Classification', "No classification available")
        }
    
    except Exception as e:
        return {
            'visualization_path': None,
            'analysis': f"Error in full analysis: {str(e)}"
        }

# Update enhanced_query function to handle the new visualization return format
def enhanced_query(query, callbacks=None):
    try:
        # Check if it's a full analysis request
        if "full information" in query.lower() or "full analysis" in query.lower():
            # Extract SMILES from query using regex - handle both simple and complex SMILES
            # More comprehensive SMILES pattern
            smiles_pattern = r"([A-Za-z0-9@\[\]\.\+\-\=\#\:\(\)\\\/;\$\%\|\{\}]+>>[A-Za-z0-9@\[\]\.\+\-\=\#\:\(\)\\\/;\$\%\|\{\}]*)"
            match = re.search(smiles_pattern, query)
            
            if match:
                reaction_smiles = match.group(1)
                return handle_full_info(query, reaction_smiles)
            else:
                # Try to extract product SMILES if not a full reaction
                product_pattern = r"rxn\s+([A-Za-z0-9@\[\]\.\+\-\=\#\:\(\)\\\/;\$\%\|\{\}]+)"
                product_match = re.search(product_pattern, query)
                
                if product_match:
                    # For single product, create a simple reaction SMILES
                    product_smiles = product_match.group(1)
                    # Simple reaction with generic reactant
                    return handle_full_info(query, f"[R]>>{product_smiles}")
                
                return {"visualization_path": None, "analysis": "Could not extract reaction SMILES from the query. Please provide a valid reaction SMILES or product SMILES."}
        
        # NEW ADDITION: Check if it's just a visualization request
        elif any(term in query.lower() for term in ["visual", "picture", "image", "show", "draw", "representation"]):
            # Extract SMILES from query using regex
            smiles_pattern = r"([A-Za-z0-9@\[\]\.\+\-\=\#\:\(\)\\\/;\$\%\|\{\}]+>>[A-Za-z0-9@\[\]\.\+\-\=\#\:\(\)\\\/;\$\%\|\{\}]*)"
            match = re.search(smiles_pattern, query)
            
            if match:
                reaction_smiles = match.group(1)
                # Only use the visualizer tool
                tool_dict = {tool.name.lower(): tool for tool in tools}
                visualizer_tool = tool_dict.get("chemvisualizer")
                
                if visualizer_tool:
                    try:
                        viz_path = visualizer_tool.run(reaction_smiles)
                        return {
                            "visualization_path": viz_path,
                            "analysis": f"Visual representation of the reaction: {reaction_smiles}"
                        }
                    except Exception as e:
                        return {"visualization_path": None, "analysis": f"Error visualizing reaction: {str(e)}"}
                else:
                    return {"visualization_path": None, "analysis": "ChemVisualizer tool not found"}
            else:
                return {"visualization_path": None, "analysis": "Could not extract reaction SMILES from the visualization request. Please provide a valid reaction SMILES."}

        # Otherwise, use normal agent
        result = agent.invoke({"input": query}, {"callbacks": callbacks} if callbacks else {})
        return {"visualization_path": None, "analysis": extract_final_answer(result.get("output", ""))}

    except Exception as e:
        return {"visualization_path": None, "analysis": f"Error occurred: {str(e)}"}

# Retrosynthesis API function
def get_retrosynthesis(compound_name):
    """
    Call the retrosynthesis API to get synthesis pathways for a compound
    """
    try:
        # Format the request according to the API's requirements
        payload = {
            "material": compound_name,
            "num_results": 10,
            "alignment": True,
            "expansion": True,
            "filtration": False
        }
        print(f"Sending retrosynthesis API request with payload: {payload}")
        
        response = requests.post(
            "http://localhost:8000/retro-synthesis/",
            json=payload
        )
        print(f"Received response with status code: {response.status_code}")
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API error response: {response.text}")
            return {
                "status": "error",
                "message": f"API returned status code {response.status_code}: {response.text}"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error calling retrosynthesis API: {str(e)}"
        }
# Run
if __name__ == "__main__":
    # Test retrosynthesis
    # retro_result = get_retrosynthesis("flubendiamide")
    # print(retro_result)
    
    # Test analysis
    query = "Give full information about this rxn O=C(NC(=O)c1c(F)cccc1F)c2ccc(Cl)c([N+](=O)[O-])c2"
    print(enhanced_query(query))