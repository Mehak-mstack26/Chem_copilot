# # # app.py - Place this file in the root directory of your CHEM_COPILOT folder
# from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
# import streamlit as st
# import re
# import sys
# import os

# # Add the project root to the Python path for imports
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# # Import your function from the test.py file
# from test import enhanced_query

# # Set up the Streamlit page
# st.set_page_config(
#     page_title="ChemCopilot - Chemistry Assistant",
#     page_icon="🧪",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for styling
# st.markdown("""
#     <style>
#     .main-header {
#         font-size: 42px;
#         font-weight: bold;
#         color: #2e7d32;
#         margin-bottom: 0px;
#     }
#     .sub-header {
#         font-size: 20px;
#         color: #5c5c5c;
#         margin-bottom: 30px;
#     }
#     .stButton>button {
#         background-color: #2e7d32;
#         color: white;
#         border: none;
#         padding: 10px 24px;
#         border-radius: 4px;
#         font-weight: bold;
#     }
#     .stButton>button:hover {
#         background-color: #005005;
#     }
#     .tool-card {
#         background-color: #f5f5f5;
#         padding: 10px;
#         border-radius: 5px;
#         margin-bottom: 10px;
#     }
#     .result-area {
#         background-color: #f9f9f9;
#         padding: 20px;
#         border-radius: 8px;
#         border-left: 4px solid #2e7d32;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # App Header
# st.markdown('<p class="main-header">ChemCopilot</p>', unsafe_allow_html=True)
# st.markdown('<p class="sub-header">Your Expert Chemistry Assistant</p>', unsafe_allow_html=True)

# # Sidebar with tools info and examples
# with st.sidebar:
#     st.markdown("## Tools Available")
    
#     with st.expander("🔍 SMILES2Name"):
#         st.markdown("Converts SMILES notation to chemical names.")
#         st.markdown("*Example:* `C1=CC=CC=C1` → `Benzene`")
    
#     with st.expander("📝 Name2SMILES"):
#         st.markdown("Converts chemical names to SMILES notation.")
#         st.markdown("*Example:* `Ethanol` → `CCO`")
    
#     with st.expander("🧪 FuncGroups"):
#         st.markdown("Analyzes functional groups in molecules.")
#         st.markdown("*Example:* `C(O)(=O)C1=CC=CC=C1` → `Carboxylic acid, Aromatic ring`")
    
#     with st.expander("⚛️ BondChangeAnalyzer"):
#         st.markdown("Analyzes bond changes in chemical reactions.")
#         st.markdown("*Example:* `CCCl.CC[O-].[Na+]>>CCOCC.[Na+].[Cl-]` → `C-Cl bond broken, C-O bond formed`")
    
#     st.markdown("## Example Queries")
    
#     example1 = "What is the reaction name having this smiles CCCl.CC[O-].[Na+]>>CCOCC.[Na+].[Cl-]"
#     example2 = "Give full information about this rxn CBr.[Na+].[I-]>>CI.[Na+].[Br-]"
#     example3 = "What functional group transformations occur in this reaction: O=C(O)c1ccccc1.HN=[N+]=[N-] >> NC(=O)c1ccccc1.N#N.O"
    
#     if st.button("Reaction Name from SMILES"):
#         st.session_state.query = example1
        
#     if st.button("Full Reaction Analysis"):
#         st.session_state.query = example2
        
#     if st.button("Identify Functional Groups"):
#         st.session_state.query = example3

# # Main content area
# st.markdown("## Enter Your Chemistry Query")

# # Initialize session state for query if it doesn't exist
# if 'query' not in st.session_state:
#     st.session_state.query = ""

# # Input box for user query
# query = st.text_area(
#     "Ask me about chemical reactions and molecules:",
#     value=st.session_state.query,
#     height=100,
#     placeholder="Example: Give full information about this rxn CBr.[Na+].[I-]>>CI.[Na+].[Br-]"
# )

# # Action buttons
# col1, col2 = st.columns([1, 5])
# with col1:
#     analyze_button = st.button("Analyze")
# with col2:
#     if st.button("Clear"):
#         st.session_state.query = ""
#         st.experimental_rerun()

# # Process the query and display results
# if analyze_button and query:
#     st.session_state.query = query  # Save query to session state

#     with st.spinner("Analyzing your chemistry query..."):
#         callback_container = st.container()
#         st_callback = StreamlitCallbackHandler(callback_container)
#         try:
#             # Pass the callback to your backend function
#             result = enhanced_query(query, callbacks=[st_callback])
            
#             st.session_state.last_result = result
            
#             # Display the final answer as before
#             st.markdown("## Results")
#             st.markdown('<div class="result-area">', unsafe_allow_html=True)
#             st.markdown(result)
#             st.markdown('</div>', unsafe_allow_html=True)
            
#             # Check if it's a reaction query
#             if ">>" in query or "rxn" in query.lower():
#                 # Extract SMILES using regex
#                 smiles_match = re.search(r"([A-Za-z0-9@\[\]\.\+\-\=\#\(\)]+>>[A-Za-z0-9@\[\]\.\+\-\=\#\(\)\.]*)", query)
#                 if smiles_match:
#                     reaction_smiles = smiles_match.group(1)
#                     st.markdown("### Reaction SMILES")
#                     st.code(reaction_smiles)
                    
#         except Exception as e:
#             st.error(f"An error occurred: {str(e)}")
#             st.markdown("Please check your query format and try again.")

# elif analyze_button and not query:
#     st.error("Please enter a query first.")

# # Display previous result if it exists
# elif 'last_result' in st.session_state and st.session_state.query:
#     st.markdown("## Previous Results")
#     st.markdown('<div class="result-area">', unsafe_allow_html=True)
#     st.markdown(st.session_state.last_result)
#     st.markdown('</div>', unsafe_allow_html=True)

# # Footer
# st.markdown("---")
# col1, col2, col3 = st.columns([1, 2, 1])
# with col2:
#     st.caption("ChemCopilot - Your Expert Chemistry Assistant")






# NOW 
# import streamlit as st
# import requests
# import re
# import sys
# import os
# from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
# from langchain_openai import ChatOpenAI

# # Add the project root to the Python path for imports
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# # Import your function from the test.py file
# from Features.env.test import enhanced_query

# # Initialize LLM for name to SMILES conversion
# llm = ChatOpenAI(model="gpt-4o", temperature=0)

# # Set up the Streamlit page
# st.set_page_config(
#     page_title="ChemCopilot - Chemistry Assistant",
#     page_icon="🧪",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for styling
# st.markdown("""
#     <style>
#     .main-header {
#         font-size: 42px;
#         font-weight: bold;
#         color: #2e7d32;
#         margin-bottom: 0px;
#     }
#     .sub-header {
#         font-size: 20px;
#         color: #5c5c5c;
#         margin-bottom: 30px;
#     }
#     .stButton>button {
#         background-color: #2e7d32;
#         color: white;
#         border: none;
#         padding: 10px 24px;
#         border-radius: 4px;
#         font-weight: bold;
#     }
#     .stButton>button:hover {
#         background-color: #005005;
#     }
#     .tool-card {
#         background-color: #f5f5f5;
#         padding: 10px;
#         border-radius: 5px;
#         margin-bottom: 10px;
#     }
#     .result-area {
#         background-color: #f9f9f9;
#         padding: 20px;
#         border-radius: 8px;
#         border-left: 4px solid #2e7d32;
#     }
#     .reaction-card {
#         background-color: #f0f7f0;
#         padding: 15px;
#         border-radius: 8px;
#         margin-bottom: 15px;
#         border-left: 4px solid #2e7d32;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Initialize session state variables
# if 'query' not in st.session_state:
#     st.session_state.query = ""
# if 'retro_results' not in st.session_state:
#     st.session_state.retro_results = None
# if 'selected_rxn_smiles' not in st.session_state:
#     st.session_state.selected_rxn_smiles = None
# if 'current_tab_index' not in st.session_state:
#     st.session_state.current_tab_index = 0
# if 'auto_analyze' not in st.session_state:
#     st.session_state.auto_analyze = False
# if 'last_result' not in st.session_state:
#     st.session_state.last_result = None

# # Function to convert reactant and product names to SMILES
# def convert_to_reaction_smiles(reactants, products):
#     """Convert reactant and product names to reaction SMILES format using LLM"""
#     prompt = f"""Convert these chemical names to a reaction SMILES format.

# Reactants: {', '.join(reactants)}
# Products: {', '.join(products)}

# Format the output as reaction SMILES using the format: reactant1.reactant2>>product1.product2
# Only output the SMILES, no explanations.
# """
    
#     try:
#         response = llm.invoke(prompt)
#         reaction_smiles = response.content.strip()
        
#         # Verify that the output looks like a reaction SMILES
#         if ">>" in reaction_smiles:
#             return reaction_smiles
#         else:
#             # If LLM didn't output proper reaction SMILES, try to construct manually
#             # Join names with dots (for multiple reactants/products) and '>>' separator
#             reactants_str = '.'.join(reactants)
#             products_str = '.'.join(products)
#             return f"{reactants_str}>>{products_str}"
#     except Exception as e:
#         st.error(f"Error converting to reaction SMILES: {str(e)}")
#         return None

# # Function to create reaction SMILES from reactants and products
# def create_reaction_smiles(reactants, products):
#     """Create reaction SMILES from reactants and products, handling both SMILES and name inputs"""
#     # Check if inputs are likely SMILES or names
#     smiles_pattern = r'^[A-Za-z0-9@\[\]\.\+\-\=\#\:\(\)\\\/;\$\%\|\{\}]+$'
    
#     # Check if all reactants and products are SMILES
#     all_smiles = all(re.match(smiles_pattern, r) for r in reactants) and all(re.match(smiles_pattern, p) for p in products)
    
#     if all_smiles:
#         # Join SMILES directly
#         reactants_str = '.'.join(reactants)
#         products_str = '.'.join(products)
#         return f"{reactants_str}>>{products_str}"
#     else:
#         # Use LLM to convert names to SMILES
#         return convert_to_reaction_smiles(reactants, products)

# # Function to handle reaction analysis button clicks
# def analyze_reaction(reactants, products):
#     """Analyze reaction by converting names to SMILES if necessary"""
#     reaction_smiles = create_reaction_smiles(reactants, products)
    
#     if reaction_smiles:
#         st.session_state.selected_rxn_smiles = reaction_smiles
#         st.session_state.query = f"Give full information about this rxn {reaction_smiles}"
#         st.session_state.current_tab_index = 1  # Set to Analysis tab (index 1)
#         st.session_state.auto_analyze = True
#     else:
#         st.error("Could not create reaction SMILES. Please check the reactants and products.")

# # App Header
# st.markdown('<p class="main-header">ChemCopilot</p>', unsafe_allow_html=True)
# st.markdown('<p class="sub-header">Your Expert Chemistry Assistant</p>', unsafe_allow_html=True)

# # Sidebar with tools info and examples
# with st.sidebar:
#     st.markdown("## Tools Available")
    
#     with st.expander("🔍 SMILES2Name"):
#         st.markdown("Converts SMILES notation to chemical names.")
#         st.markdown("*Example:* `C1=CC=CC=C1` → `Benzene`")
    
#     with st.expander("📝 Name2SMILES"):
#         st.markdown("Converts chemical names to SMILES notation.")
#         st.markdown("*Example:* `Ethanol` → `CCO`")
    
#     with st.expander("🧪 FuncGroups"):
#         st.markdown("Analyzes functional groups in molecules.")
#         st.markdown("*Example:* `C(O)(=O)C1=CC=CC=C1` → `Carboxylic acid, Aromatic ring`")
    
#     with st.expander("⚛️ BondChangeAnalyzer"):
#         st.markdown("Analyzes bond changes in chemical reactions.")
#         st.markdown("*Example:* `CCCl.CC[O-].[Na+]>>CCOCC.[Na+].[Cl-]` → `C-Cl bond broken, C-O bond formed`")
    
#     st.markdown("## Example Queries")
    
#     if st.button("Search Retrosynthesis for Flubendiamide"):
#         st.session_state.query = "flubendiamide"
#         st.session_state.current_tab_index = 0  # Set to Search tab
        
#     if st.button("Search for 2-amino-5-chloro-3-methyl benzoic acid"):
#         st.session_state.query = "2-amino-5-chloro-3-methyl benzoic acid"
#         st.session_state.current_tab_index = 0  # Set to Search tab
        
#     if st.button("Reaction Analysis Example"):
#         st.session_state.query = "Give full information about this rxn CCCl.CC[O-].[Na+]>>CCOCC.[Na+].[Cl-]"
#         st.session_state.current_tab_index = 1  # Set to Analysis tab

# # Create tabs with selected parameter
# tabs = st.tabs(["Retrosynthesis Search", "Reaction Analysis"])
# selected_tab = tabs[st.session_state.current_tab_index]

# # Retrosynthesis Search Tab
# with tabs[0]:
#     st.markdown("## Search Retrosynthesis Pathways")
    
#     compound_name = st.text_input(
#         "Enter compound name (IUPAC or common name):",
#         value=st.session_state.query if st.session_state.current_tab_index == 0 else "",
#         placeholder="Example: flubendiamide, 2-amino-5-chloro-3-methyl benzoic acid",
#         key="search_compound_name"
#     )
    
#     # Simple search button without advanced settings
#     search_button = st.button("Search Retrosynthesis", key="search_retro_button")
    
#     if search_button and compound_name:
#         st.session_state.query = compound_name
#         with st.spinner("Searching for retrosynthesis pathways..."):
#             try:
#                 # Make API call to retrosynthesis service with default payload
#                 payload = {
#                     "material": compound_name,
#                     "num_results": 10,
#                     "alignment": True,
#                     "expansion": True,
#                     "filtration": False
#                 }
                
#                 response = requests.post(
#                     "http://localhost:8000/retro-synthesis/",
#                     json=payload
#                 )
                
#                 if response.status_code == 200:
#                     retro_data = response.json()
#                     if retro_data.get("status") == "success":
#                         st.session_state.retro_results = retro_data["data"]
#                         st.success(f"Found {len(retro_data['data']['reactions'])} reactions for {compound_name}")
#                     else:
#                         st.error(f"API returned an error: {retro_data.get('message', 'Unknown error')}")
#                 else:
#                     st.error(f"API request failed with status code: {response.status_code}")
#                     st.error(f"Response content: {response.text}")
#             except Exception as e:
#                 st.error(f"Failed to fetch retrosynthesis data: {str(e)}")
#                 import traceback
#                 st.error(traceback.format_exc())
    
#     # Display retrosynthesis results if available
#     if st.session_state.retro_results:
#         st.markdown("## Retrosynthesis Pathway")
        
#         # Display recommended pathway
#         st.markdown("### Recommended Synthesis Route")
#         st.markdown(st.session_state.retro_results["reasoning"])
        
#         # Display reactions
#         st.markdown("### Reactions")
        
#         for i, reaction in enumerate(st.session_state.retro_results["reactions"]):
#             with st.container():
#                 st.markdown(f'<div class="reaction-card">', unsafe_allow_html=True)
                
#                 # Highlight if this is a recommended step
#                 if reaction["idx"] in st.session_state.retro_results["recommended_indices"]:
#                     st.markdown(f"**Step {i+1} (Recommended)**: {reaction['idx']}")
#                 else:
#                     st.markdown(f"**Step {i+1}**: {reaction['idx']}")
                
#                 # Reactants and products
#                 reactants_str = " + ".join(reaction["reactants"])
#                 products_str = " + ".join(reaction["products"])
#                 st.markdown(f"**Reaction**: {reactants_str} → {products_str}")
                
#                 # Conditions
#                 st.markdown(f"**Conditions**: {reaction['conditions']}")
                
#                 # Source
#                 st.markdown(f"**Source**: {reaction['source']}")
                
#                 # Create a button that will trigger the analysis
#                 if st.button(f"Analyze Reaction {i+1}", key=f"analyze_btn_{i}"):
#                     analyze_reaction(reaction["reactants"], reaction["products"])
#                     st.rerun()
                
#                 st.markdown('</div>', unsafe_allow_html=True)

# # Reaction Analysis Tab
# # Reaction Analysis Tab
# with tabs[1]:
#     st.markdown("## Analyze Chemical Reactions")
    
#     # Input box for user query
#     query = st.text_area(
#         "Ask me about chemical reactions and molecules:",
#         value=st.session_state.query if st.session_state.current_tab_index == 1 else "",
#         height=100,
#         placeholder="Example: Give full information about this rxn CCCl.CC[O-].[Na+]>>CCOCC.[Na+].[Cl-]",
#         key="analysis_query"
#     )
    
#     # Action buttons
#     col1, col2 = st.columns([1, 5])
#     with col1:
#         analyze_button = st.button("Analyze", key="analyze_reaction_btn")
#     with col2:
#         if st.button("Clear", key="clear_analyze_btn"):
#             st.session_state.query = ""
#             st.rerun()
    
#     # Process the query and display results - either from button click or auto-analyze
#     if (analyze_button and query) or (st.session_state.auto_analyze and query):
#         # Reset auto-analyze flag
#         st.session_state.auto_analyze = False
        
#         st.session_state.query = query  # Save query to session state
        
#         with st.spinner("Analyzing your chemistry query..."):
#             callback_container = st.container()
#             st_callback = StreamlitCallbackHandler(callback_container)
#             try:
#                 # Pass the callback to your backend function
#                 result = enhanced_query(query, callbacks=[st_callback])
                
#                 st.session_state.last_result = result
                
#                 # Display visualization if available
#                 if result.get('visualization_path') and os.path.exists(result.get('visualization_path')):
#                     st.markdown("## Reaction Visualization")
#                     st.image(result.get('visualization_path'), caption="Chemical Reaction Visualization")
                
#                 # Add this line to make file path less prominent but still available
#                 with st.expander("Image file details"):
#                     st.code(result.get('visualization_path'))
                
#                 # Display the final answer
#                 st.markdown("## Results")
#                 st.markdown('<div class="result-area">', unsafe_allow_html=True)
#                 st.markdown(result.get('analysis', 'No analysis available'))
#                 st.markdown('</div>', unsafe_allow_html=True)
                
#                 # Check if it's a reaction query
#                 if ">>" in query or "rxn" in query.lower():
                    
#                     # Extract SMILES using regex
#                     smiles_match = re.search(r"([A-Za-z0-9@\[\]\.\+\-\=\#\(\)]+>>[A-Za-z0-9@\[\]\.\+\-\=\#\(\)\.]*)", query)
#                     if smiles_match:
#                         reaction_smiles = smiles_match.group(1)
#                         st.markdown("### Reaction SMILES")
#                         st.code(reaction_smiles)
                        
#             except Exception as e:
#                 st.error(f"An error occurred: {str(e)}")
#                 st.markdown("Please check your query format and try again.")
    
#     elif 'last_result' in st.session_state and st.session_state.last_result and st.session_state.query:
#         st.markdown("## Previous Results")
        
#         # Display visualization if available
#         # if isinstance(st.session_state.last_result, dict) and st.session_state.last_result.get('visualization_path') and os.path.exists(st.session_state.last_result.get('visualization_path')):
#         #     st.markdown("## Reaction Visualization")
#         #     st.image(st.session_state.last_result.get('visualization_path'), caption="Chemical Reaction Visualization")
        
#         # st.markdown('<div class="result-area">', unsafe_allow_html=True)
#         # if isinstance(st.session_state.last_result, dict):
#         #     st.markdown(st.session_state.last_result.get('analysis', 'No analysis available'))
#         # else:
#         #     st.markdown(st.session_state.last_result)
#         # st.markdown('</div>', unsafe_allow_html=True)

# # Footer
# st.markdown("---")
# col1, col2, col3 = st.columns([1, 2, 1])
# with col2:
#     st.caption("ChemCopilot - Your Expert Chemistry Assistant")


# CURRENT 
import streamlit as st
import requests
import re
import sys
import os
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI

os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
# Add the project root to the Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your function from the test.py file
from test import enhanced_query
# Import retrosynthesis function
from tools.retrosynthesis import run_retrosynthesis
# Import NameToSMILES tool
from tools.name2smiles import NameToSMILES  # Update this with the correct import path

# Initialize LLM for name to SMILES conversion
llm = ChatOpenAI(model="gpt-4o", temperature=0)
# Initialize NameToSMILES tool
name_to_smiles_tool = NameToSMILES()

# Set up the Streamlit page
st.set_page_config(
    page_title="ChemCopilot - Chemistry Assistant",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #2e7d32;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 20px;
        color: #5c5c5c;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 4px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #005005;
    }
    .tool-card {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .result-area {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #2e7d32;
    }
    .reaction-card {
        background-color: #f0f7f0;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 4px solid #2e7d32;
    }
    .analysis-section {
        margin-top: 30px;
        padding-top: 20px;
        border-top: 1px solid #ddd;
    }
    .query-box {
        background-color: #f5f7fa;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
        border-left: 4px solid #3f51b5;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state variables
if 'query' not in st.session_state:
    st.session_state.query = ""
if 'retro_results' not in st.session_state:
    st.session_state.retro_results = None
if 'selected_rxn_smiles' not in st.session_state:
    st.session_state.selected_rxn_smiles = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'rxn_query' not in st.session_state:
    st.session_state.rxn_query = ""
if 'show_analyze_section' not in st.session_state:
    st.session_state.show_analyze_section = False

# Function to get SMILES from chemical name using NameToSMILES tool
def get_smiles_from_name(name):
    """Get SMILES notation for a chemical name using the NameToSMILES tool"""
    try:
        result = name_to_smiles_tool._run(name)
        if "SMILES:" in result:
            # Extract SMILES from the result
            smiles = result.split("SMILES:")[1].split("\n")[0].strip()
            return smiles
        return None
    except Exception as e:
        st.error(f"Error converting {name} to SMILES: {str(e)}")
        return None

# Function to convert reactant and product names to SMILES
def convert_to_reaction_smiles(reactants, products):
    """Convert reactant and product names to reaction SMILES format using LLM and NameToSMILES tool"""
    reactant_smiles = []
    product_smiles = []
    
    # First try to convert each reactant and product using NameToSMILES tool
    for reactant in reactants:
        smiles = get_smiles_from_name(reactant)
        if smiles:
            reactant_smiles.append(smiles)
        else:
            reactant_smiles.append(reactant)  # Keep original name if conversion fails
    
    for product in products:
        smiles = get_smiles_from_name(product)
        if smiles:
            product_smiles.append(smiles)
        else:
            product_smiles.append(product)  # Keep original name if conversion fails
    
    # If NameToSMILES failed for some compounds, use LLM as backup
    if any(not re.match(r'^[A-Za-z0-9@\[\]\.\+\-\=\#\:\(\)\\\/;\$\%\|\{\}]+$', r) for r in reactant_smiles) or \
       any(not re.match(r'^[A-Za-z0-9@\[\]\.\+\-\=\#\:\(\)\\\/;\$\%\|\{\}]+$', p) for p in product_smiles):
        
        prompt = f"""Convert these chemical names to a reaction SMILES format.

Reactants: {', '.join(reactants)}
Products: {', '.join(products)}

Format the output as reaction SMILES using the format: reactant1.reactant2>>product1.product2
Only output the SMILES, no explanations.
"""
        
        try:
            response = llm.invoke(prompt)
            reaction_smiles = response.content.strip()
            
            # Verify that the output looks like a reaction SMILES
            if ">>" in reaction_smiles:
                return reaction_smiles
        except Exception as e:
            st.error(f"Error converting to reaction SMILES with LLM: {str(e)}")
    
    # Join SMILES directly if all conversions were successful
    reactants_str = '.'.join(reactant_smiles)
    products_str = '.'.join(product_smiles)
    return f"{reactants_str}>>{products_str}"

# Function to extract clean SMILES from reaction data
def extract_reaction_smiles(reaction):
    """Extract or generate clean reaction SMILES from reaction data"""
    # First check if there's a valid reaction_smiles that doesn't have placeholders
    if "reaction_smiles" in reaction and reaction["reaction_smiles"]:
        smiles = reaction["reaction_smiles"]
        # Check if it contains placeholders
        if not re.search(r'\[.*?_SMILES\]|\[.*?\]', smiles) and ">>" in smiles:
            return smiles
    
    # If reaction_smiles is not usable, create from reactants and products
    return convert_to_reaction_smiles(reaction["reactants"], reaction["products"])

# Function to create reaction SMILES from reactants and products
def create_reaction_smiles(reactants, products):
    """Create reaction SMILES from reactants and products, handling both SMILES and name inputs"""
    # Check if inputs are likely SMILES or names
    smiles_pattern = r'^[A-Za-z0-9@\[\]\.\+\-\=\#\:\(\)\\\/;\$\%\|\{\}]+$'
    
    # Check if all reactants and products are SMILES
    all_smiles = all(re.match(smiles_pattern, r) for r in reactants) and all(re.match(smiles_pattern, p) for p in products)
    
    if all_smiles:
        # Join SMILES directly
        reactants_str = '.'.join(reactants)
        products_str = '.'.join(products)
        return f"{reactants_str}>>{products_str}"
    else:
        # Use conversion function
        return convert_to_reaction_smiles(reactants, products)

# Function to handle reaction analysis
def analyze_reaction(reactants, products):
    """Analyze reaction by converting names to SMILES if necessary"""
    reaction_smiles = create_reaction_smiles(reactants, products)
    
    if reaction_smiles:
        st.session_state.selected_rxn_smiles = reaction_smiles
        st.session_state.query = f"Give full information about this rxn {reaction_smiles}"
        st.session_state.show_analyze_section = True
        return True
    else:
        st.error("Could not create reaction SMILES. Please check the reactants and products.")
        return False

# App Header
st.markdown('<p class="main-header">ChemCopilot</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your Expert Chemistry Assistant</p>', unsafe_allow_html=True)

# Sidebar with tools info and examples
with st.sidebar:
    st.markdown("## Tools Available")
    
    with st.expander("🔍 SMILES2Name"):
        st.markdown("Converts SMILES notation to chemical names.")
        st.markdown("*Example:* `C1=CC=CC=C1` → `Benzene`")
    
    with st.expander("📝 Name2SMILES"):
        st.markdown("Converts chemical names to SMILES notation.")
        st.markdown("*Example:* `Ethanol` → `CCO`")
    
    with st.expander("🧪 FuncGroups"):
        st.markdown("Analyzes functional groups in molecules.")
        st.markdown("*Example:* `C(O)(=O)C1=CC=CC=C1` → `Carboxylic acid, Aromatic ring`")
    
    with st.expander("⚛️ BondChangeAnalyzer"):
        st.markdown("Analyzes bond changes in chemical reactions.")
        st.markdown("*Example:* `CCCl.CC[O-].[Na+]>>CCOCC.[Na+].[Cl-]` → `C-Cl bond broken, C-O bond formed`")
    
    st.markdown("## Example Queries")
    
    if st.button("Search for Flubendiamide"):
        st.session_state.query = "flubendiamide"
        st.session_state.show_analyze_section = False
        st.rerun()
        
    if st.button("Search for 2-amino-5-chloro-3-methyl benzoic acid"):
        st.session_state.query = "2-amino-5-chloro-3-methyl benzoic acid"
        st.session_state.show_analyze_section = False
        st.rerun()
        
    if st.button("Reaction Analysis Example"):
        st.session_state.query = "Give full information about this rxn CCCl.CC[O-].[Na+]>>CCOCC.[Na+].[Cl-]"
        st.session_state.selected_rxn_smiles = "CCCl.CC[O-].[Na+]>>CCOCC.[Na+].[Cl-]"
        st.session_state.show_analyze_section = True
        st.rerun()

# Main content area - integrated into a single page
st.markdown("## Retrosynthesis & Reaction Analysis")

# Search Input Section
compound_name = st.text_input(
    "Enter compound name (IUPAC or common name):",
    value=st.session_state.query if not st.session_state.show_analyze_section else "",
    placeholder="Example: flubendiamide, 2-amino-5-chloro-3-methyl benzoic acid",
    key="search_compound_name"
)

search_button = st.button("Search Retrosynthesis", key="search_retro_button")

# Process Search
if search_button and compound_name:
    st.session_state.query = compound_name
    st.session_state.show_analyze_section = False  # Reset analysis section
    
    with st.spinner("Searching for retrosynthesis pathways..."):
        try:
            # Call your retrosynthesis function
            retro_result = run_retrosynthesis(compound_name)
            
            if retro_result.get("status") == "success":
                # Process the reactions to ensure we have usable SMILES
                for reaction in retro_result["data"]["reactions"]:
                    # Extract or create valid reaction SMILES
                    reaction["cleaned_reaction_smiles"] = extract_reaction_smiles(reaction)
                
                st.session_state.retro_results = retro_result["data"]
                st.success(f"Found {len(retro_result['data']['reactions'])} reactions for {compound_name}")
            else:
                st.error(f"Error: {retro_result.get('message', 'Unknown error')}")
                if retro_result.get('traceback'):
                    with st.expander("See error details"):
                        st.code(retro_result.get('traceback'))
        except Exception as e:
            st.error(f"Failed to fetch retrosynthesis data: {str(e)}")
            import traceback
            with st.expander("See error details"):
                st.code(traceback.format_exc())

# Display retrosynthesis results if available
if st.session_state.retro_results and not st.session_state.show_analyze_section:
    st.markdown("## Retrosynthesis Pathway")
    
    # Display recommended pathway
    st.markdown("### Recommended Synthesis Route")
    st.markdown(st.session_state.retro_results["reasoning"])
    
    # Display reactions
    st.markdown("### Reactions")
    
    for i, reaction in enumerate(st.session_state.retro_results["reactions"]):
        with st.container():
            st.markdown(f'<div class="reaction-card">', unsafe_allow_html=True)
            
            # Highlight if this is a recommended step
            if reaction["idx"] in st.session_state.retro_results["recommended_indices"]:
                st.markdown(f"**Step {i+1} (Recommended)**: {reaction['idx']}")
            else:
                st.markdown(f"**Step {i+1}**: {reaction['idx']}")
            
            # Reactants and products
            reactants_str = " + ".join(reaction["reactants"])
            products_str = " + ".join(reaction["products"])
            st.markdown(f"**Reaction**: {reactants_str} → {products_str}")
            
            # Conditions
            st.markdown(f"**Conditions**: {reaction['conditions']}")
            
            # Source
            st.markdown(f"**Source**: {reaction['source']}")
            
            # Show the cleaned reaction SMILES for debugging (can be removed in production)
            with st.expander("Reaction SMILES"):
                st.code(reaction.get("cleaned_reaction_smiles", "No valid SMILES available"))
            
            # Create a button that will trigger the analysis
            if st.button(f"Analyze Reaction {i+1}", key=f"analyze_btn_{i}"):
                # Use the cleaned SMILES if available, otherwise generate from reactants and products
                if "cleaned_reaction_smiles" in reaction and reaction["cleaned_reaction_smiles"]:
                    st.session_state.selected_rxn_smiles = reaction["cleaned_reaction_smiles"]
                    st.session_state.query = f"Give full information about this rxn {reaction['cleaned_reaction_smiles']}"
                    st.session_state.show_analyze_section = True
                    st.rerun()
                else:
                    if analyze_reaction(reaction["reactants"], reaction["products"]):
                        st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

# Analysis Section - shows either when a reaction is selected for analysis or when directly analyzing
if st.session_state.show_analyze_section:
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown("## Reaction Analysis")
    
    # Show selected reaction SMILES
    if st.session_state.selected_rxn_smiles:
        st.markdown("### Selected Reaction")
        st.code(st.session_state.selected_rxn_smiles)
    
    # Process the selected reaction for analysis
    if st.session_state.selected_rxn_smiles and not st.session_state.analysis_result:
        with st.spinner("Analyzing reaction..."):
            # Create the full analysis query
            analysis_query = f"Give full information about this rxn {st.session_state.selected_rxn_smiles}"
            
            # Use the callback handler for streaming updates
            callback_container = st.container()
            st_callback = StreamlitCallbackHandler(callback_container)
            
            try:
                # Pass the callback to your backend function
                result = enhanced_query(analysis_query, callbacks=[st_callback])
                st.session_state.analysis_result = result
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                import traceback
                with st.expander("See error details"):
                    st.code(traceback.format_exc())
    
    # Display analysis results
    if st.session_state.analysis_result:
        # Display visualization if available
        if st.session_state.analysis_result.get('visualization_path') and os.path.exists(st.session_state.analysis_result.get('visualization_path')):
            st.markdown("### Reaction Visualization")
            st.image(st.session_state.analysis_result.get('visualization_path'), caption="Chemical Reaction Visualization")
            
            # Add this line to make file path less prominent but still available
            with st.expander("Image file details"):
                st.code(st.session_state.analysis_result.get('visualization_path'))
        
        # Display the analysis text
        st.markdown("### Analysis Results")
        st.markdown('<div class="result-area">', unsafe_allow_html=True)
        st.markdown(st.session_state.analysis_result.get('analysis', 'No analysis available'))
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add a query box for follow-up questions about the reaction
        st.markdown('<div class="query-box">', unsafe_allow_html=True)
        st.markdown("### Ask About This Reaction")
        
        rxn_query = st.text_area(
            "Enter your question about this reaction:",
            value=st.session_state.rxn_query,
            height=80,
            placeholder="Example: What is the mechanism of this reaction? How does this reaction relate to industrial processes?",
            key="rxn_query_input"
        )
        
        if st.button("Ask Question", key="ask_rxn_btn") and rxn_query:
            st.session_state.rxn_query = rxn_query
            
            with st.spinner("Processing your question..."):
                try:
                    # Create a query that includes both the reaction SMILES and the user's question
                    combined_query = f"For the reaction {st.session_state.selected_rxn_smiles}, answer this question: {rxn_query}"
                    
                    # Use the callback handler for streaming updates
                    query_response_container = st.container()
                    query_st_callback = StreamlitCallbackHandler(query_response_container)
                    
                    # Get the response
                    query_result = enhanced_query(combined_query, callbacks=[query_st_callback])
                    
                    # Display the answer
                    st.markdown("### Answer")
                    st.markdown('<div class="result-area">', unsafe_allow_html=True)
                    st.markdown(query_result.get('analysis', 'No answer available'))
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.markdown("Please check your question format and try again.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add a button to clear analysis and return to retrosynthesis results
        if st.button("← Back to Retrosynthesis Results", key="back_btn"):
            st.session_state.show_analyze_section = False
            st.session_state.selected_rxn_smiles = None
            st.session_state.analysis_result = None
            st.session_state.rxn_query = ""
            st.rerun()
            
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.caption("ChemCopilot - Your Expert Chemistry Assistant")