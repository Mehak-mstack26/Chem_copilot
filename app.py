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







import streamlit as st
import requests
import re
import sys
import os
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI

# Add the project root to the Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your function from the test.py file
from test import enhanced_query

# Initialize LLM for name to SMILES conversion
llm = ChatOpenAI(model="gpt-4o", temperature=0)

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
    </style>
    """, unsafe_allow_html=True)

# Initialize session state variables
if 'query' not in st.session_state:
    st.session_state.query = ""
if 'retro_results' not in st.session_state:
    st.session_state.retro_results = None
if 'selected_rxn_smiles' not in st.session_state:
    st.session_state.selected_rxn_smiles = None
if 'current_tab_index' not in st.session_state:
    st.session_state.current_tab_index = 0
if 'auto_analyze' not in st.session_state:
    st.session_state.auto_analyze = False
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

# Function to convert reactant and product names to SMILES
def convert_to_reaction_smiles(reactants, products):
    """Convert reactant and product names to reaction SMILES format using LLM"""
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
        else:
            # If LLM didn't output proper reaction SMILES, try to construct manually
            # Join names with dots (for multiple reactants/products) and '>>' separator
            reactants_str = '.'.join(reactants)
            products_str = '.'.join(products)
            return f"{reactants_str}>>{products_str}"
    except Exception as e:
        st.error(f"Error converting to reaction SMILES: {str(e)}")
        return None

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
        # Use LLM to convert names to SMILES
        return convert_to_reaction_smiles(reactants, products)

# Function to handle reaction analysis button clicks
def analyze_reaction(reactants, products):
    """Analyze reaction by converting names to SMILES if necessary"""
    reaction_smiles = create_reaction_smiles(reactants, products)
    
    if reaction_smiles:
        st.session_state.selected_rxn_smiles = reaction_smiles
        st.session_state.query = f"Give full information about this rxn {reaction_smiles}"
        st.session_state.current_tab_index = 1  # Set to Analysis tab (index 1)
        st.session_state.auto_analyze = True
    else:
        st.error("Could not create reaction SMILES. Please check the reactants and products.")

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
    
    if st.button("Search Retrosynthesis for Flubendiamide"):
        st.session_state.query = "flubendiamide"
        st.session_state.current_tab_index = 0  # Set to Search tab
        
    if st.button("Search for 2-amino-5-chloro-3-methyl benzoic acid"):
        st.session_state.query = "2-amino-5-chloro-3-methyl benzoic acid"
        st.session_state.current_tab_index = 0  # Set to Search tab
        
    if st.button("Reaction Analysis Example"):
        st.session_state.query = "Give full information about this rxn CCCl.CC[O-].[Na+]>>CCOCC.[Na+].[Cl-]"
        st.session_state.current_tab_index = 1  # Set to Analysis tab

# Create tabs with selected parameter
tabs = st.tabs(["Retrosynthesis Search", "Reaction Analysis"])
selected_tab = tabs[st.session_state.current_tab_index]

# Retrosynthesis Search Tab
with tabs[0]:
    st.markdown("## Search Retrosynthesis Pathways")
    
    compound_name = st.text_input(
        "Enter compound name (IUPAC or common name):",
        value=st.session_state.query if st.session_state.current_tab_index == 0 else "",
        placeholder="Example: flubendiamide, 2-amino-5-chloro-3-methyl benzoic acid",
        key="search_compound_name"
    )
    
    # Simple search button without advanced settings
    search_button = st.button("Search Retrosynthesis", key="search_retro_button")
    
    if search_button and compound_name:
        st.session_state.query = compound_name
        with st.spinner("Searching for retrosynthesis pathways..."):
            try:
                # Make API call to retrosynthesis service with default payload
                payload = {
                    "material": compound_name,
                    "num_results": 10,
                    "alignment": True,
                    "expansion": True,
                    "filtration": False
                }
                
                response = requests.post(
                    "http://localhost:8000/retro-synthesis/",
                    json=payload
                )
                
                if response.status_code == 200:
                    retro_data = response.json()
                    if retro_data.get("status") == "success":
                        st.session_state.retro_results = retro_data["data"]
                        st.success(f"Found {len(retro_data['data']['reactions'])} reactions for {compound_name}")
                    else:
                        st.error(f"API returned an error: {retro_data.get('message', 'Unknown error')}")
                else:
                    st.error(f"API request failed with status code: {response.status_code}")
                    st.error(f"Response content: {response.text}")
            except Exception as e:
                st.error(f"Failed to fetch retrosynthesis data: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    
    # Display retrosynthesis results if available
    if st.session_state.retro_results:
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
                
                # Create a button that will trigger the analysis
                if st.button(f"Analyze Reaction {i+1}", key=f"analyze_btn_{i}"):
                    analyze_reaction(reaction["reactants"], reaction["products"])
                    st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)

# Reaction Analysis Tab
with tabs[1]:
    st.markdown("## Analyze Chemical Reactions")
    
    # Input box for user query
    query = st.text_area(
        "Ask me about chemical reactions and molecules:",
        value=st.session_state.query if st.session_state.current_tab_index == 1 else "",
        height=100,
        placeholder="Example: Give full information about this rxn CCCl.CC[O-].[Na+]>>CCOCC.[Na+].[Cl-]",
        key="analysis_query"
    )
    
    # Action buttons
    col1, col2 = st.columns([1, 5])
    with col1:
        analyze_button = st.button("Analyze", key="analyze_reaction_btn")
    with col2:
        if st.button("Clear", key="clear_analyze_btn"):
            st.session_state.query = ""
            st.rerun()
    
    # Process the query and display results - either from button click or auto-analyze
    if (analyze_button and query) or (st.session_state.auto_analyze and query):
        # Reset auto-analyze flag
        st.session_state.auto_analyze = False
        
        st.session_state.query = query  # Save query to session state
        
        with st.spinner("Analyzing your chemistry query..."):
            callback_container = st.container()
            st_callback = StreamlitCallbackHandler(callback_container)
            try:
                # Pass the callback to your backend function
                result = enhanced_query(query, callbacks=[st_callback])
                
                st.session_state.last_result = result
                
                # Display the final answer
                st.markdown("## Results")
                st.markdown('<div class="result-area">', unsafe_allow_html=True)
                st.markdown(result)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Check if it's a reaction query
                if ">>" in query or "rxn" in query.lower():
                    # Extract SMILES using regex
                    smiles_match = re.search(r"([A-Za-z0-9@\[\]\.\+\-\=\#\(\)]+>>[A-Za-z0-9@\[\]\.\+\-\=\#\(\)\.]*)", query)
                    if smiles_match:
                        reaction_smiles = smiles_match.group(1)
                        st.markdown("### Reaction SMILES")
                        st.code(reaction_smiles)
                        
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.markdown("Please check your query format and try again.")
    
    elif 'last_result' in st.session_state and st.session_state.last_result and st.session_state.query:
        st.markdown("## Previous Results")
        st.markdown('<div class="result-area">', unsafe_allow_html=True)
        st.markdown(st.session_state.last_result)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.caption("ChemCopilot - Your Expert Chemistry Assistant")