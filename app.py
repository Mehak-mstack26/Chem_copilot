# # app.py - Place this file in the root directory of your CHEM_COPILOT folder
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import streamlit as st
import re
import sys
import os

# Add the project root to the Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your function from the test.py file
from test import enhanced_query

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
    </style>
    """, unsafe_allow_html=True)

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
    
    example1 = "What is the reaction name having this smiles CCCl.CC[O-].[Na+]>>CCOCC.[Na+].[Cl-]"
    example2 = "Give full information about this rxn CBr.[Na+].[I-]>>CI.[Na+].[Br-]"
    example3 = "What functional group transformations occur in this reaction: O=C(O)c1ccccc1.HN=[N+]=[N-] >> NC(=O)c1ccccc1.N#N.O"
    
    if st.button("Reaction Name from SMILES"):
        st.session_state.query = example1
        
    if st.button("Full Reaction Analysis"):
        st.session_state.query = example2
        
    if st.button("Identify Functional Groups"):
        st.session_state.query = example3

# Main content area
st.markdown("## Enter Your Chemistry Query")

# Initialize session state for query if it doesn't exist
if 'query' not in st.session_state:
    st.session_state.query = ""

# Input box for user query
query = st.text_area(
    "Ask me about chemical reactions and molecules:",
    value=st.session_state.query,
    height=100,
    placeholder="Example: Give full information about this rxn CBr.[Na+].[I-]>>CI.[Na+].[Br-]"
)

# Action buttons
col1, col2 = st.columns([1, 5])
with col1:
    analyze_button = st.button("Analyze")
with col2:
    if st.button("Clear"):
        st.session_state.query = ""
        st.experimental_rerun()

# Process the query and display results
if analyze_button and query:
    st.session_state.query = query  # Save query to session state

    with st.spinner("Analyzing your chemistry query..."):
        callback_container = st.container()
        st_callback = StreamlitCallbackHandler(callback_container)
        try:
            # Pass the callback to your backend function
            result = enhanced_query(query, callbacks=[st_callback])
            
            st.session_state.last_result = result
            
            # Display the final answer as before
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

elif analyze_button and not query:
    st.error("Please enter a query first.")

# Display previous result if it exists
elif 'last_result' in st.session_state and st.session_state.query:
    st.markdown("## Previous Results")
    st.markdown('<div class="result-area">', unsafe_allow_html=True)
    st.markdown(st.session_state.last_result)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.caption("ChemCopilot - Your Expert Chemistry Assistant")







# from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
# import streamlit as st
# import re
# import sys
# import os
# import json

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from test import enhanced_query, format_funcgroups_json

# st.set_page_config(
#     page_title="ChemCopilot - Chemistry Assistant",
#     page_icon="🧪",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Your CSS and sidebar code here (unchanged)...

# st.markdown("## Enter Your Chemistry Query")

# if 'query' not in st.session_state:
#     st.session_state.query = ""

# query = st.text_area(
#     "Ask me about chemical reactions and molecules:",
#     value=st.session_state.query,
#     height=100,
#     placeholder="Example: Give full information about this rxn CBr.[Na+].[I-]>>CI.[Na+].[Br-]"
# )

# col1, col2 = st.columns([1, 5])
# with col1:
#     analyze_button = st.button("Analyze")
# with col2:
#     if st.button("Clear"):
#         st.session_state.query = ""
#         st.experimental_rerun()

# if analyze_button and query:
#     st.session_state.query = query

#     with st.spinner("Analyzing your chemistry query..."):
#         callback_container = st.container()
#         st_callback = StreamlitCallbackHandler(callback_container)
#         try:
#             name_result, funcgroups_json, bond_result = enhanced_query(query, callbacks=[st_callback])

#             if funcgroups_json is not None:
#                 st.write("DEBUG funcgroups_json:", funcgroups_json)
#                 st.text(f"Type of funcgroups_json: {type(funcgroups_json)}")
#                 st.markdown("### Functional Groups (Raw JSON):")

#                 if isinstance(funcgroups_json, dict):
#                     st.json(funcgroups_json, expanded=True)
#                 elif isinstance(funcgroups_json, str):
#                     try:
#                         funcgroups_dict = json.loads(funcgroups_json)
#                         st.json(funcgroups_dict, expanded=True)
#                     except json.JSONDecodeError as e:
#                         st.error(f"Could not parse string as JSON: {e}")
#                         st.text(funcgroups_json)
#                 else:
#                     st.error("Unknown format for funcgroups_json")
#                     st.text(str(funcgroups_json))


#                 funcgroups_text = format_funcgroups_json(funcgroups_json)
#                 st.write("DEBUG formatted summary:", funcgroups_text)
#                 st.markdown("### Functional Groups (Summary):")
#                 st.markdown(f"```\n{funcgroups_text}\n```")

#             # ⬇️ Now comes the final answer
#             st.markdown("## Final Answer")
#             st.markdown('<div class="result-area">', unsafe_allow_html=True)
#             st.markdown(name_result)
#             st.markdown('</div>', unsafe_allow_html=True)


#         except Exception as e:
#             st.error(f"An error occurred: {str(e)}")



# elif analyze_button and not query:
#     st.error("Please enter a query first.")

# elif 'last_result' in st.session_state and st.session_state.query:
#     st.markdown("## Previous Results")
#     st.markdown('<div class="result-area">', unsafe_allow_html=True)
#     st.markdown(st.session_state.last_result)
#     st.markdown('</div>', unsafe_allow_html=True)

# st.markdown("---")
# col1, col2, col3 = st.columns([1, 2, 1])
# with col2:
#     st.caption("ChemCopilot - Your Expert Chemistry Assistant")
