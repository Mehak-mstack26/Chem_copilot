# app.py
import streamlit as st
# import requests # Keep for potential future API calls - currently unused directly in app.py
import re
import sys # sys import seems unused, can be removed if not needed elsewhere
import os
import api_config # Ensures API key is set from environment, used by ChatOpenAI
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI

os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

# Import from test.py
from test import enhanced_query, tools as agent_tools # Import tools list

# --- NameToSMILES tool setup for app.py's direct use ---
name_to_smiles_tool_instance = None
for tool_in_agent_list in agent_tools:
    if tool_in_agent_list.name == "NameToSMILES": # Use the exact name defined in make_tools
        name_to_smiles_tool_instance = tool_in_agent_list
        break

if name_to_smiles_tool_instance is None:
    # This is a critical failure if the tool is expected for pre-processing.
    # We can let Streamlit show an error, or handle it by disabling the feature.
    st.error("Fatal Error: NameToSMILES tool not found in the ChemCopilot agent's toolset. Name conversion will not work.")
    # Depending on requirements, you might want to st.stop() or disable parts of the UI.
    # For now, we'll allow the app to run, but get_smiles_from_name_app will fail gracefully.

# Set up the Streamlit page
st.set_page_config(
    page_title="ChemCopilot - Chemistry Assistant",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling (remains the same)
st.markdown("""
    <style>
    .main-header {font-size: 42px;font-weight: bold;color: #2e7d32;margin-bottom: 0px;}
    .sub-header {font-size: 20px;color: #5c5c5c;margin-bottom: 30px;}
    .stButton>button {background-color: #2e7d32;color: white;border: none;padding: 10px 24px;border-radius: 4px;font-weight: bold;}
    .stButton>button:hover {background-color: #005005;}
    .tool-card {background-color: #f5f5f5;padding: 10px;border-radius: 5px;margin-bottom: 10px;}
    .result-area {background-color: #f9f9f9;padding: 20px;border-radius: 8px;border-left: 4px solid #2e7d32;}
    .analysis-section {margin-top: 30px;padding-top: 20px;border-top: 1px solid #ddd;}
    .query-box {background-color: #f5f7fa;padding: 15px;border-radius: 8px;margin-top: 20px;border-left: 4px solid #3f51b5;}
    </style>
    """, unsafe_allow_html=True)

# Initialize session state variables
if 'user_query_input' not in st.session_state:
    st.session_state.user_query_input = ""
if 'query_to_analyze' not in st.session_state: # This is the query passed to enhanced_query
    st.session_state.query_to_analyze = ""
if 'original_user_provided_name' not in st.session_state: # Stores name if app converted it
    st.session_state.original_user_provided_name = None
if 'analyzed_smiles' not in st.session_state: # SMILES confirmed/used by enhanced_query
    st.session_state.analyzed_smiles = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'follow_up_question' not in st.session_state:
    st.session_state.follow_up_question = ""
if 'show_analyze_section' not in st.session_state:
    st.session_state.show_analyze_section = False

# Function to get SMILES from chemical name using NameToSMILES tool
def get_smiles_from_name_app(name):
    """Get SMILES notation for a chemical name using the NameToSMILES tool."""
    if name_to_smiles_tool_instance is None:
        st.warning("NameToSMILES tool is not available. Cannot convert name to SMILES.")
        return None
    try:
        # LangChain tools are typically run with .run() or ._run()
        # The input to tool.run() should match what the tool expects.
        result_str = name_to_smiles_tool_instance.run(name) # Use .run() for Langchain tools
        
        # Attempt to parse SMILES from the tool's string output
        # This parsing is dependent on the specific format of NameToSMILES tool's output
        match = re.search(r"SMILES:\s*([^\s]+)", result_str, re.IGNORECASE)
        if match:
            smiles = match.group(1).strip()
            # Basic SMILES check for single compound (no '>>', common chars)
            if re.match(r"^[A-Za-z0-9@\[\]\.\+\-\=\#\:\(\)\\\/;\$\%\|\{\}]+$", smiles) and ">>" not in smiles:
                return smiles
            st.warning(f"Potentially invalid compound SMILES '{smiles}' from name '{name}'. Tool output: {result_str}")
            return None
        st.error(f"Could not parse SMILES from NameToSMILES tool output for '{name}': {result_str}")
        return None
    except Exception as e:
        st.error(f"Error converting '{name}' to SMILES using NameToSMILES tool: {str(e)}")
        return None

# App Header
st.markdown('<p class="main-header">ChemCopilot</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your Expert Chemistry Assistant</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## Tools Available (via ChemCopilot Agent)")
    with st.expander("🔍 SMILES2Name"): st.markdown("Converts SMILES to chemical names.")
    with st.expander("📝 Name2SMILES"): st.markdown("Converts chemical names to SMILES.")
    with st.expander("🧪 FuncGroups"): st.markdown("Analyzes functional groups.")
    with st.expander("⚛️ BondChangeAnalyzer"): st.markdown("Analyzes bond changes in reactions.")
    with st.expander("📊 ReactionClassifier"): st.markdown("Classifies reactions and provides info.")
    with st.expander("🖼️ ChemVisualizer"): st.markdown("Visualizes molecules and reactions.")

    st.markdown("## Example Queries")
    example_queries = {
        "Info on Aspirin": "Give full information about Aspirin",
        "SMILES for Ethanol": "What is the SMILES for Ethanol?",
        "Analyze Reaction": "Give full information about this rxn CC(=O)Cl.OCCO>>CC(=O)OCCO.Cl",
        "Functional groups of CCO": "What are the functional groups in CCO?"
    }
    for desc, query_text in example_queries.items():
        if st.button(desc, key=f"example_{desc.replace(' ', '_')}"):
            st.session_state.user_query_input = query_text
            st.session_state.query_to_analyze = query_text
            st.session_state.show_analyze_section = True
            st.session_state.analysis_result = None
            st.session_state.analyzed_smiles = None
            st.session_state.original_user_provided_name = None
            st.rerun()

# Main content area
st.markdown("## Compound & Reaction Analysis")

user_input = st.text_input(
    "Enter Compound Name, SMILES, Reaction SMILES, or a Question:",
    value=st.session_state.user_query_input,
    placeholder="e.g., 'full information about Glucose', 'CCO', 'What is an SN2 reaction?'",
    key="main_query_input"
)

if st.button("Submit Query", key="submit_query_button"):
    if user_input:
        st.session_state.user_query_input = user_input
        st.session_state.query_to_analyze = user_input # Initial query to process
        st.session_state.show_analyze_section = True
        st.session_state.analysis_result = None # Clear previous results
        st.session_state.analyzed_smiles = None
        st.session_state.original_user_provided_name = None
        # No rerun here, let the flow continue to analysis section
    else:
        st.warning("Please enter a query.")

if st.session_state.show_analyze_section and st.session_state.query_to_analyze:
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown("## Analysis Workspace")

    query_for_processing = st.session_state.query_to_analyze
    original_name_for_query_param = None # Will be set if name conversion happens

    # Pre-processing for "full information about [name]" or "full analysis of [name]"
    full_info_name_match = re.match(r"(full information about|full analysis of|analyze|details on)\s+(.+)", query_for_processing, re.IGNORECASE)
    
    if full_info_name_match:
        subject_name_or_smiles = full_info_name_match.group(2).strip()
        # Heuristic: if it doesn't look like a SMILES/reaction SMILES already, try to convert name.
        is_likely_name = ">>" not in subject_name_or_smiles and \
                         not re.search(r"[\[\]@\.\+\-\=\#\:\(\)\\\/;\$\%\|\{\}\d]{2,}", subject_name_or_smiles) and \
                         (" " in subject_name_or_smiles or subject_name_or_smiles.isalpha() or len(subject_name_or_smiles.split()) > 1)

        if is_likely_name:
            with st.spinner(f"Converting '{subject_name_or_smiles}' to SMILES..."):
                smiles = get_smiles_from_name_app(subject_name_or_smiles)
            if smiles:
                st.info(f"Converted '{subject_name_or_smiles}' to SMILES: `{smiles}` for analysis.")
                # Reformulate query for the agent, providing both SMILES and original name context.
                # This helps `enhanced_query` route to the general agent for single compounds.
                query_for_processing = (
                    f"Give me full information about the compound with SMILES {smiles}. "
                    f"This compound was originally referred to as '{subject_name_or_smiles}'."
                )
                st.session_state.analyzed_smiles = smiles # Tentatively set, might be updated by agent's response
                st.session_state.original_user_provided_name = subject_name_or_smiles
                original_name_for_query_param = subject_name_or_smiles
            else:
                st.warning(f"Could not convert '{subject_name_or_smiles}' to SMILES. Proceeding with the original query. The agent will attempt conversion if needed.")
                # query_for_processing remains original; original_name_for_query_param remains None
    
    st.markdown("### Query for ChemCopilot")
    st.code(query_for_processing) # Show the (potentially reformulated) query
    
    if st.session_state.original_user_provided_name:
        st.markdown(f"**Original Name Context:** `{st.session_state.original_user_provided_name}`")
    if st.session_state.analyzed_smiles: # If pre-conversion set a SMILES
         st.markdown(f"**Initial Target SMILES:** `{st.session_state.analyzed_smiles}` (may be refined by agent)")


    if not st.session_state.analysis_result: # Only run if no result yet for this query
        with st.spinner("ChemCopilot is thinking..."):
            callback_container = st.container() # For StreamlitCallbackHandler logs
            st_callback = StreamlitCallbackHandler(callback_container)
            
            try:
                # Pass original_name_for_query_param if set, otherwise it's None
                result = enhanced_query(
                    query_for_processing, 
                    callbacks=[st_callback],
                    original_compound_name=original_name_for_query_param 
                )
                st.session_state.analysis_result = result

                # Update analyzed_smiles based on what enhanced_query actually processed
                # This requires enhanced_query to return 'processed_smiles_for_tools'
                if 'processed_smiles_for_tools' in result:
                    processed_smiles = result['processed_smiles_for_tools']
                    if processed_smiles: # If a SMILES was indeed processed
                        st.session_state.analyzed_smiles = processed_smiles
                        # If original_user_provided_name wasn't set by app's pre-processing,
                        # but agent found SMILES, original name context might be missing or be the SMILES itself.
                        if not st.session_state.original_user_provided_name:
                             # If query was directly a SMILES, original_name_for_query_param would be None.
                             # test.py's save_analysis_to_file handles original_compound_name=None or SMILES.
                             pass
                    else: # Agent did not focus on a specific SMILES
                        if not original_name_for_query_param: # If not a name-to-smiles pre-process case
                            st.session_state.analyzed_smiles = None # Clear if no SMILES was focused on by agent
                
                if st.session_state.analyzed_smiles:
                    st.markdown(f"**SMILES Processed by Agent:** `{st.session_state.analyzed_smiles}`")

            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                import traceback
                with st.expander("See error details"):
                    st.code(traceback.format_exc())
    
    if st.session_state.analysis_result:
        viz_path = st.session_state.analysis_result.get('visualization_path')
        if viz_path and isinstance(viz_path, str) and not viz_path.lower().startswith("error"):
            if os.path.exists(viz_path):
                st.markdown("### Visualization")
                try:
                    st.image(viz_path, caption="Chemical Visualization")
                except Exception as e:
                    st.warning(f"Could not display image: {e}")
            else:
                 st.warning(f"Visualization image not found at path: {viz_path}")
        elif viz_path: # It's an error message or non-path string
             st.info(f"Visualization message: {viz_path}")
        
        st.markdown("### Analysis / Answer")
        st.markdown('<div class="result-area">', unsafe_allow_html=True)
        st.markdown(st.session_state.analysis_result.get('analysis', 'No analysis available or an error occurred.'))
        st.markdown('</div>', unsafe_allow_html=True)
        
        # --- Follow-up Section ---
        st.markdown('<div class="query-box">', unsafe_allow_html=True)
        st.markdown("### Ask a Follow-up Question")
        
        follow_up_context_prefix = ""
        if st.session_state.analyzed_smiles:
            # If we have an original name, use it for better readability in context
            if st.session_state.original_user_provided_name and st.session_state.original_user_provided_name != st.session_state.analyzed_smiles:
                 follow_up_context_prefix = f"Regarding '{st.session_state.original_user_provided_name}' (SMILES: {st.session_state.analyzed_smiles}): "
            else:
                follow_up_context_prefix = f"Regarding the SMILES {st.session_state.analyzed_smiles}: "
        elif st.session_state.query_to_analyze: # Fallback to original query if no specific SMILES
             original_user_query_for_context = st.session_state.query_to_analyze # The query that led to current analysis
             follow_up_context_prefix = f"In the context of the previous query ('{original_user_query_for_context[:50]}...'): "

        follow_up_q_text = st.text_area(
            "Enter your follow-up question:",
            value=st.session_state.follow_up_question,
            height=80,
            placeholder="e.g., What is the mechanism? Is this compound toxic?",
            key="follow_up_input"
        )
        
        if st.button("Ask Follow-up", key="ask_follow_up_btn"):
            if follow_up_q_text:
                st.session_state.follow_up_question = follow_up_q_text # Save for display in text_area
                
                with st.spinner("Processing your follow-up question..."):
                    try:
                        combined_follow_up_query = f"{follow_up_context_prefix}{follow_up_q_text}".strip()
                        st.markdown(f"**Sending to ChemCopilot:** `{combined_follow_up_query}`")

                        follow_up_callback_container = st.container()
                        follow_up_st_callback = StreamlitCallbackHandler(follow_up_callback_container)
                        
                        # Determine original_compound_name for the follow-up call
                        # This helps test.py save files with consistent naming for a conversation thread
                        original_context_for_followup_save = st.session_state.original_user_provided_name or st.session_state.analyzed_smiles

                        follow_up_result = enhanced_query(
                            combined_follow_up_query, 
                            callbacks=[follow_up_st_callback],
                            original_compound_name=original_context_for_followup_save
                        )
                        
                        st.markdown("### Answer to Follow-up")
                        st.markdown('<div class="result-area">', unsafe_allow_html=True)
                        st.markdown(follow_up_result.get('analysis', 'No answer available or an error occurred.'))
                        st.markdown('</div>', unsafe_allow_html=True)

                        # Update analyzed_smiles if the follow-up changed context or identified a new SMILES
                        if 'processed_smiles_for_tools' in follow_up_result:
                            processed_smiles_followup = follow_up_result['processed_smiles_for_tools']
                            if processed_smiles_followup and processed_smiles_followup != st.session_state.analyzed_smiles:
                                st.session_state.analyzed_smiles = processed_smiles_followup
                                # If follow-up changes the SMILES, original_user_provided_name might become stale or less relevant
                                # For simplicity, we don't try to get a new name here.
                                st.session_state.original_user_provided_name = None # Or set to new SMILES if no name
                                st.info(f"Follow-up focused on a new SMILES: {st.session_state.analyzed_smiles}")
                        
                    except Exception as e:
                        st.error(f"An error occurred during follow-up: {str(e)}")
                        import traceback
                        with st.expander("See error details for follow-up"):
                           st.code(traceback.format_exc())
            else:
                st.warning("Please enter a follow-up question.")
        st.markdown('</div>', unsafe_allow_html=True) # End query-box
            
    if st.button("Clear Analysis / New Query", key="clear_analysis_btn"):
        st.session_state.show_analyze_section = False
        st.session_state.query_to_analyze = ""
        # st.session_state.user_query_input = "" # Optionally clear the main input box, or leave for editing
        st.session_state.analyzed_smiles = None
        st.session_state.original_user_provided_name = None
        st.session_state.analysis_result = None
        st.session_state.follow_up_question = "" # Clear follow-up text
        st.rerun()
            
    st.markdown('</div>', unsafe_allow_html=True) # End analysis-section

st.markdown("---")
st.caption("ChemCopilot - Your Expert Chemistry Assistant")