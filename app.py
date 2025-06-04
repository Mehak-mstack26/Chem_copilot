import streamlit as st
import os
import sys
import re
import traceback # Ensure traceback is imported

# --- Add the current directory to sys.path to find chem_copilot_autogen_main ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.append(APP_DIR)

# Now import your main script. This will also run its initial setup (API key, etc.)
try:
    import chem_copilot_autogen_main as main_script
    print("[StreamlitApp] Successfully imported chem_copilot_autogen_main.")
except ModuleNotFoundError:
    st.error(
        "CRITICAL ERROR: Could not import `chem_copilot_autogen_main.py`. "
        "Ensure it's in the same directory as `app.py` and that all its dependencies are installed in your environment."
    )
    st.stop()
except Exception as e:
    st.error(f"CRITICAL ERROR during import of `chem_copilot_autogen_main.py`: {e}")
    st.stop()


# --- Streamlit UI Configuration ---
st.set_page_config(page_title="Chem Copilot", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS (Optional - for more advanced styling) ---
# You can create a styles.css file in the same directory and load it
# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# local_css("styles.css") # If you create a styles.css

st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px;
        padding: 10px 15px;
        margin-bottom: 10px;
    }
    /* You can add more specific styles here */
</style>
""", unsafe_allow_html=True)


# --- Main Page Title ---
col1_title, col2_title = st.columns([1, 10]) # Adjust column ratio as needed
with col1_title:
    st.markdown("<h1 style='font-size: 3rem; margin-top: -10px;'>🧪</h1>", unsafe_allow_html=True) # Display emoji as large text
with col2_title:
    st.title("Chem Copilot")
    st.caption("AI-Powered Chemistry Assistant with Autogen")
st.markdown("---")


# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "moi_name" not in st.session_state:
    st.session_state.moi_name = None
if "moi_smiles" not in st.session_state:
    st.session_state.moi_smiles = None
if "last_original_name" not in st.session_state:
    st.session_state.last_original_name = None


# --- Helper function to update MOI display ---
def update_moi_from_backend():
    st.session_state.moi_name = main_script._current_moi_context.get("name")
    st.session_state.moi_smiles = main_script._current_moi_context.get("smiles")

# --- Sidebar for Controls and MOI Display ---
with st.sidebar:
    st.header("⚙️ Controls & Context")
    st.markdown("---")
    if st.button("🗑️ Clear Chat & MOI Context", use_container_width=True):
        main_script.clear_chatbot_memory_autogen()
        st.session_state.messages = []
        update_moi_from_backend()
        st.session_state.last_original_name = None
        st.success("Chat history and MOI context cleared!")
        st.rerun()

    st.markdown("---")
    st.subheader("Molecule of Interest (MOI)")
    if st.session_state.moi_name or st.session_state.moi_smiles:
        st.success(f"**Name:** `{st.session_state.moi_name or 'Not Set'}`\n\n"
                   f"**SMILES:** `{st.session_state.moi_smiles or 'Not Set'}`")

        # Example: Add buttons for common actions if MOI is set
        # if st.session_state.moi_smiles:
        #     if st.button("Get Functional Groups for MOI", use_container_width=True):
        #         st.session_state.predefined_query = f"What are the functional groups of {st.session_state.moi_smiles}?"
        #     if st.button("Visualize MOI", use_container_width=True):
        #         st.session_state.predefined_query = f"Visualize {st.session_state.moi_smiles}"

    else:
        st.info("No MOI set. Use a priming message or ask for 'full info'.")
    
    st.markdown("---")
    with st.expander("💡 Example Priming Message"):
        st.code("Let's discuss Aspirin with SMILES CC(=O)OC1=CC=CC=C1C(=O)O. Please acknowledge.")
    
    with st.expander("📋 Example Queries"):
        st.markdown("""
        - `full info for CCO`
        - `tell me everything about the reaction CCO>>CC=O`
        - `give me the full analysis of Aspirin`
        - (After priming MOI) `What are its functional groups?`
        - `Visualize C1=CC=CS1`
        - `What is the name of C1=CC=CS1?`
        """)


# --- Main Chat Area ---
chat_container = st.container() # Use a container for better control if needed

with chat_container:
    for message in st.session_state.messages:
        avatar_icon = "🧑‍🔬" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar_icon):
            # For assistant messages that are "full info", make them expandable
            if message["role"] == "assistant" and ("Overview of the Compound" in message["content"] or "Reaction Analysis Summary" in message["content"] or len(message["content"]) > 500):
                with st.expander("See Full Analysis Details", expanded=False):
                    st.markdown(message["content"], unsafe_allow_html=True) # Allow HTML for better formatting if your backend produces it
            else:
                st.markdown(message["content"], unsafe_allow_html=True)

            if message.get("image_path"):
                try:
                    if os.path.exists(message["image_path"]):
                        st.image(message["image_path"])
                    else:
                        st.warning(f"Image not found: {message['image_path']}")
                except Exception as e:
                    st.error(f"Could not load image: {message['image_path']}. Error: {e}")

# --- User Input Handling ---
# Check for predefined query from sidebar buttons
# query_to_run = None
# if "predefined_query" in st.session_state and st.session_state.predefined_query:
#     query_to_run = st.session_state.predefined_query
#     st.session_state.predefined_query = None # Clear it after use

user_input = st.chat_input("Ask Chem Copilot...") # Removed 'query_to_run or' for now for simplicity

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍🔬"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response_text = ""
        image_to_display = None

        with st.spinner("Chem Copilot is thinking..."):
            try:
                current_original_name_for_saving = None
                name_match_for_saving = re.search(
                    r"full\s+(?:info|information|analysis|detail)\s+(?:for|of|about)\s+named?\s*['\"]?([^'\"\n\.,]+?)['\"]?(?:\s|$|\.|\?)",
                    user_input, re.IGNORECASE
                ) or re.search(
                     r"full\s+(?:info|information|analysis|detail)\s+(?:for|of|about)\s+([^'\"\n\.,\s]+)(?:\s|$|\.|\?)(?!.*smiles)",
                     user_input, re.IGNORECASE
                )
                if name_match_for_saving:
                    potential_name_for_saving = name_match_for_saving.group(1).strip()
                    if not (">>" in potential_name_for_saving or main_script.extract_single_compound_smiles(potential_name_for_saving)):
                         current_original_name_for_saving = potential_name_for_saving
                         st.session_state.last_original_name = current_original_name_for_saving

                priming_match_loop = re.match(r"Let's discuss the molecule of interest: (.*?) with SMILES .*\. Please acknowledge\.", user_input, re.IGNORECASE)
                if priming_match_loop:
                    st.session_state.last_original_name = priming_match_loop.group(1).strip()

                result = main_script.enhanced_query(
                    user_input,
                    original_compound_name=current_original_name_for_saving or st.session_state.last_original_name
                )

                if result and isinstance(result, dict):
                    full_response_text = result.get("analysis", "No analysis text provided.")
                    if result.get("error"):
                        full_response_text += f"\n\n**Error:** {result.get('error')}"
                    image_to_display = result.get("visualization_path")
                    update_moi_from_backend()

                    analysis_context = result.get('analysis_context', '')
                    if ('full_direct_openai_summary_generated' in analysis_context or \
                        'compound_openai_summary_with_disconnections' in analysis_context or \
                        'full_info_from_name' in analysis_context) and \
                       result.get('processed_smiles_for_tools'):
                        if current_original_name_for_saving:
                             st.session_state.last_original_name = current_original_name_for_saving
                        elif main_script._current_moi_context.get("smiles") == result.get('processed_smiles_for_tools') and main_script._current_moi_context.get("name"):
                            st.session_state.last_original_name = main_script._current_moi_context.get("name")
                else:
                    full_response_text = "Error: Received an unexpected result format from the backend."

            except Exception as e:
                full_response_text = f"An unexpected error occurred in the Streamlit app: {str(e)}\n{traceback.format_exc()}"
                st.error(full_response_text) # Display error prominently

        # Display logic after spinner
        if "Overview of the Compound" in full_response_text or "Reaction Analysis Summary" in full_response_text or len(full_response_text) > 500:
            with message_placeholder.expander("See Full Analysis Details", expanded=True): # Expand by default if it's a new full analysis
                 st.markdown(full_response_text, unsafe_allow_html=True)
        else:
            message_placeholder.markdown(full_response_text, unsafe_allow_html=True)

        if image_to_display:
            try:
                if os.path.exists(image_to_display):
                    st.image(image_to_display)
                else:
                    st.warning(f"Visualization image not found: {image_to_display}")
            except Exception as e_img:
                st.error(f"Error displaying image {image_to_display}: {e_img}")

    st.session_state.messages.append({"role": "assistant", "content": full_response_text, "image_path": image_to_display})
    # No st.rerun() here to allow continued interaction