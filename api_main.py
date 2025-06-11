import os
import sys
from fastapi import FastAPI, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse # For serving the frontend
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uvicorn
import traceback

# --- Path Setup ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = APP_DIR # This refers to the root of the API application itself
sys.path.insert(0, PROJECT_ROOT) # Add API app's root to path
print(f"[API INFO] Added '{PROJECT_ROOT}' to sys.path.")

# --- Import Chem Copilot Components ---
try:
    # MODIFIED IMPORT STATEMENT:
    import chem_copilot_integrated as chem_analyzer
    print("[API INFO] Successfully imported 'chem_copilot_integrated' as chem_analyzer.")
    # Now, chem_analyzer.PROJECT_ROOT_DIR refers to the root of the chem_analyzer module,
    # which is important for its internal file access (like pricing files and visualization output dir)
except ImportError as e:
    # MODIFIED ERROR MESSAGE:
    print(f"[API FATAL ERROR] Could not import chem_copilot_integrated: {e}")
    print(f"  Ensure 'chem_copilot_integrated.py' (your integrated script) is in an importable location.")
    print(f"  Current sys.path: {sys.path}")
    sys.exit(1)
except Exception as e_init:
    print(f"[API FATAL ERROR] An unexpected error during import/initialization of chem_analyzer: {e_init}")
    traceback.print_exc()
    sys.exit(1)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Chem Copilot API",
    description="API for chemical analysis and chat using ChemCopilot's integrated logic.",
    version="0.4.1" # Incremented version slightly for this change
)

# --- Pydantic Models ---
class GeneralQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, example="What are the functional groups in CCO?")
    original_name_for_saving: Optional[str] = Field(None, example="Ethanol_FG_Query")
    clear_moi_before_query: Optional[bool] = Field(False, description="If true, MOI context is cleared before processing this query.")

# MODIFIED AnalysisResponse Model
class AnalysisResponse(BaseModel):
    analysis: Optional[Any] = "No analysis provided." 
    visualization_path: Optional[str] = None
    # Removed: processed_smiles, error, query_context_for_filename, current_moi_name, current_moi_smiles
    # If you want to include 'error' only when it's not null, that requires more complex logic
    # or sending it and letting the client ignore it if null.
    # For strict removal of the key itself when error is null, FastAPI's `response_model_exclude_none=True`
    # would handle it if 'error' was Optional[str] = None in the Pydantic model.
    # But since you want to remove 'error' key entirely from the *model definition* when error is null,
    # we'd have to conditionally build the response.
    # A simpler approach for now is to define it as Optional and let exclude_none handle it.
    error: Optional[str] = None # Keep error for now, will be excluded if None by response_model_exclude_none

class ChatRequest(BaseModel): 
    query: Optional[str] = Field(None, min_length=1, example="What are its functional groups?")
    clear_history: Optional[bool] = Field(False)

class ChatResponse(BaseModel): 
    reply: Optional[str] = "No reply generated."
    visualization_path: Optional[str] = None
    error: Optional[str] = None

# --- FastAPI Event Handler for Startup ---
@app.on_event("startup")
async def startup_event():
    print("[API INFO] FastAPI application startup...")
    # Assuming chem_analyzer (your integrated script) has DEFAULT_LLM_PROVIDER defined
    if not os.getenv("OPENAI_API_KEY") and getattr(chem_analyzer, 'DEFAULT_LLM_PROVIDER', 'openai') == "openai":
        print("[API STARTUP WARNING] OPENAI_API_KEY IS NOT SET and OpenAI is a potential provider.")
    elif os.getenv("OPENAI_API_KEY"):
        print("[API STARTUP] OPENAI_API_KEY is set.")

    if not os.getenv("PERPLEXITY_API_KEY") and getattr(chem_analyzer, 'DEFAULT_LLM_PROVIDER', 'openai') == "perplexity":
        print("[API STARTUP WARNING] PERPLEXITY_API_KEY IS NOT SET and Perplexity is a potential provider.")
    elif os.getenv("PERPLEXITY_API_KEY"):
        print("[API STARTUP] PERPLEXITY_API_KEY is set.")


    print("[API INFO] Initializing Autogen agents and tool wrappers from chem_analyzer...")
    try:
        # These functions should prepare the agents within chem_analyzer
        if hasattr(chem_analyzer, 'get_tool_agents'):
            chem_analyzer.get_tool_agents()
        if hasattr(chem_analyzer, 'get_chatbot_agents'):
            chem_analyzer.get_chatbot_agents()
        # if hasattr(chem_analyzer, 'get_tools_for_core_logic'): # This function was not in the provided chem_copilot_integrated.py
        #     chem_analyzer.get_tools_for_core_logic()
        print("[API INFO] Autogen components initialization attempted via chem_analyzer.")
    except Exception as e:
        print(f"[API STARTUP ERROR] Failed to initialize Autogen components via chem_analyzer: {e}")
        traceback.print_exc()

    # Mount static directory for frontend UI files (relative to this api_main.py file)
    frontend_static_dir = os.path.join(APP_DIR, "frontend_static")
    if os.path.isdir(frontend_static_dir):
        app.mount("/ui", StaticFiles(directory=frontend_static_dir, html=True), name="frontend")
        print(f"[API INFO] Mounted frontend static files from: {frontend_static_dir} at /ui")
    else:
        print(f"[API WARNING] Frontend static directory not found at '{frontend_static_dir}'. UI may not be accessible via / or /ui.")

    # Mount static directory for tool-generated visualizations
    # This path comes from chem_analyzer's PROJECT_ROOT_DIR, which is where it saves visualizations.
    # Ensure chem_analyzer has PROJECT_ROOT_DIR defined at its module level
    if hasattr(chem_analyzer, 'PROJECT_ROOT_DIR'):
        tool_visualizations_output_base_dir = os.path.join(chem_analyzer.PROJECT_ROOT_DIR, "static", "autogen_visualizations")
        tool_visualizations_serve_dir = os.path.join(chem_analyzer.PROJECT_ROOT_DIR, "static")

        if os.path.isdir(tool_visualizations_serve_dir):
            if not os.path.exists(tool_visualizations_output_base_dir):
                try:
                    os.makedirs(tool_visualizations_output_base_dir, exist_ok=True)
                    print(f"[API INFO] Created tool visualization output directory: {tool_visualizations_output_base_dir}")
                except Exception as e_dir:
                    print(f"[API WARNING] Could not create tool visualization output directory {tool_visualizations_output_base_dir}: {e_dir}")

            app.mount("/static", StaticFiles(directory=tool_visualizations_serve_dir), name="tool_visualizations")
            print(f"[API INFO] Mounted tool visualization static files from (serving '{tool_visualizations_serve_dir}' at '/static')")
        else:
            print(f"[API WARNING] Tool visualizations base directory '{tool_visualizations_serve_dir}' (from chem_analyzer.PROJECT_ROOT_DIR/static) not found.")
    else:
        print("[API WARNING] `chem_analyzer.PROJECT_ROOT_DIR` not found. Cannot reliably mount tool visualization directory.")


# --- Helper for Visualization Path ---
def _make_web_accessible_viz_path(local_path: Optional[str]) -> Optional[str]:
    if not local_path or not isinstance(local_path, str):
        return None
    
    if not hasattr(chem_analyzer, 'PROJECT_ROOT_DIR'):
        print("[API VIZ_PATH ERROR] chem_analyzer.PROJECT_ROOT_DIR is not defined. Cannot make path web accessible.")
        return local_path # Return as is, might be external or already correct

    chem_analyzer_static_dir = os.path.join(chem_analyzer.PROJECT_ROOT_DIR, "static")
    
    if os.path.isabs(local_path):
        if local_path.startswith(chem_analyzer_static_dir):
            relative_path = os.path.relpath(local_path, chem_analyzer_static_dir)
            web_path = f"/static/{relative_path}".replace("\\", "/")
            print(f"[API VIZ_PATH] Converted absolute path '{local_path}' to web path '{web_path}'")
            return web_path
        else:
            print(f"[API VIZ_PATH WARNING] Absolute path '{local_path}' is not within chem_analyzer's static dir '{chem_analyzer_static_dir}'. Cannot make web accessible.")
            return None
    elif local_path.startswith("static/"): 
        web_path = f"/{local_path}".replace("\\", "/")
        print(f"[API VIZ_PATH] Relative path '{local_path}' made web path '{web_path}'")
        return web_path
    
    print(f"[API VIZ_PATH WARNING] Could not reliably convert path '{local_path}' to web accessible.")
    return local_path


# --- Root Endpoint (Serves the HTML UI) ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False, tags=["Frontend UI"])
async def serve_frontend_ui():
    frontend_html_path = os.path.join(APP_DIR, "frontend_static", "index.html")
    if os.path.exists(frontend_html_path):
        return FileResponse(frontend_html_path)
    return HTMLResponse(content="<h1>ChemCopilot API</h1><p>Frontend UI not found at root. Try <a href='/ui/'>/ui/</a> if available.</p><p>API docs at <a href='/docs'>/docs</a>.</p>", status_code=200)

# +++ NEW GENERAL COPILOT QUERY ENDPOINT +++
@app.post("/api/v1/copilot_query", response_model=AnalysisResponse, response_model_exclude_none=True, tags=["General Query"])
async def general_copilot_query_endpoint(request: GeneralQueryRequest):
    print(f"[API /api/v1/copilot_query] Query: '{request.query}', Clear MOI: {request.clear_moi_before_query}, Orig Name: '{request.original_name_for_saving}'")

    if request.clear_moi_before_query:
        try:
            if hasattr(chem_analyzer, 'clear_chatbot_memory_autogen'):
                chem_analyzer.clear_chatbot_memory_autogen()
                print("[API /api/v1/copilot_query] MOI context cleared due to request flag.")
            else:
                print("[API WARNING /api/v1/copilot_query] chem_analyzer has no 'clear_chatbot_memory_autogen' function.")
        except Exception as e_clear:
            print(f"[API WARNING /api/v1/copilot_query] Error clearing MOI: {e_clear}")

    try:
        if not hasattr(chem_analyzer, 'enhanced_query'):
            raise HTTPException(status_code=501, detail="Core 'enhanced_query' function not found in chem_analyzer module.")

        result_dict = chem_analyzer.enhanced_query(
            full_query=request.query,
            original_compound_name=request.original_name_for_saving
        )
        if not isinstance(result_dict, dict):
            print(f"[API ERROR /api/v1/copilot_query] chem_analyzer.enhanced_query did not return a dict. Got: {type(result_dict)}")
            raise HTTPException(status_code=500, detail="Internal error: Invalid response format from core logic.")

        web_viz_path = _make_web_accessible_viz_path(result_dict.get("visualization_path"))
        
        # We no longer include these in AnalysisResponse by default
        # processed_smiles_val = result_dict.get("processed_smiles_for_tools")
        # query_context_val = result_dict.get("analysis_context")
        # current_moi_name_after = getattr(chem_analyzer, '_current_moi_context', {}).get("name")
        # current_moi_smiles_after = getattr(chem_analyzer, '_current_moi_context', {}).get("smiles")
        error_val = result_dict.get("error")


        return AnalysisResponse(
            analysis=result_dict.get("analysis", "No analysis content from enhanced_query."),
            visualization_path=web_viz_path,
            error=error_val # error will be excluded if None by response_model_exclude_none=True
            # The other 4 fields are no longer part of the AnalysisResponse model
        )
    except Exception as e:
        print(f"[API ERROR /api/v1/copilot_query] Exception during enhanced_query call: {e}")
        traceback.print_exc()
        # moi_name_on_error = getattr(chem_analyzer, '_current_moi_context', {}).get("name")
        # moi_smiles_on_error = getattr(chem_analyzer, '_current_moi_context', {}).get("smiles")
        return AnalysisResponse(
            analysis=f"An internal server error occurred while processing your query: {str(e)}",
            error=str(e) # error will be excluded if None by response_model_exclude_none=True
            # The other 4 fields are no longer part of the AnalysisResponse model
        )

# --- Endpoint to Clear MOI Context Explicitly ---
@app.post("/api/v1/clear_moi_context", status_code=200, response_model=Dict[str,str], tags=["Context Management"])
async def clear_moi_context_endpoint():
    try:
        if hasattr(chem_analyzer, 'clear_chatbot_memory_autogen'):
            chem_analyzer.clear_chatbot_memory_autogen()
            return {"message": "MOI context cleared successfully."}
        else:
            raise HTTPException(status_code=501, detail="MOI clearing function not available.")
    except Exception as e:
        print(f"[API ERROR /api/v1/clear_moi_context] Exception: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error clearing MOI context: {str(e)}")

# --- Endpoint to Get Current MOI Context ---
@app.get("/api/v1/get_moi_context", response_model=Dict[str, Optional[str]], tags=["Context Management"])
async def get_current_moi_context_endpoint():
    try:
        return {
            "name": getattr(chem_analyzer, '_current_moi_context', {}).get("name"),
            "smiles": getattr(chem_analyzer, '_current_moi_context', {}).get("smiles")
        }
    except Exception as e:
        print(f"[API ERROR /api/v1/get_moi_context] Exception: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error retrieving MOI context: {str(e)}")

# --- Legacy/Direct Endpoints ---
@app.post("/analyze/full", response_model=AnalysisResponse, response_model_exclude_none=True, tags=["Specific Endpoints (Legacy/Direct)"])
async def analyze_full_information_endpoint(request: GeneralQueryRequest):
    print(f"[API /analyze/full] Query: '{request.query}', Original Name: '{request.original_name_for_saving}'")
    # This endpoint directly calls the general copilot query
    # The returned object from general_copilot_query_endpoint will already be an AnalysisResponse
    # instance adhering to the new (reduced) model structure.
    return await general_copilot_query_endpoint(request)

@app.post("/query/tool_agent", response_model=AnalysisResponse, response_model_exclude_none=True, tags=["Specific Endpoints (Legacy/Direct)"])
async def query_with_tool_agent_endpoint(request: GeneralQueryRequest):
    print(f"[API /query/tool_agent] Query: '{request.query}'")
    try:
        if not hasattr(chem_analyzer, 'run_autogen_tool_agent_query'):
             raise HTTPException(status_code=501, detail="Tool agent query function not available.")
        
        result_dict = chem_analyzer.run_autogen_tool_agent_query(request.query)
        if not isinstance(result_dict, dict):
             print(f"[API ERROR /query/tool_agent] Tool agent did not return a dict. Got: {type(result_dict)}")
             raise HTTPException(status_code=500, detail="Internal error: Invalid response from tool agent.")

        web_viz_path = _make_web_accessible_viz_path(result_dict.get("visualization_path"))
        
        return AnalysisResponse(
            analysis=result_dict.get("analysis", "No analysis content from tool agent."),
            visualization_path=web_viz_path,
            error=result_dict.get("error") # Assuming run_autogen_tool_agent_query might return an error
        )
    except Exception as e:
        print(f"[API ERROR /query/tool_agent] Exception: {e}")
        traceback.print_exc()
        return AnalysisResponse(error=f"Internal server error: {str(e)}")

@app.post("/chat/conversational", response_model=ChatResponse, response_model_exclude_none=True, tags=["Specific Endpoints (Legacy/Direct)"])
async def conversational_chat_endpoint(request: ChatRequest):
    print(f"[API /chat/conversational] Query: '{request.query}', Clear: {request.clear_history}")
    try:
        # ... (your existing logic for clear_history and query check) ...
        if not hasattr(chem_analyzer, 'clear_chatbot_memory_autogen') or not hasattr(chem_analyzer, 'run_autogen_chatbot_query'):
            raise HTTPException(status_code=501, detail="Chatbot functions not available.")

        if request.clear_history:
            chem_analyzer.clear_chatbot_memory_autogen()
            # ChatResponse now only has reply, viz_path, error
            return ChatResponse(reply="Chat history and MOI context has been cleared.")

        if request.query is None:
            raise HTTPException(status_code=400, detail="Query is required if not clearing history.")

        result_dict = chem_analyzer.run_autogen_chatbot_query(request.query)
        if not isinstance(result_dict, dict):
            print(f"[API ERROR /chat/conversational] Chatbot did not return a dict. Got: {type(result_dict)}")
            raise HTTPException(status_code=500, detail="Internal error: Invalid response from chatbot.")

        web_viz_path = _make_web_accessible_viz_path(result_dict.get("visualization_path"))

        return ChatResponse(
            reply=result_dict.get("analysis", "Chatbot did not provide a clear reply."), # Map 'analysis' to 'reply'
            visualization_path=web_viz_path,
            error=result_dict.get("error")
        )
    except Exception as e:
        print(f"[API ERROR /chat/conversational] Exception: {e}")
        traceback.print_exc()
        return ChatResponse(reply="An internal server error occurred.", error=str(e))
    
# --- Main block to run Uvicorn ---
if __name__ == "__main__":
    print("Starting Chem Copilot API server with Uvicorn...")
    print(f"API docs (Swagger UI) available at: http://localhost:8000/docs")
    print(f"API docs (ReDoc) available at: http://localhost:8000/redoc")
    print(f"Frontend UI (if index.html is in frontend_static) available at: http://localhost:8000/ or http://localhost:8000/ui/")
    uvicorn.run("api_main:app", host="0.0.0.0", port=8000, reload=True)