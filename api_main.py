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
PROJECT_ROOT = APP_DIR
sys.path.insert(0, PROJECT_ROOT)
print(f"[API INFO] Added '{PROJECT_ROOT}' to sys.path.")

# --- Import Chem Copilot Components ---
try:
    import chem_copilot_autogen_main as chem_analyzer
    print("[API INFO] Successfully imported 'chem_copilot_autogen_main' as chem_analyzer.")
except ImportError as e:
    print(f"[API FATAL ERROR] Could not import chem_copilot_autogen_main: {e}")
    print(f"  Ensure 'chem_copilot_autogen_main.py' is in '{PROJECT_ROOT}' or on sys.path.")
    print(f"  Current sys.path: {sys.path}")
    sys.exit(1)
except Exception as e_init:
    print(f"[API FATAL ERROR] An unexpected error during import/initialization: {e_init}")
    traceback.print_exc()
    sys.exit(1)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Chem Copilot API",
    description="API for chemical analysis and chat using Autogen agents.",
    version="0.3.0" # Incremented version
)

# --- Pydantic Models ---
class GeneralQueryRequest(BaseModel): # Used by specific endpoints & new general endpoint
    query: str = Field(..., min_length=1, example="What are the functional groups in CCO?")
    original_name_for_saving: Optional[str] = Field(None, example="Ethanol_FG_Query")
    # Add a flag if frontend wants to explicitly clear MOI before this query
    clear_moi_before_query: Optional[bool] = Field(False, description="If true, MOI context is cleared before processing this query.")


class AnalysisResponse(BaseModel): # Used by specific endpoints & new general endpoint
    analysis: Optional[str] = "No analysis provided." # Made optional with default
    visualization_path: Optional[str] = None
    processed_smiles: Optional[str] = None
    error: Optional[str] = None
    query_context_for_filename: Optional[str] = None # For file saving context
    # Add MOI context to all responses so frontend can stay in sync
    current_moi_name: Optional[str] = None
    current_moi_smiles: Optional[str] = None

# Pydantic models for existing specific chat endpoint (can be deprecated if general endpoint is preferred)
class ChatRequest(BaseModel):
    query: Optional[str] = Field(None, min_length=1, example="What are its functional groups?")
    clear_history: Optional[bool] = Field(False)

class ChatResponse(BaseModel):
    reply: str
    visualization_path: Optional[str] = None
    error: Optional[str] = None
    current_moi_name: Optional[str] = None # Add MOI here too
    current_moi_smiles: Optional[str] = None

# --- FastAPI Event Handler for Startup ---
@app.on_event("startup")
async def startup_event():
    print("[API INFO] FastAPI application startup...")
    if not os.getenv("OPENAI_API_KEY"):
        print("[API STARTUP CRITICAL] OPENAI_API_KEY IS NOT SET.")
    else:
        print("[API STARTUP] OPENAI_API_KEY is set.")

    print("[API INFO] Initializing Autogen agents and tool wrappers...")
    try:
        chem_analyzer.get_tool_agents()
        chem_analyzer.get_chatbot_agents()
        chem_analyzer.get_tools_for_core_logic()
        print("[API INFO] Autogen components initialized.")
    except Exception as e:
        print(f"[API STARTUP ERROR] Failed to initialize Autogen components: {e}")

    # Mount static directory for frontend UI files
    frontend_static_dir = os.path.join(APP_DIR, "frontend_static")
    if os.path.isdir(frontend_static_dir):
        app.mount("/ui", StaticFiles(directory=frontend_static_dir, html=True), name="frontend") # html=True serves index.html on /ui/
        print(f"[API INFO] Mounted frontend static files from: {frontend_static_dir} at /ui")
    else:
        print(f"[API WARNING] Frontend static directory not found at '{frontend_static_dir}'. UI may not be accessible.")

    # Mount static directory for tool-generated visualizations
    tool_visualizations_dir = os.path.join(chem_analyzer.PROJECT_ROOT_DIR, "static")
    if os.path.isdir(tool_visualizations_dir):
        app.mount("/static", StaticFiles(directory=tool_visualizations_dir), name="tool_visualizations")
        print(f"[API INFO] Mounted tool visualization static files from: {tool_visualizations_dir} at /static")
    else:
        print(f"[API WARNING] Tool visualizations directory not found at '{tool_visualizations_dir}'.")


# --- Root Endpoint (Serves the HTML UI) ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False, tags=["Frontend UI"])
async def serve_frontend_ui():
    """Serves the main HTML frontend UI."""
    frontend_html_path = os.path.join(APP_DIR, "frontend_static", "index.html")
    if os.path.exists(frontend_html_path):
        return FileResponse(frontend_html_path)
    raise HTTPException(status_code=404, detail="Frontend HTML (index.html) not found in 'frontend_static' directory.")

# +++ NEW GENERAL COPILOT QUERY ENDPOINT +++
@app.post("/api/v1/copilot_query", response_model=AnalysisResponse, tags=["General Query"])
async def general_copilot_query_endpoint(request: GeneralQueryRequest):
    """
    A general endpoint to interact with ChemCopilot.
    It internally uses enhanced_query to route the request to the appropriate handler
    (full analysis, tool agent, or conversational chatbot with MOI context).
    """
    print(f"[API /api/v1/copilot_query] Query: '{request.query}', Clear MOI: {request.clear_moi_before_query}")

    if request.clear_moi_before_query:
        chem_analyzer.clear_chatbot_memory_autogen()
        print("[API /api/v1/copilot_query] MOI context cleared due to request flag.")

    try:
        result_dict = chem_analyzer.enhanced_query(
            full_query=request.query,
            original_compound_name=request.original_name_for_saving
        )

        # Ensure visualization_path is web-accessible
        vis_path = result_dict.get("visualization_path")
        if vis_path and os.path.exists(vis_path):
            abs_static_path = os.path.join(chem_analyzer.PROJECT_ROOT_DIR, "static")
            if vis_path.startswith(abs_static_path):
                 vis_path = vis_path.replace(abs_static_path, "/static", 1).replace("\\", "/")
            elif vis_path.startswith("static/"):
                 vis_path = "/" + vis_path.replace("\\", "/")
        else:
            vis_path = None # Ensure it's None if not valid

        return AnalysisResponse(
            analysis=result_dict.get("analysis", "No analysis content."),
            visualization_path=vis_path,
            processed_smiles=result_dict.get("processed_smiles_for_tools"),
            query_context_for_filename=result_dict.get("analysis_context"),
            error=result_dict.get("error"),
            current_moi_name=chem_analyzer._current_moi_context.get("name"),
            current_moi_smiles=chem_analyzer._current_moi_context.get("smiles")
        )
    except Exception as e:
        print(f"[API ERROR /api/v1/copilot_query] Exception: {e}")
        traceback.print_exc()
        moi_name_on_error = chem_analyzer._current_moi_context.get("name")
        moi_smiles_on_error = chem_analyzer._current_moi_context.get("smiles")
        # Return a structured error response
        return AnalysisResponse(
            analysis=f"An internal server error occurred: {str(e)}",
            error=str(e),
            current_moi_name=moi_name_on_error,
            current_moi_smiles=moi_smiles_on_error
        )


# --- Endpoint to Clear MOI Context Explicitly ---
@app.post("/api/v1/clear_moi_context", status_code=200, tags=["Context Management"])
async def clear_moi_context_endpoint():
    """Clears the current Molecule of Interest (MOI) context on the backend."""
    try:
        chem_analyzer.clear_chatbot_memory_autogen()
        return {"message": "MOI context cleared successfully."} # This is a JSON response
    except Exception as e:
        print(f"[API ERROR /api/v1/clear_moi_context] Exception: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error clearing MOI context: {str(e)}")

# --- Endpoint to Get Current MOI Context ---
@app.get("/api/v1/get_moi_context", response_model=Dict[str, Optional[str]], tags=["Context Management"])
async def get_current_moi_context_endpoint():
    """Gets the current Molecule of Interest (MOI) context from the backend."""
    try:
        return {
            "name": chem_analyzer._current_moi_context.get("name"),
            "smiles": chem_analyzer._current_moi_context.get("smiles")
        }
    except Exception as e:
        print(f"[API ERROR /api/v1/get_moi_context] Exception: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error retrieving MOI context: {str(e)}")


# --- Existing Specific Endpoints (can be kept for direct API use or deprecated over time) ---

@app.post("/analyze/full", response_model=AnalysisResponse, tags=["Specific Endpoints (Legacy/Direct)"])
async def analyze_full_information_endpoint(request: GeneralQueryRequest):
    print(f"[API /analyze/full] Query: '{request.query}', Original Name: '{request.original_name_for_saving}'")
    try:
        result_dict = chem_analyzer.enhanced_query(
            full_query=request.query,
            original_compound_name=request.original_name_for_saving
        )
        vis_path = result_dict.get("visualization_path")
        if vis_path and os.path.exists(vis_path): # Convert path
            abs_static_path = os.path.join(chem_analyzer.PROJECT_ROOT_DIR, "static")
            if vis_path.startswith(abs_static_path): vis_path = vis_path.replace(abs_static_path, "/static", 1).replace("\\", "/")
            elif vis_path.startswith("static/"): vis_path = "/" + vis_path.replace("\\", "/")
            else: vis_path = None
        else: vis_path = None

        return AnalysisResponse(
            analysis=result_dict.get("analysis", "No analysis content."),
            visualization_path=vis_path,
            processed_smiles=result_dict.get("processed_smiles_for_tools"),
            query_context_for_filename=result_dict.get("analysis_context"),
            current_moi_name=chem_analyzer._current_moi_context.get("name"),
            current_moi_smiles=chem_analyzer._current_moi_context.get("smiles")
        )
    except Exception as e:
        print(f"[API ERROR /analyze/full] Exception: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/query/tool_agent", response_model=AnalysisResponse, tags=["Specific Endpoints (Legacy/Direct)"])
async def query_with_tool_agent_endpoint(request: GeneralQueryRequest):
    print(f"[API /query/tool_agent] Query: '{request.query}'")
    try:
        result_dict = chem_analyzer.run_autogen_tool_agent_query(request.query)
        vis_path = result_dict.get("visualization_path") # Path from tool agent is already relative
        if vis_path and not vis_path.startswith("/"): vis_path = "/" + vis_path # Ensure it's web root relative

        return AnalysisResponse(
            analysis=result_dict.get("analysis", "No analysis content."),
            visualization_path=vis_path,
            processed_smiles=result_dict.get("processed_smiles_for_tools"),
            current_moi_name=chem_analyzer._current_moi_context.get("name"), # MOI might not be relevant here
            current_moi_smiles=chem_analyzer._current_moi_context.get("smiles")
        )
    except Exception as e:
        print(f"[API ERROR /query/tool_agent] Exception: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/chat/conversational", response_model=ChatResponse, tags=["Specific Endpoints (Legacy/Direct)"])
async def conversational_chat_endpoint(request: ChatRequest):
    print(f"[API /chat/conversational] Query: '{request.query}', Clear: {request.clear_history}")
    try:
        if request.clear_history:
            chem_analyzer.clear_chatbot_memory_autogen()
            return ChatResponse(
                reply="Chat history has been cleared.",
                current_moi_name=chem_analyzer._current_moi_context.get("name"), # Should be None
                current_moi_smiles=chem_analyzer._current_moi_context.get("smiles") # Should be None
                )

        if request.query is None:
            raise HTTPException(status_code=400, detail="Query is required if not clearing history.")

        result_dict = chem_analyzer.run_autogen_chatbot_query(request.query)
        vis_path = result_dict.get("visualization_path") # Path from chatbot agent
        if vis_path and not vis_path.startswith("/"): vis_path = "/" + vis_path

        return ChatResponse(
            reply=result_dict.get("analysis", "Chatbot did not provide a clear reply."),
            visualization_path=vis_path,
            error=result_dict.get("error"),
            current_moi_name=chem_analyzer._current_moi_context.get("name"),
            current_moi_smiles=chem_analyzer._current_moi_context.get("smiles")
        )
    except Exception as e:
        print(f"[API ERROR /chat/conversational] Exception: {e}")
        traceback.print_exc()
        return ChatResponse(
            reply="An internal server error occurred.", error=str(e),
            current_moi_name=chem_analyzer._current_moi_context.get("name"),
            current_moi_smiles=chem_analyzer._current_moi_context.get("smiles")
            )

# --- Main block to run Uvicorn ---
if __name__ == "__main__":
    print("Starting Chem Copilot API server with Uvicorn...")
    print(f"API docs (Swagger UI) available at: http://localhost:8000/docs")
    print(f"API docs (ReDoc) available at: http://localhost:8000/redoc")
    print(f"Frontend UI (if index.html is in frontend_static) available at: http://localhost:8000/")
    uvicorn.run("api_main:app", host="0.0.0.0", port=8000, reload=True)