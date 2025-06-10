# ChemCopilot - Chemistry Assistant

ChemCopilot is an advanced chemistry assistant designed for analyzing chemical reactions and performing retrosynthesis analysis. This tool provides chemists with comprehensive information about reactions, including bond changes, functional group transformations, visualization, and more.

## 📋 Prerequisites

- Python 3.11
- Conda (recommended for environment management)
- OpenAI API key
- Perplexity API Key
- Git

## ⚙️ Installation

### Step 1: Clone the ChemCopilot Repository
```bash
git clone https://github.com/yourusername/chemcopilot.git
cd chemcopilot
```

### Step 2: Create and Activate a Conda Environment

This command will create an environment named autogen_chem_copilot (as per your YAML file) with Python 3.11 and install all specified dependencies.

```bash
conda env create -f autogen_chem_env.yml
conda activate autogen_chem_copilot
```

If you prefer a different environment name, you can change the name: field in autogen_chem_env.yml before running the command, or create an environment manually and then install:

```bash
# Alternative manual creation (if not using YAML directly for env creation)
# conda create -n chemcopilot_env python=3.11
# conda activate chemcopilot_env
# conda install --file autogen_chem_env.yml # This might not work directly, YAML is usually for `env create`
# If the above conda install --file doesn't work, after creating and activating the env,
# ensure pip is available in the conda env and then use pip for requirements if you generate a requirements.txt from the yml.
# OR, more simply, ensure the YAML installs everything correctly with `conda env create`.
```

### Step 3: Configure Environment Variables
Create a .env file in the main chemcopilot directory with the following content. Replace placeholder values with your actual keys and paths.

```bash
# --- API KEYS ---

API_KEY="sk-proj-YOUR_OPENAI_API_KEY_HERE"

# Required for core LLM functionalities if DEFAULT_LLM_PROVIDER is openai or as fallback
OPENAI_API_KEY="sk-proj-YOUR_OPENAI_API_KEY_HERE"

# Optional: For using Perplexity as an LLM provider
PERPLEXITY_API_KEY="pplx-YOUR_PERPLEXITY_API_KEY_HERE"

# --- DEFAULT LLM PROVIDER ---
# Set to "openai" or "perplexity". If perplexity is chosen and key is missing, will try to fallback to openai.
DEFAULT_LLM_PROVIDER="openai"

# --- DATASET PATHS ---
# Ensure these paths are correct for your system
REACTION_DATASET_PATH1="/path/to/your/orderly_retro.parquet"
REACTION_DATASET_PATH2="/path/to/your/reaction_classification_checkpoint.parquet"

# --- Other Optional Environment Variables (if any) ---
# EXAMPLE_VARIABLE="example_value"
```

Note:
The API_KEY variable you had at the top of your example .env content (starting with sk-proj-JO-W...) seems to be an OpenAI key. I've used OPENAI_API_KEY as the standard variable name in the template above, as used in your Python script. Adjust if your script uses API_KEY directly for OpenAI.
For dataset paths, replace /path/to/your/... with the actual absolute paths to your .parquet files on your system.


### Step 4: (If applicable) Install any remaining packages

If autogen_chem_env.yml doesn't cover all dependencies (e.g., if you added new ones directly to requirements.txt that aren't in the YAML, or if some packages are better installed via pip within the conda environment):

```bash 
# (Activate your conda environment first if not already active)
# conda activate autogen_chem_copilot
pip install -r requirements.txt # If you have a separate requirements.txt
```

However, it's best practice to keep autogen_chem_env.yml as the single source of truth for dependencies.

## 🚀 Running the Application

Currently, ChemCopilot is run as an interactive command-line script.

1. Activate your Conda environment:
```bash
conda activate autogen_chem_copilot
```

2. Navigate to the project directory:
```bash 
cd path/to/your/chemcopilot
```

3. Start the API server using Uvicorn:
```bash 
uvicorn api_main:app --reload --port 8000
```

- This assumes your FastAPI application instance is named app within the api_main.py file.
- --reload enables auto-reloading for development. Remove it for production.
- The API will typically be available at http://localhost:8000. You can access API documentation (e.g., Swagger UI) at http://localhost:8000/docs or http://localhost:8000/redoc if you're using FastAPI defaults.

4. User Interface (UI):

- If you have a separate frontend application (e.g., built with React, Vue, Angular, or another Streamlit app that consumes the API), run that application according to its own instructions.
- The UI will then make requests to the ChemCopilot API running at http://localhost:8000.

## 📊 Core Functionalities & Tools

ChemCopilot leverages a ChemicalAnalysisAgent and integrates several underlying capabilities, often exposed through Autogen agent tools, made available via API endpoints:

### FuncGroups
- Identifies functional groups in molecules or reactions
- Analyzes functional group transformations

### NameToSMILES
- Converts chemical names (IUPAC or common) to SMILES notation

### SMILES2Name
- Converts SMILES strings to chemical names
- Provides both IUPAC and common names when available

### BondChangeAnalyzer
- Identifies bonds broken, formed, and changed in reactions
- Explains bond transformation patterns

### ChemVisualizer
- Generates visual representations of molecules and reactions
- Creates clear reaction diagrams showing transformations

### ReactionClassifier
- Classifies reaction types based on reaction SMILES
- Provides educational information about reaction mechanisms

### Follow-up Questions
1. After viewing a reaction analysis, use the question box
2. Ask specific questions about the reaction
3. Get tailored answers about mechanisms, applications, etc.

## 📝 Important Notes

- Ensure your API keys in the .env file are correct and have necessary permissions/billing enabled.
- Verify that the paths to your .parquet dataset files in the .env file are accurate for the server environment.
- The quality of LLM-generated information can vary. Always critically evaluate the results.
- The autogen_chem_env.yml file is the primary source for Python dependencies for the backend.

<!--
Note: To properly display the diagrams, please save the following images:
1. Save the system architecture diagram as "images/system_architecture.png"
2. Save the workflow diagram as "images/workflow.png"
-->
