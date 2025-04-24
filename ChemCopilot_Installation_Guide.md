# ChemCopilot Installation Guide

This guide provides detailed instructions for installing and setting up ChemCopilot on your system.

## Prerequisites

Before installing ChemCopilot, ensure you have the following prerequisites:

- **Python**: Version 3.9 or higher
- **Git**: For cloning the repository
- **OpenAI API Key**: Required for LLM-based analysis
- **Sci-Hub Access**: For literature retrieval (optional but recommended)
- **Sufficient Disk Space**: At least 2GB for the application and dependencies

## Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/chem_copilot.git
cd chem_copilot
```

## Step 2: Set Up a Virtual Environment

It's recommended to use a virtual environment to avoid conflicts with other Python packages.

### For macOS/Linux:

```bash
python -m venv env
source env/bin/activate
```

### For Windows:

```bash
python -m venv env
env\Scripts\activate
```

## Step 3: Install Dependencies

Install all required packages using pip:

```bash
pip install -r requirements.txt
```

If the requirements.txt file is not available, you can install the core dependencies manually:

```bash
pip install openai fastapi uvicorn streamlit rdkit-pypi pymupdf langchain langchain_openai scholarly python-dotenv graphviz pydantic pubchempy tqdm pillow requests
```

## Step 4: Configure Environment Variables

Create a `.env` file in the project root directory with the following variables:

```
# OpenAI API Configuration
API_KEY=your_openai_api_key
BASE_URL=https://api.openai.com/v1  # Optional if using OpenAI directly

# Sci-Hub Access (for PDF downloading)
HEADERS={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
COOKIES={"session": "your_session_cookie_if_needed"}
```

Replace `your_openai_api_key` with your actual OpenAI API key.

## Step 5: Create Required Directories

Ensure the necessary directories exist for storing PDFs and results:

```bash
mkdir -p pdf_pi res_pi tree_pi
```

## Step 6: Verify Installation

### Test RetroSynthesisAgent:

```bash
cd RetroSynthesisAgent
python -c "from RetroSynAgent.GPTAPI import GPTAPI; api = GPTAPI(); print('API Connection Successful')"
```

If successful, you should see "API Connection Successful" printed to the console.

### Test Features Module:

```bash
cd Features
python -c "from tools.name2smiles import NameToSMILES; tool = NameToSMILES(); print('Features Module Loaded')"
```

If successful, you should see "Features Module Loaded" printed to the console.

## Step 7: Start the Services

### Start the RetroSynthesisAgent API:

```bash
cd RetroSynthesisAgent
uvicorn api:app --reload --port 8000
```

The API will be available at http://localhost:8000

### Start the Web Interface:

```bash
cd Features
streamlit run app.py
```

The web interface will be available at http://localhost:8501

## Troubleshooting

### Common Installation Issues

1. **SSL Certificate Verification Errors**:
   
   If you encounter SSL certificate verification errors when downloading PDFs or accessing APIs, you can temporarily disable SSL verification by uncommenting the SSL bypass block at the top of `main.py`. Note that this is not recommended for production environments.

2. **Missing Dependencies**:
   
   If you encounter "ModuleNotFoundError", install the missing package:
   ```bash
   pip install <package_name>
   ```

3. **OpenAI API Authentication Errors**:
   
   Verify your API key is correct and has sufficient permissions. Check your billing status if you're getting rate limit errors.

4. **PDF Download Issues**:
   
   If PDF downloads fail, check your internet connection and Sci-Hub access. You may need to update the headers and cookies in your `.env` file.

5. **Graphviz Installation**:
   
   If you encounter issues with tree visualization, ensure Graphviz is installed on your system:
   
   - **Ubuntu/Debian**: `sudo apt-get install graphviz`
   - **macOS**: `brew install graphviz`
   - **Windows**: Download and install from the [Graphviz website](https://graphviz.org/download/)

## System Requirements

- **CPU**: Multi-core processor recommended
- **RAM**: Minimum 8GB, 16GB recommended
- **Disk Space**: 2GB for application and dependencies, plus additional space for downloaded PDFs
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+ recommended)
- **Internet Connection**: Required for API access and PDF downloads

## Next Steps

After successful installation, refer to the main documentation for:

- Detailed usage instructions
- API documentation
- Advanced configuration options
- Troubleshooting specific functionality issues
