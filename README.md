# Chem Copilot

Chem CoPilot is an AI-powered chemistry assistant designed to help researchers, students, and professionals extract, analyze, and understand chemical information from various sources, including patents and structured datasets.

## Tools & Libraries Used

The following libraries are used in this project:

* **LangChain:** For building applications with LLMs (Large Language Models).
* **OpenAI:** API for GPT-based models.
* **RDKit:** Python library for cheminformatics.
* **RXNMapper:** For mapping chemical reactions.
* **Streamlit:** For building the app's user interface.
* **Pydantic:** Data validation and settings management.
* **Requests:** For making HTTP requests.
* **Python-dotenv:** For loading environment variables from `.env`.

## Installation Guide

### Prerequisites:

* **Python version:** 3.9.18
* **Conda:** To install RDKit.

### Step-by-Step Setup:

1.  **Create a Virtual Environment:**
    To ensure a clean environment for your project, it's recommended to use a virtual environment. You can do so with the following commands:

    ```bash
    python3.9 -m venv chem_env
    source chem_env/bin/activate   # On macOS/Linux
    .\chem_env\Scripts\activate    # On Windows
    ```

2.  **Install Dependencies:**
    Once the virtual environment is set up, install the dependencies listed in the `requirements.txt` file using pip:

    ```bash
    pip install -r requirements.txt
    ```

    For RDKit, which is installed via conda, run:

    ```bash
    conda install -c conda-forge rdkit=2022.9.5
    ```

3.  **Set Up OpenAI API Key:**
    Create a `.env` file in the root of the project and add the OpenAI API key as follows:

    ```ini
    OPENAI_API_KEY=your-openai-api-key-here
    ```

4.  **Running the Application:**
    To run the app, use the following Streamlit command:

    ```bash
    python -m streamlit run app.py
    ```

    This will start the Streamlit server, and you can view the application by navigating to the provided local URL (typically `http://localhost:8501`).
