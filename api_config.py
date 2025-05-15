import os
from dotenv import load_dotenv

def setup_api_key():
    """Set up the API key globally for all OpenAI clients.
    
    This function should be called at the start of your application.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API key from environment
    api_key = os.getenv("API_KEY")
    
    if not api_key:
        raise ValueError("API_KEY environment variable is not set. Please set it in your .env file or environment.")
    
    # Set API key for OpenAI in the environment
    os.environ["OPENAI_API_KEY"] = api_key
    
    return api_key

# Set the API key when this module is imported
api_key = setup_api_key()