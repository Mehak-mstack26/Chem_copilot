import os
from dotenv import load_dotenv

def setup_api_key():
    load_dotenv()
    api_key = os.getenv("API_KEY") 
    if not api_key:
        raise ValueError("API_KEY ...")
    os.environ["OPENAI_API_KEY"] = api_key 
    return api_key

api_key = setup_api_key() 