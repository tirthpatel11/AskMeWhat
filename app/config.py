import os
from dotenv import load_dotenv

def load_config():
    return load_dotenv()

def get_groq_api():
    return "api_key"

groq_api_key = get_groq_api()

if groq_api_key:
    print(f"Groq API Key is: {groq_api_key}")
else:
    print("Groq API Key is not set. Please provide it in the code.")
