
import os
from dotenv import load_dotenv

def load_config():
    load_dotenv()
    
    config = {
        "LANGCHAIN_PROJECT": os.getenv("LANGSMITH_PROJECT", "default-project"),
        "OPENAI_API_BASE": os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        "MODEL_NAME": os.getenv("MODEL_NAME", "gpt-4o-mini"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    }
    
    print("✅ Config loaded successfully")
    print(f"   LangSmith project : {config['LANGCHAIN_PROJECT']}")
    print(f"   OpenAI endpoint   : {config['OPENAI_API_BASE']}")
    print(f"   Default LLM model : {config['MODEL_NAME']}")
    print(f"   Embedding model   : {config['EMBEDDING_MODEL']}")
    
    return config

if __name__ == "__main__":
    load_config()
