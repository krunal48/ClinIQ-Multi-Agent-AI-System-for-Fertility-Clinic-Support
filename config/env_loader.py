import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True), override=False)

def get(key: str, default=None):
    return os.getenv(key, default)
