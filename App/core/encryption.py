from cryptography.fernet import Fernet
import os
from dotenv import load_dotenv
load_dotenv()

ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
if not ENCRYPTION_KEY:
    raise ValueError("ENCRYPTION_KEY is missing")

fernet = Fernet(ENCRYPTION_KEY.encode())
def encrypt_value(value) -> str:
    if value is None:
        return None
    return fernet.encrypt(str(value).encode()).decode()

def decrypt_value(encrypted_value: str) -> str:
    if encrypted_value is None:
        return None
    return fernet.decrypt(encrypted_value.encode()).decode()