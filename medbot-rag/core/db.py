import os
from typing import Optional
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi

load_dotenv()

_CLIENT: Optional[MongoClient] = None

def get_client(uri: Optional[str] = None, timeout_ms: int = 10_000) -> MongoClient:
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    uri = uri or os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError(
            "Missing MONGODB_URI. Add it to your .env, e.g.\n"
            "MONGODB_URI='mongodb+srv://<user>:<pass>@<cluster>/<db>?retryWrites=true&w=majority&appName=Embedding'"
        )

    _CLIENT = MongoClient(
        uri,
        server_api=ServerApi("1"),
        serverSelectionTimeoutMS=timeout_ms,
        connectTimeoutMS=timeout_ms,
        socketTimeoutMS=timeout_ms,
    )

    _CLIENT.admin.command("ping")
    return _CLIENT

def get_db(name: str):
    return get_client()[name]

def get_collection(db_name: str, col_name: str):
    return get_client()[db_name][col_name]

def close_client() -> None:
    global _CLIENT
    if _CLIENT is not None:
        _CLIENT.close()
        _CLIENT = None