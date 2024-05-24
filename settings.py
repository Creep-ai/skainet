from os import getenv

from dotenv import load_dotenv

load_dotenv()

DB_USER = getenv("DB_USER", "postgres")
DB_PASSWORD = getenv("DB_PASSWORD", "postgres")
DB_HOST = getenv("DB_HOST", "localhost")
DB_PORT = getenv("DB_PORT", 5432)
DB_NAME = getenv("DB_NAME", "dvdrental")
GPT_SQL_MODEL = getenv("GPT_SQL_MODEL", "gpt-4o-2024-05-13")
GPT_PYTHON_MODEL = getenv("GPT_PYTHON_MODEL", "gpt-4-turbo")
