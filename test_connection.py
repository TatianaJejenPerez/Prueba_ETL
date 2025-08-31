from dotenv import load_dotenv, find_dotenv
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import os, sys

# 1) Cargar .env
env_path = find_dotenv()
print("🔎 .env encontrado en:", env_path if env_path else "NO ENCONTRADO")
load_dotenv(env_path, override=True)

# 2) Leer SUPABASE_URI directo
supa_uri = os.getenv("SUPABASE_URI")

# Debug: ver qué llegó (repr muestra espacios ocultos)
print("SUPABASE_URI =", repr(supa_uri))

# 3) Si no hay SUPABASE_URI, intentar construirlo desde partes
if not supa_uri:
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "postgres")
    user = os.getenv("DB_USER", "postgres")
    pwd  = os.getenv("DB_PASSWORD")
    if all([host, user, pwd]):
        supa_uri = f"postgresql+psycopg2://{user}:{quote_plus(pwd)}@{host}:{port}/{name}?sslmode=require"
        print("⚙️  SUPABASE_URI construido:", supa_uri)
    else:
        print("❌ Falta SUPABASE_URI y/o variables DB_* para construirlo.")
        sys.exit(1)

# 4) Probar conexión
try:
    engine = create_engine(supa_uri)
    with engine.connect() as conn:
        row = conn.execute(text("SELECT current_database(), current_user;")).fetchone()
        print("✅ Conexión OK:", row)
except Exception as e:
    print("❌ Error de conexión:", e)
    sys.exit(2)
