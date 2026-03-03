import os
from dotenv import load_dotenv
from pathlib import Path



# Directorio raíz del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
# Cargar variables de entorno
load_dotenv()
# Directorios de datos
DATA_DIR = PROJECT_ROOT / "data"

# Directorio específico para chunks
CHUNKS_DIR = DATA_DIR /"processed" / "chunks"

# Crear directorios si no existen
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("⚠️  ADVERTENCIA: OPENAI_API_KEY no está configurada en el archivo .env")

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Configuración del modelo
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# Parámetros del paper Microsoft GraphRAG
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
MAX_GLEANINGS = int(os.getenv("MAX_GLEANINGS", "2"))


# Función para verificar la configuración
def verificar_configuracion():
    """Verifica que la configuración sea válida"""
    print("🔧 Verificando configuración...")
    print(f"📁 Directorio chunks: {CHUNKS_DIR}")
    print(f"   - Existe: {CHUNKS_DIR.exists()}")
    print(f"📊 CHUNK_SIZE: {CHUNK_SIZE}")
    print(f"📊 MAX_GLEANINGS: {MAX_GLEANINGS}")

    if OPENAI_API_KEY:
        print(f"✅ OpenAI API Key configurada")
    else:
        print("❌ OpenAI API Key NO configurada")

    if NEO4J_PASSWORD:
        print(f"✅ Neo4j configurado en {NEO4J_URI}")
    else:
        print("❌ Neo4j password NO configurada")


# Ejecutar verificación si se ejecuta directamente
if __name__ == "__main__":
    verificar_configuracion()