# config/llm_client.py
import os

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from openai import OpenAI

load_dotenv()

BASE_URL = os.getenv("LLM_BASE_URL")
API_KEY  = os.getenv("PAID_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"

# ── Modelos por paso ──────────────────────────────────────────────────────────
#
# Paso 1 — detección de spans (tarea simple, modelo pequeño suficiente)
#   • qwen/qwen3-32b: potente, sin razonamiento forzado, buen seguimiento
#     de instrucciones JSON
#
# Paso 2 — clasificación + relaciones (tarea compleja, contexto largo)
#   • qwen/qwen3-32b: mejor opción local para seguir esquemas estrictos
#   • Alternativa si falla: deepseek/deepseek-r1-0528-qwen3-8b
#     (reasoning pero más manejable que R1-32B)
#
MODEL_PASO1 = "gpt-4.1-mini"  # "qwen/qwen3-32b"
MODEL_PASO2 = "gpt-4.1-mini"  # "qwen/qwen3-32b"

NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

_LLM_TIMEOUT = 600  # segundos — aplicado via request_timeout de LangChain


def get_langchain_llm(temperatura: float = 0.0) -> ChatOpenAI:
    """Compatibilidad con código existente — usa el modelo del Paso 2."""
    return get_langchain_llm_paso2(temperatura=temperatura)


def get_langchain_llm_paso1(temperatura: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_PASO1,
        temperature=temperatura,
        #max_tokens=10_000,
        request_timeout=_LLM_TIMEOUT,
        openai_api_key=API_KEY,
        #openai_api_base=BASE_URL,
    )


def get_langchain_llm_paso2(temperatura: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_PASO2,
        temperature=temperatura,
        #max_tokens=4096,
        request_timeout=_LLM_TIMEOUT,
        openai_api_key=API_KEY,
        #openai_api_base=BASE_URL,
    )


def get_neo4j_graph() -> Neo4jGraph:
    return Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
    )


def get_embeddings_client() -> OpenAI:
    return OpenAI(api_key=API_KEY)


def get_embeddings(texts: list[str]) -> list[list[float]]:
    client = get_embeddings_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        encoding_format="float",
    )
    return [item.embedding for item in response.data]