"""
Graph Builder para crear grafos Neo4j desde documentos usando LangChain.
Inspirado en el paper Microsoft GraphRAG.
"""

import os
from typing import List, Tuple, Optional, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document

# Carga variables de entorno
load_dotenv()


class GraphBuilder:
    """
    Constructor de grafos que transforma documentos en nodos y relaciones.
    Los documentos se pasan al método construir(), no al constructor.
    """

    def __init__(
            self,
            model_name: str = "gpt-4o-mini",
            temperature: float = 0,
            batch_size: int = 10,
            verbose: bool = True
    ):
        """
        Inicializa el builder (sin documentos).

        Args:
            model_name: Modelo OpenAI (gpt-4o-mini es económico)
            temperature: 0 = determinista
            batch_size: Documentos por lote
            verbose: Muestra información detallada
        """
        self.batch_size = batch_size
        self.verbose = verbose

        # 1. Verificar API key de OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY no encontrada en .env")

        # 2. Inicializar LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )

        # 3. Transformador de grafos (con descripciones, como en el paper)
        self.transformer = LLMGraphTransformer(
            llm=self.llm,
            node_properties=["description"],
            relationship_properties=["description"]
        )

        # 4. Conexión a Neo4j
        self.graph = self._conectar_neo4j()

        if self.verbose:
            print(f"✅ GraphBuilder inicializado")
            print(f"   Modelo: {model_name}")
            print(f"   Batch: {batch_size}")

    def _conectar_neo4j(self) -> Neo4jGraph:
        """Conecta a Neo4j"""
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")

        if not password:
            raise ValueError("❌ NEO4J_PASSWORD no encontrada en .env")

        try:
            graph = Neo4jGraph(
                url=uri,
                username=username,
                password=password
            )
            if self.verbose:
                print(f"✅ Conectado a Neo4j: {uri}")
            return graph
        except Exception as e:
            raise ConnectionError(f"❌ Error conectando a Neo4j: {e}")

    def construir(self, documentos: List[Document], limpiar: bool = True) -> Tuple[int, int]:
        """
        Construye el grafo a partir de los documentos.

        Args:
            documentos: Lista de documentos LangChain (del loader)
            limpiar: Si True, borra el grafo existente

        Returns:
            (total_entidades, total_relaciones) creadas
        """
        if not documentos:
            print("⚠️ No hay documentos para procesar")
            return (0, 0)

        print("\n" + "=" * 60)
        print("🕸️  CONSTRUYENDO GRAFO")
        print("=" * 60)
        print(f"📄 Documentos recibidos: {len(documentos)}")

        if limpiar:
            self._limpiar_grafo()

        total_entidades = 0
        total_relaciones = 0

        # Procesar por lotes
        for i in range(0, len(documentos), self.batch_size):
            lote = documentos[i:i + self.batch_size]
            entidades, relaciones = self._procesar_lote(lote, i)

            total_entidades += entidades
            total_relaciones += relaciones

        # Estadísticas finales
        print("\n" + "=" * 60)
        print("✅ GRAFO COMPLETADO")
        print("=" * 60)
        print(f"📊 Total entidades: {total_entidades}")
        print(f"🔗 Total relaciones: {total_relaciones}")

        self._mostrar_estadisticas()

        return (total_entidades, total_relaciones)

    def _procesar_lote(self, lote: List[Document], indice_inicio: int) -> Tuple[int, int]:
        """Procesa un lote de documentos"""
        print(f"\n📦 Lote {indice_inicio // self.batch_size + 1} ({len(lote)} docs)...")

        try:
            # Extraer grafo del lote
            graph_documents = self.transformer.convert_to_graph_documents(lote)

            # Añadir a Neo4j
            self.graph.add_graph_documents(
                graph_documents,
                baseEntityLabel=True,
                include_source=True
            )

            # Contar
            entidades = sum(len(gd.nodes) for gd in graph_documents)
            relaciones = sum(len(gd.relationships) for gd in graph_documents)

            print(f"   ✅ Lote procesado: {entidades} entidades, {relaciones} relaciones")

            return (entidades, relaciones)

        except Exception as e:
            print(f"   ❌ Error en lote: {e}")
            return (0, 0)

    def _limpiar_grafo(self):
        """Elimina todos los nodos y relaciones del grafo"""
        if self.verbose:
            print("\n🧹 Limpiando grafo existente...")

        query = "MATCH (n) DETACH DELETE n"
        self.graph.query(query)

        if self.verbose:
            print("✅ Grafo limpiado")

    def _mostrar_estadisticas(self):
        """Muestra estadísticas del grafo actual"""
        queries = {
            "total_nodos": "MATCH (n) RETURN count(n) as count",
            "total_relaciones": "MATCH ()-[r]->() RETURN count(r) as count",
        }

        print("\n📊 ESTADÍSTICAS DEL GRAFO:")

        try:
            # Total nodos
            result = self.graph.query(queries["total_nodos"])
            if result:
                print(f"   • Nodos totales: {result[0]['count']}")

            # Total relaciones
            result = self.graph.query(queries["total_relaciones"])
            if result:
                print(f"   • Relaciones totales: {result[0]['count']}")

        except Exception as e:
            print(f"   ⚠️ No se pudieron obtener estadísticas: {e}")

    def consultar(self, query: str) -> List[Dict[str, Any]]:
        """Ejecuta una consulta Cypher en Neo4j"""
        return self.graph.query(query)