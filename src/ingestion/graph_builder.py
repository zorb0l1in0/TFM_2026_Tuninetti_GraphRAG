"""
Graph Builder para crear grafos Neo4j desde documentos usando LangChain.
Inspirado en el paper Microsoft GraphRAG.
"""

import os
from typing import List, Tuple, Dict, Any
import pandas as pd
import json
import ast
from dotenv import load_dotenv

from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document

# 👇 IMPORTAR el extractor (NO importar LLMGraphTransformer directamente)
from src.extraction.entity_relation_extractor import EntityRelationExtractor

load_dotenv()


class GraphBuilder:
    """
    Constructor de grafos que USA el EntityRelationExtractor
    para obtener nodos y relaciones, y los guarda en Neo4j.
    """

    def __init__(
            self,
            model_name="gpt-4o-mini",
            batch_size=5,
            verbose=True
    ):
        """
        Inicializa el builder con un extractor interno.
        """
        self.batch_size = batch_size
        self.verbose = verbose

        # 1. Verificar API key (lo hará el extractor, pero verificamos antes)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY no encontrada en .env")

        # 2. 👇 CREAR EXTRACTOR INTERNO (él maneja el LLM y transformer)
        self.extractor = EntityRelationExtractor(
            model_name=model_name,
            verbose=verbose
        )

        # 3. Conexión a Neo4j (solo el builder guarda, el extractor no)
        self.graph = self._conectar_neo4j()

        if self.verbose:
            print(f"✅ GraphBuilder inicializado con extractor interno")
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
        Construye el grafo a partir de los documentos USANDO EL EXTRACTOR.
        """
        if not documentos:
            return (0, 0)

        print("\n🕸️ Construyendo grafo...")
        if limpiar:
            self.graph.query("MATCH (n) DETACH DELETE n")
            print("✅ Grafo limpiado")

        total_entidades = 0
        total_relaciones = 0

        for i in range(0, len(documentos), self.batch_size):
            lote = documentos[i:i + self.batch_size]
            entidades, relaciones = self._procesar_lote(lote)
            total_entidades += entidades
            total_relaciones += relaciones

        print(f"\n✅ Total: {total_entidades} entidades, {total_relaciones} relaciones")
        return (total_entidades, total_relaciones)

    def _procesar_lote(self, lote: List[Document]) -> Tuple[int, int]:
        """
        Procesa un lote USANDO EL EXTRACTOR.
        """
        try:
            # 👇 USAR EL EXTRACTOR (NO transformer directamente)
            graph_documents, entidades, relaciones = self.extractor.extract(
                documents=lote,
                batch_size=len(lote)  # Un solo lote
            )

            # Guardar en Neo4j (responsabilidad del builder)
            self.graph.add_graph_documents(
                graph_documents,
                baseEntityLabel=True,
                include_source=True
            )

            print(f"   📦 Lote: {entidades} ent, {relaciones} rel")
            return (entidades, relaciones)

        except Exception as e:
            print(f"   ❌ Error lote: {e}")
            return (0, 0)

    def asociar_embeddings(self, loader):
        """
        Asocia los embeddings del CSV a los nodos del grafo.
        """
        print("\n🔗 Asociando embeddings...")

        if not hasattr(loader, 'df') or 'embedding' not in loader.df.columns:
            print("❌ No hay columna 'embedding' en CSV")
            return 0

        total_nodos = 0
        for idx, row in loader.df.iterrows():
            chunk_id = row.get('id')
            embedding_val = row.get('embedding')

            if pd.isna(embedding_val) or not chunk_id:
                continue

            try:
                # Parsear embedding
                if isinstance(embedding_val, str):
                    try:
                        embedding = json.loads(embedding_val)
                    except:
                        try:
                            embedding = ast.literal_eval(embedding_val)
                        except:
                            clean = embedding_val.strip('[]')
                            embedding = [float(x.strip()) for x in clean.split(',') if x.strip()]
                else:
                    embedding = embedding_val

                # Actualizar nodos
                query = """
                MATCH (e:__Entity__)<-[:MENTIONED_IN]-(c:Chunk {id: $chunk_id})
                WHERE e.embedding IS NULL
                CALL db.create.setNodeVectorProperty(e, 'embedding', $embedding)
                RETURN count(e) as actualizados
                """
                result = self.graph.query(query, {
                    'chunk_id': str(chunk_id),
                    'embedding': embedding
                })
                if result:
                    total_nodos += result[0]['actualizados']
            except Exception as e:
                continue

        # Crear índice vectorial
        if total_nodos > 0:
            try:
                dim = self.graph.query(
                    "MATCH (e:__Entity__) WHERE e.embedding IS NOT NULL RETURN size(e.embedding) as dim LIMIT 1"
                )[0]['dim']
                self.graph.query(f"""
                CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
                FOR (n:__Entity__) ON (n.embedding)
                OPTIONS {{ indexConfig: {{ `vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine' }} }}
                """)
                print(f"✅ Índice vectorial creado (dim {dim})")
            except Exception as e:
                print(f"⚠️ Error creando índice: {e}")

        print(f"✅ {total_nodos} nodos actualizados")
        return total_nodos

    def consultar(self, query: str) -> List[Dict[str, Any]]:
        """Ejecuta una consulta Cypher"""
        return self.graph.query(query)