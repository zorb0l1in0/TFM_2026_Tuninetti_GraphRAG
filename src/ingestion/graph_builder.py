"""
Graph Builder para crear grafos Neo4j desde documentos usando LangChain.
Inspirado en el paper Microsoft GraphRAG.
"""

import os
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import json
import ast
from dotenv import load_dotenv

from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from src.extraction.entity_relation_extractor import EntityRelationExtractor
from src.extraction.entity_relation_labels_extractor import HybridDocumentAnalyzer
load_dotenv()


class GraphBuilder:
    """
    Constructor de grafos que USA el EntityRelationExtractor
    para obtener nodos y relaciones, y los guarda en Neo4j.

    Ahora soporta descubrimiento automático de vocabulario con HybridDocumentAnalyzer.
    """

    def __init__(
            self,
            model_name="gpt-4o-mini",
            batch_size=5,
            verbose=True,
            auto_discover_vocabulary: bool = False,
            vocabulary_document: Optional[Path] = None,
            allowed_nodes: Optional[List[str]] = None,
            allowed_relationships: Optional[List[str]] = None,
            top_n: int = 15,
            threshold_percentual: float = 0.05,  # 👈 5% di default
            min_frecuencia: int = 2
    ):
        """
        Inicializa el builder con un extractor interno.

        Args:
            auto_discover_vocabulary: si True, analiza el documento para descubrir nodos y relaciones
            vocabulary_document: documento específico para extraer vocabulario (opcional)
            allowed_nodes: lista manual de nodos permitidos (opcional)
            allowed_relationships: lista manual de relaciones permitidas (opcional)
            top_n: para el analizador híbrido
            min_frecuencia: para el analizador híbrido
        """
        self.batch_size = batch_size
        self.verbose = verbose
        self.auto_discover_vocabulary = auto_discover_vocabulary
        self.vocabulary_document = vocabulary_document
        self.top_n = top_n
        self.min_frecuencia = min_frecuencia
        self.threshold_percentual = threshold_percentual

        # 1. Verificar API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY no encontrada en .env")

        # 2. Preparar vocabulario (si es necesario)
        final_allowed_nodes = allowed_nodes
        final_allowed_relationships = allowed_relationships

        # Si hay descubrimiento automático, analizar el documento
        if auto_discover_vocabulary:
            if vocabulary_document:
                # Usar documento específico para vocabulario
                final_allowed_nodes, final_allowed_relationships = self._descubre_vocabulario(vocabulary_document)
            else:
                # Descubrimiento diferido: se hará cuando se llame a construir()
                # con los primeros documentos
                if self.verbose:
                    print("🔍 Modo descubrimiento automático: se analizarán los primeros documentos")
        elif not allowed_nodes and not allowed_relationships:
            if self.verbose:
                print("⚠️ Sin vocabulario definido. El extractor usará todas las entidades posibles.")

        # 3. 👇 CREAR EXTRACTOR INTERNO con el vocabulario (si lo tenemos)
        self.extractor = EntityRelationExtractor(
            model_name=model_name,
            verbose=verbose,
            allowed_nodes=final_allowed_nodes,
            allowed_relationships=final_allowed_relationships,
            auto_discover=(auto_discover_vocabulary and not vocabulary_document),
            threshold_percentual=threshold_percentual,  # 👈 PASSA QUI
            min_frecuencia=min_frecuencia,
            top_n=top_n
        )

        # 4. Conexión a Neo4j
        self.graph = self._conectar_neo4j()

        # 5. Analyzer para descubrimiento (si se necesita)
        if auto_discover_vocabulary:
            self.analyzer = HybridDocumentAnalyzer(
                modelo_spacy="es_core_news_lg",
                patron_tabla=["Fila de tabla", "Tabla"]
            )
        else:
            self.analyzer = None

        if self.verbose:
            print(f"✅ GraphBuilder inicializado con extractor interno")
            print(f"   Modelo: {model_name}")
            print(f"   Batch: {batch_size}")
            if final_allowed_nodes:
                print(f"   📋 Nodos permitidos: {len(final_allowed_nodes)}")
            if final_allowed_relationships:
                print(f"   🔗 Relaciones permitidas: {len(final_allowed_relationships)}")

    def _descubre_vocabulario(self, documento: Path) -> Tuple[List[str], List[str]]:
        """
        Descubre nodos y relaciones usando HybridDocumentAnalyzer.

        Args:
            documento: Path al documento para analizar

        Returns:
            (allowed_nodes, allowed_relationships)
        """
        if self.verbose:
            print(f"\n🔍 Descubriendo vocabulario desde: {documento.name}")

        if not self.analyzer:
            self.analyzer = HybridDocumentAnalyzer(
                modelo_spacy="es_core_news_lg",
                patron_tabla=["Fila de tabla", "Tabla"]
            )

        resultado = self.analyzer.analiza(
            documento,
            top_n=self.top_n,
            min_frecuencia=self.min_frecuencia,
            verbose=self.verbose
        )

        nodes = resultado['allowed_nodes']
        relations = resultado['allowed_relationships']

        if self.verbose:
            print(f"\n✅ Vocabulario descubierto:")
            print(f"   📋 Nodos ({len(nodes)}): {nodes}")
            print(f"   🔗 Relaciones ({len(relations)}): {relations}")

        return nodes, relations

    def _prepara_vocabulario_desde_documentos(self, documentos: List[Document]):
        """
        Prepara vocabulario analizando los primeros documentos.
        Útil cuando no se especifica un documento de vocabulario.
        """
        if not documentos:
            return

        if self.verbose:
            print("\n🔍 Analizando documentos para descubrir vocabulario...")

        # Crear archivo temporal con el contenido de los primeros documentos
        temp_content = "\n\n".join([doc.page_content[:2000] for doc in documentos[:3]])
        temp_path = Path("temp_vocab.txt")
        temp_path.write_text(temp_content, encoding="utf-8")

        # Descubrir vocabulario
        nodes, relations = self._descubre_vocabulario(temp_path)

        # Limpiar
        temp_path.unlink()

        # Actualizar extractor
        self.extractor.allowed_nodes = nodes
        self.extractor.allowed_relationships = relations

        # Reinicializar transformer con nuevo vocabulario
        self.extractor.transformer = LLMGraphTransformer(
            llm=self.extractor.llm,
            node_properties=["description"],
            relationship_properties=["description"],
            allowed_nodes=nodes,
            allowed_relationships=relations
        )

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

        Si está en modo auto_discover_vocabulary y no hay vocabulario,
        lo descubre de los primeros documentos.
        """
        if not documentos:
            return (0, 0)

        # 👇 NUEVO: Si estamos en modo descubrimiento pero no tenemos vocabulario aún
        if self.auto_discover_vocabulary and not self.vocabulary_document:
            if not self.extractor.allowed_nodes:
                self._prepara_vocabulario_desde_documentos(documentos)

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
            # Usar el extractor
            graph_documents, entidades, relaciones = self.extractor.extract(
                documents=lote,
                batch_size=len(lote)  # Un solo lote
            )

            # Guardar en Neo4j
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

    def construir_con_vocabulario(
            self,
            documentos: List[Document],
            documento_vocabulario: Path,
            limpiar: bool = True
    ) -> Tuple[int, int]:
        """
        Construye el grafo usando un documento específico para el vocabulario.

        Args:
            documentos: documentos a procesar
            documento_vocabulario: documento del que extraer nodos y relaciones
            limpiar: si debe limpiar el grafo antes

        Returns:
            (total_entidades, total_relaciones)
        """
        # Descubrir vocabulario
        nodes, relations = self._descubre_vocabulario(documento_vocabulario)

        # Actualizar extractor
        self.extractor.allowed_nodes = nodes
        self.extractor.allowed_relationships = relations

        # Reinicializar transformer
        from langchain_experimental.graph_transformers import LLMGraphTransformer
        self.extractor.transformer = LLMGraphTransformer(
            llm=self.extractor.llm,
            node_properties=["description"],
            relationship_properties=["description"],
            allowed_nodes=nodes,
            allowed_relationships=relations
        )

        # Construir grafo
        return self.construir(documentos, limpiar)

    def asociar_embeddings(self, loader):
        """Asocia los embeddings del CSV a los nodos del grafo."""
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