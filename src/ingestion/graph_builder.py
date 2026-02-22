"""
Graph Builder para crear grafos Neo4j desde documentos usando LangChain.
Inspirado en el paper Microsoft GraphRAG.
"""

import os
from typing import List, Tuple, Optional, Dict, Any

import httpx
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
            model_name="gpt-4o-mini",  # Este funcionará seguro
            batch_size=5,
            verbose=True
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
            api_key=os.getenv("OPENAI_API_KEY"),
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

    def asociar_embeddings(self, loader):
        """
        Asocia los embeddings del CSV a los nodos del grafo.

        Args:
            loader: Instancia de CSVLoader con los datos originales

        Returns:
            Número de nodos actualizados
        """
        print("\n🔗 ASOCIANDO EMBEDDINGS A NODOS")
        print("=" * 60)

        # 1. Verificar que hay embeddings en el loader
        if not hasattr(loader, 'df') or 'embedding' not in loader.df.columns:
            print("❌ No se encontró columna 'embedding' en el CSV")
            return 0

        import json
        import ast

        total_nodos = 0
        chunks_con_embedding = 0
        errores = 0

        # 2. Iterar sobre los chunks que tienen embedding
        for idx, row in loader.df.iterrows():
            chunk_id = row.get('id')
            embedding_val = row.get('embedding')

            if pd.isna(embedding_val) or not chunk_id:
                continue

            try:
                # 3. Parsear embedding (de string a lista)
                if isinstance(embedding_val, str):
                    embedding_str = embedding_val.strip()

                    # Intentar diferentes formatos
                    try:
                        embedding = json.loads(embedding_str)
                    except:
                        try:
                            embedding = ast.literal_eval(embedding_str)
                        except:
                            # Formato simple: números separados por comas
                            clean = embedding_str.strip('[]')
                            embedding = [float(x.strip()) for x in clean.split(',') if x.strip()]
                else:
                    embedding = embedding_val

                # 4. Verificar que es una lista válida
                if not isinstance(embedding, list):
                    errores += 1
                    continue

                # 5. Actualizar nodos asociados a este chunk
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

                if result and result[0]['actualizados'] > 0:
                    total_nodos += result[0]['actualizados']
                    chunks_con_embedding += 1

                if idx % 20 == 0 and idx > 0:
                    print(f"   ✅ Procesados {idx} chunks, {total_nodos} nodos actualizados")

            except Exception as e:
                errores += 1
                if errores <= 5:
                    print(f"   ⚠️ Error chunk {chunk_id}: {e}")

        # 6. Crear índice vectorial
        if total_nodos > 0:
            try:
                # Obtener dimensión del embedding
                dim_query = """
                MATCH (e:__Entity__) 
                WHERE e.embedding IS NOT NULL 
                RETURN size(e.embedding) as dim 
                LIMIT 1
                """
                result = self.graph.query(dim_query)
                if result:
                    dim = result[0]['dim']

                    # Crear índice
                    self.graph.query(f"""
                    CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
                    FOR (n:__Entity__) ON (n.embedding)
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {dim},
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                    """)
                    print(f"\n📊 Índice vectorial creado (dimensión {dim})")
            except Exception as e:
                print(f"⚠️ Error creando índice: {e}")

        print("\n" + "=" * 60)
        print(f"✅ RESULTADOS:")
        print(f"   • Chunks con embedding: {chunks_con_embedding}")
        print(f"   • Nodos actualizados: {total_nodos}")
        print(f"   • Errores: {errores}")
        print("=" * 60)

        return total_nodos


if __name__ == "__main__":
    """
    Prueba del GraphBuilder con documentos de ejemplo.
    """
    import sys
    from pathlib import Path

    print("=" * 60)
    print("🧪 PRUEBA DEL GRAPH BUILDER")
    print("=" * 60)

    # 1. Crear documentos de ejemplo (simulando lo que cargaría el CSVLoader)
    print("\n📄 Creando documentos de ejemplo...")

    documentos_prueba = [
        Document(
            page_content="El contrato de arrendamiento establece que el pago mensual es de 500 euros. El inquilino debe realizar el pago antes del día 5 de cada mes.",
            metadata={
                "titulo": "Contrato Alquiler",
                "tipo": "Contrato",
                "pagina": 1,
                "id_chunk": "chunk_001",
                "archivo": "contrato_alquiler.pdf"
            }
        ),
        Document(
            page_content="La duración del contrato es de un año prorrogable. El propietario se llama Juan Pérez y el inquilino es María García.",
            metadata={
                "titulo": "Contrato Alquiler",
                "tipo": "Contrato",
                "pagina": 2,
                "id_chunk": "chunk_002",
                "archivo": "contrato_alquiler.pdf"
            }
        ),
        Document(
            page_content="El informe anual muestra un incremento del 15% en ventas. Las ventas online representan el 40% del total.",
            metadata={
                "titulo": "Informe Anual 2024",
                "tipo": "Informe",
                "pagina": 1,
                "id_chunk": "chunk_003",
                "archivo": "informe_2024.pdf"
            }
        ),
        Document(
            page_content="Se recomienda invertir en marketing digital para 2025. El presupuesto recomendado es de 50.000 euros.",
            metadata={
                "titulo": "Informe Anual 2024",
                "tipo": "Informe",
                "pagina": 2,
                "id_chunk": "chunk_004",
                "archivo": "informe_2024.pdf"
            }
        )
    ]

    print(f"✅ {len(documentos_prueba)} documentos de prueba creados")

    # 2. Mostrar resumen de documentos de prueba
    print("\n📊 RESUMEN DOCUMENTOS PRUEBA:")

    # Agrupar por tipo
    tipos = {}
    for doc in documentos_prueba:
        tipo = doc.metadata.get('tipo', 'Desconocido')
        tipos[tipo] = tipos.get(tipo, 0) + 1

    for tipo, count in tipos.items():
        print(f"   • {tipo}: {count} documentos")

    # 3. Inicializar GraphBuilder
    print("\n🕸️  Inicializando GraphBuilder...")

    try:
        builder = GraphBuilder(
            model_name="gpt-4o-mini",
            batch_size=2,
            verbose=True
        )

    except Exception as e:
        print(f"❌ Error inicializando GraphBuilder: {e}")
        sys.exit(1)

    # 4. Construir grafo
    print("\n" + "=" * 60)
    print("🚀 CONSTRUYENDO GRAFO DE PRUEBA")
    print("=" * 60)

    try:
        entidades, relaciones = builder.construir(
            documentos=documentos_prueba,
            limpiar=True  # Limpiar grafo existente
        )

    # 5. Verificación
    print("\n" + "=" * 60)
    print("🔍 VERIFICACIÓN")
    print("=" * 60)

    # Consultas de ejemplo
    consultas = [
        ("📊 Total nodos:", "MATCH (n) RETURN count(n) as total"),
        ("🔗 Total relaciones:", "MATCH ()-[r]->() RETURN count(r) as total"),
        ("🏷️ Tipos de nodo:", """
            MATCH (n)
            RETURN labels(n) as tipo, count(n) as count
            ORDER BY count DESC
            LIMIT 5
        """),
        ("📑 Tipos de relación:", """
            MATCH ()-[r]->()
            RETURN type(r) as tipo, count(r) as count
            ORDER BY count DESC
            LIMIT 5
        """),
        ("📄 Documentos fuente:", """
            MATCH (d:Document)
            RETURN d.id as id, d.source as fuente
            LIMIT 3
        """)
    ]

    for titulo, query in consultas:
        print(f"\n{titulo}")
        try:
            resultados = builder.consultar(query)
            for r in resultados:
                print(f"   {r}")
        except Exception as e:
            print(f"   ⚠️ Error en consulta: {e}")

    # 6. Explorar el grafo (primeros nodos)
    print("\n" + "=" * 60)
    print("🌐 PRIMEROS NODOS DEL GRAFO")
    print("=" * 60)

    query_nodos = """
    MATCH (n)
    RETURN n.id as id, labels(n) as tipo, n.description as descripcion
    LIMIT 5
    """

    try:
        resultados = builder.consultar(query_nodos)
        for i, r in enumerate(resultados, 1):
            print(f"\n--- Nodo {i} ---")
            print(f"   ID: {r.get('id', 'N/A')}")
            print(f"   Tipo: {r.get('tipo', 'N/A')}")
            desc = r.get('descripcion', '')
            if desc:
                print(f"   Desc: {desc[:100]}...")
    except Exception as e:
        print(f"   ⚠️ Error obteniendo nodos: {e}")

    # 7. Resumen final
    print("\n" + "=" * 60)
    print("✅ PRUEBA COMPLETADA")
    print("=" * 60)
    print(f"📊 Entidades creadas: {entidades}")
    print(f"🔗 Relaciones creadas: {relaciones}")
    print("\n🌐 Explora el grafo en: http://localhost:7474")

    except Exception as e:
        print(f"\n❌ Error durante la construcción del grafo: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # 5. ASOCIAR EMBEDDINGS (nuevo paso)
    print("\n" + "=" * 60)
    print("🔗 PASO 5: ASOCIAR EMBEDDINGS")
    print("=" * 60)

    try:
        # Cargar el CSV original para obtener embeddings
        from src.ingestion.loader import CSVLoader

        cargador_csv = CSVLoader("chunks.csv", cargar_embeddings=True)
        nodos_con_embedding = builder.asociar_embeddings(cargador_csv)

        if nodos_con_embedding > 0:
            print(f"\n✅ {nodos_con_embedding} nodos ahora tienen embeddings")

            # Verificación rápida
            result = builder.consultar("""
            MATCH (e:__Entity__)
            WHERE e.embedding IS NOT NULL
            RETURN count(e) as total, size(e.embedding) as dim
            LIMIT 1
            """)
            if result:
                print(f"📊 Nodos con embedding: {result[0]['total']}")
                print(f"📏 Dimensión: {result[0]['dim']}")
        else:
            print("⚠️ No se asociaron embeddings")

    except Exception as e:
        print(f"⚠️ Error asociando embeddings: {e}")
