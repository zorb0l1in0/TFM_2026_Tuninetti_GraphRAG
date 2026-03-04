#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de ingestión de datos - Construye el grafo desde CSV
Ahora procesa CADA documento con su propio vocabulario descubierto
"""

import sys
from pathlib import Path
import pandas as pd
import tempfile

sys.path.append(str(Path(__file__).parent.parent))

from src.name_entity_recognition.cargador_chunks import CSVLoader
from src.graph_builder import GraphBuilder
from src.name_entity_recognition.ontology_extractor import HybridDocumentAnalyzer
from src.extraction.entity_relation_extractor import EntityRelationExtractor
from langchain_core.documents import Document


def main():
    """Función principal de ingestión"""

    print("\n" + "=" * 60)
    print("🚀 INGESTIÓN DE DATOS - GRAPHRAG")
    print("=" * 60)

    # =====================================================
    # CONFIGURACIÓN
    # =====================================================
    NOMBRE_CSV = "chunks_con_embeddings.csv"
    THRESHOLD_PERCENTUAL = 0.05  # 5% del máximo para filtrar nodos
    MIN_FRECUENCIA = 2
    BATCH_SIZE = 5

    # =====================================================
    # 1. Cargar CSV
    # =====================================================
    print("\n📂 PASO 1: CARGAR CSV")
    print("-" * 30)

    try:
        cargador = CSVLoader(NOMBRE_CSV, cargar_embeddings=True)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # =====================================================
    # 2. Mostrar resumen
    # =====================================================
    print(f"\n📊 RESUMEN:")
    print(f"   Total chunks: {len(cargador.df)}")

    # Obtener documentos únicos
    if 'archivo_origen' in cargador.df.columns:
        documentos_unicos = cargador.df['archivo_origen'].unique()
        print(f"\n📄 Documentos originales ({len(documentos_unicos)}):")
        for doc in documentos_unicos:
            chunks = len(cargador.df[cargador.df['archivo_origen'] == doc])
            tipo = cargador.df[cargador.df['archivo_origen'] == doc]['tipo'].iloc[0]
            print(f"   • {doc}: {chunks} chunks ({tipo})")
    else:
        print("❌ Columna 'archivo_origen' no encontrada en CSV")
        return

    # =====================================================
    # 3. Inicializar builder (sin vocabulario global)
    # =====================================================
    print("\n" + "=" * 60)
    print("🕸️  PASO 2: INICIALIZAR BUILDER")
    print("=" * 60)

    builder = GraphBuilder(
        model_name="gpt-4o-mini",
        batch_size=BATCH_SIZE,
        verbose=True,
        auto_discover_vocabulary=False  # Nosotros gestionamos el descubrimiento
    )

    # Limpiar grafo al inicio
    builder.graph.query("MATCH (n) DETACH DELETE n")
    print("✅ Grafo limpiado")

    # =====================================================
    # 4. Inicializar analizador híbrido
    # =====================================================
    analyzer = HybridDocumentAnalyzer(
        modelo_spacy="es_core_news_lg",
        patron_tabla=["Fila de tabla", "Tabla"]
    )

    # =====================================================
    # 5. PROCESAR CADA DOCUMENTO CON SU VOCABULARIO
    # =====================================================
    print("\n" + "=" * 60)
    print("🔍 PASO 3: PROCESAR DOCUMENTOS INDIVIDUALMENTE")
    print("=" * 60)

    total_entidades = 0
    total_relaciones = 0
    documentos_procesados = 0

    for doc_nombre in documentos_unicos:
        print(f"\n{'=' * 50}")
        print(f"📄 DOCUMENTO: {doc_nombre}")
        print(f"{'=' * 50}")

        # Obtener chunks de este documento
        chunks_doc = cargador.df[cargador.df['archivo_origen'] == doc_nombre]
        print(f"   📊 Chunks: {len(chunks_doc)}")

        # =====================================================
        # 5.1 Reconstruir texto original para análisis de vocabulario
        # =====================================================
        # Buscar columna de texto
        columna_texto = None
        for col in ['raw_text', 'texto', 'contenido', 'content']:
            if col in chunks_doc.columns:
                columna_texto = col
                break

        if not columna_texto:
            print(f"   ❌ No se encontró columna de texto en el CSV")
            continue

        # Reconstruir texto completo (aproximado)
        textos_validos = chunks_doc[columna_texto].dropna().tolist()
        if not textos_validos:
            print(f"   ❌ No hay textos válidos en este documento")
            continue

        texto_completo = "\n".join([str(t) for t in textos_validos])

        # Crear archivo temporal para análisis
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', encoding='utf-8', delete=False) as f:
            f.write(texto_completo)
            temp_path = Path(f.name)

        # =====================================================
        # 5.2 Descubrir vocabulario específico para este documento
        # =====================================================
        try:
            print(f"   🔍 Descubriendo vocabulario específico...")

            resultado = analyzer.analiza(
                temp_path,
                threshold_percentual=THRESHOLD_PERCENTUAL,
                min_frecuencia=MIN_FRECUENCIA,
                verbose=False
            )

            nodes = resultado['allowed_nodes']
            relations = resultado['allowed_relationships']

            print(f"   ✅ Vocabulario: {len(nodes)} nodos, {len(relations)} relaciones")
            if nodes:
                print(f"      Nodos: {nodes[:10]}{'...' if len(nodes) > 10 else ''}")
            if relations:
                print(f"      Relaciones: {relations[:5]}{'...' if len(relations) > 5 else ''}")

            # =====================================================
            # 5.3 Crear extractor TEMPORAL para este documento
            # =====================================================
            extractor = EntityRelationExtractor(
                model_name="gpt-4o-mini",
                verbose=False,
                allowed_nodes=nodes,
                allowed_relationships=relations,
                auto_discover=False
            )

            # =====================================================
            # 5.4 Crear documentos LangChain para los chunks
            # =====================================================
            documentos_doc = []
            for idx, row in chunks_doc.iterrows():
                # Extraer texto
                texto = str(row[columna_texto]) if pd.notna(row[columna_texto]) else ""
                if not texto:
                    continue

                # Metadatos básicos
                metadata = {
                    "titulo": doc_nombre,
                    "tipo": row.get('tipo', 'desconocido') if 'tipo' in row else 'desconocido',
                    "pagina": int(row['page']) if 'page' in row and pd.notna(row['page']) else 0,
                    "id_chunk": row.get('id', f"{doc_nombre}_{idx}") if 'id' in row else f"{doc_nombre}_{idx}",
                    "archivo": doc_nombre,
                }

                # Embedding si existe
                if 'embedding' in row and pd.notna(row['embedding']):
                    embedding = cargador._extraer_embedding(row)
                    if embedding:
                        metadata["embedding"] = embedding

                documentos_doc.append(Document(page_content=texto, metadata=metadata))

            print(f"   📄 Documentos LangChain creados: {len(documentos_doc)}")

            # =====================================================
            # 5.5 Procesar chunks en lotes
            # =====================================================
            print(f"   🕸️  Extrayendo entidades y relaciones...")

            for i in range(0, len(documentos_doc), BATCH_SIZE):
                lote = documentos_doc[i:i + BATCH_SIZE]

                try:
                    graph_documents, entidades, relaciones = extractor.extract(
                        documents=lote,
                        batch_size=len(lote)
                    )

                    # Guardar en Neo4j
                    builder.graph.add_graph_documents(
                        graph_documents,
                        baseEntityLabel=True,
                        include_source=True
                    )

                    total_entidades += entidades
                    total_relaciones += relaciones
                    documentos_procesados += len(lote)

                    print(f"      Lote {i // BATCH_SIZE + 1}: {entidades} ent, {relaciones} rel")

                except Exception as e:
                    print(f"      ❌ Error lote: {e}")

        finally:
            # Limpiar archivo temporal
            temp_path.unlink()

    # =====================================================
    # 6. ASOCIAR EMBEDDINGS
    # =====================================================
    print("\n" + "=" * 60)
    print("🔗 PASO 4: ASOCIAR EMBEDDINGS A NODOS")
    print("=" * 60)

    if 'embedding' in cargador.df.columns:
        # Usar el loader para asociar embeddings (usa su propio método)
        nodos_actualizados = builder.asociar_embeddings(cargador)

        if nodos_actualizados > 0:
            print(f"\n✅ {nodos_actualizados} nodos actualizados con embeddings")

            # Verificar embeddings en Neo4j
            result = builder.consultar("""
                MATCH (e:__Entity__)
                WHERE e.embedding IS NOT NULL
                RETURN count(e) as total, size(e.embedding) as dim
                LIMIT 1
                """)
            if result:
                print(f"📊 Nodos con embedding: {result[0]['total']}")
                if 'dim' in result[0]:
                    print(f"📏 Dimensión embeddings: {result[0]['dim']}")
    else:
        print("⚠️ No se encontró columna 'embedding' en el CSV")

    # =====================================================
    # 7. VERIFICACIÓN FINAL
    # =====================================================
    print("\n" + "=" * 60)
    print("🔍 VERIFICACIÓN FINAL")
    print("=" * 60)

    result = builder.consultar("MATCH (n) RETURN count(n) as total")
    if result:
        print(f"\n📊 Nodos en Neo4j: {result[0]['total']}")

    result = builder.consultar("MATCH ()-[r]->() RETURN count(r) as total")
    if result:
        print(f"🔗 Relaciones en Neo4j: {result[0]['total']}")

    print("\n🌐 Puedes explorar el grafo en:")
    print("   http://localhost:7474")

    # =====================================================
    # 8. RESUMEN FINAL
    # =====================================================
    print("\n" + "=" * 60)
    print("✅ INGESTIÓN COMPLETADA")
    print("=" * 60)
    print(f"📊 Total chunks en CSV: {len(cargador.df)}")
    print(f"📄 Documentos procesados: {len(documentos_unicos)}")
    print(f"🕸️  Chunks procesados: {documentos_procesados}")
    print(f"🕸️  Total entidades: {total_entidades}")
    print(f"🔗 Total relaciones: {total_relaciones}")


if __name__ == "__main__":
    main()