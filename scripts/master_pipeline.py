"""
run_pipeline.py
---------------
Script maestro que coordina todo el pipeline:

  STEP 1 — Chunking + embeddings   (ingestion/)
     Si el CSV ya existe y no hay archivos .md más nuevos → lo salta

  STEP 2 — Ontología               (ner/)
     Si ontology.yaml existe → lo usa
     Si no existe → descubre vocabulario, genera borrador_ontologia.json
                    y DETIENE el pipeline con instrucciones al usuario

  STEP 3 — NER                     (ner/)
     Si ner_resultados.json existe y es más reciente que el CSV → lo salta
     Si no → procesa los chunks y guarda ner_resultados.json

  STEP 4 — Grafo Neo4j             (graph_building/)
     Fase A: entity resolution y deduplicación (entity_resolver.py)
     Fase B: summarización de entidades y relaciones (entity_summarizer.py)
     Fase C: community detection multi-nivel Leiden (community_detector.py)

Uso:
  python run_pipeline.py --solo-summaries
  python run_pipeline.py --forzar-embeddings   # regenera CSV aunque exista
  python run_pipeline.py --forzar-ner          # re-ejecuta NER
  python run_pipeline.py --forzar-grafo        # re-construye el grafo
  python run_pipeline.py --max-chunks 20       # limita chunks para pruebas
  python run_pipeline.py --skip-neo4j          # salta el STEP 4
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Rutas del proyecto ────────────────────────────────────────────────────────
RUTA_RAW        = Path("../data/raw")
RUTA_CSV        = Path("../data/processed/chunks/chunks_con_embeddings.csv")
RUTA_ONTOLOGIA  = Path("../data/ner/ontologia/ontology.yaml")
RUTA_BORRADOR   = Path("../data/ner/ontologia/borrador_ontologia.json")
RUTA_RESULTADOS = Path("../data/ner/ner_resultados.json")
RUTA_ACRONIMOS  = Path("../data/acronimos.yaml")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Chunking + Embeddings
# ─────────────────────────────────────────────────────────────────────────────

def paso_ingestion(forzar: bool = False) -> bool:
    print("\n" + "=" * 60)
    print("📦 STEP 1: CHUNKING + EMBEDDINGS")
    print("=" * 60)

    archivos_md = list(RUTA_RAW.glob("**/*.md"))
    if not archivos_md:
        print(f"❌ No se encontraron archivos .md en {RUTA_RAW}")
        return False

    print(f"   📄 Archivos .md encontrados: {len(archivos_md)}")
    for f in archivos_md:
        print(f"      • {f.name}")

    if not forzar and RUTA_CSV.exists():
        csv_mtime    = RUTA_CSV.stat().st_mtime
        md_más_nuevo = max(f.stat().st_mtime for f in archivos_md)

        if csv_mtime >= md_más_nuevo:
            print(f"\n✅ CSV actualizado: {RUTA_CSV}")
            print("   (usa --forzar-embeddings para regenerar)")
            return True
        else:
            print("\n⚠️  Hay archivos .md más nuevos que el CSV → regenerando...")

    print("\n🔄 Ejecutando chunking + embeddings...")
    try:
        from src.ingestion.pipeline import IngestionPipeline

        pipeline = IngestionPipeline()
        RUTA_CSV.parent.mkdir(parents=True, exist_ok=True)
        pipeline.ejecutar(
            carpeta_entrada=str(RUTA_RAW),
            archivo_salida=str(RUTA_CSV),
        )
        print(f"\n✅ CSV generado: {RUTA_CSV}")
        return True

    except Exception as e:
        print(f"\n❌ Error en ingestion: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Ontología
# ─────────────────────────────────────────────────────────────────────────────

def paso_ontologia() -> bool:
    print("\n" + "=" * 60)
    print("📋 STEP 2: ONTOLOGÍA")
    print("=" * 60)

    if RUTA_ONTOLOGIA.exists():
        print(f"✅ Ontología encontrada: {RUTA_ONTOLOGIA}")
        return True

    print(f"⚠️  No se encontró ontology.yaml en {RUTA_ONTOLOGIA.parent}")
    print("🔍 Descubriendo vocabulario desde los archivos .md...")

    try:
        from src.name_entity_recognition.ontologia import HybridDocumentAnalyzer

        archivos_md = list(RUTA_RAW.glob("**/*.md"))
        temp = RUTA_BORRADOR.parent / "_temp_vocab.md"
        temp.parent.mkdir(parents=True, exist_ok=True)
        temp.write_text(
            "\n\n".join(f.read_text(encoding="utf-8") for f in archivos_md),
            encoding="utf-8",
        )

        analizador = HybridDocumentAnalyzer(
            modelo_spacy="es_core_news_lg",
            patron_tabla=["Fila de tabla", "Tabla"],
        )
        resultado = analizador.analiza(temp, verbose=True)
        temp.unlink()

        analizador.guardar_ontology_json(
            ruta_salida=RUTA_BORRADOR,
            nodos=resultado["allowed_nodes"],
            relaciones=resultado["allowed_relationships"],
            archivo_fuente=str(RUTA_RAW),
        )

    except Exception as e:
        print(f"\n❌ Error descubriendo vocabulario: {e}")
        return False

    print()
    print("=" * 60)
    print("🛑  PIPELINE DETENIDO — ACCIÓN REQUERIDA")
    print("=" * 60)
    print()
    print(f"  Se ha generado un borrador de ontología en:")
    print(f"  📄 {RUTA_BORRADOR}")
    print()
    print("  Pasos para continuar:")
    print("  1. Abre borrador_ontologia.json y revisa nodos y relaciones")
    print("  2. Crea el YAML definitivo con esta estructura:")
    print()
    print("     entities:")
    print("       NombreEntidad:")
    print("         description: '...'")
    print("         patterns: ['patrón1', 'patrón2']")
    print("     relations:")
    print("       nombreRelacion:")
    print("         description: '...'")
    print("         domain: [EntidadOrigen]")
    print("         range:  [EntidadDestino]")
    print("     constraints: []")
    print()
    print(f"  3. Guárdalo como: {RUTA_ONTOLOGIA}")
    print("  4. Vuelve a ejecutar: python run_pipeline.py")
    print()
    return False


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — NER
# ─────────────────────────────────────────────────────────────────────────────

def paso_ner(forzar: bool = False, max_chunks: int = None) -> bool:
    print("\n" + "=" * 60)
    print("🏷️  STEP 3: NER")
    print("=" * 60)

    if not forzar and RUTA_RESULTADOS.exists():
        csv_mtime = RUTA_CSV.stat().st_mtime if RUTA_CSV.exists() else 0
        ner_mtime = RUTA_RESULTADOS.stat().st_mtime

        if ner_mtime >= csv_mtime:
            print(f"✅ ner_resultados.json actualizado: {RUTA_RESULTADOS}")
            print("   (usa --forzar-ner para re-ejecutar)")
            return True
        else:
            print("⚠️  El CSV es más nuevo que ner_resultados.json → re-ejecutando...")

    try:
        from src.name_entity_recognition.cargador_chunks import CargadorChunksCSV
        from src.name_entity_recognition.pipeline import PipelineNERDosPasos

        cargador = CargadorChunksCSV(str(RUTA_CSV))
        cargador.resumen()

        chunks = cargador.cargar_chunks(max_chunks=max_chunks)
        if not chunks:
            print("❌ No se cargaron chunks.")
            return False

        print(f"   📊 Chunks a procesar: {len(chunks)}")

        pipeline = PipelineNERDosPasos(ruta_ontologia=str(RUTA_ONTOLOGIA))

        todos_resultados = []
        for i, chunk in enumerate(chunks, 1):
            print(f"\n>>> Chunk {i}/{len(chunks)} | {chunk['titulo']} [{chunk['chunk_id']}]")
            resultado = pipeline.ejecutar(chunk["texto"])
            resultado.update({
                "chunk_id": chunk["chunk_id"],
                "titulo":   chunk["titulo"],
                "tipo":     chunk["tipo"],
                "fuente":   chunk["fuente"],
            })
            todos_resultados.append(resultado)

        RUTA_RESULTADOS.parent.mkdir(parents=True, exist_ok=True)
        with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
            json.dump(todos_resultados, f, ensure_ascii=False, indent=2)

        print(f"\n✅ NER completado: {RUTA_RESULTADOS}")
        print(f"   Chunks procesados : {len(todos_resultados)}")
        print(f"   Entidades totales : {sum(len(r.get('entidades', [])) for r in todos_resultados)}")
        print(f"   Relaciones totales: {sum(len(r.get('relaciones', [])) for r in todos_resultados)}")
        return True

    except Exception as e:
        print(f"\n❌ Error en NER: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Grafo Neo4j
# ─────────────────────────────────────────────────────────────────────────────

def paso_grafo(forzar: bool = False) -> bool:
    print("\n" + "=" * 60)
    print("🕸️  STEP 4: GRAFO NEO4J")
    print("=" * 60)

    if not RUTA_RESULTADOS.exists():
        print(f"❌ ner_resultados.json no encontrado: {RUTA_RESULTADOS}")
        return False

    try:
        from src.graph_building.entity_resolver import EntityResolver
        from src.graph_building.graph_builder import GraphBuilder
        from src.graph_building.entity_summarizer import EntitySummarizer
        from src.communities.community_detector import CommunityDetector

        # Fase A: resolución de entidades
        resolver   = EntityResolver(
            fuzzy_threshold=0.93,
            ruta_acronimos=RUTA_ACRONIMOS,
            verbose=True,
        )
        entity_map = resolver.resolve(str(RUTA_RESULTADOS))
        entity_map.print_stats()

        # Construcción del grafo con canónicos
        builder = GraphBuilder(
            verbose=True,
            solo_relaciones_validas=True,
            confianza_minima="medium",
            entity_map=entity_map,
        )
        builder.construir_desde_json(
            ruta_json=str(RUTA_RESULTADOS),
            limpiar=forzar,
        )

        # Fase B: summarización de entidades y relaciones
        summarizer = EntitySummarizer(graph=builder.graph, verbose=True)
        summarizer.summarize(entity_map)
        summarizer.summarize_relations()

        # Fase C: detección de comunidades
        detector = CommunityDetector(graph=builder.graph, verbose=True)
        detector.detect_and_summarize()

        builder.estadisticas()
        return True

    except Exception as e:
        print(f"\n❌ Error construyendo grafo: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pipeline maestro GraphRAG")
    parser.add_argument("--forzar-embeddings", action="store_true",
                        help="Regenera el CSV aunque ya exista")
    parser.add_argument("--forzar-ner", action="store_true",
                        help="Re-ejecuta el NER aunque ner_resultados.json sea reciente")
    parser.add_argument("--forzar-grafo", action="store_true",
                        help="Limpia y re-construye el grafo Neo4j")
    parser.add_argument("--skip-neo4j", action="store_true",
                        help="Salta el STEP 4 (construcción del grafo)")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Limita el número de chunks para pruebas")
    parser.add_argument("--solo-summaries", action="store_true",
                        help="Re-ejecuta solo summarización y community detection")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🚀 PIPELINE MAESTRO — GRAPHRAG")
    print("=" * 60)

    # STEP 1: Chunking + Embeddings
    if not paso_ingestion(forzar=args.forzar_embeddings):
        sys.exit(1)

    # STEP 2: Ontología (detiene si no hay YAML)
    if not paso_ontologia():
        sys.exit(0)

    # STEP 3: NER
    if not paso_ner(forzar=args.forzar_ner, max_chunks=args.max_chunks):
        sys.exit(1)

    # STEP 4: Grafo Neo4j
    if args.skip_neo4j:
        print("\n⏭️  STEP 4 saltado (--skip-neo4j)")
    elif args.solo_summaries:
        # Salta entity resolver e graph builder, riesegue solo B e C
        try:
            from src.graph_building.entity_resolver import EntityResolver
            from src.graph_building.graph_builder import GraphBuilder
            from src.graph_building.entity_summarizer import EntitySummarizer
            from src.communities.community_detector import CommunityDetector

            resolver = EntityResolver(
                fuzzy_threshold=0.93,
                ruta_acronimos=RUTA_ACRONIMOS,
                verbose=True,
            )
            entity_map = resolver.resolve(str(RUTA_RESULTADOS))

            builder = GraphBuilder(verbose=True, entity_map=entity_map)
            # No llama a construir_desde_json: el grafo ya existe en Neo4j

            summarizer = EntitySummarizer(graph=builder.graph, verbose=True)
            summarizer.summarize(entity_map)
            summarizer.summarize_relations()

            detector = CommunityDetector(graph=builder.graph, verbose=True)
            detector.detect_and_summarize()
        except Exception as e:
            print(f"\n❌ Error en summaries: {e}")
            sys.exit(1)
    else:
        if not paso_grafo(forzar=args.forzar_grafo):
            sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETADO")
    print("=" * 60)
    print(f"   CSV          : {RUTA_CSV}")
    print(f"   Ontología    : {RUTA_ONTOLOGIA}")
    print(f"   Acrónimos    : {RUTA_ACRONIMOS}")
    print(f"   Resultados   : {RUTA_RESULTADOS}")
    print(f"   Neo4j        : http://localhost:7474")


if __name__ == "__main__":
    main()