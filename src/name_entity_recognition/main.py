"""
master_pipeline.py
------------------
Pipeline maestro: Markdown → Chunks/Embeddings → Ontología → NER → Grafo Neo4j

Lógica condicional:
  1. Si falta data/processed/chunks/chunks_con_embeddings.csv
       → ejecuta IngestionPipeline (chunking + embeddings)

  2. Si falta data/ontology.json
       → avisa al usuario de que genere el borrador con PipelineNERDosPasos
       → se detiene

  3. Si existe todo
       → ejecuta NER sobre todos los chunks
       → escribe ner_resultados.json
       → construye el grafo en Neo4j

Uso rápido:
    python master_pipeline.py

Uso avanzado:
    python master_pipeline.py \\
        --raw       data/raw \\
        --processed data/processed/chunks/chunks_con_embeddings.csv \\
        --ontology  data/ontology.json \\
        --ner-out   data/ner/ner_resultados.json \\
        --max-chunks 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

from src.common.clients import get_neo4j_graph

# ─────────────────────────────────────────────────────────────────────────────
# Rutas por defecto
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_RAW      = "data/raw"
DEFAULT_CSV      = "data/processed/chunks/chunks_con_embeddings.csv"
DEFAULT_ONTOLOGY = "data/ontology.json"
DEFAULT_NER_OUT  = "data/ner/ner_resultados.json"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – Ingestion (chunking + embeddings)
# ─────────────────────────────────────────────────────────────────────────────

def step_ingestion(
    carpeta_raw: Path,
    archivo_csv: Path,
    max_caracteres: int = 1500,
    batch_size: int = 20,
) -> pd.DataFrame:
    print("\n" + "═" * 60)
    print("📦 STEP 1 — Ingestion: Markdown → Chunks + Embeddings")
    print("═" * 60)

    try:
        from src.ingestion.pipeline import IngestionPipeline
    except ImportError as e:
        _abort(f"No se puede importar ingestion.pipeline: {e}")

    pipeline = IngestionPipeline(
        max_caracteres_chunk=max_caracteres,
        batch_size=batch_size,
    )
    return pipeline.ejecutar(
        carpeta_entrada=str(carpeta_raw),
        archivo_salida=str(archivo_csv),
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – NER
# ─────────────────────────────────────────────────────────────────────────────

def step_ner(
    ruta_csv: Path,
    ruta_ontologia: Path,
    carpeta_raw: Path,
    ruta_salida: Path,
    max_chunks: Optional[int] = None,
) -> List[Dict]:
    print("\n" + "═" * 60)
    print("🔍 STEP 2 — NER: extracción de entidades y relaciones")
    print("═" * 60)

    try:
        from ner.cargador_chunks import CargadorChunksCSV   # type: ignore
        from ner.pipeline import PipelineNERDosPasos        # type: ignore
    except ImportError as e:
        _abort(f"No se puede importar ner.*: {e}")

    # Vocabulario: concatena todos los .md en un archivo temporal
    archivos_md = list(carpeta_raw.glob("**/*.md"))
    temp_vocab  = ruta_salida.parent / "_temp_vocab.md"
    temp_vocab.parent.mkdir(parents=True, exist_ok=True)
    temp_vocab.write_text(
        "\n\n".join(a.read_text(encoding="utf-8") for a in archivos_md),
        encoding="utf-8",
    )

    # Carga chunks
    cargador = CargadorChunksCSV(str(ruta_csv))
    cargador.resumen()
    chunks = cargador.cargar_chunks(max_chunks=max_chunks)

    if not chunks:
        _abort("No se cargaron chunks. Verifica el CSV.")

    ner_pipeline = PipelineNERDosPasos(ruta_vocabulario=temp_vocab)
    todos: List[Dict] = []

    for i, chunk in enumerate(chunks, 1):
        print(f"\n  >>> Chunk {i}/{len(chunks)} | {chunk['titulo']} [{chunk['chunk_id']}]")
        resultado = ner_pipeline.ejecutar(chunk["texto"])
        resultado.update({
            "chunk_id": chunk["chunk_id"],
            "titulo":   chunk["titulo"],
            "tipo":     chunk["tipo"],
            "fuente":   chunk.get("fuente", ""),
        })
        todos.append(resultado)
        print(ner_pipeline.formatear_resultado(resultado))

    temp_vocab.unlink(missing_ok=True)

    # Guarda JSON
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    ruta_salida.write_text(
        json.dumps(todos, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  📄 NER completado: {ruta_salida} ({len(todos)} chunks)")

    # Guarda CSV entidades
    filas = [
        {
            "chunk_id":      r.get("chunk_id"),
            "titulo":        r.get("titulo"),
            "tipo":          r.get("tipo"),
            "fuente":        r.get("fuente"),
            "texto_entidad": e.get("text"),
            "tipo_entidad":  e.get("entity_type"),
            "confianza":     e.get("confidence"),
        }
        for r in todos
        for e in r.get("entidades", [])
    ]
    if filas:
        csv_ent = ruta_salida.parent / "ner_entidades.csv"
        pd.DataFrame(filas).to_csv(csv_ent, index=False)
        print(f"  📊 CSV entidades: {csv_ent} ({len(filas)} filas)")

    return todos


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – Construcción grafo Neo4j
# ─────────────────────────────────────────────────────────────────────────────

def step_neo4j(
    ner_resultados: List[Dict],
    ruta_csv: Path,
    ruta_ontologia: Path,
) -> None:
    print("\n" + "═" * 60)
    print("🕸️  STEP 3 — Construcción grafo Neo4j")
    print("═" * 60)

    # Carga ontología para etiquetas canónicas
    ontologia: Dict = {}
    if ruta_ontologia.exists():
        try:
            ontologia = json.loads(ruta_ontologia.read_text(encoding="utf-8"))
        except Exception:
            print("  ⚠️  Ontología no legible, se continúa sin ella.")

    graph = get_neo4j_graph()

    # ── Restricciones/índices ─────────────────────────────────────────────
    _run(graph, "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Documento) REQUIRE d.id IS UNIQUE")
    _run(graph, "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
    _run(graph, "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entidad) REQUIRE (e.texto, e.tipo) IS NODE KEY")
    _run(graph, "CREATE CONSTRAINT IF NOT EXISTS FOR (t:TipoEntidad) REQUIRE t.nombre IS UNIQUE")

    # ── Tipos de entidad desde la ontología ──────────────────────────────
    for tipo in ontologia.get("entidades", []):
        _run(graph,
             "MERGE (t:TipoEntidad {nombre: $nombre}) SET t.descripcion = $desc",
             {"nombre": tipo.get("nombre", ""), "desc": tipo.get("descripcion", "")})

    # ── Carga metadatos chunks desde CSV ─────────────────────────────────
    df_chunks: Dict[str, Dict] = {}
    if ruta_csv.exists():
        df = pd.read_csv(ruta_csv)
        for _, row in df.iterrows():
            df_chunks[str(row.get("id_chunk", row.get("chunk_id", "")))] = row.to_dict()

    documentos_creados: set = set()

    for resultado in ner_resultados:
        chunk_id = resultado.get("chunk_id", "")
        titulo   = resultado.get("titulo", "sin_titulo")
        tipo_doc = resultado.get("tipo", "Desconocido")
        fuente   = resultado.get("fuente", "")

        # Documento
        doc_id = _slugify(titulo)
        if doc_id not in documentos_creados:
            _run(graph,
                 "MERGE (d:Documento {id: $id}) SET d.titulo = $titulo, d.tipo = $tipo, d.fuente = $fuente",
                 {"id": doc_id, "titulo": titulo, "tipo": tipo_doc, "fuente": fuente})
            documentos_creados.add(doc_id)

        # Chunk
        meta = df_chunks.get(chunk_id, {})
        _run(graph,
             "MERGE (c:Chunk {id: $id}) "
             "SET c.texto = $texto, c.titulo_seccion = $ts, "
             "c.caracteres = $chars, c.palabras = $words, c.pagina = $pagina",
             {
                 "id":    chunk_id,
                 "texto": resultado.get("texto_original", meta.get("texto", ""))[:2000],
                 "ts":    meta.get("titulo_seccion", ""),
                 "chars": int(meta.get("caracteres", 0)),
                 "words": int(meta.get("palabras", 0)),
                 "pagina": int(meta.get("pagina", 0)),
             })

        # Documento → Chunk
        _run(graph,
             "MATCH (d:Documento {id: $did}), (c:Chunk {id: $cid}) MERGE (d)-[:TIENE_CHUNK]->(c)",
             {"did": doc_id, "cid": chunk_id})

        # Entidades
        for entidad in resultado.get("entidades", []):
            texto_e = entidad.get("text", "").strip()
            tipo_e  = entidad.get("entity_type", "UNKNOWN")
            conf    = float(entidad.get("confidence", 0.0))
            if not texto_e:
                continue

            _run(graph,
                 "MERGE (e:Entidad {texto: $texto, tipo: $tipo}) "
                 "ON CREATE SET e.confianza_promedio = $conf "
                 "ON MATCH SET e.confianza_promedio = round((e.confianza_promedio + $conf) / 2.0, 4)",
                 {"texto": texto_e, "tipo": tipo_e, "conf": conf})

            _run(graph,
                 "MATCH (c:Chunk {id: $cid}), (e:Entidad {texto: $texto, tipo: $tipo}) "
                 "MERGE (c)-[r:MENCIONA]->(e) SET r.confianza = $conf",
                 {"cid": chunk_id, "texto": texto_e, "tipo": tipo_e, "conf": conf})

            _run(graph,
                 "MERGE (t:TipoEntidad {nombre: $tipo}) "
                 "WITH t MATCH (e:Entidad {texto: $texto, tipo: $tipo}) "
                 "MERGE (e)-[:ES_DE_TIPO]->(t)",
                 {"tipo": tipo_e, "texto": texto_e})

        # Relaciones NER
        for rel in resultado.get("relaciones", []):
            origen  = rel.get("origen", "").strip()
            destino = rel.get("destino", "").strip()
            tipo_r  = _safe_rel_type(rel.get("tipo", "RELATED_TO"))
            if not origen or not destino:
                continue
            _run(graph,
                 f"MATCH (a:Entidad {{texto: $orig}}), (b:Entidad {{texto: $dest}}) "
                 f"MERGE (a)-[:{tipo_r}]->(b)",
                 {"orig": origen, "dest": destino})

    # ── Estadísticas finales ──────────────────────────────────────────────
    stats = {
        "Documentos":   graph.query("MATCH (d:Documento) RETURN count(d) AS n")[0]["n"],
        "Chunks":       graph.query("MATCH (c:Chunk) RETURN count(c) AS n")[0]["n"],
        "Entidades":    graph.query("MATCH (e:Entidad) RETURN count(e) AS n")[0]["n"],
        "TiposEntidad": graph.query("MATCH (t:TipoEntidad) RETURN count(t) AS n")[0]["n"],
    }

    print("\n  🕸️  Grafo Neo4j construido con éxito!")
    for label, n in stats.items():
        print(f"     {label:<14}: {n}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run(graph: Any, cypher: str, params: dict = {}) -> Any:
    try:
        return graph.query(cypher, params)
    except Exception as e:
        print(f"  ⚠️  Cypher error: {e}\n     Query: {cypher[:80]}")


def _slugify(texto: str) -> str:
    import re
    return re.sub(r"[^a-z0-9_]", "_", texto.lower())[:80]


def _safe_rel_type(tipo: str) -> str:
    import re
    return re.sub(r"[^A-Z0-9_]", "_", tipo.upper())[:50] or "RELATED_TO"


def _abort(msg: str) -> None:
    print(f"\n❌ ERROR: {msg}")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Orquestador principal
# ─────────────────────────────────────────────────────────────────────────────

def run(
    carpeta_raw: Path,
    ruta_csv: Path,
    ruta_ontologia: Path,
    ruta_ner: Path,
    max_chunks: Optional[int] = None,
    max_caracteres: int = 1500,
    batch_size: int = 20,
    skip_neo4j: bool = False,
) -> None:
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         MASTER PIPELINE  —  MD → KG Neo4j               ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Raw        : {carpeta_raw}")
    print(f"  CSV chunks : {ruta_csv}")
    print(f"  Ontología  : {ruta_ontologia}")
    print(f"  NER output : {ruta_ner}")

    # ── STEP 1: Ingestion ─────────────────────────────────────────────────────
    if not ruta_csv.exists():
        print(f"\n  ℹ️  CSV no encontrado ({ruta_csv}), iniciando ingestion...")
        step_ingestion(
            carpeta_raw=carpeta_raw,
            archivo_csv=ruta_csv,
            max_caracteres=max_caracteres,
            batch_size=batch_size,
        )
    else:
        print(f"\n  ✅ STEP 1 — CSV ya presente: {ruta_csv}")

    # ── STEP 2: Ontología ─────────────────────────────────────────────────────
    if not ruta_ontologia.exists():
        print(f"\n  ⚠️  Ontología no encontrada ({ruta_ontologia})")
        print("  → Generando borrador + ontología final automáticamente con LLM...")

        try:
            import json
            from .ontologia import HybridDocumentAnalyzer  # same module used for borrador
            from langchain_core.messages import HumanMessage
            from src.common.clients import get_langchain_llm_paso2
            from .prompts import PROMPT_GENERAR_ONTOLOGIA
            # 1. Crear borrador desde todos los .md
            print("  🟦 Generando borrador_ontologia.json...")
            archivos_md = list(carpeta_raw.glob("**/*.md"))
            bloque = "\n\n".join(a.read_text(encoding="utf-8") for a in archivos_md)

            borr_temp = ruta_ontologia.parent / "borrador_ontologia.json"
            borr_temp.write_text(bloque, encoding="utf-8")

            analizador = HybridDocumentAnalyzer(
                modelo_spacy="es_core_news_lg",
                patron_tabla=["Fila de tabla", "Tabla"],
            )
            resultado = analizador.analiza(borr_temp, verbose=False)

            borrador = {
                "allowed_nodes": resultado["allowed_nodes"],
                "allowed_relationships": resultado["allowed_relationships"],
                "archivo_fuente": str(carpeta_raw)
            }

            borr_temp.write_text(
                json.dumps(borrador, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )

            print("  🟩 Borrador generado correctamente.")

            # 3. Llamada LLM
            llm = get_langchain_llm_paso2(temperatura=0.0)

            entrada = PROMPT_GENERAR_ONTOLOGIA + "\n\nBORRADOR:\n" + json.dumps(
                borrador, indent=2, ensure_ascii=False
            )

            respuesta = llm.invoke([HumanMessage(content=entrada)])
            ontologia_json = respuesta.content

            # 4. Guardar ontología final
            ruta_ontologia.write_text(ontologia_json, encoding="utf-8")
            print(f"  🟩 Ontología generada automáticamente en: {ruta_ontologia}. ATENCIÓN: revisa su contenido y corrige posibles errores.")

        except Exception as e:
            print(f"\n❌ Error generando ontología automáticamente: {e}")
            sys.exit(1)

    else:
        print(f"  ✅ STEP 2 — Ontología encontrada: {ruta_ontologia}")

    # ── STEP 3: NER ───────────────────────────────────────────────────────────
    if not ruta_ner.exists():
        print(f"\n  ℹ️  NER output no encontrado ({ruta_ner}), iniciando NER...")
        ner_resultados = step_ner(
            ruta_csv=ruta_csv,
            ruta_ontologia=ruta_ontologia,
            carpeta_raw=carpeta_raw,
            ruta_salida=ruta_ner,
            max_chunks=max_chunks,
        )
    else:
        print(f"  ✅ STEP 3 — NER ya ejecutado: {ruta_ner}")
        ner_resultados = json.loads(ruta_ner.read_text(encoding="utf-8"))

    # ── STEP 4: Neo4j ─────────────────────────────────────────────────────────
    if skip_neo4j:
        print("\n  ⏭️  STEP 4 — Neo4j omitido (--skip-neo4j)")
    else:
        step_neo4j(
            ner_resultados=ner_resultados,
            ruta_csv=ruta_csv,
            ruta_ontologia=ruta_ontologia,
        )

    print("\n" + "═" * 60)
    print("🎉 ¡Pipeline completada con éxito!")
    print("═" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Master pipeline: Markdown → Chunks → NER → Grafo Neo4j",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw",            default=DEFAULT_RAW,      help="Carpeta documentos Markdown")
    p.add_argument("--processed",      default=DEFAULT_CSV,      help="CSV chunks+embeddings")
    p.add_argument("--ontology",       default=DEFAULT_ONTOLOGY, help="Ontología JSON finalizada")
    p.add_argument("--ner-out",        default=DEFAULT_NER_OUT,  help="Output NER JSON")
    p.add_argument("--max-chunks",     type=int, default=None,   help="Límite chunks para NER")
    p.add_argument("--max-caracteres", type=int, default=1500,   help="Máx caracteres por chunk")
    p.add_argument("--batch-size",     type=int, default=20,     help="Batch size embeddings")
    p.add_argument("--skip-neo4j",     action="store_true",      help="Omitir construcción grafo")
    p.add_argument("--reset-ner",      action="store_true",      help="Forzar re-ejecución NER")
    p.add_argument("--reset-all",      action="store_true",      help="Forzar re-ejecución completa")
    return p


def main():
    load_dotenv()
    args = _build_parser().parse_args()

    ruta_csv       = Path(args.processed)
    ruta_ontologia = Path(args.ontology)
    ruta_ner       = Path(args.ner_out)

    if args.reset_all:
        for p in [ruta_csv, ruta_ner]:
            p.unlink(missing_ok=True)
        print("♻️  Reset completo: CSV y NER eliminados.")
    elif args.reset_ner:
        ruta_ner.unlink(missing_ok=True)
        print("♻️  Reset NER: ner_resultados.json eliminado.")

    run(
        carpeta_raw    = Path(args.raw),
        ruta_csv       = ruta_csv,
        ruta_ontologia = ruta_ontologia,
        ruta_ner       = ruta_ner,
        max_chunks     = args.max_chunks,
        max_caracteres = args.max_caracteres,
        batch_size     = args.batch_size,
        skip_neo4j     = args.skip_neo4j,
    )


if __name__ == "__main__":
    main()