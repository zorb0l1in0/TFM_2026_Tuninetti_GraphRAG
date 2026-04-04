"""
master_pipeline.py
------------------
Pipeline maestro GraphRAG:

  STEP 1 — Chunking + embeddings
  STEP 2 — Ontología (AUTOMÁTICA)
  STEP 3 — NER (paralelizado)
  STEP 4 — Grafo Neo4j
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from src.common.clients import MODEL_PASO1
from dotenv import load_dotenv
load_dotenv()

# Forza flush immediato su Windows/PyCharm
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# -------------------------------
# Rutas del proyecto
# -------------------------------
_ROOT           = Path(__file__).resolve().parent.parent
RUTA_RAW        = _ROOT / "data" / "raw"
RUTA_CSV        = _ROOT / "data" / "processed" / "chunks" / "chunks_con_embeddings.csv"
RUTA_ONTOLOGIA  = _ROOT / "data" / "ner" / "ontologia" / "ontology.yaml"
RUTA_BORRADOR   = _ROOT / "data" / "ner" / "ontologia" / "borrador_ontologia.json"
RUTA_RESULTADOS = _ROOT / "data" / "ner" / "ner_resultados.json"
RUTA_ACRONIMOS  = _ROOT / "data" / "acronimos.yaml"
_print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    msg = sep.join(str(a) for a in args) + end
    with _print_lock:
        sys.stdout.write(msg)
        sys.stdout.flush()


# ============================================================================
# KEEPALIVE — evita que el servidor descargue el modelo por inactividad
# ============================================A<================================
def _keepalive_loop(stop_event: threading.Event):
    """Manda una petición mínima cada 30s para mantener el modelo cargado."""
    from openai import OpenAI
    base_url = os.getenv("LLM_BASE_URL")
    api_key  = os.getenv("LLM_API_KEY")
    model = MODEL_PASO1

    client = OpenAI(api_key=api_key, base_url=base_url)
    while not stop_event.is_set():
        stop_event.wait(30)
        if stop_event.is_set():
            break
        try:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ok"}],
                max_tokens=1,
                timeout=10,
            )
            safe_print("  🔄 Keepalive OK", flush=True)
        except Exception as e:
            safe_print(f"  ⚠ Keepalive fallito: {e}")


def iniciar_keepalive() -> threading.Event:
    stop = threading.Event()
    threading.Thread(target=_keepalive_loop, args=(stop,), daemon=True).start()
    safe_print("  🟢 Keepalive iniciado (ping cada 30s)")
    return stop


# ============================================================================
# STEP 1 — INGESTIÓN
# ============================================================================
def paso_ingestion(forzar: bool = False) -> bool:
    safe_print("\n" + "=" * 60)
    safe_print("📦 STEP 1: CHUNKING + EMBEDDINGS")
    safe_print("=" * 60)

    archivos_md = list(RUTA_RAW.glob("**/*.md"))
    if not archivos_md:
        safe_print(f"❌ No se encontraron archivos .md en {RUTA_RAW}")
        return False

    if not forzar and RUTA_CSV.exists():
        csv_mtime = RUTA_CSV.stat().st_mtime
        md_newest = max(f.stat().st_mtime for f in archivos_md)
        if csv_mtime >= md_newest:
            safe_print("✅ CSV actualizado.")
            return True

    safe_print("🔄 Ejecutando ingestion...")
    try:
        from src.ingestion.pipeline import IngestionPipeline
        pipeline = IngestionPipeline()
        RUTA_CSV.parent.mkdir(parents=True, exist_ok=True)
        pipeline.ejecutar(str(RUTA_RAW), str(RUTA_CSV))
        safe_print(f"✅ CSV generado: {RUTA_CSV}")
        return True
    except Exception as e:
        safe_print(f"❌ Error en ingestion: {e}")
        return False


# ============================================================================
# STEP 2 — ONTOLOGÍA (AUTOMÁTICA ✔)
# ============================================================================
def paso_ontologia() -> bool:
    safe_print("\n" + "=" * 60)
    safe_print("📋 STEP 2: ONTOLOGÍA")
    safe_print("=" * 60)

    if RUTA_ONTOLOGIA.exists():
        safe_print(f"✅ Ontología ya existe: {RUTA_ONTOLOGIA}")
        return True

    safe_print("⚠️  Ontología no encontrada. Generación automática...")

    try:
        from src.name_entity_recognition.ontologia import HybridDocumentAnalyzer
        from langchain_core.messages import HumanMessage
        from src.common.clients import get_langchain_llm_paso2
        from src.name_entity_recognition.prompts import PROMPT_GENERAR_ONTOLOGIA

        archivos_md = list(RUTA_RAW.glob("**/*.md"))
        contenido_md = "\n\n".join(
            f.read_text(encoding="utf-8") for f in archivos_md
        )

        safe_print("🔧 Generando borrador_ontologia.json...")
        RUTA_BORRADOR.parent.mkdir(parents=True, exist_ok=True)
        temp_md = RUTA_BORRADOR.parent / "_temp_vocab.md"
        temp_md.write_text(contenido_md, encoding="utf-8")

        analizador = HybridDocumentAnalyzer(
            modelo_spacy="es_core_news_lg",
            patron_tabla=["Fila de tabla", "Tabla"],
        )
        resultado = analizador.analiza(temp_md, verbose=False)
        temp_md.unlink()

        borrador = {
            "allowed_nodes": resultado["allowed_nodes"],
            "allowed_relationships": resultado["allowed_relationships"],
            "archivo_fuente": str(RUTA_RAW),
        }
        RUTA_BORRADOR.write_text(
            json.dumps(borrador, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        safe_print(f"🟦 Borrador creado: {RUTA_BORRADOR}")

        safe_print("🤖 LLM generando ontology.yaml final...")
        llm = get_langchain_llm_paso2(0.0)
        entrada = (
            PROMPT_GENERAR_ONTOLOGIA
            + "\n\nBORRADOR:\n"
            + json.dumps(borrador, indent=2, ensure_ascii=False)
        )
        respuesta = llm.invoke([HumanMessage(content=entrada)])
        yaml_generado = respuesta.content
        yaml_generado = re.sub(r"^```(?:yaml)?\s*\n?", "", yaml_generado.strip())
        yaml_generado = re.sub(r"\n?```\s*$", "", yaml_generado.strip())
        yaml_generado = yaml_generado.strip()

        RUTA_ONTOLOGIA.parent.mkdir(parents=True, exist_ok=True)
        RUTA_ONTOLOGIA.write_text(yaml_generado, encoding="utf-8")
        safe_print(f"🟩 Ontología generada correctamente: {RUTA_ONTOLOGIA}")
        safe_print("   (Revisa su contenido por si necesita ajustes finos)")
        return True

    except Exception as e:
        safe_print(f"❌ Error generando ontología: {e}")
        return False


# ============================================================================
# STEP 3 — NER
# ============================================================================
def paso_ner(forzar: bool = False, max_chunks: int = None, workers: int = 1) -> bool:
    safe_print("\n" + "=" * 60)
    safe_print("🏷️  STEP 3: NER")
    safe_print("=" * 60)

    if not forzar and RUTA_RESULTADOS.exists():
        csv_mtime = RUTA_CSV.stat().st_mtime if RUTA_CSV.exists() else 0
        ner_mtime = RUTA_RESULTADOS.stat().st_mtime
        if ner_mtime >= csv_mtime:
            safe_print("✅ ner_resultados.json ya actualizado.")
            return True

    try:
        from src.name_entity_recognition.cargador_chunks import CargadorChunksCSV
        from src.name_entity_recognition.pipeline import PipelineNERDosPasos
        from src.name_entity_recognition.relation_canonicalizer import canonicalizar_post_ner

        cargador = CargadorChunksCSV(str(RUTA_CSV))
        cargador.resumen()

        chunks = cargador.cargar_chunks(max_chunks=max_chunks)
        if not chunks:
            safe_print("❌ No se cargaron chunks.")
            return False

        pipeline = PipelineNERDosPasos(ruta_ontologia=str(RUTA_ONTOLOGIA))

        todos = [None] * len(chunks)
        errores = []
        MAX_RETRY = 3

        # Keepalive — mantiene el modelo cargado durante todo el NER
        #stop_ka = iniciar_keepalive()

        def procesar(args):
            idx, ch = args
            titulo_corto = ch['titulo'][:80] + ("…" if len(ch['titulo']) > 80 else "")
            for intento in range(1, MAX_RETRY + 1):
                safe_print(f"  → [{idx+1:02d}/{len(chunks)}] {ch['chunk_id']} (intento {intento}/{MAX_RETRY})")
                try:
                    res = pipeline.ejecutar(ch["texto"])
                    res.update({k: v for k, v in ch.items() if k != "texto"})
                    res.pop("_log", "")
                    safe_print(
                        f"\n┌─ [{idx+1:02d}/{len(chunks)}] {ch['chunk_id']} "
                        f"({len(ch['texto'])} chars)\n"
                        f"│  {titulo_corto}\n"
                        f"└{'─' * 58}"
                    )
                    return idx, res, None
                except Exception as e:
                    nombre = type(e).__name__
                    es_timeout = "Timeout" in nombre or "timeout" in str(e).lower()
                    es_unloaded = "unloaded" in str(e).lower() or "canceled" in str(e).lower()
                    es_bad_request_cancel = "BadRequestError" in nombre and "canceled" in str(e).lower()
                    if (es_timeout or es_unloaded or es_bad_request_cancel) and intento < MAX_RETRY:
                        espera = 30 * intento
                        safe_print(f"  ⚠ [{ch['chunk_id']}] {nombre} (intento {intento}) — reintentando en {espera}s...")
                        time.sleep(espera)
                        continue
                    import traceback
                    safe_print(
                        f"\n┌─ [{idx+1:02d}/{len(chunks)}] {ch['chunk_id']} ❌ ERROR tras {intento} intentos\n"
                        f"│  {titulo_corto}\n"
                        f"│  {e}\n"
                        + "\n".join(f"│  {l}" for l in traceback.format_exc().splitlines())
                        + f"\n└{'─' * 58}"
                    )
                    return idx, None, str(e)

        if workers == 1:
            for i, ch in enumerate(chunks):
                idx, resultado, error = procesar((i, ch))
                if resultado:
                    todos[idx] = resultado
                else:
                    errores.append((idx, error))
        else:
            completados = [0]
            stop_hb = threading.Event()
            t_inicio = time.time()

            def heartbeat():
                while not stop_hb.is_set():
                    stop_hb.wait(15)
                    if not stop_hb.is_set():
                        elapsed = int(time.time() - t_inicio)
                        safe_print(
                            f"  ⏳ NER en curso... "
                            f"{completados[0]}/{len(chunks)} chunks completados "
                            f"({elapsed}s, {workers} workers)"
                        )

            threading.Thread(target=heartbeat, daemon=True).start()

            with ThreadPoolExecutor(max_workers=workers) as ex:
                futuros = {ex.submit(procesar, (i, ch)): i for i, ch in enumerate(chunks)}
                for fut in as_completed(futuros):
                    idx, resultado, error = fut.result()
                    completados[0] += 1
                    if resultado:
                        todos[idx] = resultado
                    else:
                        errores.append((idx, error))

            stop_hb.set()


        todos = [x for x in todos if x is not None]
        RUTA_RESULTADOS.write_text(
            json.dumps(todos, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        safe_print("\n" + "=" * 60)
        # DOPO:
        safe_print(f"✅ NER completado — {len(todos)} chunks procesados, {len(errores)} errores")
        safe_print("=" * 60)

        # EDC: canonicalizza i nomi non in ontologia prima della revisione manuale D/R
        safe_print("\n🔗 EDC — Canonicalizzazione relazioni non-ontologiche...")
        n_mappate, n_grigia, n_scartate = canonicalizar_post_ner(
            ruta_ner_json=str(RUTA_RESULTADOS),
            ruta_ontologia=str(RUTA_ONTOLOGIA),
            soglia=0.75,
            soglia_grigia=0.60,
            verbose=True,
        )
        safe_print(f"   Mappate: {n_mappate} | Zona grigia: {n_grigia} | Scartate: {n_scartate}")
        return True

    except Exception as e:
        safe_print(f"❌ Error NER: {e}")
        return False


# ============================================================================
# STEP 4 — NEO4J
# ============================================================================

def paso_grafo(forzar: bool = False) -> bool:
    safe_print("\n" + "=" * 60)
    safe_print("🕸️  STEP 4: GRAFO NEO4J")
    safe_print("=" * 60)

    if not RUTA_RESULTADOS.exists():
        safe_print("❌ No existe ner_resultados.json")
        return False

    try:
        from src.graph_building.entity_resolver import EntityResolver
        from src.graph_building.graph_builder import GraphBuilder
        from src.graph_building.entity_summarizer import EntitySummarizer
        from src.communities.community_detector import CommunityDetector
        from src.name_entity_recognition.relation_canonicalizer import canonicalizar_post_ner

        resolver = EntityResolver(
            embedding_threshold=0.92,
            usar_embedding_merge=True,
            ruta_acronimos=RUTA_ACRONIMOS,
            verbose=True
        )
        entity_map = resolver.resolve(str(RUTA_RESULTADOS))

        builder = GraphBuilder(
            verbose=True,
            solo_relaciones_validas=True,
            confianza_minima="medium",
            entity_map=entity_map,
        )
        builder.construir_desde_json(str(RUTA_RESULTADOS), limpiar=forzar)

        summarizer = EntitySummarizer(graph=builder.graph, verbose=True)
        summarizer.summarize(entity_map)
        summarizer.summarize_relations()

        detector = CommunityDetector(graph=builder.graph, verbose=True, min_community_size=4, gamma=0.5)
        detector.detect_and_summarize()

        builder.estadisticas()
        safe_print("🟩 Grafo finalizado en Neo4j")
        return True

    except Exception as e:
        safe_print(f"❌ Error grafo: {e}")
        return False


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Pipeline maestro GraphRAG")
    parser.add_argument("--forzar-embeddings", action="store_true")
    parser.add_argument("--forzar-ner", action="store_true")
    parser.add_argument("--forzar-grafo", action="store_true")
    parser.add_argument("--skip-neo4j", action="store_true")
    parser.add_argument("--solo-ner", action="store_true", help="Esegue solo fino a ner_resultados.json + EDC, senza costruire il grafo")
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--solo-summaries", action="store_true")
    parser.add_argument("--workers", type=int, default=1,
                        help="Número de workers paralelos (default=1, recomendado para LLM local)")

    args = parser.parse_args()

    safe_print("\n" + "=" * 60)
    safe_print("🚀 PIPELINE MAESTRO — GRAPHRAG")
    safe_print("=" * 60)

    if not paso_ingestion(forzar=args.forzar_embeddings):
        sys.exit(1)

    if not paso_ontologia():
        sys.exit(1)

    if not paso_ner(
        forzar=args.forzar_ner,
        max_chunks=args.max_chunks,
        workers=args.workers
    ):
        sys.exit(1)

    if args.solo_ner:
        safe_print("\n" + "=" * 60)
        safe_print("⏹  --solo-ner: pipeline fermata dopo NER + EDC")
        safe_print("=" * 60)
        sys.exit(0)

    if not args.skip_neo4j:
        paso_grafo(forzar=args.forzar_grafo)


    safe_print("\n" + "=" * 60)
    safe_print("🎉 PIPELINE COMPLETADO")
    safe_print("=" * 60)


if __name__ == "__main__":
    main()