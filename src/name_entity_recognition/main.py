"""
master_pipeline.py
------------------
Pipeline maestro: Markdown → Chunks/Embeddings → Ontologia → NER → Grafo Neo4j

Logica condizionale:
  1. Se manca data/processed/chunks/chunks_con_embeddings.csv
       → esegue IngestionPipeline (chunking + embeddings)

  2. Se manca data/ontology.json
       → genera borrador_ontologia.json dai documenti raw
       → avvisa l'utente di completarla come YAML e si ferma

  3. Se esiste tutto
       → esegue NER su tutti i chunk
       → scrive ner_resultados.json
       → costruisce il grafo su Neo4j

Uso rapido:
    python master_pipeline.py

Uso avanzato:
    python master_pipeline.py \\
        --raw       data/raw \\
        --processed data/processed/chunks/chunks_con_embeddings.csv \\
        --ontology  data/ontology.json \\
        --ner-out   data/ner/ner_resultados.json \\
        --neo4j-uri bolt://localhost:7687 \\
        --neo4j-user neo4j \\
        --neo4j-password secret \\
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


# ─────────────────────────────────────────────────────────────────────────────
# Percorsi di default
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_RAW        = "data/raw"
DEFAULT_CSV        = "data/processed/chunks/chunks_con_embeddings.csv"
DEFAULT_ONTOLOGY   = "data/ontology.json"
DEFAULT_NER_OUT    = "data/ner/ner_resultados.json"
DEFAULT_NEO4J_URI  = "bolt://localhost:7687"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASS = "neo4j"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – Ingestion (chunking + embeddings)
# ─────────────────────────────────────────────────────────────────────────────

def step_ingestion(
    carpeta_raw: Path,
    archivo_csv: Path,
    api_key: str,
    max_caracteres: int = 1500,
    batch_size: int = 20,
) -> pd.DataFrame:
    """
    Esegue chunking + embeddings e salva il CSV.
    Importa IngestionPipeline in modo lazy per non richiedere
    la dipendenza se il CSV esiste già.
    """
    print("\n" + "═" * 60)
    print("📦 STEP 1 — Ingestion: Markdown → Chunks + Embeddings")
    print("═" * 60)

    try:
        from ingestion.pipeline import IngestionPipeline  # type: ignore
    except ImportError as e:
        _abort(f"Impossibile importare ingestion.pipeline: {e}\n"
               "Assicurati che il modulo 'ingestion' sia nel PYTHONPATH.")

    pipeline = IngestionPipeline(
        api_key=api_key,
        max_caracteres_chunk=max_caracteres,
        batch_size=batch_size,
    )
    df = pipeline.ejecutar(
        carpeta_entrada=str(carpeta_raw),
        archivo_salida=str(archivo_csv),
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – Generazione bozza ontologia
# ─────────────────────────────────────────────────────────────────────────────

def step_genera_ontologia(
    carpeta_raw: Path,
    ruta_ontologia: Path,
    api_key: str,
) -> None:
    """
    Legge tutti i .md in carpeta_raw, chiede a GPT una bozza di ontologia
    e la salva come borrador_ontologia.json.
    Avvisa l'utente che deve finalizzarla come YAML.
    """
    print("\n" + "═" * 60)
    print("🧬 STEP 2 — Generazione bozza ontologia")
    print("═" * 60)

    archivos = list(carpeta_raw.glob("**/*.md"))
    if not archivos:
        _abort(f"Nessun file .md trovato in {carpeta_raw}")

    testo_concatenato = "\n\n".join(
        a.read_text(encoding="utf-8") for a in archivos
    )

    print(f"  📂 {len(archivos)} file Markdown letti")
    print("  🤖 Chiamo il modello per estrarre la bozza di ontologia…")

    prompt = f"""Sei un knowledge engineer esperto.
Analizza i seguenti documenti Markdown e genera una bozza di ontologia per un Knowledge Graph in formato JSON.

L'ontologia deve contenere:
- "entidades": lista di tipi di entità (es. Persona, Organizzazione, Luogo, Concetto…)
  Ogni tipo ha: "nombre", "descripcion", "atributos" (lista di attributi chiave)
- "relaciones": lista di relazioni tra entità
  Ogni relazione ha: "nombre", "origen", "destino", "descripcion"
- "notas": eventuali note o decisioni di design

Rispondi SOLO con JSON valido, senza testo aggiuntivo né backtick.

DOCUMENTI:
\"\"\"
{testo_concatenato[:12000]}
\"\"\"
"""

    import urllib.request, urllib.error  # stdlib per non aggiungere dipendenze

    payload = json.dumps({
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }).encode()

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        _abort(f"Errore API OpenAI: {e.code} {e.reason}\n{e.read().decode()}")

    contenuto = data["choices"][0]["message"]["content"]

    # Pulizia robusta del JSON
    contenuto = contenuto.strip()
    if contenuto.startswith("```"):
        contenuto = contenuto.split("```", 2)[1]
        if contenuto.startswith("json"):
            contenuto = contenuto[4:]
        contenuto = contenuto.rsplit("```", 1)[0].strip()

    try:
        ontologia = json.loads(contenuto)
    except json.JSONDecodeError as e:
        _abort(f"Il modello non ha restituito JSON valido: {e}\nOutput:\n{contenuto[:500]}")

    # Salva bozza
    borrador = ruta_ontologia.parent / "borrador_ontologia.json"
    borrador.parent.mkdir(parents=True, exist_ok=True)
    borrador.write_text(json.dumps(ontologia, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  ✅ Bozza ontologia salvata: {borrador}")
    print()
    print("  ⚠️  ATTENZIONE — Azione manuale richiesta:")
    print("  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  1. Rivedi e modifica:  {borrador.name:<27}│")
    print("  │  2. Converti in YAML → data/ontology.yaml           │")
    print(f"  │  3. Crea anche:         {ruta_ontologia.name:<27}│")
    print("  │     (copia/rinomina il JSON finalizzato)             │")
    print("  │  4. Rilancia master_pipeline.py                     │")
    print("  └─────────────────────────────────────────────────────┘")
    print()
    sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – NER
# ─────────────────────────────────────────────────────────────────────────────

def step_ner(
    ruta_csv: Path,
    ruta_ontologia: Path,
    carpeta_raw: Path,
    ruta_salida: Path,
    max_chunks: Optional[int] = None,
) -> List[Dict]:
    """
    Carica i chunk, esegue NER e salva ner_resultados.json + ner_entidades.csv.
    """
    print("\n" + "═" * 60)
    print("🔍 STEP 3 — NER: estrazione entità e relazioni")
    print("═" * 60)

    try:
        from ner.cargador_chunks import CargadorChunksCSV   # type: ignore
        from ner.pipeline import PipelineNERDosPasos        # type: ignore
    except ImportError as e:
        _abort(f"Impossibile importare ner.*: {e}\n"
               "Assicurati che il modulo 'ner' sia nel PYTHONPATH.")

    # Vocabolario: concatena tutti gli MD raw in un file temporaneo
    archivos_md = list(carpeta_raw.glob("**/*.md"))
    temp_vocab  = ruta_salida.parent / "_temp_vocab.md"
    temp_vocab.parent.mkdir(parents=True, exist_ok=True)
    temp_vocab.write_text(
        "\n\n".join(a.read_text(encoding="utf-8") for a in archivos_md),
        encoding="utf-8",
    )

    # Carica chunks
    cargador = CargadorChunksCSV(str(ruta_csv))
    cargador.resumen()
    chunks = cargador.cargar_chunks(max_chunks=max_chunks)

    if not chunks:
        _abort("Nessun chunk caricato. Verifica il CSV.")

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

    # Pulizia
    temp_vocab.unlink(missing_ok=True)

    # Salva JSON
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    ruta_salida.write_text(
        json.dumps(todos, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  📄 NER completato: {ruta_salida} ({len(todos)} chunk)")

    # Salva CSV entidades
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
        print(f"  📊 CSV entità: {csv_ent} ({len(filas)} righe)")

    return todos


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – Costruzione grafo Neo4j
# ─────────────────────────────────────────────────────────────────────────────

def step_neo4j(
    ner_resultados: List[Dict],
    ruta_csv: Path,
    ruta_ontologia: Path,
    uri: str,
    user: str,
    password: str,
) -> None:
    """
    Costruisce il grafo Neo4j a partire dai risultati NER.

    Struttura del grafo:
      (:Documento {id, titulo, tipo, fuente})
      (:Chunk     {id, texto, titulo_seccion, caracteres, palabras, pagina})
      (:Entidad   {texto, tipo, confianza_promedio})
      (:TipoEntidad {nombre})

      (:Documento)-[:TIENE_CHUNK]->(:Chunk)
      (:Chunk)-[:MENCIONA {confianza}]->(:Entidad)
      (:Entidad)-[:ES_DE_TIPO]->(:TipoEntidad)
      (relaciones NER opzionali)-[:RELATED_TO]->
    """
    print("\n" + "═" * 60)
    print("🕸️  STEP 4 — Costruzione grafo Neo4j")
    print("═" * 60)

    try:
        from neo4j import GraphDatabase  # type: ignore
    except ImportError:
        _abort("neo4j non installato. Esegui: pip install neo4j")

    # Carica ontologia per etichette canoniche
    ontologia: Dict = {}
    if ruta_ontologia.exists():
        try:
            ontologia = json.loads(ruta_ontologia.read_text(encoding="utf-8"))
        except Exception:
            print("  ⚠️  Ontologia non leggibile, si procede senza.")

    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        # ── Vincoli/indici ────────────────────────────────────────────────
        _run(session, "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Documento) REQUIRE d.id IS UNIQUE")
        _run(session, "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
        _run(session, "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entidad) REQUIRE (e.texto, e.tipo) IS NODE KEY")
        _run(session, "CREATE CONSTRAINT IF NOT EXISTS FOR (t:TipoEntidad) REQUIRE t.nombre IS UNIQUE")

        # ── Tipi entità dall'ontologia ────────────────────────────────────
        for tipo in ontologia.get("entidades", []):
            _run(session,
                 "MERGE (t:TipoEntidad {nombre: $nombre}) "
                 "SET t.descripcion = $desc",
                 nombre=tipo.get("nombre", ""),
                 desc=tipo.get("descripcion", ""))

        # ── Carica metadati chunk dal CSV ─────────────────────────────────
        df_chunks: Dict[str, Dict] = {}
        if ruta_csv.exists():
            df = pd.read_csv(ruta_csv)
            for _, row in df.iterrows():
                df_chunks[str(row.get("id_chunk", row.get("chunk_id", "")))] = row.to_dict()

        documentos_creati: set = set()

        for resultado in ner_resultados:
            chunk_id = resultado.get("chunk_id", "")
            titulo   = resultado.get("titulo", "sin_titulo")
            tipo_doc = resultado.get("tipo", "Desconocido")
            fuente   = resultado.get("fuente", "")

            # Documento (deduplica per titolo)
            doc_id = _slugify(titulo)
            if doc_id not in documentos_creati:
                _run(session,
                     "MERGE (d:Documento {id: $id}) "
                     "SET d.titulo = $titulo, d.tipo = $tipo, d.fuente = $fuente",
                     id=doc_id, titulo=titulo, tipo=tipo_doc, fuente=fuente)
                documentos_creati.add(doc_id)

            # Chunk
            meta = df_chunks.get(chunk_id, {})
            _run(session,
                 "MERGE (c:Chunk {id: $id}) "
                 "SET c.texto          = $texto, "
                 "    c.titulo_seccion = $ts, "
                 "    c.caracteres     = $chars, "
                 "    c.palabras       = $words, "
                 "    c.pagina         = $pagina",
                 id=chunk_id,
                 texto=resultado.get("texto_original", meta.get("texto", ""))[:2000],
                 ts=meta.get("titulo_seccion", ""),
                 chars=int(meta.get("caracteres", 0)),
                 words=int(meta.get("palabras", 0)),
                 pagina=int(meta.get("pagina", 0)))

            # Documento → Chunk
            _run(session,
                 "MATCH (d:Documento {id: $did}), (c:Chunk {id: $cid}) "
                 "MERGE (d)-[:TIENE_CHUNK]->(c)",
                 did=doc_id, cid=chunk_id)

            # Entità
            for entidad in resultado.get("entidades", []):
                texto_e  = entidad.get("text", "").strip()
                tipo_e   = entidad.get("entity_type", "UNKNOWN")
                conf     = float(entidad.get("confidence", 0.0))
                if not texto_e:
                    continue

                _run(session,
                     "MERGE (e:Entidad {texto: $texto, tipo: $tipo}) "
                     "ON CREATE SET e.confianza_promedio = $conf "
                     "ON MATCH  SET e.confianza_promedio = "
                     "  round((e.confianza_promedio + $conf) / 2.0, 4)",
                     texto=texto_e, tipo=tipo_e, conf=conf)

                # Chunk → Entità
                _run(session,
                     "MATCH (c:Chunk {id: $cid}), (e:Entidad {texto: $texto, tipo: $tipo}) "
                     "MERGE (c)-[r:MENCIONA]->(e) "
                     "SET r.confianza = $conf",
                     cid=chunk_id, texto=texto_e, tipo=tipo_e, conf=conf)

                # Entità → TipoEntidad
                _run(session,
                     "MERGE (t:TipoEntidad {nombre: $tipo}) "
                     "WITH t "
                     "MATCH (e:Entidad {texto: $texto, tipo: $tipo}) "
                     "MERGE (e)-[:ES_DE_TIPO]->(t)",
                     tipo=tipo_e, texto=texto_e)

            # Relazioni NER (se presenti)
            for rel in resultado.get("relaciones", []):
                origen  = rel.get("origen", "").strip()
                destino = rel.get("destino", "").strip()
                tipo_r  = _safe_rel_type(rel.get("tipo", "RELATED_TO"))
                if not origen or not destino:
                    continue
                cypher = (
                    f"MATCH (a:Entidad {{texto: $orig}}), (b:Entidad {{texto: $dest}}) "
                    f"MERGE (a)-[:{tipo_r}]->(b)"
                )
                _run(session, cypher, orig=origen, dest=destino)

        # ── Statistiche finali ────────────────────────────────────────────
        stats = {
            "Documenti":  session.run("MATCH (d:Documento) RETURN count(d) AS n").single()["n"],
            "Chunk":      session.run("MATCH (c:Chunk)     RETURN count(c) AS n").single()["n"],
            "Entità":     session.run("MATCH (e:Entidad)   RETURN count(e) AS n").single()["n"],
            "TipiEntità": session.run("MATCH (t:TipoEntidad) RETURN count(t) AS n").single()["n"],
        }

    driver.close()

    print("\n  🕸️  Grafo Neo4j costruito con successo!")
    print(f"  URI: {uri}")
    for label, n in stats.items():
        print(f"     {label:<12}: {n}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run(session: Any, cypher: str, **params) -> Any:
    """Esegue una query Cypher con gestione errori silenziosa."""
    try:
        return session.run(cypher, **params)
    except Exception as e:
        print(f"  ⚠️  Cypher error: {e}\n     Query: {cypher[:80]}")


def _slugify(testo: str) -> str:
    import re
    return re.sub(r"[^a-z0-9_]", "_", testo.lower())[:80]


def _safe_rel_type(tipo: str) -> str:
    import re
    return re.sub(r"[^A-Z0-9_]", "_", tipo.upper())[:50] or "RELATED_TO"


def _abort(msg: str) -> None:
    print(f"\n❌ ERRORE: {msg}")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestratore principale
# ─────────────────────────────────────────────────────────────────────────────

def run(
    carpeta_raw: Path,
    ruta_csv: Path,
    ruta_ontologia: Path,
    ruta_ner: Path,
    api_key: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
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
    print(f"  Ontologia  : {ruta_ontologia}")
    print(f"  NER output : {ruta_ner}")
    print(f"  Neo4j URI  : {neo4j_uri}")

    # ── STEP 1: Ingestion ─────────────────────────────────────────────────────
    if not ruta_csv.exists():
        print(f"\n  ℹ️  CSV non trovato ({ruta_csv})")
        if not api_key:
            _abort("API key OpenAI richiesta per generare embeddings.\n"
                   "Passa --api-key oppure imposta OPENAI_API_KEY nell'ambiente.")
        step_ingestion(
            carpeta_raw=carpeta_raw,
            archivo_csv=ruta_csv,
            api_key=api_key,
            max_caracteres=max_caracteres,
            batch_size=batch_size,
        )
    else:
        print(f"\n  ✅ STEP 1 — CSV già presente: {ruta_csv}")

    # ── STEP 2: Ontologia ─────────────────────────────────────────────────────
    if not ruta_ontologia.exists():
        print(f"\n  ℹ️  Ontologia non trovata ({ruta_ontologia})")
        if not api_key:
            _abort("API key OpenAI richiesta per generare la bozza ontologia.")
        step_genera_ontologia(
            carpeta_raw=carpeta_raw,
            ruta_ontologia=ruta_ontologia,
            api_key=api_key,
        )
        # step_genera_ontologia fa sys.exit(0) dopo aver stampato le istruzioni
    else:
        print(f"  ✅ STEP 2 — Ontologia trovata: {ruta_ontologia}")

    # ── STEP 3: NER ───────────────────────────────────────────────────────────
    if not ruta_ner.exists():
        print(f"\n  ℹ️  NER output non trovato ({ruta_ner}), avvio NER…")
        ner_resultados = step_ner(
            ruta_csv=ruta_csv,
            ruta_ontologia=ruta_ontologia,
            carpeta_raw=carpeta_raw,
            ruta_salida=ruta_ner,
            max_chunks=max_chunks,
        )
    else:
        print(f"  ✅ STEP 3 — NER già eseguito: {ruta_ner}")
        ner_resultados = json.loads(ruta_ner.read_text(encoding="utf-8"))

    # ── STEP 4: Neo4j ─────────────────────────────────────────────────────────
    if skip_neo4j:
        print("\n  ⏭️  STEP 4 — Neo4j saltato (--skip-neo4j)")
    else:
        step_neo4j(
            ner_resultados=ner_resultados,
            ruta_csv=ruta_csv,
            ruta_ontologia=ruta_ontologia,
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
        )

    print("\n" + "═" * 60)
    print("🎉 Pipeline completata con successo!")
    print("═" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Master pipeline: Markdown → Chunks → NER → Grafo Neo4j",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw",          default=DEFAULT_RAW,        help="Cartella documenti Markdown")
    p.add_argument("--processed",    default=DEFAULT_CSV,        help="CSV chunks+embeddings")
    p.add_argument("--ontology",     default=DEFAULT_ONTOLOGY,   help="Ontologia JSON finalizzata")
    p.add_argument("--ner-out",      default=DEFAULT_NER_OUT,    help="Output NER JSON")
    p.add_argument("--api-key",      default=None,               help="Chiave OpenAI")
    p.add_argument("--neo4j-uri",    default=DEFAULT_NEO4J_URI,  help="URI Neo4j")
    p.add_argument("--neo4j-user",   default=DEFAULT_NEO4J_USER, help="Utente Neo4j")
    p.add_argument("--neo4j-password", default=DEFAULT_NEO4J_PASS, help="Password Neo4j")
    p.add_argument("--max-chunks",   type=int, default=None,     help="Limite chunk per NER")
    p.add_argument("--max-caracteres", type=int, default=1500,   help="Max caratteri per chunk")
    p.add_argument("--batch-size",   type=int, default=20,       help="Batch size embeddings")
    p.add_argument("--skip-neo4j",   action="store_true",        help="Salta costruzione grafo")
    p.add_argument("--reset-ner",    action="store_true",        help="Forza riesecuzione NER")
    p.add_argument("--reset-all",    action="store_true",        help="Forza riesecuzione completa")
    return p


def main():
    load_dotenv()
    args = _build_parser().parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")

    ruta_csv       = Path(args.processed)
    ruta_ontologia = Path(args.ontology)
    ruta_ner       = Path(args.ner_out)

    # Reset selettivi
    if args.reset_all:
        for p in [ruta_csv, ruta_ner]:
            p.unlink(missing_ok=True)
        print("♻️  Reset completo: CSV e NER rimossi.")
    elif args.reset_ner:
        ruta_ner.unlink(missing_ok=True)
        print("♻️  Reset NER: ner_resultados.json rimosso.")

    run(
        carpeta_raw    = Path(args.raw),
        ruta_csv       = ruta_csv,
        ruta_ontologia = ruta_ontologia,
        ruta_ner       = ruta_ner,
        api_key        = api_key,
        neo4j_uri      = args.neo4j_uri,
        neo4j_user     = args.neo4j_user,
        neo4j_password = args.neo4j_password,
        max_chunks     = args.max_chunks,
        max_caracteres = args.max_caracteres,
        batch_size     = args.batch_size,
        skip_neo4j     = args.skip_neo4j,
    )


if __name__ == "__main__":
    main()