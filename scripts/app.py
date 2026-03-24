"""
ner_debug/app.py
----------------
App di debug per il pipeline GraphRAG NER.
Versión con explicaciones en español para tribunal/relatori.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import secrets
import uvicorn
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, JSONResponse
from pydantic import BaseModel

from src.name_entity_recognition.pipeline import PipelineNERDosPasos
from src.graph_building.entity_resolver import (
    EntityResolver, EntityInstance, CanonicalEntity, EntityMap
)
from src.graph_building.entity_summarizer import EntitySummarizer

load_dotenv()

ONTOLOGY_PATH  = os.getenv("ONTOLOGY_PATH", "../data/ner/ontologia/ontology.yaml")
ACRONIMOS_PATH = os.getenv("ACRONIMOS_PATH", "../data/acronimos.yaml")
LLM_BASE_URL   = os.getenv("LLM_BASE_URL")
LLM_API_KEY    = os.getenv("LLM_API_KEY", "no-key")
MODEL_PASO1    = os.getenv("MODEL_PASO1", "gpt-4o-mini")
MODEL_PASO2    = os.getenv("MODEL_PASO2", "gpt-4o-mini")

print(f"ONTOLOGY_PATH : {ONTOLOGY_PATH}")
print(f"ACRONIMOS_PATH: {ACRONIMOS_PATH}")

# ── Autenticazione ───────────────────────────────────────────────────────────
# Imposta DEMO_PASSWORD nel .env oppure viene generata casualmente all'avvio.
DEMO_PASSWORD = os.getenv("DEMO_PASSWORD", secrets.token_urlsafe(8))

app = FastAPI()

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """
    Protezione semplice tramite query param ?pw=PASSWORD.
    Il browser la ricorda nell'URL finché la sessione è aperta.
    Rotte libere: nessuna (tutto protetto).
    """
    # Permetti OPTIONS (preflight CORS) senza auth
    if request.method == "OPTIONS":
        return await call_next(request)
    pw = request.query_params.get("pw", "")
    if not secrets.compare_digest(pw, DEMO_PASSWORD):
        return JSONResponse(
            status_code=401,
            content={"detail": "Acceso denegado. Añade ?pw=PASSWORD a la URL."},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await call_next(request)

ner_pipeline = PipelineNERDosPasos(ruta_ontologia=ONTOLOGY_PATH)
_ultimo_risultato: list = []

DEFAULT_CHUNKS = [
    # Chunk 1 — variantes léxicas de la misma entidad
    """El Vicerrectorado de Ordenación Académica de la ULL ha publicado la instrucción reguladora del proceso de matrícula para las enseñanzas oficiales de Grado. Dicha instrucción fue remitida por el Vicerrectorado de Ordenación Académica y Espacio Europeo de Educación Superior, que actúa por delegación del Rector de la Universidad de La Laguna. Los estudios oficiales de Grado y los estudios de Grado oficial quedan sujetos a las mismas condiciones de admisión.""",
    # Chunk 2 — decretos con números similares
    """La matrícula de los estudiantes admitidos en virtud del Real Decreto 534/2024 se regirá por las mismas normas generales que la matrícula ordinaria. Sin embargo, los admitidos al amparo del Real Decreto 534/2023 deberán acreditar documentación adicional según la resolución del Consejo de Gobierno de 15 de junio de 2023. La convocatoria ordinaria y la convocatoria extraordinaria tienen plazos distintos e incompatibles entre sí.""",
    # Chunk 3 — misma persona, roles distintos
    """El Rector de la Universidad de La Laguna, en su condición de máxima autoridad académica, delegó en el Vicerrector de Estudiantes la resolución de los expedientes de matrícula. El Vicerrector de Estudiantes, actuando por delegación rectoral, comunicó al Servicio de Gestión Académica y al Servicio de Alumnado las instrucciones oportunas. El Servicio de Alumnado y el Servicio de Gestión de Alumnado tramitarán los expedientes según el procedimiento establecido en el Reglamento de Matrícula aprobado por el Consejo de Gobierno.""",
]

DEFAULT_CHUNK_LABELS = [
    "Chunk 1 — Variantes léxicas de la misma entidad",
    "Chunk 2 — Decretos normativos con números similares",
    "Chunk 3 — Misma persona, roles distintos",
]

DEFAULT_CHUNK_HINTS = [
    "Prueba: ¿fusiona ULL↔Universidad de La Laguna? ¿Distingue los dos Vicerrectorados? ¿Une 'estudios oficiales de Grado' y 'enseñanzas oficiales de Grado'?",
    "Prueba: ¿separa RD 534/2024 de RD 534/2023? ¿No fusiona 'convocatoria ordinaria' y 'extraordinaria'?",
    "Prueba: ¿separa 'Servicio de Alumnado' de 'Servicio de Gestión de Alumnado'? ¿No confunde Rector con Vicerrector?",
]


def load_ontology(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def get_entity_types(onto: dict) -> List[str]:
    if "entities" in onto and isinstance(onto["entities"], dict):
        return list(onto["entities"].keys())
    return []

def get_relation_types(onto: dict) -> List[str]:
    if "relations" in onto and isinstance(onto["relations"], dict):
        return list(onto["relations"].keys())
    return []

def paso1_span_detection(texto: str) -> List[dict]:
    return ner_pipeline.ejecutar_paso1(texto)

def paso2_classify(texto: str, spans: List[dict]) -> Tuple[List[dict], List[dict]]:
    entidades, relaciones = ner_pipeline.ejecutar_paso2(texto, spans)
    relaciones_out = []
    for r in relaciones:
        relaciones_out.append({
            "sujeto":   r.get("sujeto", ""),
            "relacion": r.get("relacion", ""),
            "objeto":   r.get("objeto", ""),
            "valida":   r.get("valida", False),
            "motivo":   r.get("motivo", ""),
        })
    return entidades, relaciones_out

def paso3_entity_resolver(chunks_entidades: List[Tuple[str, List[dict]]]) -> Tuple[List[dict], List[dict]]:
    resolver = EntityResolver(
        embedding_threshold=0.92,
        usar_embedding_merge=True,
        ruta_acronimos=Path(ACRONIMOS_PATH),
        verbose=False,
    )
    instances = []
    for chunk_id, entidades in chunks_entidades:
        for e in entidades:
            if e.get("text", "").strip():
                instances.append(EntityInstance(
                    text        = e.get("text", ""),
                    entity_type = e.get("entity_type", ""),
                    chunk_id    = chunk_id,
                    confidence  = e.get("confidence", "medium"),
                    descripcion = e.get("descripcion", ""),
                ))

    entity_map = resolver._exact_merge(instances)
    resolver._abbreviation_merge(entity_map)
    if resolver.usar_embedding_merge:
        ruta_cache = Path(ACRONIMOS_PATH).parent / "debug_embedding_cache.json"
        resolver._embedding_merge(entity_map, ruta_cache)

    canonical_entities = []
    merge_log = []
    for ce in entity_map.all_canonicals():
        all_texts = ce.all_texts
        canonical_entities.append({
            "canonical_id":   ce.canonical_id,
            "canonical_text": ce.canonical_text,
            "entity_type":    ce.entity_type,
            "n_instances":    len(ce.instances),
            "n_chunks":       ce.n_chunks,
            "descriptions":   ce.descriptions,
            "variants":       all_texts,
        })
        if len(all_texts) > 1 or ce.n_chunks > 1:
            merge_log.append({
                "canonical_text": ce.canonical_text,
                "entity_type":    ce.entity_type,
                "variants":       all_texts,
                "n_instances":    len(ce.instances),
                "n_chunks":       ce.n_chunks,
            })
    return canonical_entities, merge_log

def paso4_summarize(canonical_entities: List[dict]) -> List[dict]:
    entity_map = EntityMap()
    for ce_dict in canonical_entities:
        ce = CanonicalEntity(
            canonical_id   = ce_dict["canonical_id"],
            canonical_text = ce_dict["canonical_text"],
            entity_type    = ce_dict["entity_type"],
            descriptions   = ce_dict.get("descriptions", []),
        )
        entity_map.register(ce_dict["canonical_text"].lower(), ce)

    summarizer = EntitySummarizer(graph=None, verbose=False)
    results = []
    for ce in entity_map.all_canonicals():
        descs = [d for d in ce.descriptions if d.strip()]
        if len(descs) <= 1:
            ce.descripcion_consolidada = descs[0] if descs else ""
            used_llm = False
        else:
            ce.descripcion_consolidada = summarizer._consolidar(ce)
            used_llm = True
        results.append({
            "canonical_id":            ce.canonical_id,
            "canonical_text":          ce.canonical_text,
            "entity_type":             ce.entity_type,
            "n_instances":             len(ce.instances),
            "descriptions":            ce.descriptions,
            "variants":                ce.all_texts,
            "descripcion_consolidada": ce.descripcion_consolidada,
            "used_llm":                used_llm,
        })
    return results


class RunRequest(BaseModel):
    chunks: List[str]
    ontology_path: str = ONTOLOGY_PATH

@app.post("/run")
def run_pipeline(req: RunRequest):
    global _ultimo_risultato
    onto = load_ontology(req.ontology_path)
    t0 = time.time()

    chunk_results = []
    chunks_entidades = []

    for i, texto in enumerate(req.chunks):
        texto = texto.strip()
        if not texto:
            continue
        chunk_id = f"debug_chunk_{i+1}"

        t1s = time.time()
        spans = paso1_span_detection(texto)
        t1e = time.time()

        t2s = time.time()
        entidades, relaciones = paso2_classify(texto, spans)
        t2e = time.time()

        chunk_results.append({
            "chunk_id": chunk_id,
            "texto":    texto,
            "timings":  {"paso1_ms": round((t1e-t1s)*1000), "paso2_ms": round((t2e-t2s)*1000)},
            "paso1":    {"spans": spans},
            "paso2":    {"entidades": entidades, "relaciones": relaciones},
        })
        chunks_entidades.append((chunk_id, entidades))

    t3s = time.time()
    canonical_entities, merge_log = paso3_entity_resolver(chunks_entidades)
    t3e = time.time()

    t4s = time.time()
    summarized = paso4_summarize(canonical_entities)
    t4e = time.time()

    total_ms = round((t4e - t0) * 1000)

    _ultimo_risultato = [
        {
            "chunk_id":    cr["chunk_id"],
            "texto":       cr["texto"],
            "spans_paso1": cr["paso1"]["spans"],
            "entidades":   cr["paso2"]["entidades"],
            "relaciones":  cr["paso2"]["relaciones"],
        }
        for cr in chunk_results
    ]

    return {
        "total_ms":      total_ms,
        "n_chunks":      len(chunk_results),
        "chunk_results": chunk_results,
        "paso3": {
            "canonical_entities": canonical_entities,
            "merge_log":          merge_log,
            "timings":            {"paso3_ms": round((t3e-t3s)*1000)},
        },
        "paso4": {
            "summarized": summarized,
            "timings":    {"paso4_ms": round((t4e-t4s)*1000)},
        },
        "ontology_info": {
            "path":           req.ontology_path,
            "entity_types":   get_entity_types(onto),
            "relation_types": get_relation_types(onto),
        },
    }

@app.get("/download-resultados")
def download_resultados():
    if not _ultimo_risultato:
        raise HTTPException(status_code=404, detail="Sin resultados — ejecute primero el pipeline")
    content = json.dumps(_ultimo_risultato, ensure_ascii=False, indent=2)
    return Response(
        content=content.encode("utf-8"),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=ner_resultados.json"},
    )

@app.get("/ontology-info")
def ontology_info(path: str = ONTOLOGY_PATH):
    onto = load_ontology(path)
    return {
        "path":           path,
        "entity_types":   get_entity_types(onto),
        "relation_types": get_relation_types(onto),
        "raw_keys":       list(onto.keys()),
    }

@app.get("/default-chunks")
def get_default_chunks():
    return {
        "chunks": DEFAULT_CHUNKS,
        "labels": DEFAULT_CHUNK_LABELS,
        "hints":  DEFAULT_CHUNK_HINTS,
    }


HTML = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GraphRAG NER — Demo ULL</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
  --bg:      #07090f;
  --bg2:     #0d1117;
  --bg3:     #111820;
  --bg4:     #181f2c;
  --border:  #1e2c40;
  --border2: #28405a;
  --text:    #d0e2f8;
  --muted:   #5a7898;
  --muted2:  #384e68;
  --accent:  #5a9cf8;
  --green:   #28cc7a;
  --amber:   #f0a830;
  --red:     #e05870;
  --pink:    #cc6af0;
  --indigo:  #8878f6;
  --r: 9px;
  --mono: 'IBM Plex Mono', monospace;
  --sans: 'IBM Plex Sans', system-ui, sans-serif;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: var(--sans); font-size: 14px; line-height: 1.65; }

/* ── Header ── */
.hdr {
  padding: 16px 24px; border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 14px;
  background: #080b14;
}
.hdr-logo {
  width: 36px; height: 36px; border-radius: 9px; flex-shrink: 0;
  background: linear-gradient(135deg, var(--accent), var(--indigo));
  display: flex; align-items: center; justify-content: center; font-size: 17px;
}
.hdr h1 { font-size: 16px; font-weight: 600; color: #e4f0ff; }
.hdr .sub { font-size: 11px; color: var(--muted); }
.hdr-badge {
  margin-left: auto; background: var(--bg3); border: 1px solid var(--border2);
  border-radius: 20px; padding: 4px 13px; font-size: 10px;
  font-family: var(--mono); color: #3ddbc8; letter-spacing: .06em;
}

/* ── Layout: sidebar + main ── */
.layout {
  display: grid;
  grid-template-columns: 370px 1fr;
  height: calc(100vh - 69px);
}
.sidebar {
  border-right: 1px solid var(--border);
  background: var(--bg2);
  display: flex; flex-direction: column; overflow: hidden;
}
.sidebar-inner {
  flex: 1; overflow-y: auto; padding: 16px;
  display: flex; flex-direction: column; gap: 14px;
}
.main { display: flex; flex-direction: column; overflow: hidden; }

/* ── Sidebar: banner, steps, form ── */
.banner {
  background: #0b0d1e; border: 1px solid #1c2040;
  border-left: 3px solid #505898; border-radius: var(--r);
  padding: 13px 15px; font-size: 12.5px; color: #6878a0; line-height: 1.7;
}
.banner strong { color: #a8b8d8; }
.banner-title { font-size: 10px; font-weight: 700; color: #8890c0; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 7px; }

.steps {
  display: grid; grid-template-columns: repeat(4, 1fr);
  border: 1px solid var(--border); border-radius: var(--r); overflow: hidden;
}
.step {
  padding: 8px 4px; text-align: center; border-right: 1px solid var(--border);
  font-size: 10px; color: var(--muted); line-height: 1.3;
}
.step:last-child { border-right: none; }
.step .n { font-family: var(--mono); font-weight: 700; font-size: 14px; display: block; }
.s1 .n{color:var(--accent)} .s2 .n{color:var(--green)} .s3 .n{color:var(--amber)} .s4 .n{color:var(--pink)}

label.lbl { display: block; font-size: 10px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: .07em; margin-bottom: 5px; }
input[type=text] {
  width: 100%; background: var(--bg3); border: 1px solid var(--border);
  border-radius: var(--r); color: var(--text); font-size: 12px;
  padding: 9px 12px; outline: none; font-family: var(--mono); transition: border-color .15s;
}
input[type=text]:focus { border-color: var(--accent); }
.onto-box { background: var(--bg3); border: 1px solid var(--border); border-radius: var(--r); padding: 9px 12px; margin-top: 7px; }
.pills { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }
.pill { border-radius: 20px; padding: 2px 8px; font-size: 10px; font-family: var(--mono); }
.pill.t { background: #0d1838; border: 1px solid #1a3268; color: #7aa8f0; }
.pill.r { background: #0d2218; border: 1px solid #1a4428; color: #44cc88; }

.chunk-list { display: flex; flex-direction: column; gap: 11px; }
.chunk-item { display: flex; flex-direction: column; gap: 4px; }
.chunk-hdr { display: flex; align-items: center; gap: 7px; }
.chunk-num { font-size: 10px; font-weight: 700; color: var(--accent); font-family: var(--mono); }
.chunk-lbl { font-size: 11px; color: var(--text); flex: 1; }
.rm-btn { background: none; border: none; color: var(--muted2); cursor: pointer; font-size: 13px; transition: color .15s; }
.rm-btn:hover { color: var(--red); }
.chunk-hint {
  font-size: 10.5px; color: var(--muted); font-style: italic;
  padding: 5px 9px; background: var(--bg4); border-radius: 5px;
  border-left: 2px solid var(--border2); line-height: 1.5;
}
textarea {
  width: 100%; background: var(--bg3); border: 1px solid var(--border);
  border-radius: var(--r); color: var(--text); font-size: 11.5px;
  padding: 9px 11px; resize: vertical; outline: none;
  font-family: var(--mono); line-height: 1.5; min-height: 90px; transition: border-color .15s;
}
textarea:focus { border-color: var(--accent); }

.add-btn {
  width: 100%; padding: 9px; background: var(--bg3); color: var(--muted);
  font-size: 12px; border: 1px dashed var(--border2); border-radius: var(--r);
  cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 6px;
  transition: all .15s; font-family: var(--sans);
}
.add-btn:hover { border-color: var(--accent); color: var(--accent); }
.run-btn {
  width: 100%; padding: 12px; background: var(--accent); color: #fff;
  font-weight: 600; font-size: 14px; border: none; border-radius: var(--r);
  cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 8px;
  transition: background .15s, opacity .15s; font-family: var(--sans);
}
.run-btn:hover { background: #6aaaff; }
.run-btn:disabled { opacity: .4; cursor: not-allowed; }
.dl-btn {
  width: 100%; padding: 10px; background: var(--bg3); color: var(--muted);
  font-weight: 600; font-size: 12px; border: 1px solid var(--border); border-radius: var(--r);
  cursor: pointer; display: none; align-items: center; justify-content: center; gap: 7px;
  transition: all .15s; font-family: var(--sans);
}
.dl-btn:hover { background: var(--bg4); color: var(--text); }

/* ── Main: summary bar + nav + content ── */
.stat-bar {
  background: #0d1520; border-bottom: 1px solid var(--border);
  padding: 10px 20px; display: flex; gap: 18px; align-items: center; flex-shrink: 0;
}
.stat { display: flex; flex-direction: column; align-items: center; }
.stat-v { font-size: 20px; font-weight: 700; font-family: var(--mono); color: var(--amber); line-height: 1.1; }
.stat-l { font-size: 10px; color: var(--muted); }
.stat-sep { width: 1px; height: 28px; background: var(--border); }
.t-pill { margin-left: auto; background: var(--bg3); border: 1px solid var(--border); border-radius: 20px; padding: 3px 11px; font-size: 10px; font-family: var(--mono); color: var(--muted); }

/* ── Carosello nav ── */
.car-nav {
  display: flex; align-items: stretch; background: #0a1018;
  border-bottom: 2px solid var(--border); flex-shrink: 0; height: 56px;
}
.car-arr {
  width: 54px; border: none; background: none; color: var(--muted);
  font-size: 22px; cursor: pointer; transition: color .15s, background .15s;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.car-arr:hover:not(:disabled) { color: var(--text); background: var(--bg3); }
.car-arr:disabled { opacity: .2; cursor: default; }
.car-arr.L { border-right: 1px solid var(--border); }
.car-arr.R { border-left:  1px solid var(--border); }
.car-mid {
  flex: 1; display: flex; align-items: center; gap: 12px; padding: 0 16px; min-width: 0;
}
.car-label { font-size: 13px; font-weight: 600; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex: 1; min-width: 0; }
.car-dots { display: flex; gap: 6px; align-items: center; flex-shrink: 0; }
.car-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--border2); cursor: pointer; transition: all .2s; }
.car-dot.on { width: 24px; border-radius: 4px; background: var(--accent); }
.car-cnt { font-size: 11px; font-family: var(--mono); color: var(--muted); flex-shrink: 0; }

/* ── Slide content ── */
.car-content {
  flex: 1; overflow-y: auto; padding: 20px 24px;
  display: flex; flex-direction: column; gap: 16px;
}

/* ── Pannelli spiegazione (indigo scuro) ── */
.explain {
  background: #0c0f22; border: 1px solid #1c2248;
  border-left: 4px solid #5060a8; border-radius: var(--r);
  padding: 13px 16px; font-size: 13px; color: #7882aa; line-height: 1.75;
}
.explain .etitle {
  font-size: 11px; font-weight: 700; color: #9298c8; text-transform: uppercase;
  letter-spacing: .08em; margin-bottom: 8px;
}
.explain ul { padding-left: 18px; }
.explain li { margin-bottom: 4px; }
.explain strong { color: #b8c4e0; }

/* ── Pannelli risultato (verde scuro) ── */
.result {
  background: #061610; border: 1px solid #103a1e;
  border-left: 4px solid var(--green); border-radius: var(--r);
  padding: 13px 16px; font-size: 13px; color: #38a868; line-height: 1.75;
}
.result .etitle { font-size: 11px; font-weight: 700; color: var(--green); text-transform: uppercase; letter-spacing: .08em; margin-bottom: 8px; }
.result strong { color: #60cc90; }
.result.warn { background: #150e03; border-color: #3a2006; border-left-color: var(--amber); color: #9a7030; }
.result.warn .etitle { color: var(--amber); }
.result.warn strong { color: #c8a050; }

/* ── Spans ── */
.span-cloud { display: flex; flex-wrap: wrap; gap: 8px; }
.span-tag {
  background: #0e1a42; border: 1px solid #1e3268;
  color: #88b4f8; border-radius: 6px; padding: 5px 12px;
  font-size: 12.5px; font-family: var(--mono); line-height: 1.4;
  white-space: normal; word-break: break-word;
}

/* ── Sezione label ── */
.sec-label { font-size: 11px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: .07em; }

/* ── Entity cards ── */
.ent-grid { display: flex; flex-direction: column; gap: 7px; }
.ent-card {
  background: #0e1520; border: 1px solid #1e2c40; border-radius: 8px;
  padding: 10px 14px; display: flex; align-items: flex-start; gap: 10px;
}
.type-badge { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; border-radius: 4px; padding: 3px 7px; flex-shrink: 0; font-family: var(--mono); }
.conf-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; margin-top: 8px; }
.ch { background: var(--green); } .cm { background: var(--amber); } .cl { background: var(--red); }
.ent-main { flex: 1; min-width: 0; }
.ent-text { font-weight: 600; font-size: 13.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ent-desc { font-size: 12px; color: var(--muted); margin-top: 2px; }

/* ── Relaciones ── */
.rel-list { display: flex; flex-direction: column; gap: 6px; }
.rel-row {
  background: #0e1520; border: 1px solid #1e2c40; border-radius: 7px;
  padding: 9px 13px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; font-size: 13px;
}
.rel-valid   { border-left: 3px solid var(--green); }
.rel-invalid { border-left: 3px solid #3a1820; opacity: .6; }
.rel-subj, .rel-obj { font-weight: 600; }
.rel-arr { color: var(--muted2); font-size: 10px; }
.rel-type { background: #0a2414; color: var(--green); border-radius: 4px; padding: 2px 8px; font-family: var(--mono); font-size: 11px; }
.rel-invalid .rel-type { background: #2a0d10; color: #d06070; }
.rel-why { font-size: 10px; color: var(--red); margin-left: auto; max-width: 240px; text-align: right; opacity: .85; }

/* ── Canonical cards ── */
.can-grid { display: flex; flex-direction: column; gap: 7px; }
.can-card { background: #0e1520; border: 1px solid #1e2c40; border-radius: 8px; padding: 11px 14px; }
.can-card.merged { background: #0a1408; border-color: #1e3a18; }
.can-head { display: flex; align-items: center; gap: 7px; flex-wrap: wrap; }
.can-text { font-weight: 600; font-size: 13.5px; }
.can-id { font-size: 9px; color: var(--muted2); font-family: var(--mono); margin: 3px 0 6px; }
.variants { display: flex; flex-wrap: wrap; gap: 4px; }
.vtag { background: var(--bg); border: 1px solid var(--border); border-radius: 4px; padding: 2px 8px; font-size: 11px; color: var(--muted); font-family: var(--mono); }
.vtag.pri { border-color: var(--accent); color: var(--accent); }
.n-inst { font-size: 11px; color: var(--muted); margin-left: auto; font-family: var(--mono); }
.badge-chunk { font-size: 10px; background: #0a1e28; color: #3ddbc8; border-radius: 4px; padding: 2px 7px; border: 1px solid #1a3848; }
.badge-merged { font-size: 10px; background: #102408; color: var(--green); border-radius: 4px; padding: 2px 7px; border: 1px solid #204810; }

/* ── Merge section ── */
.merge-box { background: #07130a; border: 1px solid #183018; border-radius: 8px; padding: 12px 15px; }
.merge-title { font-size: 11px; color: var(--green); font-weight: 700; text-transform: uppercase; letter-spacing: .07em; margin-bottom: 9px; }
.merge-row { font-size: 13px; color: #50a870; padding: 5px 0; border-bottom: 1px solid #182818; display: flex; align-items: baseline; gap: 7px; flex-wrap: wrap; }
.merge-row:last-child { border-bottom: none; }
.merge-type { font-size: 10px; color: var(--muted); font-family: var(--mono); background: #0a0e14; border-radius: 3px; padding: 1px 6px; }

/* ── Summary cards ── */
.sum-card { background: #0e1520; border: 1px solid #1e2c40; border-radius: 8px; padding: 12px 15px; }
.sum-head { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; }
.sum-text { font-size: 13px; color: var(--text); line-height: 1.6; }
.llm-b { font-size: 10px; background: #160a2e; color: var(--indigo); border-radius: 4px; padding: 2px 8px; font-weight: 600; border: 1px solid #301860; }
.nollm-b { font-size: 10px; background: var(--bg3); color: var(--muted); border-radius: 4px; padding: 2px 8px; border: 1px solid var(--border); }
.sum-descs { margin-top: 9px; border-top: 1px solid var(--border); padding-top: 9px; }
.sum-descs-lbl { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .07em; margin-bottom: 5px; }
.sum-desc-item { font-size: 12px; color: var(--muted); padding: 3px 0 3px 10px; border-left: 2px solid var(--border2); margin-bottom: 3px; }

/* ── Type color classes ── */
.tc0{background:#0d1638;color:#7aa8f0;border:1px solid #1a2e60}
.tc1{background:#0d2218;color:#44cc88;border:1px solid #1a4428}
.tc2{background:#221608;color:#dda040;border:1px solid #382a10}
.tc3{background:#220d0d;color:#e07080;border:1px solid #381818}
.tc4{background:#0d1e22;color:#3adac0;border:1px solid #1a3840}
.tc5{background:#1e0d22;color:#cc80f0;border:1px solid #381848}
.tc6{background:#181e0d;color:#b0d060;border:1px solid #283818}
.tc7{background:#0d0d22;color:#90b0f0;border:1px solid #1a1a40}

/* ── Misc ── */
.empty { color: var(--muted2); font-size: 13px; text-align: center; padding: 30px 0; }
.skel { background: #0e1520; border-radius: 7px; animation: pulse 1.2s ease-in-out infinite; }
@keyframes pulse { 0%,100%{opacity:.3} 50%{opacity:.7} }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
</style>
</head>
<body>

<!-- HEADER -->
<div class="hdr">
  <div class="hdr-logo">⚗</div>
  <div>
    <div class="hdr h1" style="font-size:16px;font-weight:600;color:#e4f0ff">GraphRAG NER Pipeline — Demo ULL</div>
    <div class="sub">Detección de entidades · Resolución canónica · Summarización · Corpus universitario español</div>
  </div>
  <div class="hdr-badge">TFM · GraphRAG</div>
</div>

<div class="layout">

<!-- SIDEBAR -->
<div class="sidebar">
<div class="sidebar-inner">

  <div class="banner">
    <div class="banner-title">¿Qué hace este sistema?</div>
    Este demo muestra un pipeline de <strong>extracción de conocimiento</strong> sobre textos universitarios.
    A partir de texto en lenguaje natural, construye automáticamente un <strong>grafo de conocimiento</strong>
    con entidades (órganos, personas, normativas…) y sus relaciones.
    Resuelve el problema fundamental de los sistemas <em>chunk-by-chunk</em>: la misma entidad puede aparecer
    con nombres distintos en distintos fragmentos.
  </div>

  <div class="steps">
    <div class="step s1"><span class="n">1</span>Span<br>Detection</div>
    <div class="step s2"><span class="n">2</span>NER +<br>Relaciones</div>
    <div class="step s3"><span class="n">3</span>Entity<br>Resolver</div>
    <div class="step s4"><span class="n">4</span>Summa-<br>rizer</div>
  </div>

  <div>
    <label class="lbl">Ruta de ontología (.yaml)</label>
    <input type="text" id="onto-path" value="__ONTO_PATH__">
    <div id="onto-box" class="onto-box"><span style="color:var(--muted);font-size:11px">Cargando…</span></div>
  </div>

  <div>
    <label class="lbl">Fragmentos de texto (chunks)</label>
    <div class="chunk-list" id="chunk-list"></div>
  </div>

  <button class="add-btn" onclick="addChunk()">＋ Añadir chunk vacío</button>
  <button class="run-btn" id="run-btn" onclick="runPipeline()">
    <span id="run-icon">▶</span><span id="run-text">Ejecutar Pipeline</span>
  </button>
  <button class="dl-btn" id="dl-btn" onclick="window.open('/download-resultados','_blank')">
    ⬇ Descargar ner_resultados.json
  </button>

</div>
</div>

<!-- MAIN -->
<div class="main" id="main">
  <div class="car-content" style="justify-content:center">
    <div class="empty">Selecciona los chunks y ejecuta el pipeline para ver los resultados.</div>
  </div>
</div>

</div><!-- /layout -->

<script>
// ── Helpers ──────────────────────────────────────────────────────────────────
const TC = ['tc0','tc1','tc2','tc3','tc4','tc5','tc6','tc7'];
const TYPE_MAP = {};
let typeIdx = 0;
function tc(type) {
  if (!TYPE_MAP[type]) TYPE_MAP[type] = TC[typeIdx++ % TC.length];
  return TYPE_MAP[type];
}
function esc(s) {
  return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function conf(c) { return c==='high'?'ch':c==='medium'?'cm':'cl'; }

// ── Chunk management ──────────────────────────────────────────────────────────
let chunkN = 0;
let defLabels = [], defHints = [];

function addChunk(text='', label='', hint='') {
  chunkN++;
  const id = chunkN;
  const el = document.createElement('div');
  el.className = 'chunk-item';
  el.id = `ci-${id}`;
  el.innerHTML = `
    <div class="chunk-hdr">
      <span class="chunk-num">#${id}</span>
      <span class="chunk-lbl">${label ? esc(label) : '<span style="color:var(--muted)">Chunk personalizado</span>'}</span>
      <button class="rm-btn" onclick="rmChunk(${id})">✕</button>
    </div>
    ${hint ? `<div class="chunk-hint">🎯 ${esc(hint)}</div>` : ''}
    <textarea id="ct-${id}" placeholder="Pega aquí el texto…">${esc(text)}</textarea>
  `;
  document.getElementById('chunk-list').appendChild(el);
}
function rmChunk(id) { document.getElementById(`ci-${id}`)?.remove(); }
function getChunks() {
  return [...document.querySelectorAll('.chunk-list textarea')]
    .map(t=>t.value.trim()).filter(Boolean);
}

// ── Ontology preview ──────────────────────────────────────────────────────────
async function loadOnto() {
  const path = document.getElementById('onto-path').value.trim();
  const box  = document.getElementById('onto-box');
  box.innerHTML = '<span style="color:var(--muted);font-size:11px">Cargando…</span>';
  try {
    const d = await (await fetch('/ontology-info?path='+encodeURIComponent(path))).json();
    box.innerHTML = `
      <div style="font-size:10px;color:var(--muted);font-family:var(--mono);margin-bottom:4px">${esc(d.path)}</div>
      <div style="font-size:10px;color:var(--muted);margin-bottom:3px">${d.entity_types.length} tipos de entidad:</div>
      <div class="pills">${d.entity_types.map(t=>`<span class="pill t">${esc(t)}</span>`).join('')||'<span style="color:var(--red);font-size:10px">Ninguno</span>'}</div>
      <div style="font-size:10px;color:var(--muted);margin:6px 0 3px">${d.relation_types.length} relaciones:</div>
      <div class="pills">${d.relation_types.map(t=>`<span class="pill r">${esc(t)}</span>`).join('')||'<span style="color:var(--muted);font-size:10px">Ninguna</span>'}</div>
    `;
  } catch(e) { box.innerHTML = `<span style="color:var(--red);font-size:11px">Error: ${esc(e.message)}</span>`; }
}
document.getElementById('onto-path').addEventListener('change', loadOnto);

async function loadDefaults() {
  try {
    const d = await (await fetch('/default-chunks')).json();
    defLabels = d.labels || []; defHints = d.hints || [];
    d.chunks.forEach((t,i) => addChunk(t, defLabels[i]||'', defHints[i]||''));
  } catch { addChunk(); }
}

loadOnto();
loadDefaults();

// ── Pipeline ──────────────────────────────────────────────────────────────────
async function runPipeline() {
  const chunks = getChunks();
  if (!chunks.length) return;
  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  document.getElementById('run-icon').textContent = '⏳';
  document.getElementById('run-text').textContent = `Procesando ${chunks.length} chunk${chunks.length>1?'s':''}…`;
  document.getElementById('dl-btn').style.display = 'none';

  const main = document.getElementById('main');
  main.innerHTML = `
    <div class="car-content" style="gap:12px">
      <div class="skel" style="height:44px"></div>
      <div class="skel" style="height:120px"></div>
      <div class="skel" style="height:160px"></div>
      <div class="skel" style="height:120px"></div>
    </div>`;

  try {
    const resp = await fetch('/run', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({chunks, ontology_path: document.getElementById('onto-path').value.trim()})
    });
    if (!resp.ok) { showError(await resp.text()); return; }
    renderResults(await resp.json());
    document.getElementById('dl-btn').style.display = 'flex';
  } catch(e) { showError(e.message); }
  finally {
    btn.disabled = false;
    document.getElementById('run-icon').textContent = '▶';
    document.getElementById('run-text').textContent = 'Ejecutar Pipeline';
  }
}
function showError(msg) {
  document.getElementById('main').innerHTML =
    `<div class="car-content"><div style="background:#1a0808;border:1px solid #4a1010;border-radius:8px;padding:14px;font-family:var(--mono);font-size:12px;color:var(--red)">${esc(msg)}</div></div>`;
}

// ── Render ────────────────────────────────────────────────────────────────────
let slides = []; // array di { label, node }
let cur = 0;

function renderResults(data) {
  const totalEnt   = data.chunk_results.reduce((s,c)=>s+c.paso2.entidades.length,0);
  const totalRel   = data.chunk_results.reduce((s,c)=>s+c.paso2.relaciones.length,0);
  const totalValid = data.chunk_results.reduce((s,c)=>s+c.paso2.relaciones.filter(r=>r.valida).length,0);
  const nMerges    = data.paso3.merge_log.length;
  const nCanon     = data.paso3.canonical_entities.length;
  const pct        = totalEnt>0 ? Math.round((1-nCanon/totalEnt)*100) : 0;

  slides = [];
  cur = 0;

  // ── Una slide per Paso1 di ogni chunk ──
  data.chunk_results.forEach((cr,i) => {
    const label = defLabels[i] || cr.chunk_id;
    const hint  = defHints[i]  || '';
    slides.push({
      label: `Paso 1 — Span Detection · Chunk ${i+1}`,
      node: buildSpanSlide(cr.paso1, label, hint, cr.timings.paso1_ms),
    });
    slides.push({
      label: `Paso 2 — NER + Relaciones · Chunk ${i+1}`,
      node: buildNerSlide(cr.paso2, label, hint, cr.timings.paso2_ms),
    });
  });

  // ── Resolver e Summarizer ──
  slides.push({
    label: `Paso 3 — Entity Resolver`,
    node: buildResolverSlide(data.paso3, totalEnt),
  });
  slides.push({
    label: `Paso 4 — Entity Summarizer`,
    node: buildSummarizerSlide(data.paso4),
  });

  // ── Costruisce il DOM del main ──
  const main = document.getElementById('main');
  main.innerHTML = '';

  // stat bar
  const bar = document.createElement('div');
  bar.className = 'stat-bar';
  bar.innerHTML = `
    <div class="stat"><span class="stat-v">${data.n_chunks}</span><span class="stat-l">chunks</span></div>
    <div class="stat-sep"></div>
    <div class="stat"><span class="stat-v">${totalEnt}</span><span class="stat-l">entidades raw</span></div>
    <div class="stat-sep"></div>
    <div class="stat"><span class="stat-v">${nCanon}</span><span class="stat-l">canonicadas</span></div>
    <div class="stat-sep"></div>
    <div class="stat"><span class="stat-v">${nMerges}</span><span class="stat-l">fusiones</span></div>
    <div class="stat-sep"></div>
    <div class="stat"><span class="stat-v">${totalValid}/${totalRel}</span><span class="stat-l">rel. válidas</span></div>
    <div class="t-pill">⏱ ${data.total_ms}ms</div>
  `;
  main.appendChild(bar);

  // risultato globale
  const globalRes = document.createElement('div');
  globalRes.style.cssText = 'padding:10px 24px 0;flex-shrink:0';
  globalRes.innerHTML = `
    <div class="result">
      <div class="etitle">✅ Resultado del pipeline</div>
      De <strong>${totalEnt} menciones</strong> en ${data.n_chunks} chunks → <strong>${nCanon} entidades canónicas</strong>
      (reducción del ${pct}%, <strong>${nMerges} fusiones</strong>).
      Relaciones válidas: <strong>${totalValid}/${totalRel}</strong>. Tiempo total: <strong>${data.total_ms}ms</strong>.
    </div>`;
  main.appendChild(globalRes);

  // nav
  const nav = document.createElement('div');
  nav.className = 'car-nav';
  nav.innerHTML = `
    <button class="car-arr L" id="cp" onclick="go(-1)">‹</button>
    <div class="car-mid">
      <div class="car-label" id="cl"></div>
      <div class="car-dots" id="cd"></div>
      <div class="car-cnt"  id="cc"></div>
    </div>
    <button class="car-arr R" id="cn" onclick="go(1)">›</button>
  `;
  main.appendChild(nav);

  // content area (una sola div visibile alla volta)
  const content = document.createElement('div');
  content.className = 'car-content';
  content.id = 'car-content';
  main.appendChild(content);

  updateCarousel();

  document.onkeydown = e => {
    if (e.key==='ArrowRight') go(1);
    if (e.key==='ArrowLeft')  go(-1);
  };
}

function go(d) { cur = Math.max(0, Math.min(slides.length-1, cur+d)); updateCarousel(); }
function goTo(i) { cur=i; updateCarousel(); }

function updateCarousel() {
  const n = slides.length;
  // aggiorna label, dots, counter
  document.getElementById('cl').textContent = slides[cur]?.label || '';
  document.getElementById('cc').textContent = `${cur+1} / ${n}`;
  document.getElementById('cd').innerHTML = slides.map((s,i)=>
    `<div class="car-dot${i===cur?' on':''}" onclick="goTo(${i})" title="${esc(s.label)}"></div>`
  ).join('');
  document.getElementById('cp').disabled = cur===0;
  document.getElementById('cn').disabled = cur===n-1;

  // sostituisce il contenuto — NO carosello CSS, solo show/hide
  const c = document.getElementById('car-content');
  c.innerHTML = '';
  if (slides[cur]) c.appendChild(slides[cur].node.cloneNode(true));
  c.scrollTop = 0;
}

// ── Builders ──────────────────────────────────────────────────────────────────
function buildSpanSlide(d, label, hint, ms) {
  const spans = d.spans || [];
  const el = document.createElement('div');
  el.style.display = 'contents';

  const expl = document.createElement('div');
  expl.className = 'explain';
  expl.innerHTML = `
    <div class="etitle">ℹ️ Paso 1 — Detección de spans · ${esc(label)}</div>
    El LLM recorre el texto e identifica los <strong>candidatos a entidad</strong> sin clasificarlos aún.
    Solo se delimitan los fragmentos de texto potencialmente relevantes para el dominio universitario.
    En este paso no se asigna ningún tipo.
    ${hint ? `<br><br>🎯 <em>${esc(hint)}</em>` : ''}
    <div style="margin-top:8px;font-size:11px;color:var(--muted);font-family:var(--mono)">⏱ ${ms}ms</div>
  `;
  el.appendChild(expl);

  const label2 = document.createElement('div');
  label2.className = 'sec-label';
  label2.textContent = `${spans.length} spans detectados`;
  el.appendChild(label2);

  if (spans.length) {
    const cloud = document.createElement('div');
    cloud.className = 'span-cloud';
    cloud.innerHTML = spans.map(s=>`<span class="span-tag">${esc(s.text)}</span>`).join('');
    el.appendChild(cloud);
  } else {
    const e = document.createElement('div'); e.className='empty'; e.textContent='Ningún span detectado';
    el.appendChild(e);
  }
  return el;
}

function buildNerSlide(d, label, hint, ms) {
  const ents = d.entidades  || [];
  const rels = d.relaciones || [];
  const valid   = rels.filter(r=>r.valida).length;
  const invalid = rels.length - valid;
  const el = document.createElement('div');
  el.style.display = 'contents';

  // spiegazione
  const expl = document.createElement('div');
  expl.className = 'explain';
  expl.innerHTML = `
    <div class="etitle">ℹ️ Paso 2 — NER + Relaciones · ${esc(label)}</div>
    Cada span detectado se <strong>clasifica</strong> según la ontología universitaria
    (Órgano, Resolución, Titulación…) y se extraen las <strong>relaciones</strong> entre entidades.
    Las relaciones se validan contra los pares dominio→rango de la ontología.
    ${hint ? `<br><br>🎯 <em>${esc(hint)}</em>` : ''}
    <div style="margin-top:8px;font-size:11px;color:var(--muted);font-family:var(--mono)">⏱ ${ms}ms</div>
  `;
  el.appendChild(expl);

  // entidades
  const el2 = document.createElement('div'); el2.className='sec-label'; el2.textContent=`${ents.length} entidades clasificadas`;
  el.appendChild(el2);
  if (ents.length) {
    const grid = document.createElement('div'); grid.className='ent-grid';
    grid.innerHTML = ents.map(e=>`
      <div class="ent-card">
        <div class="conf-dot ${conf(e.confidence)}"></div>
        <span class="type-badge ${tc(e.entity_type)}">${esc(e.entity_type)}</span>
        <div class="ent-main">
          <div class="ent-text">${esc(e.text)}</div>
          ${e.descripcion?`<div class="ent-desc">${esc(e.descripcion)}</div>`:''}
        </div>
      </div>`).join('');
    el.appendChild(grid);
  } else { const e2=document.createElement('div');e2.className='empty';e2.textContent='Ninguna entidad';el.appendChild(e2); }

  // relaciones
  const el3=document.createElement('div');el3.className='sec-label';el3.textContent=`${rels.length} relaciones (${valid} válidas, ${invalid} rechazadas)`;
  el.appendChild(el3);
  if (rels.length) {
    const list=document.createElement('div');list.className='rel-list';
    list.innerHTML = rels.map(r=>`
      <div class="rel-row ${r.valida?'rel-valid':'rel-invalid'}">
        <span class="rel-subj">${esc(r.sujeto)}</span>
        <span class="rel-arr">→</span>
        <span class="rel-type">${esc(r.relacion)}</span>
        <span class="rel-arr">→</span>
        <span class="rel-obj">${esc(r.objeto)}</span>
        ${!r.valida&&r.motivo?`<span class="rel-why" title="${esc(r.motivo)}">⚠ ${esc(r.motivo)}</span>`:''}
      </div>`).join('');
    el.appendChild(list);
  }

  // risultato
  const res = document.createElement('div');
  res.className = invalid>0 ? 'result warn' : 'result';
  res.innerHTML = invalid>0
    ? `<div class="etitle">⚠️ ${invalid} relación${invalid>1?'es':''} rechazada${invalid>1?'s':''}</div>
       El tipo de entidad origen o destino no coincide con los dominios permitidos en la ontología.`
    : `<div class="etitle">✅ Todas las relaciones son válidas</div>Las ${valid} relaciones respetan la ontología.`;
  el.appendChild(res);
  return el;
}

function buildResolverSlide(d, totalEntRaw) {
  const cans    = d.canonical_entities || [];
  const merges  = d.merge_log || [];
  const pct     = totalEntRaw>0 ? Math.round((1-cans.length/totalEntRaw)*100) : 0;
  const varM    = merges.filter(m=>m.variants.length>1).length;
  const crossC  = merges.filter(m=>m.n_chunks>1).length;
  const el = document.createElement('div');
  el.style.display = 'contents';

  // spiegazione
  const expl = document.createElement('div');
  expl.className = 'explain';
  expl.innerHTML = `
    <div class="etitle">ℹ️ Paso 3 — Entity Resolver</div>
    El resolver agrega las entidades de <strong>todos los chunks</strong> en un conjunto canónico único.
    Aplica tres estrategias en cascada:
    <ul>
      <li><strong>Exact merge:</strong> agrupa variantes que difieren solo en mayúsculas/minúsculas.</li>
      <li><strong>Abbreviation merge:</strong> expande acrónimos conocidos (ej. <em>ULL → Universidad de La Laguna</em>).</li>
      <li><strong>Embedding merge:</strong> similitud coseno ≥ 0.92 con <em>text-embedding-3-small</em>.
        Se excluyen tipos normativos y textos con números.</li>
    </ul>
  `;
  el.appendChild(expl);

  // resultado
  const res = document.createElement('div');
  res.className = 'result';
  res.innerHTML = `
    <div class="etitle">✅ Resultado del resolver</div>
    <strong>${totalEntRaw} menciones raw</strong> → <strong>${cans.length} entidades canónicas</strong>
    (reducción del ${pct}%).
    ${merges.length ? `<strong>${varM}</strong> fusiones por variante léxica/embedding,
    <strong>${crossC}</strong> entidades reconocidas en múltiples chunks.` : 'Sin fusiones — todas las entidades eran léxicamente únicas.'}
  `;
  el.appendChild(res);

  // fusioni
  if (merges.length) {
    const box=document.createElement('div');box.className='merge-box';
    box.innerHTML=`<div class="merge-title">⚡ Fusiones realizadas (${merges.length})</div>`+
      merges.map(m=>`
        <div class="merge-row">
          <span class="merge-type">${esc(m.entity_type)}</span>
          <strong>${esc(m.canonical_text)}</strong>
          ${m.variants.length>1?'← '+m.variants.filter(v=>v!==m.canonical_text).map(v=>`<em>${esc(v)}</em>`).join(', '):''}
          <span style="margin-left:auto;font-size:10px;color:var(--muted);font-family:var(--mono)">${m.n_instances} inst. · ${m.n_chunks} chunk${m.n_chunks>1?'s':''}</span>
        </div>`).join('');
    el.appendChild(box);
  }

  // canoniche
  const lbl=document.createElement('div');lbl.className='sec-label';lbl.textContent=`${cans.length} entidades canónicas`;
  el.appendChild(lbl);
  if (cans.length) {
    const grid=document.createElement('div');grid.className='can-grid';
    grid.innerHTML=cans.map(ce=>`
      <div class="can-card${ce.variants&&ce.variants.length>1?' merged':''}">
        <div class="can-head">
          <span class="type-badge ${tc(ce.entity_type)}">${esc(ce.entity_type)}</span>
          <span class="can-text">${esc(ce.canonical_text)}</span>
          ${ce.n_chunks>1?`<span class="badge-chunk">×${ce.n_chunks} chunks</span>`:''}
          ${ce.variants&&ce.variants.length>1?`<span class="badge-merged">fusionado</span>`:''}
          <span class="n-inst">${ce.n_instances} inst.</span>
        </div>
        <div class="can-id">id: ${esc(ce.canonical_id)}</div>
        ${ce.variants&&ce.variants.length>1?`<div class="variants">${ce.variants.map((v,i)=>`<span class="vtag${i===0?' pri':''}">${esc(v)}</span>`).join('')}</div>`:''}
      </div>`).join('');
    el.appendChild(grid);
  }
  return el;
}

function buildSummarizerSlide(d) {
  const sums   = d.summarized || [];
  const llmN   = sums.filter(s=>s.used_llm).length;
  const el = document.createElement('div');
  el.style.display = 'contents';

  const expl=document.createElement('div');expl.className='explain';
  expl.innerHTML=`
    <div class="etitle">ℹ️ Paso 4 — Entity Summarizer</div>
    Cada entidad canónica puede haber acumulado <strong>varias descripciones</strong> de distintos chunks.
    El summarizer las consolida en una única descripción canónica:
    <ul>
      <li><strong>Una sola descripción</strong> → se usa directamente (badge <em>descripción directa</em>).</li>
      <li><strong>Varias descripciones</strong> → el LLM genera una síntesis coherente (badge <em>LLM consolidado</em>).</li>
    </ul>
    Esta descripción se almacenará como propiedad del nodo en el grafo Neo4j.
  `;
  el.appendChild(expl);

  const res=document.createElement('div');res.className='result';
  res.innerHTML=`
    <div class="etitle">✅ Resultado del summarizer</div>
    <strong>${llmN}</strong> entidades consolidadas con LLM (tenían descripciones múltiples),
    <strong>${sums.length-llmN}</strong> con descripción directa.
    Se muestran solo las <strong>${sums.filter(s=>s.descripcion_consolidada?.trim()).length}</strong> entidades con descripción.
  `;
  el.appendChild(res);

  const sumsConDesc = sums.filter(s => s.descripcion_consolidada && s.descripcion_consolidada.trim());
  if (sumsConDesc.length) {
    sumsConDesc.forEach(s=>{
      const card=document.createElement('div');card.className='sum-card';
      card.innerHTML=`
        <div class="sum-head">
          <span class="type-badge ${tc(s.entity_type)}">${esc(s.entity_type)}</span>
          <span style="font-weight:600;font-size:13.5px">${esc(s.canonical_text)}</span>
          ${s.used_llm?'<span class="llm-b">LLM consolidado</span>':'<span class="nollm-b">descripción directa</span>'}
        </div>
        ${s.descripcion_consolidada
          ?`<div class="sum-text">${esc(s.descripcion_consolidada)}</div>`
          :'<div style="font-size:11px;color:var(--muted2)">— sin descripción —</div>'}
        ${s.descriptions&&s.descriptions.length>1?`
          <div class="sum-descs">
            <div class="sum-descs-lbl">Descripciones originales (${s.descriptions.length})</div>
            ${s.descriptions.map(d=>`<div class="sum-desc-item">${esc(d)}</div>`).join('')}
          </div>`:''}
      `;
      el.appendChild(card);
    });
  } else { const e=document.createElement('div');e.className='empty';e.textContent='Ninguna entidad con descripción';el.appendChild(e); }
  return el;
}
</script>
</body>
</html>
""".replace("__ONTO_PATH__", ONTOLOGY_PATH)



@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(HTML)


if __name__ == "__main__":
    print("=" * 55)
    print("  GraphRAG NER Debug — Demo ULL")
    print(f"  Ontología : {ONTOLOGY_PATH}")
    print(f"  Acrónimos : {ACRONIMOS_PATH}")
    print(f"  LLM URL   : {LLM_BASE_URL or '(OpenAI default)'}")
    print(f"  Modelos   : {MODEL_PASO1} / {MODEL_PASO2}")
    print("=" * 55)
    print(f"  🔑 Password : {DEMO_PASSWORD}")
    print(f"  Acceso local  : http://localhost:8000/?pw={DEMO_PASSWORD}")
    print(f"  Acceso remoto : https://<tunnel-url>/?pw={DEMO_PASSWORD}")
    print("=" * 55)
    uvicorn.run(app, host="0.0.0.0", port=8000)