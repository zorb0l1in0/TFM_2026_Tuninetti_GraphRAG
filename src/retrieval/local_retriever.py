"""
comparar_retrievers.py  (v3 — evaluación con ground truth)
-----------------------------------------------------------
Aggiunta rispetto alla v2:

  - Classe EvaluadorRetrieval che calcola Precision@K e Recall@K
    per entrambi i retriever, a partire da un file ground_truth.json.
  - La valutazione viene eseguita dopo il confronto e aggiunta ai
    risultati JSON finali.
  - Stampa una tabella di riepilogo con P@K, R@K e F1@K per domanda
    e medie globali.

Formato ground_truth.json:
  [
    {
      "pregunta": "...",
      "chunks_relevantes": ["chunk_id_1", "chunk_id_2"],
      "nota": "Spiegazione opzionale"
    },
    ...
  ]
"""

from __future__ import annotations

import ast
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.common.clients import get_embeddings, get_neo4j_graph
from src.name_entity_recognition.pipeline import PipelineNERDosPasos
from src.graph_building.entity_resolver import EntityMap, EntityResolver


# ── Preguntas de evaluación ───────────────────────────────────────────────────

PREGUNTAS = [
    "¿Cuál es la fórmula para calcular la nota de admisión por convalidación parcial de estudios extranjeros?",
    "¿Ante qué juzgado puede interponerse recurso contencioso-administrativo?",
    "¿Qué documentación debe presentar un estudiante extranjero para solicitar la convalidación parcial?",
    "¿Qué dice la Ley 39/2015 en relación con la presentación de documentación?",
    "¿A través de qué sede y mediante qué procedimiento electrónico se presentan las solicitudes de cambio de universidad?",
    "¿Cuáles son los períodos en que permanecen cerradas las instalaciones de la Universidad de La Laguna en 2026?",
]

# ── Rutas ─────────────────────────────────────────────────────────────────────

_ROOT           = Path(__file__).resolve().parent.parent.parent
RUTA_CSV        = _ROOT / "data" / "processed" / "chunks" / "chunks_con_embeddings.csv"
RUTA_SALIDA     = _ROOT / "data" / "retrieval" / "comparacion_resultados.json"
RUTA_ONTOLOGIA  = _ROOT / "data" / "ner" / "ontologia" / "ontology.yaml"
RUTA_NER_JSON   = _ROOT / "data" / "ner" / "ner_resultados.json"
RUTA_ACRONIMOS  = _ROOT / "data" / "acronimos.yaml"
RUTA_GT         = _ROOT / "data" / "retrieval" / "ground_truth.json"

# ── Constantes ────────────────────────────────────────────────────────────────

_SCORE_DECAY = 0.5

_KEYWORDS_MULTIHOP = {
    "a través de", "mediante", "por medio de", "quién aprueba", "quién regula",
    "cómo llega", "qué órgano", "cuál es el camino", "relación entre",
    "conectado", "vinculado", "dependiente", "competente para",
}
_KEYWORDS_SIMPLE = {
    "qué es", "define", "menciona", "aparece", "cuándo", "dónde",
    "qué dice", "lista", "enumera",
}


# ── Dataclass compartida ──────────────────────────────────────────────────────

@dataclass
class ChunkRecuperado:
    chunk_id: str
    titulo:   str
    fuente:   str
    texto:    str
    score:    float = 1.0
    hop:      int   = 0


# ═════════════════════════════════════════════════════════════════════════════
# EVALUADOR  — Precision@K, Recall@K, F1@K
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class MetricasRetriever:
    precision: float = 0.0
    recall:    float = 0.0
    f1:        float = 0.0
    recuperados:   List[str] = field(default_factory=list)
    relevantes:    List[str] = field(default_factory=list)
    verdaderos_pos: List[str] = field(default_factory=list)


class EvaluadorRetrieval:
    """
    Calcula Precision@K, Recall@K y F1@K comparando los chunks
    recuperados por cada retriever contra una ground truth manual.

    Precision@K = TP / K
        De los K chunks devueltos, ¿cuántos son realmente relevantes?
        Penaliza el ruido.

    Recall@K = TP / |relevantes|
        De todos los chunks relevantes que existen, ¿cuántos encontró
        el retriever en sus K resultados?
        Penaliza los chunks perdidos.

    F1@K = 2 * P * R / (P + R)
        Media armónica entre precisión y recall. Útil para comparar
        con un solo número cuando ambas métricas importan.
    """

    def __init__(self, ruta_ground_truth: Path = RUTA_GT):
        if not ruta_ground_truth.exists():
            raise FileNotFoundError(
                f"Ground truth no encontrada: {ruta_ground_truth}\n"
                "Crea el archivo con el formato descrito en el docstring del módulo."
            )
        with open(ruta_ground_truth, encoding="utf-8") as f:
            datos = json.load(f)

        # índice por pregunta (texto exacto)
        self._gt: Dict[str, List[str]] = {
            d["pregunta"]: d["chunks_relevantes"]
            for d in datos
        }

    def evaluar(
        self,
        pregunta:   str,
        recuperados: List[str],   # chunk_ids en orden de ranking
    ) -> MetricasRetriever:
        """
        Calcula las métricas para una pregunta y una lista de chunk_ids.
        El orden de `recuperados` no afecta al cálculo (usamos @K = len(recuperados)).
        """
        relevantes = self._gt.get(pregunta, [])
        if not relevantes:
            # sin ground truth para esta pregunta → métricas nulas
            return MetricasRetriever(relevantes=[], recuperados=recuperados)

        set_rec = set(recuperados)
        set_rel = set(relevantes)
        tp      = set_rec & set_rel

        k         = len(recuperados)
        precision = len(tp) / k          if k              else 0.0
        recall    = len(tp) / len(set_rel) if set_rel      else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        return MetricasRetriever(
            precision       = round(precision, 3),
            recall          = round(recall,    3),
            f1              = round(f1,         3),
            recuperados     = recuperados,
            relevantes      = relevantes,
            verdaderos_pos  = sorted(tp),
        )

    def tiene_ground_truth(self, pregunta: str) -> bool:
        return pregunta in self._gt


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS — EntityMap
# ═════════════════════════════════════════════════════════════════════════════

def cargar_entity_map(ruta_ner_json: Path, ruta_acronimos: Path) -> EntityMap:
    resolver = EntityResolver(
        usar_embedding_merge=False,
        ruta_acronimos=ruta_acronimos,
        verbose=False,
    )
    return resolver.resolve(str(ruta_ner_json))


# ═════════════════════════════════════════════════════════════════════════════
# RETRIEVER GRAFO + NER
# ═════════════════════════════════════════════════════════════════════════════

class RetrieverGrafo:
    def __init__(
        self,
        graph        = None,
        entity_map:  Optional[EntityMap] = None,
        verbose:     bool = False,
        max_chunks:  int  = 8,
        max_hops:    int  = 2,
        ruta_ontologia: Path = RUTA_ONTOLOGIA,
        ruta_ner_json:  Path = RUTA_NER_JSON,
        ruta_acronimos: Path = RUTA_ACRONIMOS,
    ):
        self.graph      = graph or get_neo4j_graph()
        self.verbose    = verbose
        self.max_chunks = max_chunks
        self.max_hops   = max_hops

        self.ner_pipeline = PipelineNERDosPasos(
            ruta_ontologia=str(ruta_ontologia),
            temperatura=0.0,
        )
        self.entity_map = entity_map or cargar_entity_map(ruta_ner_json, ruta_acronimos)

        if verbose:
            s = self.entity_map.stats()
            print(f"   📖 EntityMap: {s['nodi_canonici']} nodos canónicos")

    def recuperar(self, consulta: str) -> Tuple[List[ChunkRecuperado], List[str], int]:
        entidades = self._extraer_entidades_ner(consulta)
        hops      = self._determinar_hops(consulta)
        if not entidades:
            return [], [], hops
        nodos = self._buscar_en_grafo(entidades)
        if not nodos:
            return [], entidades, hops
        chunks_raw = self._traversal(nodos, hops)
        chunks     = self._rankear(chunks_raw)
        return chunks, entidades, hops

    def _extraer_entidades_ner(self, consulta: str) -> List[str]:
        try:
            resultado = self.ner_pipeline.ejecutar(consulta)
        except Exception as e:
            if self.verbose:
                print(f"   ⚠ Error NER pipeline: {e}")
            return []

        entidades_raw = resultado.get("entidades", [])
        if not entidades_raw:
            return []

        textos = []
        for ent in entidades_raw:
            texto = ent.get("text", "").strip()
            if not texto:
                continue
            canonical = self.entity_map.lookup(texto)
            texto_final = canonical.canonical_text if canonical else texto
            if self.verbose:
                tag = f"→ '{texto_final}'" if canonical else "(sin canónico)"
                print(f"   NER '{texto}' {tag}")
            if texto_final not in textos:
                textos.append(texto_final)
        return textos

    def _determinar_hops(self, consulta: str) -> int:
        q = consulta.lower()
        for kw in _KEYWORDS_MULTIHOP:
            if kw in q:
                return min(3, self.max_hops)
        for kw in _KEYWORDS_SIMPLE:
            if kw in q:
                return 1
        if len(consulta) > 120 or consulta.count("?") > 1:
            return min(3, self.max_hops)
        return min(2, self.max_hops)

    def _buscar_en_grafo(self, entidades: List[str]) -> List[Dict]:
        encontrados = []
        ids_vistos  = set()
        for texto in entidades:
            filas = self.graph.query(
                "MATCH (e:Entidad) WHERE toLower(e.text) = toLower($texto) "
                "RETURN e.id AS id, e.text AS text LIMIT 5",
                {"texto": texto},
            )
            if not filas:
                filas = self.graph.query(
                    "MATCH (e:Entidad) "
                    "WHERE toLower(e.text) CONTAINS toLower($texto) "
                    "   OR toLower($texto) CONTAINS toLower(e.text) "
                    "RETURN e.id AS id, e.text AS text "
                    "ORDER BY size(e.text) DESC LIMIT 3",
                    {"texto": texto},
                )
            for fila in filas:
                if fila["id"] not in ids_vistos:
                    ids_vistos.add(fila["id"])
                    encontrados.append(fila)
        return encontrados

    def _traversal(self, nodos: List[Dict], hops: int) -> List[Dict]:
        ids_semilla = [n["id"] for n in nodos]
        chunks_raw  = []
        directos = self.graph.query(
            """
            MATCH (e:Entidad)-[:MENTIONED_IN]->(c:Chunk)
            WHERE e.id IN $ids
            RETURN e.id AS entity_id, c.chunk_id AS chunk_id,
                   c.titulo AS titulo, c.fuente AS fuente,
                   c.texto AS texto, 0 AS hop, 1.0 AS score_decay
            """,
            {"ids": ids_semilla},
        )
        chunks_raw.extend(directos)
        for hop in range(1, hops + 1):
            decay  = _SCORE_DECAY ** (hop - 1)
            cypher = f"""
            MATCH path = (semilla:Entidad)-[*1..{hop}]-(vecino:Entidad)
            WHERE semilla.id IN $ids
              AND ALL(r IN relationships(path)
                      WHERE type(r) <> 'MENTIONED_IN' AND type(r) <> 'BELONGS_TO')
            WITH DISTINCT vecino
            MATCH (vecino)-[:MENTIONED_IN]->(c:Chunk)
            RETURN vecino.id AS entity_id, c.chunk_id AS chunk_id,
                   c.titulo AS titulo, c.fuente AS fuente,
                   c.texto AS texto, {hop} AS hop, {decay:.4f} AS score_decay
            """
            try:
                chunks_raw.extend(self.graph.query(cypher, {"ids": ids_semilla}))
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠ Error traversal hop={hop}: {e}")
        return chunks_raw

    def _rankear(self, chunks_raw: List[Dict]) -> List[ChunkRecuperado]:
        agregado: Dict[str, Dict] = {}
        for fila in chunks_raw:
            cid = fila["chunk_id"]
            if cid not in agregado:
                agregado[cid] = {
                    "chunk_id": cid,
                    "titulo":   fila.get("titulo", ""),
                    "fuente":   fila.get("fuente", ""),
                    "texto":    fila.get("texto",  ""),
                    "score":    0.0,
                    "hop":      fila.get("hop", 0),
                }
            agregado[cid]["score"] += float(fila.get("score_decay", 1.0))
            agregado[cid]["hop"]    = min(agregado[cid]["hop"], fila.get("hop", 0))
        ordenados = sorted(agregado.values(), key=lambda x: (-x["score"], x["hop"]))
        return [
            ChunkRecuperado(
                chunk_id = r["chunk_id"],
                titulo   = r["titulo"],
                fuente   = r["fuente"],
                texto    = r["texto"],
                score    = round(r["score"], 4),
                hop      = r["hop"],
            )
            for r in ordenados[: self.max_chunks]
        ]


# ═════════════════════════════════════════════════════════════════════════════
# RETRIEVER EMBEDDING
# ═════════════════════════════════════════════════════════════════════════════

class RetrieverEmbedding:
    def __init__(self, ruta_csv: Path = RUTA_CSV, top_k: int = 8, verbose: bool = False):
        self.top_k   = top_k
        self.verbose = verbose
        self._cargar_csv(ruta_csv)

    def _cargar_csv(self, ruta: Path):
        if not ruta.exists():
            raise FileNotFoundError(f"CSV no encontrado: {ruta}")
        df = pd.read_csv(ruta)

        def parsear(val):
            if isinstance(val, str):
                return np.array(ast.literal_eval(val), dtype=np.float32)
            return np.array(val, dtype=np.float32)

        df["_emb"] = df["embedding"].apply(parsear)
        self.df    = df
        mat        = np.stack(df["_emb"].values)
        normas     = np.linalg.norm(mat, axis=1, keepdims=True)
        self.mat_norm = mat / (normas + 1e-9)
        if self.verbose:
            print(f"   📂 CSV: {len(df)} chunks, dim={mat.shape[1]}")

    def recuperar(self, consulta: str) -> List[ChunkRecuperado]:
        vec   = np.array(get_embeddings([consulta])[0], dtype=np.float32)
        norma = np.linalg.norm(vec)
        vec_n = vec / (norma + 1e-9)
        scores  = self.mat_norm @ vec_n
        indices = np.argsort(scores)[::-1][: self.top_k]
        return [
            ChunkRecuperado(
                chunk_id = str(self.df.iloc[i].get("id_chunk",
                               self.df.iloc[i].get("chunk_id", f"idx_{i}"))),
                titulo   = str(self.df.iloc[i].get("titulo", "")),
                fuente   = str(self.df.iloc[i].get("archivo_origen",
                               self.df.iloc[i].get("fuente", ""))),
                texto    = str(self.df.iloc[i].get("texto", "")),
                score    = round(float(scores[i]), 4),
                hop      = 0,
            )
            for i in indices
        ]


# ═════════════════════════════════════════════════════════════════════════════
# COMPARACIÓN + EVALUACIÓN
# ═════════════════════════════════════════════════════════════════════════════

def comparar(
    preguntas:           List[str],
    retriever_grafo:     RetrieverGrafo,
    retriever_embedding: RetrieverEmbedding,
    evaluador:           Optional[EvaluadorRetrieval] = None,
) -> List[Dict]:

    resultados = []

    for i, pregunta in enumerate(preguntas, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(preguntas)}] {pregunta[:75]}")
        print("="*60)

        # ── Grafo+NER ─────────────────────────────────────────────
        print("  🕸️  Grafo+NER...")
        t0 = time.time()
        chunks_grafo, entidades, hops = retriever_grafo.recuperar(pregunta)
        t_grafo   = round(time.time() - t0, 2)
        ids_grafo = [ch.chunk_id for ch in chunks_grafo]

        # ── Embedding ─────────────────────────────────────────────
        print("  📐 Embedding...")
        t0 = time.time()
        chunks_emb = retriever_embedding.recuperar(pregunta)
        t_emb   = round(time.time() - t0, 2)
        ids_emb = [ch.chunk_id for ch in chunks_emb]

        # ── Jaccard ───────────────────────────────────────────────
        set_g   = set(ids_grafo)
        set_e   = set(ids_emb)
        overlap = set_g & set_e
        union   = set_g | set_e
        jaccard = round(len(overlap) / len(union), 3) if union else 0.0

        # ── Evaluación con ground truth ───────────────────────────
        eval_grafo = None
        eval_emb   = None
        if evaluador and evaluador.tiene_ground_truth(pregunta):
            m_g = evaluador.evaluar(pregunta, ids_grafo)
            m_e = evaluador.evaluar(pregunta, ids_emb)
            eval_grafo = {
                "precision": m_g.precision,
                "recall":    m_g.recall,
                "f1":        m_g.f1,
                "verdaderos_pos": m_g.verdaderos_pos,
                "relevantes":    m_g.relevantes,
            }
            eval_emb = {
                "precision": m_e.precision,
                "recall":    m_e.recall,
                "f1":        m_e.f1,
                "verdaderos_pos": m_e.verdaderos_pos,
                "relevantes":    m_e.relevantes,
            }

        resultado = {
            "pregunta": pregunta,
            "grafo": {
                "chunks":    [{"chunk_id": c.chunk_id, "titulo": c.titulo, "score": c.score} for c in chunks_grafo],
                "n_chunks":  len(chunks_grafo),
                "entidades": entidades,
                "hops":      hops,
                "tiempo_s":  t_grafo,
                "evaluacion": eval_grafo,
            },
            "embedding": {
                "chunks":   [{"chunk_id": c.chunk_id, "titulo": c.titulo, "score": c.score} for c in chunks_emb],
                "n_chunks": len(chunks_emb),
                "tiempo_s": t_emb,
                "evaluacion": eval_emb,
            },
            "overlap": {
                "comunes":        sorted(overlap),
                "solo_grafo":     sorted(set_g - set_e),
                "solo_embedding": sorted(set_e - set_g),
                "jaccard":        jaccard,
            },
        }
        resultados.append(resultado)
        _imprimir_pregunta(resultado)

    return resultados


def _imprimir_pregunta(r: Dict):
    g  = r["grafo"]
    e  = r["embedding"]
    ov = r["overlap"]
    print(f"\n  Grafo+NER  : {g['n_chunks']} chunks | entidades={g['entidades']} | hops={g['hops']} | {g['tiempo_s']}s")
    print(f"  Embedding  : {e['n_chunks']} chunks | {e['tiempo_s']}s")
    print(f"  Jaccard    : {ov['jaccard']:.3f}  |  comunes={ov['comunes']}")

    # métricas de evaluación si existen
    if g.get("evaluacion"):
        mg = g["evaluacion"]
        me = e["evaluacion"]
        print(f"\n  {'':15} {'P@K':>6}  {'R@K':>6}  {'F1@K':>6}  TP")
        print(f"  {'Grafo+NER':15} {mg['precision']:>6.3f}  {mg['recall']:>6.3f}  {mg['f1']:>6.3f}  {mg['verdaderos_pos']}")
        print(f"  {'Embedding':15} {me['precision']:>6.3f}  {me['recall']:>6.3f}  {me['f1']:>6.3f}  {me['verdaderos_pos']}")


def _imprimir_resumen_global(resultados: List[Dict]):
    print(f"\n{'='*60}")
    print("📊 RESUMEN GLOBAL")
    print("="*60)

    n             = len(resultados)
    jaccard_medio = round(sum(r["overlap"]["jaccard"] for r in resultados) / n, 3)
    t_grafo_total = round(sum(r["grafo"]["tiempo_s"]     for r in resultados), 2)
    t_emb_total   = round(sum(r["embedding"]["tiempo_s"] for r in resultados), 2)

    print(f"  Jaccard medio       : {jaccard_medio}")
    print(f"  Tiempo grafo total  : {t_grafo_total}s")
    print(f"  Tiempo emb. total   : {t_emb_total}s")

    # resumen evaluación
    evals_g = [r["grafo"]["evaluacion"]     for r in resultados if r["grafo"]["evaluacion"]]
    evals_e = [r["embedding"]["evaluacion"] for r in resultados if r["embedding"]["evaluacion"]]

    if evals_g:
        print(f"\n  {'':15} {'P@K':>6}  {'R@K':>6}  {'F1@K':>6}")
        print(f"  {'-'*38}")
        pg = round(sum(e["precision"] for e in evals_g) / len(evals_g), 3)
        rg = round(sum(e["recall"]    for e in evals_g) / len(evals_g), 3)
        fg = round(sum(e["f1"]        for e in evals_g) / len(evals_g), 3)
        pe = round(sum(e["precision"] for e in evals_e) / len(evals_e), 3)
        re_ = round(sum(e["recall"]   for e in evals_e) / len(evals_e), 3)
        fe = round(sum(e["f1"]        for e in evals_e) / len(evals_e), 3)
        print(f"  {'Grafo+NER (media)':20} {pg:>6.3f}  {rg:>6.3f}  {fg:>6.3f}")
        print(f"  {'Embedding (media)':20} {pe:>6.3f}  {re_:>6.3f}  {fe:>6.3f}")

    print(f"\n  {'#':<3} {'Jaccard':>8}  {'G-P':>6}  {'G-R':>6}  {'E-P':>6}  {'E-R':>6}  Pregunta")
    print(f"  {'-'*72}")
    for i, r in enumerate(resultados, 1):
        eg = r["grafo"]["evaluacion"]     or {}
        ee = r["embedding"]["evaluacion"] or {}
        print(
            f"  {i:<3} {r['overlap']['jaccard']:>8.3f}"
            f"  {eg.get('precision', '-'):>6}  {eg.get('recall', '-'):>6}"
            f"  {ee.get('precision', '-'):>6}  {ee.get('recall', '-'):>6}"
            f"  {r['pregunta'][:40]}"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔬 COMPARACIÓN: GRAFO+NER vs EMBEDDING  (v3 — con evaluación)")
    print("="*60)

    print("\n⚙️  Inicializando...")
    graph = get_neo4j_graph()

    print("📖 Cargando EntityMap...")
    entity_map = cargar_entity_map(RUTA_NER_JSON, RUTA_ACRONIMOS)
    entity_map.print_stats()

    retriever_grafo = RetrieverGrafo(
        graph=graph,
        entity_map=entity_map,
        verbose=True,
        max_chunks=8,
        max_hops=2,
        ruta_ontologia=RUTA_ONTOLOGIA,
        ruta_ner_json=RUTA_NER_JSON,
        ruta_acronimos=RUTA_ACRONIMOS,
    )
    retriever_embedding = RetrieverEmbedding(ruta_csv=RUTA_CSV, top_k=8, verbose=True)

    # Evaluador — opzionale: se il file non esiste, il confronto gira lo stesso
    evaluador = None
    if RUTA_GT.exists():
        print(f"📏 Ground truth caricata: {RUTA_GT}")
        evaluador = EvaluadorRetrieval(ruta_ground_truth=RUTA_GT)
    else:
        print(f"⚠️  Ground truth non trovata ({RUTA_GT}) — esecuzione senza metriche P/R")

    resultados = comparar(PREGUNTAS, retriever_grafo, retriever_embedding, evaluador)
    _imprimir_resumen_global(resultados)

    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Resultados guardados en {RUTA_SALIDA}")