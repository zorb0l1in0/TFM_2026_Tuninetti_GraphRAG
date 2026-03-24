"""
entity_resolver.py  (optimizado)
----------------------------------
Fase A del enfoque GraphRAG: resolución y fusión de entidades
extraídas chunk a chunk por la pipeline NER.

Cambios respecto a la versión original:

  1. BATCH ÚNICO — una sola llamada a la API para TODAS las entidades
     de todos los tipos, en lugar de N llamadas (una por tipo).
     Reduce latencia y coste.

  2. CACHÉ EN DISCO — los embeddings se guardan en un archivo .json
     junto a ner_resultados.json. En ejecuciones posteriores se reutilizan
     si el texto canónico no ha cambiado. Solo los textos nuevos se
     re-embedan.

  3. BUCLE TRIANGULAR CORREGIDO — j empieza desde i+1, cada par se
     compara una sola vez. La versión original comparaba cada par dos veces.

  4. FILTRO ANTICIPADO — los textos con dígitos o números romanos se
     excluyen antes de pedir embeddings, reduciendo el payload de la API.

  Todo lo demás (exact merge, abbreviation merge, exclusiones de seguridad,
  estructura EntityMap/CanonicalEntity) permanece invariado.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml


# ── Tipos excluidos del embedding merge ───────────────────────────────────────
# Convocatoria: 'ordinaria' ≠ 'extraordinaria' aunque sean semánticamente cercanas
# Plaza: 'plazas vacantes' ≠ 'plazas ofertadas'
# Resolución: documentos normativos distintos con texto similar
# Fecha/Período: periodos temporales distintos
_TIPOS_NO_EMBEDDING: Set[str] = {
    "Período", "Fecha", "Convocatoria", "Resolución", "Plaza"
}

# Regex para detectar textos con dígitos arábigos o números romanos aislados.
# Excluidos del embedding merge para evitar fusionar entidades normativas distintas.
# Ej: 'Real Decreto 534/2024' ≠ 'Real Decreto 123/2020'
_RE_DIGITS = re.compile(r"\d")
_RE_ROMAN  = re.compile(r"\b(I{1,3}|IV|VI{0,3}|IX|X{1,3}|XL|L|XC|C)\b")

_RUTA_ACRONIMOS_DEFAULT = Path(__file__).parent / "acronimos.yaml"


def cargar_acronimos(ruta: Optional[Path] = None) -> Dict[str, str]:
    """
    Carga el diccionario de acrónimos desde un YAML externo.

    El YAML debe tener la estructura:
        acronimos:
          ULL: Universidad de La Laguna
          TFG: Trabajo de Fin de Grado

    Las claves se normalizan a minúsculas para que el lookup sea insensible
    a mayúsculas. Si el archivo no existe, devuelve un diccionario vacío.
    """
    ruta = Path(ruta) if ruta else _RUTA_ACRONIMOS_DEFAULT
    if not ruta.exists():
        print(f"⚠️  acronimos.yaml no encontrado en {ruta}. "
              f"Se continúa sin expansión de acrónimos.")
        return {}
    with open(ruta, encoding="utf-8") as f:
        datos = yaml.safe_load(f)
    acronimos = datos.get("acronimos", {}) or {}
    return {k.lower(): v for k, v in acronimos.items() if k and v}


# ── Dataclasses (sin cambios) ─────────────────────────────────────────────────

@dataclass
class EntityInstance:
    """Una única ocurrencia de entidad extraída de un chunk."""
    text:        str
    entity_type: str
    chunk_id:    str
    confidence:  str = "medium"
    descripcion: str = ""


@dataclass
class CanonicalEntity:
    """Nodo canónico que agrega una o más EntityInstance."""
    canonical_id:            str
    canonical_text:          str
    entity_type:             str
    instances:               List[EntityInstance] = field(default_factory=list)
    chunk_ids:               List[str]            = field(default_factory=list)
    descriptions:            List[str]            = field(default_factory=list)
    descripcion_consolidada: str                  = ""

    @property
    def n_chunks(self) -> int:
        return len(set(self.chunk_ids))

    @property
    def all_texts(self) -> List[str]:
        return list({i.text for i in self.instances})


# ── EntityMap (sin cambios) ───────────────────────────────────────────────────

class EntityMap:
    """
    Diccionario bidireccional:
      text_raw (minúsculas) → CanonicalEntity
      canonical_id          → CanonicalEntity
    """

    def __init__(self):
        self._by_text: Dict[str, CanonicalEntity] = {}
        self._by_id:   Dict[str, CanonicalEntity] = {}

    def lookup(self, text_raw: str) -> Optional[CanonicalEntity]:
        return self._by_text.get(text_raw.strip().lower())

    def canonical_id(self, text_raw: str) -> Optional[str]:
        ce = self.lookup(text_raw)
        return ce.canonical_id if ce else None

    def all_canonicals(self) -> List[CanonicalEntity]:
        return list(self._by_id.values())

    def register(self, text_lower: str, canonical: CanonicalEntity):
        self._by_text[text_lower] = canonical
        self._by_id[canonical.canonical_id] = canonical

    def stats(self) -> Dict[str, int]:
        raw_texts   = len(self._by_text)
        canonicals  = len(self._by_id)
        merged      = raw_texts - canonicals
        cross_chunk = sum(1 for c in self._by_id.values() if c.n_chunks > 1)
        return {
            "testi_raw":        raw_texts,
            "nodi_canonici":    canonicals,
            "fusioni_totali":   merged,
            "nodi_cross_chunk": cross_chunk,
        }

    def print_stats(self):
        s = self.stats()
        print("\n📊 Estadísticas de EntityMap:")
        print(f"   Textos raw (únicos):    {s['testi_raw']}")
        print(f"   Nodos canónicos:        {s['nodi_canonici']}")
        print(f"   Fusiones realizadas:    {s['fusioni_totali']}")
        print(f"   Nodos cross-chunk:      {s['nodi_cross_chunk']}")

    def print_merges(self, min_instances: int = 2):
        """Muestra los nodos canónicos que agrupan más de una variante."""
        print(f"\n🔗 Fusiones (canónicos con ≥{min_instances} variantes):")
        merged = [c for c in self._by_id.values() if len(c.all_texts) >= min_instances]
        merged.sort(key=lambda c: -len(c.instances))
        for ce in merged:
            variants = ce.all_texts
            print(f"\n  [{ce.entity_type}] '{ce.canonical_text}'  "
                  f"({len(ce.instances)} instancias, {ce.n_chunks} chunks)")
            for v in variants:
                if v != ce.canonical_text:
                    print(f"    ← '{v}'")


# ── CacheEmbeddings ───────────────────────────────────────────────────────────

class CacheEmbeddings:
    """
    Caché persistente en disco para embeddings.

    Formato del archivo JSON:
        { "texto_canónico": [0.123, -0.456, ...], ... }

    Lógica:
      - Al cargar, lee todos los embeddings ya calculados.
      - obtener_o_calcular() devuelve los cacheados y llama a la API
        solo para los textos que faltan, en un único batch.
      - Al terminar, guarda todo en disco.
    """

    def __init__(self, ruta_cache: Path, verbose: bool = True):
        self.ruta_cache = ruta_cache
        self.verbose    = verbose
        self._cache: Dict[str, List[float]] = {}
        self._cargar()

    def _cargar(self):
        if self.ruta_cache.exists():
            with open(self.ruta_cache, encoding="utf-8") as f:
                self._cache = json.load(f)
            if self.verbose:
                print(f"   💾 Caché de embeddings cargada: "
                      f"{len(self._cache)} entradas desde {self.ruta_cache.name}")
        else:
            if self.verbose:
                print(f"   💾 Caché no encontrada, se creará en {self.ruta_cache.name}")

    def guardar(self):
        with open(self.ruta_cache, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False)
        if self.verbose:
            print(f"   💾 Caché guardada: {len(self._cache)} entradas → {self.ruta_cache.name}")

    def obtener_o_calcular(self, textos: List[str]) -> Dict[str, List[float]]:
        """
        Devuelve un dict {texto: embedding} para todos los textos.
        Llama a la API solo para los textos que no están en caché,
        en un único batch.
        """
        from src.common.clients import get_embeddings

        faltantes = [t for t in textos if t not in self._cache]

        if faltantes:
            if self.verbose:
                print(f"   🌐 API de embeddings — {len(faltantes)} textos nuevos "
                      f"({len(textos) - len(faltantes)} ya en caché)")
            # UNA SOLA llamada a la API para todos los textos que faltan
            vectores = get_embeddings(faltantes)
            for texto, vec in zip(faltantes, vectores):
                self._cache[texto] = vec
        else:
            if self.verbose:
                print(f"   ✅ Todos los embeddings desde caché ({len(textos)} textos)")

        return {t: self._cache[t] for t in textos}


# ── EntityResolver ────────────────────────────────────────────────────────────

class EntityResolver:
    """
    Construye un EntityMap a partir de ner_resultados.json aplicando:
      1. Exact merge
      2. Abbreviation merge
      3. Embedding merge (optimizado: batch único + caché en disco)

    Uso típico:
        resolver = EntityResolver()
        entity_map = resolver.resolve("ner_resultados.json")
        entity_map.print_stats()
        entity_map.print_merges()
    """

    def __init__(
        self,
        embedding_threshold:  float = 0.92,
        usar_embedding_merge: bool  = True,
        min_text_length:      int   = 3,
        ruta_acronimos:       Optional[Path] = None,
        abbreviations:        Optional[Dict[str, str]] = None,
        verbose:              bool  = True,
    ):
        """
        Args:
            embedding_threshold:  umbral de similitud coseno (0–1).
                                  0.92 es conservador: evita fusiones incorrectas
                                  entre entidades semánticamente cercanas pero distintas.
            usar_embedding_merge: si False, omite la Fase 3 (útil para debug).
            min_text_length:      descarta textos demasiado cortos (ruido NER).
            ruta_acronimos:       ruta al YAML de acrónimos.
            abbreviations:        acrónimos adicionales en memoria (prioridad sobre YAML).
            verbose:              imprime log de cada fase.
        """
        self.embedding_threshold  = embedding_threshold
        self.usar_embedding_merge = usar_embedding_merge
        self.min_text_length      = min_text_length
        self.verbose              = verbose

        self.abbrev_map: Dict[str, str] = cargar_acronimos(ruta_acronimos)
        if abbreviations:
            self.abbrev_map.update({k.lower(): v for k, v in abbreviations.items()})

        if verbose:
            print(f"📖 Acrónimos cargados: {len(self.abbrev_map)}")

    # ── Punto de entrada ──────────────────────────────────────────────────────

    def resolve(self, ner_json_path: str) -> EntityMap:
        """
        Lee el JSON NER y construye el EntityMap completo.

        Returns:
            EntityMap lista para ser consumida por graph_builder.py
        """
        path = Path(ner_json_path)
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {path}")

        with open(path, encoding="utf-8") as f:
            chunks: List[Dict] = json.load(f)

        if self.verbose:
            print(f"\n🔍 EntityResolver — cargando {path.name}")
            print(f"   {len(chunks)} chunks, iniciando recopilación de instancias...")

        # Recopila todas las instancias
        instances = self._collect_instances(chunks)
        if self.verbose:
            print(f"   {len(instances)} instancias recopiladas")

        # Fase 1: exact merge
        entity_map = self._exact_merge(instances)
        if self.verbose:
            s = entity_map.stats()
            print(f"\n✅ Exact merge: {s['testi_raw']} textos → {s['nodi_canonici']} canónicos "
                  f"({s['fusioni_totali']} fusiones)")

        # Fase 2: abbreviation merge
        n_abbrev = self._abbreviation_merge(entity_map)
        if self.verbose:
            print(f"✅ Abbrev merge: {n_abbrev} fusiones")

        # Fase 3: embedding merge
        n_emb = 0
        if self.usar_embedding_merge:
            # La caché se guarda junto al archivo NER
            ruta_cache = path.parent / (path.stem + "_embedding_cache.json")
            n_emb = self._embedding_merge(entity_map, ruta_cache)
            if self.verbose:
                print(f"✅ Embedding merge (umbral={self.embedding_threshold}): "
                      f"{n_emb} fusiones")

        return entity_map

    # ── Fase 0: recopilación de instancias ───────────────────────────────────

    def _collect_instances(self, chunks: List[Dict]) -> List[EntityInstance]:
        instances = []
        for chunk in chunks:
            cid = chunk.get("chunk_id", "")
            for ent in chunk.get("entidades", []):
                text = ent.get("text", "").strip()
                if not text or len(text) < self.min_text_length:
                    continue
                instances.append(EntityInstance(
                    text        = text,
                    entity_type = ent.get("entity_type", "Desconocido"),
                    chunk_id    = cid,
                    confidence  = ent.get("confidence", "medium"),
                    descripcion = ent.get("descripcion", ""),
                ))
        return instances

    # ── Fase 1: exact merge ───────────────────────────────────────────────────

    def _exact_merge(self, instances: List[EntityInstance]) -> EntityMap:
        """
        Agrupa por (entity_type, text_lower).
        El texto canónico es la variante más frecuente.
        """
        groups: Dict[Tuple[str, str], List[EntityInstance]] = defaultdict(list)
        for inst in instances:
            key = (inst.entity_type, inst.text.lower())
            groups[key].append(inst)

        entity_map = EntityMap()

        for (etype, text_lower), insts in groups.items():
            variant_counter = Counter(i.text for i in insts)
            canonical_text  = variant_counter.most_common(1)[0][0]
            canonical_id    = _make_id(etype, canonical_text)

            existing = entity_map._by_id.get(canonical_id)
            if existing is None:
                ce = CanonicalEntity(
                    canonical_id   = canonical_id,
                    canonical_text = canonical_text,
                    entity_type    = etype,
                )
                entity_map._by_id[canonical_id] = ce
            else:
                ce = existing

            for inst in insts:
                ce.instances.append(inst)
                ce.chunk_ids.append(inst.chunk_id)
                if inst.descripcion and inst.descripcion not in ce.descriptions:
                    ce.descriptions.append(inst.descripcion)

            entity_map.register(text_lower, ce)

        return entity_map

    # ── Fase 2: abbreviation merge ────────────────────────────────────────────

    def _abbreviation_merge(self, entity_map: EntityMap) -> int:
        """
        Colapsa las abreviaturas conocidas hacia su texto canónico.
        Ej: 'ULL' → 'Universidad de La Laguna'
        """
        merges = 0
        for abbrev_lower, target_text in self.abbrev_map.items():
            abbrev_ce = entity_map.lookup(abbrev_lower)
            target_ce = entity_map.lookup(target_text.lower())

            if abbrev_ce is None:
                continue
            if target_ce is None:
                abbrev_ce.canonical_text = target_text
                entity_map.register(target_text.lower(), abbrev_ce)
                merges += 1
                continue
            if abbrev_ce.canonical_id == target_ce.canonical_id:
                continue

            self._merge_into(abbrev_ce, target_ce, entity_map)
            merges += 1

        return merges

    # ── Fase 3: embedding merge (OPTIMIZADO) ──────────────────────────────────

    def _embedding_merge(self, entity_map: EntityMap, ruta_cache: Path) -> int:
        """
        Fusiona entidades del mismo tipo cuya similitud coseno supera el umbral.

        Optimizaciones respecto al original:

        A) BATCH ÚNICO — recoge TODOS los textos candidatos de todos los tipos
           y hace UNA sola llamada a la API (o cero si todo está en caché).
           El original hacía una llamada por tipo.

        B) CACHÉ EN DISCO — los embeddings se persisten entre ejecuciones.
           Solo los textos nuevos se re-embedan.

        C) BUCLE TRIANGULAR — j empieza desde i+1, cada par se examina
           una sola vez. El original examinaba cada par dos veces.

        D) FILTRO ANTICIPADO — los textos con dígitos o números romanos
           se excluyen antes de pedir embeddings, reduciendo el payload.
        """
        try:
            import numpy as np
        except ImportError:
            if self.verbose:
                print("  ⚠ numpy no disponible, embedding merge omitido")
            return 0

        # ── Recopilación de candidatos por tipo (con filtro anticipado) ───────
        # Excluimos tipos peligrosos y textos con dígitos/romanos ANTES
        # de construir el payload de la API, no solo antes de comparar.
        by_type: Dict[str, List[CanonicalEntity]] = defaultdict(list)
        for ce in entity_map.all_canonicals():
            if ce.entity_type in _TIPOS_NO_EMBEDDING:
                continue
            t = ce.canonical_text
            if _RE_DIGITS.search(t) or _RE_ROMAN.search(t):
                continue
            by_type[ce.entity_type].append(ce)

        # Solo tipos con al menos 2 candidatos (si no, no hay nada que comparar)
        by_type = {k: v for k, v in by_type.items() if len(v) >= 2}

        if not by_type:
            if self.verbose:
                print("  ℹ️  Ningún tipo con ≥2 candidatos, embedding merge omitido")
            return 0

        # ── Batch único: todos los textos únicos a embeddar ───────────────────
        todos_los_textos: List[str] = list({
            ce.canonical_text
            for candidatos in by_type.values()
            for ce in candidatos
        })

        if self.verbose:
            total_candidatos = sum(len(v) for v in by_type.values())
            print(f"\n  📐 Embedding merge — {total_candidatos} candidatos, "
                  f"{len(todos_los_textos)} textos únicos, {len(by_type)} tipos")

        # UNA llamada a la API (o cero si todo está en caché)
        cache = CacheEmbeddings(ruta_cache, verbose=self.verbose)
        texto_a_vec = cache.obtener_o_calcular(todos_los_textos)
        cache.guardar()

        # ── Comparación por tipo con bucle triangular ─────────────────────────
        fusiones    = 0
        merged_ids: Set[str] = set()

        for etype, candidatos in by_type.items():
            # Ordena por frecuencia desc: el más frecuente es el destino de la fusión
            candidatos.sort(key=lambda c: -len(c.instances))
            n = len(candidatos)

            # Matriz normalizada para el tipo actual
            mat = np.array(
                [texto_a_vec[ce.canonical_text] for ce in candidatos],
                dtype=np.float32
            )
            norms    = np.linalg.norm(mat, axis=1, keepdims=True)
            mat_norm = mat / (norms + 1e-9)

            for i in range(n):
                ce_a = candidatos[i]
                if ce_a.canonical_id in merged_ids:
                    continue

                # ▼ BUCLE TRIANGULAR: j empieza desde i+1 ▼
                for j in range(i + 1, n):
                    ce_b = candidatos[j]
                    if ce_b.canonical_id in merged_ids:
                        continue
                    if ce_a.canonical_id == ce_b.canonical_id:
                        continue

                    sim = float(np.dot(mat_norm[i], mat_norm[j]))

                    if sim >= self.embedding_threshold:
                        if self.verbose:
                            print(f"  🔗 Embedding merge [{etype}] sim={sim:.3f}")
                            print(f"     '{ce_a.canonical_text}'")
                            print(f"     ← '{ce_b.canonical_text}'")
                        # ce_b (menos frecuente, j > i) se fusiona en ce_a
                        self._merge_into(ce_b, ce_a, entity_map)
                        merged_ids.add(ce_b.canonical_id)
                        fusiones += 1

        return fusiones

    # ── Utilidad de fusión (sin cambios) ──────────────────────────────────────

    def _merge_into(
        self,
        source: CanonicalEntity,
        target: CanonicalEntity,
        entity_map: EntityMap,
    ):
        """
        Mueve todas las instancias de `source` a `target`.
        Actualiza todos los punteros del entity_map que apuntaban a source.
        """
        target.instances.extend(source.instances)
        target.chunk_ids.extend(source.chunk_ids)
        for desc in source.descriptions:
            if desc and desc not in target.descriptions:
                target.descriptions.append(desc)

        # Redirige todos los alias de source → target
        for text_lower, ce in list(entity_map._by_text.items()):
            if ce.canonical_id == source.canonical_id:
                entity_map._by_text[text_lower] = target

        # Elimina source del índice por id
        entity_map._by_id.pop(source.canonical_id, None)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_id(entity_type: str, text: str) -> str:
    """ID determinístico y estable: hash SHA1 truncado de type+text_lower."""
    raw = f"{entity_type}::{text.strip().lower()}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


# ── CLI / debug ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    json_path      = sys.argv[1] if len(sys.argv) > 1 else "ner_resultados.json"
    ruta_acronimos = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    resolver = EntityResolver(
        embedding_threshold=0.92,
        usar_embedding_merge=True,
        ruta_acronimos=ruta_acronimos,
        verbose=True,
    )
    entity_map = resolver.resolve(json_path)

    entity_map.print_stats()
    entity_map.print_merges(min_instances=1)

    # Ejemplo de lookup
    print("\n🔎 Ejemplo de lookup:")
    for test in ["Universidad de la Laguna", "ULL", "enseñanzas oficiales de Grado"]:
        ce = entity_map.lookup(test)
        if ce:
            print(f"  '{test}' → [{ce.entity_type}] '{ce.canonical_text}' "
                  f"(id={ce.canonical_id}, {ce.n_chunks} chunks)")
        else:
            print(f"  '{test}' → no encontrado")

    # Guarda el mapa en JSON para inspección
    out = {
        ce.canonical_id: {
            "canonical_text": ce.canonical_text,
            "entity_type":    ce.entity_type,
            "n_instances":    len(ce.instances),
            "n_chunks":       ce.n_chunks,
            "all_texts":      ce.all_texts,
            "descriptions":   ce.descriptions,
        }
        for ce in entity_map.all_canonicals()
    }
    out_path = Path(json_path).parent / "entity_map.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n💾 entity_map.json guardado en {out_path}")