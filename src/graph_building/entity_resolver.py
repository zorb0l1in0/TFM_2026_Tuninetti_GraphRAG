"""
entity_resolver.py
------------------
Fase A del enfoque GraphRAG: resolución y fusión de entidades
extraídas chunk a chunk por la pipeline NER.

Problema que resuelve:
  La pipeline NER procesa cada chunk de forma aislada → la misma entidad
  ("Universidad de La Laguna", "Universidad de la Laguna", "ULL")
  genera 3 nodos separados en el grafo. Este módulo los colapsa en un
  único nodo canónico ANTES de que graph_builder.py ejecute los MERGE en Neo4j.

Pipeline:
  1. Recopila todas las entidades de los chunks (622 instancias → 395 textos únicos)
  2. Exact merge    : insensible a mayúsculas/minúsculas, siempre seguro
  3. Abbrev merge   : mapa de abreviaturas conocidas (ULL, PADG, ...)
  4. Fuzzy merge    : SequenceMatcher ratio > umbral, solo mismo tipo,
                      excluye Período y textos con dígitos (fechas peligrosas)
  5. Produce EntityMap: text_raw → CanonicalEntity

Salida principal:
  EntityMap.lookup(text_raw) → CanonicalEntity
  EntityMap.canonical_id(text_raw) → str  (usado por graph_builder.py)
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml


# ── Tipos excluidos del fuzzy matching ────────────────────────────────────────
# Período contiene fechas específicas: '14 de mayo' ≈ '8 de mayo' (ratio 0.91)
# pero son entidades DISTINTAS. El mismo riesgo aplica a Convocatoria ordinaria
# vs extraordinaria. Resolución: 'ordinaria' ≠ 'extraordinaria' aunque el ratio
# sea alto.
_TIPOS_NO_FUZZY: Set[str] = {"Período", "Fecha", "Convocatoria", "Resolución"}

# Regex para detectar textos con dígitos arábigos O números romanos aislados.
# Ej: 'Anexo II', 'Anexo III', 'artículo 44', '2026-2027'
# Estos textos se excluyen del fuzzy merge para evitar fusiones incorrectas.
_RE_DIGITS = re.compile(r"\d")
_RE_ROMAN  = re.compile(r"\b(I{1,3}|IV|VI{0,3}|IX|X{1,3}|XL|L|XC|C)\b")

# ── Ruta por defecto del YAML de acrónimos ────────────────────────────────────
# Puede sobreescribirse pasando ruta_acronimos al constructor de EntityResolver.
_RUTA_ACRONIMOS_DEFAULT = Path(__file__).parent / "acronimos.yaml"


def cargar_acronimos(ruta: Optional[Path] = None) -> Dict[str, str]:
    """
    Carga el diccionario de acrónimos desde un YAML externo.

    El YAML debe tener la estructura:
        acronimos:
          ULL: Universidad de La Laguna
          TFG: Trabajo de Fin de Grado

    Si el archivo no existe, devuelve un diccionario vacío y avisa por consola.
    Las claves se normalizan a minúsculas para que el lookup sea insensible
    a mayúsculas.

    Args:
        ruta: Ruta al archivo YAML. Si es None usa _RUTA_ACRONIMOS_DEFAULT.

    Returns:
        Dict con las abreviaturas en minúsculas como claves.
    """
    ruta = Path(ruta) if ruta else _RUTA_ACRONIMOS_DEFAULT

    if not ruta.exists():
        print(f"⚠️  acronimos.yaml no encontrado en {ruta}. "
              f"Se continúa sin expansión de acrónimos.")
        return {}

    with open(ruta, encoding="utf-8") as f:
        datos = yaml.safe_load(f)

    acronimos = datos.get("acronimos", {}) or {}
    # Normaliza claves a minúsculas
    return {k.lower(): v for k, v in acronimos.items() if k and v}


# ── Dataclasses ───────────────────────────────────────────────────────────────

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
    canonical_id:   str                      # hash determinístico
    canonical_text: str                      # texto representativo
    entity_type:    str
    instances:      List[EntityInstance] = field(default_factory=list)
    chunk_ids:      List[str]            = field(default_factory=list)
    descriptions:   List[str]            = field(default_factory=list)
    # Rellenado por la Fase B (summarización con LLM)
    descripcion_consolidada: str = ""

    @property
    def n_chunks(self) -> int:
        return len(set(self.chunk_ids))

    @property
    def all_texts(self) -> List[str]:
        return list({i.text for i in self.instances})


# ── EntityMap ─────────────────────────────────────────────────────────────────

class EntityMap:
    """
    Diccionario bidireccional:
      text_raw (minúsculas) → CanonicalEntity
      canonical_id          → CanonicalEntity
    """

    def __init__(self):
        self._by_text: Dict[str, CanonicalEntity] = {}   # text_lower → canónico
        self._by_id:   Dict[str, CanonicalEntity] = {}   # canonical_id → canónico

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
        """Muestra todos los nodos canónicos que agrupan más de una variante."""
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


# ── EntityResolver ────────────────────────────────────────────────────────────

class EntityResolver:
    """
    Construye un EntityMap a partir de ner_resultados.json aplicando
    exact merge, abbreviation merge y fuzzy merge.

    Uso típico:
        resolver = EntityResolver()
        entity_map = resolver.resolve("ner_resultados.json")
        entity_map.print_stats()
        entity_map.print_merges()

        # En graph_builder.py:
        cid = entity_map.canonical_id("Universidad de la Laguna")
    """

    def __init__(
        self,
        fuzzy_threshold:    float = 0.93,
        min_text_length:    int   = 3,
        ruta_acronimos:     Optional[Path] = None,
        abbreviations:      Optional[Dict[str, str]] = None,
        verbose:            bool  = True,
    ):
        """
        Args:
            fuzzy_threshold:  umbral de SequenceMatcher (0–1). 0.93 es conservador:
                              evita fusiones incorrectas en fechas y variantes semánticas.
            min_text_length:  descarta textos demasiado cortos (ruido de la NER).
            ruta_acronimos:   ruta al YAML de acrónimos. Si es None busca
                              acronimos.yaml en el mismo directorio que este módulo.
            abbreviations:    acrónimos adicionales como dict en memoria.
                              Tienen prioridad sobre el YAML si hay conflictos.
            verbose:          imprime log de cada fase.
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.min_text_length = min_text_length
        self.verbose         = verbose

        # Carga el YAML como base
        self.abbrev_map: Dict[str, str] = cargar_acronimos(ruta_acronimos)

        # Los acrónimos pasados en memoria sobreescriben el YAML
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

        # Fase 1: exact merge (insensible a mayúsculas/minúsculas)
        entity_map = self._exact_merge(instances)
        if self.verbose:
            s = entity_map.stats()
            print(f"\n✅ Exact merge: {s['testi_raw']} textos → {s['nodi_canonici']} canónicos "
                  f"({s['fusioni_totali']} fusiones)")

        # Fase 2: abbreviation merge
        n_abbrev = self._abbreviation_merge(entity_map)
        if self.verbose:
            print(f"✅ Abbrev merge: {n_abbrev} fusiones")

        # Fase 3: fuzzy merge
        n_fuzzy = self._fuzzy_merge(entity_map)
        if self.verbose:
            print(f"✅ Fuzzy merge (umbral={self.fuzzy_threshold}): {n_fuzzy} fusiones")

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
        El texto canónico es la variante más frecuente; en caso de empate, la más larga.
        """
        # Agrupa: clave = (entity_type, text_lower) → lista de instancias
        groups: Dict[Tuple[str, str], List[EntityInstance]] = defaultdict(list)
        for inst in instances:
            key = (inst.entity_type, inst.text.lower())
            groups[key].append(inst)

        entity_map = EntityMap()

        for (etype, text_lower), insts in groups.items():
            # Texto canónico = variante más frecuente
            variant_counter = Counter(i.text for i in insts)
            canonical_text  = variant_counter.most_common(1)[0][0]

            # ID determinístico: hash de (entity_type + canonical_text_lower)
            canonical_id = _make_id(etype, canonical_text)

            # Recupera o crea el nodo canónico
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

            # Añade instancias
            for inst in insts:
                ce.instances.append(inst)
                ce.chunk_ids.append(inst.chunk_id)
                if inst.descripcion and inst.descripcion not in ce.descriptions:
                    ce.descriptions.append(inst.descripcion)

            # Registra todos los alias → mismo canónico
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
            abbrev_ce  = entity_map.lookup(abbrev_lower)
            target_ce  = entity_map.lookup(target_text.lower())

            if abbrev_ce is None:
                continue  # abreviatura no presente en el corpus

            if target_ce is None:
                # El texto expandido no existe: renombra el canónico de la abreviatura
                abbrev_ce.canonical_text = target_text
                entity_map.register(target_text.lower(), abbrev_ce)
                merges += 1
                continue

            if abbrev_ce.canonical_id == target_ce.canonical_id:
                continue  # ya fusionados

            # Fusiona abbrev_ce → target_ce (target queda como canónico)
            self._merge_into(abbrev_ce, target_ce, entity_map)
            merges += 1

        return merges

    # ── Fase 3: fuzzy merge ───────────────────────────────────────────────────

    def _fuzzy_merge(self, entity_map: EntityMap) -> int:
        """
        Compara por pares los nodos canónicos del mismo entity_type.
        Fusiona aquellos con SequenceMatcher ratio > fuzzy_threshold,
        excluyendo Período y textos con dígitos (fechas peligrosas).
        """
        merges = 0

        # Agrupa canónicos por entity_type
        by_type: Dict[str, List[CanonicalEntity]] = defaultdict(list)
        for ce in entity_map.all_canonicals():
            by_type[ce.entity_type].append(ce)

        for etype, canonicals in by_type.items():
            if etype in _TIPOS_NO_FUZZY:
                continue
            if len(canonicals) < 2:
                continue

            # Ordena por frecuencia descendente (el más frecuente se convierte en destino)
            canonicals.sort(key=lambda c: -len(c.instances))

            merged_ids: Set[str] = set()

            for i, ce_a in enumerate(canonicals):
                if ce_a.canonical_id in merged_ids:
                    continue
                for ce_b in canonicals[i + 1:]:
                    if ce_b.canonical_id in merged_ids:
                        continue
                    if self._should_fuzzy_merge(ce_a, ce_b):
                        # Fusiona ce_b → ce_a (ce_a es el más frecuente)
                        self._merge_into(ce_b, ce_a, entity_map)
                        merged_ids.add(ce_b.canonical_id)
                        merges += 1

        return merges

    def _should_fuzzy_merge(self, ce_a: CanonicalEntity, ce_b: CanonicalEntity) -> bool:
        """
        Decide si dos nodos canónicos deben fusionarse.
        Criterios:
          - mismo entity_type
          - ningún texto contiene dígitos (evita mezclar fechas/artículos)
          - ratio SequenceMatcher > fuzzy_threshold
        """
        if ce_a.entity_type != ce_b.entity_type:
            return False

        t_a = ce_a.canonical_text.lower()
        t_b = ce_b.canonical_text.lower()

        # Evita la fusión si alguno de los textos contiene dígitos arábigos o romanos.
        # Ej: 'Anexo II' ≠ 'Anexo III', 'artículo 44' ≠ 'artículo 45'.
        # Nota: el control de dígitos usa t_a/t_b (minúsculas); el de romanos
        # usa los textos canónicos originales (sensible a mayúsculas, los romanos van en mayúscula).
        if _RE_DIGITS.search(t_a) or _RE_DIGITS.search(t_b):
            return False
        if _RE_ROMAN.search(ce_a.canonical_text) or _RE_ROMAN.search(ce_b.canonical_text):
            return False

        # Evita la fusión de textos muy cortos (alta probabilidad de falsos positivos)
        if len(t_a) < 8 or len(t_b) < 8:
            return False

        ratio = SequenceMatcher(None, t_a, t_b).ratio()
        return ratio >= self.fuzzy_threshold

    # ── Utilidad de fusión ────────────────────────────────────────────────────

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
        # Transfiere instancias, chunk_ids, descripciones
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

    resolver   = EntityResolver(
        fuzzy_threshold=0.93,
        ruta_acronimos=ruta_acronimos,
        verbose=True,
    )
    entity_map = resolver.resolve(json_path)

    entity_map.print_stats()
    entity_map.print_merges(min_instances=2)

    # Ejemplo de lookup
    print("\n🔎 Ejemplo de lookup:")
    for test in ["Universidad de la Laguna", "ULL", "PADG", "doble titulación"]:
        ce = entity_map.lookup(test)
        if ce:
            print(f"  '{test}' → [{ce.entity_type}] '{ce.canonical_text}' "
                  f"(id={ce.canonical_id}, {ce.n_chunks} chunks)")
        else:
            print(f"  '{test}' → no encontrado")

    # Guarda el mapa en JSON para inspección
    out = {}
    for ce in entity_map.all_canonicals():
        out[ce.canonical_id] = {
            "canonical_text": ce.canonical_text,
            "entity_type":    ce.entity_type,
            "n_instances":    len(ce.instances),
            "n_chunks":       ce.n_chunks,
            "all_texts":      ce.all_texts,
            "descriptions":   ce.descriptions,
        }
    out_path = Path(json_path).parent / "entity_map.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n💾 entity_map.json guardado en {out_path}")