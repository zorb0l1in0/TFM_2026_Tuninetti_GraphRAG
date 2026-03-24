"""
ontologia.py
------------
Due classi per la gestione dell'ontologia:

  HybridDocumentAnalyzer  — scopre nodi e relazioni candidati da un documento
                            (testo + tabelle) usando spaCy + regex
  CargadorOntologia       — carica un YAML di ontologia strutturata e lo
                            serializza per i prompt del pipeline NER
"""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import spacy
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# HybridDocumentAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class HybridDocumentAnalyzer:
    """
    Analizza documenti con parti testuali e tabulari per scoprire
    nodi e relazioni candidati da usare nell'ontologia.

    - Righe con il pattern "Fila de tabla" → elaborazione regex
    - Tutto il resto → elaborazione spaCy
    """

    def __init__(self, modelo_spacy: str = "es_core_news_lg", patron_tabla="Fila de tabla"):
        print(f"🔍 Cargando modelo spaCy: {modelo_spacy}...")
        self.nlp = spacy.load(modelo_spacy)

        self.patrones_tabla = [patron_tabla] if isinstance(patron_tabla, str) else patron_tabla

        self.blacklist_nodos = {
            'caso', 'número', 'vez', 'forma', 'tipo', 'nivel', 'punto',
            'parte', 'conjunto', 'total', 'objeto', 'efecto', 'modo',
            'base', 'paso', 'lugar', 'término', 'carácter', 'criterio',
            'condición', 'situación', 'proceso', 'sistema', 'estructura',
            'fecha', 'tiempo', 'plazo', 'periodo', 'duración', 'año',
            'artículo', 'párrafo', 'apartado', 'disposición', 'página',
            'tabla', 'fila', 'anexo',
        }

        self.blacklist_relaciones = {
            'ser', 'estar', 'haber', 'tener', 'hacer', 'poder', 'deber',
            'ir', 'venir', 'dar', 'decir', 'ver', 'saber', 'querer',
            'llevar', 'seguir', 'encontrar', 'llamar', 'parecer', 'quedar',
            'conducir', 'constituir', 'formar', 'considerar', 'unidad',
        }

        self.cliticos = {'él', 'la', 'lo', 'le', 'se', 'nos', 'os', 'me', 'te', 'ello'}

    # ── Separazione righe ─────────────────────────────────────────────────────

    def es_linea_tabla(self, linea: str) -> bool:
        linea_lower = linea.lower()
        return any(p.lower() in linea_lower for p in self.patrones_tabla)

    def separa_lineas(self, texto: str):
        lineas_tabla, lineas_texto = [], []
        for linea in texto.split("\n"):
            linea = linea.strip()
            if not linea:
                continue
            if self.es_linea_tabla(linea):
                lineas_tabla.append(linea)
            elif len(linea) > 15:
                lineas_texto.append(linea)
        return lineas_tabla, lineas_texto

    # ── Estrazione da tabelle ─────────────────────────────────────────────────

    def extrae_de_tabla(self, lineas_tabla) -> Counter:
        nodos = Counter()
        patron_nota     = r'Titulación de Grado es (.*?),\s*Nota de corte es (\d+,\d+)'
        patron_precio   = r'Precio en euros por crédito.*?es (\d+,\d+)'
        patron_generico = r'(\w+)\s+es\s+([^,\.]+)'

        for linea in lineas_tabla:
            for titulacion, _ in re.findall(patron_nota, linea):
                nodos[titulacion.strip().capitalize()] += 3
                nodos['Nota de corte'] += 2
                nodos['Grado'] += 1
            if re.findall(patron_precio, linea):
                nodos['Precio'] += 2
                nodos['Crédito'] += 1
            for concepto, _ in re.findall(patron_generico, linea):
                if len(concepto) > 3 and concepto.lower() not in self.blacklist_nodos:
                    nodos[concepto.capitalize()] += 1
        return nodos

    # ── Estrazione da testo ───────────────────────────────────────────────────

    def extrae_de_texto(self, lineas_texto, top_n: int = 15, min_frecuencia: int = 2):
        if not lineas_texto:
            return Counter(), Counter()

        texto = "\n".join(lineas_texto)
        texto = re.sub(r'[#*_`]', ' ', texto)
        texto = re.sub(r'\d+', ' ', texto)
        texto = re.sub(r'\n+', ' ', texto)

        print(f"📊 Analizando {len(lineas_texto)} líneas con spaCy...")
        doc = self.nlp(texto[:100000])

        contador_sustantivos      = Counter()
        sustantivos_estructurales = Counter()

        for token in doc:
            if token.pos_ == "NOUN" and not token.is_stop and len(token.text) > 3:
                lema = token.lemma_.lower()
                if lema not in self.blacklist_nodos:
                    contador_sustantivos[lema] += 1
                    if token.dep_ in ("nsubj", "obj", "iobj", "nsubjpass"):
                        sustantivos_estructurales[lema] += 1

        candidatos     = {p for p, _ in contador_sustantivos.most_common(top_n * 2)}
        verbos_dominio = Counter()

        for token in doc:
            if token.pos_ != "VERB":
                continue
            if token.lemma_.lower() in self.blacklist_relaciones or len(token.text) <= 3:
                continue
            if any(p in self.cliticos for p in token.lemma_.lower().split()):
                continue
            hijos  = list(token.children)
            sujeto = any(h.lemma_.lower() in candidatos for h in hijos if h.dep_ in ("nsubj", "nsubjpass"))
            objeto = any(h.lemma_.lower() in candidatos for h in hijos if h.dep_ in ("obj", "iobj"))
            if sujeto or objeto:
                verbos_dominio[token.lemma_.lower()] += 1

        puntuacion_nodos = Counter({
            p: freq + sustantivos_estructurales.get(p, 0) * 2
            for p, freq in contador_sustantivos.items()
            if freq >= min_frecuencia
        })

        return puntuacion_nodos, verbos_dominio

    # ── Analisi principale ────────────────────────────────────────────────────

    def analiza(
        self,
        ruta_archivo: Path,
        top_n: int = 15,
        min_frecuencia: int = 2,
        verbose: bool = True,
        threshold_percentual: float = 0.05,
    ) -> dict:
        if verbose:
            print(f"\n📄 Analizando: {ruta_archivo.name}")

        texto = ruta_archivo.read_text(encoding="utf-8")
        texto = re.sub(r'---.*?---', ' ', texto, flags=re.DOTALL)

        lineas_tabla, lineas_texto = self.separa_lineas(texto)
        if verbose:
            print(f"   📊 Líneas tabla: {len(lineas_tabla)} | 📝 Líneas texto: {len(lineas_texto)}")

        nodos_tabla               = self.extrae_de_tabla(lineas_tabla) if lineas_tabla else Counter()
        nodos_texto, relaciones_texto = self.extrae_de_texto(lineas_texto, top_n, min_frecuencia)

        nodos_combinados = nodos_texto.copy()
        for nodo, peso in nodos_tabla.items():
            nodos_combinados[nodo] += peso

        nodos_finales = []
        threshold     = 0
        max_score     = 0

        if nodos_combinados:
            max_score = max(nodos_combinados.values())
            threshold = max_score * threshold_percentual

            if verbose:
                print(f"\n📊 Filtro: max={max_score}, threshold={threshold:.1f} ({threshold_percentual*100}%)")

            filtrados = sorted(
                [(n, s) for n, s in nodos_combinados.items()
                 if s >= threshold and n.lower() not in self.blacklist_nodos and len(n) > 2],
                key=lambda x: x[1], reverse=True,
            )
            nodos_finales = self._normaliza_nodos([n.capitalize() for n, _ in filtrados])

        relaciones_finales = [
            r.upper()
            for r, freq in relaciones_texto.most_common(top_n)
            if freq >= min_frecuencia and r.lower() not in self.blacklist_relaciones
        ]

        if verbose:
            print(f"\n✅ Nodos finales: {len(nodos_finales)} | Relaciones: {len(relaciones_finales)}")

        return {
            "allowed_nodes":         nodos_finales,
            "allowed_relationships": relaciones_finales,
            "estadisticas": {
                "lineas_tabla":        len(lineas_tabla),
                "lineas_texto":        len(lineas_texto),
                "nodos_de_tabla":      len(nodos_tabla),
                "nodos_de_texto":      len(nodos_texto),
                "relaciones_de_texto": len(relaciones_texto),
                "threshold_usado":     threshold,
                "max_score":           max_score,
            },
            "nodos_completos":      nodos_combinados.most_common(30),
            "relaciones_completas": relaciones_texto.most_common(20),
        }

    def guardar_ontology_json(
        self,
        ruta_salida: Path,
        nodos: list,
        relaciones: list,
        archivo_fuente: str,
    ):
        """Salva nodi e relazioni candidati in un file ontology.json."""
        estructura = {
            "metadata": {
                "generated_from": archivo_fuente,
                "generator":      "HybridDocumentAnalyzer",
                "version":        "0.1.0",
            },
            "candidate_entities":  sorted(nodos),
            "candidate_relations": sorted(relaciones),
        }
        ruta_salida.write_text(
            json.dumps(estructura, indent=4, ensure_ascii=False), encoding="utf-8"
        )
        print(f"📁 ontology.json generado en: {ruta_salida}")

    # ── Helper ────────────────────────────────────────────────────────────────

    def _normaliza_nodos(self, nodos: list) -> list:
        risultato, vistos = [], set()
        for nodo in nodos:
            chiave = "".join(
                c for c in unicodedata.normalize("NFD", nodo.lower())
                if unicodedata.category(c) != "Mn"
            )
            if chiave not in vistos:
                vistos.add(chiave)
                risultato.append(nodo)
        return risultato


# ─────────────────────────────────────────────────────────────────────────────
# CargadorOntologia
# ─────────────────────────────────────────────────────────────────────────────

class CargadorOntologia:
    """
    Carica il YAML dell'ontologia strutturata e lo serializza
    per i prompt del pipeline NER a due passi.
    """

    def __init__(self, ruta_yaml: str):
        with open(ruta_yaml, "r", encoding="utf-8") as f:
            self.ontologia = yaml.safe_load(f)

    def descripcion_entidades(self) -> str:
        lineas = []
        for nombre, datos in self.ontologia.get("entities", {}).items():
            patrones = ", ".join(datos.get("patterns", []))
            desc     = datos.get("description", "")
            lineas.append(f"- **{nombre}**: {desc} (patrones: {patrones})")
        return "\n".join(lineas)

    def descripcion_relaciones(self) -> str:
        lineas = []
        for nombre, datos in self.ontologia.get("relations", {}).items():
            dominio = ", ".join(datos.get("domain", []))
            rango   = ", ".join(datos.get("range",  []))
            desc    = datos.get("description", "")
            lineas.append(
                f"- **{nombre}**: {desc}  [dominio: {dominio}] → [rango: {rango}]"
            )
        return "\n".join(lineas)

    def texto_restricciones(self) -> str:
        return "\n".join(f"  - {c}" for c in self.ontologia.get("constraints", []))

    def tipos_entidad_validos(self) -> list[str]:
        return list(self.ontologia.get("entities", {}).keys())

    def restricciones_relaciones(self) -> dict:
        result = {}
        for nombre, datos in self.ontologia.get("relations", {}).items():
            entrada = {"domain": datos.get("domain", [])}
            if "range" in datos:
                entrada["range"] = datos["range"]
            result[nombre] = entrada
        return result
    def descripcion_entidades_compacta(self) -> str:
        lineas = []
        for nombre, datos in self.ontologia.get("entities", {}).items():
            patrones = ", ".join(datos.get("patterns", []))
            lineas.append(f"- {nombre}: {patrones}")
        return "\n".join(lineas)

    def descripcion_relaciones_compacta(self) -> str:
        lineas = []
        for nombre, datos in self.ontologia.get("relations", {}).items():
            domain = ", ".join(datos.get("domain", []))
            rango  = ", ".join(datos.get("range",  []))
            lineas.append(f"- {nombre}: [{domain}] → [{rango}]")
        return "\n".join(lineas)


# ─────────────────────────────────────────────────────────────────────────────
# __main__
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    analizador = HybridDocumentAnalyzer(
        modelo_spacy="es_core_news_lg",
        patron_tabla=["Fila de tabla", "Tabla"],
    )

    ruta = Path("../../data/raw/reglamento_ull_simultaneidad_dobles_titulaciones.md")
    if ruta.exists():
        resultado = analizador.analiza(ruta, verbose=True, min_frecuencia=1)

        print("\n" + "=" * 50)
        print("RESULTADOS FINALES")
        print("=" * 50)
        print(f"\nallowed_nodes         = {resultado['allowed_nodes']}")
        print(f"allowed_relationships = {resultado['allowed_relationships']}")

        salida = Path("../../data/ner/ontologia/ontology.json")
        salida.parent.mkdir(parents=True, exist_ok=True)
        analizador.guardar_ontology_json(
            ruta_salida=salida,
            nodos=resultado["allowed_nodes"],
            relaciones=resultado["allowed_relationships"],
            archivo_fuente=ruta.name,
        )
    else:
        print(f"❌ File non trovato: {ruta}")