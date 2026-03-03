import re
from collections import Counter
from pathlib import Path
import spacy
import unicodedata  # Importado aquí para usarlo en toda la clase

class HybridDocumentAnalyzer:
    """
    Analizador híbrido que gestiona documentos con partes textuales y tabulares.
    - Líneas con "Fila de tabla" se procesan como tablas (regex)
    - Todo lo demás se procesa con spaCy (método original)
    """

    def __init__(self, modelo_spacy="es_core_news_lg", patron_tabla="Fila de tabla"):
        """
        Inicializa el analizador con el modelo spaCy y el patrón para reconocer tablas.

        Args:
            modelo_spacy: nombre del modelo spaCy a cargar
            patron_tabla: string o lista de strings que identifican líneas de tabla
        """
        print(f"🔍 Cargando modelo spaCy: {modelo_spacy}...")
        self.nlp = spacy.load(modelo_spacy)

        # Patrones para reconocer líneas de tabla
        if isinstance(patron_tabla, str):
            self.patrones_tabla = [patron_tabla]
        else:
            self.patrones_tabla = patron_tabla

        # Blacklist para nodos demasiado genéricos
        self.blacklist_nodos = {
            'caso', 'número', 'vez', 'forma', 'tipo', 'nivel', 'punto',
            'parte', 'conjunto', 'total', 'objeto', 'efecto', 'modo',
            'base', 'paso', 'lugar', 'término', 'carácter', 'criterio',
            'condición', 'situación', 'proceso', 'sistema', 'estructura',
            'fecha', 'tiempo', 'plazo', 'periodo', 'duración', 'año',
            'artículo', 'párrafo', 'apartado', 'disposición', 'página',
            'tabla', 'fila', 'anexo'  # Añadidos términos técnicos del documento
        }

        # Blacklist para verbos demasiado genéricos
        self.blacklist_relaciones = {
            'ser', 'estar', 'haber', 'tener', 'hacer', 'poder', 'deber',
            'ir', 'venir', 'dar', 'decir', 'ver', 'saber', 'querer',
            'llevar', 'seguir', 'encontrar', 'llamar', 'parecer', 'quedar',
            'conducir', 'constituir', 'formar', 'considerar', 'unidad'
        }

        self.cliticos = {'él', 'la', 'lo', 'le', 'se', 'nos', 'os', 'me', 'te', 'ello'}

    def es_linea_tabla(self, linea):
        """
        Verifica si una línea es una línea de tabla según los patrones configurados.

        Args:
            linea: string a verificar

        Returns:
            True si la línea corresponde a un patrón de tabla
        """
        linea_lower = linea.lower()
        for patron in self.patrones_tabla:
            if patron.lower() in linea_lower:
                return True
        return False

    def separa_lineas(self, texto):
        """
        Separa las líneas del documento en líneas de tabla y líneas de texto.

        Args:
            texto: string completo del documento

        Returns:
            tuple: (lineas_tabla, lineas_texto) como listas de strings
        """
        lineas = texto.split('\n')
        lineas_tabla = []
        lineas_texto = []

        for linea in lineas:
            linea_limpia = linea.strip()
            if not linea_limpia:  # Salta líneas vacías
                continue

            if self.es_linea_tabla(linea_limpia):
                lineas_tabla.append(linea_limpia)
            else:
                # Ignora líneas demasiado cortas o probables artefactos
                if len(linea_limpia) > 15:  # Umbral mínimo para texto significativo
                    lineas_texto.append(linea_limpia)

        return lineas_tabla, lineas_texto

    def extrae_de_tabla(self, lineas_tabla):
        """
        Extrae nodos de las líneas de tabla usando patrones regex específicos.
        Adaptado al formato "Fila de tabla ... Titulación de Grado es X, Nota de corte es Y"

        Args:
            lineas_tabla: lista de líneas de tabla

        Returns:
            Counter: nodos extraídos de las tablas con sus pesos
        """
        nodos = Counter()

        # Patrón para extraer titulaciones y notas
        patron_nota = r'Titulación de Grado es (.*?),\s*Nota de corte es (\d+,\d+)'
        # Patrón para extraer precios (del decreto)
        patron_precio = r'Precio en euros por crédito.*?es (\d+,\d+)'
        # Patrón genérico para extraer conceptos clave con "es"
        patron_generico = r'(\w+)\s+es\s+([^,\.]+)'

        for linea in lineas_tabla:
            # Busca titulaciones y notas de corte
            matches = re.findall(patron_nota, linea)
            for titulacion, nota in matches:
                # Limpia y añade nodos
                titulacion_limpia = titulacion.strip().capitalize()
                nodos[titulacion_limpia] += 3  # Peso alto para datos explícitos
                nodos['Nota de corte'] += 2
                nodos['Grado'] += 1

            # Busca precios
            matches_precio = re.findall(patron_precio, linea)
            for precio in matches_precio:
                nodos['Precio'] += 2
                nodos['Crédito'] += 1

            # Busca otros conceptos
            matches_generico = re.findall(patron_generico, linea)
            for concepto, valor in matches_generico:
                if len(concepto) > 3 and concepto.lower() not in self.blacklist_nodos:
                    nodos[concepto.capitalize()] += 1

        return nodos

    def extrae_de_texto(self, lineas_texto, top_n=15, min_frecuencia=2):
        """
        Extrae nodos y relaciones del texto usando spaCy.
        Este es tu método original adaptado.

        Args:
            lineas_texto: lista de líneas de texto
            top_n: número máximo de nodos/relaciones a devolver
            min_frecuencia: frecuencia mínima para considerar un término

        Returns:
            tuple: (contador_nodos, contador_relaciones)
        """
        if not lineas_texto:
            return Counter(), Counter()

        texto_completo = "\n".join(lineas_texto)

        # Limpieza básica
        texto_completo = re.sub(r'[#*_`]', ' ', texto_completo)
        texto_completo = re.sub(r'\d+', ' ', texto_completo)
        texto_completo = re.sub(r'\n+', ' ', texto_completo)

        print(f"📊 Analizando {len(lineas_texto)} líneas de texto con spaCy...")
        doc = self.nlp(texto_completo[:100000])  # Límite por rendimiento

        # Contadores
        contador_sustantivos = Counter()
        sustantivos_estructurales = Counter()

        # Primera pasada: sustantivos
        for token in doc:
            if token.pos_ == "NOUN" and not token.is_stop and len(token.text) > 3:
                lema = token.lemma_.lower()
                if lema not in self.blacklist_nodos:
                    contador_sustantivos[lema] += 1

                    # Bonus para sujetos y objetos
                    if token.dep_ in ("nsubj", "obj", "iobj", "nsubjpass"):
                        sustantivos_estructurales[lema] += 1

        # Segunda pasada: verbos que conectan entidades
        candidatos_nodos = {p for p, _ in contador_sustantivos.most_common(top_n * 2)}
        verbos_dominio = Counter()

        for token in doc:
            if token.pos_ != "VERB":
                continue
            if token.lemma_.lower() in self.blacklist_relaciones:
                continue
            if len(token.text) <= 3:
                continue
            if any(parte in self.cliticos for parte in token.lemma_.lower().split()):
                continue

            # Verifica si conecta entidades del dominio
            hijos = list(token.children)
            sujeto = any(
                h.lemma_.lower() in candidatos_nodos
                for h in hijos
                if h.dep_ in ("nsubj", "nsubjpass")
            )
            objeto = any(
                h.lemma_.lower() in candidatos_nodos
                for h in hijos
                if h.dep_ in ("obj", "iobj")
            )

            if sujeto or objeto:
                verbos_dominio[token.lemma_.lower()] += 1

        # Calcula puntuación final para nodos
        puntuacion_nodos = Counter()
        for palabra, freq in contador_sustantivos.items():
            if freq < min_frecuencia:
                continue
            puntuacion = freq
            puntuacion += sustantivos_estructurales.get(palabra, 0) * 2
            puntuacion_nodos[palabra] = puntuacion

        return puntuacion_nodos, verbos_dominio

    def _normaliza_nodos(self, nodos):
        """Normaliza y elimina duplicados de la lista de nodos"""
        nodos_normalizados = []
        vistos = set()

        for nodo in nodos:
            # Normaliza: quita acentos, pasa a minúsculas para comparar
            nodo_sin_acentos = ''.join(
                c for c in unicodedata.normalize('NFD', nodo.lower())
                if unicodedata.category(c) != 'Mn'
            )

            if nodo_sin_acentos not in vistos:
                vistos.add(nodo_sin_acentos)
                nodos_normalizados.append(nodo)

        return nodos_normalizados

    def analiza(self, ruta_archivo, top_n=15, min_frecuencia=2, verbose=True, threshold_percentual=0.05):
        """
        Método principal: analiza un documento separando tablas y texto.

        Args:
            threshold_percentual: porcentaje del máximo para filtrar nodos (0.05 = 5%)
        """
        if verbose:
            print(f"\n📄 Analizando: {ruta_archivo.name}")

        # Lee archivo
        texto = ruta_archivo.read_text(encoding="utf-8")

        # Elimina markdown/metadata
        texto = re.sub(r'---.*?---', ' ', texto, flags=re.DOTALL)

        # Separa líneas
        lineas_tabla, lineas_texto = self.separa_lineas(texto)

        if verbose:
            print(f"   📊 Líneas tabla: {len(lineas_tabla)}")
            print(f"   📝 Líneas texto: {len(lineas_texto)}")

        # Extrae de tablas
        nodos_tabla = Counter()
        if lineas_tabla:
            nodos_tabla = self.extrae_de_tabla(lineas_tabla)
            if verbose:
                print(f"   ✅ Extraídos {len(nodos_tabla)} nodos de las tablas")

        # Extrae de texto
        nodos_texto, relaciones_texto = self.extrae_de_texto(
            lineas_texto, top_n, min_frecuencia
        )

        if verbose and lineas_texto:
            print(f"   ✅ Extraídos {len(nodos_texto)} nodos y {len(relaciones_texto)} relaciones del texto")

        # Combina nodos (dando más peso a los de tabla)
        nodos_combinados = nodos_texto.copy()
        for nodo, peso in nodos_tabla.items():
            nodos_combinados[nodo] += peso

        # 👇 NUEVO: FILTRO PERCENTUALE ADATTIVO
        if nodos_combinados:
            max_score = max(nodos_combinados.values())
            threshold = max_score * threshold_percentual

            if verbose:
                print(
                    f"\n📊 Filtro percentuale: max={max_score}, threshold={threshold:.1f} ({threshold_percentual * 100}%)")

            # Filtra per soglia percentuale
            nodos_filtrados = [
                (n, s) for n, s in nodos_combinados.items()
                if s >= threshold and n.lower() not in self.blacklist_nodos and len(n) > 2
            ]

            # Ordina per score decrescente
            nodos_filtrados.sort(key=lambda x: x[1], reverse=True)

            nodos_finales = [n.capitalize() for n, _ in nodos_filtrados]

            if verbose:
                print(f"   Nodi dopo filtro: {len(nodos_finales)}")
                print(f"   Scores: {[s for _, s in nodos_filtrados[:5]]}...")
        else:
            nodos_finales = []

        # Filtra y formatea relaciones (igual que antes)
        relaciones_finales = [
            r.upper()
            for r, freq in relaciones_texto.most_common(top_n)
            if freq >= min_frecuencia and r.lower() not in self.blacklist_relaciones
        ]

        # Estadísticas
        estadisticas = {
            'lineas_tabla': len(lineas_tabla),
            'lineas_texto': len(lineas_texto),
            'nodos_de_tabla': len(nodos_tabla),
            'nodos_de_texto': len(nodos_texto),
            'relaciones_de_texto': len(relaciones_texto),
            'threshold_usado': threshold if nodos_combinados else 0,
            'max_score': max_score if nodos_combinados else 0
        }

        if verbose:
            print(f"\n✅ Análisis completado")
            print(f"   Nodos finales: {len(nodos_finales)}")
            print(f"   Relaciones finales: {len(relaciones_finales)}")

        return {
            'allowed_nodes': nodos_finales,
            'allowed_relationships': relaciones_finales,
            'estadisticas': estadisticas,
            'nodos_completos': nodos_combinados.most_common(30),
            'relaciones_completas': relaciones_texto.most_common(20)
        }


# Ejemplo de uso
if __name__ == "__main__":
    # Inicializa el analizador
    analizador = HybridDocumentAnalyzer(
        modelo_spacy="es_core_news_lg",
        patron_tabla=["Fila de tabla", "Tabla"]  # Puedes añadir más patrones
    )

    # Analiza un documento
    ruta = Path("../../data/raw/notas_de_corte_grados_cupo_general.md")
    if ruta.exists():
        resultado = analizador.analiza(ruta, verbose=True, min_frecuencia=1)  # min_frecuencia=1 para capturar relaciones

        print("\n" + "="*50)
        print("RESULTADOS FINALES")
        print("="*50)
        print(f"\nallowed_nodes = {resultado['allowed_nodes']}")
        print(f"\nallowed_relationships = {resultado['allowed_relationships']}")

        print("\n" + "="*50)
        print("ESTADÍSTICAS")
        print("="*50)
        for k, v in resultado['estadisticas'].items():
            print(f"  {k}: {v}")