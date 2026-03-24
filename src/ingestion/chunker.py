"""
chunker.py
----------
Divide documenti Markdown in chunks basati sulle sezioni (headings #/##/###)
rispettando un limite massimo di caratteri.
"""

import re
from typing import Any, Dict, List, Optional


class ChunkerSeccionesMarkdown:

    def __init__(
        self,
        max_caracteres: int = 1500,
        min_caracteres: int = 200,
        niveles_heading: List[int] = [1, 2, 3],
    ):
        self.max_caracteres  = max_caracteres
        self.min_caracteres  = min_caracteres
        self.niveles_heading = niveles_heading
        min_lvl = min(niveles_heading)
        max_lvl = max(niveles_heading)
        self.patron_heading = re.compile(
            rf'^(#{{{min_lvl},{max_lvl}}})\s+(.+)$', re.MULTILINE
        )

    def _limpiar_frontmatter(self, texto: str) -> str:
        return re.sub(r'---\n(?:.*?\n)*?---\n', '', texto, flags=re.DOTALL).strip()

    def chunk_documento(self, texto: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        if metadata is None:
            metadata = {}
        texto     = self._limpiar_frontmatter(texto)
        secciones = self._extraer_secciones(texto)
        if not secciones:
            return [self._crear_chunk(texto, metadata, "Documento completo", 0)]

        chunks, chunk_actual, titulos_secciones, tamaño_actual = [], [], [], 0

        for seccion in secciones:
            tamaño_seccion = len(seccion["texto"])
            if tamaño_seccion > self.max_caracteres:
                if chunk_actual:
                    chunks.append(self._fusionar_secciones(chunk_actual, titulos_secciones, metadata, len(chunks)))
                    chunk_actual, titulos_secciones, tamaño_actual = [], [], 0
                for j, sub in enumerate(self._dividir_texto(seccion["texto"], seccion["titulo"])):
                    chunks.append(self._crear_chunk(
                        texto=sub["texto"], metadata=metadata,
                        titulo_seccion=seccion["titulo"], indice=len(chunks),
                        subtitulo=f"Parte {j+1}",
                    ))
                continue

            if tamaño_actual + tamaño_seccion > self.max_caracteres and chunk_actual:
                chunks.append(self._fusionar_secciones(chunk_actual, titulos_secciones, metadata, len(chunks)))
                chunk_actual, titulos_secciones, tamaño_actual = [seccion], [seccion["titulo"]], tamaño_seccion
            else:
                chunk_actual.append(seccion)
                titulos_secciones.append(seccion["titulo"])
                tamaño_actual += tamaño_seccion

        if chunk_actual:
            chunks.append(self._fusionar_secciones(chunk_actual, titulos_secciones, metadata, len(chunks)))
        return chunks

    def _extraer_secciones(self, texto: str) -> List[Dict]:
        matches = list(self.patron_heading.finditer(texto))
        if not matches:
            return []
        secciones = []
        for i, match in enumerate(matches):
            inicio = match.end()
            fin    = matches[i + 1].start() if i + 1 < len(matches) else len(texto)
            texto_seccion = texto[inicio:fin].strip()
            secciones.append({
                "titulo": match.group(2).strip(),
                "texto":  f"{match.group(0)}\n{texto_seccion}",
                "nivel":  len(match.group(1)),
            })
        return secciones

    def _dividir_texto(self, texto: str, titulo: str) -> List[Dict]:
        """
        Divide un blocco di testo in chunks <= max_caracteres.
        Strategia: paragrafi (\n\n) → orazioni → forza taglio a max_caracteres.
        Non usa _extraer_secciones per evitare ricorsione.
        """
        # 1. Paragrafi con \n\n
        parrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]
        if len(parrafos) > 1:
            result = self._agrupar(parrafos, titulo)
            if len(result) > 1:
                return result

        # 2. Orazioni
        oraciones = [o.strip() for o in re.split(r'(?<=[.!?])\s+', texto) if o.strip()]
        if len(oraciones) > 1:
            result = self._agrupar(oraciones, titulo)
            if len(result) > 1:
                return result

        # 3. Taglio forzato a max_caracteres
        partes = []
        for i in range(0, len(texto), self.max_caracteres):
            partes.append({"titulo": titulo, "texto": texto[i:i + self.max_caracteres]})
        return partes if partes else [{"titulo": titulo, "texto": texto}]

    def _agrupar(self, unidades: List[str], titulo: str) -> List[Dict]:
        chunks: List[Dict] = []
        chunk_actual: List[str] = []
        tamaño_actual = 0
        for u in unidades:
            nuevo = tamaño_actual + len(u) + (2 if chunk_actual else 0)
            if nuevo > self.max_caracteres and chunk_actual:
                chunks.append({"titulo": titulo, "texto": "\n\n".join(chunk_actual)})
                chunk_actual, tamaño_actual = [u], len(u)
            else:
                chunk_actual.append(u)
                tamaño_actual = nuevo
        if chunk_actual:
            chunks.append({"titulo": titulo, "texto": "\n\n".join(chunk_actual)})
        return chunks

    def _fusionar_secciones(self, secciones, titulos, metadata, indice):
        texto_completo = "\n\n".join(s["texto"] for s in secciones)
        titulo_chunk   = titulos[0] if len(titulos) == 1 else f"{titulos[0]} y otras {len(titulos)-1} secciones"
        return self._crear_chunk(texto_completo, metadata, titulo_chunk, indice, secciones_incluidas=titulos)

    def _crear_chunk(self, texto, metadata, titulo_seccion, indice, subtitulo=None, secciones_incluidas=None):
        chunk = {
            "texto":          texto,
            "titulo_seccion": titulo_seccion,
            "indice_chunk":   indice,
            "caracteres":     len(texto),
            "palabras":       len(texto.split()),
        }
        if subtitulo:
            chunk["subtitulo"] = subtitulo
        if secciones_incluidas:
            chunk["secciones_incluidas"] = secciones_incluidas
        chunk.update(metadata)
        return chunk