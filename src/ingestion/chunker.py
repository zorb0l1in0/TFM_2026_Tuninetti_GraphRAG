"""
chunker.py
----------
Divide documenti Markdown in chunks basati sulle sezioni (headings #/##/###)
rispettando un limite massimo di caratteri.
"""

import re
from typing import Any, Dict, List, Optional


class ChunkerSeccionesMarkdown:
    """
    Divide documenti Markdown in chunks per sezioni.
    Ogni chunk contiene una o più sezioni complete, mai spezzate a metà.
    ~4 caratteri = 1 token.
    """

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

    # ── Pulizia ───────────────────────────────────────────────────────────────

    def _limpiar_frontmatter(self, texto: str) -> str:
        """Rimuove i blocchi ---...--- inseriti da LlamaParser tra le pagine."""
        return re.sub(r'---\n.*?---\n', '', texto, flags=re.DOTALL).strip()

    # ── Entry point ───────────────────────────────────────────────────────────

    def chunk_documento(
        self,
        texto: str,
        metadata: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Divide un documento in chunks per sezioni.

        Args:
            texto:    Contenuto Markdown del documento
            metadata: Metadati base da includere in ogni chunk

        Returns:
            Lista di dict con: texto, titulo_seccion, indice_chunk,
            caracteres, palabras, + campi di metadata
        """
        if metadata is None:
            metadata = {}

        texto     = self._limpiar_frontmatter(texto)
        secciones = self._extraer_secciones(texto)

        if not secciones:
            return [self._crear_chunk(texto, metadata, "Documento completo", 0)]

        chunks            = []
        chunk_actual      = []
        titulos_secciones = []
        tamaño_actual     = 0

        for seccion in secciones:
            tamaño_seccion = len(seccion["texto"])

            if tamaño_seccion > self.max_caracteres:
                if chunk_actual:
                    chunks.append(self._fusionar_secciones(
                        chunk_actual, titulos_secciones, metadata, len(chunks)
                    ))
                    chunk_actual, titulos_secciones, tamaño_actual = [], [], 0

                subpartes = self._dividir_seccion_grande(seccion)
                for j, sub in enumerate(subpartes):
                    chunks.append(self._crear_chunk(
                        texto=sub["texto"],
                        metadata=metadata,
                        titulo_seccion=seccion["titulo"],
                        indice=len(chunks),
                        subtitulo=f"Parte {j+1}" if len(subpartes) > 1 else None,
                    ))
                continue

            if tamaño_actual + tamaño_seccion > self.max_caracteres and chunk_actual:
                chunks.append(self._fusionar_secciones(
                    chunk_actual, titulos_secciones, metadata, len(chunks)
                ))
                chunk_actual      = [seccion]
                titulos_secciones = [seccion["titulo"]]
                tamaño_actual     = tamaño_seccion
            else:
                chunk_actual.append(seccion)
                titulos_secciones.append(seccion["titulo"])
                tamaño_actual += tamaño_seccion

        if chunk_actual:
            chunks.append(self._fusionar_secciones(
                chunk_actual, titulos_secciones, metadata, len(chunks)
            ))

        return chunks

    # ── Estrazione sezioni ────────────────────────────────────────────────────

    def _extraer_secciones(self, texto: str) -> List[Dict]:
        matches = list(self.patron_heading.finditer(texto))
        if not matches:
            return []

        secciones = []
        for i, match in enumerate(matches):
            inicio        = match.end()
            fin           = matches[i + 1].start() if i + 1 < len(matches) else len(texto)
            texto_seccion = texto[inicio:fin].strip()
            secciones.append({
                "titulo": match.group(2).strip(),
                "texto":  f"{match.group(0)}\n{texto_seccion}",
                "nivel":  len(match.group(1)),
            })
        return secciones

    # ── Divisione sezioni grandi ──────────────────────────────────────────────

    def _dividir_seccion_grande(self, seccion: Dict) -> List[Dict]:
        texto  = seccion["texto"]
        titulo = seccion["titulo"]

        # 1. Subseciones internas
        subsecciones = self._extraer_secciones(texto)
        if subsecciones:
            return [{"titulo": f"{titulo} - {s['titulo']}", "texto": s["texto"]}
                    for s in subsecciones]

        # 2. Párrafos
        parrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]
        if len(parrafos) > 1:
            chunks, chunk_actual, tamaño_actual = [], [], 0
            for p in parrafos:
                if tamaño_actual + len(p) > self.max_caracteres and chunk_actual:
                    chunks.append({"titulo": titulo, "texto": "\n\n".join(chunk_actual)})
                    chunk_actual, tamaño_actual = [p], len(p)
                else:
                    chunk_actual.append(p)
                    tamaño_actual += len(p) + 2
            if chunk_actual:
                chunks.append({"titulo": titulo, "texto": "\n\n".join(chunk_actual)})
            return chunks

        # 3. Oraciones (último recurso)
        return self._dividir_por_oraciones(seccion)

    def _dividir_por_oraciones(self, seccion: Dict) -> List[Dict]:
        texto     = seccion["texto"]
        titulo    = seccion["titulo"]
        oraciones = [o.strip() for o in re.split(r'(?<=[.!?])\s+', texto) if o.strip()]

        chunks, chunk_actual, tamaño_actual = [], [], 0
        for oracion in oraciones:
            if tamaño_actual + len(oracion) > self.max_caracteres and chunk_actual:
                chunks.append({"titulo": titulo, "texto": " ".join(chunk_actual)})
                chunk_actual, tamaño_actual = [oracion], len(oracion)
            else:
                chunk_actual.append(oracion)
                tamaño_actual += len(oracion) + 1
        if chunk_actual:
            chunks.append({"titulo": titulo, "texto": " ".join(chunk_actual)})
        return chunks

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fusionar_secciones(
        self,
        secciones: List[Dict],
        titulos: List[str],
        metadata: Dict,
        indice: int,
    ) -> Dict[str, Any]:
        texto_completo = "\n\n".join(s["texto"] for s in secciones)
        titulo_chunk   = (
            titulos[0] if len(titulos) == 1
            else f"{titulos[0]} y otras {len(titulos)-1} secciones"
        )
        return self._crear_chunk(
            texto=texto_completo,
            metadata=metadata,
            titulo_seccion=titulo_chunk,
            indice=indice,
            secciones_incluidas=titulos,
        )

    def _crear_chunk(
        self,
        texto: str,
        metadata: Dict,
        titulo_seccion: str,
        indice: int,
        subtitulo: Optional[str] = None,
        secciones_incluidas: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
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