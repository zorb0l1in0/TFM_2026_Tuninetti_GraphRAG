#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chunker para Markdown que divide por secciones (#, ##, ###)
y respeta un límite máximo de caracteres.
"""

import re
from typing import List, Dict, Any, Optional
from pathlib import Path


class ChunkerSeccionesMarkdown:
    """
    Divide documentos Markdown en chunks basados en las secciones (headings).
    Cada chunk contiene una o más secciones completas, nunca partidas a la mitad.
    Límite en caracteres (aproximadamente 4 caracteres = 1 token).
    """

    def __init__(self,
                 max_caracteres: int = 1500,
                 min_caracteres: int = 200,
                 niveles_heading: List[int] = [1, 2, 3]):
        """
        Args:
            max_caracteres: Caracteres máximos por chunk (default: 1500)
            min_caracteres: Caracteres mínimos recomendados (default: 200)
            niveles_heading: Niveles de heading a considerar como separadores
        """
        self.max_caracteres = max_caracteres
        self.min_caracteres = min_caracteres
        self.niveles_heading = niveles_heading

        # Patrón para encontrar headings (ej. #, ##, ###)
        min_level = min(niveles_heading)
        max_level = max(niveles_heading)
        patron = rf'^(#{{{min_level},{max_level}}})\s+(.+)$'
        self.patron_heading = re.compile(patron, re.MULTILINE)

    # ------------------------------------------------------------------
    # Pulizia frontmatter LlamaParser
    # ------------------------------------------------------------------

    def _limpiar_frontmatter(self, texto: str) -> str:
        """
        Rimuove i blocchi ---...--- inseriti da LlamaParser tra le pagine.
        Esempio:
            ---
            page: 3
            processor: Llama-Parse
            source: file.pdf
            total_pages: 11
            ---
        """
        return re.sub(r'---\n.*?---\n', '', texto, flags=re.DOTALL).strip()

    # ------------------------------------------------------------------
    # Entry point principale
    # ------------------------------------------------------------------

    def chunk_documento(self, texto: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Divide un documento in chunks per sezioni.

        Args:
            texto: Contenido del documento
            metadata: Metadatos base a incluir en cada chunk

        Returns:
            Lista de chunks con texto, caracteres, y metadatos
        """
        if metadata is None:
            metadata = {}

        # 0. Pulire il frontmatter di LlamaParser prima di processare
        texto = self._limpiar_frontmatter(texto)

        # 1. Extraer todas las secciones
        secciones = self._extraer_secciones(texto)

        if not secciones:
            # No hay headings → tratar todo como una única sección
            return [self._crear_chunk(texto, metadata, "Documento completo", 0)]

        # 2. Agrupar secciones en chunks respetando el límite
        chunks = []
        chunk_actual = []
        tamaño_actual = 0
        titulos_secciones = []

        for i, seccion in enumerate(secciones):
            tamaño_seccion = len(seccion['texto'])

            # Si la sección sola es más grande que el máximo, la dividimos forzosamente
            if tamaño_seccion > self.max_caracteres:
                # Primero cerrar el chunk actual si existe
                if chunk_actual:
                    chunks.append(self._fusionar_secciones(
                        chunk_actual,
                        titulos_secciones,
                        metadata,
                        len(chunks)
                    ))
                    chunk_actual = []
                    titulos_secciones = []
                    tamaño_actual = 0

                # Dividir la sección gigante en subpartes
                subpartes = self._dividir_seccion_grande(seccion)
                for j, subparte in enumerate(subpartes):
                    chunks.append(self._crear_chunk(
                        texto=subparte['texto'],
                        metadata=metadata,
                        titulo_seccion=seccion['titulo'],
                        subtitulo=f"Parte {j+1}" if len(subpartes) > 1 else None,
                        indice=len(chunks)
                    ))
                continue

            # Si al añadir esta sección se supera el límite
            if tamaño_actual + tamaño_seccion > self.max_caracteres and chunk_actual:
                # Guardar el chunk actual
                chunks.append(self._fusionar_secciones(
                    chunk_actual,
                    titulos_secciones,
                    metadata,
                    len(chunks)
                ))
                # Empezar nuevo chunk con esta sección
                chunk_actual = [seccion]
                titulos_secciones = [seccion['titulo']]
                tamaño_actual = tamaño_seccion
            else:
                # Añadir al chunk actual
                chunk_actual.append(seccion)
                titulos_secciones.append(seccion['titulo'])
                tamaño_actual += tamaño_seccion

        # Último chunk
        if chunk_actual:
            chunks.append(self._fusionar_secciones(
                chunk_actual,
                titulos_secciones,
                metadata,
                len(chunks)
            ))

        return chunks

    # ------------------------------------------------------------------
    # Estrazione sezioni
    # ------------------------------------------------------------------

    def _extraer_secciones(self, texto: str) -> List[Dict[str, str]]:
        """
        Extrae las secciones del documento basadas en los headings.
        Devuelve lista de dict con 'titulo' y 'texto'.
        """
        matches = list(self.patron_heading.finditer(texto))

        if not matches:
            return []

        secciones = []
        for i, match in enumerate(matches):
            titulo = match.group(2).strip()

            # Determinar el texto de la sección (desde este match hasta el siguiente)
            inicio = match.end()
            fin = matches[i + 1].start() if i + 1 < len(matches) else len(texto)
            texto_seccion = texto[inicio:fin].strip()

            # Incluir el heading en el texto de la sección
            texto_completo = f"{match.group(0)}\n{texto_seccion}"

            secciones.append({
                'titulo': titulo,
                'texto': texto_completo,
                'nivel': len(match.group(1))  # número de #
            })

        return secciones

    # ------------------------------------------------------------------
    # Divisione sezioni grandi
    # ------------------------------------------------------------------

    def _dividir_seccion_grande(self, seccion: Dict) -> List[Dict]:
        """
        Divide una sección demasiado larga en subpartes.
        Intenta dividir por párrafos o subsecciones.
        """
        texto = seccion['texto']
        titulo = seccion['titulo']

        # Buscar subsecciones (headings de nivel superior)
        subsecciones = self._extraer_secciones(texto)

        if subsecciones:
            # Ya tiene subsecciones internas → usarlas
            return [{
                'titulo': f"{titulo} - {s['titulo']}",
                'texto': s['texto']
            } for s in subsecciones]

        # Dividir por párrafos
        parrafos = [p.strip() for p in texto.split('\n\n') if p.strip()]

        if len(parrafos) > 1:
            chunks = []
            chunk_actual = []
            tamaño_actual = 0

            for p in parrafos:
                if tamaño_actual + len(p) > self.max_caracteres and chunk_actual:
                    chunks.append({
                        'titulo': titulo,
                        'texto': '\n\n'.join(chunk_actual)
                    })
                    chunk_actual = [p]
                    tamaño_actual = len(p)
                else:
                    chunk_actual.append(p)
                    tamaño_actual += len(p) + 2

            if chunk_actual:
                chunks.append({
                    'titulo': titulo,
                    'texto': '\n\n'.join(chunk_actual)
                })

            return chunks

        # Último recurso: dividir por frases
        return self._dividir_por_oraciones(seccion)

    def _dividir_por_oraciones(self, seccion: Dict) -> List[Dict]:
        """Divide por oraciones cuando no hay otras estructuras"""
        texto = seccion['texto']
        titulo = seccion['titulo']

        # Patrón simple para oraciones (. ! ?)
        oraciones = re.split(r'(?<=[.!?])\s+', texto)
        oraciones = [o.strip() for o in oraciones if o.strip()]

        chunks = []
        chunk_actual = []
        tamaño_actual = 0

        for oracion in oraciones:
            if tamaño_actual + len(oracion) > self.max_caracteres and chunk_actual:
                chunks.append({
                    'titulo': titulo,
                    'texto': ' '.join(chunk_actual)
                })
                chunk_actual = [oracion]
                tamaño_actual = len(oracion)
            else:
                chunk_actual.append(oracion)
                tamaño_actual += len(oracion) + 1

        if chunk_actual:
            chunks.append({
                'titulo': titulo,
                'texto': ' '.join(chunk_actual)
            })

        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fusionar_secciones(self,
                            secciones: List[Dict],
                            titulos: List[str],
                            metadata: Dict,
                            indice: int) -> Dict[str, Any]:
        """Une múltiples secciones en un único chunk"""
        texto_completo = '\n\n'.join([s['texto'] for s in secciones])

        if len(titulos) == 1:
            titulo_chunk = titulos[0]
        else:
            titulo_chunk = f"{titulos[0]} y otras {len(titulos)-1} secciones"

        return self._crear_chunk(
            texto=texto_completo,
            metadata=metadata,
            titulo_seccion=titulo_chunk,
            indice=indice,
            secciones_incluidas=titulos
        )

    def _crear_chunk(self,
                     texto: str,
                     metadata: Dict,
                     titulo_seccion: str,
                     indice: int,
                     subtitulo: Optional[str] = None,
                     secciones_incluidas: Optional[List[str]] = None) -> Dict[str, Any]:
        """Crea un chunk con todos los metadatos"""
        chunk = {
            "texto": texto,
            "titulo_seccion": titulo_seccion,
            "indice_chunk": indice,
            "caracteres": len(texto),
            "palabras": len(texto.split())
        }

        if subtitulo:
            chunk["subtitulo"] = subtitulo

        if secciones_incluidas:
            chunk["secciones_incluidas"] = secciones_incluidas

        # Añadir metadatos del documento
        chunk.update(metadata)

        return chunk