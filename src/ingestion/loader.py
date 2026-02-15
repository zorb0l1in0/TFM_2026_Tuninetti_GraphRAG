import pandas as pd
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from config import settings


class CSVLoader:
    """
    Cargador simple que extrae texto y metadatos fundamentales del CSV.
    """

    def __init__(self, nombre_archivo: str):
        """
        Args:
            nombre_archivo: Nombre del archivo CSV en data/processed/chunks/
        """
        self.ruta_csv = settings.CHUNKS_DIR / nombre_archivo
        if not self.ruta_csv.exists():
            raise FileNotFoundError(f"❌ CSV no encontrado: {self.ruta_csv}")

        print(f"📂 Cargando CSV: {self.ruta_csv.name}")
        self.df = pd.read_csv(self.ruta_csv)
        print(f"✅ Total filas: {len(self.df)}")
        print(f"📋 Columnas: {', '.join(self.df.columns[:5])}...")

    def cargar_documentos(self) -> List[Document]:
        """
        Convierte el CSV en documentos de LangChain.

        Metadatos:
        - titulo: nombre del documento
        - tipo: clase de documento (guía, boletín, etc.)
        - pagina: número de página
        - id_chunk: identificador único del fragmento
        - archivo: ruta del archivo original
        """
        documentos = []
        errores = 0

        for idx, fila in self.df.iterrows():
            try:
                # 1. Texto del chunk
                texto = self._extraer_texto(fila)
                if not texto:  # Saltar filas vacías
                    continue

                # 2. Metadatos esenciales
                metadata = {
                    "titulo": self._extraer_titulo(fila),
                    "tipo": self._extraer_tipo(fila),
                    "pagina": self._extraer_pagina(fila),
                    "id_chunk": self._extraer_id_chunk(fila, idx),
                    "archivo": self._extraer_archivo(fila),
                }

                # 3. Crear documento
                documentos.append(Document(
                    page_content=texto,
                    metadata=metadata
                ))

            except Exception as e:
                errores += 1
                if errores <= 3:  # Mostrar solo primeros errores
                    print(f"⚠️ Error fila {idx}: {e}")

        print(f"✅ Documentos cargados: {len(documentos)}")
        if errores:
            print(f"⚠️ Filas con error: {errores}")

        return documentos

    def _extraer_texto(self, fila) -> str:
        """Busca el texto en columnas posibles"""
        for col in ['raw_text', 'texto', 'contenido', 'content']:
            if col in self.df.columns and pd.notna(fila[col]):
                return str(fila[col]).strip()
        return ""

    def _extraer_titulo(self, fila) -> str:
        """Busca el título/nombre del documento"""
        for col in ['nombre_documento', 'titulo', 'title']:
            if col in self.df.columns and pd.notna(fila[col]):
                return str(fila[col]).strip()
        return "Sin título"

    def _extraer_tipo(self, fila) -> str:
        """Busca el tipo/clase de documento"""
        for col in ['clase_documento', 'tipo_documento', 'type']:
            if col in self.df.columns and pd.notna(fila[col]):
                return str(fila[col]).strip()
        return "Desconocido"

    def _extraer_pagina(self, fila) -> int:
        """Extrae el número de página"""
        for col in ['page', 'pagina', 'page_number']:
            if col in self.df.columns and pd.notna(fila[col]):
                try:
                    return int(float(fila[col]))
                except:
                    return 0
        return 0

    def _extraer_id_chunk(self, fila, idx_default) -> str:
        """Extrae el ID del chunk"""
        for col in ['id', 'chunk_id', 'chunk_index']:
            if col in self.df.columns and pd.notna(fila[col]):
                return str(fila[col])
        return f"chunk_{idx_default:04d}"

    def _extraer_archivo(self, fila) -> str:
        """Extrae el nombre del archivo original"""
        for col in ['source_file', 'archivo_origen', 'file_name']:
            if col in self.df.columns and pd.notna(fila[col]):
                return str(fila[col])
        return "desconocido"

    def resumen(self):
        """Muestra estadísticas básicas"""
        print("\n" + "=" * 40)
        print("📊 RESUMEN")
        print("=" * 40)
        print(f"Total chunks: {len(self.df)}")

        if 'clase_documento' in self.df:
            print("\n📑 Tipos:")
            for tipo, count in self.df['clase_documento'].value_counts().items():
                print(f"  • {tipo}: {count}")

        if 'nombre_documento' in self.df:
            print(f"\n📄 Documentos: {self.df['nombre_documento'].nunique()}")

        if 'page' in self.df:
            print(f"📌 Páginas: {self.df['page'].min()} - {self.df['page'].max()}")

        print("=" * 40)


if __name__ == "__main__":
    """Prueba rápida del cargador"""
    import sys
    from pathlib import Path

    print("=" * 50)
    print("🧪 PRUEBA CARGADOR")
    print("=" * 50)

    # Buscar CSV en la carpeta chunks
    chunks_dir = Path(__file__).parent.parent.parent / "data" / "chunks"
    csvs = list(chunks_dir.glob("*.csv"))

    if not csvs:
        print("❌ No hay archivos CSV")
        sys.exit(1)

    # Mostrar archivos disponibles
    print("\n📁 Archivos:")
    for i, csv in enumerate(csvs):
        print(f"  {i + 1}. {csv.name} ({csv.stat().st_size / 1024:.0f} KB)")

    # Seleccionar archivo
    seleccion = input(f"\n📌 Selecciona (1-{len(csvs)}): ").strip()
    try:
        idx = int(seleccion) - 1 if seleccion else 0
        archivo = csvs[idx].name
    except:
        archivo = csvs[0].name

    print(f"\n📂 Usando: {archivo}")

    # Probar cargador
    cargador = CSVLoader(archivo)
    cargador.resumen()

    documentos = cargador.cargar_documentos()

    # Mostrar ejemplos
    if documentos:
        print("\n📋 EJEMPLOS:")
        for i, doc in enumerate(documentos[:3]):
            print(f"\n--- Documento {i + 1} ---")
            print(f"📌 Título: {doc.metadata['titulo']}")
            print(f"📌 Tipo: {doc.metadata['tipo']}")
            print(f"📌 Página: {doc.metadata['pagina']}")
            print(f"📝 Texto: {doc.page_content[:150]}...")

    print(f"\n✅ Prueba completada")