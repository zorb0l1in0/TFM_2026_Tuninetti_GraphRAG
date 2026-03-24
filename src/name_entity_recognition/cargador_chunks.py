import sys
import pandas as pd
from pathlib import Path
from typing import Optional


def _print(msg: str):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


class CargadorChunksCSV:
    """
    Carga los chunks del CSV y los prepara como entrada para la pipeline NER.

    Columnas esperadas (con múltiples alias por columna):
      texto  → raw_text | texto | contenido | content
      título → nombre_documento | titulo | title
      tipo   → clase_documento | tipo_documento | type
      fuente → source_file | archivo_origen | file_name
    """

    def __init__(self, ruta_csv: str):
        self.ruta = Path(ruta_csv)
        if not self.ruta.exists():
            raise FileNotFoundError(f"CSV no encontrado: {self.ruta}")

        _print(f"📂 Cargando CSV: {self.ruta.name}")
        self.df = pd.read_csv(self.ruta)
        _print(f"✅ Filas totales: {len(self.df)}")
        _print(f"📋 Columnas: {', '.join(self.df.columns.tolist())}")

    def cargar_chunks(
        self,
        max_chunks: Optional[int] = None,
        filtro_documento: Optional[str] = None,
        filtro_tipo: Optional[str] = None,
    ) -> list[dict]:
        """
        Devuelve lista de dicts con campos:
          texto, titulo, tipo, fuente, chunk_id

        Args:
            max_chunks:        limita el número de chunks a procesar
            filtro_documento:  filtra por nombre de documento (nombre_documento)
            filtro_tipo:       filtra por tipo de documento (clase_documento)
        """
        df = self.df.copy()

        if filtro_documento:
            col = next((c for c in ['nombre_documento', 'titulo', 'title'] if c in df.columns), None)
            if col:
                df = df[df[col].astype(str).str.contains(filtro_documento, case=False, na=False)]
                _print(f"🔍 Filtro documento '{filtro_documento}': {len(df)} filas")

        if filtro_tipo:
            col = next((c for c in ['clase_documento', 'tipo_documento', 'type'] if c in df.columns), None)
            if col:
                df = df[df[col].astype(str).str.contains(filtro_tipo, case=False, na=False)]
                _print(f"🔍 Filtro tipo '{filtro_tipo}': {len(df)} filas")

        if max_chunks:
            df = df.head(max_chunks)
            _print(f"✂️  Limitado a {max_chunks} chunks")

        chunks = []
        for idx, fila in df.iterrows():
            texto = self._obtener(fila, ['raw_text', 'texto', 'contenido', 'content'])
            if not texto:
                continue
            chunks.append({
                "chunk_id": self._obtener(fila, ['id', 'chunk_id', 'chunk_index']) or f"chunk_{idx:04d}",
                "texto":    texto,
                "titulo":   self._obtener(fila, ['nombre_documento', 'titulo', 'title']) or "Sin título",
                "tipo":     self._obtener(fila, ['clase_documento', 'tipo_documento', 'type']) or "Desconocido",
                "fuente":   self._obtener(fila, ['source_file', 'archivo_origen', 'file_name']) or "desconocido",
            })

        _print(f"✅ Chunks listos para NER: {len(chunks)}")
        return chunks

    def _obtener(self, fila, columnas: list[str]) -> str:
        """Busca el valor en la primera columna disponible."""
        for col in columnas:
            if col in self.df.columns and pd.notna(fila.get(col)):
                return str(fila[col]).strip()
        return ""

    def resumen(self):
        """Muestra estadísticas básicas del CSV cargado."""
        _print("\n" + "=" * 40)
        _print("📊 RESUMEN CSV")
        _print("=" * 40)
        _print(f"Chunks totales: {len(self.df)}")

        for col_tipo in ['clase_documento', 'tipo_documento']:
            if col_tipo in self.df.columns:
                _print("\n📑 Tipos de documento:")
                for tipo, count in self.df[col_tipo].value_counts().items():
                    _print(f"  • {tipo}: {count}")
                break

        for col_doc in ['nombre_documento', 'titulo']:
            if col_doc in self.df.columns:
                _print(f"\n📄 Documentos únicos: {self.df[col_doc].nunique()}")
                break

        _print("=" * 40)


if __name__ == "__main__":
    import sys

    print("=" * 50)
    print("🧪 PRUEBA CARGADOR CHUNKS CSV")
    print("=" * 50)

    # Buscar CSV disponibles
    carpeta = Path(__file__).parent
    csvs = list(carpeta.glob("*.csv"))

    if not csvs:
        print("❌ No hay archivos CSV en la carpeta actual")
        sys.exit(1)

    print("\n📁 Archivos disponibles:")
    for i, csv in enumerate(csvs):
        print(f"  {i + 1}. {csv.name} ({csv.stat().st_size / 1024:.0f} KB)")

    seleccion = input(f"\n📌 Selecciona (1-{len(csvs)}): ").strip()
    try:
        idx = int(seleccion) - 1 if seleccion else 0
        archivo = str(csvs[idx])
    except (ValueError, IndexError):
        archivo = str(csvs[0])

    print(f"\n📂 Usando: {Path(archivo).name}")

    cargador = CargadorChunksCSV(archivo)
    cargador.resumen()

    chunks = cargador.cargar_chunks(max_chunks=5)

    print("\n📋 EJEMPLOS:")
    for chunk in chunks:
        print(f"\n--- {chunk['chunk_id']} ---")
        print(f"  📌 Título : {chunk['titulo']}")
        print(f"  📌 Tipo   : {chunk['tipo']}")
        print(f"  📌 Fuente : {chunk['fuente']}")
        print(f"  📝 Texto  : {chunk['texto'][:150]}...")

    print("\n✅ Prueba completada")