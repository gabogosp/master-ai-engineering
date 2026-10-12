"""Genera los PDFs sintéticos calibrados para el runner de stress
(Bloque 3 / Paso 3).

Produce `attach_{5,20,50,100}kb.pdf` en esta misma carpeta, donde el
sufijo refiere al TAMAÑO DEL TEXTO EXTRAÍDO (lo que `pypdf` va a sacar de
ahí) y no al tamaño del archivo en disco -- al runner le importa cuánto le
suma al prompt, no cuántos bytes pesa el PDF. El attachment de "0 KB" de
la consigna no es un archivo: es directamente no adjuntar nada, así que
acá no se genera.

El generador es determinístico: mismo párrafo, mismo patrón de repetición
por tamaño. Correrlo de nuevo produce PDFs idénticos byte a byte -- por
eso no se comitean, solo este script (ver .gitignore).

Verificación: después de escribir cada PDF, se vuelve a leer con
`pypdf.PdfReader` (el mismo lector que usa `app.attachments.extract_text`
en producción) para confirmar que el texto extraído por el pipeline real
cae cerca del tamaño pedido. fpdf2 no garantiza una relación 1:1 entre
"caracteres escritos" y "caracteres que pypdf extrae de vuelta" (depende
de cómo cada librería maneja saltos de línea/espacios), así que no alcanza
con confiar en el tamaño del párrafo repetido -- hay que medir contra el
mismo extractor que usa el sistema.

Correr con::

    uv run python -m evals.stress.fixtures.build_pdfs
"""

import sys
from pathlib import Path

from fpdf import FPDF
from pypdf import PdfReader

_TARGETS_KB = (5, 20, 50, 100)

_PARRAFO = (
    "El portal de proveedores debe reemplazar el flujo manual existente. "
    "Los proveedores suben PDFs con sus certificados de cumplimiento, que "
    "el equipo de compras revisa por completitud, vencimiento y "
    "conformidad con el estándar de la empresa. Los proveedores aprobados "
    "quedan habilitados para generar órdenes de compra; los rechazados "
    "deben corregir antes del próximo ciclo de revisión. Cada cambio de "
    "estado queda registrado en un log de auditoría inmutable, retenido "
    "por siete años según la política interna. Los revisores pueden "
    "agregar comentarios a un certificado puntual; esos comentarios son "
    "visibles para el proveedor y forman parte del registro formal. El "
    "portal debe integrarse con el ERP existente para sincronizar los "
    "datos maestros de proveedores, propagando cambios en menos de "
    "quince minutos. El inicio de sesión único es obligatorio para "
    "usuarios internos; los proveedores externos usan un enlace mágico "
    "por email, acotado a un solo tenant. Se espera un P95 de carga de "
    "página bajo dos segundos con doscientos usuarios concurrentes, y "
    "una disponibilidad con menos de una hora de caída por trimestre, "
    "sin contar ventanas de mantenimiento planificadas fuera de horario. "
)


def _build_pdf(target_kb: int, output_path: Path) -> tuple[int, int]:
    """Arma un PDF cuyo TEXTO EXTRAÍDO ronda `target_kb` KB.

    Devuelve (caracteres_escritos, bytes_del_archivo). La relación entre
    texto escrito y bytes del archivo depende de la compresión interna de
    fpdf2 -- no es 1:1, por eso el nombre del archivo describe el texto,
    no el peso en disco.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    target_chars = target_kb * 1024
    written_chars = 0
    while written_chars < target_chars:
        pdf.multi_cell(0, 5, _PARRAFO)
        written_chars += len(_PARRAFO)
        if written_chars < target_chars:
            pdf.ln(3)  # línea en blanco -- separador visual entre párrafos

    pdf.output(str(output_path))
    file_bytes = output_path.stat().st_size
    return written_chars, file_bytes


def _verify_extracted(path: Path) -> int:
    """Relee el PDF con pypdf (el mismo extractor que usa
    app.attachments.extract_text en producción) y devuelve cuántos
    caracteres saca de verdad -- la cifra que realmente le va a llegar al
    LLM, no la que escribimos nosotros."""
    reader = PdfReader(str(path))
    parts = [page.extract_text() or "" for page in reader.pages]
    return len("\n".join(parts).strip())


def main() -> int:
    here = Path(__file__).parent
    print(f"Escribiendo PDFs en: {here}")
    for kb in _TARGETS_KB:
        path = here / f"attach_{kb}kb.pdf"
        written_chars, file_bytes = _build_pdf(kb, path)
        extracted_chars = _verify_extracted(path)
        print(
            f"  {path.name}: escritos={written_chars} "
            f"bytes_archivo={file_bytes} pypdf_extraidos={extracted_chars}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
