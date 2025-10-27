# app.py — бэкенд для расчёта и импозиции книги
import io
import os
import math
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

from fastapi import FastAPI, UploadFile, Form, Request, File
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib.units import mm  # 1 mm -> points

# ---------------------------------------------------------------------
# FastAPI setup
# ---------------------------------------------------------------------
app = FastAPI(title="Идеальная книга — калькулятор и верстка", version="2.0")
BASE_DIR = os.path.dirname(__file__)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def has_cmd(cmd: str) -> bool:
    from shutil import which
    return which(cmd) is not None

def run(args: list) -> int:
    try:
        return subprocess.call(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return 1

# ---------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------
@dataclass
class Params:
    total_pages: int
    signature_size: int
    pages_per_sheet_side: int = 2
    duplex: bool = True
    page_format: str = "A5"
    sheet_format: str = "A4"
    sheet_width_mm: float = 210.0
    sheet_height_mm: float = 297.0
    # формат КНИЖНОЙ страницы (для справки/будущих смещений)
    page_width_mm: float = 148.0
    page_height_mm: float = 210.0
    creep_mm_per_leaf: float = 0.1
    gsm: int = 80

def _round2(x: float) -> float:
    return round(x + 1e-9, 2)

def page_order_for_signature(first: int, last: int) -> List[Dict]:
    """Классическая типографская раскладка для одной тетради (размер тетради кратен 4)."""
    s = last - first + 1
    assert s % 4 == 0
    sheets = []
    L, R = last, first
    for i in range(s // 4):
        front = [L, R]
        back  = [R + 1, L - 1]
        sheets.append({"sheet": i + 1, "front": front, "back": back})
        L -= 2
        R += 2
    return sheets

def build_plan(p: Params) -> Dict:
    sig = p.signature_size
    total = p.total_pages
    if sig % 4 != 0:
        raise ValueError("Размер тетради должен быть кратен 4")

    full = total // sig
    rem = total % sig
    padded_total = total if rem == 0 else total + (sig - rem)

    # список тетрадей (диапазоны страниц)
    signatures = []
    page = 1
    for _ in range(full):
        signatures.append({"start": page, "end": page + sig - 1})
        page += sig
    if rem:
        signatures.append({"start": page, "end": page + sig - 1})  # последняя с добивкой

    # листовой план каждой тетради
    sig_plans = []
    for sgn in signatures:
        sheets = page_order_for_signature(sgn["start"], sgn["end"])
        sig_plans.append({"range": sgn, "sheets": sheets})

    pages_per_sheet = p.pages_per_sheet_side * (2 if p.duplex else 1)
    sheets_needed = padded_total // pages_per_sheet

    # прикидки по корешку и макс. уводу
    caliper_map = {60:0.08,70:0.09,80:0.1,90:0.11,100:0.12,115:0.135,120:0.145,130:0.155,150:0.175}
    leaf_thickness = caliper_map.get(p.gsm, 0.1)  # мм на лист
    spine_mm = _round2((padded_total/2) * leaf_thickness)
    max_creep = _round2((sig/2) * p.creep_mm_per_leaf)

    return {
        "input": asdict(p),
        "padded_total_pages": padded_total,
        "signatures": sig_plans,
        "sheets_needed": sheets_needed,
        "pages_per_sheet": pages_per_sheet,
        "spine_mm_estimate": spine_mm,
        "max_creep_mm_estimate": max_creep,
        "blanks_added": padded_total - total
    }

# ---------------------------------------------------------------------
# PDF imposition (2-up на лист печати в горизонтальной ориентации)
# ---------------------------------------------------------------------
def impose_pdf_two_up(pdf_bytes: bytes, p: Params) -> bytes:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    total = len(reader.pages)

    plan = build_plan(p)

    sheet_w_pt = p.sheet_width_mm * mm   # 210 мм -> 595 pt
    sheet_h_pt = p.sheet_height_mm * mm  # 297 мм -> 842 pt

    # Горизонтальная ориентация (ландшафт): ширина и высота меняются местами
    dst_width = sheet_h_pt
    dst_height = sheet_w_pt
    half_w = dst_width / 2.0

    writer = PdfWriter()

    def get_src(n: int):
        if 1 <= n <= total:
            return reader.pages[n-1]
        return None  # пустая, если добивка

    for sig in plan["signatures"]:
        for sheet in sig["sheets"]:
            for side in ("front","back"):
                dst = writer.add_blank_page(width=dst_width, height=dst_height)

                left_num, right_num = sheet[side]
                left = get_src(left_num)
                right = get_src(right_num)

                if left is not None:
                    lw, lh = float(left.mediabox.width), float(left.mediabox.height)
                    s = min(half_w / lw, dst_height / lh)
                    t = Transformation().scale(s, s).translate(0, 0)
                    dst.merge_transformed_page(left, t)

                if right is not None:
                    rw, rh = float(right.mediabox.width), float(right.mediabox.height)
                    s = min(half_w / rw, dst_height / rh)
                    t = Transformation().scale(s, s).translate(half_w, 0)
                    dst.merge_transformed_page(right, t)

    out = io.BytesIO()
    writer.write(out)
    out.seek(0)
    return out.read()

# ---------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------
def detect_pdf_pages(pdf_bytes: bytes) -> int:
    return len(PdfReader(io.BytesIO(pdf_bytes)).pages)

def convert_any_to_pdf(bytes_in: bytes, ext: str) -> Optional[bytes]:
    """Конвертирует входной файл в PDF (LibreOffice/Calibre). Возвращает PDF-байты или None."""
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "input"+ext)
        with open(in_path, "wb") as f:
            f.write(bytes_in)
        out_pdf = os.path.join(td, "out.pdf")
        ok = False
        if has_cmd("soffice"):
            code = run(["soffice","--headless","--convert-to","pdf","--outdir",td,in_path])
            guess = os.path.join(td, os.path.splitext(os.path.basename(in_path))[0]+".pdf")
            if code == 0 and os.path.exists(guess):
                os.replace(guess, out_pdf)
                ok = True
        elif has_cmd("ebook-convert"):
            run(["ebook-convert", in_path, out_pdf])
            ok = os.path.exists(out_pdf)
        if ok and os.path.exists(out_pdf):
            with open(out_pdf, "rb") as f:
                return f.read()
        return None

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/calculate")
async def calculate_endpoint(
    total_pages: int = Form(...),
    signature_size: int = Form(16),                   # размер тетради (стр)
    sheets_per_signature: Optional[int] = Form(None), # листов A4 на тетрадь (приоритет)
    num_signatures: Optional[int] = Form(None),       # количество тетрадей (второй приоритет)
    pages_per_sheet_side: int = Form(2),
    duplex: bool = Form(True),
    page_format: str = Form("A5"),
    sheet_format: str = Form("A4"),
    sheet_width_mm: float = Form(210),
    sheet_height_mm: float = Form(297),
    page_width_mm: float = Form(148),
    page_height_mm: float = Form(210),
    creep_mm_per_leaf: float = Form(0.1),
    gsm: int = Form(80),
):
    # Приоритет: листов на тетрадь → число тетрадей → явный размер тетради
    if sheets_per_signature and sheets_per_signature > 0:
        signature_size = int(sheets_per_signature) * 4
    elif num_signatures and num_signatures > 0:
        raw_sig = math.ceil(total_pages / num_signatures)
        signature_size = int(math.ceil(raw_sig / 4) * 4)

    p = Params(
        total_pages=total_pages,
        signature_size=signature_size,
        pages_per_sheet_side=pages_per_sheet_side,
        duplex=duplex,
        page_format=page_format,
        sheet_format=sheet_format,
        sheet_width_mm=sheet_width_mm,
        sheet_height_mm=sheet_height_mm,
        page_width_mm=page_width_mm,
        page_height_mm=page_height_mm,
        creep_mm_per_leaf=creep_mm_per_leaf,
        gsm=gsm,
    )

    try:
        plan = build_plan(p)
        return JSONResponse(plan)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.post("/impose")
async def impose_endpoint(
    file: UploadFile = File(...),
    total_pages_hint: Optional[int] = Form(None),
    signature_size: int = Form(16),
    sheets_per_signature: Optional[int] = Form(None),
    num_signatures: Optional[int] = Form(None),
    sheet_width_mm: float = Form(210),
    sheet_height_mm: float = Form(297),
    page_width_mm: float = Form(148),
    page_height_mm: float = Form(210),
):
    # 1) читаем файл
    try:
        data = await file.read()
    except Exception as e:
        return JSONResponse({"error": f"Не удалось прочитать файл: {e}"}, status_code=400)

    name = file.filename or "input"
    ext = (os.path.splitext(name)[1] or "").lower()

    try:
        # 2) получаем PDF-байты (PDF — сразу; DOCX/EPUB/FB2 — через конвертацию)
        if ext == ".pdf":
            pdf_bytes = data
        else:
            pdf_bytes = convert_any_to_pdf(data, ext)
            if pdf_bytes is None:
                return JSONResponse(
                    {"error": "Формат не поддержан. Установите LibreOffice (`soffice`) или Calibre (`ebook-convert`) для конвертации в PDF."},
                    status_code=400
                )

        # 3) определяем фактическое число страниц
        detected_pages = detect_pdf_pages(pdf_bytes)
        total_pages = detected_pages if not total_pages_hint or total_pages_hint < detected_pages else total_pages_hint

        # 4) финализируем размер тетради по приоритетам
        if sheets_per_signature and sheets_per_signature > 0:
            signature_size = int(sheets_per_signature) * 4
        elif num_signatures and num_signatures > 0:
            raw_sig = math.ceil(total_pages / num_signatures)
            signature_size = int(math.ceil(raw_sig / 4) * 4)
        # иначе берём переданный signature_size

        # 5) параметры для импозиции
        p = Params(
            total_pages=total_pages,
            signature_size=signature_size,
            sheet_width_mm=sheet_width_mm,
            sheet_height_mm=sheet_height_mm,
            page_width_mm=page_width_mm,
            page_height_mm=page_height_mm,
        )

        # валидация/добивка пустыми
        _ = build_plan(p)

        # 6) генерим буклет
        imposed = impose_pdf_two_up(pdf_bytes, p)
        if not imposed or len(imposed) < 1000:
            return JSONResponse({"error": "PDF сгенерирован пустым — проверьте исходный файл."}, status_code=500)

        headers = {
            "Content-Disposition": 'attachment; filename="booklet.pdf"',
            "Content-Type": "application/pdf",
            "X-Detected-Pages": str(detected_pages),
            "X-Signature-Size": str(p.signature_size),
        }
        return StreamingResponse(io.BytesIO(imposed), media_type="application/pdf", headers=headers)

    except Exception as e:
        return JSONResponse({"error": f"Сбой при подготовке PDF: {e}"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
