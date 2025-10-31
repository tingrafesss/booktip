# app.py — бэкенд для расчёта и импозиции книги (экономная версия: только PDF, стрим в файл)
import io
import os
import math
import subprocess
import tempfile
import shutil
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

from fastapi import FastAPI, UploadFile, Form, Request, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import FileResponse

from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib.units import mm  # 1 mm -> points

# ---------------------------------------------------------------------
# FastAPI setup
# ---------------------------------------------------------------------
app = FastAPI(title="Идеальная книга — калькулятор и верстка", version="2.1-stream")
BASE_DIR = os.path.dirname(__file__)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def has_cmd(cmd: str) -> bool:
    from shutil import which
    return which(cmd) is not None

def run_quiet(args: list) -> int:
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
    page_width_mm: float = 148.0
    page_height_mm: float = 210.0
    creep_mm_per_leaf: float = 0.1
    gsm: int = 80

def _round2(x: float) -> float:
    return round(x + 1e-9, 2)

def page_order_for_signature(first: int, last: int) -> List[Dict]:
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

    signatures = []
    page = 1
    for _ in range(full):
        signatures.append({"start": page, "end": page + sig - 1})
        page += sig
    if rem:
        signatures.append({"start": page, "end": page + sig - 1})  # последняя с добивкой

    sig_plans = []
    for sgn in signatures:
        sheets = page_order_for_signature(sgn["start"], sgn["end"])
        sig_plans.append({"range": sgn, "sheets": sheets})

    pages_per_sheet = p.pages_per_sheet_side * (2 if p.duplex else 1)
    sheets_needed = padded_total // pages_per_sheet

    caliper_map = {60:0.08,70:0.09,80:0.1,90:0.11,100:0.12,115:0.135,120:0.145,130:0.155,150:0.175}
    leaf_thickness = caliper_map.get(p.gsm, 0.1)
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
# PDF imposition — потоково (лист за листом) с дописыванием к файлу
# ---------------------------------------------------------------------
def impose_pdf_streaming_to_path(pdf_bytes: bytes, p: Params, out_path: str) -> None:
    """
    Экономная по памяти импозиция:
    - создаём лист (две страницы) -> сохраняем во временный PDF
    - дописываем его к out_path через `pdfunite` (из пакета poppler-utils)
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    total = len(reader.pages)
    plan = build_plan(p)

    sheet_w_pt = p.sheet_width_mm * mm
    sheet_h_pt = p.sheet_height_mm * mm
    dst_width, dst_height = sheet_h_pt, sheet_w_pt
    half_w = dst_width / 2.0

    with tempfile.TemporaryDirectory() as td:
        # создаём пустой out_path
        open(out_path, "wb").close()
        have_any = False

        def append_pdf(in_pdf: str, out_pdf: str):
            nonlocal have_any
            if not have_any:
                shutil.copyfile(in_pdf, out_pdf)
                have_any = True
            else:
                prev = os.path.join(td, "prev.pdf")
                os.replace(out_pdf, prev)
                # объединяем: prev + in_pdf -> out_pdf
                ret = run_quiet(["pdfunite", prev, in_pdf, out_pdf])
                if ret != 0:
                    # на всякий случай восстановим предыдущую версию
                    shutil.copyfile(prev, out_pdf)
                    raise RuntimeError("Не удалось склеить PDF (pdfunite)")

        def get_src(n: int):
            if 1 <= n <= total:
                return reader.pages[n-1]
            return None

        for sig in plan["signatures"]:
            for sheet in sig["sheets"]:
                for side in ("front", "back"):
                    writer = PdfWriter()
                    dst = writer.add_blank_page(width=dst_width, height=dst_height)

                    left_num, right_num = sheet[side]
                    left = get_src(left_num)
                    right = get_src(right_num)

                    # если хочется, чтобы "оборот" выглядел визуально как при печати, раскомментируй:
                    # if side == "back":
                    #     left, right = right, left

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

                    # пишем текущий лист и тут же дописываем его к итоговому PDF
                    one_sheet = os.path.join(td, "sheet.pdf")
                    with open(one_sheet, "wb") as f:
                        writer.write(f)
                    append_pdf(one_sheet, out_path)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def detect_pdf_pages(pdf_bytes: bytes) -> int:
    return len(PdfReader(io.BytesIO(pdf_bytes)).pages)

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/calculate")
async def calculate_endpoint(
    total_pages: int = Form(...),
    signature_size: int = Form(16),
    sheets_per_signature: Optional[int] = Form(None),
    num_signatures: Optional[int] = Form(None),
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

    # Временно поддерживаем только PDF (экономим память, без LibreOffice)
    if ext != ".pdf":
        raise HTTPException(status_code=400, detail="Только PDF-файлы поддерживаются на данный момент")

    pdf_bytes = data

    try:
        # 3) определяем фактическое число страниц
        detected_pages = detect_pdf_pages(pdf_bytes)
        total_pages = detected_pages if not total_pages_hint or total_pages_hint < detected_pages else total_pages_hint

        # 4) финализируем размер тетради по приоритетам
        if sheets_per_signature and sheets_per_signature > 0:
            signature_size = int(sheets_per_signature) * 4
        elif num_signatures and num_signatures > 0:
            raw_sig = math.ceil(total_pages / num_signatures)
            signature_size = int(math.ceil(raw_sig / 4) * 4)

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

        # 6) генерим буклет во временный файл и отдаём потоково
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            out_path = tmp.name

        try:
            impose_pdf_streaming_to_path(pdf_bytes, p, out_path)
            if os.path.getsize(out_path) < 1000:
                return JSONResponse({"error": "PDF сгенерирован пустым — проверьте исходный файл."}, status_code=500)

            resp = FileResponse(path=out_path, media_type="application/pdf", filename="booklet.pdf")
            resp.headers["X-Detected-Pages"] = str(detected_pages)
            resp.headers["X-Signature-Size"] = str(p.signature_size)
            return resp
        finally:
            # Файл удалится системой позже; можно добавить периодическую уборку, если захочешь
            pass

    except Exception as e:
        return JSONResponse({"error": f"Сбой при подготовке PDF: {e}"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
