const calcBtn = document.getElementById('calcBtn');
const planDiv = document.getElementById('plan');
const msgDiv = document.getElementById('msg');
const downloadBtn = document.getElementById('downloadBtn');

function setHidden(fileForm, name, value) {
  let el = fileForm.querySelector(`input[name="${name}"]`);
  if (!el) {
    el = document.createElement('input');
    el.type = 'hidden';
    el.name = name;
    fileForm.appendChild(el);
  }
  el.value = value ?? '';
}

/* === живой мост между формами === */
function syncCalcToFile() {
  const calcForm = document.getElementById('calcForm');
  const fileForm = document.getElementById('fileForm');
  const fd = new FormData(calcForm);

  const duplexEl = calcForm.querySelector('[name="duplex"]');
  if (duplexEl) fd.set('duplex', duplexEl.checked ? 'true' : 'false');

  const keys = [
    'total_pages',
    'signature_size',
    'sheet_width_mm',
    'sheet_height_mm',
    'page_width_mm',
    'page_height_mm',
    'sheets_per_signature',
    'num_signatures',
    'duplex'
  ];
  keys.forEach(k => setHidden(fileForm, k, fd.get(k)));
}

/* === визуальная схема === */
function renderVisualPlan(plan) {
  const old = planDiv.querySelector('.visual');
  if (old) old.remove();

  const container = document.createElement('div');
  container.classList.add('visual');

  plan.signatures.forEach((sig, idx) => {
    const sigDiv = document.createElement('div');
    sigDiv.classList.add('sig');

    const t = document.createElement('h4');
    t.textContent = `Тетрадь ${idx + 1} — стр. ${sig.range.start}–${sig.range.end}`;
    sigDiv.appendChild(t);

    sig.sheets.forEach((sheet) => {
      const lblF = document.createElement('div');
      lblF.classList.add('small','muted');
      lblF.textContent = 'аверс';
      sigDiv.appendChild(lblF);

      const front = document.createElement('div');
      front.classList.add('sheet');
      sheet.front.forEach((n) => {
        const cell = document.createElement('div');
        cell.classList.add('pagebox');
        cell.textContent = n;
        front.appendChild(cell);
      });
      sigDiv.appendChild(front);

      const lblB = document.createElement('div');
      lblB.classList.add('small','muted');
      lblB.textContent = 'реверс';
      sigDiv.appendChild(lblB);

      const back = document.createElement('div');
      back.classList.add('sheet');
      sheet.back.forEach((n) => {
        const cell = document.createElement('div');
        cell.classList.add('pagebox');
        cell.textContent = n;
        back.appendChild(cell);
      });
      sigDiv.appendChild(back);
    });

    container.appendChild(sigDiv);
  });

  planDiv.appendChild(container);
}

/* === табличный вывод === */
function renderPlan(plan) {
  planDiv.classList.remove('hidden');

  const total = plan.input.total_pages;
  const padded = plan.padded_total_pages;
  const blanks = plan.blanks_added;

  const badges = `
    <span class="badge">листов: ${plan.sheets_needed}</span>
    <span class="badge">стр/лист: ${plan.pages_per_sheet}</span>
    <span class="badge">корешок ≈ ${plan.spine_mm_estimate} мм</span>
    <span class="badge">макс. увод ≈ ${plan.max_creep_mm_estimate} мм</span>
  `;

  let html = `<h3>Результат расчёта ${badges}</h3>
  <p>Всего страниц: <b>${total}</b>. После добивки: <b>${padded}</b> (пустых добавлено: ${blanks}).</p>`;

  if (plan.input && plan.input.signature_size) {
    html += `<p class="small muted">Размер тетради (фактический): <b>${plan.input.signature_size}</b> стр.</p>`;
  }

  plan.signatures.forEach((sig, idx) => {
    html += `<h4>Тетрадь ${idx + 1} — страницы ${sig.range.start}–${sig.range.end}</h4>`;
    html += `<table><thead><tr><th>Лист</th><th>Сторона 1 (аверс)</th><th>Сторона 2 (реверс)</th></tr></thead><tbody>`;
    sig.sheets.forEach((sh) => {
      html += `<tr><td>${sh.sheet}</td><td>${sh.front[0]}, ${sh.front[1]}</td><td>${sh.back[0]}, ${sh.back[1]}</td></tr>`;
    });
    html += `</tbody></table>`;
  });

  planDiv.innerHTML = `<div class="card">${html}</div>`;
  renderVisualPlan(plan);
}

/* === Рассчитать === */
calcBtn.addEventListener('click', async () => {
  const form = document.getElementById('calcForm');
  const fd = new FormData(form);
  const duplexEl = form.querySelector('[name="duplex"]');
  if (duplexEl) fd.set('duplex', duplexEl.checked ? 'true' : 'false');

  const sheetsPerSig = parseInt(fd.get('sheets_per_signature') || 0, 10);
  const totalPages = parseInt(fd.get('total_pages') || 0, 10);
  if (sheetsPerSig > 0 && totalPages > 0) {
    const sigSize = sheetsPerSig * 4;
    const nSigs = Math.ceil(totalPages / sigSize);
    fd.set('num_signatures', String(nSigs));
    const numEl = form.querySelector('[name="num_signatures"]');
    if (numEl) numEl.value = nSigs;
  }

  const res = await fetch('/calculate', { method: 'POST', body: fd });
  let data;
  try { data = await res.json(); } catch {
    planDiv.classList.remove('hidden');
    planDiv.innerHTML = `<div class="card">Ошибка: сервер вернул не-JSON.</div>`;
    return;
  }
  if (!res.ok) {
    planDiv.classList.remove('hidden');
    planDiv.innerHTML = `<div class="card">Ошибка: ${data.error || 'неизвестная'}</div>`;
    return;
  }

  renderPlan(data);

  const sigSize = data?.input?.signature_size ?? fd.get('signature_size');
  const fileForm = document.getElementById('fileForm');
  [
    'total_pages','sheet_width_mm','sheet_height_mm',
    'page_width_mm','page_height_mm','sheets_per_signature','num_signatures'
  ].forEach(k => setHidden(fileForm, k, fd.get(k)));

  setHidden(fileForm, 'signature_size', sigSize);
});

/* === Скачать готовый файл === */
downloadBtn.addEventListener('click', async () => {
  msgDiv.textContent = '';
  syncCalcToFile();
  const fileForm = document.getElementById('fileForm');
  const fileField = fileForm.querySelector('input[name="file"]');
  if (!fileField || !fileField.files || fileField.files.length === 0) {
    msgDiv.textContent = 'Пожалуйста, выберите файл.';
    return;
  }

  const fd = new FormData(fileForm);
  const res = await fetch('/impose', { method: 'POST', body: fd });

  const ct = res.headers.get('Content-Type') || '';
  if (!res.ok || !ct.includes('application/pdf')) {
    let txt = '';
    try {
      if (ct.includes('application/json')) {
        const e = await res.json();
        txt = e.error || JSON.stringify(e);
      } else {
        txt = await res.text();
      }
    } catch {}
    console.error('Impose error:', txt);
    msgDiv.textContent = txt || 'Ошибка подготовки файла.';
    return;
  }

  const blob = await res.blob();
  const disp = res.headers.get('Content-Disposition') || 'attachment; filename="booklet.pdf"';
  const filename = /filename="([^"]+)"/.exec(disp)?.[1] || 'booklet.pdf';

  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url + `#${Date.now()}`; // уникальный URL
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    URL.revokeObjectURL(url);
    a.remove();
  }, 0);

  const detected = res.headers.get('X-Detected-Pages');
  if (detected) msgDiv.textContent = `Определено страниц: ${detected}. Файл готов к печати.`;
});

/* === Пресеты === */
const formatSelect = document.getElementById('pageFormatSelect');
if (formatSelect) {
  formatSelect.addEventListener('change', () => {
    const w = document.querySelector('[name="page_width_mm"]');
    const h = document.querySelector('[name="page_height_mm"]');
    switch (formatSelect.value) {
      case 'A5': w.value = 148; h.value = 210; break;
      case 'Pocket': w.value = 107; h.value = 174; break;
      case 'A6': w.value = 105; h.value = 148; break;
      default: break;
    }
    syncCalcToFile();
  });
}

const sheetSelect = document.getElementById('sheetFormatSelect');
if (sheetSelect) {
  sheetSelect.addEventListener('change', () => {
    const w = document.querySelector('[name="sheet_width_mm"]');
    const h = document.querySelector('[name="sheet_height_mm"]');
    switch (sheetSelect.value) {
      case 'A4': w.value = 210; h.value = 297; break;
      default: break;
    }
    syncCalcToFile();
  });
}

['calcForm'].forEach(id => {
  const form = document.getElementById(id);
  if (!form) return;
  form.addEventListener('input', syncCalcToFile);
  form.addEventListener('change', syncCalcToFile);
});
syncCalcToFile();
