"""The page served at GET /: record or upload audio, get a speaker-labelled transcript.

Kept as a Python string so every host (Docker, Modal) ships it with app.py and no
static-file setup. Talks to /transcribe on the same origin, so no CORS config.
"""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kalam</title>
<meta name="description" content="Speaker-attributed transcription for hearings and meetings, in English and Arabic.">
<style>
:root{
  --bg:#f6f5f2;--panel:#ffffff;--ink:#1c1b19;--muted:#6b6862;--line:#e3e0da;
  --accent:#1f4d3a;--accent-ink:#ffffff;--soft:#eef3ef;--danger:#a3312a;--rec:#c0392b;
  --s0:#1f4d3a;--s1:#8a4b12;--s2:#2b4f8a;--s3:#7a2d5c;--s4:#4d6b12;--s5:#5c4a8a;
  --radius:14px;--shadow:0 1px 2px rgba(0,0,0,.04),0 4px 16px rgba(0,0,0,.05);
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#141413;--panel:#1d1c1a;--ink:#ecebe7;--muted:#a19d95;--line:#2f2d2a;
  --accent:#7fc7a4;--accent-ink:#10231a;--soft:#1f2a24;--danger:#f08a80;--rec:#ff6b5e;
  --s0:#7fc7a4;--s1:#e6a96b;--s2:#8fb3f0;--s3:#e290c2;--s4:#b6d36e;--s5:#b9a8f0;
  --shadow:none;
}}
:root[data-theme="dark"]{
  --bg:#141413;--panel:#1d1c1a;--ink:#ecebe7;--muted:#a19d95;--line:#2f2d2a;
  --accent:#7fc7a4;--accent-ink:#10231a;--soft:#1f2a24;--danger:#f08a80;--rec:#ff6b5e;
  --s0:#7fc7a4;--s1:#e6a96b;--s2:#8fb3f0;--s3:#e290c2;--s4:#b6d36e;--s5:#b9a8f0;
  --shadow:none;
}
*{box-sizing:border-box}
[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--ink);
  font:16px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans Arabic",sans-serif}
.wrap{max-width:760px;margin:0 auto;padding:40px 16px 64px}
header{margin-bottom:28px}
.brand{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}
h1{font-size:34px;letter-spacing:-.02em;margin:0}
.ar{font-size:26px;color:var(--muted)}
.tag{color:var(--muted);margin:6px 0 0}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow);padding:20px}
.tabs{display:flex;gap:6px;background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:4px;width:fit-content;margin-bottom:18px}
.tabs button{border:0;background:transparent;color:var(--muted);padding:7px 14px;border-radius:7px;font:inherit;font-size:14px;cursor:pointer}
.tabs button[aria-selected="true"]{background:var(--panel);color:var(--ink);box-shadow:var(--shadow)}
.source{min-height:96px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.rec-btn{width:64px;height:64px;border-radius:50%;border:2px solid var(--line);background:var(--panel);cursor:pointer;display:grid;place-items:center;flex:none}
.rec-btn span{width:24px;height:24px;border-radius:50%;background:var(--rec);transition:all .15s}
.rec-btn.on span{border-radius:5px;width:20px;height:20px}
.rec-btn.on{border-color:var(--rec)}
.timer{font-variant-numeric:tabular-nums;font-size:22px}
.hint{color:var(--muted);font-size:14px}
.drop{flex:1;border:1.5px dashed var(--line);border-radius:12px;padding:22px;text-align:center;cursor:pointer;color:var(--muted)}
.drop.over{border-color:var(--accent);background:var(--soft)}
.drop strong{color:var(--ink)}
audio{width:100%;margin-top:12px}
.opts{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:18px 0 6px}
label{display:block;font-size:13px;color:var(--muted);margin-bottom:4px}
select,input[type=text]{width:100%;padding:9px 10px;border:1px solid var(--line);border-radius:9px;background:var(--panel);color:var(--ink);font:inherit;font-size:15px}
.full{grid-column:1/-1}
.go{margin-top:14px;width:100%;padding:13px;border:0;border-radius:10px;background:var(--accent);color:var(--accent-ink);font:inherit;font-weight:600;font-size:16px;cursor:pointer}
.go:disabled{opacity:.45;cursor:not-allowed}
.status{margin-top:14px;font-size:14px;color:var(--muted);display:none;align-items:center;gap:10px}
.status.show{display:flex}
.spin{width:16px;height:16px;border:2px solid var(--line);border-top-color:var(--accent);border-radius:50%;animation:s 1s linear infinite;flex:none}
@keyframes s{to{transform:rotate(360deg)}}
.err{margin-top:14px;color:var(--danger);font-size:14px;display:none}
.result{margin-top:22px;display:none}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px}
.chip{font-size:13px;padding:3px 10px;border-radius:999px;background:var(--soft);color:var(--ink)}
.actions{display:flex;gap:8px;margin-left:auto}
.actions button{border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:8px;padding:5px 12px;font:inherit;font-size:13px;cursor:pointer}
.turn{display:grid;grid-template-columns:120px 1fr;gap:12px;padding:12px 0;border-top:1px solid var(--line)}
.who{font-weight:600;font-size:14px}
.who input{font:inherit;font-weight:600;color:inherit;background:transparent;border:0;border-bottom:1px dashed transparent;width:100%;padding:0}
.who input:hover,.who input:focus{border-bottom-color:var(--line);outline:none}
.when{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
.text{white-space:pre-wrap;word-wrap:break-word}
.text[dir="rtl"]{font-size:18px;line-height:1.8}
footer{margin-top:28px;color:var(--muted);font-size:13px}
@media (max-width:560px){
  .wrap{padding-top:24px}
  h1{font-size:28px}
  .turn{grid-template-columns:1fr;gap:2px}
  .actions{margin-left:0}
}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand"><h1>Kalam</h1><span class="ar" lang="ar">كلام</span></div>
    <p class="tag">Speaker-labelled transcripts of hearings and meetings, in English and Arabic.</p>
  </header>

  <main class="panel">
    <div class="tabs" role="tablist">
      <button role="tab" aria-selected="true" data-tab="record">Record</button>
      <button role="tab" aria-selected="false" data-tab="upload">Upload a file</button>
    </div>

    <section id="tab-record" class="source">
      <button class="rec-btn" id="recBtn" aria-label="Start recording"><span></span></button>
      <div>
        <div class="timer" id="timer">0:00</div>
        <div class="hint" id="recHint">Tap to record. Tap again to stop.</div>
      </div>
    </section>

    <section id="tab-upload" class="source" hidden>
      <label class="drop" id="drop" for="file">
        <strong>Choose an audio file</strong> or drop it here<br>
        <span class="hint">mp3, m4a, wav, webm, mp4 — up to 200 MB</span>
        <input type="file" id="file" accept="audio/*,video/mp4,video/webm" hidden>
      </label>
    </section>

    <audio id="preview" controls hidden></audio>

    <div class="opts">
      <div>
        <label for="language">Language</label>
        <select id="language">
          <option value="en">English</option>
          <option value="ar">Arabic</option>
          <option value="mixed">Arabic + English mixed</option>
          <option value="auto">Detect automatically</option>
        </select>
      </div>
      <div id="dialectBox" hidden>
        <label for="dialect">Arabic dialect</label>
        <select id="dialect">
          <option value="auto">Detect automatically</option>
          <option value="msa">Modern Standard / formal</option>
          <option value="gulf">Gulf (Khaleeji)</option>
          <option value="levantine">Levantine</option>
          <option value="egyptian">Egyptian</option>
          <option value="maghrebi">Maghrebi</option>
        </select>
      </div>
      <div>
        <label for="speakers">Number of speakers</label>
        <select id="speakers">
          <option value="">Not sure</option>
          <option>1</option><option>2</option><option>3</option><option>4</option>
          <option>5</option><option>6</option><option>7</option><option>8</option>
        </select>
      </div>
      <div class="full">
        <label for="vocab">Names or terms to spell correctly (optional)</label>
        <input type="text" id="vocab" maxlength="500" placeholder="e.g. Judge Al Mansoori, Emaar, Article 35" dir="auto">
      </div>
    </div>

    <button class="go" id="go" disabled>Transcribe</button>
    <div class="status" id="status"><div class="spin"></div><span id="statusText"></span></div>
    <div class="err" id="err"></div>

    <section class="result" id="result">
      <div class="meta" id="meta">
        <div class="actions">
          <button id="copy">Copy</button>
          <button id="download">Download .txt</button>
        </div>
      </div>
      <div id="turns"></div>
    </section>
  </main>

  <footer>Audio is processed and then deleted; nothing is stored. The first transcription after a quiet spell takes about a minute longer while the server wakes up.</footer>
</div>

<script>
(() => {
  const $ = id => document.getElementById(id);
  let blob = null, fileName = "recording", recorder = null, chunks = [], t0 = 0, tick = null, names = {};

  // tabs
  document.querySelectorAll('[role=tab]').forEach(b => b.onclick = () => {
    document.querySelectorAll('[role=tab]').forEach(x => x.setAttribute('aria-selected', x === b));
    $('tab-record').hidden = b.dataset.tab !== 'record';
    $('tab-upload').hidden = b.dataset.tab !== 'upload';
  });

  $('language').onchange = () => { $('dialectBox').hidden = !['ar','mixed'].includes($('language').value); };

  const fmt = s => { s = Math.max(0, Math.round(s)); const h = Math.floor(s/3600), m = Math.floor(s%3600/60), x = String(s%60).padStart(2,'0'); return h ? `${h}:${String(m).padStart(2,'0')}:${x}` : `${m}:${x}`; };

  function setAudio(b, name) {
    blob = b; fileName = name;
    const p = $('preview'); p.src = URL.createObjectURL(b); p.hidden = false;
    $('go').disabled = false;
  }

  // recording
  $('recBtn').onclick = async () => {
    if (recorder && recorder.state === 'recording') { recorder.stop(); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: true } });
      const type = ['audio/webm;codecs=opus','audio/webm','audio/mp4'].find(t => window.MediaRecorder && MediaRecorder.isTypeSupported(t)) || '';
      recorder = new MediaRecorder(stream, type ? { mimeType: type } : undefined);
      chunks = [];
      recorder.ondataavailable = e => e.data.size && chunks.push(e.data);
      recorder.onstop = () => {
        stream.getTracks().forEach(t => t.stop());
        clearInterval(tick);
        $('recBtn').classList.remove('on'); $('recBtn').setAttribute('aria-label','Start recording');
        $('recHint').textContent = 'Recorded. Tap to record again.';
        const mime = recorder.mimeType || 'audio/webm';
        setAudio(new Blob(chunks, { type: mime }), 'recording.' + (mime.includes('mp4') ? 'm4a' : 'webm'));
      };
      recorder.start(1000);
      t0 = Date.now(); $('timer').textContent = '0:00';
      tick = setInterval(() => $('timer').textContent = fmt((Date.now()-t0)/1000), 500);
      $('recBtn').classList.add('on'); $('recBtn').setAttribute('aria-label','Stop recording');
      $('recHint').textContent = 'Recording… tap to stop.';
    } catch (e) {
      showErr('Microphone not available: ' + (e.message || e) + '. You can upload a file instead.');
    }
  };

  // upload
  const pick = f => { if (f) setAudio(f, f.name); };
  $('file').onchange = e => pick(e.target.files[0]);
  const drop = $('drop');
  ['dragenter','dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave','drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', e => pick(e.dataTransfer.files[0]));

  function showErr(m) { const e = $('err'); e.textContent = m; e.style.display = m ? 'block' : 'none'; }

  // transcribe
  $('go').onclick = async () => {
    if (!blob) return;
    showErr(''); $('result').style.display = 'none';
    $('go').disabled = true;
    const st = $('status'), stt = $('statusText'); st.classList.add('show');
    const start = Date.now();
    const upd = () => {
      const s = (Date.now()-start)/1000;
      stt.textContent = `Transcribing… ${fmt(s)}` + (s > 20 ? ' — waking the server and loading the model can add about a minute' : '');
    };
    upd(); const iv = setInterval(upd, 1000);

    const fd = new FormData();
    fd.append('audio', blob, fileName);
    fd.append('language', $('language').value);
    if (!$('dialectBox').hidden) fd.append('dialect', $('dialect').value);
    if ($('speakers').value) fd.append('num_speakers', $('speakers').value);
    if ($('vocab').value.trim()) fd.append('vocabulary', $('vocab').value.trim());

    try {
      const r = await fetch('transcribe', { method: 'POST', body: fd });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || `Server error ${r.status}`);
      render(data);
    } catch (e) {
      showErr((e.message || String(e)) + (String(e).includes('Failed to fetch') ? ' (the request may have timed out — try a shorter clip)' : ''));
    } finally {
      clearInterval(iv); st.classList.remove('show'); $('go').disabled = false;
    }
  };

  const LANG = { en: 'English', ar: 'Arabic' };
  const DIAL = { msa: 'Formal / MSA', gulf: 'Gulf', levantine: 'Levantine', egyptian: 'Egyptian', maghrebi: 'Maghrebi' };
  let last = null;

  function label(spk) {
    if (names[spk]) return names[spk];
    const m = /(\d+)$/.exec(spk || '');
    return m ? `Speaker ${parseInt(m[1], 10) + 1}` : 'Speaker';
  }

  function render(data) {
    last = data; names = {};
    const meta = $('meta');
    meta.querySelectorAll('.chip').forEach(c => c.remove());
    const chips = [
      fmt(data.duration || 0) + ' of audio',
      (data.speakers || 0) + ' speaker' + (data.speakers === 1 ? '' : 's'),
      LANG[data.language] || data.language,
    ];
    if (data.dialect) chips.push(DIAL[data.dialect] + (data.dialect_detected ? ' (detected)' : ''));
    chips.reverse().forEach(t => { const c = document.createElement('span'); c.className = 'chip'; c.textContent = t; meta.prepend(c); });

    const order = [...new Set(data.segments.map(s => s.speaker))];
    const wrap = $('turns'); wrap.innerHTML = '';
    if (!data.segments.length) { wrap.innerHTML = '<p class="hint">No speech was found in this recording.</p>'; }
    data.segments.forEach(s => {
      const row = document.createElement('div'); row.className = 'turn';
      const color = `var(--s${order.indexOf(s.speaker) % 6})`;
      const who = document.createElement('div');
      const inp = document.createElement('input');
      inp.value = label(s.speaker); inp.dataset.spk = s.speaker; inp.title = 'Click to rename';
      inp.oninput = () => { names[s.speaker] = inp.value; document.querySelectorAll(`input[data-spk="${s.speaker}"]`).forEach(x => { if (x !== inp) x.value = inp.value; }); };
      const w = document.createElement('div'); w.className = 'who'; w.style.color = color; w.appendChild(inp);
      const when = document.createElement('div'); when.className = 'when'; when.textContent = fmt(s.start);
      who.append(w, when);
      const txt = document.createElement('div'); txt.className = 'text'; txt.dir = 'auto'; txt.textContent = s.text;
      if (/[؀-ۿ]/.test(s.text)) txt.dir = 'rtl';
      row.append(who, txt); wrap.appendChild(row);
    });
    $('result').style.display = 'block';
    $('result').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  const asText = () => last ? last.segments.map(s => `[${fmt(s.start)}] ${label(s.speaker)}: ${s.text}`).join('\n') : '';
  $('copy').onclick = async () => { try { await navigator.clipboard.writeText(asText()); $('copy').textContent = 'Copied'; setTimeout(() => $('copy').textContent = 'Copy', 1500); } catch {} };
  $('download').onclick = () => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([asText()], { type: 'text/plain;charset=utf-8' }));
    a.download = 'kalam-transcript.txt'; a.click();
  };
})();
</script>
</body>
</html>
"""
