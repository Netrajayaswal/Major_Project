from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse
import hashlib
import json
import mimetypes
import os
import threading
import time
import webbrowser

from signlang.pipeline import IncompleteTranslationError, TextToSignPipeline, UnsupportedGlossError


mimetypes.add_type("video/mp4", ".mp4")
mimetypes.add_type("video/webm", ".webm")


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Sign Language Translator</title>
  <style>
    :root {
      --bg: #09080d;
      --panel: #1e1f26;
      --panel-soft: #26242d;
      --line: rgba(207, 140, 255, 0.22);
      --text: #f6f1ff;
      --muted: #aaa6b4;
      --purple: #bd75ff;
      --purple-strong: #8d21ff;
      --blue: #4f87ff;
      --good: #74f0a7;
      --warn: #ffcf7b;
      --bad: #ff7c9b;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background:
        radial-gradient(circle at 25% 5%, rgba(141, 33, 255, 0.18), transparent 32rem),
        radial-gradient(circle at 75% 35%, rgba(79, 135, 255, 0.12), transparent 32rem),
        var(--bg);
    }

    .announcement {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 1rem;
      padding: 0.7rem 1rem;
      color: #17111c;
      background: #e7c8ff;
      font-weight: 800;
      letter-spacing: 0.01em;
    }

    .announcement span {
      padding: 0.6rem 1.3rem;
      border-radius: 999px;
      background: white;
      box-shadow: 0 0.3rem 0.8rem rgba(0, 0, 0, 0.22);
    }

    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 1.15rem 1.5rem;
      background: rgba(24, 24, 28, 0.92);
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      backdrop-filter: blur(16px);
    }

    .brand {
      font-size: clamp(1.35rem, 2vw, 1.9rem);
      font-weight: 850;
      letter-spacing: -0.04em;
    }

    .brand .dot { color: var(--purple); }
    .brand .by { color: #f7f0ff; opacity: 0.82; font-weight: 650; }

    nav {
      display: flex;
      gap: 0.55rem;
      align-items: center;
    }

    nav a {
      color: var(--text);
      text-decoration: none;
      font-weight: 800;
      padding: 0.65rem 0.85rem;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.035);
    }

    nav a.active {
      background: linear-gradient(135deg, var(--purple-strong), #6e16f2);
      box-shadow: 0 0.5rem 1.4rem rgba(141, 33, 255, 0.35);
    }

    main {
      width: min(1380px, calc(100vw - 2rem));
      margin: 1.15rem auto 2rem;
    }

    .tool-switch {
      display: inline-flex;
      align-items: center;
      gap: 0.6rem;
      padding: 0.7rem 1rem;
      margin: 0 0 1.1rem 0.2rem;
      color: #d99eff;
      border: 1px solid rgba(217, 158, 255, 0.6);
      border-radius: 0.35rem;
      background: rgba(60, 30, 80, 0.22);
      font-weight: 850;
    }

    .notice {
      max-width: 665px;
      margin: 0 auto 1.05rem;
      padding: 0.85rem 1.15rem;
      text-align: center;
      color: var(--muted);
      border: 1px solid rgba(141, 33, 255, 0.35);
      border-radius: 999px;
      background: rgba(58, 16, 84, 0.34);
      font-weight: 700;
    }

    .notice strong { color: var(--purple-strong); }

    .translator {
      overflow: hidden;
      border: 1px solid var(--line);
      border-top: 5px solid transparent;
      border-radius: 1.6rem 1.6rem 0.7rem 0.7rem;
      background:
        linear-gradient(var(--panel), var(--panel)) padding-box,
        linear-gradient(90deg, var(--blue), var(--purple-strong)) border-box;
      box-shadow: 0 1.5rem 5rem rgba(0, 0, 0, 0.35);
    }

    .language-bar {
      display: grid;
      grid-template-columns: 1fr auto 1fr;
      align-items: center;
      min-height: 3.75rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      background: rgba(255, 255, 255, 0.015);
    }

    .tabs {
      display: flex;
      align-items: center;
      gap: 0.3rem;
      padding: 0 1.1rem;
      color: var(--muted);
      font-weight: 850;
      letter-spacing: 0.02em;
    }

    .tab {
      padding: 1.25rem 1rem 1.05rem;
      border-bottom: 2px solid transparent;
      white-space: nowrap;
    }

    .tab.active {
      color: #d99eff;
      border-color: #d99eff;
    }

    .swap {
      color: var(--muted);
      font-size: 1.35rem;
      padding: 0.5rem 0.9rem;
      border-inline: 1px solid rgba(255, 255, 255, 0.08);
    }

    .panels {
      display: grid;
      grid-template-columns: 1fr 1fr;
      min-height: clamp(430px, 63vh, 720px);
    }

    .left, .right {
      position: relative;
      min-height: 100%;
      background: linear-gradient(145deg, rgba(255,255,255,0.018), transparent 40%);
    }

    .left { border-right: 1px solid rgba(255, 255, 255, 0.08); }

    textarea {
      width: 100%;
      height: 100%;
      min-height: clamp(430px, 63vh, 720px);
      padding: 2rem 1.35rem 4.5rem;
      resize: none;
      border: 0;
      outline: 0;
      color: var(--text);
      background: transparent;
      font: inherit;
      font-size: clamp(1.2rem, 1.8vw, 1.8rem);
      line-height: 1.45;
    }

    textarea::placeholder { color: rgba(246, 241, 255, 0.38); }

    .left-footer, .right-footer {
      position: absolute;
      inset-inline: 1.2rem;
      bottom: 1rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.8rem;
      color: var(--muted);
      font-size: 0.93rem;
    }

    .buttons {
      display: flex;
      flex-wrap: wrap;
      gap: 0.55rem;
      align-items: center;
    }

    button, .pill {
      border: 0;
      border-radius: 999px;
      color: var(--text);
      background: rgba(255, 255, 255, 0.075);
      padding: 0.65rem 0.9rem;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
    }

    button.primary {
      background: linear-gradient(135deg, var(--purple-strong), #5d7cff);
      box-shadow: 0 0.5rem 1.5rem rgba(141, 33, 255, 0.28);
    }

    button:disabled {
      cursor: not-allowed;
      opacity: 0.55;
    }

    label.auto {
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      color: var(--muted);
      font-weight: 800;
      user-select: none;
    }

    input[type="checkbox"] { accent-color: var(--purple-strong); }

    .video-wrap {
      position: absolute;
      inset: 1.2rem;
      display: grid;
      place-items: center;
      overflow: hidden;
      border-radius: 1.1rem;
      background:
        linear-gradient(135deg, rgba(141, 33, 255, 0.12), rgba(79, 135, 255, 0.08)),
        #101116;
      border: 1px solid rgba(255, 255, 255, 0.07);
    }

    video {
      width: min(100%, 640px);
      max-height: calc(100% - 3rem);
      border-radius: 0.9rem;
      background: #000;
      box-shadow: 0 1rem 3rem rgba(0, 0, 0, 0.45);
    }

    .empty {
      max-width: 27rem;
      padding: 2rem;
      text-align: center;
      color: var(--muted);
    }

    .empty .icon {
      width: 4.3rem;
      height: 4.3rem;
      display: grid;
      place-items: center;
      margin: 0 auto 1rem;
      border-radius: 1.2rem;
      background: rgba(141, 33, 255, 0.16);
      color: #d99eff;
      font-size: 2rem;
    }

    .empty h2 {
      margin: 0 0 0.5rem;
      color: var(--text);
      letter-spacing: -0.02em;
    }

    .meta {
      display: grid;
      gap: 0.55rem;
      padding: 0.9rem 1.1rem;
      border-top: 1px solid rgba(255, 255, 255, 0.08);
      background: rgba(0, 0, 0, 0.16);
    }

    .meta-row {
      display: flex;
      flex-wrap: wrap;
      gap: 0.55rem;
      align-items: center;
    }

    .status {
      color: var(--muted);
      font-weight: 850;
    }

    .status.good { color: var(--good); }
    .status.warn { color: var(--warn); }
    .status.bad { color: var(--bad); }

    .chip {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      max-width: 100%;
      padding: 0.45rem 0.65rem;
      border-radius: 999px;
      color: #e9ddff;
      background: rgba(141, 33, 255, 0.15);
      border: 1px solid rgba(141, 33, 255, 0.22);
      font-size: 0.88rem;
      font-weight: 800;
    }

    .examples {
      display: flex;
      flex-wrap: wrap;
      gap: 0.55rem;
      margin-top: 1rem;
      color: var(--muted);
      align-items: center;
    }

    .examples button {
      color: #dec4ff;
      border: 1px solid rgba(217, 158, 255, 0.24);
      background: rgba(217, 158, 255, 0.08);
    }

    @media (max-width: 900px) {
      header, nav { flex-wrap: wrap; }
      .announcement { flex-direction: column; gap: 0.45rem; text-align: center; }
      .language-bar { grid-template-columns: 1fr; }
      .swap { display: none; }
      .panels { grid-template-columns: 1fr; }
      .left { border-right: 0; border-bottom: 1px solid rgba(255, 255, 255, 0.08); }
      .video-wrap { position: relative; min-height: 420px; }
      .right-footer { position: relative; margin: 0 1rem 1rem; inset: auto; }
    }
  </style>
</head>
<body>
    <div class="announcement">
    <div>Local trained sign-language translator is ready</div>
    <span>Real-time local + remote sign translation</span>
  </div>

  <header>
    <div class="brand">sign<span class="dot">.</span>Translate <span class="by">hybrid text-to-skeleton mode</span></div>
    <nav aria-label="Navigation">
      <a href="#" class="active">Home</a>
      <a href="#examples">Examples</a>
      <a href="#model">Model</a>
    </nav>
  </header>

  <main>
    <div class="tool-switch">⌘ Text</div>
    <div class="notice">Trained signs are used when available. Unknown text can use remote sign translation for full-sentence skeleton output.</div>

    <section class="translator" aria-label="Text to sign translator">
      <div class="language-bar">
        <div class="tabs">
          <div class="tab">Detect language</div>
          <div class="tab active">English</div>
          <div class="tab">Sentence</div>
        </div>
        <div class="swap">⇄</div>
        <div class="tabs">
          <div class="tab active">🇺🇸 Skeleton Sign</div>
          <div class="tab">Trained Clips</div>
          <div class="tab">Video</div>
        </div>
      </div>

      <div class="panels">
        <div class="left">
          <textarea id="sourceText" maxlength="500" placeholder="Type any sentence, for example: The robot is dancing on Mars"></textarea>
          <div class="left-footer">
            <div class="buttons">
              <button class="primary" id="generateBtn">Generate</button>
              <button id="clearBtn">Clear</button>
              <label class="auto"><input id="autoToggle" type="checkbox" checked /> Real-time</label>
            </div>
            <span><span id="charCount">0</span> / 500</span>
          </div>
        </div>

        <div class="right">
          <div class="video-wrap">
            <div class="empty" id="emptyState">
              <div class="icon">☝</div>
              <h2>Output video appears here</h2>
              <p>Start typing on the left. The app generates one skeleton video using trained data and remote fallback when needed.</p>
            </div>
            <video id="resultVideo" controls autoplay muted loop playsinline hidden></video>
          </div>
          <div class="right-footer">
            <div class="status" id="statusText">Waiting for text…</div>
            <span class="pill" id="modelStatus">Loading model…</span>
          </div>
        </div>
      </div>

      <div class="meta" id="model">
        <div class="meta-row">
          <span class="chip">Gloss: <span id="glossText">—</span></span>
          <span class="chip">Mode: <span id="modeText">—</span></span>
          <span class="chip">Fallback: <span id="skippedText">—</span></span>
        </div>
        <div class="meta-row" id="examples">
          <span class="status">Try:</span>
          <button data-example="Tall wide clean beautiful">tall wide clean</button>
          <button data-example="The robot is dancing on Mars">random text demo</button>
          <button data-example="Beautiful young happy">beautiful young happy</button>
          <button data-example="Old narrow dirty">old narrow dirty</button>
        </div>
      </div>
    </section>
  </main>

  <script>
    const sourceText = document.getElementById("sourceText");
    const resultVideo = document.getElementById("resultVideo");
    const emptyState = document.getElementById("emptyState");
    const statusText = document.getElementById("statusText");
    const modelStatus = document.getElementById("modelStatus");
    const glossText = document.getElementById("glossText");
    const modeText = document.getElementById("modeText");
    const skippedText = document.getElementById("skippedText");
    const charCount = document.getElementById("charCount");
    const generateBtn = document.getElementById("generateBtn");
    const clearBtn = document.getElementById("clearBtn");
    const autoToggle = document.getElementById("autoToggle");

    let debounceTimer = null;
    let activeRequest = 0;
    let isRendering = false;
    let queuedText = "";
    let lastSubmittedText = "";

    function setStatus(message, tone = "") {
      statusText.textContent = message;
      statusText.className = `status ${tone}`.trim();
    }

    function resetOutput(message = "Waiting for text…") {
      resultVideo.pause();
      resultVideo.removeAttribute("src");
      resultVideo.hidden = true;
      emptyState.hidden = false;
      glossText.textContent = "—";
      modeText.textContent = "—";
      skippedText.textContent = "—";
      setStatus(message);
    }

    function updateCount() {
      charCount.textContent = sourceText.value.length;
    }

    function scheduleTranslate() {
      updateCount();
      clearTimeout(debounceTimer);
      const text = sourceText.value.trim();
      if (!text) {
        resetOutput();
        return;
      }
      if (!autoToggle.checked) {
        setStatus("Ready. Press Generate.", "warn");
        return;
      }
      setStatus("Typing detected…", "warn");
      debounceTimer = setTimeout(translate, 700);
    }

    async function translate() {
      const text = sourceText.value.trim();
      updateCount();
      if (!text) {
        resetOutput();
        return;
      }

      if (isRendering) {
        queuedText = text;
        setStatus("Finishing previous render, then updating…", "warn");
        return;
      }

      isRendering = true;
      lastSubmittedText = text;
      const requestId = ++activeRequest;
      generateBtn.disabled = true;
      setStatus("Rendering skeleton video…", "warn");

      try {
        const response = await fetch("/api/translate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        const contentType = response.headers.get("content-type") || "";
        const data = contentType.includes("application/json")
          ? await response.json()
          : { ok: false, message: await response.text() };
        if (requestId !== activeRequest) return;

        if (!response.ok || !data.ok) {
          resultVideo.hidden = true;
          emptyState.hidden = false;
          glossText.textContent = data.gloss || "—";
          modeText.textContent = data.mode || "—";
          skippedText.textContent = (data.fallback_text_words || data.ignored_text_words || data.unsupported_glosses || []).join(", ") || "—";
          setStatus(data.message || "Could not generate a trained sign.", "bad");
          return;
        }

        glossText.textContent = data.gloss || "—";
        modeText.textContent = data.mode || "—";
        skippedText.textContent = (data.fallback_text_words || data.ignored_text_words || []).join(", ") || "—";
        resultVideo.src = data.video_url;
        resultVideo.hidden = false;
        emptyState.hidden = true;
        resultVideo.load();
        resultVideo.play().catch(() => {});
        setStatus("Video ready.", "good");
      } catch (error) {
        setStatus(error.message ? `Render connection error: ${error.message}` : "Server error while rendering.", "bad");
      } finally {
        isRendering = false;
        if (requestId === activeRequest) {
          generateBtn.disabled = false;
        }
        const currentText = sourceText.value.trim();
        if (queuedText && queuedText !== lastSubmittedText && queuedText === currentText) {
          queuedText = "";
          setTimeout(translate, 120);
        } else {
          queuedText = "";
        }
      }
    }

    async function loadStatus() {
      try {
        const response = await fetch("/api/status");
        const data = await response.json();
        const count = data.supported_glosses ? data.supported_glosses.length : 0;
        modelStatus.textContent = `${count} trained signs + remote`;
      } catch {
        modelStatus.textContent = "Model status unavailable";
      }
    }

    sourceText.addEventListener("input", scheduleTranslate);
    resultVideo.addEventListener("canplay", () => setStatus("Video ready.", "good"));
    resultVideo.addEventListener("error", () => setStatus("Video rendered, but the browser could not play it.", "bad"));
    generateBtn.addEventListener("click", translate);
    clearBtn.addEventListener("click", () => {
      sourceText.value = "";
      updateCount();
      resetOutput();
      sourceText.focus();
    });
    document.querySelectorAll("[data-example]").forEach((button) => {
      button.addEventListener("click", () => {
        sourceText.value = button.dataset.example;
        scheduleTranslate();
        sourceText.focus();
      });
    });

    updateCount();
    loadStatus();
  </script>
</body>
</html>
"""


class SignTranslatorApp:
    def __init__(
        self,
        project_root=None,
        text_model_dir="outputs/checkpoints/text2gloss",
        pose_checkpoint_path="outputs/checkpoints/pose_transformer.pt",
        output_dir="outputs/videos/ui",
        pose_source="auto",
        fallback_mode="remote",
        pretrained_model_dir="outputs/checkpoints/pretrained_text2sign",
        remote_spoken_language="en",
        remote_signed_language="ase",
        allow_partial=True,
        fps=20,
        canvas_size=512,
    ):
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.output_dir = self._project_path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pose_source = pose_source
        self.fallback_mode = fallback_mode
        self.remote_spoken_language = remote_spoken_language
        self.remote_signed_language = remote_signed_language
        self.allow_partial = allow_partial
        self.fps = fps
        self.canvas_size = canvas_size
        self.pipeline = TextToSignPipeline(
            text_model_dir=self._project_path(text_model_dir),
            pose_checkpoint_path=self._project_path(pose_checkpoint_path),
            pretrained_model_dir=self._project_path(pretrained_model_dir),
            remote_spoken_language=self.remote_spoken_language,
            remote_signed_language=self.remote_signed_language,
            keypoint_dirs=[
                self.project_root / "data/keypoints/train",
                self.project_root / "data/keypoints/valid",
            ],
        )
        self._lock = threading.Lock()
        self._cache = {}

    def translate(self, text):
        text = " ".join(str(text or "").split())
        if not text:
            return {
                "ok": True,
                "empty": True,
                "message": "Waiting for text.",
            }

        cache_key = self._cache_key(text)
        with self._lock:
            cached = self._cache.get(cache_key)
            cached_video_path = Path(cached.get("browser_output_path", "")) if cached else None
            if cached and cached_video_path and _is_playable_video(cached_video_path):
                return dict(cached, video_url=self._video_url(cached_video_path))
            if cached_video_path and cached_video_path.exists() and not _is_playable_video(cached_video_path):
                cached_video_path.unlink(missing_ok=True)

            output_path = self.output_dir / f"sign_{cache_key}_{int(time.time() * 1000)}.mp4"
            try:
                result = self.pipeline.generate(
                    text,
                    output_path,
                    fps=self.fps,
                    canvas_size=self.canvas_size,
                    demo_if_missing=False,
                    pose_source=self.pose_source,
                    fallback_mode=self.fallback_mode,
                    require_complete=not self.allow_partial and self.fallback_mode == "strict",
                )
            except IncompleteTranslationError as exc:
                return {
                    "ok": False,
                        "message": (
                            "No accurate complete local video yet. Add training videos for missing words, "
                            "or switch to remote fallback for full sentence rendering."
                        ),
                    "gloss": exc.attempted_gloss,
                    "unsupported_glosses": exc.missing_words,
                    "supported_glosses": list(exc.supported_glosses),
                    "mode": "missing_trained_signs",
                }
            except UnsupportedGlossError as exc:
                unsupported = [
                    token for token in str(exc.attempted_gloss or "").split()
                    if token not in set(exc.supported_glosses)
                ]
                return {
                    "ok": False,
                    "message": "No trained sign was found for this input. Add videos for those words, then run prepare/train again.",
                    "gloss": exc.attempted_gloss,
                    "unsupported_glosses": unsupported,
                    "supported_glosses": list(exc.supported_glosses),
                    "mode": "unsupported",
                }
            except Exception as exc:
                if self.fallback_mode == "pretrained":
                    return {
                        "ok": False,
                        "message": (
                            "Pretrained text-to-sign model is not installed or could not run. "
                            "Run `python main.py download-pretrained`, then restart with "
                            "`python main.py ui --fallback pretrained`."
                        ),
                        "mode": "pretrained_unavailable",
                    }
                if self.fallback_mode == "remote":
                    return {
                        "ok": False,
                        "message": (
                            "Remote sign translation failed. Check internet and remote language codes, "
                            "or use local trained mode."
                        ),
                        "mode": "remote_unavailable",
                    }
                return {
                    "ok": False,
                    "message": f"Generation failed: {exc}",
                    "mode": "error",
                }

            source_output_path = Path(result["output_path"]).resolve()
            if not _is_playable_video(source_output_path):
                source_output_path.unlink(missing_ok=True)
                raise RuntimeError("Generated video has no playable frames.")
            browser_output_path = self._ensure_browser_video(source_output_path)
            if not _is_playable_video(browser_output_path):
                browser_output_path = source_output_path
            response = {
                "ok": True,
                "input_text": text,
                "output_path": str(source_output_path),
                "browser_output_path": str(browser_output_path),
                "video_url": self._video_url(browser_output_path),
                "gloss": result.get("gloss", ""),
                "attempted_gloss": result.get("attempted_gloss", ""),
                "mode": result.get("mode", ""),
                "missing": result.get("missing", []),
                "unsupported_glosses": result.get("unsupported_glosses", []),
                "ignored_text_words": result.get("ignored_text_words", []),
                "matched_text_words": result.get("matched_text_words", []),
                "fallback_text_words": result.get("fallback_text_words", []),
                "text_source": result.get("text_source", ""),
                "remote_error": result.get("remote_error", ""),
            }
            self._cache[cache_key] = response
            return response

    def status(self):
        supported_glosses = sorted(self.pipeline.supported_glosses())
        return {
            "ok": True,
            "pose_checkpoint": self.pipeline.pose_checkpoint_path.exists(),
            "text_checkpoint": self.pipeline._text_model_available(),
            "pose_source": self.pose_source,
            "fallback_mode": self.fallback_mode,
            "remote_spoken_language": self.remote_spoken_language,
            "remote_signed_language": self.remote_signed_language,
            "allow_partial": self.allow_partial,
            "fps": self.fps,
            "canvas_size": self.canvas_size,
            "supported_glosses": supported_glosses,
        }

    def video_path_from_url(self, request_path):
        name = unquote(Path(request_path).name)
        if not name:
            return None
        candidate = (self.output_dir / name).resolve()
        try:
            candidate.relative_to(self.output_dir)
        except ValueError:
            return None
        return candidate if candidate.exists() and candidate.is_file() else None

    def _project_path(self, path):
        path = Path(path)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    def _cache_key(self, text):
        payload = json.dumps(
            {
                "text": text,
                "pose_source": self.pose_source,
                "fallback_mode": self.fallback_mode,
                "remote_spoken_language": self.remote_spoken_language,
                "remote_signed_language": self.remote_signed_language,
                "allow_partial": self.allow_partial,
                "fps": self.fps,
                "canvas_size": self.canvas_size,
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def _video_url(self, output_path):
        return f"/videos/{Path(output_path).name}?v={int(Path(output_path).stat().st_mtime)}"

    def _ensure_browser_video(self, source_path):
        source_path = Path(source_path)
        target_path = source_path.with_suffix(".webm")
        if (
            target_path.exists()
            and target_path.stat().st_mtime >= source_path.stat().st_mtime
            and _is_playable_video(target_path)
        ):
            return target_path.resolve()
        try:
            converted = _convert_video_to_webm(source_path, target_path, self.fps, self.canvas_size)
        except Exception:
            return source_path.resolve()
        if converted.exists() and _is_playable_video(converted):
            return converted.resolve()
        target_path.unlink(missing_ok=True)
        return source_path.resolve()


class SignTranslatorRequestHandler(BaseHTTPRequestHandler):
    server_version = "SignTranslatorHTTP/1.0"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send_html(INDEX_HTML)
            return
        if parsed.path == "/api/status":
            self._send_json(self.server.app.status())
            return
        if parsed.path.startswith("/videos/"):
            self._send_video(parsed.path)
            return
        self.send_error(404, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/translate":
            self.send_error(404, "Not found")
            return

        try:
            payload = self._read_json()
        except ValueError as exc:
            self._send_json({"ok": False, "message": str(exc)}, status=400)
            return

        try:
            result = self.server.app.translate(payload.get("text", ""))
        except Exception as exc:
            result = {
                "ok": False,
                "message": f"Server failed while rendering: {exc}",
                "mode": "error",
            }
        self._send_json(result, status=200 if result.get("ok") else 422)

    def log_message(self, format, *args):
        return

    def _read_json(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length > 20_000:
            raise ValueError("Request is too large.")
        raw = self.rfile.read(content_length)
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON body.") from exc

    def _send_html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self._safe_write(body)

    def _send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self._safe_write(body)

    def _send_video(self, request_path):
        path = self.server.app.video_path_from_url(request_path)
        if path is None:
            self.send_error(404, "Video not found")
            return

        file_size = path.stat().st_size
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        range_header = self.headers.get("Range")
        start = 0
        end = file_size - 1
        status = 200

        if range_header and range_header.startswith("bytes="):
            status = 206
            byte_range = range_header.replace("bytes=", "", 1).split("-", 1)
            if byte_range[0]:
                start = int(byte_range[0])
            if len(byte_range) > 1 and byte_range[1]:
                end = int(byte_range[1])
            end = min(end, file_size - 1)

        if start < 0 or start >= file_size or end < start:
            self.send_error(416, "Requested range not satisfiable")
            return

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()

        with open(path, "rb") as file:
            file.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = file.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                if not self._safe_write(chunk):
                    break
                remaining -= len(chunk)

    def _safe_write(self, body):
        try:
            self.wfile.write(body)
            return True
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return False


class SignTranslatorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, app):
        super().__init__(server_address, SignTranslatorRequestHandler)
        self.app = app


def _convert_video_to_webm(source_path, target_path, fps, canvas_size):
    import cv2
    import numpy as np

    source_path = Path(source_path)
    target_path = Path(target_path)
    temporary_path = target_path.with_suffix(".tmp.webm")
    temporary_path.unlink(missing_ok=True)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        return source_path

    writer = cv2.VideoWriter(
        str(temporary_path),
        cv2.VideoWriter_fourcc(*"VP80"),
        fps,
        (canvas_size, canvas_size),
    )
    if not writer.isOpened():
        capture.release()
        return source_path

    written = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(_fit_frame_to_square(frame, canvas_size))
            written += 1
    finally:
        capture.release()
        writer.release()

    if written == 0:
        temporary_path.unlink(missing_ok=True)
        return source_path

    temporary_path.replace(target_path)
    return target_path


def _fit_frame_to_square(frame, canvas_size):
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    scale = min(canvas_size / max(width, 1), canvas_size / max(height, 1))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    x_offset = (canvas_size - resized_width) // 2
    y_offset = (canvas_size - resized_height) // 2
    canvas[y_offset : y_offset + resized_height, x_offset : x_offset + resized_width] = resized
    return canvas


def _is_playable_video(path):
    import cv2

    path = Path(path)
    if not path.exists() or path.stat().st_size <= 64:
        return False
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return False
    ok, frame = capture.read()
    capture.release()
    return bool(ok and frame is not None and frame.size > 0)


def run_web_app(
    host="127.0.0.1",
    port=7860,
    project_root=None,
    text_model_dir="outputs/checkpoints/text2gloss",
    pose_checkpoint_path="outputs/checkpoints/pose_transformer.pt",
    output_dir="outputs/videos/ui",
    pose_source="auto",
    fallback_mode="remote",
    pretrained_model_dir="outputs/checkpoints/pretrained_text2sign",
    remote_spoken_language="en",
    remote_signed_language="ase",
    allow_partial=True,
    fps=20,
    canvas_size=512,
    open_browser=True,
):
    project_root = Path(project_root or Path.cwd()).resolve()
    os.chdir(project_root)
    app = SignTranslatorApp(
        project_root=project_root,
        text_model_dir=text_model_dir,
        pose_checkpoint_path=pose_checkpoint_path,
        output_dir=output_dir,
        pose_source=pose_source,
        fallback_mode=fallback_mode,
        pretrained_model_dir=pretrained_model_dir,
        remote_spoken_language=remote_spoken_language,
        remote_signed_language=remote_signed_language,
        allow_partial=allow_partial,
        fps=fps,
        canvas_size=canvas_size,
    )
    server = SignTranslatorServer((host, port), app)
    url = f"http://{host}:{port}"

    print(f"Sign language translator UI running at {url}")
    print("Type text in the left panel; one skeleton video appears on the right.")
    print("Press Ctrl+C to stop.")

    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping UI server.")
    finally:
        server.server_close()
