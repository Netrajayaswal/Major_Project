from pathlib import Path
import csv
import json
import re

import numpy as np
import torch

from signlang.demo import make_demo_frames
from signlang.fingerspell import make_fingerspell_frames
from signlang.models.pose_transformer import PoseTransformerConfig, PoseTransformerModel, Vocabulary
from signlang.pretrained_text2sign import HuggingFaceText2SignProvider
from signlang.remote_signmt import fetch_signmt_pose_bytes, signmt_pose_to_frames
from signlang.render import draw_skeleton_frame, render_keypoints_video


GRAMMAR_STOPWORDS = {
    "A",
    "AN",
    "AND",
    "ARE",
    "BE",
    "FOR",
    "IN",
    "IS",
    "IT",
    "OF",
    "ON",
    "PLEASE",
    "SHOW",
    "SIGN",
    "THE",
    "THIS",
    "TO",
    "VERY",
}


def normalize_text_to_gloss_guess(text):
    tokens = re.findall(r"[A-Za-z0-9]+", text.upper())
    return " ".join(tokens) if tokens else "SIGN"


class UnsupportedGlossError(ValueError):
    def __init__(self, text, attempted_gloss, supported_glosses):
        self.text = text
        self.attempted_gloss = attempted_gloss
        self.supported_glosses = sorted(supported_glosses)
        message = (
            f"No trained sign was found for input '{text}'. "
            f"Attempted gloss: '{attempted_gloss or 'NONE'}'. "
            f"Supported trained glosses: {', '.join(self.supported_glosses) or 'none'}."
        )
        super().__init__(message)


class IncompleteTranslationError(ValueError):
    def __init__(self, text, attempted_gloss, missing_words, supported_glosses):
        self.text = text
        self.attempted_gloss = attempted_gloss
        self.missing_words = list(missing_words)
        self.supported_glosses = sorted(supported_glosses)
        message = (
            "No accurate complete sign video can be generated yet. "
            f"Missing trained signs for: {', '.join(self.missing_words) or 'none'}."
        )
        super().__init__(message)


class TextToSignPipeline:
    def __init__(
        self,
        text_model_dir="outputs/checkpoints/text2gloss",
        pose_checkpoint_path="outputs/checkpoints/pose_transformer.pt",
        keypoint_dirs=None,
        pretrained_model_dir="outputs/checkpoints/pretrained_text2sign",
        remote_spoken_language="en",
        remote_signed_language="ase",
        device=None,
    ):
        self.text_model_dir = Path(text_model_dir)
        self.pose_checkpoint_path = Path(pose_checkpoint_path)
        self.keypoint_dirs = [Path(path) for path in (keypoint_dirs or ["data/keypoints/train", "data/keypoints/valid"])]
        self.pretrained_model_dir = Path(pretrained_model_dir)
        self.remote_spoken_language = remote_spoken_language
        self.remote_signed_language = remote_signed_language
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._text_model = None
        self._tokenizer = None
        self._pose_model = None
        self._pose_vocab = None
        self._clip_index = None
        self._pretrained_provider = None

    def generate(
        self,
        text,
        output_path,
        fps=20,
        canvas_size=512,
        demo_if_missing=False,
        pose_source="auto",
        fallback_mode="strict",
        require_complete=False,
    ):
        missing = []
        if not self._text_model_available():
            missing.append("text2gloss")
        if not self.pose_checkpoint_path.exists():
            missing.append("pose")

        clip_index = self._load_clip_index()
        model_glosses = self.model_supported_glosses()
        clip_glosses = set(clip_index.keys())
        supported_glosses = model_glosses | clip_glosses

        if fallback_mode == "pretrained":
            return self.pretrained_text_to_video(text, output_path, missing=missing, fps=fps, canvas_size=canvas_size)
        if fallback_mode == "remote":
            try:
                return self.generate(
                    text=text,
                    output_path=output_path,
                    fps=fps,
                    canvas_size=canvas_size,
                    demo_if_missing=demo_if_missing,
                    pose_source=pose_source,
                    fallback_mode="strict",
                    require_complete=True,
                )
            except (IncompleteTranslationError, UnsupportedGlossError, FileNotFoundError, RuntimeError):
                try:
                    return self.remote_text_to_video(
                        text,
                        output_path,
                        missing=missing,
                        fps=fps,
                        canvas_size=canvas_size,
                    )
                except RuntimeError as exc:
                    fallback_result = self.text_to_any_video(
                        text,
                        output_path,
                        supported_glosses=supported_glosses,
                        model_glosses=model_glosses,
                        clip_glosses=clip_glosses,
                        missing=missing,
                        fps=fps,
                        canvas_size=canvas_size,
                        pose_source=pose_source,
                    )
                    fallback_result["mode"] = f"{fallback_result['mode']}_remote_unavailable"
                    fallback_result["remote_error"] = str(exc)
                    return fallback_result
        if fallback_mode == "fingerspell":
            return self.text_to_any_video(
                text,
                output_path,
                supported_glosses=supported_glosses,
                model_glosses=model_glosses,
                clip_glosses=clip_glosses,
                missing=missing,
                fps=fps,
                canvas_size=canvas_size,
                pose_source=pose_source,
            )
        if fallback_mode != "strict":
            raise ValueError("fallback_mode must be one of: strict, fingerspell, pretrained, remote")

        has_clip_data = bool(clip_index)
        if "pose" in missing and not has_clip_data:
            if not demo_if_missing:
                raise FileNotFoundError(
                    "Missing pose checkpoint and no keypoint clips were found. "
                    "Run prepare/train-pose first, or use --allow-preview for renderer testing."
                )
            gloss = normalize_text_to_gloss_guess(text)
            frames = make_demo_frames(gloss, fps=fps)
            render_keypoints_video(frames, output_path, fps=fps, canvas_size=canvas_size)
            return {
                "output_path": str(output_path),
                "gloss": gloss,
                "mode": "untrained_preview",
                "missing": missing,
            }

        if self._text_model_available():
            attempted_gloss = self.text_to_gloss(text)
            text_source = "text2gloss_checkpoint"
            text_analysis = {
                "matched_text_words": [],
                "ignored_text_words": [],
            }
            gloss, unsupported_glosses = self._filter_supported_glosses(attempted_gloss, supported_glosses)
            local_attempted_gloss, local_analysis = self.local_text_to_gloss(
                text,
                supported_glosses,
                return_analysis=True,
            )
            local_gloss, local_unsupported = self._filter_supported_glosses(local_attempted_gloss, supported_glosses)
            local_complete = not local_analysis.get("ignored_text_words") and not local_unsupported
            model_complete = not unsupported_glosses
            model_token_count = len(gloss.split()) if gloss else 0
            local_token_count = len(local_gloss.split()) if local_gloss else 0
            input_tokens = [token for token in _text_tokens(text) if token not in GRAMMAR_STOPWORDS]
            model_tokens = [token for token in gloss.split() if token]
            model_semantic_mismatch = (
                bool(model_tokens)
                and bool(input_tokens)
                and not (set(model_tokens) & set(input_tokens))
            )
            prefer_local = (
                (require_complete and not model_complete and local_complete)
                or (model_token_count == 0 and local_token_count > 0)
                or (local_token_count > model_token_count)
                or model_semantic_mismatch
            )
            if prefer_local:
                attempted_gloss = local_attempted_gloss
                gloss = local_gloss
                unsupported_glosses = local_unsupported
                text_source = "local_trained_gloss_match"
                text_analysis = local_analysis
        else:
            attempted_gloss, text_analysis = self.local_text_to_gloss(
                text,
                supported_glosses,
                return_analysis=True,
            )
            text_source = "local_trained_gloss_match"
            gloss, unsupported_glosses = self._filter_supported_glosses(attempted_gloss, supported_glosses)

        missing_text_words = text_analysis.get("ignored_text_words", [])
        if require_complete and missing_text_words:
            raise IncompleteTranslationError(text, attempted_gloss, missing_text_words, supported_glosses)
        if require_complete and unsupported_glosses:
            raise IncompleteTranslationError(text, attempted_gloss, unsupported_glosses, supported_glosses)
        if not gloss:
            if demo_if_missing:
                frames = make_demo_frames(attempted_gloss or normalize_text_to_gloss_guess(text), fps=fps)
                render_keypoints_video(frames, output_path, fps=fps, canvas_size=canvas_size)
                return {
                    "output_path": str(output_path),
                    "gloss": attempted_gloss,
                    "mode": "untrained_preview",
                    "missing": missing,
                    "unsupported_glosses": unsupported_glosses,
                    "text_source": text_source,
                    "ignored_text_words": text_analysis.get("ignored_text_words", []),
                }
            raise UnsupportedGlossError(text, attempted_gloss, supported_glosses)

        source = self._resolve_pose_source(pose_source, gloss, model_glosses, clip_glosses)
        if source == "clips":
            if self.source_clips_to_video(gloss, output_path, fps=fps, canvas_size=canvas_size):
                mode = "training_source_video_retrieval"
            else:
                frames = self.clips_to_frames(gloss)
                render_keypoints_video(frames, output_path, fps=fps, canvas_size=canvas_size)
                mode = "training_keypoint_clip_retrieval"
        else:
            frames = self.gloss_to_frames(gloss)
            render_keypoints_video(frames, output_path, fps=fps, canvas_size=canvas_size)
            mode = "trained_pose_model"

        return {
            "output_path": str(output_path),
            "gloss": gloss,
            "attempted_gloss": attempted_gloss,
            "mode": mode,
            "missing": missing,
            "unsupported_glosses": unsupported_glosses,
            "text_source": text_source,
            "ignored_text_words": text_analysis.get("ignored_text_words", []),
            "matched_text_words": text_analysis.get("matched_text_words", []),
        }

    def text_to_any_video(
        self,
        text,
        output_path,
        supported_glosses=None,
        model_glosses=None,
        clip_glosses=None,
        missing=None,
        fps=20,
        canvas_size=512,
        pose_source="auto",
    ):
        supported_glosses = supported_glosses if supported_glosses is not None else self.supported_glosses()
        model_glosses = model_glosses if model_glosses is not None else self.model_supported_glosses()
        clip_glosses = clip_glosses if clip_glosses is not None else set(self._load_clip_index().keys())
        missing = missing or []
        segments = _segment_text_for_universal_output(text, supported_glosses)
        if not segments:
            segments = [{"kind": "fingerspell", "text": "SIGN", "gloss": "FS:SIGN"}]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import cv2

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (canvas_size, canvas_size),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {output_path}")

        trained_glosses = []
        fingerspelled_words = []
        unsupported_glosses = []
        written = 0
        try:
            for segment_index, segment in enumerate(segments):
                if segment["kind"] == "trained":
                    ok = self._write_trained_segment(
                        writer,
                        segment["gloss"],
                        fps=fps,
                        canvas_size=canvas_size,
                        pose_source=pose_source,
                        model_glosses=model_glosses,
                        clip_glosses=clip_glosses,
                    )
                    if ok:
                        trained_glosses.append(segment["gloss"])
                        written += 1
                    else:
                        unsupported_glosses.append(segment["gloss"])
                        written += _write_fingerspell_segment(
                            writer,
                            segment["text"],
                            fps=fps,
                            canvas_size=canvas_size,
                            label=f"FS {segment['text']}",
                        )
                        fingerspelled_words.append(segment["text"])
                else:
                    written += _write_fingerspell_segment(
                        writer,
                        segment["text"],
                        fps=fps,
                        canvas_size=canvas_size,
                        label=f"FS {segment['text']}",
                    )
                    fingerspelled_words.append(segment["text"])

                if segment_index < len(segments) - 1:
                    written += _write_blank_frames(writer, canvas_size, max(2, fps // 8))
        finally:
            writer.release()

        if written == 0 and output_path.exists():
            output_path.unlink(missing_ok=True)
            raise RuntimeError("No frames were written for the requested text.")

        if fingerspelled_words and trained_glosses:
            mode = "trained_plus_fingerspell_fallback"
        elif fingerspelled_words:
            mode = "fingerspell_fallback"
        else:
            mode = "trained_universal_sequence"

        gloss = " ".join(
            segment["gloss"] if segment["kind"] == "trained" else segment["gloss"]
            for segment in segments
        )
        return {
            "output_path": str(output_path),
            "gloss": gloss,
            "attempted_gloss": normalize_text_to_gloss_guess(text),
            "mode": mode,
            "missing": missing,
            "unsupported_glosses": unsupported_glosses,
            "text_source": "local_universal_text_match",
            "ignored_text_words": [],
            "matched_text_words": [segment["text"] for segment in segments if segment["kind"] == "trained"],
            "fallback_text_words": fingerspelled_words,
        }

    def pretrained_text_to_video(self, text, output_path, missing=None, fps=20, canvas_size=512):
        provider = self._load_pretrained_provider(fps=fps, canvas_size=canvas_size)
        provider.generate_video(text, output_path)
        return {
            "output_path": str(output_path),
            "gloss": normalize_text_to_gloss_guess(text),
            "attempted_gloss": normalize_text_to_gloss_guess(text),
            "mode": "pretrained_text2sign",
            "missing": missing or [],
            "unsupported_glosses": [],
            "text_source": "pretrained_text2sign_provider",
            "ignored_text_words": [],
            "matched_text_words": _text_tokens(text),
            "fallback_text_words": [],
        }

    def remote_text_to_video(self, text, output_path, missing=None, fps=20, canvas_size=512):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            pose_bytes = fetch_signmt_pose_bytes(
                text=text,
                spoken_language=self.remote_spoken_language,
                signed_language=self.remote_signed_language,
            )
            frames, remote_fps = signmt_pose_to_frames(pose_bytes)
            if len(frames) == 0:
                raise RuntimeError("Remote pose output contained zero frames.")
            target_fps = fps or max(1, int(round(remote_fps)))
            render_keypoints_video(frames, output_path, fps=target_fps, canvas_size=canvas_size)
        except Exception as exc:
            raise RuntimeError(f"Remote sign.mt translation failed: {exc}") from exc
        return {
            "output_path": str(output_path),
            "gloss": normalize_text_to_gloss_guess(text),
            "attempted_gloss": normalize_text_to_gloss_guess(text),
            "mode": "remote_signmt_pose_render",
            "missing": missing or [],
            "unsupported_glosses": [],
            "text_source": "remote_signmt_api",
            "ignored_text_words": [],
            "matched_text_words": _text_tokens(text),
            "fallback_text_words": [],
        }

    def text_to_gloss(self, text, max_length=64):
        self._load_text_model()
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        generated = self._text_model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
        )
        return self._tokenizer.decode(generated[0], skip_special_tokens=True).strip().upper()

    def gloss_to_frames(self, gloss, max_frames=160):
        self._load_pose_model()
        unsupported = [
            token for token in gloss.split()
            if token not in self._pose_vocab.token_to_id
        ]
        if unsupported:
            raise UnsupportedGlossError(gloss, gloss, self.model_supported_glosses())
        token_ids = self._pose_vocab.encode(gloss)
        gloss_tokens = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        frames = self._pose_model.greedy_decode(gloss_tokens, max_frames=max_frames)
        return frames.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def clips_to_frames(self, gloss):
        chunks = []
        for data in self._clip_records_for_gloss(gloss):
            frames = np.asarray(data["frames"], dtype=np.float32)
            chunks.append(frames)
            chunks.append(np.zeros((4, frames.shape[1]), dtype=np.float32))
        if not chunks:
            raise UnsupportedGlossError(gloss, gloss, self._load_clip_index().keys())
        return np.concatenate(chunks, axis=0)

    def source_clips_to_video(self, gloss, output_path, fps=20, canvas_size=512, pause_frames=4):
        records = self._clip_records_for_gloss(gloss)
        if not records or not all(_is_rendered_source_record(record) for record in records):
            return False

        import cv2

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (canvas_size, canvas_size),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {output_path}")

        written = 0
        try:
            for record_index, record in enumerate(records):
                cap = cv2.VideoCapture(str(record["source_video"]))
                try:
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        writer.write(_fit_frame_to_canvas(frame, canvas_size))
                        written += 1
                finally:
                    cap.release()
                if record_index < len(records) - 1:
                    blank = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
                    for _ in range(pause_frames):
                        writer.write(blank)
                        written += 1
        finally:
            writer.release()

        if written == 0:
            output_path.unlink(missing_ok=True)
            return False
        return True

    def _write_trained_segment(
        self,
        writer,
        gloss,
        fps=20,
        canvas_size=512,
        pose_source="auto",
        model_glosses=None,
        clip_glosses=None,
    ):
        model_glosses = model_glosses if model_glosses is not None else self.model_supported_glosses()
        clip_glosses = clip_glosses if clip_glosses is not None else set(self._load_clip_index().keys())
        try:
            source = self._resolve_pose_source(pose_source, gloss, model_glosses, clip_glosses)
        except Exception:
            source = "clips" if self._sequence_supported(gloss, clip_glosses) else "model"

        if source == "clips" and self._write_source_clip_segment(writer, gloss, canvas_size=canvas_size):
            return True

        try:
            frames = self.clips_to_frames(gloss) if source == "clips" else self.gloss_to_frames(gloss)
        except Exception:
            try:
                frames = self.clips_to_frames(gloss)
            except Exception:
                try:
                    frames = self.gloss_to_frames(gloss)
                except Exception:
                    return False

        _write_keypoint_frames(writer, frames, canvas_size=canvas_size)
        return True

    def _write_source_clip_segment(self, writer, gloss, canvas_size=512):
        records = self._clip_records_for_gloss(gloss)
        if not records or not all(_is_rendered_source_record(record) for record in records):
            return False

        import cv2

        written = 0
        for record in records:
            cap = cv2.VideoCapture(str(record["source_video"]))
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    writer.write(_fit_frame_to_canvas(frame, canvas_size))
                    written += 1
            finally:
                cap.release()
        return written > 0

    def local_text_to_gloss(self, text, supported_glosses, return_analysis=False):
        phrase_gloss = self._lookup_phrase_gloss(text)
        if phrase_gloss:
            analysis = {
                "matched_text_words": _text_tokens(text),
                "ignored_text_words": [],
            }
            return (phrase_gloss, analysis) if return_analysis else phrase_gloss

        text_tokens = _text_tokens(text)
        matched, matched_indices = _match_supported_glosses_in_order(text_tokens, supported_glosses)
        analysis = {
            "matched_text_words": [text_tokens[index] for index in sorted(matched_indices)],
            "ignored_text_words": [
                token for index, token in enumerate(text_tokens)
                if index not in matched_indices and token not in GRAMMAR_STOPWORDS
            ],
        }
        gloss = " ".join(matched)
        return (gloss, analysis) if return_analysis else gloss

    def supported_glosses(self):
        return self.model_supported_glosses() | set(self._load_clip_index().keys())

    def model_supported_glosses(self):
        glosses = set()
        if self.pose_checkpoint_path.exists():
            self._load_pose_model()
            glosses.update(
                token
                for token in self._pose_vocab.token_to_id
                if not token.startswith("<")
            )
        return glosses

    def _load_text_model(self):
        if self._text_model is not None:
            return
        from transformers import MT5ForConditionalGeneration, MT5Tokenizer

        model_dir = self._resolve_text_model_dir()
        self._tokenizer = MT5Tokenizer.from_pretrained(model_dir)
        self._text_model = MT5ForConditionalGeneration.from_pretrained(model_dir)
        self._text_model.to(self.device)
        self._text_model.eval()

    def _load_pretrained_provider(self, fps=20, canvas_size=512):
        if self._pretrained_provider is None:
            self._pretrained_provider = HuggingFaceText2SignProvider(
                model_dir=self.pretrained_model_dir,
                device=self.device,
                fps=fps,
                canvas_size=canvas_size,
            )
        return self._pretrained_provider

    def _load_pose_model(self):
        if self._pose_model is not None:
            return
        checkpoint = torch.load(self.pose_checkpoint_path, map_location=self.device)
        config = PoseTransformerConfig(**checkpoint["config"])
        vocab = Vocabulary.from_dict(checkpoint["vocab"])
        model = PoseTransformerModel(config)
        model.load_state_dict(checkpoint["model_state"])
        model.to(self.device)
        model.eval()
        self._pose_model = model
        self._pose_vocab = vocab

    def _load_clip_index(self):
        if self._clip_index is not None:
            return self._clip_index
        clip_index = {}
        for directory in self.keypoint_dirs:
            if not directory.exists():
                continue
            for path in sorted(directory.rglob("*.json")):
                try:
                    with open(path, "r", encoding="utf-8") as file:
                        data = json.load(file)
                except (OSError, json.JSONDecodeError):
                    continue
                gloss = str(data.get("gloss", "")).strip().upper()
                if gloss:
                    clip_index.setdefault(gloss, []).append(path)
        self._clip_index = clip_index
        return self._clip_index

    def _lookup_phrase_gloss(self, text):
        normalized_text = _normalize_phrase(text)
        for csv_path in [Path("data/text_gloss/train.csv"), Path("data/text_gloss/valid.csv")]:
            if not csv_path.exists():
                continue
            with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    sentence = row.get("sentence", "")
                    gloss = (row.get("gloss_sequence") or row.get("gloss") or "").strip().upper()
                    if gloss and _normalize_phrase(sentence) == normalized_text:
                        return gloss
        return ""

    def _filter_supported_glosses(self, gloss, supported_glosses):
        gloss = gloss.strip().upper()
        if gloss in supported_glosses:
            return gloss, []
        supported = []
        unsupported = []
        for token in gloss.split():
            token = token.strip().upper()
            if not token:
                continue
            if token in supported_glosses:
                supported.append(token)
            else:
                unsupported.append(token)
        return " ".join(supported), unsupported

    def _resolve_pose_source(self, pose_source, gloss, model_glosses, clip_glosses):
        if pose_source not in {"auto", "model", "clips"}:
            raise ValueError("pose_source must be one of: auto, model, clips")
        model_ready = self._sequence_supported(gloss, model_glosses)
        clips_ready = self._sequence_supported(gloss, clip_glosses)
        if pose_source == "clips":
            if not clips_ready:
                raise FileNotFoundError(f"No keypoint clips found for gloss '{gloss}'.")
            return "clips"
        if pose_source == "model":
            if not self.pose_checkpoint_path.exists():
                raise FileNotFoundError("No pose checkpoint found for --pose-source model.")
            if not model_ready:
                raise UnsupportedGlossError(gloss, gloss, model_glosses)
            return "model"
        if clips_ready and self._has_rendered_source_clips(gloss):
            return "clips"
        if model_ready:
            return "model"
        if clips_ready:
            return "clips"
        raise UnsupportedGlossError(gloss, gloss, model_glosses | clip_glosses)

    def _text_model_available(self):
        model_dir = self._resolve_text_model_dir()
        return model_dir.exists() and (model_dir / "config.json").exists()

    def _sequence_supported(self, gloss, glosses):
        if gloss in glosses:
            return True
        tokens = [token for token in gloss.split() if token]
        return bool(tokens) and all(token in glosses for token in tokens)

    def _resolve_text_model_dir(self):
        if (self.text_model_dir / "config.json").exists():
            return self.text_model_dir
        checkpoints = sorted(
            (
                path for path in self.text_model_dir.glob("checkpoint-*")
                if path.is_dir() and (path / "config.json").exists()
            ),
            key=_checkpoint_step,
            reverse=True,
        )
        if checkpoints:
            return checkpoints[0]
        return self.text_model_dir

    def _clip_records_for_gloss(self, gloss):
        clip_index = self._load_clip_index()
        keys = [gloss] if gloss in clip_index else [token for token in gloss.split() if token]
        records = []
        for key in keys:
            paths = clip_index.get(key, [])
            if not paths:
                return []
            with open(paths[0], "r", encoding="utf-8") as file:
                record = json.load(file)
            record["_json_path"] = str(paths[0])
            source_video = record.get("source_video")
            if source_video:
                record["source_video"] = Path(source_video)
            records.append(record)
        return records

    def _has_rendered_source_clips(self, gloss):
        records = self._clip_records_for_gloss(gloss)
        return bool(records) and all(_is_rendered_source_record(record) for record in records)


def _normalize_phrase(text):
    return " ".join(re.findall(r"[A-Za-z0-9]+", text.upper()))


def _text_tokens(text):
    return re.findall(r"[A-Za-z0-9]+", text.upper())


def _match_supported_glosses_in_order(text_tokens, supported_glosses):
    candidates = []
    for gloss in supported_glosses:
        gloss_words = _text_tokens(gloss)
        if gloss_words:
            candidates.append((gloss, gloss_words))
    candidates.sort(key=lambda item: (-len(item[1]), -len(item[0]), item[0]))

    matched = []
    matched_indices = set()
    index = 0
    while index < len(text_tokens):
        best = None
        for gloss, gloss_words in candidates:
            end = index + len(gloss_words)
            if text_tokens[index:end] == gloss_words:
                best = (gloss, gloss_words)
                break
        if best is None:
            index += 1
            continue
        gloss, gloss_words = best
        matched.append(gloss)
        matched_indices.update(range(index, index + len(gloss_words)))
        index += len(gloss_words)
    return matched, matched_indices


def _segment_text_for_universal_output(text, supported_glosses):
    text_tokens = _text_tokens(text)
    candidates = []
    for gloss in supported_glosses:
        gloss_words = _text_tokens(gloss)
        if gloss_words:
            candidates.append((gloss, gloss_words))
    candidates.sort(key=lambda item: (-len(item[1]), -len(item[0]), item[0]))

    segments = []
    index = 0
    while index < len(text_tokens):
        best = None
        for gloss, gloss_words in candidates:
            end = index + len(gloss_words)
            if text_tokens[index:end] == gloss_words:
                best = (gloss, gloss_words)
                break
        if best is None:
            token = text_tokens[index]
            segments.append({"kind": "fingerspell", "text": token, "gloss": f"FS:{token}"})
            index += 1
            continue
        gloss, gloss_words = best
        segments.append({"kind": "trained", "text": " ".join(gloss_words), "gloss": gloss})
        index += len(gloss_words)
    return segments


def _write_fingerspell_segment(writer, text, fps=20, canvas_size=512, label=None):
    frames = make_fingerspell_frames(text, fps=fps)
    return _write_keypoint_frames(writer, frames, canvas_size=canvas_size, label=label)


def _write_keypoint_frames(writer, frames, canvas_size=512, label=None):
    count = 0
    for frame in np.asarray(frames, dtype=np.float32):
        image = draw_skeleton_frame(frame, canvas_size=canvas_size)
        if label:
            _draw_label(image, label)
        writer.write(image)
        count += 1
    return count


def _write_blank_frames(writer, canvas_size, count):
    blank = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    for _ in range(max(0, count)):
        writer.write(blank)
    return max(0, count)


def _draw_label(image, label):
    import cv2

    cv2.putText(
        image,
        str(label)[:24],
        (18, image.shape[0] - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (210, 210, 255),
        2,
        cv2.LINE_AA,
    )


def _is_rendered_source_record(record):
    source_video = record.get("source_video")
    return (
        str(record.get("extractor", "")).lower() == "rendered"
        and source_video
        and Path(source_video).exists()
    )


def _fit_frame_to_canvas(frame, canvas_size):
    import cv2

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


def _checkpoint_step(path):
    match = re.search(r"checkpoint-(\d+)$", Path(path).name)
    return int(match.group(1)) if match else -1
