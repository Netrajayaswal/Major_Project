from pathlib import Path
import importlib.util
import sys


class PretrainedText2SignUnavailable(RuntimeError):
    pass


class HuggingFaceText2SignProvider:
    def __init__(
        self,
        model_dir="outputs/checkpoints/pretrained_text2sign",
        checkpoint_name="checkpoint_epoch_70.pt",
        device=None,
        num_inference_steps=25,
        guidance_scale=7.5,
        fps=8,
        canvas_size=512,
    ):
        self.model_dir = Path(model_dir)
        self.checkpoint_name = checkpoint_name
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.fps = fps
        self.canvas_size = canvas_size
        self._pipeline = None

    def available(self):
        return self._checkpoint_path() is not None

    def generate_video(self, text, output_path):
        pipeline = self._load_pipeline()
        frames = pipeline(
            text,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        )[0]
        return _write_pil_frames_video(frames, output_path, fps=self.fps, canvas_size=self.canvas_size)

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        checkpoint_path = self._checkpoint_path()
        if checkpoint_path is None:
            raise PretrainedText2SignUnavailable(
                f"No pretrained Text2Sign checkpoint found in {self.model_dir}. "
                "Run `python main.py download-pretrained` first, or use trained-only mode."
            )

        import torch

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        repo_dir = checkpoint_path.parent.resolve()
        sys.path.insert(0, str(repo_dir))
        try:
            pipeline_path = repo_dir / "pipeline.py"
            spec = importlib.util.spec_from_file_location("_external_text2sign_pipeline", pipeline_path)
            if spec is None or spec.loader is None:
                raise PretrainedText2SignUnavailable(f"Could not load external pipeline from {pipeline_path}")
            external_pipeline = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(external_pipeline)
            text2sign_pipeline = external_pipeline.Text2SignPipeline
            self._pipeline = text2sign_pipeline.from_pretrained(str(checkpoint_path), device=device)
        finally:
            try:
                sys.path.remove(str(repo_dir))
            except ValueError:
                pass
        return self._pipeline

    def _checkpoint_path(self):
        if self.model_dir.is_file() and self.model_dir.name == self.checkpoint_name:
            return self.model_dir.resolve()
        direct = self.model_dir / self.checkpoint_name
        if direct.exists():
            return direct.resolve()
        matches = sorted(self.model_dir.rglob(self.checkpoint_name)) if self.model_dir.exists() else []
        return matches[0].resolve() if matches else None


def _write_pil_frames_video(frames, output_path, fps=8, canvas_size=512):
    import cv2
    import numpy as np

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

    try:
        for frame in frames:
            rgb = np.asarray(frame.convert("RGB"))
            resized = cv2.resize(rgb, (canvas_size, canvas_size), interpolation=cv2.INTER_CUBIC)
            writer.write(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return output_path
