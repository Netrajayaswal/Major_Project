from pathlib import Path
import argparse
import csv
import os
import sys
import warnings

import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signlang.metrics import exact_match, simple_bleu


class TextGlossDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_input_length=128, max_target_length=64):
        self.rows = read_rows(csv_path)
        if not self.rows:
            raise ValueError(f"No sentence/gloss_sequence rows found in {csv_path}")
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        source = self.tokenizer(
            row["sentence"],
            truncation=True,
            max_length=self.max_input_length,
        )
        target = self.tokenizer(
            text_target=row["gloss_sequence"],
            truncation=True,
            max_length=self.max_target_length,
        )
        source["labels"] = target["input_ids"]
        return source


def main():
    parser = argparse.ArgumentParser(description="Fine-tune google/mt5-small for text-to-gloss.")
    parser.add_argument("--model-name", default="google/mt5-small")
    parser.add_argument("--train-csv", default="data/text_gloss/train.csv")
    parser.add_argument("--valid-csv", default="data/text_gloss/valid.csv")
    parser.add_argument("--output-dir", default="outputs/checkpoints/text2gloss")
    parser.add_argument("--log-dir", default="outputs/logs/text2gloss")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-input-length", type=int, default=128)
    parser.add_argument("--max-target-length", type=int, default=64)
    parser.add_argument("--allow-peft", action="store_true", help="Allow transformers to import PEFT integrations.")
    args = parser.parse_args()

    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    warnings.filterwarnings(
        "ignore",
        message="urllib3 .* doesn't match a supported version",
    )

    if not args.allow_peft:
        _disable_peft_integration()

    try:
        from transformers import (
            DataCollatorForSeq2Seq,
            MT5ForConditionalGeneration,
            MT5Tokenizer,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit("transformers and sentencepiece are required. Install requirements.txt first.") from exc

    tokenizer = MT5Tokenizer.from_pretrained(args.model_name)
    model = MT5ForConditionalGeneration.from_pretrained(args.model_name)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    train_dataset = TextGlossDataset(
        args.train_csv,
        tokenizer,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
    )
    valid_dataset = TextGlossDataset(
        args.valid_csv,
        tokenizer,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.log_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        fp16=torch.cuda.is_available(),
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda prediction: compute_metrics(prediction, tokenizer),
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics_path = Path("outputs/logs") / "text2gloss_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=sorted(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    print(f"saved model to {args.output_dir}")
    print(f"saved metrics to {metrics_path}")


def _disable_peft_integration():
    try:
        import transformers.utils.import_utils as import_utils
    except Exception:
        return
    if getattr(import_utils, "is_peft_available", None) and import_utils.is_peft_available():
        import_utils._peft_available = False


def read_rows(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            sentence = (row.get("sentence") or "").strip()
            gloss = (row.get("gloss_sequence") or row.get("gloss") or "").strip().upper()
            if sentence and gloss:
                rows.append({"sentence": sentence, "gloss_sequence": gloss})
    return rows


def compute_metrics(eval_prediction, tokenizer):
    predictions, labels = eval_prediction
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    labels = labels.copy()
    labels[labels == -100] = tokenizer.pad_token_id
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_predictions = [prediction.strip().upper() for prediction in decoded_predictions]
    decoded_labels = [label.strip().upper() for label in decoded_labels]
    return {
        "bleu": simple_bleu(decoded_predictions, decoded_labels),
        "exact_match": exact_match(decoded_predictions, decoded_labels),
    }


if __name__ == "__main__":
    main()
