import argparse
import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModel,
    pipeline
)

# HuggingFace modeli
MODEL_NAME = "savasy/bert-base-turkish-sentiment-cased"


def load_reviews(path: Path):
    """
    Trendyol: [ {...}, {...} ]  (list)
    Hepsiburada: { "reviews": [ {...}, {...} ], ... } (dict)
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data, data  # wrapper, reviews
    if isinstance(data, dict) and isinstance(data.get("reviews"), list):
        return data, data["reviews"]

    raise ValueError(f"Beklenmeyen JSON formatı: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Girdi JSON dosyası")
    ap.add_argument("--out", dest="out_path", required=True, help="Çıktı JSON dosyası")
    ap.add_argument("--overwrite", action="store_true", help="Çıktı varsa üzerine yaz")
    args = ap.parse_args()

    IN_FILE = Path(args.in_path)
    OUT_FILE = Path(args.out_path)

    if not IN_FILE.exists():
        raise SystemExit(f"Girdi bulunamadı: {IN_FILE}")

    if OUT_FILE.exists() and not args.overwrite:
        raise SystemExit(
            f"Çıktı zaten var: {OUT_FILE}\n"
            f"Farklı --out ver veya --overwrite kullan."
        )

    print("Model indiriliyor / yükleniyor (ilk sefer biraz sürebilir)...")
    clf_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    base_model = AutoModel.from_pretrained(MODEL_NAME)  # embedding için
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    sa = pipeline("sentiment-analysis", tokenizer=tokenizer, model=clf_model)
    print("Model hazır.")

    def get_comment_embedding(text: str):
        text = (text or "").strip()
        if not text:
            return None

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=False
        )
        with torch.no_grad():
            outputs = base_model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]  # CLS token
        return cls_emb.squeeze(0).tolist()

    wrapper, reviews = load_reviews(IN_FILE)
    print(f"Toplam yorum sayısı: {len(reviews)}")

    for i, item in enumerate(reviews):
        text = (item.get("comment") or "").strip()

        # Boş yorumsa her şeyi None yap
        if not text:
            item["sentiment_label"] = None
            item["sentiment_model_label"] = None
            item["sentiment_raw_label"] = None
            item["sentiment_score"] = None
            item["embedding"] = None
            continue

        # --- Sentiment ---
        result = sa(text)[0]
        raw_label = result["label"]  # LABEL_0 / LABEL_1
        score = float(result["score"])

        # LABEL_1 = pozitif, LABEL_0 = negatif
        if raw_label == "LABEL_1":
            nice_label = "positive"
        elif raw_label == "LABEL_0":
            nice_label = "negative"
        else:
            nice_label = raw_label

        # Rating'e göre düzeltme katmanı
        rating = item.get("rating")
        try:
            rating_int = int(float(rating)) if rating is not None else None
        except (ValueError, TypeError):
            rating_int = None

        final_label = nice_label

        # 4–5 yıldız olup negatif çıkanları pozitife düzelt
        if rating_int in (4, 5) and nice_label == "negative":
            final_label = "positive"
        # 1–2 yıldız olup pozitif çıkanları negatife düzelt
        elif rating_int in (1, 2) and nice_label == "positive":
            final_label = "negative"

        # --- Embedding ---
        emb = get_comment_embedding(text)

        # Sonuçları yaz
        item["sentiment_label"] = final_label
        item["sentiment_model_label"] = nice_label
        item["sentiment_raw_label"] = raw_label
        item["sentiment_score"] = score
        item["embedding"] = emb

        if i < 3:
            print(f"[{i}] {text[:60]}...")
            print(
                f"    model_label={nice_label}, final_label={final_label}, "
                f"score={score:.3f}, rating={rating}, embedding_dim={len(emb) if emb else 0}"
            )

    # Kaydet (formatı bozmadan)
    if isinstance(wrapper, dict) and isinstance(wrapper.get("reviews"), list):
        wrapper["reviews"] = reviews
        to_save = wrapper
    else:
        to_save = reviews

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(to_save, f, ensure_ascii=False, indent=2)

    print("Bitti.")
    print("IN  :", IN_FILE)
    print("OUT :", OUT_FILE)


if __name__ == "__main__":
    main()
