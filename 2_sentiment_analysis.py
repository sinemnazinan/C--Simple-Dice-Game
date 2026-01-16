import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Model (AYNI KALIYOR)
MODEL_NAME = "savasy/bert-base-turkish-sentiment-cased"

_norm_re = re.compile(r"[^0-9a-zçğıöşü\s]+", re.IGNORECASE)
_space_re = re.compile(r"\s+")


def load_reviews(path: Path) -> Tuple[Any, List[Dict[str, Any]]]:
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


def normalize_text(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    t = t.casefold()
    t = _norm_re.sub(" ", t)
    t = _space_re.sub(" ", t).strip()
    return t


def sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def parse_rating(rating_val: Any) -> Optional[int]:
    if rating_val is None:
        return None
    try:
        r = int(float(rating_val))
        if 1 <= r <= 5:
            return r
        return None
    except (ValueError, TypeError):
        return None


def rating_to_label(rating_int: Optional[int]) -> Optional[str]:
    if rating_int in (4, 5):
        return "positive"
    if rating_int in (1, 2):
        return "negative"
    return None  # 3 veya bilinmiyor


def mean_pooling(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    last_hidden: [B, T, H]
    attention_mask: [B, T]
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # [B, T, 1]
    summed = (last_hidden * mask).sum(dim=1)                  # [B, H]
    counts = mask.sum(dim=1).clamp(min=1e-9)                  # [B, 1]
    return summed / counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Girdi JSON dosyası")
    ap.add_argument("--out", dest="out_path", required=True, help="Çıktı JSON dosyası")
    ap.add_argument("--overwrite", action="store_true", help="Çıktı varsa üzerine yaz")

    # Ek ayarlar (istersen default ile kullan, istersen parametreyle değiştir)
    ap.add_argument("--batch_size", type=int, default=32, help="Batch boyutu (GPU varsa yükseltebilirsin)")
    ap.add_argument("--max_length", type=int, default=256, help="Tokenizer max_length")
    ap.add_argument(
        "--confidence_keep_model",
        type=float,
        default=0.85,
        help="Rating ile çelişiyorsa, model skoru bu eşikten yüksekse model etiketini koru"
    )
    ap.add_argument("--no_l2norm", action="store_true", help="Embedding L2 normalize etme (varsayılan: normalize eder)")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Model indiriliyor / yükleniyor (ilk sefer biraz sürebilir)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    print("Model hazır.")

    wrapper, reviews = load_reviews(IN_FILE)
    print(f"Toplam yorum sayısı: {len(reviews)}")

    # Duplicate tespiti (ürün içi)
    seen_hashes = set()

    idxs: List[int] = []
    texts: List[str] = []

    for i, item in enumerate(reviews):
        text = (item.get("comment") or "").strip()

        if not text:
            item["review_id"] = None
            item["comment_hash"] = None
            item["is_duplicate"] = None

            item["rating_int"] = parse_rating(item.get("rating"))
            item["rating_label"] = rating_to_label(item["rating_int"])

            item["sentiment_label"] = None
            item["sentiment_model_label"] = None
            item["sentiment_raw_label"] = None
            item["sentiment_score"] = None
            item["sentiment_is_flipped"] = None
            item["sentiment_final_source"] = None

            item["embedding"] = None
            continue

        norm = normalize_text(text)
        h = sha1_hex(norm) if norm else sha1_hex(text)
        is_dup = h in seen_hashes
        if not is_dup:
            seen_hashes.add(h)

        item["review_id"] = sha1_hex(f"{h}:{i}")
        item["comment_hash"] = h
        item["is_duplicate"] = bool(is_dup)

        item["rating_int"] = parse_rating(item.get("rating"))
        item["rating_label"] = rating_to_label(item["rating_int"])

        idxs.append(i)
        texts.append(text)

    if texts:
        bs = max(1, int(args.batch_size))
        max_len = max(32, int(args.max_length))
        do_l2 = not args.no_l2norm
        conf_keep = float(args.confidence_keep_model)

        id2label = getattr(model.config, "id2label", None) or {0: "LABEL_0", 1: "LABEL_1"}

        def to_nice_label(raw: str) -> str:
            if raw == "LABEL_1":
                return "positive"
            if raw == "LABEL_0":
                return "negative"
            return raw

        for start in range(0, len(texts), bs):
            chunk_texts = texts[start:start + bs]
            chunk_idxs = idxs[start:start + bs]

            enc = tokenizer(
                chunk_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_len,
                padding=True
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                outputs = model(**enc, output_hidden_states=True, return_dict=True)

                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                scores, pred_ids = probs.max(dim=-1)

                last_hidden = outputs.hidden_states[-1]
                pooled = mean_pooling(last_hidden, enc["attention_mask"])
                if do_l2:
                    pooled = F.normalize(pooled, p=2, dim=-1)

            scores_cpu = scores.detach().cpu().tolist()
            pred_ids_cpu = pred_ids.detach().cpu().tolist()
            pooled_cpu = pooled.detach().cpu().to(torch.float32).tolist()

            for bi, review_i in enumerate(chunk_idxs):
                item = reviews[review_i]

                raw_label = id2label.get(int(pred_ids_cpu[bi]), f"LABEL_{int(pred_ids_cpu[bi])}")
                nice_label = to_nice_label(str(raw_label))
                score = float(scores_cpu[bi])

                rating_lbl = item.get("rating_label")
                final_label = nice_label
                final_source = "model"
                flipped = False

                # Rating ile çelişiyorsa: model eminse modeli koru, değilse rating override yap
                if rating_lbl and rating_lbl in ("positive", "negative") and rating_lbl != nice_label:
                    if score < conf_keep:
                        final_label = rating_lbl
                        final_source = "rating_override_low_conf"
                        flipped = True
                    else:
                        final_label = nice_label
                        final_source = "model_high_conf"

                item["sentiment_label"] = final_label
                item["sentiment_model_label"] = nice_label
                item["sentiment_raw_label"] = str(raw_label)
                item["sentiment_score"] = score
                item["sentiment_is_flipped"] = bool(flipped)
                item["sentiment_final_source"] = final_source

                item["embedding"] = pooled_cpu[bi]

        # örnek çıktı
        printed = 0
        for i, item in enumerate(reviews):
            if item.get("sentiment_label") is None:
                continue
            print(f"[{i}] {(item.get('comment') or '')[:60]}...")
            print(
                f"    model={item.get('sentiment_model_label')} "
                f"final={item.get('sentiment_label')} "
                f"score={float(item.get('sentiment_score')):.3f} "
                f"rating={item.get('rating')} "
                f"dup={item.get('is_duplicate')} "
                f"emb_dim={len(item.get('embedding') or [])}"
            )
            printed += 1
            if printed >= 3:
                break
    else:
        print("İşlenecek yorum yok.")

    # Formatı bozmadan kaydet
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
