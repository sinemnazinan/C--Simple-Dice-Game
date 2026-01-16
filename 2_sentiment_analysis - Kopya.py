import json
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModel,
    pipeline
)
import torch

# 1) HuggingFace modelini hazırla
MODEL_NAME = "savasy/bert-base-turkish-sentiment-cased"

print("Model indiriliyor / yükleniyor (ilk sefer biraz sürebilir)...")
clf_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
base_model = AutoModel.from_pretrained(MODEL_NAME)  # embedding için
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
sa = pipeline("sentiment-analysis", tokenizer=tokenizer, model=clf_model)
print("Model hazır.")

# 1.1) Embedding fonksiyonu
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
        # CLS token (ilk token) embedding
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [1, hidden_size]
    # JSON'a yazılabilmesi için listeye çevir
    return cls_emb.squeeze(0).tolist()


# 2) collect_comments.py'nin ürettiği JSON'u oku
with open("trendyol_reviews.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Toplam yorum sayısı: {len(data)}")

# 3) Her yorum için duygu analizi + embedding yap
for i, item in enumerate(data):
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
    result = sa(text)[0]   # [{'label': 'LABEL_1', 'score': 0.98}] gibi

    raw_label = result["label"]        # LABEL_0 veya LABEL_1
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
        rating_int = int(rating) if rating is not None else None
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
    item["sentiment_label"] = final_label          # rating ile düzeltilmiş etiket
    item["sentiment_model_label"] = nice_label     # modelin rating'siz tahmini
    item["sentiment_raw_label"] = raw_label        # LABEL_0 / LABEL_1
    item["sentiment_score"] = score
    item["embedding"] = emb

    if i < 3:  # ilk 3 taneyi debug için yazdır
        print(f"[{i}] {text[:60]}...")
        print(
            f"    model_label={nice_label}, final_label={final_label}, "
            f"score={score:.3f}, rating={rating}, embedding_dim={len(emb) if emb else 0}"
        )

# 4) Yeni JSON olarak kaydet
with open("trendyol_reviews_with_sentiment_and_embeds.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# # 5) Excel'e de at (istersen)
# df = pd.DataFrame(data)
# df.to_excel("trendyol_reviews_with_sentiment_and_embeds.xlsx", index=False)

print("Bitti.")
print("JSON  : trendyol_reviews_with_sentiment_and_embeds.json")
print("Excel : trendyol_reviews_with_sentiment_and_embeds.xlsx")
