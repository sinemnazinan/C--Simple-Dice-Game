import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

INPUT_FILE = "trendyol_reviews_with_sentiment_and_embeds.json"
OUTPUT_FILE = "trendyol_reviews_with_clusters.json"

# 1) Veriyi oku
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Toplam kayıt: {len(data)}")

# 2) Embedding'i olan satırları filtrele
embeds = []
valid_indices = []  # embedding'i olan kayıtların indexleri
for idx, item in enumerate(data):
    emb = item.get("embedding")
    if emb is not None:
        embeds.append(emb)
        valid_indices.append(idx)

if not embeds:
    raise RuntimeError("Hiç embedding bulunamadı. Önce sentiment_analysis.py'yi çalıştırdığından emin ol.")

X = np.array(embeds, dtype=np.float32)
print(f"Embedding boyutu: {X.shape}")  # (N, hidden_size)

# 3) Küme sayısını seç (örnek: 5)
N_CLUSTERS = 5

print(f"{N_CLUSTERS} küme ile KMeans eğitiliyor...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

# 4) Küme id'lerini data'ya geri yaz
for idx, cluster_id in zip(valid_indices, labels):
    data[idx]["cluster_id"] = int(cluster_id)

# 5) Sonucu yeni JSON'a kaydet
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Küme etiketleri eklendi. Kaydedilen dosya: {OUTPUT_FILE}")

# 6) Küme özetlerini yazdır
df = pd.DataFrame(data)

print("\n--- Küme Özeti ---")
for cid in range(N_CLUSTERS):
    cluster_df = df[df["cluster_id"] == cid]
    if cluster_df.empty:
        continue

    size = len(cluster_df)
    avg_rating = cluster_df["rating"].mean() if "rating" in cluster_df else None

    print(f"\nKüme {cid}:")
    print(f"  Yorum sayısı   : {size}")
    if avg_rating is not None:
        print(f"  Ortalama rating: {avg_rating:.2f}")

    # Örnek birkaç yorum
    print("  Örnek yorumlar:")
    for comment in cluster_df["comment"].head(5):
        print("   -", str(comment)[:120])
