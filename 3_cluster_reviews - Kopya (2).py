import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


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
    ap.add_argument("--in", dest="in_path", required=True, help="Girdi JSON (embedding içermeli)")
    ap.add_argument("--out", dest="out_path", required=True, help="Çıktı JSON (cluster_id eklenecek)")
    ap.add_argument("--overwrite", action="store_true", help="Çıktı varsa üzerine yaz")
    ap.add_argument("--clusters", type=int, default=5, help="KMeans küme sayısı (default: 5)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
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

    wrapper, reviews = load_reviews(IN_FILE)
    print(f"Toplam kayıt: {len(reviews)}")

    # Embedding'i olan satırları filtrele
    embeds = []
    valid_indices = []
    for idx, item in enumerate(reviews):
        emb = item.get("embedding")
        # emb None değilse ve listeyse al
        if isinstance(emb, list) and len(emb) > 0:
            embeds.append(emb)
            valid_indices.append(idx)

    if not embeds:
        raise RuntimeError(
            "Hiç embedding bulunamadı. Önce 2_sentiment_analysis.py çıktısını kullanmalısın."
        )

    X = np.array(embeds, dtype=np.float32)
    print(f"Embedding matrisi: {X.shape}")  # (N, hidden_size)

    n_clusters = int(args.clusters)
    print(f"{n_clusters} küme ile KMeans eğitiliyor...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10)
    labels = kmeans.fit_predict(X)

    # Küme id'lerini geri yaz
    for ridx, cluster_id in zip(valid_indices, labels):
        reviews[ridx]["cluster_id"] = int(cluster_id)

    # Formatı bozmadan kaydet
    if isinstance(wrapper, dict) and isinstance(wrapper.get("reviews"), list):
        wrapper["reviews"] = reviews
        to_save = wrapper
    else:
        to_save = reviews

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(to_save, f, ensure_ascii=False, indent=2)

    print(f"Küme etiketleri eklendi. Kaydedilen dosya: {OUT_FILE}")

    # Küme özetleri (konsolda)
    df = pd.DataFrame(reviews)
    print("\n--- Küme Özeti ---")
    for cid in range(n_clusters):
        if "cluster_id" not in df.columns:
            break

        cluster_df = df[df["cluster_id"] == cid]
        if cluster_df.empty:
            continue

        size = len(cluster_df)
        avg_rating = None
        if "rating" in cluster_df.columns:
            # rating sayısal değilse NaN olabilir, pandas zaten atlar
            avg_rating = pd.to_numeric(cluster_df["rating"], errors="coerce").mean()

        print(f"\nKüme {cid}:")
        print(f"  Yorum sayısı   : {size}")
        if avg_rating is not None and not np.isnan(avg_rating):
            print(f"  Ortalama rating: {avg_rating:.2f}")

        print("  Örnek yorumlar:")
        if "comment" in cluster_df.columns:
            for comment in cluster_df["comment"].head(5):
                print("   -", str(comment)[:120])


if __name__ == "__main__":
    main()
