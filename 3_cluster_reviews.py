import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


def compute_default_kmax(n: int, hard_cap: int) -> int:
    # ürün bağımsız, n'e göre k_max sınırı
    # sqrt(n) iyi bir genel heuristik
    k = int(math.sqrt(n))
    k = max(3, k)
    return min(hard_cap, k)


def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> Optional[float]:
    # silhouette için en az 2 cluster ve her cluster en az 2 örnek gibi koşullar pratikte önemli
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return None
    # tüm örnekler tek cluster'a düşmüş gibi durumlar
    if len(uniq) >= len(labels):
        return None
    try:
        return float(silhouette_score(X, labels, metric="euclidean"))
    except Exception:
        return None


def choose_k_auto(
    X: np.ndarray,
    k_min: int,
    k_max: int,
    seed: int,
    n_init: int,
    sample_size: int,
    min_cluster_size: int,
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    k aralığında dener, silhouette'i en iyi olan k'yı seçer.
    Büyük N ise silhouette'ı sample üzerinde hesaplar (hız için).
    """
    N = X.shape[0]
    rng = np.random.RandomState(seed)

    if N > sample_size:
        sample_idx = rng.choice(N, size=sample_size, replace=False)
        X_eval = X[sample_idx]
    else:
        X_eval = X

    results: List[Dict[str, Any]] = []
    best_k = None
    best_score = -1e9

    for k in range(k_min, k_max + 1):
        if k >= X_eval.shape[0]:
            break

        km = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
        labels = km.fit_predict(X_eval)

        # min_cluster_size kontrolü (çok küçük kümeler kaliteyi bozar)
        counts = np.bincount(labels, minlength=k)
        if (counts < min_cluster_size).any():
            sil = None
        else:
            sil = safe_silhouette(X_eval, labels)

        inertia = float(km.inertia_)
        row = {
            "k": int(k),
            "silhouette": None if sil is None else float(sil),
            "inertia": inertia,
            "min_cluster": int(counts.min()) if len(counts) else 0,
        }
        results.append(row)

        # seçim: silhouette varsa en büyüğü
        if sil is not None and sil > best_score:
            best_score = sil
            best_k = k

    # fallback: silhouette yoksa, k_min'i seç (en güvenli)
    if best_k is None:
        best_k = max(k_min, 2)

    return int(best_k), results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Girdi JSON (embedding içermeli)")
    ap.add_argument("--out", dest="out_path", required=True, help="Çıktı JSON (cluster_id eklenecek)")
    ap.add_argument("--overwrite", action="store_true", help="Çıktı varsa üzerine yaz")

    # Eski kullanım bozulmasın: --clusters hâlâ var
    ap.add_argument("--clusters", type=int, default=5, help="Sabit KMeans küme sayısı (default: 5)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    ap.add_argument("--n_init", type=int, default=10, help="KMeans n_init (default: 10)")

    # Yeni: otomatik k seçimi (varsayılan AÇIK)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--auto_k", action="store_true", help="Otomatik k seç (varsayılan: açık)")
    g.add_argument("--no_auto_k", action="store_true", help="Otomatik k seçmeyi kapat (sabit --clusters kullan)")

    ap.add_argument("--k_min", type=int, default=3, help="auto-k min k (default: 3)")
    ap.add_argument("--k_max", type=int, default=12, help="auto-k max k üst sınır (default: 12)")
    ap.add_argument("--sample_size", type=int, default=2000, help="silhouette için max örnek (default: 2000)")
    ap.add_argument("--min_cluster_size", type=int, default=5, help="auto-k için minimum küme boyu (default: 5)")

    # Embedding normalizasyonu
    ap.add_argument("--no_l2norm", action="store_true", help="Embedding L2 normalize etme (varsayılan: eder)")

    # Duplicate’leri clustering sırasında yok saymak istersen (default: hayır)
    ap.add_argument("--drop_duplicates", action="store_true", help="is_duplicate==True olanları clustering'e sokma")

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

    embeds: List[List[float]] = []
    valid_indices: List[int] = []

    # Embedding'i olan satırları filtrele
    for idx, item in enumerate(reviews):
        if args.drop_duplicates and item.get("is_duplicate") is True:
            continue

        emb = item.get("embedding")
        if isinstance(emb, list) and len(emb) > 0:
            embeds.append(emb)
            valid_indices.append(idx)

    if not embeds:
        raise RuntimeError("Hiç embedding bulunamadı. Önce 2_sentiment_analysis.py çıktısını kullanmalısın.")

    X = np.array(embeds, dtype=np.float32)
    print(f"Embedding matrisi: {X.shape}")  # (N, hidden_size)

    # Cosine uyumu için L2 normalize (varsayılan açık)
    if not args.no_l2norm:
        X = l2_normalize(X)

    # auto-k varsayılan: açık
    auto_k_enabled = True
    if args.no_auto_k:
        auto_k_enabled = False
    if args.auto_k:
        auto_k_enabled = True

    N = X.shape[0]
    k_min = max(2, int(args.k_min))
    hard_kmax = int(args.k_max)
    # n'e göre makul üst sınır
    kmax_by_n = compute_default_kmax(N, hard_kmax)
    k_max = max(k_min, kmax_by_n)

    if auto_k_enabled:
        chosen_k, eval_rows = choose_k_auto(
            X=X,
            k_min=k_min,
            k_max=k_max,
            seed=int(args.seed),
            n_init=int(args.n_init),
            sample_size=int(args.sample_size),
            min_cluster_size=int(args.min_cluster_size),
        )

        print("\n--- auto-k değerlendirme ---")
        df_eval = pd.DataFrame(eval_rows)
        if not df_eval.empty:
            print(df_eval.to_string(index=False))
        print(f"\nSeçilen k: {chosen_k}")
        n_clusters = chosen_k
    else:
        n_clusters = int(args.clusters)
        print(f"auto-k kapalı. Sabit k kullanılıyor: {n_clusters}")

    if n_clusters >= N:
        # çok küçük veri setinde k şişerse
        n_clusters = max(2, min(n_clusters, N - 1))
        print(f"Uyarı: k veri sayısından büyük/eşit olamaz. k düzeltildi: {n_clusters}")

    print(f"{n_clusters} küme ile KMeans eğitiliyor...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=int(args.seed), n_init=int(args.n_init))
    labels = kmeans.fit_predict(X)

    # Küme id'lerini geri yaz
    for ridx, cluster_id in zip(valid_indices, labels):
        reviews[ridx]["cluster_id"] = int(cluster_id)

    # embedding yoksa cluster_id yazma (unclustered)
    for item in reviews:
        if not (isinstance(item.get("embedding"), list) and len(item.get("embedding")) > 0):
            item.pop("cluster_id", None)

    # Formatı bozmadan kaydet
    if isinstance(wrapper, dict) and isinstance(wrapper.get("reviews"), list):
        wrapper["reviews"] = reviews
        to_save = wrapper
    else:
        to_save = reviews

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(to_save, f, ensure_ascii=False, indent=2)

    print(f"\nKüme etiketleri eklendi. Kaydedilen dosya: {OUT_FILE}")

    # Konsol özet (ürün içi)
    df = pd.DataFrame(reviews)
    if "cluster_id" in df.columns:
        print("\n--- Küme Özeti ---")
        for cid in range(n_clusters):
            cluster_df = df[df["cluster_id"] == cid]
            if cluster_df.empty:
                continue

            size = len(cluster_df)
            avg_rating = None
            if "rating" in cluster_df.columns:
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
