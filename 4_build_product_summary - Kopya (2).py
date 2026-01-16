import argparse
import json
import statistics
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np


def load_reviews(path: Path):
    """
    Desteklenen formatlar:
    - Trendyol eski: [ {...}, {...} ]
    - Wrapper: { "reviews": [ {...}, {...} ], ... }
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data, data  # wrapper, reviews
    if isinstance(data, dict) and isinstance(data.get("reviews"), list):
        return data, data["reviews"]

    raise ValueError(f"Beklenmeyen JSON formatı: {path}")


def safe_rating(item):
    r = item.get("rating")
    if isinstance(r, (int, float)):
        return float(r)
    try:
        return float(r)
    except Exception:
        return None


def rating_bucket(r):
    try:
        ri = int(round(float(r)))
        return ri if 1 <= ri <= 5 else None
    except Exception:
        return None


def snippet(text: str, limit: int):
    t = (text or "").strip().replace("\r", " ")
    if len(t) <= limit:
        return t
    return t[:limit].rstrip() + "..."


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - (a @ b) / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Girdi JSON (_3_clusters.json)")
    ap.add_argument("--out", dest="out_path", required=True, help="Çıktı JSON (Gemini input paketi)")
    ap.add_argument("--overwrite", action="store_true", help="Çıktı varsa üzerine yaz")

    # Gemini tokenlarını şişirmemek için örnek sayıları
    ap.add_argument("--typical", type=int, default=6, help="Cluster başına tipik örnek yorum sayısı")
    ap.add_argument("--negative", type=int, default=4, help="Cluster başına negatif/düşük puan örnek sayısı")
    ap.add_argument("--positive", type=int, default=4, help="Cluster başına pozitif/yüksek puan örnek sayısı")

    ap.add_argument("--min-len", type=int, default=20, help="Çok kısa yorumları örneğe alma eşiği")
    ap.add_argument("--max-chars", type=int, default=450, help="Örnek yorum başına max karakter")
    ap.add_argument("--max-clusters", type=int, default=12, help="En kalabalık N cluster'ı koy (default: 12)")
    args = ap.parse_args()

    IN_FILE = Path(args.in_path)
    OUT_FILE = Path(args.out_path)

    if not IN_FILE.exists():
        raise SystemExit(f"Girdi bulunamadı: {IN_FILE}")

    if OUT_FILE.exists() and not args.overwrite:
        raise SystemExit(
            f"Çıktı zaten var: {OUT_FILE}\nFarklı --out ver veya --overwrite kullan."
        )

    wrapper, reviews = load_reviews(IN_FILE)

    # ---- overall istatistikler ----
    all_ratings = []
    overall_rating_counts = Counter()
    overall_sentiment_counts = Counter()

    # ---- cluster'a ayır ----
    clusters = defaultdict(list)
    no_cluster = []

    for r in reviews:
        rt = safe_rating(r)
        if rt is not None:
            all_ratings.append(rt)
            rb = rating_bucket(rt)
            if rb:
                overall_rating_counts[rb] += 1

        sent = r.get("sentiment_label") or "unknown"
        overall_sentiment_counts[sent] += 1

        cid = r.get("cluster_id")
        if cid is None:
            no_cluster.append(r)
            continue
        try:
            clusters[int(cid)].append(r)
        except Exception:
            no_cluster.append(r)

    overall_avg_rating = statistics.mean(all_ratings) if all_ratings else None

    # Cluster'ları kalabalığa göre sırala, en kalabalık N tanesini koy
    sorted_clusters = sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)
    sorted_clusters = sorted_clusters[: int(args.max_clusters)]

    cluster_summaries = []

    for cid, items in sorted_clusters:
        # cluster stats
        c_ratings = [safe_rating(x) for x in items]
        c_ratings = [x for x in c_ratings if x is not None]
        c_avg = statistics.mean(c_ratings) if c_ratings else None

        c_rating_counts = Counter()
        for x in items:
            rb = rating_bucket(safe_rating(x))
            if rb:
                c_rating_counts[rb] += 1

        c_sent_counts = Counter((x.get("sentiment_label") or "unknown") for x in items)

        # --- Tipik örnekler: centroid'e en yakın ---
        emb_items = []
        for x in items:
            txt = (x.get("comment") or "").strip()
            emb = x.get("embedding")
            if len(txt) < int(args.min_len):
                continue
            if isinstance(emb, list) and len(emb) > 0:
                emb_items.append((np.array(emb, dtype=np.float32), x))

        representative_typical = []
        if emb_items:
            centroid = np.mean([e for e, _ in emb_items], axis=0)
            scored = []
            for e, x in emb_items:
                d = cosine_distance(e, centroid)
                scored.append((d, x))
            scored.sort(key=lambda z: z[0])

            for _, x in scored[: int(args.typical)]:
                representative_typical.append({
                    "rating": safe_rating(x),
                    "sentiment_label": x.get("sentiment_label"),
                    "comment": snippet(x.get("comment", ""), int(args.max_chars)),
                })

        # --- Negatif örnekler: düşük rating / negative öncelik ---
        neg_pool = []
        for x in items:
            txt = (x.get("comment") or "").strip()
            if len(txt) < int(args.min_len):
                continue
            rrt = safe_rating(x)
            sent = x.get("sentiment_label") or "unknown"
            # sıralama: düşük rating öne, negative öne, daha uzun öne
            neg_pool.append((
                rrt if rrt is not None else 99.0,
                0 if sent == "negative" else 1,
                -len(txt),
                x
            ))
        neg_pool.sort(key=lambda z: (z[0], z[1], z[2]))

        representative_negative = []
        for _, _, _, x in neg_pool[: int(args.negative)]:
            representative_negative.append({
                "rating": safe_rating(x),
                "sentiment_label": x.get("sentiment_label"),
                "comment": snippet(x.get("comment", ""), int(args.max_chars)),
            })

        # --- Pozitif örnekler: yüksek rating / positive öncelik ---
        pos_pool = []
        for x in items:
            txt = (x.get("comment") or "").strip()
            if len(txt) < int(args.min_len):
                continue
            rrt = safe_rating(x)
            sent = x.get("sentiment_label") or "unknown"
            # sıralama: yüksek rating öne, positive öne, daha uzun öne
            pos_pool.append((
                -(rrt if rrt is not None else -1.0),
                0 if sent == "positive" else 1,
                -len(txt),
                x
            ))
        pos_pool.sort(key=lambda z: (z[0], z[1], z[2]))

        representative_positive = []
        for _, _, _, x in pos_pool[: int(args.positive)]:
            representative_positive.append({
                "rating": safe_rating(x),
                "sentiment_label": x.get("sentiment_label"),
                "comment": snippet(x.get("comment", ""), int(args.max_chars)),
            })

        cluster_summaries.append({
            "cluster_id": cid,
            "n_reviews": len(items),
            "avg_rating": c_avg,
            "rating_counts": dict(c_rating_counts),
            "sentiment_counts": dict(c_sent_counts),

            # Gemini'nin kanıt olarak kullanacağı örnekler:
            "representative_typical": representative_typical,
            "representative_negative": representative_negative,
            "representative_positive": representative_positive,
        })

    # ürün meta varsa taşı
    product_meta = None
    if isinstance(wrapper, dict):
        product_meta = {k: wrapper.get(k) for k in [
            "source", "product_url", "product_title", "scraped_at", "comment_count"
        ] if k in wrapper} or None

    summary = {
        "product_meta": product_meta,
        "input_file": IN_FILE.name,

        "total_reviews": len(reviews),
        "overall_avg_rating": overall_avg_rating,
        "overall_rating_counts": dict(overall_rating_counts),
        "overall_sentiment_counts": dict(overall_sentiment_counts),

        "clusters_included": len(cluster_summaries),
        "clusters": cluster_summaries,

        "unclustered_count": len(no_cluster),
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Bitti.")
    print("IN :", IN_FILE)
    print("OUT:", OUT_FILE)


if __name__ == "__main__":
    main()
