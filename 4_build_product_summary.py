import argparse
import json
import math
import statistics
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

# TF-IDF (ürün bağımsız keyword çıkarımı)
from sklearn.feature_extraction.text import TfidfVectorizer


def load_reviews(path: Path):
    """
    Desteklenen formatlar:
    - Liste: [ {...}, {...} ]
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
    t = (text or "").strip().replace("\r", " ").replace("\n", " ")
    if len(t) <= limit:
        return t
    return t[:limit].rstrip() + "..."


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - (a @ b) / (na * nb))


# küçük ve genel TR stopwords (ürün bağımsız)
TR_STOP = {
    "ve", "veya", "ama", "fakat", "ancak", "çok", "az", "daha", "en", "gibi", "kadar",
    "bir", "bu", "şu", "o", "ben", "sen", "biz", "siz", "onlar", "de", "da", "ki",
    "ile", "için", "mi", "mı", "mu", "mü", "ise", "diye", "her", "hiç", "hep",
    "çok", "şey", "olarak", "ürün", "urun", "kullan", "kullandım", "kullaniyorum",
}


def pack_example(x, max_chars: int) -> dict:
    # eğitime uygun: kanıt + karar kaynağı
    return {
        "review_id": x.get("review_id"),
        "comment_hash": x.get("comment_hash"),
        "is_duplicate": x.get("is_duplicate"),

        "rating": safe_rating(x),
        "rating_int": x.get("rating_int"),
        "rating_label": x.get("rating_label"),

        "sentiment_label": x.get("sentiment_label"),
        "sentiment_model_label": x.get("sentiment_model_label"),
        "sentiment_score": x.get("sentiment_score"),
        "sentiment_is_flipped": x.get("sentiment_is_flipped"),
        "sentiment_final_source": x.get("sentiment_final_source"),

        "comment": snippet(x.get("comment", ""), max_chars),
    }


def should_use_for_examples(x: dict, min_len: int, include_duplicates: bool) -> bool:
    txt = (x.get("comment") or "").strip()
    if len(txt) < min_len:
        return False
    if not include_duplicates and x.get("is_duplicate") is True:
        return False
    return True


def top_keywords_for_clusters(
    cluster_to_items: dict,
    cluster_ids_included: list,
    min_len: int,
    include_duplicates: bool,
    topn: int,
    ngram_max: int,
    max_docs: int,
) -> dict:
    """
    Global TF-IDF fit (included cluster’ların metinleri) -> her cluster için mean TF-IDF -> top terms.
    """
    docs = []
    doc_cluster = []

    for cid in cluster_ids_included:
        items = cluster_to_items.get(cid, [])
        for x in items:
            if not should_use_for_examples(x, min_len=min_len, include_duplicates=include_duplicates):
                continue
            docs.append((x.get("comment") or "").strip())
            doc_cluster.append(cid)
            if len(docs) >= max_docs:
                break
        if len(docs) >= max_docs:
            break

    if len(docs) < 10:
        return {cid: [] for cid in cluster_ids_included}

    # token_pattern Türkçe harfleri kapsar
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=list(TR_STOP),
        token_pattern=r"(?u)\b[0-9a-zçğıöşü]{2,}\b",
        ngram_range=(1, max(1, int(ngram_max))),
        max_features=40000,
    )

    X = vectorizer.fit_transform(docs)  # (M, V)
    terms = np.array(vectorizer.get_feature_names_out())

    cid_to_rows = defaultdict(list)
    for i, cid in enumerate(doc_cluster):
        cid_to_rows[cid].append(i)

    out = {}
    for cid in cluster_ids_included:
        rows = cid_to_rows.get(cid, [])
        if not rows:
            out[cid] = []
            continue
        # mean TF-IDF
        mean_vec = X[rows].mean(axis=0)  # (1, V)
        mean_arr = np.asarray(mean_vec).ravel()
        if mean_arr.size == 0:
            out[cid] = []
            continue

        top_idx = mean_arr.argsort()[::-1][:topn]
        kws = []
        for j in top_idx:
            score = float(mean_arr[j])
            if score <= 0:
                continue
            kws.append({"term": str(terms[j]), "score": score})
        out[cid] = kws

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Girdi JSON (_3_clusters.json)")
    ap.add_argument("--out", dest="out_path", required=True, help="Çıktı JSON (model eğitim paketi / LLM input)")
    ap.add_argument("--overwrite", action="store_true", help="Çıktı varsa üzerine yaz")

    # örnek sayıları
    ap.add_argument("--typical", type=int, default=6, help="Cluster başına tipik örnek yorum sayısı")
    ap.add_argument("--negative", type=int, default=4, help="Cluster başına negatif/düşük puan örnek sayısı")
    ap.add_argument("--positive", type=int, default=4, help="Cluster başına pozitif/yüksek puan örnek sayısı")

    ap.add_argument("--min-len", type=int, default=20, help="Çok kısa yorumları örneğe alma eşiği")
    ap.add_argument("--max-chars", type=int, default=450, help="Örnek yorum başına max karakter")
    ap.add_argument("--max-clusters", type=int, default=12, help="En kalabalık N cluster'ı koy (default: 12)")

    # yeni: keyword çıkarımı
    ap.add_argument("--keywords", type=int, default=12, help="Cluster başına TF-IDF keyword sayısı (default: 12)")
    ap.add_argument("--ngram-max", type=int, default=3, help="TF-IDF ngram max (default: 3)")
    ap.add_argument("--tfidf-max-docs", type=int, default=20000, help="TF-IDF için maksimum yorum (default: 20000)")

    # yeni: top cluster listeleri
    ap.add_argument("--top-pos", type=int, default=5, help="Top pozitif cluster sayısı (default: 5)")
    ap.add_argument("--top-neg", type=int, default=5, help="Top negatif cluster sayısı (default: 5)")
    ap.add_argument("--min-cluster-for-top", type=int, default=15, help="Top listelerine girmek için min cluster boyutu")

    # yeni: örneklerde duplicate filtre (default: duplicate hariç)
    ap.add_argument("--include-duplicates", action="store_true", help="Örneklerde duplicate yorumları da kullan")

    args = ap.parse_args()

    IN_FILE = Path(args.in_path)
    OUT_FILE = Path(args.out_path)

    if not IN_FILE.exists():
        raise SystemExit(f"Girdi bulunamadı: {IN_FILE}")

    if OUT_FILE.exists() and not args.overwrite:
        raise SystemExit(f"Çıktı zaten var: {OUT_FILE}\nFarklı --out ver veya --overwrite kullan.")

    wrapper, reviews = load_reviews(IN_FILE)

    # ---- data quality ----
    embedding_dims = set()
    missing_embedding = 0
    duplicate_count = 0
    flipped_count = 0
    final_source_counts = Counter()

    for r in reviews:
        emb = r.get("embedding")
        if isinstance(emb, list) and len(emb) > 0:
            embedding_dims.add(len(emb))
        else:
            missing_embedding += 1

        if r.get("is_duplicate") is True:
            duplicate_count += 1

        if r.get("sentiment_is_flipped") is True:
            flipped_count += 1

        src = r.get("sentiment_final_source")
        if src:
            final_source_counts[src] += 1

    # ---- overall istatistikler ----
    all_ratings = []
    overall_rating_counts = Counter()
    overall_sentiment_counts = Counter()
    overall_model_sentiment_counts = Counter()

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

        msent = r.get("sentiment_model_label") or "unknown"
        overall_model_sentiment_counts[msent] += 1

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
    included_cluster_ids = [cid for cid, _ in sorted_clusters]

    # ---- Keyword çıkarımı (included cluster’lar için) ----
    kw_map = top_keywords_for_clusters(
        cluster_to_items=clusters,
        cluster_ids_included=included_cluster_ids,
        min_len=int(args.min_len),
        include_duplicates=bool(args.include_duplicates),
        topn=int(args.keywords),
        ngram_max=int(args.ngram_max),
        max_docs=int(args.tfidf_max_docs),
    )

    cluster_summaries = []
    cluster_scores_for_top = []

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
        c_model_sent_counts = Counter((x.get("sentiment_model_label") or "unknown") for x in items)

        c_flip = sum(1 for x in items if x.get("sentiment_is_flipped") is True)
        c_source_counts = Counter(x.get("sentiment_final_source") for x in items if x.get("sentiment_final_source"))

        n_reviews = len(items)
        pos = c_sent_counts.get("positive", 0)
        neg = c_sent_counts.get("negative", 0)
        pos_rate = pos / n_reviews if n_reviews else 0.0
        neg_rate = neg / n_reviews if n_reviews else 0.0

        # --- Tipik örnekler: centroid'e en yakın (çeşitlilik filtresiyle) ---
        emb_items = []
        for x in items:
            if not should_use_for_examples(x, min_len=int(args.min_len), include_duplicates=bool(args.include_duplicates)):
                continue
            txt = (x.get("comment") or "").strip()
            emb = x.get("embedding")
            if isinstance(emb, list) and len(emb) > 0 and len(txt) >= int(args.min_len):
                emb_items.append((np.array(emb, dtype=np.float32), x))

        representative_typical = []
        seen_hashes = set()

        if emb_items:
            centroid = np.mean([e for e, _ in emb_items], axis=0)
            scored = []
            for e, x in emb_items:
                d = cosine_distance(e, centroid)
                scored.append((d, x))
            scored.sort(key=lambda z: z[0])

            for _, x in scored:
                h = x.get("comment_hash") or x.get("review_id") or snippet(x.get("comment", ""), 80)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                representative_typical.append(pack_example(x, int(args.max_chars)))
                if len(representative_typical) >= int(args.typical):
                    break

        # --- Negatif örnekler: düşük rating / negative öncelik (çeşitlilik filtresiyle) ---
        neg_pool = []
        for x in items:
            if not should_use_for_examples(x, min_len=int(args.min_len), include_duplicates=bool(args.include_duplicates)):
                continue
            txt = (x.get("comment") or "").strip()
            rrt = safe_rating(x)
            sent = x.get("sentiment_label") or "unknown"
            neg_pool.append((
                rrt if rrt is not None else 99.0,
                0 if sent == "negative" else 1,
                -len(txt),
                x
            ))
        neg_pool.sort(key=lambda z: (z[0], z[1], z[2]))

        representative_negative = []
        for _, _, _, x in neg_pool:
            h = x.get("comment_hash") or x.get("review_id") or snippet(x.get("comment", ""), 80)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            representative_negative.append(pack_example(x, int(args.max_chars)))
            if len(representative_negative) >= int(args.negative):
                break

        # --- Pozitif örnekler: yüksek rating / positive öncelik (çeşitlilik filtresiyle) ---
        pos_pool = []
        for x in items:
            if not should_use_for_examples(x, min_len=int(args.min_len), include_duplicates=bool(args.include_duplicates)):
                continue
            txt = (x.get("comment") or "").strip()
            rrt = safe_rating(x)
            sent = x.get("sentiment_label") or "unknown"
            pos_pool.append((
                -(rrt if rrt is not None else -1.0),
                0 if sent == "positive" else 1,
                -len(txt),
                x
            ))
        pos_pool.sort(key=lambda z: (z[0], z[1], z[2]))

        representative_positive = []
        for _, _, _, x in pos_pool:
            h = x.get("comment_hash") or x.get("review_id") or snippet(x.get("comment", ""), 80)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            representative_positive.append(pack_example(x, int(args.max_chars)))
            if len(representative_positive) >= int(args.positive):
                break

        cluster_keywords = kw_map.get(cid, [])
        cluster_summaries.append({
            "cluster_id": cid,
            "n_reviews": n_reviews,
            "avg_rating": c_avg,
            "rating_counts": dict(c_rating_counts),

            # final sentiment (sentiment_label) dağılımı
            "sentiment_counts": dict(c_sent_counts),
            "positive_rate": pos_rate,
            "negative_rate": neg_rate,

            # model sentiment dağılımı (diagnostic)
            "model_sentiment_counts": dict(c_model_sent_counts),

            # flip & kaynak (diagnostic + eğitimde kalite)
            "flip_count": int(c_flip),
            "sentiment_final_source_counts": dict(c_source_counts),

            # ürün bağımsız anahtar ifadeler
            "cluster_keywords": cluster_keywords,

            # kanıt örnekleri
            "representative_typical": representative_typical,
            "representative_negative": representative_negative,
            "representative_positive": representative_positive,
        })

        # top-list skorları
        if n_reviews >= int(args.min_cluster_for_top):
            size_w = math.log(1.0 + n_reviews)
            pos_score = pos_rate * size_w
            neg_score = neg_rate * size_w
            cluster_scores_for_top.append((cid, n_reviews, c_avg, pos_rate, neg_rate, pos_score, neg_score))

    # Top cluster listeleri (5_gemini_summarize promptu için doğrudan işe yarar)
    # 5. dosya "En sık 5 olumlu / 5 problem" istiyor. :contentReference[oaicite:3]{index=3}
    top_pos = sorted(cluster_scores_for_top, key=lambda t: (t[5], t[1]), reverse=True)[: int(args.top_pos)]
    top_neg = sorted(cluster_scores_for_top, key=lambda t: (t[6], t[1]), reverse=True)[: int(args.top_neg)]

    # cluster_summaries’den hızlı lookup
    cs_map = {c["cluster_id"]: c for c in cluster_summaries}

    def top_pack(tup):
        cid, n, avg, pr, nr, ps, ns = tup
        c = cs_map.get(cid, {})
        return {
            "cluster_id": cid,
            "n_reviews": n,
            "avg_rating": avg,
            "positive_rate": pr,
            "negative_rate": nr,
            "cluster_keywords": c.get("cluster_keywords", []),
            # kanıt için birkaç review_id
            "evidence_review_ids": [
                x.get("review_id")
                for x in (c.get("representative_typical", [])[:2] + c.get("representative_negative", [])[:2] + c.get("representative_positive", [])[:2])
                if x.get("review_id")
            ][:6],
        }

    top_positive_clusters = [top_pack(t) for t in top_pos]
    top_negative_clusters = [top_pack(t) for t in top_neg]

    # ürün meta varsa taşı (whitelist)
    product_meta = None
    if isinstance(wrapper, dict):
        product_meta = {k: wrapper.get(k) for k in [
            "source", "product_url", "product_title", "scraped_at", "comment_count"
        ] if k in wrapper} or None

    summary = {
        "schema_version": "4.1",
        "product_meta": product_meta,
        "input_file": IN_FILE.name,

        "total_reviews": len(reviews),
        "overall_avg_rating": overall_avg_rating,
        "overall_rating_counts": dict(overall_rating_counts),

        # final sentiment
        "overall_sentiment_counts": dict(overall_sentiment_counts),
        # model sentiment (diagnostic)
        "overall_model_sentiment_counts": dict(overall_model_sentiment_counts),

        # kalite / izlenebilirlik
        "data_quality": {
            "embedding_dims": sorted(list(embedding_dims)),
            "missing_embedding_count": int(missing_embedding),
            "duplicate_count": int(duplicate_count),
            "flipped_count": int(flipped_count),
            "sentiment_final_source_counts": dict(final_source_counts),
            "include_duplicates_in_examples": bool(args.include_duplicates),
        },

        "clusters_included": len(cluster_summaries),
        "clusters": cluster_summaries,

        # 5. adımı besleyen “hazır” listeler
        "top_positive_clusters": top_positive_clusters,
        "top_negative_clusters": top_negative_clusters,

        "unclustered_count": len(no_cluster),
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Bitti.")
    print("IN :", IN_FILE)
    print("OUT:", OUT_FILE)
    print("clusters_included:", len(cluster_summaries))
    print("top_positive_clusters:", len(top_positive_clusters))
    print("top_negative_clusters:", len(top_negative_clusters))


if __name__ == "__main__":
    main()
