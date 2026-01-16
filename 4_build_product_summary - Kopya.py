import json
import statistics
from collections import Counter, defaultdict

INPUT_FILE = "trendyol_reviews_with_clusters.json"
OUTPUT_FILE = "product_summary_for_gemini.json"


def safe_rating(item):
    """rating değerini güvenli şekilde float'a çevir (yoksa None)."""
    r = item.get("rating")
    if isinstance(r, (int, float)):
        return float(r)
    try:
        return float(r)
    except (TypeError, ValueError):
        return None


def main():
    # 1) Veriyi oku
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reviews = json.load(f)

    if not isinstance(reviews, list):
        raise ValueError("Beklenen format: liste. JSON yapısı farklı görünüyor.")

    # 2) Genel rating & sentiment dağılımı
    all_ratings = []
    sentiment_counts = Counter()

    clusters = defaultdict(list)

    for r in reviews:
        rating = safe_rating(r)
        if rating is not None:
            all_ratings.append(rating)

        sent = r.get("sentiment_label") or "unknown"
        sentiment_counts[sent] += 1

        cid = r.get("cluster_id")
        if cid is not None:
            try:
                cid = int(cid)
            except (TypeError, ValueError):
                continue
            clusters[cid].append(r)

    overall_avg_rating = statistics.mean(all_ratings) if all_ratings else None

    # 3) Küme bazlı özetler
    cluster_summaries = []

    for cid, items in clusters.items():
        cluster_ratings = [safe_rating(x) for x in items]
        cluster_ratings = [x for x in cluster_ratings if x is not None]

        cluster_sent_counts = Counter(
            (x.get("sentiment_label") or "unknown") for x in items
        )

        # Temsili yorum: en uzun 5 yorumu al (detaylı olanlar)
        comments = [x.get("comment", "") for x in items if x.get("comment")]
        comments_sorted = sorted(comments, key=len, reverse=True)
        sample_comments = comments_sorted[:5]

        cluster_summaries.append(
            {
                "cluster_id": cid,
                "n_reviews": len(items),
                "avg_rating": statistics.mean(cluster_ratings)
                if cluster_ratings
                else None,
                "sentiment_counts": dict(cluster_sent_counts),
                "sample_comments": sample_comments,
            }
        )

    # Küme özetlerini id'ye göre sırala
    cluster_summaries.sort(key=lambda x: x["cluster_id"])

    # 4) Tek bir özet JSON'u oluştur
    summary = {
        "total_reviews": len(reviews),
        "overall_avg_rating": overall_avg_rating,
        "overall_sentiment_counts": dict(sentiment_counts),
        "clusters": cluster_summaries,
    }

    # 5) Kaydet
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Özet dosyası oluşturuldu: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
