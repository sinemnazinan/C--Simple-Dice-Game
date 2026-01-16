import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from google import genai

PROMPT_VERSION = "5.8"
DEFAULT_MODEL = "gemini-2.5-flash"


PROMPT_GEN = """\
Sen bir e-ticaret ürün analisti yapay zekâsın. TÜRKÇE cevap ver.

Aşağıdaki input, 4_build_product_summary.py çıktısıdır (özet paket).
SADECE aşağıdaki şemaya uygun, SADECE JSON üret. Başka hiçbir şey yazma.

ZORUNLU KURALLAR:
- overall_summary: 5-7 cümle
- pros: tam 5 madde
- cons: tam 5 madde
- Pros cluster_id sadece şu listeden: {pros_cluster_ids}
- Cons cluster_id sadece şu listeden: {cons_cluster_ids}
- Her pros/cons maddesinde en az {min_evidence} adet evidence_review_ids olmalı.
- evidence_review_ids sadece EVIDENCE_POOL içinden seçilmeli ve ilgili cluster_id’nin havuzundan gelmeli.
- numbers alanını inputtaki değerlerle AYNEN doldur: total_reviews, overall_avg_rating, rating_counts.

ŞEMA:
{{
  "overall_summary": "5-7 cümle",
  "pros": [
    {{
      "title": "kısa başlık",
      "detail": "1 cümle açıklama",
      "cluster_id": 0,
      "evidence_review_ids": ["..."]
    }}
  ],
  "cons": [
    {{
      "title": "kısa başlık",
      "detail": "1 cümle açıklama",
      "cluster_id": 0,
      "evidence_review_ids": ["..."]
    }}
  ],
  "advice": "1 paragraf",
  "numbers": {{
    "total_reviews": 0,
    "overall_avg_rating": 0.0,
    "rating_counts": {{"1":0,"2":0,"3":0,"4":0,"5":0}}
  }}
}}

EVIDENCE_POOL (cluster_id -> review_id listesi):
{evidence_pool_json}

INPUT SUMMARY_JSON:
{summary_json}
"""


PROMPT_CONVERT = """\
Aşağıdaki metin/yanıt JSON değil. Bunu aşağıdaki şemaya uygun SADECE JSON'a dönüştür.

ZORUNLU KURALLAR:
- SADECE JSON, başka hiçbir şey yazma.
- overall_summary: 5-7 cümle
- pros: tam 5 madde
- cons: tam 5 madde
- Pros cluster_id sadece şu listeden: {pros_cluster_ids}
- Cons cluster_id sadece şu listeden: {cons_cluster_ids}
- Her pros/cons maddesinde en az {min_evidence} adet evidence_review_ids olmalı.
- evidence_review_ids sadece EVIDENCE_POOL içinden seçilmeli ve ilgili cluster_id’nin havuzundan gelmeli.
- numbers alanını inputtaki değerlerle AYNEN doldur: total_reviews, overall_avg_rating, rating_counts.

ŞEMA:
{{
  "overall_summary": "5-7 cümle",
  "pros": [{{"title":"", "detail":"", "cluster_id":0, "evidence_review_ids":["..."]}}],
  "cons": [{{"title":"", "detail":"", "cluster_id":0, "evidence_review_ids":["..."]}}],
  "advice": "",
  "numbers": {{
    "total_reviews": 0,
    "overall_avg_rating": 0.0,
    "rating_counts": {{"1":0,"2":0,"3":0,"4":0,"5":0}}
  }}
}}

EVIDENCE_POOL:
{evidence_pool_json}

METİN:
{text}
"""


PROMPT_REPAIR = """\
Aşağıdaki JSON şemaya uymuyor. SADECE düzeltilmiş JSON üret.

HATALAR:
{errors}

Kurallar aynı:
- overall_summary 5-7 cümle
- pros 5, cons 5
- pros cluster_id sadece {pros_cluster_ids}
- cons cluster_id sadece {cons_cluster_ids}
- her maddede en az {min_evidence} evidence_review_ids
- evidence_review_ids EVIDENCE_POOL içinden ve ilgili cluster’dan
- numbers: inputtaki ile AYNEN aynı olmalı

EVIDENCE_POOL:
{evidence_pool_json}

INPUT NUMBERS (AYNEN):
{numbers_json}

MEVCUT JSON:
{bad_json}
"""


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def extract_json(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t).strip()

    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    first = t.find("{")
    last = t.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError("JSON blok bulunamadı.")
    blob = t[first:last + 1]
    obj = json.loads(blob)
    if not isinstance(obj, dict):
        raise ValueError("JSON dict değil.")
    return obj


def sentence_count(text: str) -> int:
    parts = [s.strip() for s in re.split(r"[.!?]+", (text or "").strip()) if s.strip()]
    return len(parts)


def numbers_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    total_reviews = summary.get("total_reviews")
    overall_avg = summary.get("overall_avg_rating")
    rating_counts = summary.get("overall_rating_counts") or summary.get("rating_counts") or {}

    rc_str = {}
    for k in ["1", "2", "3", "4", "5"]:
        v = rating_counts.get(k, rating_counts.get(int(k), 0))
        try:
            rc_str[k] = int(v)
        except Exception:
            rc_str[k] = 0

    if total_reviews is None:
        total_reviews = sum(rc_str.values())
    try:
        total_reviews = int(total_reviews)
    except Exception:
        total_reviews = sum(rc_str.values())

    try:
        overall_avg = float(overall_avg)
    except Exception:
        overall_avg = 0.0

    return {
        "total_reviews": total_reviews,
        "overall_avg_rating": overall_avg,
        "rating_counts": rc_str,
    }


def build_evidence_pool(summary: Dict[str, Any]) -> Dict[int, List[str]]:
    pool: Dict[int, Set[str]] = {}

    for c in summary.get("clusters") or []:
        cid = c.get("cluster_id")
        if not isinstance(cid, int):
            continue
        s = pool.get(cid, set())
        for key in ["representative_typical", "representative_negative", "representative_positive"]:
            for ex in c.get(key) or []:
                rid = ex.get("review_id")
                if isinstance(rid, str) and rid:
                    s.add(rid)
        pool[cid] = s

    for key in ["top_positive_clusters", "top_negative_clusters"]:
        for c in summary.get(key) or []:
            cid = c.get("cluster_id")
            if not isinstance(cid, int):
                continue
            s = pool.get(cid, set())
            for rid in c.get("evidence_review_ids") or []:
                if isinstance(rid, str) and rid:
                    s.add(rid)
            pool[cid] = s

    out = {cid: sorted(list(ids)) for cid, ids in pool.items()}
    return out


def pick_disjoint_clusters(summary: Dict[str, Any], k_pos: int = 2, k_neg: int = 2) -> Tuple[List[int], List[int]]:
    """
    4'ten gelen sıralı top listeleri kullanır.
    Mümkünse pros ve cons cluster setlerini çakıştırmaz.
    """
    top_pos = [x.get("cluster_id") for x in (summary.get("top_positive_clusters") or []) if isinstance(x.get("cluster_id"), int)]
    top_neg = [x.get("cluster_id") for x in (summary.get("top_negative_clusters") or []) if isinstance(x.get("cluster_id"), int)]

    # fallback: clusters
    if not top_pos:
        top_pos = [c.get("cluster_id") for c in (summary.get("clusters") or []) if isinstance(c.get("cluster_id"), int)]
    if not top_neg:
        top_neg = [c.get("cluster_id") for c in (summary.get("clusters") or []) if isinstance(c.get("cluster_id"), int)]

    pos: List[int] = []
    for cid in top_pos:
        if cid not in pos:
            pos.append(cid)
        if len(pos) >= max(1, k_pos):
            break

    neg: List[int] = []
    for cid in top_neg:
        if cid in neg:
            continue
        # çakışmayı mümkünse engelle
        if cid in pos and len(top_neg) > 1:
            continue
        neg.append(cid)
        if len(neg) >= max(1, k_neg):
            break

    # hâlâ boşsa
    if not neg:
        neg = [top_neg[0]] if top_neg else []

    return pos, neg


def fill_evidence(item: Dict[str, Any], evidence_pool: Dict[int, List[str]], min_evidence: int) -> None:
    cid = item.get("cluster_id")
    if not isinstance(cid, int):
        return
    pool = evidence_pool.get(cid, [])
    cur = item.get("evidence_review_ids")
    if not isinstance(cur, list):
        cur = []
    cur = [x for x in cur if isinstance(x, str) and x]

    # sadece havuzda olanları tut
    pool_set = set(pool)
    cur = [x for x in cur if x in pool_set]

    # tamamla
    if len(cur) < min_evidence:
        for rid in pool:
            if rid not in cur:
                cur.append(rid)
            if len(cur) >= min_evidence:
                break

    item["evidence_review_ids"] = cur


def validate(obj: Dict[str, Any],
             pros_ids: Set[int],
             cons_ids: Set[int],
             evidence_pool: Dict[int, Set[str]],
             min_evidence: int,
             numbers: Dict[str, Any]) -> List[str]:
    errs: List[str] = []

    for k in ["overall_summary", "pros", "cons", "advice", "numbers"]:
        if k not in obj:
            errs.append(f"Eksik alan: {k}")

    if "overall_summary" in obj:
        sc = sentence_count(str(obj.get("overall_summary", "")))
        if sc < 5 or sc > 7:
            errs.append(f"overall_summary 5-7 cümle değil (şu an {sc}).")

    pros = obj.get("pros")
    cons = obj.get("cons")
    if not isinstance(pros, list) or len(pros) != 5:
        errs.append("pros tam 5 değil.")
    if not isinstance(cons, list) or len(cons) != 5:
        errs.append("cons tam 5 değil.")

    def check_side(items: Any, allowed: Set[int], side: str):
        if not isinstance(items, list):
            return
        titles = []
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                errs.append(f"{side}[{i}] dict değil.")
                continue
            for kk in ["title", "detail", "cluster_id", "evidence_review_ids"]:
                if kk not in it:
                    errs.append(f"{side}[{i}] eksik: {kk}")
            cid = it.get("cluster_id")
            if not isinstance(cid, int) or cid not in allowed:
                errs.append(f"{side}[{i}] cluster_id allowed değil: {cid}")
                continue

            titles.append(str(it.get("title", "")).strip().lower())

            ev = it.get("evidence_review_ids")
            if not isinstance(ev, list) or len(ev) < min_evidence:
                errs.append(f"{side}[{i}] evidence_review_ids < {min_evidence}.")
                continue

            pool = evidence_pool.get(cid, set())
            if pool:
                for rid in ev:
                    if not isinstance(rid, str) or rid not in pool:
                        errs.append(f"{side}[{i}] evidence id havuzda değil: {rid} (cid={cid})")

        # aynı tarafta başlık tekrarı istemiyoruz
        if len(titles) == len(items):
            if len(set(titles)) != len(titles):
                errs.append(f"{side} içinde tekrar eden title var.")

    check_side(pros, pros_ids, "pros")
    check_side(cons, cons_ids, "cons")

    # numbers birebir aynı
    obj["numbers"] = numbers
    return errs


def call_gemini(client: Any, model: str, prompt: str) -> str:
    resp = client.models.generate_content(model=model, contents=prompt)
    text = getattr(resp, "text", None)
    if not isinstance(text, str) or not text.strip():
        # bazı sürümlerde text boş dönebiliyor; stringe çevirip dene
        text = str(resp)
    text = (text or "").strip()
    if not text:
        raise RuntimeError("Gemini boş çıktı döndürdü.")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Girdi: *_4_summary.json")
    ap.add_argument("--out", dest="out_path", required=True, help="Çıktı: *_5_gemini.json")
    ap.add_argument("--overwrite", action="store_true", help="Çıktı varsa üzerine yaz")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--sleep", type=float, default=2.0)
    ap.add_argument("--min-evidence", type=int, default=2)
    ap.add_argument("--k-pos", type=int, default=2)
    ap.add_argument("--k-neg", type=int, default=2)
    args = ap.parse_args()

    in_file = Path(args.in_path)
    out_file = Path(args.out_path)

    if not in_file.exists():
        raise SystemExit(f"Girdi bulunamadı: {in_file}")

    if out_file.exists() and not args.overwrite:
        raise SystemExit(f"Çıktı zaten var: {out_file}\nFarklı --out ver veya --overwrite kullan.")

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY (veya GOOGLE_API_KEY) ortam değişkeni bulunamadı.")

    client = genai.Client(api_key=api_key)

    summary = load_json(in_file)
    summary_json = compact_json(summary)
    input_hash = sha1_hex(summary_json)

    numbers = numbers_from_summary(summary)
    evidence_pool_list = build_evidence_pool(summary)
    evidence_pool_json = json.dumps(evidence_pool_list, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    evidence_pool_set = {cid: set(ids) for cid, ids in evidence_pool_list.items()}

    pros_ids, cons_ids = pick_disjoint_clusters(summary, k_pos=args.k_pos, k_neg=args.k_neg)
    pros_ids_set = set(pros_ids)
    cons_ids_set = set(cons_ids)

    base_prompt = PROMPT_GEN.format(
        pros_cluster_ids=json.dumps(pros_ids, ensure_ascii=False),
        cons_cluster_ids=json.dumps(cons_ids, ensure_ascii=False),
        min_evidence=int(args.min_evidence),
        evidence_pool_json=evidence_pool_json,
        summary_json=summary_json,
    )

    last_text = ""
    last_err: Optional[str] = None
    structured: Optional[Dict[str, Any]] = None

    for attempt in range(1, int(args.retries) + 1):
        try:
            # 1) üret
            text = call_gemini(client, args.model, base_prompt)
            last_text = text

            # 2) parse dene; olmazsa convert
            try:
                obj = extract_json(text)
            except Exception:
                convert_prompt = PROMPT_CONVERT.format(
                    text=text,
                    pros_cluster_ids=json.dumps(pros_ids, ensure_ascii=False),
                    cons_cluster_ids=json.dumps(cons_ids, ensure_ascii=False),
                    min_evidence=int(args.min_evidence),
                    evidence_pool_json=evidence_pool_json,
                )
                text2 = call_gemini(client, args.model, convert_prompt)
                last_text = text2
                obj = extract_json(text2)

            # 3) numbers kesin doğru olacak
            obj["numbers"] = numbers

            # 4) evidence eksikse kod tamamlar
            if isinstance(obj.get("pros"), list):
                for it in obj["pros"]:
                    if isinstance(it, dict):
                        fill_evidence(it, evidence_pool_list, int(args.min_evidence))
            if isinstance(obj.get("cons"), list):
                for it in obj["cons"]:
                    if isinstance(it, dict):
                        fill_evidence(it, evidence_pool_list, int(args.min_evidence))

            # 5) validate; hata varsa repair iste
            errs = validate(
                obj=obj,
                pros_ids=pros_ids_set,
                cons_ids=cons_ids_set,
                evidence_pool=evidence_pool_set,
                min_evidence=int(args.min_evidence),
                numbers=numbers,
            )

            if errs:
                repair_prompt = PROMPT_REPAIR.format(
                    errors="\n".join(errs),
                    pros_cluster_ids=json.dumps(pros_ids, ensure_ascii=False),
                    cons_cluster_ids=json.dumps(cons_ids, ensure_ascii=False),
                    min_evidence=int(args.min_evidence),
                    evidence_pool_json=evidence_pool_json,
                    numbers_json=json.dumps(numbers, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
                    bad_json=json.dumps(obj, ensure_ascii=False),
                )
                text3 = call_gemini(client, args.model, repair_prompt)
                last_text = text3
                obj2 = extract_json(text3)

                # repair sonrası da numbers ve evidence kesinleştir
                obj2["numbers"] = numbers
                if isinstance(obj2.get("pros"), list):
                    for it in obj2["pros"]:
                        if isinstance(it, dict):
                            fill_evidence(it, evidence_pool_list, int(args.min_evidence))
                if isinstance(obj2.get("cons"), list):
                    for it in obj2["cons"]:
                        if isinstance(it, dict):
                            fill_evidence(it, evidence_pool_list, int(args.min_evidence))

                errs2 = validate(
                    obj=obj2,
                    pros_ids=pros_ids_set,
                    cons_ids=cons_ids_set,
                    evidence_pool=evidence_pool_set,
                    min_evidence=int(args.min_evidence),
                    numbers=numbers,
                )
                if errs2:
                    raise RuntimeError("Repair sonrası hâlâ hata var: " + "; ".join(errs2))

                obj = obj2

            structured = obj

            payload = {
                "prompt_version": PROMPT_VERSION,
                "model": args.model,
                "input_file": in_file.name,
                "input_hash": input_hash,
                "pros_cluster_ids": pros_ids,
                "cons_cluster_ids": cons_ids,
                "min_evidence": int(args.min_evidence),
                "prompt": base_prompt,
                "response_text": last_text,
                "structured": structured,
            }

            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            print("Bitti.")
            print("IN :", in_file)
            print("OUT:", out_file)
            return

        except Exception as e:
            last_err = str(e)
            print(f"[Deneme {attempt}/{args.retries}] Hata: {e}")
            time.sleep(float(args.sleep))

    # başarısız olsa bile --out yaz (tek dosya)
    fallback = {
        "prompt_version": PROMPT_VERSION,
        "model": args.model,
        "input_file": in_file.name,
        "input_hash": input_hash,
        "pros_cluster_ids": pros_ids,
        "cons_cluster_ids": cons_ids,
        "min_evidence": int(args.min_evidence),
        "prompt": base_prompt,
        "response_text": last_text,
        "structured": structured,
        "error": last_err,
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(fallback, ensure_ascii=False, indent=2), encoding="utf-8")
    raise SystemExit(f"Başarısız. Hata OUT içine yazıldı: {out_file}")


if __name__ == "__main__":
    main()
