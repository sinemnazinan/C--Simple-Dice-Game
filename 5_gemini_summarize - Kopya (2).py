import argparse
import json
import os
import re
import time
from pathlib import Path

from google import genai


PROMPT_TEMPLATE = """\
Sen bir e-ticaret ürün analisti yapay zekâsın.
Lütfen TÜRKÇE cevap ver.

Aşağıda bir ürün için müşteri yorumlarından çıkarılmış özet veriler (JSON) var:

{summary_json}

Bu verilere dayanarak şunları üret:

1) Ürünün genel memnuniyetini ve öne çıkan özelliklerini anlatan 5-7 cümlelik bir genel özet.
2) En sık dile getirilen 5 olumlu noktayı madde madde yaz (her madde: kısa başlık + 1 cümle açıklama).
3) En sık dile getirilen 5 problem / şikâyeti madde madde yaz.
4) Yeni bir müşteri için "kimler almalı, nelere dikkat etmeli" tarzında 1 paragraf tavsiye yaz.

Yalnızca metin çıktısı ver.
"""


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def has_midword_ellipsis(text: str) -> bool:
    # herhangi bir "kelime...kelime" kırpmasını yakalar (unicode dahil)
    return bool(re.search(r"\w\.\.\.\w", text, flags=re.UNICODE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Girdi: *_4_summary.json")
    ap.add_argument("--out", dest="out_path", required=True, help="Çıktı: *_5_gemini.json")
    ap.add_argument("--model", default="gemini-2.5-flash", help="Örn: gemini-2.5-flash")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=2.0)
    args = ap.parse_args()

    in_file = Path(args.in_path)
    out_file = Path(args.out_path)

    if not in_file.exists():
        raise SystemExit(f"Girdi bulunamadı: {in_file}")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY ortam değişkeni bulunamadı.")

    client = genai.Client(api_key=api_key)

    summary = load_json(in_file)

    # Token tüketimini biraz azaltmak için compact JSON (ama içerik kaybı yok)
    summary_json = json.dumps(summary, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    prompt = PROMPT_TEMPLATE.format(summary_json=summary_json)

    last_err = None
    for attempt in range(1, args.retries + 1):
        try:
            resp = client.models.generate_content(
                model=args.model,
                contents=prompt,
            )
            text = (resp.text or "").strip()
            if not text:
                raise RuntimeError("Gemini boş çıktı döndürdü.")

            payload = {
                "model": args.model,
                "input_file": in_file.name,
                "text": text,
                "lines": text.splitlines(),
            }

            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            # hızlı kalite kontrol
            trunc = has_midword_ellipsis(text)
            print("Bitti.")
            print("IN :", in_file)
            print("OUT:", out_file)
            print("text_len:", len(text))
            print("midword_ellipsis_truncation:", trunc)
            return

        except Exception as e:
            last_err = e
            print(f"[Deneme {attempt}/{args.retries}] Hata: {e}")
            time.sleep(args.sleep)

    raise SystemExit(f"Başarısız. Son hata: {last_err}")


if __name__ == "__main__":
    main()
