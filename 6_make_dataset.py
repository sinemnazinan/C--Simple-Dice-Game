import argparse
import json
from pathlib import Path

def load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def find_pairs(in_dir: Path):
    pairs = []
    for p4 in sorted(in_dir.glob("*_4_summary.json")):
        base = p4.name.replace("_4_summary.json", "")
        cand = sorted(in_dir.glob(f"{base}_5_gemini*.json"))
        if not cand:
            continue
        pairs.append((base, p4, cand[0]))
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="4 ve 5 dosyalarının olduğu klasör")
    ap.add_argument("--out", required=True, help="Çıktı train.jsonl dosyası")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out = Path(args.out)

    if out.exists() and not args.overwrite:
        raise SystemExit(f"Çıktı var: {out} (overwrite kullan)")

    pairs = find_pairs(in_dir)
    if not pairs:
        raise SystemExit("DATA içinde *_4_summary.json ve *_5_gemini*.json eşleşmesi yok.")

    n = 0
    with out.open("w", encoding="utf-8") as f:
        for base, p4, p5 in pairs:
            s4 = load_json(p4)
            s5 = load_json(p5)

            structured = s5.get("structured")
            if not isinstance(structured, dict):
                print(f"[SKIP] {base}: 5 dosyasında structured yok/bozuk")
                continue

            row = {
                "id": base,
                "input": s4,          # 4'ün tamamı
                "output": structured  # 5'in structured'ı
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    print(f"Bitti. Satır sayısı: {n}. Yazıldı: {out}")

if __name__ == "__main__":
    main()
