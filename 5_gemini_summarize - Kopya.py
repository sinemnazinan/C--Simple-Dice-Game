import json
import os
from google import genai

# 1) Client oluştur
# Eğer GEMINI_API_KEY ortam değişkenini ayarladıysan, böyle tek satır yeterli:
client = genai.Client()

# Eğer ortam değişkeni kullanmadıysan, alttaki satırı kullan:
client = genai.Client(api_key="AIzaSyBPz1MxZgz51LN1f5sJphA2iAMVtJPBRXQ")
# Ortam değişkeninden al
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY ortam değişkeni bulunamadı.")

client = genai.Client(api_key=api_key)



# 2) Özet JSON'unu oku
with open("product_summary_for_gemini.json", "r", encoding="utf-8") as f:
    summary = json.load(f)

# 3) Gemini'ye gidecek prompt'u hazırla
prompt = f"""
Sen bir e-ticaret ürün analisti yapay zekâsın.
Lütfen TÜRKÇE cevap ver.

Aşağıda bir ürün için müşteri yorumlarından çıkarılmış özet veriler (JSON) var:

{json.dumps(summary, ensure_ascii=False, indent=2)}

Bu verilere dayanarak şunları üret:

1) Ürünün genel memnuniyetini ve öne çıkan özelliklerini anlatan 5-7 cümlelik bir genel özet.
2) En sık dile getirilen 5 olumlu noktayı madde madde yaz (her madde: kısa başlık + 1 cümle açıklama).
3) En sık dile getirilen 5 problem / şikâyeti madde madde yaz.
4) Yeni bir müşteri için "kimler almalı, nelere dikkat etmeli" tarzında 1 paragraf tavsiye yaz.

Yalnızca metin çıktısı ver.
"""

# 4) Gemini'den cevap al
response = client.models.generate_content(
    model="gemini-2.5-flash",   # istersen "gemini-2.5-pro" vb. ile değiştirebilirsin
    contents=prompt,
)

# 5) Cevabı ekrana yaz
print(response.text)
