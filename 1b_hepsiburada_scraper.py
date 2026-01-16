import time
import json
import random
import re
import argparse

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# -----------------------------
# Hepsiburada tarafı filtreler
# -----------------------------

HB_MONTH_NAMES = [
    "ocak", "şubat", "subat", "mart", "nisan", "mayıs", "mayis",
    "haziran", "temmuz", "ağustos", "agustos", "eylül", "eylul",
    "ekim", "kasım", "kasim", "aralık", "aralik"
]

HB_DAY_ABBR = [
    "pts", "pzt", "sal", "çar", "car", "per", "cum", "cts", "cmt", "paz"
]

HB_NOISE_EXACT = {
    "bu değerlendirme faydalı mı?",
    "bu degerlendirme faydali mi?",
    "teşekkür ederiz.",
    "tesekkur ederiz.",
    "kullanıcı bu ürünü",
    "kullanici bu urunu",
    "satıcısından aldı.",
    "saticisindan aldi.",
}


def is_hepsiburada_noise(text: str) -> bool:
    """
    Hepsiburada yorum kartı içindeki:
      - tarih satırları (09 Ekim, Per)
      - isimler (F**** E****, Beste esma H****, a**** b**** ...)
      - ölçüler (200 x 220 cm)
      - meta yazılar (Bu değerlendirme faydalı mı?, Teşekkür Ederiz. vb.)
    gibi satırları elemek için.
    Yorum DEĞİLSE True döner.
    """
    if not text:
        return True

    t = " ".join(text.split())
    lower = t.lower()

    # Çok kısa olanlar zaten işimize yaramıyor
    if len(t) < 3:
        return True

    # Tam eşleşen meta cümleler
    if lower in HB_NOISE_EXACT:
        return True

    # --- 1) Tarih satırları: "09 Ekim, Per" vb. ---
    if any(m in lower for m in HB_MONTH_NAMES) and re.search(r"\d", lower):
        # Ay adı + gün kısaltması, kısa ise büyük ihtimalle sadece tarih
        if re.search(r"\d{1,2}\s+[a-zçğıöşü]+,\s*\w+", lower) and len(t) <= 40:
            return True
        if any(d in lower for d in HB_DAY_ABBR) and len(t) <= 40:
            return True

    # --- 2) Ölçü satırları: "200 x 220 cm" vb. ---
    if "cm" in lower and "x" in lower and re.search(r"\d", lower) and len(t) <= 30:
        return True

    # --- 3) Yıldızlı isim satırları: "m**** c****", "Beste esma H****" vb. ---
    if "*" in t and len(t) <= 40 and not re.search(r"\d", t):
        return True

    # --- 4) Kısa ama muhtemelen marka satırı olan ifadeler ---
    # Örn: "BELLA MAISON", "XYZ HOME" vb.
    words = t.split()
    if 1 < len(words) <= 3 and len(t) <= 20:
        # Sadece harf ve boşluktan oluşuyorsa
        if re.fullmatch(r"[a-zçğıöşüA-ZÇĞİÖŞÜ\s]+", t):
            # Tipik kısa yorum kelimeleri varsa BIRAK (yorumdur)
            short_review_keywords = [
                "güzel", "guzel", "harika", "müthiş", "muthis",
                "kaliteli", "iyi", "kötü", "kotu", "berbat",
                "fena", "değil", "degil", "ürün", "urun"
            ]
            if not any(w.lower() in short_review_keywords for w in words):
                return True

    # Buraya kadar hiçbir kalıba uymadıysa, büyük ihtimalle gerçek yorum cümlesidir
    return False


# -----------------------------
# Selenium driver kurulumu
# -----------------------------

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Arka planda çalışsın
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )

    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
    except Exception as e:
        print(f"Hata: WebDriver başlatılamadı. Hata detayı: {e}")
        return None

    return driver


# -----------------------------
# Yorum kartından yıldız puanı çıkarma
# -----------------------------

def extract_star_rating(card):
    """
    Bir yorum kartının içinden 1-5 arası yıldız puanını çekmeye çalışır.
    Bulamazsa None döner.
    """
    # 1) aria-label="4 Yıldız"
    star_el = card.select_one('[aria-label*="Yıldız"], [aria-label*="yıldız"]')
    if star_el:
        label = star_el.get("aria-label") or ""
        m = re.search(r"(\d+)", label)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass

    # 2) title="4 Yıldız"
    star_el = card.select_one('[title*="Yıldız"], [title*="yıldız"]')
    if star_el:
        title = star_el.get("title") or ""
        m = re.search(r"(\d+)", title)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass

    # 3) data-rating="4" gibi bir attribute varsa
    star_el = card.select_one("[data-rating]")
    if star_el:
        val = star_el.get("data-rating")
        if val:
            try:
                return int(round(float(val)))
            except ValueError:
                pass

    # 4) Son çare: kart içinde "5 Yıldız" yazan düz metin
    text = card.get_text(" ", strip=True)
    m = re.search(r"(\d+)\s*yıldız", text, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass

    # 5) class adında "star" geçen elementlere göre dolu yıldız say
    star_elems = card.select('[class*="star"], [class*="Star"]')
    if star_elems:
        full_tokens = ("full", "filled", "on", "active", "selected", "solid")
        empty_tokens = ("empty", "off", "outline", "inactive")
        full = 0
        total = 0

        for el in star_elems:
            cls_attr = el.get("class", [])
            if isinstance(cls_attr, list):
                cls = " ".join(cls_attr).lower()
            else:
                cls = str(cls_attr).lower()

            if not cls:
                continue

            if any(tok in cls for tok in full_tokens):
                full += 1
                total += 1
            elif any(tok in cls for tok in empty_tokens):
                total += 1

        # Makul bir sonuçsa döndür (1–5 arası dolu yıldız)
        if full and 1 <= full <= 5:
            return full

    return None


# -----------------------------
# Tek sayfadaki yorumları çek
# -----------------------------

def scrape_current_page(driver, reviews_list, max_reviews):
    """
    Mevcut sayfadaki yorumları çeker ve verilen listeye ekler.
    Her yorum kartı için:
      - hermes-ReviewCard div'ini bul
      - içindeki p/span'lardan en mantıklı metni seç
      - isim / tarih / ölçü / meta ise kartı at
      - yıldız puanını da (varsa) ekle
    """
    # Lazy-load varsa diye sayfanın altına kaydır
    try:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(0.5)
    except Exception:
        pass

    soup = BeautifulSoup(driver.page_source, "html.parser")

    # Her yorum kartını bul
    cards = soup.select('div[class^="hermes-ReviewCard-module-"]')
    if not cards:
        print("Bilgi: Bu sayfada yorum kartı bulunamadı.")
        return

    new_comments_found = 0

    for card in cards:
        if len(reviews_list) >= max_reviews:
            break

        candidates = []

        # 1) Eski mantık: kart içindeki class'sız span (varsa)
        span_no_class = card.select_one('span:not([class])')
        if span_no_class:
            txt = span_no_class.get_text(" ", strip=True)
            if txt:
                candidates.append(txt)

        # 2) Kart içindeki tüm p / span'lerden aday metinler topla
        for el in card.select("p, span"):
            # Öne çıkan özellikler kutusundaysa SKIP
            if el.find_parent("div", class_=re.compile("^hermes-KeyFeatureBox-module-")):
                continue
            if el.find_parent("div", class_=re.compile("^hermes-key-feature-")):
                continue

            txt = el.get_text(" ", strip=True)
            if not txt:
                continue
            txt_norm = " ".join(txt.split())
            if len(txt_norm) < 8:
                continue  # tek kelimelik "Güzel" vb. çok zayıf

            low = txt_norm.lower()

            # "öne çıkan özellikler" başlığı ve benzeri metalar
            if "öne çıkan özellikler" in low or "one cikan ozellikler" in low:
                continue

            # Bariz meta yazıları at
            if any(pat in low for pat in [
                "bu değerlendirme faydalı mı",
                "bu degerlendirme faydali mi",
                "teşekkür ederiz",
                "tesekkur ederiz",
                "kullanıcı bu ürünü",
                "kullanici bu urunu",
            ]):
                continue

            candidates.append(txt_norm)

        if not candidates:
            continue

        # En uzun aday metni yorum kabul et
        comment_text = max(candidates, key=len)
        comment_text = " ".join(comment_text.split())

        # Tarih / isim / ölçü / meta ise kartı tamamen atla
        if is_hepsiburada_noise(comment_text):
            continue

        # Aynı yorum daha önce eklendiyse tekrar ekleme
        if any(r["comment"] == comment_text for r in reviews_list):
            continue

        # Yıldız puanını çek
        star = extract_star_rating(card)

        review_obj = {"comment": comment_text}
        if star is not None:
            review_obj["rating"] = star

        reviews_list.append(review_obj)
        new_comments_found += 1

    print(f"Bilgi: Bu sayfadan {new_comments_found} yeni yorum eklendi. Toplam: {len(reviews_list)}")


# -----------------------------
# Hepsiburada yorum çekme
# -----------------------------

def fetch_reviews_hepsiburada(url, max_reviews=9999):
    reviews = []
    print(f"Hepsiburada için yorum çekme işlemi başlatıldı: {url}")

    driver = None
    try:
        driver = setup_driver()
        if not driver:
            return []

        base_product_url = url.split('?')[0]
        print(f"Bilgi: Ana ürün sayfasına gidiliyor: {base_product_url}")
        driver.get(base_product_url)

        # Çerez pop-up
        try:
            WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
            ).click()
            print("Bilgi: Çerez pop-up'ı kapatıldı.")
            time.sleep(random.uniform(1.0, 2.0))
        except Exception:
            print("Bilgi: Çerez pop-up'ı bulunamadı veya zaten kapalı.")

        # "Değerlendirmeler" sekmesine geç
        try:
            print("Bilgi: 'Değerlendirmeler' sekmesini bulmak için bekleniyor...")
            selector = (By.XPATH, "//a[contains(@href, '-yorumlari')]")
            reviews_tab = WebDriverWait(driver, 15).until(
                EC.element_to_be_clickable(selector)
            )
            print("Bilgi: 'Değerlendirmeler' sekmesi bulundu. Tıklanıyor...")
            driver.execute_script("arguments[0].click();", reviews_tab)
            print("Bilgi: 'Değerlendirmeler' sekmesine başarıyla tıklandı.")
            time.sleep(3)  # Yorumların ilk sayfasının yüklenmesi için bekle

        except Exception as e:
            print(f"\nKRİTİK HATA: 'Değerlendirmeler' sekmesi bulunamadı veya tıklanamadı. Hata: {e}")
            if driver:
                driver.quit()
            return []

        # --- SAYFALANDIRMA (PAGINATION) DÖNGÜSÜ ---
        print("\n--- Sayfalandırma Döngüsü Başlatılıyor ---")

        # 1. İlk sayfayı çek
        print("Bilgi: 1. sayfa taranıyor...")
        scrape_current_page(driver, reviews, max_reviews)

        page_to_click = 2
        while len(reviews) < max_reviews:
            try:
                # Sonraki sayfanın butonunu bul
                print(f"Bilgi: {page_to_click}. sayfa butonu aranıyor...")

                # XPath: İçindeki span'in metni bir sonraki sayfa numarası olan `li` elementini bulur.
                page_button_xpath = f"//li[.//span[text()='{page_to_click}']]"

                page_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, page_button_xpath))
                )

                # Butona tıkla
                driver.execute_script("arguments[0].click();", page_button)
                print(f"Bilgi: {page_to_click}. sayfaya başarıyla geçildi.")

                # Yeni sayfanın yüklenmesini bekle
                time.sleep(random.uniform(0.5, 1.5))

                # Yeni sayfadaki yorumları çek
                print(f"Bilgi: {page_to_click}. sayfa taranıyor...")
                scrape_current_page(driver, reviews, max_reviews)

                # Bir sonraki sayfa için sayacı artır
                page_to_click += 1

            except (TimeoutException, NoSuchElementException):
                print(f"Bilgi: {page_to_click}. sayfa butonu bulunamadı. Muhtemelen son sayfa.")
                break  # Döngüyü sonlandır

    except Exception as e:
        print(f"Hata: İşlem sırasında beklenmedik bir hata oluştu: {e}")
    finally:
        if driver:
            driver.quit()

    print(f"\nToplam {len(reviews)} adet benzersiz yorum başarıyla çekildi.")
    return reviews


# -----------------------------
# JSON kaydetme & CLI
# -----------------------------

def save_reviews_to_json(url, reviews, output_filename):
    output_data = {
        "source_url": url,
        "scraped_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "comment_count": len(reviews),
        "reviews": reviews
    }

    final_json = json.dumps(output_data, ensure_ascii=False, indent=4)

    print("\n--- SONUÇ (JSON ÖZETİ) ---")
    print(final_json)

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(final_json)

    print(f"\nSonuçlar '{output_filename}' dosyasına kaydedildi.")


def main():
    parser = argparse.ArgumentParser(
        description="Sadece Hepsiburada ürün sayfalarından yorum çeker ve JSON'a kaydeder."
    )
    parser.add_argument(
        "--url",
        required=True,
        help="Hepsiburada ürün URL'si"
    )
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=9999,
        help="Çekilecek maksimum yorum sayısı (varsayılan: 9999)"
    )
    parser.add_argument(
        "--output",
        default="hepsiburada_yorumlar.json",
        help="Çıktı JSON dosya adı (varsayılan: hepsiburada_yorumlar.json)"
    )

    args = parser.parse_args()

    print(f"İşlem başlatıldı: {args.url}")
    reviews = fetch_reviews_hepsiburada(args.url, max_reviews=args.max_reviews)

    if not reviews:
        print("İşlem tamamlandı ancak hiç yorum çekilemedi.")
        return

    save_reviews_to_json(args.url, reviews, args.output)


if __name__ == "__main__":
    main()
