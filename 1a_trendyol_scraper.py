import sys
import os
import json
import time
import keyboard
import tkinter as tk
from tkinter import simpledialog

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import StaleElementReferenceException


# Konsol kodlamasını kontrol et ve gerekirse UTF-8 olarak ayarla
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def click_all_read_more(driver):
    """
    Sayfadaki tüm 'devamını oku' linklerini tıklar.
    """
    while True:
        buttons = driver.find_elements(By.CSS_SELECTOR, "a.read-more")
        if not buttons:
            break

        clicked_any = False
        for btn in buttons:
            try:
                driver.execute_script(
                    "arguments[0].scrollIntoView({block:'center', inline:'center'});",
                    btn,
                )
                time.sleep(0.1)
                btn.click()
                clicked_any = True
                time.sleep(0.1)
            except Exception:
                continue

        if not clicked_any:
            break


def _parse_px(value):
    """
    '52px' gibi değeri float(52.0) yapar; parse edemezse None döner.
    """
    if not value:
        return None
    try:
        if isinstance(value, (int, float)):
            return float(value)
        s = str(value).strip()
        if s.endswith("px"):
            s = s[:-2]
        return float(s.replace(",", "."))
    except Exception:
        return None


def extract_rating_from_block(block, driver, debug=False, index=0):
    """
    Bir review bloğundan yıldız sayısını 1–5 arası integer olarak döndürür.
    Trendyol yapısında rating,
    .star-rating-full-star elemanının padding-inline-end değerinden türetiliyor.
    """
    star_container = None

    # Önce class ile bul
    try:
        star_container = block.find_element(
            By.CSS_SELECTOR, "div.star-rating-star-container.review-stars"
        )
    except Exception:
        pass

    # Bulamazsak data-testid ile dene
    if star_container is None:
        try:
            star_container = block.find_element(
                By.CSS_SELECTOR, '[data-testid="star-rating"]'
            )
        except Exception:
            if debug and index < 3:
                print(f"[DEBUG] #{index+1} star container bulunamadı")
            return None

    # İçteki full-star elemanı
    try:
        full_star = star_container.find_element(
            By.CSS_SELECTOR, "div.star-rating-full-star"
        )
    except Exception:
        full_star = None

    # 1) CSS değişkeni (--rating / --value) varsa direkt onu kullan
    try:
        rating_var = driver.execute_script(
            "var el = arguments[0]; "
            "var st = window.getComputedStyle(el); "
            "return st.getPropertyValue('--rating') || st.getPropertyValue('--value');",
            star_container,
        )
        rating_var = (rating_var or "").strip()
        if rating_var:
            val = float(rating_var.replace(",", "."))
            rating = max(1, min(5, round(val)))
            if debug and index < 3:
                print(f"[DEBUG] #{index+1} CSS var --rating -> {rating_var} -> {rating}")
            return rating
    except Exception:
        pass

    # 2) padding-inline-end / width oranına göre hesap
    try:
        # Container genişliği
        raw_container_width = driver.execute_script(
            "var el = arguments[0]; "
            "var st = window.getComputedStyle(el); "
            "return st.getPropertyValue('width');",
            star_container,
        )
        container_w = _parse_px(raw_container_width)

        if full_star is None or not container_w or container_w <= 0:
            raise RuntimeError("container_w veya full_star yok")

        # full-star padding-inline-end (veya padding-right)
        raw_pad = driver.execute_script(
            "var el = arguments[0]; "
            "var st = window.getComputedStyle(el); "
            "return st.getPropertyValue('padding-inline-end') "
            "   || st.getPropertyValue('padding-right');",
            full_star,
        )
        pad = _parse_px(raw_pad)
        if pad is None:
            raise RuntimeError("padding okunamadı")

        # Her yıldız genişliği ~ container/5
        star_w = container_w / 5.0
        if star_w <= 0:
            raise RuntimeError("star_w <= 0")

        # padding arttıkça görünür yıldız sayısı azalıyor:
        # pad ≈ (5 - rating) * star_w
        rating_float = 5.0 - (pad / star_w)
        rating = max(1, min(5, round(rating_float)))

        if debug and index < 3:
            print(
                f"[DEBUG] #{index+1} cont_w={container_w:.2f}px, pad={pad:.2f}px, "
                f"star_w≈{star_w:.2f}px -> rating≈{rating_float:.2f} -> {rating}"
            )

        return rating

    except Exception as e:
        if debug and index < 3:
            print(f"[DEBUG] #{index+1} padding/width yöntemi çalışmadı: {e}")

    # 3) Eski heuristikler (width%, ikon sayısı) – fallback
    try:
        import re
        style = star_container.get_attribute("style") or ""
        m = re.search(r"width:\s*([0-9]+)%", style)
        if m:
            percent = int(m.group(1))
            rating = max(1, min(5, round(percent / 20)))
            if debug and index < 3:
                print(
                    f"[DEBUG] #{index+1} container style width% "
                    f"-> {percent}% -> {rating}"
                )
            return rating
    except Exception:
        pass

    try:
        if full_star is not None:
            import re
            style = full_star.get_attribute("style") or ""
            m = re.search(r"width:\s*([0-9]+)%", style)
            if m:
                percent = int(m.group(1))
                rating = max(1, min(5, round(percent / 20)))
                if debug and index < 3:
                    print(
                        f"[DEBUG] #{index+1} full-star style width% "
                        f"-> {percent}% -> {rating}"
                    )
                return rating
    except Exception:
        pass

    try:
        full_icons = star_container.find_elements(
            By.CSS_SELECTOR, ".full-star, .full, .filled"
        )
        if full_icons:
            rating = len(full_icons)
            if debug and index < 3:
                print(f"[DEBUG] #{index+1} icon count -> {rating}")
            return rating
    except Exception:
        pass

    if debug and index < 3:
        print(f"[DEBUG] #{index+1} rating bulunamadı")
    return None


def collect_data():
    driver = None
    comments = []
    stars = []

    try:
        # URL giriş penceresi
        root = tk.Tk()
        root.withdraw()
        website_url = simpledialog.askstring(
            "URL Girişi", "Trendyol yorum bağlantısını girin (Çıkmak için 'exit' yazın):"
        )
        if not website_url or website_url.lower() == "exit":
            print("URL girişi iptal edildi.")
            return [], []

        # WebDriver başlat
        options = webdriver.ChromeOptions()
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options,
        )

        driver.get(website_url)

                # Sayfa içerisinde hareket etmek için "body" elementini bulma
        body = driver.find_element(By.TAG_NAME, 'body')

        for m in range(5000):
            body.send_keys(Keys.PAGE_DOWN)
            comments_elements = driver.find_elements(By.CSS_SELECTOR, 'div.comment-text > p') #yorum sayısı kontrol

            if m < 1000 and m % 50 == 0:
                for i in range(5):
                    body.send_keys(Keys.PAGE_UP)
                    os.system('cls')
                    
            elif m % 50 == 0:
                # 1000'den sonra her 50 basışta PAGE_UP sayısını arttır
                for i in range(5):
                    body.send_keys(Keys.PAGE_UP)
                    os.system('cls')
                    
            # Sayfa yüklenmesini beklemek için kısa bir süre bekletme
            time.sleep(2)
            
            print((f"Sayaç sayısı: {m}"))
            print((f"Anlık toplanan yorum sayısı: {len(comments_elements)}"))

            
            #Döngu bitmeden çıkmak için q ya basarak çıkmak
            if keyboard.is_pressed('q'):
                print("Cikis yapildi!")
                break

        # 'Devamını oku' linklerini tıkla
        click_all_read_more(driver)

        # Tüm review bloklarını bul
        review_blocks = driver.find_elements(
            By.CSS_SELECTOR, "div.review-list > div.review"
        )
        print(f"Bulunan review sayısı: {len(review_blocks)}")

        for idx, block in enumerate(review_blocks):
            # Yorum metni
            comment_text = ""
            try:
                comment_elem = block.find_element(
                    By.CSS_SELECTOR, "div.review-comment span.review-comment"
                )
                comment_text = comment_elem.text.strip()
            except StaleElementReferenceException:
                comment_text = ""
            except Exception:
                comment_text = ""

            # Yıldız değeri
            rating_value = extract_rating_from_block(
                block, driver, debug=True, index=idx
            )

            comments.append(comment_text)
            stars.append(rating_value)

            if idx < 3:
                print(f"[DEBUG] #{idx+1} Yorum: {comment_text[:60]!r}")
                print(f"[DEBUG] #{idx+1} Yıldız (son): {rating_value}")

    except Exception as e:
        print(f"Bir hata oluştu: {e}")

    finally:
        if driver is not None:
            driver.quit()

    return comments, stars


if __name__ == "__main__":
    from itertools import zip_longest

    comments, stars = collect_data()

    print(f"Toplam Yorum Sayısı: {len(comments)}")
    print(f"Toplam Yıldız Sayısı: {len(stars)}")

    data = []
    for comment, star in zip_longest(comments, stars):
        data.append(
            {
                "comment": comment,
                "rating": star,
            }
        )

    # JSON kaydet
    with open("trendyol_reviews.json", "w", encoding="utf-8") as jf:
        json.dump(data, jf, ensure_ascii=False, indent=2)

    print("JSON: trendyol_reviews.json")
    
    # xlsx kaydet
    df = pd.DataFrame(data)
    df.to_excel("trendyol_reviews.xlsx", index=False)
    print("Excel: trendyol_reviews.xlsx")
    print("Veri toplama tamamlandı.")