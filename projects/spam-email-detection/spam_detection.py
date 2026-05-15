import pandas as pd                # Veri işleme için pandas kütüphanesi
import re                         # Düzenli ifadelerle metin temizleme için
import nltk                       # Doğal dil işleme kütüphanesi
from nltk.corpus import stopwords # Gereksiz kelimeleri almak için
from nltk.stem import PorterStemmer # Kelimeleri köklerine indirgemek için
from sklearn.feature_extraction.text import TfidfVectorizer # Metni sayısal verilere çevirmek için
from sklearn.model_selection import train_test_split       # Veriyi eğitim ve test olarak bölmek için
from sklearn.linear_model import LogisticRegression         # Lojistik regresyon sınıflandırıcısı
from sklearn.metrics import accuracy_score, classification_report # Model performansını ölçmek için

# Stopwords listesini indir (ilk kullanımda)
nltk.download("stopwords")

# Kökleyici (stemmer) ve İngilizce stopwords seti oluştur
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# Veri dosyasını oku ve sadece "v1" ve "v2" sütunlarını al
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]

# Sütun isimlerini daha anlamlı hale getir
df.columns = ["etiket", "mesaj"]

# Etiketleri sayısal değerlere dönüştür ("ham" = 0, "spam" = 1)
df["etiket"] = df["etiket"].map({"ham": 0, "spam": 1})

# Metinleri temizlemek için fonksiyon yaz
def metin_on_isleme(text):
    text = re.sub(r"\W", " ", text)       # Metindeki özel karakterleri boşlukla değiştir
    text = text.lower()                    # Metni küçük harfe çevir
    kelimeler = text.split()               # Kelimelere ayır
    kelimeler = [stemmer.stem(k) for k in kelimeler if k not in stop_words]  # Stopword'leri çıkar, kelimeleri köklerine indir
    return " ".join(kelimeler)             # Kelimeleri tekrar birleştir

# Tüm mesajlara temizleme fonksiyonunu uygula, yeni sütuna yaz
df["temizlenmis_mesaj"] = df["mesaj"].apply(metin_on_isleme)

# TF-IDF vektörleştirici nesnesi oluştur
vectorizer = TfidfVectorizer(max_features=3000)

# Metinleri sayısal verilere dönüştür (öğrenme için)
X = vectorizer.fit_transform(df["temizlenmis_mesaj"])
y = df["etiket"]
print(df.head())

# Veriyi eğitim ve test olarak ayır (yüzde 80 eğitim, yüzde 20 test)
X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lojistik regresyon modeli oluştur ve eğit
model = LogisticRegression()
model.fit(X_egitim, y_egitim)

# Test verisi ile tahmin yap
y_tahmin = model.predict(X_test)

# Modelin doğruluk oranını ve detaylı raporu yazdır
print(f"Doğruluk: {accuracy_score(y_test, y_tahmin) * 100:.2f}%")
print(classification_report(y_test, y_tahmin))

# Yeni gelen e-postayı sınıflandırmak için fonksiyon
def e_posta_tahmin(etk_metni):
    temiz_met = metin_on_isleme(etk_metni)           # Ön işleme yap
    vektorlestirilmis = vectorizer.transform([temiz_met])  # Vektörleştir
    tahmin = model.predict(vektorlestirilmis)        # Model ile tahmin yap
    return "Spam" if tahmin[0] == 1 else "Spam Değil"

# Örnek e-posta ile deneme yap
email = "go jurong point crazi avail bugi n great world..."
print(f"E-posta: {email}\nTahmin: {e_posta_tahmin(email)}")
