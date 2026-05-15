import pandas as pd
import random
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split  # Hata düzeltildi
from sklearn.preprocessing import LabelEncoder

# NLTK kütüphanesinden VADER sözlüğünü indir
nltk.download("vader_lexicon")


def load_data(file_path):
    """CSV dosyasını pandas ile oku ve bir DataFrame'e yükle"""
    df = pd.read_csv(file_path)
    return df


def drop_missing(df):
    """'tweet' veya 'date' sütunlarında eksik değer içeren satırları sil"""
    df = df.dropna(subset=["tweet", "date"])
    return df


def convert_date_timezone(df):
    """'date' sütununu datetime formatına çevir ve GMT+03:00 dilimine ayarla"""
    df["date"] = pd.to_datetime(df["date"])
    df["date"] = df["date"].dt.tz_convert("Etc/GMT-3")
    return df


def extract_date_features(df):
    """Tarihten çeşitli özellikler çıkar (mevsim, gün, periyot)"""

    def mevsim_cek(month):
        """Ay numarasına göre mevsimi döndür"""
        if month in [12, 1, 2]:
            return "Kış"
        elif month in [3, 4, 5]:
            return "İlkbahar"
        elif month in [6, 7, 8]:
            return "Yaz"
        else:
            return "Sonbahar"

    def periyot_4saat(hour):
        """Saate göre 4 saatlik periyodu döndür"""
        if 22 <= hour <= 23 or 0 <= hour < 2:
            return "22-02"
        elif 2 <= hour < 6:
            return "02-06"
        elif 6 <= hour < 10:
            return "06-10"
        elif 10 <= hour < 14:
            return "10-14"
        elif 14 <= hour < 18:
            return "14-18"
        else:
            return "18-22"

    # Tarih özelliklerini çıkar
    df["Ay"] = df["date"].dt.month
    df["Mevsim"] = df["Ay"].apply(mevsim_cek)

    # Gün isimlerini Türkçe'ye çevir
    df["Gün"] = df["date"].dt.day_name()
    df["Gün"] = df["Gün"].replace({
        "Monday": "Pazartesi",
        "Tuesday": "Salı",
        "Wednesday": "Çarşamba",
        "Thursday": "Perşembe",
        "Friday": "Cuma",
        "Saturday": "Cumartesi",
        "Sunday": "Pazar"
    })

    # Saat bilgisi ve periyotları ekle
    df["Saat"] = df["date"].dt.hour
    df["4_Saatlik_Periyot"] = df["Saat"].apply(periyot_4saat)

    return df


def create_label(df):
    """VADER kullanarak tweet'lere duygu etiketi oluştur"""
    sia = SentimentIntensityAnalyzer()

    def vader_label(tweet):
        """Tweet'in duygu puanına göre etiket döndür (1: pozitif, -1: negatif, 0: nötr)"""
        score = sia.polarity_scores(str(tweet))["compound"]
        if score > 0:
            return 1
        elif score < 0:
            return -1
        else:
            return 0

    df["label"] = df["tweet"].apply(vader_label)
    return df


def get_numeric_categorical(df):
    """DataFrame'deki numerik ve kategorik sütunları ayır"""
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric_cols, categorical_cols


def target_analysis(df):
    """Hedef değişkenin (label) dağılımını analiz et ve yazdır"""
    print("Etiket Dağılımı (Sayı):")
    print(df["label"].value_counts())
    print("\nEtiket Dağılımı (Oran):")
    print(df["label"].value_counts(normalize=True))


def predict_random_tweet(df):
    """Rastgele bir tweet seç ve VADER ile duygu analizi yap"""
    sia = SentimentIntensityAnalyzer()
    random_tweet = df["tweet"].sample(n=1).iloc[0]
    score = sia.polarity_scores(str(random_tweet))["compound"]

    if score > 0:
        label = "Pozitif"
    elif score < 0:
        label = "Negatif"
    else:
        label = "Nötr"

    print("\nRastgele Tweet Örneği:")
    print(f"Tweet: {random_tweet}")
    print(f"Duygu: {label} (Skor: {score:.4f})")


def feature_engineering_pipeline(file_path):
    """Tüm özellik mühendisliği adımlarını sırayla uygula"""
    df = load_data(file_path)
    df = drop_missing(df)
    df = convert_date_timezone(df)
    df = extract_date_features(df)
    df = create_label(df)
    numeric_cols, categorical_cols = get_numeric_categorical(df)
    target_analysis(df)
    return df, numeric_cols, categorical_cols


def main():
    """Ana program akışı"""
    # Veri dosyasının yolunu belirt
    file_path = "tweets_labeled.csv"

    # Özellik mühendisliği pipeline'ını çalıştır
    df, numeric_cols, categorical_cols = feature_engineering_pipeline(file_path)

    # Veri setinin ilk birkaç satırını ve sütun tiplerini yazdır
    print("\nVeri Seti Örneği:")
    print(df.head())
    print("\nNumerik Sütunlar:", numeric_cols)
    print("Kategorik Sütunlar:", categorical_cols)

    # Rastgele bir tweet seç ve duygu tahminini yap
    predict_random_tweet(df)

    # Metin ön işleme: Tweet'leri küçük harfe çevir
    df['tweet'] = df['tweet'].astype(str).str.lower()
    df['label'] = df['label'].astype(str)

    # Etiketleri sayısal değerlere dönüştür
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])

    # Metin verisini TF-IDF vektörlerine dönüştür
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['tweet'])
    y = df['label']

    # Veriyi eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression modelini oluştur ve eğit
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Test seti üzerinde tahmin yap ve doğruluk oranını hesapla
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Modelin doğruluk oranını yazdır
    print(f"\nLogistic Regression Doğruluk Oranı: {accuracy:.4f}")


# Programı çalıştır
if __name__ == "__main__":
    main()