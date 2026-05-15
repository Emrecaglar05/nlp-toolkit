# Sahte Haber Tespit Sistemi
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns


# nltk.download("stopwords")  # İlk çalıştırmada aktif et
# nltk.download("punkt")      # İlk çalıştırmada aktif et

def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/lutzhamel/fake-news/refs/heads/master/data/fake_or_real_news.csv")
    print(f"Veri boyutu: {df.shape}")
    print(f"Label dağılımı:\n{df['label'].value_counts()}")  # REAL/FAKE oranı
    return df


def clean_text(text):
    """Metin temizleme ve tokenizasyon"""
    text = text.lower()  # Küçük harfe çevir
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Özel karakterleri temizle
    tokens = word_tokenize(text)  # Kelimelere ayır
    stop_words = set(stopwords.words("english"))  # Stop words seti
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]  # Filtrele
    return " ".join(tokens)


def preprocess_data(df):
    print("Metin temizleme...")
    df["cleaned_text"] = df["text"].apply(clean_text)  # Temizlenmiş metin kolonu ekledik
    return df


def vectorize_text(df):
    """TF-IDF vektörizasyonu"""
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # En sık geçen 5000 özellik seçtik 2 kelimelik kombinasyonlarla.
    X = vectorizer.fit_transform(df["cleaned_text"])  # Metni vektöre çevir
    y = df["label"].map({"REAL": 1, "FAKE": 0})  # Kategorik sutunu sayısala çevirdik modelin anlaması için
    print(f"Özellik boyutu: {X.shape}")
    return X, y, vectorizer


def split_and_train(X, y):
    """Veri bölme ve model eğitimi"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # %80 eğitim

    model = LogisticRegression(max_iter=1000)  # Model tanımla
    model.fit(X_train, y_train)  # Eğit

    y_pred = model.predict(X_test)  # Test et
    accuracy = accuracy_score(y_test, y_pred)  # Doğruluk hesapla

    print(f"Doğruluk: {accuracy * 100:.2f}%")
    print("\nDetay rapor:")
    print(classification_report(y_test, y_pred, target_names=['SAHTE', 'GERÇEK']))

    return model


def predict_news(model, vectorizer, text):
    """Yeni haber tahmini"""
    cleaned = clean_text(text)  # Metni temizle
    vectorized = vectorizer.transform([cleaned])  # Vektöre çevir
    prediction = model.predict(vectorized)[0]  # Tahmin yap
    proba = model.predict_proba(vectorized)[0].max()  # Güven skoru

    result = "GERÇEK" if prediction == 1 else "SAHTE"
    return result, proba * 100


def create_wordclouds(df):
    """Sahte vs Gerçek haberlerin kelime bulutları"""
    plt.figure(figsize=(15, 8))

    # Gerçek haberler kelime bulutu
    real_text = ' '.join(df[df['label'] == 'REAL']['cleaned_text'])
    plt.subplot(1, 2, 1)
    wordcloud_real = WordCloud(width=600, height=400,
                               background_color='white',
                               colormap='Blues').generate(real_text)
    plt.imshow(wordcloud_real, interpolation='bilinear')
    plt.title('GERÇEK Haberlerde En Çok Geçen Kelimeler', fontsize=14, fontweight='bold')
    plt.axis('off')

    # Sahte haberler kelime bulutu
    fake_text = ' '.join(df[df['label'] == 'FAKE']['cleaned_text'])
    plt.subplot(1, 2, 2)
    wordcloud_fake = WordCloud(width=600, height=400,
                               background_color='white',
                               colormap='Reds').generate(fake_text)
    plt.imshow(wordcloud_fake, interpolation='bilinear')
    plt.title('SAHTE Haberlerde En Çok Geçen Kelimeler', fontsize=14, fontweight='bold')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def analyze_word_frequency(df):
    """Kelime frekans analizi ve karşılaştırma"""
    # En çok kullanılan kelimeleri bul
    real_words = ' '.join(df[df['label'] == 'REAL']['cleaned_text']).split()
    fake_words = ' '.join(df[df['label'] == 'FAKE']['cleaned_text']).split()

    real_freq = Counter(real_words).most_common(20)  # Top 20 gerçek
    fake_freq = Counter(fake_words).most_common(20)  # Top 20 sahte

    # Karşılaştırmalı bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Gerçek haberler
    words_real, counts_real = zip(*real_freq)
    ax1.barh(words_real, counts_real, color='lightblue', alpha=0.8)
    ax1.set_title('Gerçek Haberlerde En Sık Kelimeler', fontweight='bold')
    ax1.set_xlabel('Frekans')

    # Sahte haberler
    words_fake, counts_fake = zip(*fake_freq)
    ax2.barh(words_fake, counts_fake, color='lightcoral', alpha=0.8)
    ax2.set_title('Sahte Haberlerde En Sık Kelimeler', fontweight='bold')
    ax2.set_xlabel('Frekans')

    plt.tight_layout()
    plt.show()

    return real_freq, fake_freq


def distinctive_words_analysis(df, vectorizer):
    """Ayırt edici kelimeler analizi - hangi kelimeler daha karakteristik"""
    # TF-IDF skorlarını al
    tfidf_matrix = vectorizer.fit_transform(df['cleaned_text'])
    feature_names = vectorizer.get_feature_names_out()

    # Her label için ortalama TF-IDF skorları
    real_indices = df[df['label'] == 'REAL'].index
    fake_indices = df[df['label'] == 'FAKE'].index

    real_scores = np.mean(tfidf_matrix[real_indices].toarray(), axis=0)
    fake_scores = np.mean(tfidf_matrix[fake_indices].toarray(), axis=0)

    # Farkları hesapla (pozitif = sahte karakteristik, negatif = gerçek karakteristik)
    score_diff = fake_scores - real_scores

    # En ayırt edici kelimeleri bul
    top_fake_words = [(feature_names[i], score_diff[i]) for i in np.argsort(score_diff)[-15:][::-1]]
    top_real_words = [(feature_names[i], abs(score_diff[i])) for i in np.argsort(score_diff)[:15]]

    # Görselleştir
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Sahte haberlere özgü kelimeler
    words_f, scores_f = zip(*top_fake_words)
    ax1.barh(words_f, scores_f, color='red', alpha=0.7)
    ax1.set_title('Sahte Haberlere Özgü Kelimeler', fontweight='bold')
    ax1.set_xlabel('Ayırt Edicilık Skoru')

    # Gerçek haberlere özgü kelimeler
    words_r, scores_r = zip(*top_real_words)
    ax2.barh(words_r, scores_r, color='blue', alpha=0.7)
    ax2.set_title('Gerçek Haberlere Özgü Kelimeler', fontweight='bold')
    ax2.set_xlabel('Ayırt Edicillik Skoru')

    plt.tight_layout()
    plt.show()

    print("\n SAHTE haberlere özgü kelimeler:")
    for word, score in top_fake_words[:10]:
        print(f"   • {word}: {score:.4f}")

    print("\n GERÇEK haberlere özgü kelimeler:")
    for word, score in top_real_words[:10]:
        print(f"   • {word}: {score:.4f}")

    return top_fake_words, top_real_words


# Ana program
def main():
    print("=== SAHTE HABER TESPİT SİSTEMİ ===\n")

    df = load_data()  # Veri yükle
    df = preprocess_data(df)  # Ön işle
    X, y, vectorizer = vectorize_text(df)  # Vektörleştir
    model = split_and_train(X, y)  # Eğit

    # VİZUALİZASYONLAR
    print("\n Görselleştirmeler oluşturuluyor...")

    # 1. Kelime bulutları
    print("Kelime bulutları...")
    create_wordclouds(df)

    # 2. Frekans analizi
    print("Kelime frekans analizi...")
    real_freq, fake_freq = analyze_word_frequency(df)

    # 3. Ayırt edici kelimeler
    print("Ayırt edici kelimeler analizi...")
    top_fake, top_real = distinctive_words_analysis(df, vectorizer)

    # Örnek tahmin
    print("\n Örnek tahmin:")
    sample = "Scientists discover new planet in solar system"
    result, confidence = predict_news(model, vectorizer, sample)
    print(f"   Haber: {sample}")
    print(f"   Sonuç: {result} (%{confidence:.1f} güven)")

    return model, vectorizer


if __name__ == "__main__":
    model, vectorizer = main()