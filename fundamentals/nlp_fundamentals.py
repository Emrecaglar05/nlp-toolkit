# ======================================================
# 📦 Gerekli Kütüphaneler
# ======================================================
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from textblob import Word, TextBlob
from wordcloud import WordCloud
from PIL import Image
from warnings import filterwarnings
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

filterwarnings("ignore")  # Uyarıları görmezden gel

# ======================================================
# 1️⃣ VERI YUKLEME
# ======================================================
df = pd.read_csv(
    "C:/Users/EXCALIBUR15AUG025/Yeni klasör/Desktop/nlp/datasets/amazon_reviews.csv"
)[["reviewText", "overall"]].dropna()  # reviewText ve overall sütunlarını al, eksikleri at
print("✅ Veri yüklendi. Boyut:", df.shape)

# ======================================================
# 2️⃣ METIN ÖN ISLEME
# ======================================================
df["reviewText"] = (
    df["reviewText"]
    .astype(str)  # stringe çevir
    .str.lower()  # küçük harfe çevir
    .str.replace(r"[^\w\s]", " ", regex=True)  # noktalama işaretlerini kaldır
    .str.replace(r"\d+", " ", regex=True)  # sayıları kaldır
)

stop_words = set(stopwords.words("english"))  # Ingilizce stopword listesi
df["reviewText"] = df["reviewText"].apply(
    lambda x: " ".join([w for w in x.split() if w not in stop_words])
)  # stopword temizleme

word_counts = pd.Series(" ".join(df["reviewText"]).split()).value_counts()  # kelime frekansları
rare = set(word_counts[word_counts == 1].index)  # sadece 1 kez geçen nadir kelimeler
df["reviewText"] = df["reviewText"].apply( lambda x: " ".join([w for w in x.split() if w not in rare]))  # nadir kelimeleri kaldır

df["tokens"] = df["reviewText"].apply(lambda x: TextBlob(x).words)  # tokenization (kelimelere ayır)
df["reviewText"] = df["reviewText"].apply(lambda x: " ".join([Word(w).lemmatize() for w in x.split()]))  # lemmatization (kelime köklerini al)

print("✅ Ön işleme tamamlandı. Örnek:")
print(df["reviewText"].head())

# ======================================================
# 3️⃣ DUYGU ETİKETLERİ OLUŞTURMA
# ======================================================
sia = SentimentIntensityAnalyzer()  # VADER sentiment analyzer

# compound skor >0 ise pozitif = 1, <=0 ise negatif = 0
df["sentiment_label"] = df["reviewText"].apply(lambda x: 1 if sia.polarity_scores(x)["compound"] > 0 else 0)
print(df["sentiment_label"].value_counts())  # her etiketin sayısı

y = df["sentiment_label"]  # hedef değişken
X = df["reviewText"]       # özellikler

# ======================================================
# 4️⃣ TF-IDF ÖZELLIKLERINI OLUSTURMA
# ======================================================
tf_idf_word_vectorizer = TfidfVectorizer(max_features=20)  # en önemli 20 kelimeyi seç
X_tf_idf_word = tf_idf_word_vectorizer.fit_transform(X)   # TF-IDF matrisi oluştur

df_tf_idf_word = pd.DataFrame(
    X_tf_idf_word.toarray(), columns=tf_idf_word_vectorizer.get_feature_names_out()
)  # DataFrame ile göster
print("\n🔹 TF-IDF (Kelime Bazlı) - İlk 5 satır")
print(df_tf_idf_word.head())

# Örnek: unigram + bigram (tek kelime ve iki kelime grubu)
tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20)

# TF-IDF matrisini oluştur
X_tf_idf_ngram = tf_idf_ngram_vectorizer.fit_transform(X)

# DataFrame ile göster
df_tf_idf_ngram = pd.DataFrame(
    X_tf_idf_ngram.toarray(), columns=tf_idf_ngram_vectorizer.get_feature_names_out()
)

print("\n🔹 TF-IDF (N-gram Bazlı) - İlk 5 satır")
print(df_tf_idf_ngram.head())
# ======================================================
# 5️⃣ MODELLEME - Logistic Regression
# ======================================================
log_model = LogisticRegression(max_iter=1000).fit(X_tf_idf_word, y)  # model eğit

# 5 katlı cross-validation ile doğruluk
hata_orani = cross_val_score(
    log_model, X_tf_idf_word, y, scoring="accuracy", cv=5
).mean()
print("\n📊 Ortalama Doğruluk (CV=5):", round(hata_orani, 3))

# ======================================================
# 6️⃣ YENI YORUM TAHMINI
# ======================================================
new_review = pd.Series(["this product is great"])  # tahmin için yeni yorum
new_review_vec = tf_idf_word_vectorizer.transform(new_review)  # fit değil, sadece transform

tahmin = log_model.predict(new_review_vec)[0]          # tahmin
prob = log_model.predict_proba(new_review_vec)[0]      # olasılık değerleri

print("\n📝 Yeni Yorum:", new_review.iloc[0])
if tahmin == 1:
    print("🔮 Tahmin: Pozitif 😊")
else:
    print("🔮 Tahmin: Negatif 😡")

print("📊 Negatif olasılığı:", round(prob[0], 3))
print("📊 Pozitif olasılığı:", round(prob[1], 3))


# ======================================================
# 7️⃣ RANDOM FOREST MODELI
# ======================================================
# Count Vectors
count_vectorizer = CountVectorizer(max_features=20)
X_count = count_vectorizer.fit_transform(X)

rf_model = RandomForestClassifier().fit(X_count, y)
print("\n🌲 RF (Count Vectors) CV Sonuç:",
      cross_val_score(rf_model, X_count, y, cv=5, n_jobs=-1).mean())

# TF-IDF Word-Level
rf_model = RandomForestClassifier().fit(X_tf_idf_word, y)
print("🌲 RF (TF-IDF Word) CV Sonuç:",
      cross_val_score(rf_model, X_tf_idf_word, y, cv=5, n_jobs=-1).mean())

# TF-IDF N-GRAM
rf_model = RandomForestClassifier().fit(X_tf_idf_ngram, y)
print("🌲 RF (TF-IDF N-gram) CV Sonuç:",
      cross_val_score(rf_model, X_tf_idf_ngram, y, cv=5, n_jobs=-1).mean())

rf_model = RandomForestClassifier(random_state=17)
rf_params = {
    "max_depth": [8, None],
    "max_features": [7, "auto"],
    "min_samples_split": [2, 5, 8],
    "n_estimators": [100, 200]
}

rf_best_grid = GridSearchCV(
    rf_model, rf_params, cv=5, n_jobs=-1, verbose=1
).fit(X_count, y)

print("\n✅ En iyi parametreler:", rf_best_grid.best_params_)
print("📊 En iyi CV skoru:", round(rf_best_grid.best_score_, 3))



























