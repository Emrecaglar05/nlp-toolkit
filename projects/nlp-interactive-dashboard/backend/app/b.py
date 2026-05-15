# frontend/streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import numpy as np
import io

# Backend modüllerini import et
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from preprocessing import TextPreprocessor

# Sayfa ayarı
st.set_page_config(
    page_title="NLP Analiz Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Başlık
st.title("📊 NLP Analiz ve Öneri Dashboard")
st.markdown("Metin verilerinizi analiz edin, görselleştirin ve öngörüler elde edin")

# Sidebar
st.sidebar.header("Ayarlar")

# Dil seçimi
language = st.sidebar.selectbox(
    "Dil Seçin",
    ("english", "turkish"),
    index=0
)

# Sektör seçimi
sector = st.sidebar.selectbox(
    "Sektör Seçin",
    ("E-Ticaret", "Finans", "Kitap", "Diğer"),
    index=0
)

# Dosya yükleme
uploaded_file = st.sidebar.file_uploader(
    "Veri dosyası yükleyin (CSV veya Excel)",
    type=["csv", "xlsx"]
)

# Örnek veri
use_sample_data = st.sidebar.checkbox("Örnek veri kullan", value=True)

# Ön işleyiciyi başlat
preprocessor = TextPreprocessor(language=language)


def load_sample_data(sector):
    """Örnek veri yükle"""
    if sector == "E-Ticaret":
        return pd.DataFrame({
            'review': [
                'Ürün çok kaliteli, hızlı kargo için teşekkürler',
                'Beklediğim gibi çıkmadı, hayal kırıklığı',
                'Fiyatına göre çok iyi, tavsiye ederim',
                'Kargo geç geldi, paket hasarlıydı',
                'Mükemmel, kesinlikle tekrar alacağım'
            ],
            'rating': [5, 2, 4, 1, 5]
        })
    elif sector == "Finans":
        return pd.DataFrame({
            'text': [
                'Borsa bugün çok iyi performans gösterdi',
                'Dolar artışı endişe verici',
                'Yatırım yapmak için iyi bir fırsat',
                'Ekonomik belirsizlik devam ediyor',
                'Faiz kararı piyasaları olumlu etkiledi'
            ]
        })
    else:
        return pd.DataFrame({
            'text': [
                'Kitap çok akıcı ve etkileyici',
                'Konu derinlemesine işlenmemiş',
                'Karakter gelişimi mükemmel',
                'Sonu hayal kırıklığı yarattı',
                'Edebiyat severler için harika bir eser'
            ]
        })


# Veri yükleme
if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.success("Veri başarıyla yüklendi!")
elif use_sample_data:
    df = load_sample_data(sector)
    st.info("Örnek veri kullanılıyor")
else:
    st.warning("Lütfen bir veri dosyası yükleyin veya örnek veri kullanın")
    st.stop()

# Veriyi göster
st.subheader("Ham Veri")
st.dataframe(df.head())

# Metin sütunu seçimi
text_columns = [col for col in df.columns if df[col].dtype == 'object']
if text_columns:
    text_column = st.selectbox("Analiz edilecek metin sütununu seçin", text_columns)
else:
    st.error("Metin sütunu bulunamadı!")
    st.stop()

# Ön işleme
if st.button("Metni Ön İşle ve Analiz Et"):
    with st.spinner("Metin ön işleniyor..."):
        df_processed = preprocessor.full_preprocess(df, text_column)

        # Duygu analizi
        sentiment_results = df_processed['cleaned_text'].apply(
            lambda x: preprocessor.analyze_sentiment(x)
        )
        df_processed['sentiment'], df_processed['sentiment_score'] = zip(*sentiment_results)

        # Sonuçları göster
        st.subheader("Ön İşlenmiş Veri")
        st.dataframe(df_processed[['cleaned_text', 'sentiment', 'sentiment_score']].head())

        # Görselleştirmeler
        col1, col2 = st.columns(2)

        with col1:
            # Duygu dağılımı
            st.subheader("Duygu Dağılımı")
            sentiment_counts = df_processed['sentiment'].value_counts()
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Duygu Dağılımı"
            )
            st.plotly_chart(fig)

        with col2:
            # Kelime bulutu
            st.subheader("Kelime Bulutu")
            all_text = ' '.join(df_processed['cleaned_text'].dropna())
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white'
            ).generate(all_text)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

        # Sektöre özel öneriler
        st.subheader("📈 Sektöre Özel Öneriler")

        if sector == "E-Ticaret":
            # E-ticaret önerileri
            negative_reviews = df_processed[df_processed['sentiment'] == 'negative']

            if not negative_reviews.empty:
                st.warning("⚠️ Dikkat: Negatif yorumlar tespit edildi")

                # Kargo ile ilgili şikayetler
                shipping_issues = negative_reviews[
                    negative_reviews['cleaned_text'].str.contains('kargo|teslimat|nakliye')
                ]

                if not shipping_issues.empty:
                    st.info("🚚 Kargo/Teslimat sorunları tespit edildi")
                    st.write("**Öneri:** Kargo süreçlerini gözden geçirin, müşterilere daha iyi takip imkanı sağlayın")

                # Kalite şikayetleri
                quality_issues = negative_reviews[
                    negative_reviews['cleaned_text'].str.contains('kalite|bozuk|hasarlı')
                ]

                if not quality_issues.empty:
                    st.info("🎯 Ürün kalitesi ile ilgili şikayetler tespit edildi")
                    st.write("**Öneri:** Tedarikçi kalite kontrol süreçlerini iyileştirin")

            # Pozitif yorumlardan öğrenme
            positive_reviews = df_processed[df_processed['sentiment'] == 'positive']
            if not positive_reviews.empty:
                st.success("✅ Güçlü yönleriniz:")
                positive_topics = []

                if positive_reviews['cleaned_text'].str.contains('hızlı|çabuk').any():
                    positive_topics.append("Hızlı servis")
                if positive_reviews['cleaned_text'].str.contains('kaliteli|iyi').any():
                    positive_topics.append("Ürün kalitesi")
                if positive_reviews['cleaned_text'].str.contains('ucuz|uygun').any():
                    positive_topics.append("Fiyat avantajı")

                for topic in positive_topics:
                    st.write(f"✓ {topic}")

        elif sector == "Finans":
            # Finans önerileri
            st.info("📊 Finansal veri analizi")

            # Piyasa duyarlılığı
            market_sentiment = df_processed['sentiment'].value_counts(normalize=True)
            positive_ratio = market_sentiment.get('positive', 0)

            if positive_ratio > 0.6:
                st.success("📈 Piyasa genel olarak olumlu")
                st.write("**Öneri:** Yatırımcılar için fırsat penceresi açık olabilir")
            elif positive_ratio < 0.4:
                st.warning("📉 Piyasa genel olarak olumsuz")
                st.write("**Öneri:** Risk yönetimi stratejilerinizi gözden geçirin")

            # Önemli kelimeler
            financial_keywords = ['faiz', 'dolar', 'borsa', 'yatırım', 'ekonomi']
            for keyword in financial_keywords:
                keyword_reviews = df_processed[df_processed['cleaned_text'].str.contains(keyword)]
                if not keyword_reviews.empty:
                    sentiment = keyword_reviews['sentiment'].value_counts().idxmax()
                    st.write(f"• '{keyword}' için genel eğilim: {sentiment}")

        elif sector == "Kitap":
            # Kitap analizi önerileri
            st.info("📚 Kitap analizi")

            # En çok geçen konular
            common_words = pd.Series(' '.join(df_processed['cleaned_text']).split()).value_counts().head(10)
            st.write("**En sık kullanılan kelimeler:**")
            for word, count in common_words.items():
                st.write(f"- {word}: {count} kez")

            # Genel değerlendirme
            avg_sentiment = df_processed['sentiment_score'].mean()
            if avg_sentiment > 0.1:
                st.success("👍 Genel olarak olumlu değerlendirmeler")
                st.write("**Öneri:** Benzer temalı kitapları önerebilirsiniz")
            else:
                st.warning("👎 Genel olarak olumsuz değerlendirmeler")
                st.write("**Öneri:** İçerik iyileştirme için geri bildirimleri dikkate alın")

        # İndirme butonu
        csv = df_processed.to_csv(index=False)
        st.download_button(
            label="Analiz Sonuçlarını İndir (CSV)",
            data=csv,
            file_name="analiz_sonuclari.csv",
            mime="text/csv"
        )

# Hakkında bölümü
st.sidebar.markdown("---")
st.sidebar.subheader("Hakkında")
st.sidebar.info(
    """
    Bu uygulama metin verilerinizi analiz eder, görselleştirir ve sektöre özel öneriler sunar.

    **Özellikler:**
    - Metin ön işleme
    - Duygu analizi
    - Kelime bulutu
    - Sektörel öneriler
    """
)