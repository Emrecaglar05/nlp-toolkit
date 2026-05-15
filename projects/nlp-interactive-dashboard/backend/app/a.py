# backend/preprocessing.py
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob, Word
import warnings

warnings.filterwarnings("ignore")

# NLTK verilerini indir
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Türkçe stopwords
try:
    from nltk.corpus import stopwords

    TURKISH_STOPWORDS = set(stopwords.words('turkish'))
except:
    # Fallback Türkçe stopwords
    TURKISH_STOPWORDS = set([
        'acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç',
        'birşey', 'biz', 'bu', 'çok', 'çünkü', 'da', 'daha', 'de', 'defa',
        'diye', 'eğer', 'en', 'gibi', 'hem', 'hep', 'hepsi', 'her', 'hiç',
        'için', 'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu', 'mü', 'nasıl',
        'ne', 'neden', 'nerde', 'nerede', 'nereye', 'niçin', 'niye', 'o',
        'sanki', 'şey', 'siz', 'şu', 'tüm', 've', 'veya', 'ya', 'yani'
    ])


class TextPreprocessor:
    def __init__(self, language='english'):
        self.language = language
        self.stop_words = set(stopwords.words(language)) if language == 'english' else TURKISH_STOPWORDS
        self.sia = SentimentIntensityAnalyzer()

    def clean_text(self, text):
        """Metni temizleme"""
        if pd.isna(text):
            return ""

        text = str(text).lower()

        # URL'leri kaldır
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Kullanıcı adlarını ve hashtag'leri kaldır
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)

        # Noktalama işaretlerini ve sayıları kaldır
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)

        # Fazla boşlukları kaldır
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def remove_stopwords(self, text):
        """Stopword'leri kaldır"""
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words and len(word) > 2]
        return ' '.join(filtered_words)

    def lemmatize_text(self, text):
        """Lemmatization işlemi - Basit Türkçe kök bulma"""
        if self.language == 'english':
            return ' '.join([Word(word).lemmatize() for word in text.split()])
        else:
            # Türkçe için basit bir kök bulma (gerçek lemmatization yerine)
            # Gerçek projede ZemberekPy veya benzeri kütüphane kullanılmalı
            return text  # Basitçe aynı metni döndür

    def tokenize_text(self, text):
        """Metni token'lara ayır"""
        return text.split()

    def full_preprocess(self, df, text_column):
        """Tam ön işleme pipeline'ı"""
        # Metin temizleme
        df['cleaned_text'] = df[text_column].apply(self.clean_text)

        # Stopword'leri kaldır
        df['cleaned_text'] = df['cleaned_text'].apply(self.remove_stopwords)

        # Lemmatization
        df['cleaned_text'] = df['cleaned_text'].apply(self.lemmatize_text)

        # Tokenization
        df['tokens'] = df['cleaned_text'].apply(self.tokenize_text)

        return df

    def analyze_sentiment(self, text):
        """Duygu analizi yap"""
        if self.language == 'english':
            scores = self.sia.polarity_scores(text)
            compound = scores['compound']

            if compound >= 0.05:
                return 'positive', compound
            elif compound <= -0.05:
                return 'negative', compound
            else:
                return 'neutral', compound
        else:
            # Türkçe için basit duygu analizi
            positive_words = ['iyi', 'güzel', 'harika', 'mükemmel', 'süper', 'muhteşem', 'teşekkür', 'tavsiye']
            negative_words = ['kötü', 'berbat', 'fena', 'rezalet', 'hayal', 'hata', 'problem', 'arıza']

            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)

            if pos_count > neg_count:
                return 'positive', 0.5
            elif neg_count > pos_count:
                return 'negative', -0.5
            else:
                return 'neutral', 0

    def extract_keywords(self, text, n=10):
        """Anahtar kelimeleri çıkar"""
        from collections import Counter
        words = text.split()
        # Sadece anlamlı kelimeleri al (uzunluk > 3)
        meaningful_words = [word for word in words if len(word) > 3]
        return dict(Counter(meaningful_words).most_common(n))