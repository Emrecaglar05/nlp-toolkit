from textblob import TextBlob
from deep_translator import GoogleTranslator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# TextBlob + Google Translate ile Türkçe cümlelerin analizini yapar
def analyze_sentiment_textblob(metin):
    try:
        ingilizce_metin = GoogleTranslator(source='tr', target='en').translate(metin)
        duygu = TextBlob(ingilizce_metin).sentiment.polarity

        if duygu > 0:
            return "Olumlu :)"
        elif duygu < 0:
            return "Olumsuz :("
        else:
            return "Doğal :|"
    except Exception as e:
        return f"Hata (TextBlob): {str(e)}"

# VADER ile İngilizce cümlelerin analizini yapar
analyzer = SentimentIntensityAnalyzer()
def analyze_sentiment_vader(text):
    try:
        sentiment_score = analyzer.polarity_scores(text)["compound"]
        if sentiment_score >= 0.05:
            return "Olumlu :)"
        elif sentiment_score <= -0.05:
            return "Olumsuz :("
        else:
            return "Doğal :|"
    except Exception as e:
        return f"Hata (VADER): {str(e)}"

# Kullanıcıdan cümle alıp her iki analiz yöntemini uygular
def analyze_user_input():
    print("=== DUYGU ANALİZİ ARACI ===")
    print("Çıkmak için 'q' yazın.\n")

    while True:
        text = input("Cümle girin: ")
        if text.lower() == 'q':
            print("Program sonlandırıldı.")
            break

        print("\n📊 Analiz Sonuçları:")
        print(f"TextBlob (Türkçe-İngilizce): {analyze_sentiment_textblob(text)}")
        print(f"VADER (İngilizce direkt):    {analyze_sentiment_vader(text)}")
        print("-" * 30)

# Ana programı çalıştır
if __name__ == "__main__":
    analyze_user_input()
