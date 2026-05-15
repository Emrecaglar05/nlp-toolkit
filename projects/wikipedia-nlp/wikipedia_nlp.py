import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from textblob import Word
from collections import Counter
from warnings import filterwarnings

filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: '%.2f' % x)
pd.set_option("display.width", 200)

# Veri yükleme
df = pd.read_csv('wiki_data.csv')

# Temizleme
def clean_text(text):
    text = text.str.lower()
    text = text.str.replace(r'[^\w\s]', " ")
    text = text.str.replace("\n", " ")
    text = text.str.replace("\d", " ")
    return text

df["text"] = clean_text(df["text"])

# Stopword kaldırma
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return text.apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))

df["text"] = remove_stopwords(df["text"])

# Lemmatization
def lemmatize_text(text):
    return text.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

df["text"] = lemmatize_text(df["text"])

# Kelime frekanslarını tüm metin üzerinden al
all_words = " ".join(df["text"]).split()
word_counts = Counter(all_words)

# 1000’den az geçen kelimeleri kaldır ve her satırı filtrele
def filter_rare_words(text, word_counts, min_count=1000):
    return " ".join([word for word in text.split() if word_counts[word] >= min_count])

df["text_filtered"] = df["text"].apply(lambda x: filter_rare_words(x, word_counts, min_count=1000))

# WordCloud için tüm kelimeleri birleştir
all_filtered_text = " ".join(df["text_filtered"].tolist())

# WordCloud
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_filtered_text)
plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Bar plot için kelime frekansları
filtered_word_counts = Counter(all_filtered_text.split())
most_common = filtered_word_counts.most_common(30)  # En sık 30 kelime

# Bar plot
words, counts = zip(*most_common)
plt.figure(figsize=(15, 7))
plt.bar(words, counts, color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.title("En Sık Kullanılan Kelimeler")
plt.show()
