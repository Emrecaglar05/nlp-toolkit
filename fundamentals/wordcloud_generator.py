from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Metni birleştir ve temizle
text = """
Makine,arasındaki,temel,fark,sence,nedir? Türk,Milleti,bağımsız,yaşamış,bağımsızlığı,varolmalarının,yegâne,koşulu,olarak,kabul,etmiş,cesur,insanların,torunlarıdır.,Bu, millet,hiçbir,zaman,hür,olmadan,yaşamamıştır,,yaşayamaz,yaşamayacaktır. Kuantum,bilgisayarlar,klasik,bilgisayarlardan,çok,daha,hızlı,işlem,yapabilir.,Bu,sayede,karmaşık,problemler,kısa,sürede,çözülebilir. Bu,tarlanın,taşı,toprağı,,taşlı,tarlaya,taş,taşıyan,taşçının,taşıdığı,taşlar,taşlandı. Düzenli,yürüyüş,sağlığa,faydalıdır,stresi,azaltır.,(Ek:,Sabah,saatlerinde,yapılan,yürüyüşler,enerji,sunar,metabolizmayı,hızlandırır.)
"""

# Noktalama işaretlerini temizle
import re
clean_text = re.sub(r'[^\w\s]', '', text)
clean_text = clean_text.replace('\n', ' ')

# Kelime bulutu oluştur
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(clean_text)

# Görselleştir
plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
