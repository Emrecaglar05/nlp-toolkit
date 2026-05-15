# Gerekli kütüphanelerin import edilmesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Veri yükleme ve ön inceleme fonksiyonu
def load_and_explore_data(file_path):
    """Veriyi yükler ve temel istatistikleri gösterir"""
    data = pd.read_csv(file_path)

    print("\nVeri Setinin İlk 5 Satırı:")
    print(data.head())

    print("\nTemel İstatistikler:")
    print(data.describe())

    print("\nEksik Veri Kontrolü:")
    print(data.isnull().sum())

    print("\nTanı Dağılımı:")
    print(data['diagnosis'].value_counts())

    return data


# Veri ön işleme fonksiyonu
def preprocess_data(data):
    """Veriyi temizler ve hazırlar"""
    # Gereksiz sütunları kaldırma
    data = data.drop('id', axis=1)

    # Kategorik değişkeni sayısala çevirme
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    return data


# Veri görselleştirme fonksiyonları
def plot_diagnosis_distribution(data):
    """Tanı dağılımını gösterir"""
    plt.figure(figsize=(8, 6))
    sns.countplot(x='diagnosis', data=data)
    plt.title('Tanı Dağılımı (0 = İyi Huylu, 1 = Kötü Huylu)')
    plt.show()


def plot_correlation_matrix(data):
    """Korelasyon matrisini çizer"""
    plt.figure(figsize=(20, 15))
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                annot_kws={"size": 8}, cbar_kws={"shrink": .82})
    plt.title('Özellikler Arası Korelasyon Matrisi', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_feature_distribution(data, feature):
    """Belirli bir özelliğin dağılımını gösterir"""
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='diagnosis', y=feature, data=data)
    plt.title(f'{feature} Özelliğinin Tanıya Göre Dağılımı')
    plt.show()

def plot_accuracy_comparison(results):
    models = list(results.keys())
    accuracy = [results[model]['accuracy'] for model in models]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=models, y=accuracy, palette=sns.color_palette("Blues_d", len(models)))
    plt.title("Modellerin Doğruluk Karşılaştırması")
    plt.ylabel("Doğruluk Skoru")
    plt.ylim(0.9, 1.0)
    for i, val in enumerate(accuracy):
        plt.text(i, val + 0.001, f"{val:.3f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.show()


# Özellik seçimi fonksiyonu
def select_features(X, y):
    """Rastgele Orman kullanarak önemli özellikleri seçer"""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    feature_importances = pd.DataFrame(
        rf.feature_importances_,
        index=X.columns,
        columns=['importance']
    ).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=feature_importances.importance, y=feature_importances.index)
    plt.title('Özellik Önem Sıralaması')
    plt.tight_layout()
    plt.show()

    top_features = feature_importances.head(10).index
    return X[top_features]


# Model eğitimi ve değerlendirme fonksiyonu
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Farklı modelleri eğitir ve değerlendirir"""
    # Model listesi
    models = {
        "Rastgele Orman": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='linear', random_state=42),
        "Sinir Ağı": MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
    }

    results = {}

    for name, model in models.items():
        # Modeli eğitme
        model.fit(X_train, y_train)

        # Tahmin yapma
        y_pred = model.predict(X_test)

        # Performans metrikleri
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "accuracy": accuracy,
            "report": report,
            "confusion_matrix": cm
        }

        # Sonuçları yazdırma
        print(f"\n{name} Modeli:")
        print("Doğruluk:", accuracy)
        print("Sınıflandırma Raporu:")
        print(report)
        print("Karışıklık Matrisi:")
        print(cm)

    return results


# Ana iş akışı fonksiyonu
def main():
    # 1. Veri yükleme ve keşif
    data = load_and_explore_data('Breast_cancer_dataset.csv')

    # 2. Veri ön işleme
    data = preprocess_data(data)

    # 3. Veri görselleştirme
    plot_diagnosis_distribution(data)
    plot_correlation_matrix(data)
    plot_feature_distribution(data, 'radius_mean')

    # 4. Özellik seçimi
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    X_selected = select_features(X, y)

    # 5. Veriyi bölme ve ölçeklendirme
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 6. Model eğitimi ve değerlendirme
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    plot_accuracy_comparison(results)

# Ana fonksiyonu çalıştır
if __name__ == "__main__":
    main()