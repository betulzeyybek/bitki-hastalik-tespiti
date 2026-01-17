# 🌿 Bitki Hastalığı Tespit Sistemi

> Yapay zeka destekli, 2 aşamalı bitki hastalığı tespit ve sınıflandırma sistemi

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Özellikler](#-özellikler)
- [Sistem Mimarisi](#-sistem-mimarisi)
- [Model Karşılaştırması](#-model-karşılaştırması)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Proje Yapısı](#-proje-yapısı)
- [Teknolojiler](#-teknolojiler)
- [Sonuçlar](#-sonuçlar)
- [Ekran Görüntüleri](#-ekran-görüntüleri)
- [Katkıda Bulunma](#-katkıda-bulunma)
- [Lisans](#-lisans)

---

## 🎯 Proje Hakkında

Bu proje, tarımda bitki hastalıklarının erken teşhisi için geliştirilmiş yapay zeka destekli bir sistemdir. **Teachable Machine** ve **Derin Öğrenme (CNN)** modellerini birleştirerek, bitki yapraklarındaki hastalıkları yüksek doğrulukla tespit eder.

### Neden Bu Proje?

- 🌾 **Tarımsal Verimlilik**: Erken teşhis ile ürün kaybını minimize eder
- 🤖 **Yapay Zeka**: Uzman bilgisi gerektirmeden otomatik tespit
- ⚡ **Hızlı Analiz**: Saniyeler içinde sonuç
- 📱 **Kullanıcı Dostu**: Basit web arayüzü ile kolay kullanım

---

## ✨ Özellikler

### 🔬 2 Aşamalı Tespit Sistemi

1. **Aşama 1: Yaprak Tespiti** (Teachable Machine)
   - Görüntüde yaprak var mı yok mu kontrolü
   - %100 başarı oranı
   - Hızlı ön eleme

2. **Aşama 2: Hastalık Sınıflandırması** (CNN)
   - 38 farklı hastalık sınıfı
   - %93.35 doğruluk oranı
   - Detaylı analiz ve güven skoru

### 🎯 Desteklenen Özellikler

- ✅ **38 Hastalık Sınıfı** - Geniş hastalık kütüphanesi
- ✅ **14 Bitki Türü** - Domates, elma, üzüm, patates vb.
- ✅ **Gerçek Zamanlı Analiz** - Anında sonuç
- ✅ **Güven Skoru** - Tahmin güvenilirliği
- ✅ **Top-5 Tahmin** - En olası 5 hastalık
- ✅ **Web Arayüzü** - Streamlit tabanlı

---

## 🏗️ Sistem Mimarisi

```
┌─────────────────┐
│ Görüntü Girişi  │
│   (Yaprak)      │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  Aşama 1: Yaprak Tespiti    │
│  (Teachable Machine)        │
│  • Yaprak var mı?           │
│  • Binary Classification    │
└────────┬────────────────────┘
         │
    ┌────┴────┐
    │         │
   VAR       YOK
    │         │
    │    ┌────▼─────┐
    │    │   ❌     │
    │    │  REDDET  │
    │    └──────────┘
    │
    ▼
┌─────────────────────────────┐
│ Aşama 2: Hastalık Analizi   │
│ (CNN - MobileNetV2)         │
│ • 38 hastalık sınıfı        │
│ • %93.35 doğruluk           │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│        SONUÇ                │
│  • Bitki Türü               │
│  • Hastalık Adı             │
│  • Güven Skoru              │
│  • Top-5 Tahmin             │
└─────────────────────────────┘
```

---

## 📊 Model Karşılaştırması

Projede 3 farklı CNN modeli eğitilmiş ve karşılaştırılmıştır:

| Model | Doğruluk (Val) | Parametre | Eğitim Süresi | Çıkarım Hızı |
|-------|----------------|-----------|---------------|--------------|
| **Custom CNN** | %88.5 | 2.5M | ~45 dk | ~80ms |
| **ResNet50** | %91.2 | 25.6M | ~120 dk | ~150ms |
| **MobileNetV2** ⭐ | **%93.35** | 2.9M | ~60 dk | ~120ms |

> ⭐ **MobileNetV2** en iyi performans/hız dengesi sunduğu için seçildi

### Metrikler

- **Precision**: %94.2
- **Recall**: %92.8
- **F1-Score**: %93.5
- **Top-5 Accuracy**: %98.7

---

## 🚀 Kurulum

### Gereksinimler

- Python 3.9 veya üzeri
- pip paket yöneticisi
- 8GB+ RAM (önerilen)
- GPU (opsiyonel, hızlandırma için)

### Adım 1: Projeyi Klonla

```bash
git clone https://github.com/betulzeyybek/bitki-hastalik-tespiti.git
cd bitki-hastalik-tespiti
```

### Adım 2: Sanal Ortam Oluştur (Önerilen)

```bash
# Sanal ortam oluştur
python3 -m venv venv

# Aktif et
source venv/bin/activate  # Mac/Linux
# VEYA
venv\Scripts\activate  # Windows
```

### Adım 3: Bağımlılıkları Yükle

```bash
pip install -r requirements.txt
```

### Adım 4: Modelleri İndir

Modeller çok büyük olduğu için GitHub'da yok. İndirmek için:

```bash
# Google Drive'dan indir (link README'de)
# VEYA kendi modelini eğit:
python src/2_model_egitimi.py
```

---

## 💻 Kullanım

### Web Arayüzü ile (Önerilen)

```bash
# Streamlit uygulamasını başlat
streamlit run app.py
```

Tarayıcıda otomatik olarak açılacak: `http://localhost:8501`

### Python ile

```python
from tensorflow import keras
from PIL import Image
import numpy as np
import json

# Modelleri yükle
leaf_detector = keras.models.load_model('models/leaf_detector.h5')
disease_model = keras.models.load_model('models/mobilenetv2.keras')

# Sınıf isimlerini yükle
with open('models/class_names.json', 'r') as f:
    class_names = json.load(f)

# Görüntüyü yükle ve işle
image = Image.open('test_image.jpg').resize((224, 224))
img_array = np.array(image) / 255.0
img_batch = np.expand_dims(img_array, axis=0)

# 1. Yaprak var mı?
leaf_pred = leaf_detector.predict(img_batch)
if leaf_pred[0][0] > 0.5:
    # 2. Hastalık tespit et
    disease_pred = disease_model.predict(img_batch)
    class_idx = np.argmax(disease_pred[0])
    confidence = disease_pred[0][class_idx]
    
    print(f"Hastalık: {class_names[class_idx]}")
    print(f"Güven: %{confidence*100:.2f}")
else:
    print("Yaprak tespit edilemedi!")
```

---

## 📁 Proje Yapısı

```
bitki-hastalik-tespiti/
│
├── app.py                          # Ana Streamlit uygulaması
│
├── models/                         # Eğitilmiş modeller
│   ├── leaf_detector.h5            # Teachable Machine (yaprak tespiti)
│   ├── leaf_labels.txt             # Yaprak tespit sınıfları
│   ├── mobilenetv2.keras           # CNN hastalık modeli
│   ├── custom_cnn.keras            # Alternatif model 1
│   ├── resnet50.keras              # Alternatif model 2
│   └── class_names.json            # 38 hastalık sınıfı
│
├── dataset/                        # Eğitim verisi (gitignore'da)
│   ├── train/                      # Eğitim seti
│   ├── validation/                 # Validasyon seti
│   └── test/                       # Test seti
│
├── src/                            # Kaynak kodlar
│   ├── 1_veri_hazirlama.py         # Veri ön işleme
│   ├── 2_model_egitimi.py          # Model eğitim scripti
│   └── 3_model_degerlendirme.py    # Model değerlendirme
│
├── results/                        # Eğitim sonuçları ve grafikler
│   ├── model_comparison/           # Model karşılaştırma grafikleri
│   └── archive/                    # Eski sonuçlar
│
├── requirements.txt                # Python bağımlılıkları
├── .gitignore                      # Git ignore kuralları
└── README.md                       # Bu dosya
```

---

## 🛠️ Teknolojiler

### Makine Öğrenmesi

- **TensorFlow / Keras** - Derin öğrenme framework
- **Google Teachable Machine** - Transfer learning
- **MobileNetV2** - CNN mimarisi
- **ImageNet** - Pretrained weights

### Veri İşleme

- **NumPy** - Sayısal hesaplamalar
- **Pandas** - Veri analizi
- **Pillow (PIL)** - Görüntü işleme
- **OpenCV** - İleri düzey görüntü işleme

### Görselleştirme

- **Matplotlib** - Grafik çizimi
- **Seaborn** - İstatistiksel görselleştirme
- **Plotly** - İnteraktif grafikler

### Web Arayüzü

- **Streamlit** - Hızlı web uygulaması
- **HTML/CSS** - Özel stil

---

## 📈 Sonuçlar

### Performans Metrikleri

- ✅ **Genel Doğruluk**: %93.35
- ✅ **Yaprak Tespiti**: %100
- ✅ **Ortalama İşlem Süresi**: ~165ms
- ✅ **Top-5 Accuracy**: %98.7

### Veri Seti

**PlantVillage Dataset**
- 📊 54,000+ etiketli görüntü
- 🌱 14 bitki türü
- 🦠 38 hastalık sınıfı
- 📏 Train/Val/Test: 70%/20%/10%

### Desteklenen Bitkiler

| Bitki | Hastalık Sayısı |
|-------|-----------------|
| 🍅 Domates | 10 |
| 🥔 Patates | 3 |
| 🍇 Üzüm | 4 |
| 🍎 Elma | 4 |
| 🌶️ Biber | 2 |
| 🍓 Çilek | 2 |
| 🍑 Şeftali | 2 |
| 🌽 Mısır | 4 |
| Ve daha fazlası... | - |

---

## 📸 Ekran Görüntüleri

### Ana Sayfa
*Web arayüzü görüntüsü eklenecek*

### Analiz Sonucu
*Başarılı tespit örneği eklenecek*

### Model Karşılaştırması
*Training grafikleri eklenecek*

---

## 🔮 Gelecek Geliştirmeler

- [ ] 📱 Mobil uygulama (iOS/Android)
- [ ] 🌍 Daha fazla bitki türü desteği
- [ ] 💊 Hastalık tedavi önerileri
- [ ] 🗣️ Çoklu dil desteği (İngilizce, Almanca, vb.)
- [ ] 🔌 REST API endpoint'leri
- [ ] 📊 Kullanıcı istatistikleri ve raporlama
- [ ] 🤝 Topluluk katkıları (crowd-sourced data)

---

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Katkıda bulunmak için:

1. Bu depoyu fork'layın
2. Yeni bir branch oluşturun (`git checkout -b feature/yeniOzellik`)
3. Değişikliklerinizi commit'leyin (`git commit -m 'Yeni özellik eklendi'`)
4. Branch'inizi push'layın (`git push origin feature/yeniOzellik`)
5. Pull Request oluşturun

### Katkı Alanları

- 🐛 Bug düzeltmeleri
- ✨ Yeni özellikler
- 📝 Dokümantasyon iyileştirmeleri
- 🧪 Test yazımı
- 🌍 Çeviri (i18n)

---

## 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakınız.

---

## 👤 Yazar

**Betül Zeybek**

- GitHub: [@betulzeyybek](https://github.com/betulzeyybek)
- Email: betul60gs@outlook.com

---

## 🙏 Teşekkürler

- **PlantVillage** - Dataset sağladığı için
- **Google Teachable Machine** - Transfer learning imkanı için
- **TensorFlow Ekibi** - Harika framework için
- **Streamlit** - Kolay web arayüzü için
- **Açık Kaynak Topluluğu** - İlham ve destek için

---

## 📚 Referanslar

1. PlantVillage Dataset: [https://github.com/spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
2. MobileNetV2 Paper: [https://arxiv.org/abs/1801.04381](https://arxiv.org/abs/1801.04381)
3. Teachable Machine: [https://teachablemachine.withgoogle.com/](https://teachablemachine.withgoogle.com/)

---

<div align="center">

**⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın! ⭐**

**Made with ❤️ and 🤖 in Turkey 🇹🇷**

</div>