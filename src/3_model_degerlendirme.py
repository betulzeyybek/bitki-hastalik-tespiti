

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
from pathlib import Path
import pandas as pd

class ModelDegerlendirici:
    def __init__(self):
        # Önce data/processed kontrol et, yoksa dataset kullan
        if Path("data/processed").exists():
            self.veri_yolu = Path("data/processed")
        else:
            self.veri_yolu = Path("dataset")
            print("ℹ️  data/processed bulunamadı, dataset/ klasörü kullanılıyor")
        
        self.model_yolu = Path("models")
        self.sonuc_yolu = Path("results")
        
        self.img_boyut = (224, 224)
        self.batch_size = 32
        
        # Sınıf isimlerini yükle
        class_names_file = self.model_yolu / 'class_names.json'
        
        if not class_names_file.exists():
            print(f"⚠️  class_names.json bulunamadı, train klasöründen oluşturuluyor...")
            # Train klasöründen sınıf isimlerini al
            train_dir = self.veri_yolu / 'train'
            if train_dir.exists():
                self.sinif_isimleri = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
                # Kaydet
                self.model_yolu.mkdir(exist_ok=True)
                with open(class_names_file, 'w', encoding='utf-8') as f:
                    json.dump(self.sinif_isimleri, f, indent=2, ensure_ascii=False)
                print(f"✅ class_names.json oluşturuldu: {len(self.sinif_isimleri)} sınıf")
            else:
                raise FileNotFoundError(f"❌ Train klasörü bulunamadı: {train_dir}")
        else:
            with open(class_names_file, 'r', encoding='utf-8') as f:
                self.sinif_isimleri = json.load(f)
            print(f"✅ {len(self.sinif_isimleri)} sınıf yüklendi")
    
    def test_yukleyici_olustur(self):
        """Test veri yükleyici oluşturur"""
        print("\n" + "=" * 70)
        print("📊 TEST VERİSİ HAZIRLANIYOR")
        print("=" * 70)
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # test veya valid klasörünü kullan
        test_path = self.veri_yolu / 'test'
        if not test_path.exists():
            test_path = self.veri_yolu / 'valid'
            print(f"ℹ️  test/ klasörü bulunamadı, valid/ klasörü kullanılıyor")
        
        self.test_generator = test_datagen.flow_from_directory(
            test_path,
            target_size=self.img_boyut,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False  # Önemli: Shuffle False olmalı
        )
        
        print(f"✅ Test örnekleri: {self.test_generator.samples}")
        print(f"✅ Sınıf sayısı: {len(self.sinif_isimleri)}")
    
    def model_yukle_ve_degerlendir(self, model_adi):
        """Modeli yükler ve test setinde değerlendirir"""
        print("\n" + "=" * 70)
        print(f"🔍 {model_adi.upper()} DEĞERLENDİRİLİYOR")
        print("=" * 70)
        
        # Modeli yükle
        model_dosyasi = self.model_yolu / f"{model_adi}.keras"
        
        if not model_dosyasi.exists():
            print(f"❌ Model dosyası bulunamadı: {model_dosyasi}")
            return None
        
        model = keras.models.load_model(model_dosyasi)
        print(f"✅ Model yüklendi: {model_dosyasi}")
        
        # Test seti üzerinde değerlendir
        print("\n📊 Test seti üzerinde değerlendiriliyor...")
        test_loss, test_acc = model.evaluate(self.test_generator, verbose=0)
        
        print(f"\n✅ Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"✅ Test Loss: {test_loss:.4f}")
        
        # Tahminleri al
        print("\n🔮 Detaylı tahminler alınıyor...")
        y_pred_probs = model.predict(self.test_generator, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = self.test_generator.classes
        
        return {
            'model': model,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_probs': y_pred_probs
        }
    
    def confusion_matrix_ciz(self, y_true, y_pred, model_adi):
        """Confusion Matrix oluşturur"""
        print(f"\n📊 {model_adi} için Confusion Matrix oluşturuluyor...")
        
        # Confusion matrix hesapla
        cm = confusion_matrix(y_true, y_pred)
        
        # Sınıf isimlerini kısalt (daha okunabilir)
        sinif_isimleri_kisa = []
        for s in self.sinif_isimleri:
            if '___' in s:
                # "Tomato___Early_blight" -> "Early_blight"
                sinif_isimleri_kisa.append(s.split('___')[-1][:20])
            else:
                sinif_isimleri_kisa.append(s[:20])
        
        # Grafik oluştur
        plt.figure(figsize=(22, 20))
        
        # Normalize edilmiş confusion matrix (daha okunabilir)
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        
        sns.heatmap(
            cm_normalized, 
            annot=False,  # Sayıları gösterme (38x38 çok büyük)
            fmt='.2f',
            cmap='Blues',
            xticklabels=sinif_isimleri_kisa,
            yticklabels=sinif_isimleri_kisa,
            cbar_kws={'label': 'Normalize Edilmiş Değer'},
            linewidths=0.1,
            linecolor='gray'
        )
        
        plt.title(f'{model_adi.upper()} - Confusion Matrix (Normalize)', 
                 fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Tahmin Edilen Sınıf', fontsize=14, fontweight='bold')
        plt.ylabel('Gerçek Sınıf', fontsize=14, fontweight='bold')
        plt.xticks(rotation=90, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        
        dosya = self.sonuc_yolu / f'5_{model_adi}_confusion_matrix.png'
        plt.savefig(dosya, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion Matrix kaydedildi: {dosya}")
        plt.close()
        
        # Doğru tahmin oranlarını hesapla
        dogruluk_oranlari = cm_normalized.diagonal()
        ortalama_dogruluk = dogruluk_oranlari.mean()
        print(f"  📊 Ortalama sınıf doğruluğu: {ortalama_dogruluk:.4f} ({ortalama_dogruluk*100:.2f}%)")
        
        return cm
    
    def classification_report_olustur(self, y_true, y_pred, model_adi):
        """Sınıf bazlı performans raporu"""
        print(f"\n📋 {model_adi} için Classification Report oluşturuluyor...")
        
        # Rapor oluştur
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=self.sinif_isimleri,
            output_dict=True,
            zero_division=0
        )
        
        # DataFrame'e çevir
        df = pd.DataFrame(report).transpose()
        
        # En iyi ve en kötü 10 sınıfı bul
        df_classes = df.iloc[:-3]  # Son 3 satırı çıkar (accuracy, macro avg, weighted avg)
        df_sorted = df_classes.sort_values('f1-score', ascending=False)
        
        # Grafik oluştur
        fig, axes = plt.subplots(1, 2, figsize=(18, 10))
        
        # En iyi 10
        top10 = df_sorted.head(10)
        colors_top = ['#2ecc71' if score > 0.9 else '#27ae60' for score in top10['f1-score']]
        bars1 = axes[0].barh(range(10), top10['f1-score'], color=colors_top, alpha=0.8, edgecolor='black')
        axes[0].set_yticks(range(10))
        # Sınıf isimlerini kısalt
        top10_labels = []
        for s in top10.index:
            if '___' in s:
                top10_labels.append(s.split('___')[-1][:35])
            else:
                top10_labels.append(s[:35])
        axes[0].set_yticklabels(top10_labels, fontsize=10)
        axes[0].set_xlabel('F1-Score', fontsize=12, fontweight='bold')
        axes[0].set_title(f'✅ En İyi 10 Sınıf (F1-Score)', fontsize=14, fontweight='bold', color='green')
        axes[0].set_xlim([0, 1.0])
        axes[0].invert_yaxis()
        axes[0].grid(axis='x', alpha=0.3)
        
        # Değerleri bar üzerine yaz
        for i, (bar, score) in enumerate(zip(bars1, top10['f1-score'])):
            axes[0].text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{score:.3f}', 
                        ha='left', va='center', fontsize=10, fontweight='bold')
        
        # En kötü 10
        bottom10 = df_sorted.tail(10)
        colors_bottom = ['#e74c3c' if score < 0.7 else '#c0392b' for score in bottom10['f1-score']]
        bars2 = axes[1].barh(range(10), bottom10['f1-score'], color=colors_bottom, alpha=0.8, edgecolor='black')
        axes[1].set_yticks(range(10))
        # Sınıf isimlerini kısalt
        bottom10_labels = []
        for s in bottom10.index:
            if '___' in s:
                bottom10_labels.append(s.split('___')[-1][:35])
            else:
                bottom10_labels.append(s[:35])
        axes[1].set_yticklabels(bottom10_labels, fontsize=10)
        axes[1].set_xlabel('F1-Score', fontsize=12, fontweight='bold')
        axes[1].set_title(f'⚠️ En Zor 10 Sınıf (F1-Score)', fontsize=14, fontweight='bold', color='red')
        axes[1].set_xlim([0, 1.0])
        axes[1].invert_yaxis()
        axes[1].grid(axis='x', alpha=0.3)
        
        # Değerleri bar üzerine yaz
        for i, (bar, score) in enumerate(zip(bars2, bottom10['f1-score'])):
            axes[1].text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{score:.3f}', 
                        ha='left', va='center', fontsize=10, fontweight='bold')
        
        plt.suptitle(f'{model_adi.upper()} - Sınıf Bazlı Performans', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        dosya = self.sonuc_yolu / f'6_{model_adi}_class_performance.png'
        plt.savefig(dosya, dpi=300, bbox_inches='tight')
        print(f"✅ Sınıf performansı kaydedildi: {dosya}")
        plt.close()
        
        # Raporu JSON olarak kaydet
        rapor_dosyasi = self.sonuc_yolu / f'{model_adi}_classification_report.json'
        with open(rapor_dosyasi, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        
        # Özet istatistikleri yazdır
        print(f"\n📊 {model_adi.upper()} - Özet Performans:")
        print(f"  • Test Accuracy: {report['accuracy']:.4f} ({report['accuracy']*100:.2f}%)")
        print(f"  • Macro Avg Precision: {report['macro avg']['precision']:.4f}")
        print(f"  • Macro Avg Recall: {report['macro avg']['recall']:.4f}")
        print(f"  • Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}")
        print(f"  • Weighted Avg F1-Score: {report['weighted avg']['f1-score']:.4f}")
        
        return report
    
    def model_karsilastirma_tablosu(self, sonuclar):
        """Tüm modellerin karşılaştırma tablosu"""
        print("\n" + "=" * 70)
        print("📊 TÜM MODELLER KARŞILAŞTIRMA TABLOSU")
        print("=" * 70)
        
        # Veri hazırla
        data = []
        for model_adi, sonuc in sonuclar.items():
            report = sonuc['report']
            data.append({
                'Model': model_adi.upper(),
                'Test Accuracy': sonuc['test_acc'],
                'Test Loss': sonuc['test_loss'],
                'Precision (Macro)': report['macro avg']['precision'],
                'Recall (Macro)': report['macro avg']['recall'],
                'F1-Score (Macro)': report['macro avg']['f1-score'],
                'F1-Score (Weighted)': report['weighted avg']['f1-score']
            })
        
        df = pd.DataFrame(data)
        
        # Konsola yazdır (formatlı)
        print("\n")
        for col in df.columns:
            if col != 'Model':
                df[col] = df[col].apply(lambda x: f"{x:.4f}")
        print(df.to_string(index=False))
        
        # CSV olarak kaydet (sayısal değerlerle)
        df_csv = pd.DataFrame([{
            'Model': d['Model'],
            'Test Accuracy': d['Test Accuracy'],
            'Test Loss': d['Test Loss'],
            'Precision (Macro)': d['Precision (Macro)'],
            'Recall (Macro)': d['Recall (Macro)'],
            'F1-Score (Macro)': d['F1-Score (Macro)'],
            'F1-Score (Weighted)': d['F1-Score (Weighted)']
        } for d in data])
        
        csv_dosyasi = self.sonuc_yolu / 'model_karsilastirma.csv'
        df_csv.to_csv(csv_dosyasi, index=False)
        print(f"\n✅ Karşılaştırma tablosu kaydedildi: {csv_dosyasi}")
        
        # Grafik oluştur (geliştirilmiş)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        metriks = [
            'Test Accuracy', 
            'Precision (Macro)', 
            'Recall (Macro)', 
            'F1-Score (Macro)',
            'F1-Score (Weighted)',
            'Test Loss'
        ]
        
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
        
        for i, metrik in enumerate(metriks):
            ax = axes[i // 3, i % 3]
            values = df_csv[metrik].values
            models = df_csv['Model'].tolist()
            
            bars = ax.bar(models, values, color=colors[:len(models)], 
                         alpha=0.8, edgecolor='black', linewidth=1.5)
            ax.set_title(metrik, fontsize=13, fontweight='bold')
            ax.set_ylabel('Değer', fontsize=11)
            
            # Y ekseni limitleri
            if metrik == 'Test Loss':
                ax.set_ylim([0, max(values) * 1.2])
            else:
                ax.set_ylim([min(values) - 0.05, 1.0])
            
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=0)
            
            # Değerleri bar üzerine yaz
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.4f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.suptitle('🏆 MODEL PERFORMANS KARŞILAŞTIRMASI', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        dosya = self.sonuc_yolu / '7_final_comparison.png'
        plt.savefig(dosya, dpi=300, bbox_inches='tight')
        print(f"✅ Final karşılaştırma grafiği: {dosya}")
        plt.close()
        
        # En iyi modeli belirle
        best_model_idx = df_csv['Test Accuracy'].values.argmax()
        best_model = df_csv.iloc[best_model_idx]['Model']
        best_acc = df_csv.iloc[best_model_idx]['Test Accuracy']
        
        print(f"\n🏆 EN İYİ MODEL: {best_model} (Accuracy: {best_acc:.4f} / {best_acc*100:.2f}%)")
    
    def calistir(self):
        """Tüm değerlendirme pipeline'ını çalıştırır"""
        print("\n" + "🌿 " * 35)
        print("BİTKİ HASTALIĞI TESPİTİ - MODEL DEĞERLENDİRME")
        print("🌿 " * 35 + "\n")
        
        # Test veri yükleyici
        self.test_yukleyici_olustur()
        
        # Değerlendirilecek modeller (3 model)
        model_adlari = ['custom_cnn', 'resnet50', 'mobilenetv2']
        
        sonuclar = {}
        
        for model_adi in model_adlari:
            # Modeli değerlendir
            sonuc = self.model_yukle_ve_degerlendir(model_adi)
            
            if sonuc is None:
                print(f"⚠️ {model_adi} atlanıyor...")
                continue
            
            # Confusion Matrix
            cm = self.confusion_matrix_ciz(
                sonuc['y_true'], 
                sonuc['y_pred'], 
                model_adi
            )
            
            # Classification Report
            report = self.classification_report_olustur(
                sonuc['y_true'],
                sonuc['y_pred'],
                model_adi
            )
            
            sonuclar[model_adi] = {
                'test_acc': sonuc['test_acc'],
                'test_loss': sonuc['test_loss'],
                'report': report
            }
        
        # Karşılaştırma tablosu (eğer en az 1 model varsa)
        if sonuclar:
            self.model_karsilastirma_tablosu(sonuclar)
        
        print("\n" + "=" * 70)
        print("✅ MODEL DEĞERLENDİRME TAMAMLANDI!")
        print("=" * 70)
        print(f"\n📊 Tüm sonuçlar: {self.sonuc_yolu.absolute()}")
        print("\n📁 Oluşturulan Dosyalar:")
        print("   • Confusion Matrix: 5_*_confusion_matrix.png")
        print("   • Class Performance: 6_*_class_performance.png")
        print("   • Final Comparison: 7_final_comparison.png")
        print("   • CSV Report: model_karsilastirma.csv")
        print("\n➡️  Bir sonraki adım: Streamlit uygulaması (app.py)")

if __name__ == "__main__":
    degerlendirici = ModelDegerlendirici()
    degerlendirici.calistir()
