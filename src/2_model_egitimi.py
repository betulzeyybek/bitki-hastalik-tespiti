import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import time

# GPU Kontrolü
print("🖥️  Sistem Kontrolü:", "GPU Mevcut" if len(tf.config.list_physical_devices('GPU')) > 0 else "CPU kullanılıyor")

class ModelEgitici:
    def __init__(self):
        # Yollar
        self.veri_yolu = Path("data/processed")
        self.model_yolu = Path("models")
        self.sonuc_yolu = Path("results")
        
        # Hiperparametreler
        self.img_boyut = (224, 224)
        self.batch_size = 32
        self.epochs = 30
        self.sinif_sayisi = 38
        
        # Klasörleri oluştur
        self.model_yolu.mkdir(exist_ok=True)
        self.sonuc_yolu.mkdir(exist_ok=True)
        
        # Sınıf isimlerini al
        train_dir = self.veri_yolu / 'train'
        if not train_dir.exists():
            raise FileNotFoundError(f"❌ HATA: {train_dir} bulunamadı!")
            
        self.sinif_isimleri = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        
    def sinif_agirliklarini_hesapla(self, generator):
        """Veri dengesizliğini yenmek için ağırlıkları hesaplar"""
        print("⚖️  Sınıf ağırlıkları hesaplanıyor...")
        labels = generator.classes
        unique_classes = np.unique(labels)
        
        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=labels
        )
        return dict(zip(unique_classes, weights))

    def veri_yukleyicileri_olustur(self):
        print("\n" + "=" * 50)
        print("📊 VERİ YÜKLEYİCİLERİ HAZIRLANIYOR")
        print("=" * 50)
        
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        self.train_generator = train_datagen.flow_from_directory(
            self.veri_yolu / 'train',
            target_size=self.img_boyut,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        self.val_generator = val_test_datagen.flow_from_directory(
            self.veri_yolu / 'validation',
            target_size=self.img_boyut,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

    def mobilenetv2_egit(self):
        print("\n" + "=" * 50)
        print("🚀 MOBILENETV2 EĞİTİMİ BAŞLIYOR")
        print("=" * 50)
        
        agirliklar = self.sinif_agirliklarini_hesapla(self.train_generator)

        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.sinif_sayisi, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
        ]
        
        baslangic = time.time()
        history = model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=self.epochs,
            class_weight=agirliklar,
            callbacks=callbacks,
            verbose=1
        )
        sure = time.time() - baslangic
        
        model.save(self.model_yolu / "mobilenetv2.keras")
        print(f"\n✅ Eğitim Tamamlandı! Süre: {sure/60:.1f} dk")
        return history

    def grafikleri_ciz(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(history.history['accuracy'], label='Train')
        ax1.plot(history.history['val_accuracy'], label='Val')
        ax1.set_title('Doğruluk')
        ax1.legend()
        
        ax2.plot(history.history['loss'], label='Train')
        ax2.plot(history.history['val_loss'], label='Val')
        ax2.set_title('Kayıp (Loss)')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.sonuc_yolu / "final_egitim_grafigi.png")
        plt.show()

if __name__ == "__main__":
    egitici = ModelEgitici()
    egitici.veri_yukleyicileri_olustur()
    history = egitici.mobilenetv2_egit()
    egitici.grafikleri_ciz(history)
