import os
import shutil
import random
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np

# Rastgelelik için seed
random.seed(42)

class VeriHazirla:
    def __init__(self):
        # Proje ana dizinini bul
        proje_dizini = Path(__file__).parent.parent.absolute()
        
        # Bölme oranları (Hatanın çözümü için burası şart)
        self.train_oran = 0.70
        self.val_oran = 0.15
        self.test_oran = 0.15
        
        # Seçenekleri sırayla dene
        secenekler = [
            proje_dizini / "data" / "plantvillage dataset" / "segmented",
            proje_dizini / "data" / "plantvillage dataset" / "color",
            Path("/Users/betulzeybek/Desktop/plant-disease-detection/data/plantvillage dataset/segmented")
        ]
        
        self.kaynak = None
        for yol in secenekler:
            if yol.exists():
                self.kaynak = yol
                break
        
        self.hedef = proje_dizini / "data" / "processed"
        
    def klasorleri_olustur(self):
        print("=" * 70)
        print("📁 KLASÖRLER OLUŞTURULUYOR")
        print("=" * 70)
        
        for bolum in ['train', 'validation', 'test']:
            klasor = self.hedef / bolum
            if klasor.exists():
                shutil.rmtree(klasor)
            klasor.mkdir(parents=True)
            print(f"✅ {bolum} klasörü oluşturuldu")
    
    def veriyi_bol(self):
        if self.kaynak is None:
            print("❌ HATA: Kaynak klasör bulunamadı!")
            return None, None

        print("\n" + "=" * 70)
        print("🔀 VERİ BÖLÜNÜYOR")
        print("=" * 70)
        
        siniflar = sorted([d.name for d in self.kaynak.iterdir() if d.is_dir()])
        print(f"\n📊 Toplam {len(siniflar)} sınıf bulundu\n")
        
        toplam_istatistik = {'train': 0, 'validation': 0, 'test': 0}
        
        for sinif_adi in siniflar:
            sinif_klasor = self.kaynak / sinif_adi
            for bolum in ['train', 'validation', 'test']:
                (self.hedef / bolum / sinif_adi).mkdir(parents=True, exist_ok=True)
            
            # Görüntüleri al
            goruntuler = list(sinif_klasor.glob('*.jpg')) + list(sinif_klasor.glob('*.JPG')) + list(sinif_klasor.glob('*.png'))
            random.shuffle(goruntuler)
            
            toplam = len(goruntuler)
            if toplam == 0: continue

            train_sayi = int(toplam * self.train_oran)
            val_sayi = int(toplam * self.val_oran)
            
            for i, img_yolu in enumerate(goruntuler):
                if i < train_sayi:
                    bolum = 'train'
                elif i < train_sayi + val_sayi:
                    bolum = 'validation'
                else:
                    bolum = 'test'
                
                hedef_yol = self.hedef / bolum / sinif_adi / img_yolu.name
                shutil.copy2(img_yolu, hedef_yol)
                toplam_istatistik[bolum] += 1
            
            print(f"✓ {sinif_adi[:40]:45} → Bitti")
        
        return siniflar, toplam_istatistik

    def calistir(self):
        print("\n" + "🌿 " * 20)
        print("BİTKİ HASTALIĞI TESPİTİ - VERİ HAZIRLAMA")
        print("🌿 " * 20 + "\n")
        
        self.klasorleri_olustur()
        siniflar, istatistik = self.veriyi_bol()
        
        if siniflar:
            print("\n" + "=" * 70)
            print("✅ VERİ HAZIRLAMA TAMAMLANDI!")
            print("=" * 70)
            print(f"\n📁 İşlenmiş veri: {self.hedef.absolute()}")

if __name__ == "__main__":
    hazirla = VeriHazirla()
    hazirla.calistir()
    