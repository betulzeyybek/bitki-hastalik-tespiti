

import streamlit as st
import numpy as np
from PIL import Image
import json
import os

try:
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.error("❌ TensorFlow yüklü değil!")


st.set_page_config(
    page_title="🌿 Bitki Hastalığı Tespit",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
<style>
    .main-title { text-align: center; color: #2E7D32; font-size: 3em; font-weight: bold; margin-bottom: 10px; }
    .subtitle { text-align: center; color: #558B2F; font-size: 1.2em; margin-bottom: 30px; }
    .success-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin: 20px 0; }
    .error-card { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 20px; border-radius: 10px; color: white; margin: 20px 0; }
    .info-box { background-color: #E8F5E9; padding: 15px; border-left: 5px solid #4CAF50; border-radius: 5px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)



PLANT_TR = {
    'Apple': 'Elma', 'Blueberry': 'Yaban Mersini', 'Cherry': 'Kiraz', 'Corn': 'Mısır',
    'Grape': 'Üzüm', 'Orange': 'Portakal', 'Peach': 'Şeftali', 'Pepper': 'Biber',
    'Potato': 'Patates', 'Raspberry': 'Ahudut', 'Soybean': 'Soya', 'Squash': 'Kabak',
    'Strawberry': 'Çilek', 'Tomato': 'Domates'
}

DISEASE_TR = {
    'healthy': 'Sağlıklı', 'Scab': 'Kabuklanma', 'Black_rot': 'Siyah Çürüklük',
    'Cedar_apple_rust': 'Pas Hastalığı', 'Powdery_mildew': 'Külleme',
    'Cercospora_leaf_spot': 'Yaprak Lekesi', 'Common_rust': 'Yaygın Pas',
    'Northern_Leaf_Blight': 'Yaprak Yanıklığı', 'Leaf_scorch': 'Yaprak Yanığı',
    'Haunglongbing': 'HLB Hastalığı', 'Bacterial_spot': 'Bakteriyel Leke',
    'Early_blight': 'Erken Yanıklık', 'Late_blight': 'Geç Yanıklık',
    'Leaf_Mold': 'Yaprak Küfü', 'Septoria_leaf_spot': 'Septoria Lekesi',
    'Spider_mites': 'Örümcek Akarı', 'Target_Spot': 'Hedef Leke',
    'Tomato_Yellow_Leaf_Curl_Virus': 'Sarı Kıvırcık Virüs',
    'Tomato_mosaic_virus': 'Mozaik Virüs', 'Leaf_blight': 'Yaprak Yanması',
    'Esca': 'Esca', 'Isariopsis_Leaf_Spot': 'Isariopsis Leke',
    'Two-spotted_spider': 'İki Nokta Örümcek'
}


@st.cache_resource
def load_models():
    models = {
        'leaf_detector': None,
        'leaf_labels': None,
        'cnn': None,
        'class_names': None
    }
    
    try:
        # Leaf Detector
        if os.path.exists('models/leaf_detector.h5'):
            models['leaf_detector'] = keras.models.load_model('models/leaf_detector.h5', compile=False)
            
        
        # Leaf Labels
        if os.path.exists('models/leaf_labels.txt'):
            with open('models/leaf_labels.txt', 'r') as f:
                models['leaf_labels'] = [line.strip() for line in f.readlines()]
        
        # CNN Model
        if os.path.exists('models/mobilenetv2.keras'):
            models['cnn'] = keras.models.load_model('models/mobilenetv2.keras')
            
        
        # Class Names
        if os.path.exists('models/class_names.json'):
            with open('models/class_names.json', 'r', encoding='utf-8') as f:
                models['class_names'] = json.load(f)
                st.sidebar.success(f"✅ {len(models['class_names'])} sınıf yüklendi")
    
    except Exception as e:
        st.error(f"Model yükleme hatası: {e}")
    
    return models

models = load_models()


def detect_leaf(image, leaf_model, leaf_labels):
    """Teachable Machine ile yaprak tespiti - AKILLI VERSİYON"""
    
    if leaf_model is None:
        return {
            'is_leaf': True,
            'confidence': 1.0,
            'leaf_score': 1.0,
            'nonleaf_score': 0.0,
            'message': '⚠️ Leaf Detector yok, doğrudan CNN analizi yapılıyor'
        }
    
    # Görüntü hazırlama
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Tahmin
    predictions = leaf_model.predict(img_batch, verbose=0)[0]
    
    # Teachable Machine formatı: labels.txt
    # 0 leaf
    # 1 nonleaf
    leaf_score = float(predictions[0])
    nonleaf_score = float(predictions[1])
    
    # AKILLI EŞİK: %40'dan fazla leaf ise kabul et
    is_leaf = (leaf_score > 0.40)
    
    # Güven durumu
    if is_leaf:
        if leaf_score > 0.80:
            confidence_level = "Yüksek"
            emoji = "✅"
        elif leaf_score > 0.60:
            confidence_level = "Orta"
            emoji = "⚠️"
        else:
            confidence_level = "Düşük"
            emoji = "⚠️"
        message = f"{emoji} Yaprak tespit edildi (Güven: {confidence_level} - %{leaf_score*100:.1f})"
    else:
        message = f"❌ Yaprak tespit edilemedi (Yaprak değil: %{nonleaf_score*100:.1f})"
    
    return {
        'is_leaf': is_leaf,
        'confidence': leaf_score if is_leaf else nonleaf_score,
        'leaf_score': leaf_score,
        'nonleaf_score': nonleaf_score,
        'message': message
    }



def analyze_disease(image, cnn_model, class_names):
    """CNN ile hastalık analizi"""
    
    if cnn_model is None or class_names is None:
        return None
    
    # Görüntü hazırlama
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Tahmin
    predictions = cnn_model.predict(img_batch, verbose=0)[0]
    top5_indices = np.argsort(predictions)[-5:][::-1]
    
    results = []
    for idx in top5_indices:
        full_name = class_names[idx]
        plant, disease = full_name.split('___') if '___' in full_name else (full_name, 'healthy')
        
        results.append({
            'plant': plant,
            'plant_tr': PLANT_TR.get(plant, plant),
            'disease': disease,
            'disease_tr': DISEASE_TR.get(disease, disease),
            'confidence': predictions[idx] * 100
        })
    
    return results



st.markdown('<div class="main-title">🌿 Bitki Hastalığı Tespit Sistemi</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Teachable Machine + CNN Hibrit Sistem</div>', unsafe_allow_html=True)



col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Fotoğraf Yükle")
    uploaded_file = st.file_uploader("Bitki fotoğrafı seçin", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="📸 Yüklenen Fotoğraf", use_container_width=True)
        
        if st.button("🔍 Analiz Et", type="primary", use_container_width=True):
            with st.spinner("🔄 Analiz ediliyor..."):
                
                # Adım 1: Leaf Detector
                st.markdown("### 1️⃣ Yaprak Tespiti")
                leaf_result = detect_leaf(image, models['leaf_detector'], models['leaf_labels'])
                
                # Detaylı sonuç göster
                if models['leaf_detector'] is not None:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("🌿 Yaprak Skoru", f"%{leaf_result['leaf_score']*100:.1f}")
                    with col_b:
                        st.metric("❌ Yaprak Değil Skoru", f"%{leaf_result['nonleaf_score']*100:.1f}")
                
                st.info(leaf_result['message'])
                
                # Adım 2: CNN Analizi
                if leaf_result['is_leaf']:
                    st.markdown("### 2️⃣ Hastalık Analizi")
                    top5 = analyze_disease(image, models['cnn'], models['class_names'])
                    
                    if top5:
                        best = top5[0]
                        
                        with col2:
                            st.markdown("### 📊 Analiz Sonuçları")
                            
                            # Ana sonuç kartı
                            st.markdown(f"""
                            <div class="success-card">
                                <h2 style="margin:0;">Tespit Sonuçları</h2>
                                <hr style="border-color: rgba(255,255,255,0.3);">
                                <h3>🌿 Bitki: {best['plant_tr']}</h3>
                                <h3>🦠 Durum: {best['disease_tr']}</h3>
                                <h3>📊 Güven: %{best['confidence']:.2f}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # TÜR HATASINI DÜZELTEN SKOR MANTIĞI
                            plant_scores = {}
                            disease_scores = {}
                            
                            for i, pred in enumerate(top5):
                                # Bitki için: İlk tahmini dominant yapar
                                weight = 1.0 if i == 0 else 0.05
                                p_name = pred['plant_tr']
                                plant_scores[p_name] = plant_scores.get(p_name, 0) + (pred['confidence'] * weight)
                                
                                # Hastalık için: Normal topla
                                d_name = pred['disease_tr']
                                disease_scores[d_name] = disease_scores.get(d_name, 0) + pred['confidence']
                            
                            # 2 Ayrı Tablo
                            sub1, sub2 = st.columns(2)
                            
                            with sub1:
                                st.markdown("### 🌿  Olası Bitki Türleri")
                                for p, s in sorted(plant_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
                                    st.write(f"**{p}:** %{min(s, 100):.1f}")
                                    st.progress(min(s/100, 1.0))
                            
                            with sub2:
                                st.markdown("### 🦠 Hastalıklar")
                                for d, s in sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
                                    st.write(f"**{d}:** %{min(s, 100):.1f}")
                                    st.progress(min(s/100, 1.0))
                            
                            # Detaylı tahminler
                            with st.expander("🔍 Detaylı Top 5"):
                                for i, pred in enumerate(top5, 1):
                                    st.write(f"{i}. {pred['plant_tr']} - {pred['disease_tr']} (%{pred['confidence']:.2f})")
                            
                            # Bilgi notu
                            st.markdown(f"""
                            <div class="info-box">
                                <strong>💡 Uzman Notu:</strong><br>
                                Görüntü <strong>{best['plant_tr']}</strong> bitkisi olarak belirlendi. 
                                Kesin teşhis için uzman görüşü alınız.
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        with col2:
                            st.error("❌ CNN analizi başarısız!")
                
                else:
                    # Yaprak tespit edilemedi
                    with col2:
                        st.markdown("### ❌ Sonuç")
                        st.markdown("""
                        <div class="error-card">
                            <h2>❌ Yaprak Tespit Edilemedi!</h2>
                            <hr style="border-color: rgba(255,255,255,0.3);">
                            <p>Lütfen daha net bir yaprak fotoğrafı yükleyin.</p>
                        </div>
                        """, unsafe_allow_html=True)

with col2:
    if not uploaded_file:
        st.markdown("### 📊 Analiz Sonuçları")
        st.info("👈 Fotoğraf yükleyin ve 'Analiz Et' butonuna basın")
        
        

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    🌿 Bitki Hastalığı Tespit Sistemi v6.3 | Teachable Machine + MobileNetV2
</div>
""", unsafe_allow_html=True)