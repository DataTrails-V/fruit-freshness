import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- KONFIGURATION ---
# WICHTIG: Ersetze diese Liste mit der Ausgabe deines neuen Trainings-Skripts!
CLASS_NAMES = [
    'freshapples', 'freshbanana', 'freshcucumber', 'freshokra', 
    'freshoranges', 'freshpatato', 'freshtamto', 'rottenapples', 
    'rottenbanana', 'rottencucumber', 'rottenokra', 'rottenoranges', 
    'rottenpatato', 'rottentamto'
]

# --- MODELL LADEN ---
@st.cache_resource
def load_model():
    # Wir nutzen MobileNetV3 Small (identisch zum Training)
    model = models.mobilenet_v3_small(pretrained=False)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, len(CLASS_NAMES))
    
    # Gewichte laden
    model.load_state_dict(torch.load("obst_frische_modell.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# --- VORHERSAGE-FUNKTION ---
def predict(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        # Softmax hinzufügen, um Wahrscheinlichkeiten zu sehen (optional für später)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        conf, pred_idx = torch.max(probabilities, 0)
    return CLASS_NAMES[pred_idx.item()], conf.item()

# --- STREAMLIT UI ---
st.set_page_config(page_title="Frische-Check KI", page_icon="🍎")

st.title("🍎 KI Frische-Scanner")
st.write("Lade ein Foto von Obst oder Gemüse hoch, um die Frische zu prüfen.")

# Modell initialisieren
try:
    model = load_model()
except Exception as e:
    st.error("Fehler beim Laden des Modells. Hast du die neue .pth Datei hochgeladen?")
    st.stop()

uploaded_file = st.file_uploader("Bild auswählen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Bild laden und für KI vorbereiten
    image = Image.open(uploaded_file).convert('RGB')
    
    # 2. Bild anzeigen
    st.image(image, caption='Hochgeladenes Bild', use_container_width=True)
    
    # 3. Analyse-Button
    if st.button("Frische prüfen"):
        with st.spinner("🔍 Analysiere..."):
            label, confidence = predict(image, model)
            
            # Logik für Verdorben/Frisch
            is_rotten = "rotten" in label.lower() or "stale" in label.lower()
            
            # Namen der Frucht extrahieren
            fruit_name = label.replace("fresh", "").replace("rotten", "").replace("_", " ").capitalize()
            
            st.divider()
            
            # Ergebnis-Anzeige mit kleinerer Schrift (HTML)
            if is_rotten:
                st.error(f"Klassifizierung: {label} ({confidence:.1%})")
                st.markdown(f"<p style='font-size:20px; margin-bottom:0;'>🇩🇪 Ergebnis: {fruit_name} ist 🔴 <b>VERDORBEN</b></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:16px; color:gray;'>🇺🇸 Result: {fruit_name} is 🔴 ROTTEN</p>", unsafe_allow_html=True)
            else:
                st.success(f"Klassifizierung: {label} ({confidence:.1%})")
                st.markdown(f"<p style='font-size:20px; margin-bottom:0;'>🇩🇪 Ergebnis: {fruit_name} ist 🟢 <b>FRISCH</b></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:16px; color:gray;'>🇺🇸 Result: {fruit_name} is 🟢 FRESH</p>", unsafe_allow_html=True)

st.info("Das Modell wurde mit Transfer Learning auf Basis von MobileNetV3 trainiert.")