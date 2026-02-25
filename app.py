import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- KONFIGURATION ---
# Die exakt gleiche Klassenliste wie in deinem Training
CLASS_NAMES = [
    'freshapples', 'freshbanana', 'freshcucumber', 'freshokra', 
    'freshoranges', 'freshpatato', 'freshtamto', 'rottenapples', 
    'rottenbanana', 'rottencucumber', 'rottenokra', 'rottenoranges', 
    'rottenpatato', 'rottentamto'
]

# --- MODELL LADEN ---
@st.cache_resource # Verhindert, dass das Modell bei jedem Klick neu geladen wird
def load_model():
    model = models.mobilenet_v3_small(pretrained=False)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, len(CLASS_NAMES))
    
    # Lade deine trainierten Gewichte
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
        _, pred_idx = torch.max(outputs, 1)
    return CLASS_NAMES[pred_idx.item()]

# --- STREAMLIT UI ---
st.set_page_config(page_title="Frische-Check KI", page_icon="🍎")

st.title("🍎 KI Frische-Scanner")
st.write("Lade ein Foto von Obst oder Gemüse hoch, um die Frische zu prüfen.")

model = load_model()

uploaded_file = st.file_uploader("Bild auswählen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild anzeigen
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Hochgeladenes Bild', use_container_width=True)
    
    st.write("🔍 Analysiere...")
    label = predict(image, model)
    
    # Ergebnis-Logik
    is_rotten = "rotten" in label.lower()
    
    if is_rotten:
        st.error(f"Ergebnis: {label}")
        st.subheader("🇩🇪 Ergebnis: Das Obst ist 🔴 VERDORBEN")
        st.subheader(f"🇺🇸 Result: The fruit is 🔴 ROTTEN")
    else:
        st.success(f"Ergebnis: {label}")
        st.subheader("🇩🇪 Ergebnis: Das Obst ist 🟢 FRISCH")
        st.subheader(f"🇺🇸 Result: The fruit is 🟢 FRESH")

st.info("Hinweis: Dieses Modell nutzt MobileNetV3 für effiziente Echtzeit-Analyse.")