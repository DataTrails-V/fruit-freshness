import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
import kagglehub
import ssl

# --- 0. SSL-FIX & FEHLER-MANAGEMENT ---
# Das hier umgeht den [SSL: CERTIFICATE_VERIFY_FAILED] Fehler
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. DATEN LADEN & PFAD-LOGIK ---
print("📥 Überprüfe Datensatz...")
base_path = kagglehub.dataset_download("swoyam2609/fresh-and-stale-classification")

# Wir graben jetzt tiefer, um die echten Klassenordner zu finden
def find_real_train_path(path):
    for root, dirs, files in os.walk(path):
        # Wenn ein Ordner 'fresh_apple' oder 'rotten_apple' heißt, sind wir richtig
        if any("fresh" in d.lower() or "rotten" in d.lower() for d in dirs):
            return root
    return path

train_path = find_real_train_path(base_path)
print(f"📂 Finaler Datenpfad gefunden: {train_path}")

# --- 2. PREPROCESSING ---
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

try:
    full_dataset = datasets.ImageFolder(train_path, data_transforms)
    class_names = full_dataset.classes
    print(f"🍎 Klassen erkannt ({len(class_names)}): {class_names}")
except Exception as e:
    print(f"❌ Fehler: Ordnerstruktur konnte nicht gelesen werden.")
    # Kleiner Tipp: Liste mal auf, was Python sieht
    if os.path.exists(train_path):
        print(f"Inhalt von {train_path}: {os.listdir(train_path)}")
    exit()

# RAM-Schutz (200 Bilder reichen für den Prototyp)
indices = torch.randperm(len(full_dataset))[:200]
train_subset = Subset(full_dataset, indices)
train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)

# --- 3. MODELL SETUP ---
print("🧠 Lade MobileNetV3 (Download startet jetzt)...")
model = models.mobilenet_v3_small(pretrained=True)
# ... restlicher Code wie vorher

# --- 3. MODELL SETUP (MobileNetV3 Small) ---
print("🧠 Initialisiere Modell...")
model = models.mobilenet_v3_small(pretrained=True)
num_ftrs = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_ftrs, len(class_names))

MODEL_FILE = "obst_frische_modell.pth"

# --- 4. TRAINING (FALLS KEIN MODELL GESPEICHERT) ---
if not os.path.exists(MODEL_FILE):
    print("🚀 Starte kurzes Training (20 Schritte für den Prototyp)...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if i % 5 == 0:
            print(f"   Batch {i} - Loss: {loss.item():.4f}")
        if i >= 19: break 
        
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"✅ Modell gespeichert als '{MODEL_FILE}'")
else:
    print(f"♻️ Lade gespeichertes Modell '{MODEL_FILE}'...")
    model.load_state_dict(torch.load(MODEL_FILE))

# --- 5. VORHERSAGE-TEST ---
def run_demo_prediction():
    model.eval()
    # Ein zufälliges Bild aus dem Datensatz ziehen
    img, label_idx = full_dataset[torch.randint(len(full_dataset), (1,)).item()]
    
    with torch.no_grad():
        output = model(img.unsqueeze(0))
        _, pred_idx = torch.max(output, 1)
    
    prediction = class_names[pred_idx]
    actual = class_names[label_idx]
    
    # Sprachausgabe Logik
    is_rotten = "rotten" in prediction.lower() or "stale" in prediction.lower()
    status_de = "🔴 VERDORBEN" if is_rotten else "🟢 FRISCH"
    status_en = "🔴 ROTTEN" if is_rotten else "🟢 FRESH"
    
    print("\n" + "="*40)
    print(f"KI-CHECK ERGEBNIS")
    print(f"Tatsächlich: {actual}")
    print(f"Erkannt:     {prediction}")
    print("-"*40)
    print(f"🇩🇪 DE: Das Obst ist {status_de}.")
    print(f"🇺🇸 EN: The fruit is {status_en}.")
    print("="*40)

# Ausführen
run_demo_prediction()