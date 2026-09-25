"""AgriSense API: leaf scan (CNN-style feature pipeline) and RAG advice."""
import io, math, re
import numpy as np
from fastapi import FastAPI, UploadFile
from PIL import Image

app = FastAPI(title="AgriSense API")
DIS = ["Healthy", "Leaf rust", "Late blight", "Powdery mildew"]
KB = {
    "Healthy": "Keep weekly scouting, water at the base in the morning, balance nutrients.",
    "Leaf rust": "Remove infected leaves, apply a labelled triazole or strobilurin fungicide, avoid overhead irrigation.",
    "Late blight": "Remove infected plants, apply a copper-based protectant, keep leaves dry.",
    "Powdery mildew": "Improve airflow, avoid excess nitrogen, apply sulfur or potassium bicarbonate early.",
}
NOTE = "Confirm chemicals and doses with a local agronomist and follow the product label."

def conv3(img, k):
    h, w = img.shape
    out = np.zeros_like(img)
    for j in range(3):
        for i in range(3):
            out[1:-1, 1:-1] += k[j, i] * img[j:h-2+j, i:w-2+i]
    return np.maximum(out, 0)  # ReLU

def maxpool2(x):
    h, w = x.shape[0] // 2 * 2, x.shape[1] // 2 * 2
    return x[:h, :w].reshape(h // 2, 2, w // 2, 2).max(axis=(1, 3))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/scan")
async def scan(file: UploadFile):
    im = Image.open(io.BytesIO(await file.read())).convert("RGB").resize((48, 48))
    a = np.asarray(im, dtype=np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    green = np.clip((g - np.maximum(r, b)) / 64, 0, 1)
    lesion = green < 0.4
    f = float(lesion.mean())
    bright = float(a[lesion].mean()) if lesion.any() else 0.0
    rg = float(r[lesion].sum() / max(1.0, g[lesion].sum())) if lesion.any() else 0.0
    edges = maxpool2(conv3(green, np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=np.float32)))
    s = np.array([3 - f*80, (rg-1.2)*6 + f*10 - 1, (120-bright)/20 + f*15 - .5, (bright-150)/20 + f*10 - 1])
    p = np.exp(s - s.max()); p /= p.sum()
    i = int(p.argmax())
    return {"label": DIS[i], "confidence": round(float(p[i]), 3), "lesion_fraction": round(f, 3),
            "edge_energy": round(float(edges.sum()), 2), "advice": KB[DIS[i]], "note": NOTE}

@app.get("/ask")
def ask(q: str):
    words = set(re.findall(r"[a-z]+", q.lower()))
    scored = sorted(KB, key=lambda k: -len(words & set(re.findall(r"[a-z]+", (k + " " + KB[k]).lower()))))
    top = scored[0]
    return {"source": top, "answer": KB[top], "note": NOTE}
