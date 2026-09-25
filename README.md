# AgriSense 3D: Agentic Crop Health Digital Twin (v2)

DESIGNED & DEVELOPED BY 
# NIKHIL CHARY SRIRAMOJU

## What changed in v2 (real-world upgrade)
- **Real CNN**: leaf photos are now classified using MobileNetV1 (a real pretrained convolutional network) via TensorFlow.js, run entirely in-browser.
- **Few-shot classification**: each uploaded photo is embedded and compared by cosine similarity to a few reference embeddings per disease class. A demo reference set is built in so the app works immediately; use "Calibrate with real photos" to upload your own labelled leaf photos and improve accuracy for real use.
- **Verified agent loop**: plan, retrieve (TF-IDF), verify the retrieved source actually names the diagnosed disease, retry with a narrower query if not, then act.
- **Explainer section** on the page itself: Problem, Approach, Honest limits, so anyone opening the link understands what's real vs. simulated.

## Path to production-grade accuracy
Few-shot matching against a handful of references is a legitimate technique but caps out well below a trained classifier. For real deployment:
1. Collect or use a labelled dataset (e.g. PlantVillage, ~54k images, 38 classes).
2. Fine-tune a CNN (MobileNet/EfficientNet) on it with Keras/PyTorch.
3. Export it and serve inference from the included FastAPI backend's `/scan` endpoint instead of doing few-shot matching in the browser.
4. Swap the frontend's MobileNet call for a fetch to your `/scan` API.
This repo's structure (Docker web + Docker api) is already set up for that swap.

## Run locally
```
docker compose up --build
```
Web: http://localhost:8080 — API docs: http://localhost:8000/docs

## GitHub Pages
Push to `main`, then Settings → Pages → Source: Deploy from a branch (main, root). Live at `https://<user>.github.io/<repo>/`.
The model loads from a CDN (jsdelivr), so it needs an internet connection.

## Honest limits
- Demo reference photos are synthetic; classification accuracy depends on the reference photos loaded (built-in demo set, or your own via Calibrate).
- Agronomic advice is generic — always confirm chemicals/doses with a local agronomist.
- The FastAPI backend mirrors the pipeline for a real deployment path but isn't wired to the frontend by default (static hosting on GitHub Pages can't call a backend without a public URL).
