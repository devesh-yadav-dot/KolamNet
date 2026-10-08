# 🪷 KolamNet

**Geometric Vision & Parametric Vector Studio** — a hybrid computational-topology and
generative-AI framework for preserving Kolam (rangoli) heritage.

Upload a photo or sketch of a Kolam and KolamNet will:

- **Extract** the structure with a classical CV pipeline (CLAHE → top-hat/Otsu → skeletonisation → pulli detection)
- **Measure** it (centroid, inertia tensor, principal axis, curvature, lattice pitch, 4-fold symmetry score)
- **Reconstruct** it either locally (medial-axis vectoriser) or with Gemini (clean Bézier SVG)
- **Play** with it: affine transforms, harmonic wave perturbation, C4 / D8 symmetry multiplication
- **Extrude** it into 3D (ribbon, wireframe, pillars) with Plotly
- **Complete** a single quadrant into a full pattern
- **Export** an archive JSON of the pulli coordinates

No image uploaded? It runs on a bundled demo Kolam.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Requires Python 3.11+.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app** → pick the repo, branch `main`, main file **`app.py`**.
3. Under **Advanced settings**, choose Python 3.11 or newer.
4. *(Optional)* paste secrets under **Advanced settings → Secrets** (see below).
5. Deploy.

### Gemini API key

The neural reconstruction needs a [Gemini API key](https://aistudio.google.com/apikey).

| Setup | Behaviour |
|---|---|
| **Nothing configured** (default) | Each visitor pastes their own key in the sidebar. Safe for a public app. |
| `GEMINI_API_KEY` in secrets | Used when the visitor leaves the field empty. ⚠️ On a public app, anyone can spend *your* quota. |
| `GEMINI_MODEL` in secrets | Overrides the model ID (default `gemini-3.6-flash`) without a code change. |

For local runs, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` (git-ignored).

## Project structure

```
app.py                      # the Streamlit app
requirements.txt
assets/
  default_kolam.png         # demo image
  background.mp4            # pre-compressed (≈1.6 MB) looping background
.streamlit/
  config.toml               # dark theme, upload limit
  secrets.toml.example
tests/test_smoke.py         # pytest + streamlit AppTest
```

## Notes

- `opencv-python-headless` is used on purpose: the regular `opencv-python` needs system GL libraries that Streamlit Cloud doesn't provide.
- The background video is inlined into the page as base64, so keep it small. `app.py` skips it if it exceeds 8 MB. To re-compress a new one:
  ```bash
  ffmpeg -i in.mp4 -vf "scale=854:-2,fps=20" -an -c:v libx264 -crf 36 -movflags +faststart assets/background.mp4
  ```
- LLM-generated SVG is sanitised and rendered with `st.html` (no script execution).

## Credits

Add attribution/licence details for the demo image and background video here before publishing.
