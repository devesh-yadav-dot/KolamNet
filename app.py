import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from PIL import Image
from skimage.morphology import skeletonize, remove_small_objects
from google import genai
from google.genai import types

import json
import re
import base64
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="KolamNet | Cultural Geometric Vision Engine",
    page_icon="🪷",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# DEFAULT FILES
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ASSETS_DIR = os.path.join(BASE_DIR, "assets")

DEFAULT_KOLAM = os.path.join(ASSETS_DIR, "default_kolam.png")
BACKGROUND_VIDEO = os.path.join(ASSETS_DIR, "background.mp4")

# Safety valve: the video is inlined as base64 into the page, so refuse
# anything large enough to make the app sluggish or exhaust Cloud memory.
MAX_BG_VIDEO_BYTES = 8 * 1024 * 1024


# ============================================================
# SECRETS / CONFIG HELPERS
# ============================================================

def get_secret(name: str, default: str = "") -> str:
    """Read from Streamlit secrets, then environment variables.

    st.secrets raises if no secrets.toml exists (e.g. local runs), so
    this never assumes it is present.
    """

    try:
        value = st.secrets.get(name)
    except Exception:
        value = None

    return str(value) if value else os.environ.get(name, default)


GEMINI_MODEL = get_secret("GEMINI_MODEL", "gemini-3.6-flash")


def sanitize_svg(svg_text: str) -> str:
    """Strip anything executable from LLM-generated SVG.

    Model output is untrusted (an uploaded image can carry prompt
    injection), so keep only the <svg>...</svg> block and drop scripts,
    foreignObject, inline event handlers and javascript: URLs.
    """

    if not isinstance(svg_text, str):
        return ""

    start = svg_text.lower().find("<svg")
    end = svg_text.lower().rfind("</svg>")

    if start == -1 or end == -1:
        return ""

    svg = svg_text[start:end + len("</svg>")]

    svg = re.sub(
        r"<\s*(script|foreignObject)\b.*?<\s*/\s*\1\s*>",
        "",
        svg,
        flags=re.IGNORECASE | re.DOTALL
    )

    svg = re.sub(
        r"\son\w+\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)",
        "",
        svg,
        flags=re.IGNORECASE
    )

    svg = re.sub(
        r"javascript\s*:",
        "",
        svg,
        flags=re.IGNORECASE
    )

    return svg


# ============================================================
# 1. LIGHTWEIGHT BACKGROUND VIDEO
# ============================================================

@st.cache_data(show_spinner=False)
def get_web_video_base64(video_path: str) -> str:
    """Return the (pre-compressed) background video as base64.

    The video in assets/ is already web-optimised, so no runtime
    transcoding (and no ffmpeg dependency) is needed on Streamlit Cloud.
    """

    if not os.path.exists(video_path):
        return ""

    if os.path.getsize(video_path) > MAX_BG_VIDEO_BYTES:
        return ""

    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def set_local_bg_video(
    video_path: str,
    overlay_opacity: float = 0.55
):

    if not os.path.exists(video_path):
        return

    b64_video = get_web_video_base64(video_path)

    if not b64_video:
        return

    bg_video_html = f"""
    <style>

    /* Global App Transparency & Typography */

    .stApp {{
        background: transparent !important;
        color: #E6EDF3 !important;
    }}

    header,
    footer {{
        background: transparent !important;
    }}

    h1,
    h2,
    h3,
    h4,
    h5,
    h6 {{
        color: #F0F6FC !important;
        font-weight: 600 !important;
    }}

    p,
    span,
    label,
    div {{
        color: #C9D1D9 !important;
    }}


    /* Glassmorphism Sidebar */

    [data-testid="stSidebar"] {{
        background-color: rgba(13, 17, 23, 0.78) !important;
        backdrop-filter: none !important;
        border-right: 1px solid rgba(255, 255, 255, 0.12);
    }}


    /* Luminous Tab Headers */

    .stTabs [data-baseweb="tab-list"] {{
        background-color: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 5px;
        backdrop-filter: none;
    }}

    .stTabs [data-baseweb="tab"] {{
        color: #E6EDF3 !important;
        font-weight: 500;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: rgba(255, 255, 255, 0.18) !important;
        border-radius: 8px;
        color: #FFFFFF !important;
        font-weight: 700;
    }}


    /* Metric Readability */

    [data-testid="stMetricValue"] {{
        color: #58FFAA !important;
        font-weight: bold !important;
    }}

    [data-testid="stMetricLabel"] {{
        color: #8B949E !important;
    }}


    /* Background Video */

    #bg-video {{
        position: fixed;
        right: 0;
        bottom: 0;
        width: 100vw;
        height: 100vh;
        object-fit: cover;
        will-change: transform;
        z-index: -2;
    }}

    #video-overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background-color: rgba(6, 10, 18, {overlay_opacity});
        z-index: -1;
    }}

    </style>

    <video
        autoplay
        loop
        muted
        playsinline
        preload="auto"
        id="bg-video"
    >
        <source
            src="data:video/mp4;base64,{b64_video}"
            type="video/mp4"
        >
    </video>

    <div id="video-overlay"></div>
    """

    st.html(bg_video_html)


# Activate background video
set_local_bg_video(
    BACKGROUND_VIDEO,
    overlay_opacity=0.55
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "🪷 KolamNet: Geometric Vision & Parametric Vector Studio"
)

st.caption(
    "A Hybrid Computational Topology & Generative AI "
    "Framework for Cultural Heritage Preservation"
)


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

with st.sidebar:

    st.header("⚙️ System Configuration")

    user_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help=(
            "Required for neural generative reconstruction. "
            "Used only for your session and never stored."
        )
    )

    # A key typed by the visitor wins; otherwise fall back to a key
    # configured in Streamlit secrets / environment (optional).
    api_key = user_api_key or get_secret("GEMINI_API_KEY")

    st.markdown("---")

    st.subheader("🔧 Preprocessing Controls")

    preset_mode = st.radio(
        "Pipeline Mode",
        [
            "📸 Real-World Chalk Photo",
            "🎨 Clean Digital Vector / Sketch"
        ]
    )

    clahe_clip = st.slider(
        "CLAHE Contrast Limit",
        1.0,
        6.0,
        3.5,
        0.5
    )

    noise_cutoff = st.slider(
        "Skeleton Denoise Min Pixels",
        10,
        150,
        45
    )

    st.markdown("---")

    st.subheader("🎨 Default Visualizer Styles")

    svg_stroke_color = st.color_picker(
        "Stroke Color",
        "#58FFAA"
    )

    svg_stroke_width = st.slider(
        "Line Thickness",
        1.0,
        6.0,
        2.4,
        0.5
    )

    svg_dot_color = st.color_picker(
        "Pulli Dot Color",
        "#FFFFFF"
    )

    canvas_transparency = st.slider(
        "Plot Panel Alpha",
        0.0,
        1.0,
        0.25,
        0.05
    )


# ============================================================
# MAIN TABS
# ============================================================

(
    tab_analysis,
    tab_playground,
    tab_3d,
    tab_inpaint,
    tab_procedural,
    tab_archive
) = st.tabs(
    [
        "🔍 Vision Pipeline & Reconstructors",
        "🎮 Parametric Vector Playground",
        "🧊 3D Structural Extruder",
        "🧩 Quarter-to-Full Inpainter",
        "♾️ Infinite Procedural Synthesizer",
        "📜 Cultural Grammar & Archive"
    ]
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Kolam Image "
    "(Smartphone Photo, Chalk on Floor, or Digital Sketch)",
    type=["jpg", "png", "jpeg"]
)


# ============================================================
# IMAGE SELECTION LOGIC
# ============================================================

if uploaded_file is not None:

    # User uploaded an image
    file_bytes = uploaded_file.getvalue()

    image_source = uploaded_file.name

    using_default = False

else:

    # No upload -> use bundled demo image
    if os.path.exists(DEFAULT_KOLAM):

        with open(DEFAULT_KOLAM, "rb") as f:
            file_bytes = f.read()

        image_source = "default_kolam.png"

        using_default = True

    else:

        file_bytes = None

        image_source = None

        using_default = False


# ============================================================
# HELPER FOR TRANSLUCENT MATPLOTLIB PLOTS
# ============================================================

def create_glass_figure(figsize=(6, 6)):

    fig, ax = plt.subplots(
        figsize=figsize
    )

    fig.patch.set_facecolor(
        "#0d1117"
    )

    fig.patch.set_alpha(
        canvas_transparency
    )

    ax.set_facecolor(
        "#0d1117"
    )

    ax.set_alpha(
        canvas_transparency
    )

    return fig, ax


# ============================================================
# 2. CACHED COMPUTER VISION PIPELINE
# ============================================================

@st.cache_data(show_spinner=False)
def run_vision_pipeline(
    file_bytes: bytes,
    mode: str,
    clip_limit: float,
    min_noise: int
):

    nparr = np.frombuffer(
        file_bytes,
        np.uint8
    )

    img_bgr = cv2.imdecode(
        nparr,
        cv2.IMREAD_COLOR
    )

    if img_bgr is None:
        raise ValueError(
            "Could not decode the supplied image."
        )

    max_dim = 1200

    original_h, original_w = img_bgr.shape[:2]

    if max(
        original_h,
        original_w
    ) > max_dim:

        scale = (
            max_dim
            / max(original_h, original_w)
        )

        img_bgr = cv2.resize(
            img_bgr,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

    h, w, _ = img_bgr.shape

    # --------------------------------------------------------
    # Stage 1: CLAHE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(8, 8)
    )

    enhanced_gray = clahe.apply(
        gray
    )

    # --------------------------------------------------------
    # Stage 2: Feature Filtering
    # --------------------------------------------------------

    if "Real-World" in mode:

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (13, 13)
        )

        bright_features = cv2.morphologyEx(
            enhanced_gray,
            cv2.MORPH_TOPHAT,
            kernel
        )

        _, binary = cv2.threshold(
            bright_features,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    else:

        if np.mean(enhanced_gray) > 127:

            enhanced_gray = cv2.bitwise_not(
                enhanced_gray
            )

        _, binary = cv2.threshold(
            enhanced_gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    # --------------------------------------------------------
    # Stage 3: Pulli Centroids
    # --------------------------------------------------------

    dist = cv2.distanceTransform(
        binary,
        cv2.DIST_L2,
        5
    )

    threshold_value = (
        0.40 * dist.max()
        if dist.max() > 0
        else 0
    )

    _, dot_peaks = cv2.threshold(
        dist,
        threshold_value,
        255,
        0
    )

    num_labels, _, stats, centroids = (
        cv2.connectedComponentsWithStats(
            np.uint8(dot_peaks)
        )
    )

    dots = []

    for i in range(1, num_labels):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        if 4 <= area <= 1200:

            dots.append(
                [
                    float(centroids[i][0]),
                    float(centroids[i][1])
                ]
            )

    # --------------------------------------------------------
    # Stage 4: Skeletonization
    # --------------------------------------------------------

    try:
        # scikit-image >= 0.26 (min_size is deprecated; max_size removes
        # objects <= its value, so min_noise - 1 keeps the old behavior).
        clean_lines = remove_small_objects(
            binary > 0,
            max_size=min_noise - 1
        )

    except TypeError:
        # scikit-image < 0.26
        clean_lines = remove_small_objects(
            binary > 0,
            min_size=min_noise
        )

    skeleton = skeletonize(
        clean_lines
    )

    skeleton_uint8 = (
        skeleton * 255
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Stage 5: Rotational Symmetry
    # --------------------------------------------------------

    sq_size = 512

    sq_skel = cv2.resize(
        skeleton_uint8,
        (sq_size, sq_size),
        interpolation=cv2.INTER_NEAREST
    )

    rot90 = np.rot90(
        sq_skel
    )

    rotational_score = (
        np.mean(
            sq_skel == rot90
        ) * 100
    )

    # --------------------------------------------------------
    # Stage 6: Extract Paths
    # --------------------------------------------------------

    path_contours, _ = cv2.findContours(
        skeleton_uint8,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE
    )

    extracted_paths = []

    for cnt in path_contours:

        if len(cnt) >= (
            min_noise // 3
        ):

            pts = cnt.squeeze()

            if len(pts.shape) == 2:

                extracted_paths.append(
                    pts.astype(np.float32)
                )

    return {

        "enhanced": enhanced_gray,

        "binary": binary,

        "skeleton": skeleton_uint8,

        "dots": (
            np.array(dots)
            if len(dots) > 0
            else np.empty((0, 2))
        ),

        "symmetry_score": rotational_score,

        "paths": extracted_paths,

        "h": h,

        "w": w,

        "raw_rgb": cv2.cvtColor(
            img_bgr,
            cv2.COLOR_BGR2RGB
        )
    }


# ============================================================
# 3. DIFFERENTIAL GEOMETRY
# ============================================================

def compute_differential_geometry(
    paths,
    dots,
    width,
    height
):

    total_vertices = sum(
        len(p)
        for p in paths
    )

    if total_vertices == 0:
        return {}

    # --------------------------------------------------------
    # Global Centroid
    # --------------------------------------------------------

    all_pts = np.vstack(
        paths
    )

    cx = np.mean(
        all_pts[:, 0]
    )

    cy = np.mean(
        all_pts[:, 1]
    )

    # --------------------------------------------------------
    # Second-order central moments
    # --------------------------------------------------------

    mu_20 = np.mean(
        (all_pts[:, 0] - cx) ** 2
    )

    mu_02 = np.mean(
        (all_pts[:, 1] - cy) ** 2
    )

    mu_11 = np.mean(
        (all_pts[:, 0] - cx)
        *
        (all_pts[:, 1] - cy)
    )

    # --------------------------------------------------------
    # Principal Axis
    # --------------------------------------------------------

    theta_rad = (
        0.5
        *
        np.arctan2(
            2 * mu_11,
            mu_20 - mu_02
        )
    )

    theta_deg = np.degrees(
        theta_rad
    )

    # --------------------------------------------------------
    # Curvature
    # --------------------------------------------------------

    longest_path = max(
        paths,
        key=len
    )

    dx = np.gradient(
        longest_path[:, 0]
    )

    dy = np.gradient(
        longest_path[:, 1]
    )

    ddx = np.gradient(
        dx
    )

    ddy = np.gradient(
        dy
    )

    numerator = np.abs(
        dx * ddy
        -
        dy * ddx
    )

    denominator = (
        (dx ** 2 + dy ** 2) ** 1.5
        + 1e-8
    )

    curvatures = (
        numerator
        /
        denominator
    )

    # --------------------------------------------------------
    # Pulli Lattice Pitch
    # --------------------------------------------------------

    avg_pitch = 0.0

    if len(dots) >= 2:

        from scipy.spatial.distance import (
            pdist,
            squareform
        )

        d_matrix = squareform(
            pdist(dots)
        )

        np.fill_diagonal(
            d_matrix,
            np.inf
        )

        nearest_distances = np.min(
            d_matrix,
            axis=1
        )

        avg_pitch = float(
            np.median(
                nearest_distances
            )
        )

    return {

        "centroid": (
            float(cx),
            float(cy)
        ),

        "inertia_tensor": [
            [
                float(mu_20),
                float(mu_11)
            ],
            [
                float(mu_11),
                float(mu_02)
            ]
        ],

        "orientation_deg": float(
            theta_deg
        ),

        "total_vertices": (
            total_vertices
        ),

        "sample_path": longest_path,

        "tangents": np.column_stack(
            [dx, dy]
        ),

        "curvatures": curvatures,

        "lattice_pitch_px": avg_pitch
    }


# ============================================================
# 4. VECTOR TRANSFORMATION ENGINE
# ============================================================

def transform_vector_set(
    paths,
    dots,
    center,
    angle_deg,
    scale_x,
    scale_y,
    shear_x,
    ripple_amp,
    ripple_freq
):

    cx, cy = center

    rad = np.radians(
        angle_deg
    )

    cos_a = np.cos(rad)
    sin_a = np.sin(rad)

    rot_mat = np.array(
        [
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ]
    )

    shear_mat = np.array(
        [
            [1.0, shear_x],
            [0.0, 1.0]
        ]
    )

    scale_mat = np.array(
        [
            [scale_x, 0.0],
            [0.0, scale_y]
        ]
    )

    affine_mat = (
        rot_mat
        @ shear_mat
        @ scale_mat
    )

    def process_pts(pts):

        if len(pts) == 0:
            return pts

        centered = (
            pts
            -
            np.array([cx, cy])
        )

        if ripple_amp > 0:

            centered[:, 0] += (
                ripple_amp
                *
                np.sin(
                    centered[:, 1]
                    *
                    ripple_freq
                )
            )

            centered[:, 1] += (
                ripple_amp
                *
                np.cos(
                    centered[:, 0]
                    *
                    ripple_freq
                )
            )

        transformed = (
            affine_mat
            @ centered.T
        ).T

        return (
            transformed
            +
            np.array([cx, cy])
        )

    transformed_paths = [
        process_pts(
            p.copy()
        )
        for p in paths
    ]

    transformed_dots = (
        process_pts(
            dots.copy()
        )
        if len(dots) > 0
        else dots
    )

    return (
        transformed_paths,
        transformed_dots
    )


# ============================================================
# 5. 3D STRUCTURAL EXTRUDER
# ============================================================

def render_3d_kolam(
    paths,
    dots,
    width,
    height,
    mode="Ribbon Surface",
    extrusion_height=40.0,
    stroke_color="#58FFAA",
    dot_color="#FFFFFF"
):

    fig = go.Figure()

    for pts in paths:

        if len(pts) < 2:
            continue

        x = pts[:, 0]

        y = height - pts[:, 1]

        n_pts = len(x)

        # ----------------------------------------------------
        # Ribbon Surface
        # ----------------------------------------------------

        if mode == "Ribbon Surface":

            mesh_x = np.concatenate(
                [x, x]
            )

            mesh_y = np.concatenate(
                [y, y]
            )

            mesh_z = np.concatenate(
                [
                    np.zeros(n_pts),
                    np.full(
                        n_pts,
                        extrusion_height
                    )
                ]
            )

            i_idx = []
            j_idx = []
            k_idx = []

            for idx in range(
                n_pts - 1
            ):

                i_idx.append(idx)

                j_idx.append(
                    idx + n_pts
                )

                k_idx.append(
                    idx + 1
                )

                i_idx.append(
                    idx + n_pts
                )

                j_idx.append(
                    idx + n_pts + 1
                )

                k_idx.append(
                    idx + 1
                )

            fig.add_trace(
                go.Mesh3d(
                    x=mesh_x,
                    y=mesh_y,
                    z=mesh_z,
                    i=i_idx,
                    j=j_idx,
                    k=k_idx,
                    color=stroke_color,
                    opacity=0.85,
                    flatshading=True,
                    showlegend=False
                )
            )

            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=np.full(
                        n_pts,
                        extrusion_height
                    ),
                    mode="lines",
                    line=dict(
                        color=stroke_color,
                        width=3
                    ),
                    showlegend=False
                )
            )

        # ----------------------------------------------------
        # Wireframe
        # ----------------------------------------------------

        elif mode == "Wireframe Skeleton":

            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=np.zeros(n_pts),
                    mode="lines",
                    line=dict(
                        color="#38BDF8",
                        width=2
                    ),
                    showlegend=False
                )
            )

            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=np.full(
                        n_pts,
                        extrusion_height
                    ),
                    mode="lines",
                    line=dict(
                        color=stroke_color,
                        width=3
                    ),
                    showlegend=False
                )
            )

            step = max(
                1,
                n_pts // 12
            )

            for xi, yi in zip(
                x[::step],
                y[::step]
            ):

                fig.add_trace(
                    go.Scatter3d(
                        x=[xi, xi],
                        y=[yi, yi],
                        z=[
                            0,
                            extrusion_height
                        ],
                        mode="lines",
                        line=dict(
                            color="#A855F7",
                            width=1.5
                        ),
                        showlegend=False
                    )
                )

        # ----------------------------------------------------
        # Pillars
        # ----------------------------------------------------

        elif mode == "3D Pillars (Vertical Struts)":

            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=np.full(
                        n_pts,
                        extrusion_height
                    ),
                    mode="lines",
                    line=dict(
                        color=stroke_color,
                        width=4
                    ),
                    showlegend=False
                )
            )

            step = max(
                1,
                n_pts // 8
            )

            for xi, yi in zip(
                x[::step],
                y[::step]
            ):

                fig.add_trace(
                    go.Scatter3d(
                        x=[xi, xi],
                        y=[yi, yi],
                        z=[
                            0,
                            extrusion_height
                        ],
                        mode="lines",
                        line=dict(
                            color="#F43F5E",
                            width=2
                        ),
                        showlegend=False
                    )
                )

    # --------------------------------------------------------
    # Pulli Dots
    # --------------------------------------------------------

    if len(dots) > 0:

        dx = dots[:, 0]

        dy = height - dots[:, 1]

        fig.add_trace(
            go.Scatter3d(
                x=dx,
                y=dy,
                z=np.zeros(len(dx)),
                mode="markers",
                marker=dict(
                    size=4,
                    color=dot_color,
                    symbol="circle",
                    opacity=0.9
                ),
                name="Pulli Dots"
            )
        )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    fig.update_layout(

        template="plotly_dark",

        paper_bgcolor=(
            "rgba(13, 17, 23, 0.4)"
        ),

        scene=dict(

            xaxis=dict(
                showbackground=False,
                showticklabels=False,
                title=""
            ),

            yaxis=dict(
                showbackground=False,
                showticklabels=False,
                title=""
            ),

            zaxis=dict(
                showbackground=False,
                showticklabels=False,
                title="",
                range=[
                    -5,
                    extrusion_height * 1.5
                ]
            ),

            camera=dict(
                eye=dict(
                    x=1.2,
                    y=-1.4,
                    z=1.1
                )
            )
        ),

        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),

        height=620
    )

    return fig


# ============================================================
# 6. QUARTER-TO-FULL INPAINTER
# ============================================================

def complete_quadrant_kolam(
    paths,
    dots,
    width,
    height,
    corner_source="Top-Left",
    overlap_x=0.0,
    overlap_y=0.0
):

    if len(paths) == 0:

        return (
            [],
            dots,
            width,
            height
        )

    # --------------------------------------------------------
    # Normalize to Top-Left
    # --------------------------------------------------------

    norm_paths = []

    for p in paths:

        x = p[:, 0].copy()

        y = p[:, 1].copy()

        if "Right" in corner_source:

            x = width - x

        if "Bottom" in corner_source:

            y = height - y

        norm_paths.append(
            np.column_stack(
                [x, y]
            )
        )

    # --------------------------------------------------------
    # Normalize dots
    # --------------------------------------------------------

    norm_dots = np.empty(
        (0, 2)
    )

    if len(dots) > 0:

        dx = dots[:, 0].copy()

        dy = dots[:, 1].copy()

        if "Right" in corner_source:

            dx = width - dx

        if "Bottom" in corner_source:

            dy = height - dy

        norm_dots = np.column_stack(
            [dx, dy]
        )

    # --------------------------------------------------------
    # Effective seams
    # --------------------------------------------------------

    effective_w = max(
        1.0,
        width - overlap_x
    )

    effective_h = max(
        1.0,
        height - overlap_y
    )

    full_paths = []

    full_dots = []

    configs = [

        (False, False),

        (True, False),

        (False, True),

        (True, True)
    ]

    # --------------------------------------------------------
    # Mirror quadrants
    # --------------------------------------------------------

    for flip_x, flip_y in configs:

        for p in norm_paths:

            px = (
                2 * effective_w - p[:, 0]
                if flip_x
                else p[:, 0]
            )

            py = (
                2 * effective_h - p[:, 1]
                if flip_y
                else p[:, 1]
            )

            full_paths.append(
                np.column_stack(
                    [px, py]
                )
            )

        if len(norm_dots) > 0:

            pdx = (
                2 * effective_w - norm_dots[:, 0]
                if flip_x
                else norm_dots[:, 0]
            )

            pdy = (
                2 * effective_h - norm_dots[:, 1]
                if flip_y
                else norm_dots[:, 1]
            )

            full_dots.append(
                np.column_stack(
                    [pdx, pdy]
                )
            )

    merged_dots = (
        np.vstack(full_dots)
        if len(full_dots) > 0
        else np.empty((0, 2))
    )

    if len(merged_dots) > 0:

        merged_dots = np.unique(
            np.round(
                merged_dots / 4.0
            ) * 4.0,
            axis=0
        )

    total_w = 2 * effective_w

    total_h = 2 * effective_h

    return (
        full_paths,
        merged_dots,
        total_w,
        total_h
    )


# ============================================================
# MAIN WORKFLOW
# ============================================================

if file_bytes is not None:

    # --------------------------------------------------------
    # Run CV pipeline
    # --------------------------------------------------------

    cv_data = run_vision_pipeline(
        file_bytes,
        preset_mode,
        clahe_clip,
        noise_cutoff
    )

    # --------------------------------------------------------
    # Active image indicator
    # --------------------------------------------------------

    if using_default:

        st.success(
            "🪷 Demo mode active — "
            "KolamNet is using `default_kolam.png`. "
            "Upload another image above to replace it."
        )

    else:

        st.info(
            f"📷 Processing uploaded image: "
            f"`{image_source}`"
        )


    # ========================================================
    # TAB 1
    # ========================================================

    with tab_analysis:

        st.subheader(
            "1. Multi-Stage Computer Vision Decomposition"
        )

        v_col1, v_col2, v_col3, v_col4 = (
            st.columns(4)
        )

        with v_col1:

            st.image(
                cv_data["raw_rgb"],
                caption="1. Raw Input Photo",
                width="stretch",
                output_format="JPEG"
            )

        with v_col2:

            st.image(
                cv_data["enhanced"],
                caption="2. CLAHE Normalized",
                width="stretch",
                output_format="JPEG"
            )

        with v_col3:

            st.image(
                cv_data["binary"],
                caption="3. Isolated Feature Mask",
                width="stretch",
                output_format="JPEG"
            )

        with v_col4:

            st.image(
                cv_data["skeleton"],
                caption="4. 1-Pixel Medial Skeleton",
                width="stretch",
                output_format="JPEG"
            )

        st.markdown("---")

        # ----------------------------------------------------
        # Differential Geometry
        # ----------------------------------------------------

        st.subheader(
            "2. Topological & Differential Vector Metrics"
        )

        st.caption(
            "Quantitative continuous-space derivations "
            "from discrete medial skeleton graphs."
        )

        math_data = compute_differential_geometry(
            cv_data["paths"],
            cv_data["dots"],
            cv_data["w"],
            cv_data["h"]
        )

        m_col1, m_col2, m_col3, m_col4 = (
            st.columns(4)
        )

        m_col1.metric(
            "Total Extracted Vertices",
            f"{math_data.get('total_vertices', 0):,}"
        )

        centroid = math_data.get(
            "centroid",
            (0, 0)
        )

        m_col2.metric(
            "Centroid (x̄, ȳ)",
            f"({centroid[0]:.1f}, {centroid[1]:.1f})"
        )

        m_col3.metric(
            "Principal Inertia Angle (θ)",
            f"{math_data.get('orientation_deg', 0.0):.2f}°"
        )

        m_col4.metric(
            "Avg Pulli Lattice Pitch",
            f"{math_data.get('lattice_pitch_px', 0.0):.1f} px"
        )

        # ----------------------------------------------------
        # Mathematical Derivations
        # ----------------------------------------------------

        with st.expander(
            "📐 View Computational Differential Geometry Derivations",
            expanded=False
        ):

            st.markdown(
                r"""
                **1. Discrete Curve Curvature (κ) Along Medial Path:**

                $$\kappa(s) =
                \frac{|\dot{x}\ddot{y}-\dot{y}\ddot{x}|}
                {(\dot{x}^2+\dot{y}^2)^{3/2}}$$

                **2. Second-Order Central Moment & Inertia Tensor (J):**

                $$J =
                \begin{bmatrix}
                \mu_{20} & \mu_{11}\\
                \mu_{11} & \mu_{02}
                \end{bmatrix}$$

                $$\theta =
                \frac{1}{2}
                \text{atan2}
                (2\mu_{11},\mu_{20}-\mu_{02})$$
                """
            )

            if math_data:

                it = math_data[
                    "inertia_tensor"
                ]

                st.code(
                    f"""
Inertia Tensor Matrix J =

[[{it[0][0]:.4f}, {it[0][1]:.4f}],
 [{it[1][0]:.4f}, {it[1][1]:.4f}]]
""",
                    language="text"
                )

        # ----------------------------------------------------
        # Tangent / Curvature Inspector
        # ----------------------------------------------------

        if (
            math_data
            and len(
                math_data.get(
                    "sample_path",
                    []
                )
            ) > 0
        ):

            st.markdown(
                "**🔬 Point-by-Point Tangent & "
                "Curvature Inspector**"
            )

            p_sample = math_data[
                "sample_path"
            ]

            t_sample = math_data[
                "tangents"
            ]

            k_sample = math_data[
                "curvatures"
            ]

            idx_slider = st.slider(
                "Inspect Path Node Index (k)",
                0,
                len(p_sample) - 1,
                len(p_sample) // 2
            )

            cur_pt = p_sample[
                idx_slider
            ]

            cur_tan = t_sample[
                idx_slider
            ]

            cur_k = k_sample[
                idx_slider
            ]

            c_data1, c_data2, c_data3 = (
                st.columns(3)
            )

            c_data1.latex(
                rf"\vec{{P}}_{{{idx_slider}}}"
                rf" = ({cur_pt[0]:.2f}, {cur_pt[1]:.2f})"
            )

            c_data2.latex(
                rf"\vec{{T}}_{{{idx_slider}}}"
                rf" = ({cur_tan[0]:.2f}, {cur_tan[1]:.2f})"
            )

            c_data3.latex(
                rf"\kappa_{{{idx_slider}}}"
                rf" = {cur_k:.5f}\ "
                rf"\text{{px}}^{{-1}}"
            )

        st.markdown("---")

        # ----------------------------------------------------
        # Dual Reconstruction
        # ----------------------------------------------------

        st.subheader(
            "3. Dual-Engine Reconstruction"
        )

        r_col1, r_col2 = (
            st.columns(2)
        )

        # ----------------------------------------------------
        # Engine A
        # ----------------------------------------------------

        with r_col1:

            st.markdown(
                "#### ⚡ Engine A: "
                "Local Algorithmic Vectorizer"
            )

            st.caption(
                "Zero-cloud deterministic "
                "Medial Axis contour tracing."
            )

            fig, ax = create_glass_figure(
                figsize=(6, 6)
            )

            for pts in cv_data["paths"]:

                ax.plot(
                    pts[:, 0],
                    cv_data["h"] - pts[:, 1],
                    color=svg_stroke_color,
                    lw=svg_stroke_width,
                    solid_capstyle="round"
                )

            if len(
                cv_data["dots"]
            ) > 0:

                ax.scatter(
                    cv_data["dots"][:, 0],
                    cv_data["h"]
                    - cv_data["dots"][:, 1],
                    color=svg_dot_color,
                    s=26,
                    zorder=5,
                    edgecolors="#A5D6FF",
                    linewidths=0.8
                )

            ax.set_xlim(
                0,
                cv_data["w"]
            )

            ax.set_ylim(
                0,
                cv_data["h"]
            )

            ax.set_aspect(
                "equal"
            )

            ax.axis("off")

            st.pyplot(
                fig
            )

            plt.close(fig)

            m1, m2 = st.columns(2)

            m1.metric(
                "Dots Extracted",
                len(
                    cv_data["dots"]
                )
            )

            m2.metric(
                "4-Fold Symmetry Match",
                f"{cv_data['symmetry_score']:.1f}%"
            )

        # ----------------------------------------------------
        # Engine B
        # ----------------------------------------------------

        with r_col2:

            st.markdown(
                "#### 🧠 Engine B: "
                "Neural Geometric Reconstructor (Gemini)"
            )

            st.caption(
                "Semantic topology reasoning & "
                "clean Bézier SVG synthesis."
            )

            if st.button(
                "🚀 Run Neural Generative Reconstruction"
            ):

                if not api_key:

                    st.error(
                        "Please enter a Gemini API Key "
                        "in the sidebar."
                    )

                else:

                    client = genai.Client(
                        api_key=api_key
                    )

                    pil_enhanced = Image.fromarray(
                        cv_data["enhanced"]
                    )

                    with st.spinner(
                        "Compiling geometric grammar "
                        "and generating SVG..."
                    ):

                        prompt = f"""
You are an expert computational geometry
and traditional Indian Kolam art analyzer.

You are given a CLAHE-enhanced image of a Kolam.

Extract topological parameters and generate clean SVG.

1. Lattice Grid dimensions
   (e.g. 5x5, 7x7 diamond).

2. Symmetry order
   (e.g. 4-fold rotational C4,
   D4 dihedral, radial).

3. Synthesize a clean,
   self-contained SVG:

   - viewBox="0 0 500 500"
   - Background transparent
   - Dots: {svg_dot_color}
   - radius r="3.5"
   - Lines/Loops: {svg_stroke_color}
   - stroke-width="{svg_stroke_width}"
   - fill="none"
   - stroke-linecap="round"
   - Use smooth continuous Bézier loops
     wrapping properly around the dots.

Return strictly valid JSON:

{{
    "dot_grid": "string description",
    "symmetry_type": "string description",
    "estimated_dots_count": int,
    "structural_rules": [
        "rule 1",
        "rule 2"
    ],
    "svg_code": "<svg ...></svg>"
}}
"""

                        try:

                            response = (
                                client.models.generate_content(
                                    model=GEMINI_MODEL,
                                    contents=[
                                        pil_enhanced,
                                        prompt
                                    ],
                                    config=types.GenerateContentConfig(
                                        response_mime_type=(
                                            "application/json"
                                        )
                                    )
                                )
                            )

                            raw_text = (
                                response.text.strip()
                            )

                            # Remove Markdown fences
                            raw_text = re.sub(
                                r"^```(?:json)?\s*",
                                "",
                                raw_text,
                                flags=re.IGNORECASE
                            )

                            raw_text = re.sub(
                                r"\s*```$",
                                "",
                                raw_text
                            )

                            parsed = json.loads(
                                raw_text
                            )

                            st.session_state[
                                "gemini_output"
                            ] = parsed

                        except Exception as e:

                            st.error(
                                f"Neural reconstruction error: {e}"
                            )

            # ------------------------------------------------
            # Display Gemini result
            # ------------------------------------------------

            if (
                "gemini_output"
                in st.session_state
            ):

                data = st.session_state[
                    "gemini_output"
                ]

                svg_code = sanitize_svg(
                    data.get(
                        "svg_code",
                        ""
                    )
                )

                if svg_code:

                    # st.html sanitizes markup (no <script>), unlike the
                    # iframe-based components.html which would execute it.
                    st.html(
                        f'<div style="max-width:100%;'
                        f'max-height:350px;display:flex;'
                        f'justify-content:center;">'
                        f'{svg_code}</div>'
                    )

                d1, d2 = st.columns(2)

                d1.metric(
                    "Lattice Grid",
                    data.get(
                        "dot_grid",
                        "N/A"
                    )
                )

                d2.metric(
                    "Symmetry Group",
                    data.get(
                        "symmetry_type",
                        "N/A"
                    )
                )

                if svg_code:

                    st.download_button(
                        "💾 Download Scalable Vector (.SVG)",
                        svg_code,
                        file_name="reconstructed_kolam.svg",
                        mime="image/svg+xml"
                    )


    # ========================================================
    # TAB 2: PARAMETRIC VECTOR PLAYGROUND
    # ========================================================

    with tab_playground:

        st.subheader(
            "🎮 Interactive Parametric Vector Studio"
        )

        st.caption(
            "Apply real-time linear algebra, "
            "wave harmonics, and symmetry operations "
            "directly onto extracted vector vertices."
        )

        p_ctrl, p_canvas = st.columns(
            [1, 2]
        )

        with p_ctrl:

            st.markdown(
                "**🎛️ Geometric Transforms**"
            )

            p_rot = st.slider(
                "Continuous Rotation (°)",
                0,
                360,
                0,
                5
            )

            p_scale_x = st.slider(
                "Scale X (Aspect Morph)",
                0.4,
                2.0,
                1.0,
                0.05
            )

            p_scale_y = st.slider(
                "Scale Y (Aspect Morph)",
                0.4,
                2.0,
                1.0,
                0.05
            )

            p_shear = st.slider(
                "Isometric Shear (Skew)",
                -1.0,
                1.0,
                0.0,
                0.05
            )

            st.markdown(
                "**🌊 Harmonic Wave Perturbation**"
            )

            p_ripple_amp = st.slider(
                "Wave Amplitude (Organic Flow)",
                0.0,
                25.0,
                0.0,
                0.5
            )

            p_ripple_freq = st.slider(
                "Wave Frequency",
                0.01,
                0.20,
                0.05,
                0.01
            )

            st.markdown(
                "**✨ Symmetry Multiplication**"
            )

            sym_multi = st.radio(
                "Symmetry Multiplier",
                [
                    "Original Extracted",
                    "Force 4-Fold (C4)",
                    "Force 8-Fold Radial (D8)"
                ]
            )

            st.markdown(
                "**🎨 Render Aesthetics**"
            )

            morph_color = st.color_picker(
                "Vector Stroke Color",
                "#FF79C6"
            )

            morph_lw = st.slider(
                "Vector Stroke Weight",
                0.5,
                5.0,
                2.2,
                0.5
            )

        with p_canvas:

            center_pt = (
                cv_data["w"] / 2.0,
                cv_data["h"] / 2.0
            )

            t_paths, t_dots = (
                transform_vector_set(
                    cv_data["paths"],
                    cv_data["dots"],
                    center_pt,
                    p_rot,
                    p_scale_x,
                    p_scale_y,
                    p_shear,
                    p_ripple_amp,
                    p_ripple_freq
                )
            )

            fig_morph, ax_morph = (
                create_glass_figure(
                    figsize=(7, 7)
                )
            )

            def plot_transformed_layer(
                paths_to_plot,
                dots_to_plot,
                angle_offset=0,
                alpha_val=0.95
            ):

                rad = np.radians(
                    angle_offset
                )

                c = np.cos(rad)

                s = np.sin(rad)

                rot_offset = np.array(
                    [
                        [c, -s],
                        [s, c]
                    ]
                )

                cx, cy = center_pt

                for pts in paths_to_plot:

                    centered = (
                        pts
                        -
                        np.array([cx, cy])
                    )

                    rotated = (
                        rot_offset
                        @ centered.T
                    ).T + np.array(
                        [cx, cy]
                    )

                    ax_morph.plot(
                        rotated[:, 0],
                        cv_data["h"]
                        - rotated[:, 1],
                        color=morph_color,
                        lw=morph_lw,
                        alpha=alpha_val,
                        solid_capstyle="round"
                    )

                if len(
                    dots_to_plot
                ) > 0:

                    centered_d = (
                        dots_to_plot
                        -
                        np.array([cx, cy])
                    )

                    rotated_d = (
                        rot_offset
                        @ centered_d.T
                    ).T + np.array(
                        [cx, cy]
                    )

                    ax_morph.scatter(
                        rotated_d[:, 0],
                        cv_data["h"]
                        - rotated_d[:, 1],
                        color="#FFFFFF",
                        s=22,
                        zorder=5,
                        alpha=alpha_val,
                        edgecolors="#79C0FF",
                        linewidths=0.8
                    )

            if sym_multi == "Original Extracted":

                plot_transformed_layer(
                    t_paths,
                    t_dots
                )

            elif sym_multi == "Force 4-Fold (C4)":

                for ang in [
                    0,
                    90,
                    180,
                    270
                ]:

                    plot_transformed_layer(
                        t_paths,
                        t_dots,
                        angle_offset=ang,
                        alpha_val=0.9
                    )

            elif sym_multi == "Force 8-Fold Radial (D8)":

                for ang in [
                    0,
                    45,
                    90,
                    135,
                    180,
                    225,
                    270,
                    315
                ]:

                    plot_transformed_layer(
                        t_paths,
                        t_dots,
                        angle_offset=ang,
                        alpha_val=0.75
                    )

            pad_w = (
                cv_data["w"] * 0.4
            )

            pad_h = (
                cv_data["h"] * 0.4
            )

            ax_morph.set_xlim(
                -pad_w,
                cv_data["w"] + pad_w
            )

            ax_morph.set_ylim(
                -pad_h,
                cv_data["h"] + pad_h
            )

            ax_morph.set_aspect(
                "equal"
            )

            ax_morph.axis("off")

            st.pyplot(
                fig_morph,
                clear_figure=False
            )

            plt.close(
                fig_morph
            )


    # ========================================================
    # TAB 3: 3D STRUCTURAL EXTRUDER
    # ========================================================

    with tab_3d:

        st.subheader(
            "🧊 3D Parametric Kolam "
            "Extrusion & Wireframe Stage"
        )

        st.caption(
            "Extrude 2D planar continuous loops "
            "into volumetric 3D spatial ribbons, "
            "skeletons, and architectural pillar frames."
        )

        col3d_ctrl, col3d_view = (
            st.columns([1, 2.5])
        )

        with col3d_ctrl:

            st.markdown(
                "**🏗️ Extrusion Geometry**"
            )

            render_style = st.selectbox(
                "Structural Model",
                [
                    "Ribbon Surface",
                    "Wireframe Skeleton",
                    "3D Pillars (Vertical Struts)"
                ]
            )

            z_height = st.slider(
                "Extrusion Height (Z-Axis)",
                10.0,
                150.0,
                45.0,
                5.0
            )

            custom_3d_stroke = (
                st.color_picker(
                    "3D Curve Color",
                    svg_stroke_color
                )
            )

            custom_3d_dot = (
                st.color_picker(
                    "3D Dot Color",
                    svg_dot_color
                )
            )

            st.info(
                "💡 **Interaction Tip**: "
                "Click and drag inside the viewport "
                "to rotate the Kolam in real-time "
                "WebGL space. Scroll to zoom."
            )

        with col3d_view:

            fig_3d = render_3d_kolam(
                cv_data["paths"],
                cv_data["dots"],
                cv_data["w"],
                cv_data["h"],
                mode=render_style,
                extrusion_height=z_height,
                stroke_color=custom_3d_stroke,
                dot_color=custom_3d_dot
            )

            st.plotly_chart(
                fig_3d,
                width="stretch"
            )


    # ========================================================
    # TAB 4: QUARTER-TO-FULL INPAINTER
    # ========================================================

    with tab_inpaint:

        st.subheader(
            "🧩 Quarter-to-Full Symmetry Inpainter"
        )

        st.caption(
            "Recover the full 4-quadrant pattern "
            "by mirroring and joining quadrant seams."
        )

        in_ctrl, in_view = (
            st.columns([1, 2])
        )

        with in_ctrl:

            corner_src = st.selectbox(
                "Input Corner Quadrant",
                [
                    "Top-Left",
                    "Top-Right",
                    "Bottom-Left",
                    "Bottom-Right"
                ]
            )

            st.markdown(
                "**🧲 Seam Alignment & Gap Closure**"
            )

            overlap_x = st.slider(
                "Horizontal Seam Pull (px)",
                0.0,
                float(
                    cv_data["w"] * 0.5
                ),
                float(
                    cv_data["w"] * 0.08
                ),
                1.0
            )

            overlap_y = st.slider(
                "Vertical Seam Pull (px)",
                0.0,
                float(
                    cv_data["h"] * 0.5
                ),
                float(
                    cv_data["h"] * 0.08
                ),
                1.0
            )

            st.markdown("---")

            inpaint_stroke = st.color_picker(
                "Completed Stroke Color",
                "#00FFA3"
            )

            inpaint_lw = st.slider(
                "Completed Stroke Thickness",
                1.0,
                6.0,
                2.5,
                0.5
            )

        with in_view:

            (
                c_paths,
                c_dots,
                total_w,
                total_h
            ) = complete_quadrant_kolam(
                cv_data["paths"],
                cv_data["dots"],
                cv_data["w"],
                cv_data["h"],
                corner_source=corner_src,
                overlap_x=overlap_x,
                overlap_y=overlap_y
            )

            fig_c, ax_c = (
                create_glass_figure(
                    figsize=(7, 7)
                )
            )

            for p in c_paths:

                ax_c.plot(
                    p[:, 0],
                    total_h - p[:, 1],
                    color=inpaint_stroke,
                    lw=inpaint_lw,
                    solid_capstyle="round"
                )

            if len(c_dots) > 0:

                ax_c.scatter(
                    c_dots[:, 0],
                    total_h - c_dots[:, 1],
                    color="#FFFFFF",
                    s=24,
                    edgecolors="#79C0FF",
                    linewidths=0.8,
                    zorder=5
                )

            ax_c.set_xlim(
                0,
                total_w
            )

            ax_c.set_ylim(
                0,
                total_h
            )

            ax_c.set_aspect(
                "equal"
            )

            ax_c.axis(
                "off"
            )

            st.pyplot(
                fig_c,
                clear_figure=False
            )

            plt.close(
                fig_c
            )


    # ========================================================
    # TAB 5: PROCEDURAL INFINITE GENERATOR
    # ========================================================

    with tab_procedural:

        st.subheader(
            "♾️ Procedural 'Infinite Kolam' Extrapolator"
        )

        st.caption(
            "Procedurally extrapolate and scale "
            "traditional symmetry grammars into "
            "arbitrary lattice dimensions."
        )

        p_ctrl2, p_view2 = (
            st.columns([1, 2])
        )

        with p_ctrl2:

            grid_scale = st.slider(
                "Matrix Size (N x N)",
                3,
                15,
                7,
                step=2
            )

            symmetry_fold = st.selectbox(
                "Symmetry Projection",
                [
                    "4-Fold Rotational (C4)",
                    "Dihedral Mirror (D4)",
                    "Concentric Weave (Kambi)"
                ]
            )

        with p_view2:

            p_fig, p_ax = (
                create_glass_figure(
                    figsize=(6, 6)
                )
            )

            px, py = np.meshgrid(
                range(grid_scale),
                range(grid_scale)
            )

            p_ax.scatter(
                px,
                py,
                color=svg_dot_color,
                s=30,
                zorder=5,
                edgecolors="#79C0FF",
                linewidths=0.8
            )

            p_mid = (
                grid_scale // 2
            )

            if "4-Fold" in symmetry_fold:

                for r in range(
                    1,
                    p_mid + 1
                ):

                    poly = plt.Polygon(
                        [
                            [
                                p_mid - r,
                                p_mid
                            ],
                            [
                                p_mid,
                                p_mid + r
                            ],
                            [
                                p_mid + r,
                                p_mid
                            ],
                            [
                                p_mid,
                                p_mid - r
                            ]
                        ],
                        fill=False,
                        edgecolor=svg_stroke_color,
                        lw=svg_stroke_width
                    )

                    p_ax.add_patch(
                        poly
                    )

            elif "Dihedral" in symmetry_fold:

                for i in range(
                    1,
                    p_mid + 1
                ):

                    p_ax.plot(
                        [
                            p_mid - i,
                            p_mid + i
                        ],
                        [
                            p_mid - i,
                            p_mid + i
                        ],
                        color=svg_stroke_color,
                        lw=svg_stroke_width
                    )

                    p_ax.plot(
                        [
                            p_mid - i,
                            p_mid + i
                        ],
                        [
                            p_mid + i,
                            p_mid - i
                        ],
                        color=svg_stroke_color,
                        lw=svg_stroke_width
                    )

            else:

                for r in np.linspace(
                    0.8,
                    p_mid,
                    p_mid * 2
                ):

                    c = plt.Circle(
                        (
                            p_mid,
                            p_mid
                        ),
                        r,
                        fill=False,
                        edgecolor=svg_stroke_color,
                        lw=svg_stroke_width
                    )

                    p_ax.add_patch(
                        c
                    )

            p_ax.set_xlim(
                -0.6,
                grid_scale - 0.4
            )

            p_ax.set_ylim(
                -0.6,
                grid_scale - 0.4
            )

            p_ax.set_aspect(
                "equal"
            )

            p_ax.axis(
                "off"
            )

            st.pyplot(
                p_fig
            )

            plt.close(
                p_fig
            )


    # ========================================================
    # TAB 6: CULTURAL ARCHIVE
    # ========================================================

    with tab_archive:

        st.subheader(
            "📜 Digital Ethnomathematics Archive & CAD Export"
        )

        st.caption(
            "Preserving oral geometric heritage "
            "into formal computational graph grammars."
        )

        archive_data = {

            "metadata": {

                "source_file": image_source,

                "using_default_demo": using_default,

                "extraction_method":
                    "Hybrid Medial Axis & Vision LLM",

                "symmetry_index":
                    cv_data["symmetry_score"],

                "dot_count":
                    len(cv_data["dots"])
            },

            "pulli_coordinates":
                (
                    cv_data["dots"].tolist()
                    if len(
                        cv_data["dots"]
                    ) > 0
                    else []
                ),

            "manufacturing_targets": [

                "Laser Cutting",

                "CNC Stone Carving",

                "Digital Textile Weaving",

                "Architectural Restoration"
            ]
        }

        st.json(
            archive_data
        )

        st.download_button(
            "Export Cultural Heritage JSON",
            json.dumps(
                archive_data,
                indent=2
            ),
            file_name="kolam_archive.json",
            mime="application/json"
        )


# ============================================================
# NO IMAGE FOUND
# ============================================================

else:

    st.warning(
        "🪷 No Kolam image found."
    )

    st.markdown(
        """
        ### Demo image missing

        Place your default Kolam image here:

        `assets/default_kolam.png`

        relative to this Streamlit application.

        You can also upload a Kolam image using the uploader above.
        """
    )