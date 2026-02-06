from __future__ import annotations

import shlex
import subprocess
import tempfile
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
CBIR_BINARY = PROJECT_ROOT / "cbir"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
FOLDER_PLACEHOLDER = "Select a folder..."
QUERY_PLACEHOLDER = "Click an image below or type a path..."
GALLERY_MAX_HEIGHT_PX = 420


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


@st.cache_data(show_spinner=False)
def list_images(database_dir_text: str) -> list[Path]:
    if not database_dir_text.strip():
        return []
    database_dir = resolve_path(database_dir_text)
    if not database_dir.exists() or not database_dir.is_dir():
        return []

    files = []
    for file_path in database_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(file_path)
    return sorted(files)


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_cbir_output(stdout_text: str) -> list[dict[str, float | str]]:
    rows = []
    for line in stdout_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.rsplit(maxsplit=1)
        if len(parts) != 2:
            continue

        image_path, distance_text = parts
        try:
            distance = float(distance_text)
        except ValueError:
            continue
        rows.append({"image": image_path, "distance": distance})
    return rows


def get_default_database_dir() -> str:
    candidates = ["data/olympus", "olympus"]
    for candidate in candidates:
        if resolve_path(candidate).is_dir():
            return candidate
    return "data/olympus"


@st.cache_data(show_spinner=False)
def list_database_dirs() -> list[Path]:
    candidates: set[Path] = set()
    data_root = PROJECT_ROOT / "data"
    roots = [data_root] if data_root.is_dir() else []
    olympus_root = PROJECT_ROOT / "olympus"
    if olympus_root.is_dir():
        roots.append(olympus_root)

    for root in roots:
        try:
            for file_path in root.rglob("*"):
                if (
                    file_path.is_file()
                    and file_path.suffix.lower() in IMAGE_EXTENSIONS
                ):
                    candidates.add(file_path.parent)
        except Exception:
            continue

    return sorted(candidates)


def sync_database_dir_from_choice() -> None:
    choice = st.session_state.get("database_dir_choice")
    if choice and choice != FOLDER_PLACEHOLDER:
        st.session_state["_selected_dir"] = choice


def set_query_image(path_text: str) -> None:
    st.session_state["_selected_query"] = path_text


st.set_page_config(page_title="CBIR GUI", layout="wide")
st.title("Content-based Image Retrieval GUI")
st.caption("Run the existing ./cbir command from a visual interface.")

if not CBIR_BINARY.exists():
    st.error("Missing ./cbir binary. Build it first with: make")
    st.stop()

feature_options = [
    "baseline",
    "histogram_rg",
    "histogram_rgb",
    "multi_histogram",
    "texture_color",
    "dnn",
    "custom_sunset",
]

left_col, right_col = st.columns([2, 1])

with left_col:
    if "_selected_dir" not in st.session_state:
        st.session_state["_selected_dir"] = ""

    available_dirs = list_database_dirs()
    dir_options = [relative_or_absolute(path) for path in available_dirs]
    dir_default = st.session_state["_selected_dir"]
    if dir_default and dir_default not in dir_options:
        dir_options.insert(0, dir_default)

    if dir_options:
        dir_options_with_placeholder = [FOLDER_PLACEHOLDER] + dir_options
        try:
            default_index = dir_options_with_placeholder.index(dir_default)
        except ValueError:
            default_index = 0
        st.selectbox(
            "Known folders",
            dir_options_with_placeholder,
            index=default_index,
            key="database_dir_choice",
            on_change=sync_database_dir_from_choice,
        )
    database_dir_text = st.text_input(
        "Database directory", value=dir_default, placeholder="e.g. data/olympus"
    )
    feature_type = st.selectbox("Feature type", feature_options)

    if feature_type == "dnn":
        distance_options = ["cosine", "ssd"]
    elif feature_type == "baseline":
        distance_options = ["ssd"]
    else:
        distance_options = ["histogram_intersection"]

    distance_metric = st.selectbox("Distance metric", distance_options)
    top_n = st.number_input("Top N results", min_value=1, max_value=50, value=5, step=1)
    show_least = st.checkbox("Show least-similar matches (--least)", value=False)

    embeddings_csv_text = st.text_input(
        "Embeddings CSV (used for dnn)", value="features/embeddings.csv"
    )

with right_col:
    upload_disabled = feature_type == "dnn"
    query_options = ["Choose from database", "Upload image"]
    query_mode = st.radio(
        "Query image source",
        query_options,
        horizontal=False,
        captions=[
            None,
            "Not available for DNN (requires pre-computed embeddings)" if upload_disabled else None,
        ],
    )
    if query_mode == "Upload image" and upload_disabled:
        st.error("DNN mode requires a pre-computed embedding. Please choose an image from the database instead.")
        st.stop()

    images = list_images(database_dir_text)
    selected_image = None
    uploaded_file = None

    if query_mode == "Choose from database":
        if "_selected_query" not in st.session_state:
            st.session_state["_selected_query"] = ""

        query_default = st.session_state["_selected_query"]
        selected_image = st.text_input(
            "Query image", value=query_default, placeholder=QUERY_PLACEHOLDER
        ).strip() or None

        if images:
            st.caption("Gallery (click Select to fill the query image field)")
            with st.container(height=GALLERY_MAX_HEIGHT_PX, border=True):
                gallery_cols = st.columns(4)
                for idx, path in enumerate(images):
                    path_text = relative_or_absolute(path)
                    is_selected = selected_image == path_text
                    with gallery_cols[idx % 4]:
                        if is_selected:
                            with st.container(border=True):
                                st.image(str(path), use_container_width=True)
                                st.caption(f"**{Path(path_text).name}**")
                        else:
                            st.image(str(path), use_container_width=True)
                            st.caption(Path(path_text).name)
                        st.button(
                            "Select" if not is_selected else "Selected",
                            key=f"select_query_{idx}",
                            on_click=set_query_image,
                            args=(path_text,),
                            type="primary" if is_selected else "secondary",
                        )
        else:
            st.warning("No images found in the selected database directory.")
    else:
        uploaded_file = st.file_uploader(
            "Upload query image",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=False,
        )
        if feature_type == "dnn":
            st.info(
                "For dnn mode, uploaded images need an embedding row in the CSV "
                "(filename must match)."
            )

submitted = st.button("Run CBIR", type="primary")


if submitted:
    database_dir = resolve_path(database_dir_text)
    if not database_dir.exists() or not database_dir.is_dir():
        st.error(f"Database directory not found: {database_dir}")
        st.stop()

    temp_path = None
    if query_mode == "Choose from database":
        if not selected_image:
            st.error("Pick a query image from the database.")
            st.stop()
        target_image_arg = selected_image
    else:
        if uploaded_file is None:
            st.error("Upload a query image.")
            st.stop()

        suffix = Path(uploaded_file.name).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix="cbir_query_", dir=tempfile.gettempdir()
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = Path(tmp_file.name)
        target_image_arg = str(temp_path)

    cmd = [
        str(CBIR_BINARY),
        target_image_arg,
        relative_or_absolute(database_dir),
        feature_type,
        distance_metric,
        str(int(top_n)),
    ]
    if feature_type == "dnn":
        cmd.append(embeddings_csv_text)
    if show_least:
        cmd.append("--least")

    st.code(shlex.join(cmd), language="bash")

    try:
        completed = subprocess.run(
            cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
        )

        if completed.returncode != 0:
            st.error("CBIR execution failed.")
            stderr_text = completed.stderr.strip() or "(no stderr output)"
            st.code(stderr_text, language="text")
            st.stop()

        results = parse_cbir_output(completed.stdout)
        if not results:
            st.warning("No results returned.")
            raw_output = completed.stdout.strip() or "(empty output)"
            st.code(raw_output, language="text")
            st.stop()

        st.success(f"Retrieved {len(results)} results.")
        st.dataframe(results, use_container_width=True, hide_index=True)

        st.subheader("Query")
        query_preview_path = resolve_path(target_image_arg)
        if query_preview_path.exists():
            st.image(str(query_preview_path), caption=target_image_arg, width=280)
        else:
            st.text(target_image_arg)

        st.subheader("Matches")
        cols = st.columns(4)
        for idx, row in enumerate(results):
            match_path_text = str(row["image"])
            match_path = resolve_path(match_path_text)
            distance_value = float(row["distance"])
            caption = f"{Path(match_path_text).name} | distance={distance_value:.6f}"
            with cols[idx % 4]:
                if match_path.exists():
                    st.image(str(match_path), caption=caption, use_container_width=True)
                else:
                    st.text(caption)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
