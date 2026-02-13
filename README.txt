================================================================================
                    CS 5330 - Pattern Recognition and Computer Vision
                        Project2 - Content-based Image Retrieval
================================================================================

Name: Joseph Defendre, Sourav Das  
Date: February 7, 2026
Course: CS 5330 Pattern Recognition and Computer Vision  
Project: Project2 - Content-based Image Retrieval

--------------------------------------------------------------------------------
OVERVIEW
--------------------------------------------------------------------------------
This project implements a C++ command-line CBIR system with classic features
(patch, histograms, texture) and deep network embeddings. The program outputs
the top N matches for a query image from a database directory.

--------------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------------
- macOS (darwin 25.2.0)
- OpenCV 4
- C++17 compiler
- Streamlit (Python)
- VS Code (development IDE)

--------------------------------------------------------------------------------
BUILDING INSTRUCTIONS
--------------------------------------------------------------------------------
Prerequisites:
- OpenCV 4 installed and discoverable by the compiler
- C++17-capable compiler (clang++ or g++)
- Make available on PATH

Build Steps (from the project root):
1. Clean previous outputs (optional):
   make clean
2. Build the cbir binary:
   make

Expected Output:
- The compiled binary is created at:
  ./cbir

Common Issues:
- If OpenCV headers are not found, ensure OpenCV 4 is installed and its include
  path is discoverable by your compiler/toolchain.
- If linking fails, verify OpenCV libraries are installed and available to the
  linker on your system.

--------------------------------------------------------------------------------
GUI (STREAMLIT)
--------------------------------------------------------------------------------
Run CBIR from a visual interface:
  pip install -r requirements.txt
  streamlit run app.py

Notes:
- Build cbir first with: make
- In the GUI, set database directory to data/olympus (or olympus if that is
  where your folder is).
- For dnn, provide the embeddings CSV path (for example features/embeddings.csv).

Streamlit Install/Run (Step-by-step):
1. (Optional) Create and activate a virtual environment:
   python -m venv .venv
   source .venv/bin/activate
2. Install Python dependencies:
   pip install -r requirements.txt
3. Launch the app:
   streamlit run app.py

--------------------------------------------------------------------------------
RUNNING INSTRUCTIONS
--------------------------------------------------------------------------------
  ./cbir <target_image> <database_dir> <feature_type> <distance_metric> <N>
        [embeddings_csv] [--least]

Feature Types:
- baseline           7x7 center patch + SSD
- histogram_rg       RG chromaticity histogram + histogram intersection
- histogram_rgb      RGB histogram + histogram intersection
- multi_histogram    top/bottom RGB histograms + weighted intersection
- texture_color      RGB histogram + Sobel magnitude histogram
- dnn                ResNet18 embeddings from CSV
- custom_sunset      3-region RGB histograms (weighted to emphasize lower region)

Distance Metrics:
- ssd
- histogram_intersection
- cosine

Examples:
Baseline (Task 1):
  ./cbir data/olympus/pic.1016.jpg data/olympus baseline ssd 4

RG histogram (Task 2):
  ./cbir data/olympus/pic.0164.jpg data/olympus histogram_rg histogram_intersection 4

Multi-histogram (Task 3):
  ./cbir data/olympus/pic.0274.jpg data/olympus multi_histogram histogram_intersection 4

Texture + color (Task 4):
  ./cbir data/olympus/pic.0535.jpg data/olympus texture_color histogram_intersection 4

Deep embeddings (Task 5):
  ./cbir data/olympus/pic.0893.jpg data/olympus dnn cosine 4 features/embeddings.csv

Custom sunset feature (Task 7):
  ./cbir data/olympus/pic.0734.jpg data/olympus custom_sunset histogram_intersection 5

Least-similar results (optional):
  ./cbir data/olympus/pic.0048.jpg data/olympus custom_sunset histogram_intersection 5 --least

--------------------------------------------------------------------------------
TESTING
--------------------------------------------------------------------------------
We used the required query images from the assignment:
- Task 1: pic.1016.jpg
- Task 2: pic.0164.jpg
- Task 3: pic.0274.jpg
- Task 4: pic.0535.jpg
- Task 5: pic.0893.jpg, pic.0164.jpg
- Task 7: pic.0048.jpg

--------------------------------------------------------------------------------
EXTENSIONS
--------------------------------------------------------------------------------
We added a Streamlit GUI to run CBIR visually. The interface lets users select
a query image and database folder, choose the feature type and distance metric,
set N, optionally show least-similar results, and view ranked thumbnail grids.
The GUI wraps the compiled C++ cbir binary and supports DNN embeddings by
accepting a CSV path.

--------------------------------------------------------------------------------
NOTES
--------------------------------------------------------------------------------
- The embeddings CSV (features/embeddings.csv) should contain filenames as the
  first column.
- For DNN matching, the program looks up embeddings by basename.
- Use (data/olympus/) as the database directory for all required queries.

--------------------------------------------------------------------------------
PROJECT STRUCTURE
--------------------------------------------------------------------------------
.
├── Makefile
├── README.md
├── README.txt
├── app.py
├── cbir
├── requirements.txt
├── report.html
├── report.md
├── report.pdf
├── CS 5330-Project 2_ Content-based Image Retrieval_l1.pdf
├── assets/
│   ├── gui_screen_1.png
│   └── gui_screen_2.png
├── data/
│   └── olympus/
├── features/
│   └── embeddings.csv
├── include/
│   ├── distance_metrics.h
│   ├── feature_extraction.h
│   └── image_io.h
└── src/
    ├── distance_metrics.cpp
    ├── feature_extraction.cpp
    ├── image_io.cpp
    ├── main.cpp
    └── readfiles.cpp

--------------------------------------------------------------------------------
REQUIRED FILES
--------------------------------------------------------------------------------
Source Files:
- src/main.cpp                - CLI entry point for CBIR retrieval
- src/feature_extraction.cpp  - Feature extraction implementations
- src/distance_metrics.cpp    - Distance and similarity metrics
- src/image_io.cpp            - Image/CSV I/O utilities
- src/readfiles.cpp           - Directory scanner sample utility

Header Files:
- include/feature_extraction.h - Feature extraction declarations
- include/distance_metrics.h   - Distance metric declarations
- include/image_io.h           - Image/CSV I/O declarations

App Files:
- app.py                      - Streamlit GUI wrapper around the cbir binary
- requirements.txt            - Python dependencies for the GUI

Data/Model Files:
- data/olympus/               - Image database for retrieval
- features/embeddings.csv     - ResNet18 embeddings (for dnn mode)

Build/Config Files:
- Makefile                    - Build configuration for the C++ binary

--------------------------------------------------------------------------------
FINAL RUN/TESTING NOTES
--------------------------------------------------------------------------------
Prerequisites:
1. Database directory: data/olympus/ contains all dataset images.
2. Embeddings file: features/embeddings.csv exists with filename + 512 values.

Verification Commands:
Task 1:
  ./cbir data/olympus/pic.1016.jpg data/olympus baseline ssd 4
Task 2:
  ./cbir data/olympus/pic.0164.jpg data/olympus histogram_rg histogram_intersection 4
Task 3:
  ./cbir data/olympus/pic.0274.jpg data/olympus multi_histogram histogram_intersection 4
Task 4:
  ./cbir data/olympus/pic.0535.jpg data/olympus texture_color histogram_intersection 4
Task 5:
  ./cbir data/olympus/pic.0893.jpg data/olympus dnn cosine 4 features/embeddings.csv
  ./cbir data/olympus/pic.0164.jpg data/olympus dnn cosine 4 features/embeddings.csv
Task 7:
  ./cbir data/olympus/pic.0048.jpg data/olympus custom_sunset histogram_intersection 5
  ./cbir data/olympus/pic.0048.jpg data/olympus custom_sunset histogram_intersection 5 --least

--------------------------------------------------------------------------------
ACKNOWLEDGMENTS
--------------------------------------------------------------------------------
- Claude (Anthropic): Used as a generative AI assistant for help with code
  implementation, debugging, and report writing.

- Streamlit: https://docs.streamlit.io/ - Used to build the lightweight GUI
  for running CBIR queries and visualizing ranked results.

- OpenCV Documentation: https://docs.opencv.org/ - Reference for image
  processing functions, video capture, and Haar cascades.

- ResNet18 (PyTorch Model Zoo):
  https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
  Pretrained embeddings referenced for the deep feature baseline and comparison.

- CS 5330 Course Materials: Lecture notes and project specifications provided
  guidance on distance metric implementation.

--------------------------------------------------------------------------------
TIME TRAVEL DAYS
--------------------------------------------------------------------------------

Time travel days used: 0
