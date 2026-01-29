# Content-based Image Retrieval (Project 2)

Authors: Joseph Defendre, Sourav Das  
Course: CS 5330 Pattern Recognition and Computer Vision  
Due: February 10, 2026

## Overview
This project implements a C++ command-line CBIR system with classic features
(patch, histograms, texture) and deep network embeddings. The program outputs
the top N matches for a query image from a database directory.

## Requirements
- macOS (darwin 25.2.0)
- OpenCV 4
- C++17 compiler

## Build
From the project root:
```
make
```

## Run
```
./cbir <target_image> <database_dir> <feature_type> <distance_metric> <N> [embeddings_csv] [--least]
```

### Feature Types
- `baseline` — 7x7 center patch + SSD
- `histogram_rg` — RG chromaticity histogram + histogram intersection
- `histogram_rgb` — RGB histogram + histogram intersection
- `multi_histogram` — top/bottom RGB histograms + weighted intersection
- `texture_color` — RGB histogram + Sobel magnitude histogram
- `dnn` — ResNet18 embeddings from CSV
- `custom_sunset` — 3-region RGB histograms (weighted to emphasize horizon)

### Distance Metrics
- `ssd`
- `histogram_intersection`
- `cosine`

### Examples
Baseline (Task 1):
```
./cbir data/olympus/pic.1016.jpg data/olympus baseline ssd 4
```

RG histogram (Task 2):
```
./cbir data/olympus/pic.0164.jpg data/olympus histogram_rg histogram_intersection 4
```

Multi-histogram (Task 3):
```
./cbir data/olympus/pic.0274.jpg data/olympus multi_histogram histogram_intersection 4
```

Texture + color (Task 4):
```
./cbir data/olympus/pic.0535.jpg data/olympus texture_color histogram_intersection 4
```

Deep embeddings (Task 5):
```
./cbir data/olympus/pic.0893.jpg data/olympus dnn cosine 4 features/embeddings.csv
```

Custom sunset feature (Task 7):
```
./cbir data/olympus/pic.0734.jpg data/olympus custom_sunset histogram_intersection 5
```

Least-similar results (optional):
```
./cbir data/olympus/pic.0048.jpg data/olympus custom_sunset histogram_intersection 5 --least
```

## Testing
Use the required query images from the assignment prompt:
- Task 1: `pic.1016.jpg`
- Task 2: `pic.0164.jpg`
- Task 3: `pic.0274.jpg`
- Task 4: `pic.0535.jpg`
- Task 5: `pic.0893.jpg`, `pic.0164.jpg`

## Extensions
No extensions implemented yet. Optional: replace `custom_sunset` with a more
specific category (bananas, trash bins) once the dataset is explored.

## Notes
- The embeddings CSV (`features/embeddings.csv`) should contain filenames as the first column.
- For DNN matching, the program looks up embeddings by basename.
- Use `data/olympus/` as the database directory for all required queries.

## Final Run/Testing Notes

### Prerequisites
1. **Database directory**: `data/olympus/` contains all dataset images.
2. **Embeddings file**: `features/embeddings.csv` exists with filename + 512 values.

### Verification Commands
Task 1:
```
./cbir data/olympus/pic.1016.jpg data/olympus baseline ssd 4
```
Task 2:
```
./cbir data/olympus/pic.0164.jpg data/olympus histogram_rg histogram_intersection 4
```
Task 3:
```
./cbir data/olympus/pic.0274.jpg data/olympus multi_histogram histogram_intersection 4
```
Task 4:
```
./cbir data/olympus/pic.0535.jpg data/olympus texture_color histogram_intersection 4
```
Task 5:
```
./cbir data/olympus/pic.0893.jpg data/olympus dnn cosine 4 features/embeddings.csv
./cbir data/olympus/pic.0164.jpg data/olympus dnn cosine 4 features/embeddings.csv
```
Task 7:
```
./cbir data/olympus/pic.0048.jpg data/olympus custom_sunset histogram_intersection 5
./cbir data/olympus/pic.0048.jpg data/olympus custom_sunset histogram_intersection 5 --least
```

## Time Travel Days
None.
