# Project 2: Content-based Image Retrieval

## 1. Project Description (≤200 words)
This project implements a content-based image retrieval (CBIR) system that
matches a query image to a database by comparing visual features. The system
supports baseline patch matching, color histograms, multi-region histograms,
texture histograms (Sobel magnitude), deep network embeddings, and a custom
feature configuration for a chosen category. Each method produces a feature
vector for the target and database images, computes a distance metric, and
returns the top-N closest matches.

## 2. Required Results

### Task 1 — Baseline Matching
- Query: pic.1016.jpg  
- Top matches:
  - pic.1016.jpg  
    ![](data/olympus/pic.1016.jpg)
  - pic.0986.jpg  
    ![](data/olympus/pic.0986.jpg)
  - pic.0641.jpg  
    ![](data/olympus/pic.0641.jpg)
  - pic.0547.jpg  
    ![](data/olympus/pic.0547.jpg)

### Task 2 — Histogram Matching
- Query: pic.0164.jpg  
- Top matches:
  - pic.0164.jpg  
    ![](data/olympus/pic.0164.jpg)
  - pic.0080.jpg  
    ![](data/olympus/pic.0080.jpg)
  - pic.1032.jpg  
    ![](data/olympus/pic.1032.jpg)
  - pic.0461.jpg  
    ![](data/olympus/pic.0461.jpg)

### Task 3 — Multi-histogram Matching
- Query: pic.0274.jpg  
- Top matches:
  - pic.0274.jpg  
    ![](data/olympus/pic.0274.jpg)
  - pic.0273.jpg  
    ![](data/olympus/pic.0273.jpg)
  - pic.1031.jpg  
    ![](data/olympus/pic.1031.jpg)
  - pic.0409.jpg  
    ![](data/olympus/pic.0409.jpg)

### Task 4 — Texture + Color
- Query: pic.0535.jpg  
- Top matches:
  - pic.0535.jpg  
    ![](data/olympus/pic.0535.jpg)
  - pic.0004.jpg  
    ![](data/olympus/pic.0004.jpg)
  - pic.0001.jpg  
    ![](data/olympus/pic.0001.jpg)
  - pic.0356.jpg  
    ![](data/olympus/pic.0356.jpg)
- Comparison vs Task 2/3: The texture+color method shifts matches toward images with similar edge structure (buildings/geometry) rather than only color similarity, compared to the histogram-only methods.

### Task 5 — Deep Network Embeddings
- Query: pic.0893.jpg  
- Top matches:
  - pic.0893.jpg  
    ![](data/olympus/pic.0893.jpg)
  - pic.0897.jpg  
    ![](data/olympus/pic.0897.jpg)
  - pic.0136.jpg  
    ![](data/olympus/pic.0136.jpg)
  - pic.0146.jpg  
    ![](data/olympus/pic.0146.jpg)
- Query: pic.0164.jpg  
- Top matches:
  - pic.0164.jpg  
    ![](data/olympus/pic.0164.jpg)
  - pic.1032.jpg  
    ![](data/olympus/pic.1032.jpg)
  - pic.0213.jpg  
    ![](data/olympus/pic.0213.jpg)
  - pic.0690.jpg  
    ![](data/olympus/pic.0690.jpg)
- Comparison vs classic features: DNN matches tend to preserve semantic content and layout even when overall color distributions differ, while classic histograms bias toward color similarity.

### Task 6 — Classic vs DNN Comparison
- Queries: pic.1072.jpg, pic.0948.jpg, pic.0734.jpg  
- Summary:
  - pic.1072.jpg: Histogram RG returns color-similar scenes; DNN returns images with similar scene structure even with different palettes.
  - pic.0948.jpg: Histogram RG emphasizes color tone; DNN yields closer scene/subject context.
  - pic.0734.jpg: DNN results show near-neighbor scene continuity, while histogram RG mixes in visually similar colors from different contexts.

### Task 7 — Custom Design (Sunset-Oriented Feature)
- Queries: pic.0048.jpg, pic.0552.jpg  
- Top 5 matches (pic.0048.jpg):
  - pic.0048.jpg  
    ![](data/olympus/pic.0048.jpg)
  - pic.0552.jpg  
    ![](data/olympus/pic.0552.jpg)
  - pic.0533.jpg  
    ![](data/olympus/pic.0533.jpg)
  - pic.1003.jpg  
    ![](data/olympus/pic.1003.jpg)
  - pic.1059.jpg  
    ![](data/olympus/pic.1059.jpg)
- Least similar (pic.0048.jpg):
  - pic.0511.jpg  
    ![](data/olympus/pic.0511.jpg)
  - pic.0558.jpg  
    ![](data/olympus/pic.0558.jpg)
  - pic.0228.jpg  
    ![](data/olympus/pic.0228.jpg)
  - pic.0890.jpg  
    ![](data/olympus/pic.0890.jpg)
  - pic.0689.jpg  
    ![](data/olympus/pic.0689.jpg)
- Top 5 matches (pic.0552.jpg):
  - pic.0552.jpg  
    ![](data/olympus/pic.0552.jpg)
  - pic.0048.jpg  
    ![](data/olympus/pic.0048.jpg)
  - pic.0324.jpg  
    ![](data/olympus/pic.0324.jpg)
  - pic.0197.jpg  
    ![](data/olympus/pic.0197.jpg)
  - pic.0958.jpg  
    ![](data/olympus/pic.0958.jpg)
- Least similar (pic.0552.jpg):
  - pic.0511.jpg  
    ![](data/olympus/pic.0511.jpg)
  - pic.0558.jpg  
    ![](data/olympus/pic.0558.jpg)
  - pic.0228.jpg  
    ![](data/olympus/pic.0228.jpg)
  - pic.0890.jpg  
    ![](data/olympus/pic.0890.jpg)
  - pic.0046.jpg  
    ![](data/olympus/pic.0046.jpg)

## 3. Extensions
None.

## 4. Reflection
Building multiple feature extractors clarified how different representations capture different aspects of visual similarity. The classic histograms were fast and intuitive but biased toward color distributions, while adding texture helped match structural cues. The DNN embeddings frequently produced more semantically coherent matches, especially when color alone was ambiguous. The custom sunset feature benefited from emphasizing top-to-bottom regions, which aligned with the sky-to-ground gradient often present in sunsets.

## 5. Acknowledgments
Sourav Das, Joseph Defendre. Course materials and OpenCV documentation.
