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
- Top matches: [add images here]

### Task 2 — Histogram Matching
- Query: pic.0164.jpg  
- Top matches: [add images here]

### Task 3 — Multi-histogram Matching
- Query: pic.0274.jpg  
- Top matches: [add images here]

### Task 4 — Texture + Color
- Query: pic.0535.jpg  
- Top matches: [add images here]
- Comparison vs Task 2/3: [add notes here]

### Task 5 — Deep Network Embeddings
- Query: pic.0893.jpg  
- Top matches: [add images here]
- Query: pic.0164.jpg  
- Top matches: [add images here]
- Comparison vs classic features: [add notes here]

### Task 6 — Classic vs DNN Comparison
- Queries: pic.1072.jpg, pic.0948.jpg, pic.0734.jpg  
- Summary: [add analysis here]

### Task 7 — Custom Design (Sunset-Oriented Feature)
- Queries: [add two target images here]  
- Top 5 matches: [add images here]  
- Least similar results: [add images here]

## 3. Extensions
None.

## 4. Reflection
[Add reflection here.]

## 5. Acknowledgments
Sourav Das, Joseph Defendre. Course materials and OpenCV documentation.
