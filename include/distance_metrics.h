/*
Authors - Joseph Defendre, Sourav Das


Declarations for distance metric functions used in CBIR.
Includes SSD, histogram intersection, and weighted multi-histogram support.
Provides cosine distance for embedding-based comparisons.
All functions operate on float vectors and validate sizes.
*/
#ifndef DISTANCE_METRICS_H
#define DISTANCE_METRICS_H

#include <vector>

// Computes sum of squared differences between two equal-length vectors.
// Throws std::runtime_error if sizes mismatch.
float ssdDistance(const std::vector<float> &a, const std::vector<float> &b);

// Computes histogram intersection similarity (higher is more similar).
// Throws std::runtime_error if sizes mismatch.
float histogramIntersectionSimilarity(
    const std::vector<float> &a,
    const std::vector<float> &b);

// Converts intersection similarity to a distance in [0, 1] (1 - similarity).
float histogramIntersectionDistance(
    const std::vector<float> &a,
    const std::vector<float> &b);

// Computes weighted intersection distance over multiple concatenated histograms.
// Expects a and b to contain histogramCount blocks of binsPerHistogram values.
float histogramIntersectionDistanceMulti(
    const std::vector<float> &a,
    const std::vector<float> &b,
    size_t binsPerHistogram,
    size_t histogramCount,
    const std::vector<float> &weights);

// Computes cosine distance (1 - cosine similarity) between two vectors.
// Returns 1.0 if either vector has zero norm.
float cosineDistance(const std::vector<float> &a, const std::vector<float> &b);

#endif
