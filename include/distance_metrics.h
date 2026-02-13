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

/**
 * Compute sum of squared differences between two equal-length vectors.
 *
 * @param a First feature vector.
 * @param b Second feature vector.
 * @return Sum of squared differences.
 * @throws std::runtime_error if the input sizes do not match.
 */
float ssdDistance(const std::vector<float> &a, const std::vector<float> &b);

/**
 * Compute histogram intersection similarity (higher is more similar).
 *
 * @param a First normalized histogram.
 * @param b Second normalized histogram.
 * @return Intersection similarity value.
 * @throws std::runtime_error if the input sizes do not match.
 */
float histogramIntersectionSimilarity(
    const std::vector<float> &a,
    const std::vector<float> &b);

/**
 * Convert histogram intersection similarity to a distance in [0, 1].
 *
 * @param a First normalized histogram.
 * @param b Second normalized histogram.
 * @return Distance value (1 - similarity).
 * @throws std::runtime_error if the input sizes do not match.
 */
float histogramIntersectionDistance(
    const std::vector<float> &a,
    const std::vector<float> &b);

/**
 * Compute weighted intersection distance over concatenated per-region histograms.
 *
 * @param a Concatenated histograms for image A.
 * @param b Concatenated histograms for image B.
 * @param binsPerHistogram Number of bins in each region histogram.
 * @param histogramCount Number of regions concatenated in the vectors.
 * @param weights Per-region weights (size must equal histogramCount).
 * @return Weighted intersection distance.
 * @throws std::runtime_error if sizes or weights do not match expectations.
 */
float histogramIntersectionDistanceMulti(
    const std::vector<float> &a,
    const std::vector<float> &b,
    size_t binsPerHistogram,
    size_t histogramCount,
    const std::vector<float> &weights);

/**
 * Compute cosine distance (1 - cosine similarity) between two vectors.
 *
 * @param a First feature vector.
 * @param b Second feature vector.
 * @return Cosine distance in [0, 2], or 1.0 if either vector has zero norm.
 * @throws std::runtime_error if the input sizes do not match.
 */
float cosineDistance(const std::vector<float> &a, const std::vector<float> &b);

#endif
