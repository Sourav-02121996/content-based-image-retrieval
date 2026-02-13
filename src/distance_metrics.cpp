/*
Authors - Joseph Defendre, Sourav Das

Implements vector distance and similarity metrics.
Provides SSD and histogram intersection utilities.
Adds weighted multi-histogram distance support.
Implements cosine distance with a zero-norm guard.
*/
#include "../include/distance_metrics.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

/**
 * Compute sum of squared differences between matching vector elements.
 *
 * @param a First feature vector.
 * @param b Second feature vector.
 * @return Sum of squared differences.
 * @throws std::runtime_error if sizes do not match.
 */
float ssdDistance(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("SSD distance size mismatch.");
    }
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

/**
 * Compute histogram intersection similarity in [0, 1] for normalized histograms.
 *
 * @param a First normalized histogram.
 * @param b Second normalized histogram.
 * @return Intersection similarity.
 * @throws std::runtime_error if sizes do not match.
 */
float histogramIntersectionSimilarity(
    const std::vector<float> &a,
    const std::vector<float> &b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Histogram intersection size mismatch.");
    }
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += std::min(a[i], b[i]);
    }
    return sum;
}

/**
 * Convert histogram intersection similarity to distance (smaller is more similar).
 *
 * @param a First normalized histogram.
 * @param b Second normalized histogram.
 * @return Distance value (1 - similarity).
 */
float histogramIntersectionDistance(
    const std::vector<float> &a,
    const std::vector<float> &b) {
    float similarity = histogramIntersectionSimilarity(a, b);
    return 1.0f - similarity;
}

/**
 * Compute a weighted average of per-region histogram intersection distances.
 *
 * @param a Concatenated histograms for image A.
 * @param b Concatenated histograms for image B.
 * @param binsPerHistogram Number of bins per region.
 * @param histogramCount Number of regions.
 * @param weights Per-region weights (size histogramCount).
 * @return Weighted intersection distance.
 * @throws std::runtime_error if sizes or weights do not match expectations.
 */
float histogramIntersectionDistanceMulti(
    const std::vector<float> &a,
    const std::vector<float> &b,
    size_t binsPerHistogram,
    size_t histogramCount,
    const std::vector<float> &weights) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Multi-histogram size mismatch.");
    }
    if (weights.size() != histogramCount) {
        throw std::runtime_error("Multi-histogram weight size mismatch.");
    }

    float weightSum = std::accumulate(weights.begin(), weights.end(), 0.0f);
    if (weightSum <= 0.0f) {
        throw std::runtime_error("Multi-histogram weights must sum to > 0.");
    }

    float total = 0.0f;
    for (size_t region = 0; region < histogramCount; ++region) {
        size_t offset = region * binsPerHistogram;
        // Slice out the histogram block for this region.
        std::vector<float> histA(a.begin() + offset, a.begin() + offset + binsPerHistogram);
        std::vector<float> histB(b.begin() + offset, b.begin() + offset + binsPerHistogram);
        float distance = histogramIntersectionDistance(histA, histB);
        total += distance * weights[region];
    }

    // Normalize by total weight to keep distance scale comparable.
    return total / weightSum;
}

/**
 * Compute cosine distance (1 - cosine similarity) with zero-norm protection.
 *
 * @param a First feature vector.
 * @param b Second feature vector.
 * @return Cosine distance (1 - similarity), or 1.0 if either norm is zero.
 * @throws std::runtime_error if sizes do not match.
 */
float cosineDistance(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Cosine distance size mismatch.");
    }
    float dot = 0.0f;
    float normA = 0.0f;
    float normB = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    if (normA <= 0.0f || normB <= 0.0f) {
        return 1.0f;
    }
    float cosine = dot / (std::sqrt(normA) * std::sqrt(normB));
    return 1.0f - cosine;
}
