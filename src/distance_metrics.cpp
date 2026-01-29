#include "../include/distance_metrics.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

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

float histogramIntersectionDistance(
    const std::vector<float> &a,
    const std::vector<float> &b) {
    float similarity = histogramIntersectionSimilarity(a, b);
    return 1.0f - similarity;
}

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
        std::vector<float> histA(a.begin() + offset, a.begin() + offset + binsPerHistogram);
        std::vector<float> histB(b.begin() + offset, b.begin() + offset + binsPerHistogram);
        float distance = histogramIntersectionDistance(histA, histB);
        total += distance * weights[region];
    }

    return total / weightSum;
}

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
