#ifndef DISTANCE_METRICS_H
#define DISTANCE_METRICS_H

#include <vector>

float ssdDistance(const std::vector<float> &a, const std::vector<float> &b);

float histogramIntersectionSimilarity(
    const std::vector<float> &a,
    const std::vector<float> &b);

float histogramIntersectionDistance(
    const std::vector<float> &a,
    const std::vector<float> &b);

float histogramIntersectionDistanceMulti(
    const std::vector<float> &a,
    const std::vector<float> &b,
    size_t binsPerHistogram,
    size_t histogramCount,
    const std::vector<float> &weights);

float cosineDistance(const std::vector<float> &a, const std::vector<float> &b);

#endif
