/*
Authors - Joseph Defendre, Sourav Das


Declarations for feature extraction routines.
Covers baseline patch, RGB/RG histograms, and Sobel texture features.
Supports multi-region and custom sunset descriptors.
Used by the CLI and GUI to build feature vectors.
*/
#ifndef FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_H

#include <opencv2/opencv.hpp>
#include <vector>

// Extracts a flattened center patch in BGR order (uint8 -> float).
std::vector<float> extractCenterPatchFeature(const cv::Mat &image, int patchSize = 7);

// Computes a normalized RGB histogram with binsPerChannel bins per channel.
std::vector<float> extractRgbHistogram(const cv::Mat &image, int binsPerChannel = 8);

// Computes a normalized r-g chromaticity histogram (illumination-invariant).
std::vector<float> extractRgChromaticityHistogram(const cv::Mat &image, int binsPerChannel = 16);

// Splits the image into horizontal regions and concatenates RGB histograms.
std::vector<float> extractMultiRegionRgbHistogram(
    const cv::Mat &image,
    int binsPerChannel = 8,
    int regionCount = 2);

// Computes a normalized histogram of Sobel gradient magnitudes.
std::vector<float> extractSobelMagnitudeHistogram(const cv::Mat &image, int bins = 16);

// Task-specific feature: multi-region RGB histogram weighted for sunsets.
std::vector<float> extractCustomSunsetHistogram(
    const cv::Mat &image,
    int binsPerChannel = 8,
    int regionCount = 3);

#endif
