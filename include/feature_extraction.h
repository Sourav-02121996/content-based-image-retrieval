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

/**
 * Extract a flattened center patch in BGR order (uint8 -> float).
 *
 * @param image Input BGR image (CV_8UC3).
 * @param patchSize Patch width/height in pixels (default 7).
 * @return Flattened patch feature vector.
 */
std::vector<float> extractCenterPatchFeature(const cv::Mat &image, int patchSize = 7);

/**
 * Compute a normalized RGB histogram for the entire image.
 *
 * @param image Input BGR image (CV_8UC3).
 * @param binsPerChannel Number of bins per channel (default 8).
 * @return Normalized RGB histogram vector.
 */
std::vector<float> extractRgbHistogram(const cv::Mat &image, int binsPerChannel = 8);

/**
 * Compute a normalized r-g chromaticity histogram (illumination-invariant).
 *
 * @param image Input BGR image (CV_8UC3).
 * @param binsPerChannel Number of bins per channel (default 16).
 * @return Normalized r-g chromaticity histogram vector.
 */
std::vector<float> extractRgChromaticityHistogram(const cv::Mat &image, int binsPerChannel = 16);

/**
 * Split the image into horizontal regions and concatenate RGB histograms.
 *
 * @param image Input BGR image (CV_8UC3).
 * @param binsPerChannel Number of bins per channel (default 8).
 * @param regionCount Number of horizontal regions (default 2).
 * @return Concatenated multi-region histogram feature.
 */
std::vector<float> extractMultiRegionRgbHistogram(
    const cv::Mat &image,
    int binsPerChannel = 8,
    int regionCount = 2);

/**
 * Compute a normalized histogram of Sobel gradient magnitudes.
 *
 * @param image Input BGR image (CV_8UC3).
 * @param bins Number of magnitude bins (default 16).
 * @return Normalized Sobel magnitude histogram vector.
 */
std::vector<float> extractSobelMagnitudeHistogram(const cv::Mat &image, int bins = 16);

/**
 * Task-specific feature: multi-region RGB histogram configured for sunsets.
 *
 * @param image Input BGR image (CV_8UC3).
 * @param binsPerChannel Number of bins per channel (default 8).
 * @param regionCount Number of horizontal regions (default 3).
 * @return Concatenated histogram feature vector.
 */
std::vector<float> extractCustomSunsetHistogram(
    const cv::Mat &image,
    int binsPerChannel = 8,
    int regionCount = 3);

#endif
