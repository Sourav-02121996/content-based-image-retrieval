#ifndef FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_H

#include <opencv2/opencv.hpp>
#include <vector>

std::vector<float> extractCenterPatchFeature(const cv::Mat &image, int patchSize = 7);

std::vector<float> extractRgbHistogram(const cv::Mat &image, int binsPerChannel = 8);

std::vector<float> extractRgChromaticityHistogram(const cv::Mat &image, int binsPerChannel = 16);

std::vector<float> extractMultiRegionRgbHistogram(
    const cv::Mat &image,
    int binsPerChannel = 8,
    int regionCount = 2);

std::vector<float> extractSobelMagnitudeHistogram(const cv::Mat &image, int bins = 16);

std::vector<float> extractCustomSunsetHistogram(
    const cv::Mat &image,
    int binsPerChannel = 8,
    int regionCount = 3);

#endif
