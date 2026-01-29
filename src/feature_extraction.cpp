#include "../include/feature_extraction.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace {
std::vector<float> normalizeHistogram(const std::vector<float> &histogram) {
    float sum = std::accumulate(histogram.begin(), histogram.end(), 0.0f);
    if (sum <= 0.0f) {
        return histogram;
    }
    std::vector<float> normalized(histogram.size());
    std::transform(histogram.begin(), histogram.end(), normalized.begin(),
                   [sum](float value) { return value / sum; });
    return normalized;
}

int clampIndex(int value, int maxValue) {
    return std::min(std::max(value, 0), maxValue);
}

cv::Mat ensureMinSize(const cv::Mat &image, int minSize) {
    if (image.rows >= minSize && image.cols >= minSize) {
        return image;
    }
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(minSize, minSize));
    return resized;
}

int binForValue(float value, int bins) {
    int index = static_cast<int>(value * bins);
    return clampIndex(index, bins - 1);
}
} // namespace

std::vector<float> extractCenterPatchFeature(const cv::Mat &image, int patchSize) {
    cv::Mat safeImage = ensureMinSize(image, patchSize);
    int centerRow = safeImage.rows / 2;
    int centerCol = safeImage.cols / 2;
    int half = patchSize / 2;
    int startRow = std::max(0, centerRow - half);
    int startCol = std::max(0, centerCol - half);
    int endRow = std::min(safeImage.rows, startRow + patchSize);
    int endCol = std::min(safeImage.cols, startCol + patchSize);

    std::vector<float> feature;
    feature.reserve(patchSize * patchSize * safeImage.channels());
    for (int row = startRow; row < endRow; ++row) {
        const auto *rowPtr = safeImage.ptr<cv::Vec3b>(row);
        for (int col = startCol; col < endCol; ++col) {
            const cv::Vec3b &pixel = rowPtr[col];
            feature.push_back(static_cast<float>(pixel[0]));
            feature.push_back(static_cast<float>(pixel[1]));
            feature.push_back(static_cast<float>(pixel[2]));
        }
    }
    return feature;
}

std::vector<float> extractRgbHistogram(const cv::Mat &image, int binsPerChannel) {
    int totalBins = binsPerChannel * binsPerChannel * binsPerChannel;
    std::vector<float> histogram(totalBins, 0.0f);

    for (int row = 0; row < image.rows; ++row) {
        const auto *rowPtr = image.ptr<cv::Vec3b>(row);
        for (int col = 0; col < image.cols; ++col) {
            const cv::Vec3b &pixel = rowPtr[col];
            float b = static_cast<float>(pixel[0]) / 255.0f;
            float g = static_cast<float>(pixel[1]) / 255.0f;
            float r = static_cast<float>(pixel[2]) / 255.0f;
            int binB = binForValue(b, binsPerChannel);
            int binG = binForValue(g, binsPerChannel);
            int binR = binForValue(r, binsPerChannel);
            int index = (binR * binsPerChannel * binsPerChannel) +
                        (binG * binsPerChannel) + binB;
            histogram[index] += 1.0f;
        }
    }

    return normalizeHistogram(histogram);
}

std::vector<float> extractRgChromaticityHistogram(const cv::Mat &image, int binsPerChannel) {
    int totalBins = binsPerChannel * binsPerChannel;
    std::vector<float> histogram(totalBins, 0.0f);

    for (int row = 0; row < image.rows; ++row) {
        const auto *rowPtr = image.ptr<cv::Vec3b>(row);
        for (int col = 0; col < image.cols; ++col) {
            const cv::Vec3b &pixel = rowPtr[col];
            float r = static_cast<float>(pixel[2]);
            float g = static_cast<float>(pixel[1]);
            float b = static_cast<float>(pixel[0]);
            float sum = r + g + b;
            float rNorm = sum > 0.0f ? r / sum : 0.0f;
            float gNorm = sum > 0.0f ? g / sum : 0.0f;
            int binR = binForValue(rNorm, binsPerChannel);
            int binG = binForValue(gNorm, binsPerChannel);
            int index = (binR * binsPerChannel) + binG;
            histogram[index] += 1.0f;
        }
    }

    return normalizeHistogram(histogram);
}

std::vector<float> extractMultiRegionRgbHistogram(
    const cv::Mat &image,
    int binsPerChannel,
    int regionCount) {
    std::vector<float> feature;
    if (regionCount <= 1) {
        return extractRgbHistogram(image, binsPerChannel);
    }

    int rowsPerRegion = image.rows / regionCount;
    for (int region = 0; region < regionCount; ++region) {
        int startRow = region * rowsPerRegion;
        int endRow = (region == regionCount - 1) ? image.rows
                                                 : (region + 1) * rowsPerRegion;
        cv::Mat slice = image.rowRange(startRow, endRow);
        auto regionHist = extractRgbHistogram(slice, binsPerChannel);
        feature.insert(feature.end(), regionHist.begin(), regionHist.end());
    }

    return feature;
}

std::vector<float> extractSobelMagnitudeHistogram(const cv::Mat &image, int bins) {
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Mat gradX;
    cv::Mat gradY;
    cv::Sobel(gray, gradX, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gradY, CV_32F, 0, 1, 3);
    cv::Mat magnitude;
    cv::magnitude(gradX, gradY, magnitude);

    double maxValue = 0.0;
    cv::minMaxLoc(magnitude, nullptr, &maxValue);
    float maxMagnitude = static_cast<float>(maxValue);
    if (maxMagnitude <= 0.0f) {
        return std::vector<float>(bins, 0.0f);
    }

    std::vector<float> histogram(bins, 0.0f);
    for (int row = 0; row < magnitude.rows; ++row) {
        const auto *rowPtr = magnitude.ptr<float>(row);
        for (int col = 0; col < magnitude.cols; ++col) {
            float normalized = rowPtr[col] / maxMagnitude;
            int bin = binForValue(normalized, bins);
            histogram[bin] += 1.0f;
        }
    }

    return normalizeHistogram(histogram);
}

std::vector<float> extractCustomSunsetHistogram(
    const cv::Mat &image,
    int binsPerChannel,
    int regionCount) {
    return extractMultiRegionRgbHistogram(image, binsPerChannel, regionCount);
}
