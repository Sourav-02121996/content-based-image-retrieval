#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

std::vector<std::string> listImageFiles(const std::string &directoryPath);

cv::Mat loadImageOrThrow(const std::string &imagePath);

bool writeFeaturesCsv(
    const std::string &outputPath,
    const std::vector<std::pair<std::string, std::vector<float>>> &features);

std::vector<std::pair<std::string, std::vector<float>>> readFeaturesCsv(
    const std::string &inputPath);

std::unordered_map<std::string, std::vector<float>> readEmbeddingsCsv(
    const std::string &inputPath);

#endif
