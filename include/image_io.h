/*
Authors - Joseph Defendre, Sourav Das


Declarations for image I/O and CSV helper utilities.
Lists image files in a directory by common extensions.
Loads images with OpenCV and reads/writes feature CSVs.
Provides an embeddings CSV reader keyed by filename.
*/
#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Returns a sorted list of image file paths under the directory.
std::vector<std::string> listImageFiles(const std::string &directoryPath);

// Loads an image from disk and throws std::runtime_error on failure.
cv::Mat loadImageOrThrow(const std::string &imagePath);

// Writes (filename, feature vector) pairs to a CSV file.
bool writeFeaturesCsv(
    const std::string &outputPath,
    const std::vector<std::pair<std::string, std::vector<float>>> &features);

// Reads (filename, feature vector) pairs from a CSV file.
std::vector<std::pair<std::string, std::vector<float>>> readFeaturesCsv(
    const std::string &inputPath);

// Reads a CSV of embeddings into a map keyed by filename.
std::unordered_map<std::string, std::vector<float>> readEmbeddingsCsv(
    const std::string &inputPath);

#endif
