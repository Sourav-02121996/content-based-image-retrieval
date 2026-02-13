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

/**
 * Return a sorted list of image file paths under the directory.
 *
 * @param directoryPath Directory to scan for images.
 * @return Sorted vector of image file paths.
 */
std::vector<std::string> listImageFiles(const std::string &directoryPath);

/**
 * Load an image from disk and throw on failure.
 *
 * @param imagePath Path to the image file.
 * @return Loaded BGR image (CV_8UC3).
 * @throws std::runtime_error if the image cannot be loaded.
 */
cv::Mat loadImageOrThrow(const std::string &imagePath);

/**
 * Write (filename, feature vector) pairs to a CSV file.
 *
 * @param outputPath Destination CSV path.
 * @param features Vector of filename/feature pairs.
 * @return True on success, false if the file cannot be opened.
 */
bool writeFeaturesCsv(
    const std::string &outputPath,
    const std::vector<std::pair<std::string, std::vector<float>>> &features);

/**
 * Read (filename, feature vector) pairs from a CSV file.
 *
 * @param inputPath Source CSV path.
 * @return Vector of filename/feature pairs.
 * @throws std::runtime_error if the file cannot be opened.
 */
std::vector<std::pair<std::string, std::vector<float>>> readFeaturesCsv(
    const std::string &inputPath);

/**
 * Read an embeddings CSV into a map keyed by filename.
 *
 * @param inputPath Source CSV path.
 * @return Map of filename to embedding vector.
 * @throws std::runtime_error if the file cannot be opened.
 */
std::unordered_map<std::string, std::vector<float>> readEmbeddingsCsv(
    const std::string &inputPath);

#endif
