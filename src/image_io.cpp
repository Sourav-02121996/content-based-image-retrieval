#include "../include/image_io.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {
bool hasImageExtension(const std::string &filename) {
    auto lower = filename;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    auto dot = lower.find_last_of('.');
    if (dot == std::string::npos) {
        return false;
    }
    auto ext = lower.substr(dot);
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

std::vector<float> parseCsvNumbers(const std::string &line) {
    std::vector<float> values;
    std::stringstream lineStream(line);
    std::string cell;
    while (std::getline(lineStream, cell, ',')) {
        if (cell.empty()) {
            values.push_back(0.0f);
            continue;
        }
        values.push_back(std::stof(cell));
    }
    return values;
}
} // namespace

std::vector<std::string> listImageFiles(const std::string &directoryPath) {
    std::vector<std::string> files;
    for (const auto &entry : std::filesystem::directory_iterator(directoryPath)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto path = entry.path();
        if (hasImageExtension(path.filename().string())) {
            files.push_back(path.string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

cv::Mat loadImageOrThrow(const std::string &imagePath) {
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + imagePath);
    }
    return image;
}

bool writeFeaturesCsv(
    const std::string &outputPath,
    const std::vector<std::pair<std::string, std::vector<float>>> &features) {
    std::ofstream outputFile(outputPath);
    if (!outputFile.is_open()) {
        return false;
    }

    for (const auto &entry : features) {
        outputFile << entry.first;
        for (float value : entry.second) {
            outputFile << "," << value;
        }
        outputFile << "\n";
    }

    return true;
}

std::vector<std::pair<std::string, std::vector<float>>> readFeaturesCsv(
    const std::string &inputPath) {
    std::ifstream inputFile(inputPath);
    if (!inputFile.is_open()) {
        throw std::runtime_error("Failed to open features CSV: " + inputPath);
    }

    std::vector<std::pair<std::string, std::vector<float>>> features;
    std::string line;
    while (std::getline(inputFile, line)) {
        if (line.empty()) {
            continue;
        }
        std::stringstream lineStream(line);
        std::string filename;
        std::getline(lineStream, filename, ',');
        std::string rest;
        std::getline(lineStream, rest);
        auto values = parseCsvNumbers(rest);
        features.emplace_back(filename, std::move(values));
    }

    return features;
}

std::unordered_map<std::string, std::vector<float>> readEmbeddingsCsv(
    const std::string &inputPath) {
    std::ifstream inputFile(inputPath);
    if (!inputFile.is_open()) {
        throw std::runtime_error("Failed to open embeddings CSV: " + inputPath);
    }

    std::unordered_map<std::string, std::vector<float>> embeddings;
    std::string line;
    while (std::getline(inputFile, line)) {
        if (line.empty()) {
            continue;
        }
        std::stringstream lineStream(line);
        std::string filename;
        std::getline(lineStream, filename, ',');
        std::string rest;
        std::getline(lineStream, rest);
        auto values = parseCsvNumbers(rest);
        embeddings.emplace(filename, std::move(values));
    }

    return embeddings;
}
