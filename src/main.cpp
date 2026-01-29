#include "../include/distance_metrics.h"
#include "../include/feature_extraction.h"
#include "../include/image_io.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <filesystem>
#include <string>
std::string basenameFromPath(const std::string &path) {
    return std::filesystem::path(path).filename().string();
}
#include <vector>

namespace {
struct Match {
    std::string filename;
    float distance;
};

void printUsage() {
    std::cout
        << "Usage:\n"
        << "  ./cbir <target_image> <database_dir> <feature_type> <distance_metric> <N> [embeddings_csv] [--least]\n\n"
        << "Feature types:\n"
        << "  baseline\n"
        << "  histogram_rg\n"
        << "  histogram_rgb\n"
        << "  multi_histogram\n"
        << "  texture_color\n"
        << "  dnn\n"
        << "  custom_sunset\n\n"
        << "Distance metrics:\n"
        << "  ssd\n"
        << "  histogram_intersection\n"
        << "  cosine\n";
}

std::vector<Match> topMatches(std::vector<Match> matches, int topN, bool descending) {
    std::sort(matches.begin(), matches.end(),
              [descending](const Match &a, const Match &b) {
                  return descending ? a.distance > b.distance : a.distance < b.distance;
              });
    if (topN < static_cast<int>(matches.size())) {
        matches.resize(topN);
    }
    return matches;
}
} // namespace

int main(int argc, char **argv) {
    if (argc < 6) {
        printUsage();
        return 1;
    }

    try {
        std::string targetImagePath = argv[1];
        std::string databaseDir = argv[2];
        std::string featureType = argv[3];
        std::string distanceMetric = argv[4];
        int topN = std::stoi(argv[5]);
        bool showLeast = false;
        std::string embeddingsPath;

        for (int i = 6; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--least") {
                showLeast = true;
            } else if (embeddingsPath.empty()) {
                embeddingsPath = arg;
            }
        }

        auto imageFiles = listImageFiles(databaseDir);
        if (imageFiles.empty()) {
            std::cerr << "No images found in directory: " << databaseDir << "\n";
            return 1;
        }

        std::vector<Match> matches;
        matches.reserve(imageFiles.size());

        if (featureType == "dnn") {
            if (embeddingsPath.empty()) {
                std::cerr << "Missing embeddings CSV path for DNN features.\n";
                return 1;
            }
            auto embeddings = readEmbeddingsCsv(embeddingsPath);
            std::string targetKey = basenameFromPath(targetImagePath);
            auto targetIt = embeddings.find(targetKey);
            if (targetIt == embeddings.end()) {
                std::cerr << "Target embedding not found in CSV.\n";
                return 1;
            }
            const auto &targetEmbedding = targetIt->second;
            for (const auto &file : imageFiles) {
                std::string key = basenameFromPath(file);
                auto embedIt = embeddings.find(key);
                if (embedIt == embeddings.end()) {
                    continue;
                }
                const auto &embedding = embedIt->second;
                float distance = 0.0f;
                if (distanceMetric == "cosine") {
                    distance = cosineDistance(targetEmbedding, embedding);
                } else {
                    distance = ssdDistance(targetEmbedding, embedding);
                }
                matches.push_back({file, distance});
            }
        } else {
            cv::Mat targetImage = loadImageOrThrow(targetImagePath);

            if (featureType == "baseline") {
                auto targetFeature = extractCenterPatchFeature(targetImage);
                for (const auto &file : imageFiles) {
                    cv::Mat image = loadImageOrThrow(file);
                    auto feature = extractCenterPatchFeature(image);
                    float distance = ssdDistance(targetFeature, feature);
                    matches.push_back({file, distance});
                }
            } else if (featureType == "histogram_rg") {
                auto targetFeature = extractRgChromaticityHistogram(targetImage);
                for (const auto &file : imageFiles) {
                    cv::Mat image = loadImageOrThrow(file);
                    auto feature = extractRgChromaticityHistogram(image);
                    float distance = histogramIntersectionDistance(targetFeature, feature);
                    matches.push_back({file, distance});
                }
            } else if (featureType == "histogram_rgb") {
                auto targetFeature = extractRgbHistogram(targetImage);
                for (const auto &file : imageFiles) {
                    cv::Mat image = loadImageOrThrow(file);
                    auto feature = extractRgbHistogram(image);
                    float distance = histogramIntersectionDistance(targetFeature, feature);
                    matches.push_back({file, distance});
                }
            } else if (featureType == "multi_histogram") {
                int binsPerChannel = 8;
                int regionCount = 2;
                auto targetFeature =
                    extractMultiRegionRgbHistogram(targetImage, binsPerChannel, regionCount);
                size_t binsPerHistogram = static_cast<size_t>(binsPerChannel * binsPerChannel * binsPerChannel);
                std::vector<float> weights(regionCount, 1.0f);
                for (const auto &file : imageFiles) {
                    cv::Mat image = loadImageOrThrow(file);
                    auto feature =
                        extractMultiRegionRgbHistogram(image, binsPerChannel, regionCount);
                    float distance = histogramIntersectionDistanceMulti(
                        targetFeature, feature, binsPerHistogram, regionCount, weights);
                    matches.push_back({file, distance});
                }
            } else if (featureType == "texture_color") {
                auto targetColor = extractRgbHistogram(targetImage);
                auto targetTexture = extractSobelMagnitudeHistogram(targetImage);
                for (const auto &file : imageFiles) {
                    cv::Mat image = loadImageOrThrow(file);
                    auto color = extractRgbHistogram(image);
                    auto texture = extractSobelMagnitudeHistogram(image);
                    float colorDistance = histogramIntersectionDistance(targetColor, color);
                    float textureDistance = histogramIntersectionDistance(targetTexture, texture);
                    float distance = (colorDistance + textureDistance) * 0.5f;
                    matches.push_back({file, distance});
                }
            } else if (featureType == "custom_sunset") {
                int binsPerChannel = 8;
                int regionCount = 3;
                auto targetFeature =
                    extractCustomSunsetHistogram(targetImage, binsPerChannel, regionCount);
                size_t binsPerHistogram = static_cast<size_t>(binsPerChannel * binsPerChannel * binsPerChannel);
                std::vector<float> weights = {0.2f, 0.3f, 0.5f};
                for (const auto &file : imageFiles) {
                    cv::Mat image = loadImageOrThrow(file);
                    auto feature =
                        extractCustomSunsetHistogram(image, binsPerChannel, regionCount);
                    float distance = histogramIntersectionDistanceMulti(
                        targetFeature, feature, binsPerHistogram, regionCount, weights);
                    matches.push_back({file, distance});
                }
            } else {
                std::cerr << "Unknown feature type: " << featureType << "\n";
                printUsage();
                return 1;
            }
        }

        auto top = topMatches(matches, topN, showLeast);
        for (const auto &match : top) {
            std::cout << match.filename << " " << match.distance << "\n";
        }
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
