APP_NAME = cbir
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)

SRC_DIR = src
SOURCES = $(SRC_DIR)/main.cpp \
		  $(SRC_DIR)/feature_extraction.cpp \
		  $(SRC_DIR)/distance_metrics.cpp \
		  $(SRC_DIR)/image_io.cpp

all: $(APP_NAME)

$(APP_NAME): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(APP_NAME) $(OPENCV_FLAGS)

clean:
	rm -f $(APP_NAME)
