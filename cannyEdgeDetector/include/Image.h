
#pragma once
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>

#include "stb_image.h"

namespace Graphics {

template <typename T>
class Image {
 private:
  std::unique_ptr<T[]> m_data;
  int m_width;
  int m_height;
  int m_channels;

 public:
  // Constructor
  Image() : m_width(0), m_height(0), m_channels(0) {}
  Image(int width, int height, int channels = 3)
      : m_width(width), m_height(height), m_channels(channels) {
    if (width > 0 && height > 0 && channels > 0) {
      m_data =
          std::make_unique<T[]>(static_cast<size_t>(width) * height * channels);
    }
  }

  // Move Constructor / Move Assignment
  Image(Image &&) noexcept = default;
  Image &operator=(Image &&) noexcept = default;

  // Copy Constructor / Copy Assignment are deleted
  Image(const Image &) = delete;
  Image &operator=(const Image &) = delete;

  // Accessors
  T *data() { return m_data.get(); }
  const T *data() const { return m_data.get(); }
  int width() const { return m_width; }
  int height() const { return m_height; }
  int channels() const { return m_channels; }
  size_t totalPixels() const { return static_cast<size_t>(m_width) * m_height; }
  size_t totalElements() const {
    return static_cast<size_t>(m_width) * m_height * m_channels;
  }

  // Operators
  T &operator()(int x, int y, int channel = 0) {
    return m_data[(static_cast<size_t>(y) * m_width + x) * m_channels +
                  channel];
  }
  const T &operator()(int x, int y, int channel = 0) const {
    return m_data[(static_cast<size_t>(y) * m_width + x) * m_channels +
                  channel];
  }

  // Load / Save
  static Image loadPNG(const std::string &filename) {
    int width, height, channels;
    unsigned char *img_data =
        stbi_load(filename.c_str(), &width, &height, &channels, 0);
    if (!img_data) {
      throw std::runtime_error("Failed to load PNG: " + filename);
    }

    Image image(width, height, channels);
    size_t total_elements = static_cast<size_t>(width) * height * channels;
    for (size_t i = 0; i < total_elements; ++i) {
      image.m_data[i] = static_cast<T>(img_data[i]);
    }

    stbi_image_free(img_data);
    return image;
  }

  static Image loadRAW(const std::string &filename, int width, int height,
                       int channels = 3) {
    if (width <= 0 || height <= 0 || channels <= 0) {
      throw std::invalid_argument("Invalid image dimensions");
    }
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open RAW file: " + filename);
    }

    std::streamsize file_size = file.tellg();
    size_t expected_size =
        static_cast<size_t>(width) * height * channels * sizeof(T);
    if (static_cast<size_t>(file_size) != expected_size) {
      throw std::runtime_error("RAW file size mismatch");
    }

    file.seekg(0, std::ios::beg);
    Image image(width, height, channels);
    file.read(reinterpret_cast<char *>(image.m_data.get()), file_size);
    if (!file) {
      throw std::runtime_error("Failed to read RAW file data");
    }
    file.close();
    return image;
  }

  void saveRAW(const std::string &filename) const {
    if (!m_data) {
      throw std::runtime_error("Cannot save empty image");
    }
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to create RAW file: " + filename);
    }

    size_t total_bytes = totalElements() * sizeof(T);
    file.write(reinterpret_cast<const char *>(m_data.get()), total_bytes);
    if (!file) {
      throw std::runtime_error("Failed to write RAW file data");
    }
    file.close();
  }

  Image clone() const {
    Image destination(m_width, m_height, m_channels);
    if (m_data) {
      std::copy(m_data.get(), m_data.get() + totalElements(),
                destination.data());
    }
    return destination;
  }
};

}  // namespace Graphics

// Include implementation of tpp
#include "src/Image.tpp"