/**
 * @file dimArray.hpp
 * @brief Provides a 2D/3D array interface on top of std::vector.
 *
 * The dimArray class template allows resizing, indexing, conversion and
 * bounds checking among 1D, 2D, and 3D views of a contiguous data buffer.
 */
#pragma once

#include <cassert>   /**< For assert checks */
#include <iostream>  /**< For debug printing */
#include <vector>    /**< Underlying data container */

enum class ShapeType { None, Dim2D, Dim3D };    ///< Dimensionality of the array

/**
 * @class dimArray
 * @brief Multi-dimensional array wrapper around std::vector.
 *
 * @tparam T Element type stored in the array.
 *
 * @note Supports 1D resize and assign, as well as 2D/3D indexing
 *       with operator() overloads.
 */
template <class T>
class dimArray {
    std::vector<T> data_;                           /**< Contiguous data buf */
    ShapeType shape_type_ = ShapeType::None;        /**< Current shape */
    int dim2d_x_ = 0, dim2d_y_ = 0;                 /**< Size of 2D shape */
    int dim3d_x_ = 0, dim3d_y_ = 0, dim3d_z_ = 0;   /**< Size of 3D shape */

public:
public:
    /// Default constructor initializes an empty array.
    dimArray() noexcept = default;

    /// Constructs a 1D array of length x.
    explicit dimArray(int x) noexcept { resize(x); }

    /// Constructs a 2D array of size x by y.
    dimArray(int x, int y) noexcept { resize(x, y); }

    /// Constructs a 3D array of size x by y by z.
    dimArray(int x, int y, int z) noexcept { resize(x, y, z); }

    /// Constructs a 2D array from an existing vector - vec.
    dimArray(const std::vector<T>& vec, int x, int y)
        : data_(vec), dim2d_x_(x), dim2d_y_(y), shape_type_(ShapeType::Dim2D) {
        assert(static_cast<int>(data_.size()) == x * y);
    }

    /// Constructs a 3D array from an existing vector - vec.
    dimArray(const std::vector<T>& vec, int x, int y, int z)
        : data_(vec), dim3d_x_(x), dim3d_y_(y), dim3d_z_(z), 
          shape_type_(ShapeType::Dim3D) {
        assert(static_cast<int>(data_.size()) == x * y * z);
    }

    /// Constructs a 2D array from an iterator range, first to last.
    template <typename InputIt>
    dimArray(InputIt first, InputIt last, int x, int y)
        : data_(first, last), dim2d_x_(x), dim2d_y_(y), shape_type_(ShapeType::Dim2D) {
        assert(std::distance(first, last) == x * y);
    }

    /// Constructs a 3D array from an iterator range, first to last.
    template <typename InputIt>
    dimArray(InputIt first, InputIt last, int x, int y, int z)
        : data_(first, last), dim3d_x_(x), dim3d_y_(y), dim3d_z_(z),
          shape_type_(ShapeType::Dim3D) {
        assert(std::distance(first, last) == x * y * z);
    }

    /// (i, j) 2D element access with bound checking.
    inline T& operator()(int i, int j) noexcept {
#ifdef DEBUG
        assert(shape_type_ == ShapeType::Dim2D);
        assert(i >= 0 && i < dim2d_x_ && j >= 0 && j < dim2d_y_);
#endif
        return data_[i * dim2d_y_ + j];
    }

    /// (i, j) const 2D element access with bound checking.
    inline const T& operator()(int i, int j) const noexcept {
#ifdef DEBUG
        assert(shape_type_ == ShapeType::Dim2D);
        assert(i >= 0 && i < dim2d_x_ && j >= 0 && j < dim2d_y_);
#endif
        return data_[i * dim2d_y_ + j];
    }

    /// (i, j, k) 3D element access with bound checking.
    inline T& operator()(int i, int j, int k) noexcept {
#ifdef DEBUG
        assert(shape_type_ == ShapeType::Dim3D);
        assert(i >= 0 && i < dim3d_x_ && j >= 0 && j < dim3d_y_ && 
               k >= 0 && k < dim3d_z_);
#endif
        return data_[i * dim3d_y_ * dim3d_z_ + j * dim3d_z_ + k];
    }

    /// (i, j, k) const 3D element access with bound checking.
    inline const T& operator()(int i, int j, int k) const noexcept {
#ifdef DEBUG
        assert(shape_type_ == ShapeType::Dim3D);
        assert(i >= 0 && i < dim3d_x_ && j >= 0 && j < dim3d_y_ && 
               k >= 0 && k < dim3d_z_);
#endif
        return data_[i * dim3d_y_ * dim3d_z_ + j * dim3d_z_ + k];
    }

    /// Resizes to a 1D array of length x.
    void resize(int x) noexcept {
        data_.resize(x, T{});
        shape_type_ = ShapeType::None;
    }

    /// Resizes to a 2D array of size x by y.
    void resize(int x, int y) {
        data_.resize(x * y, T{});
        dim2d_x_ = x;
        dim2d_y_ = y;
        shape_type_ = ShapeType::Dim2D;
    }

    /// Resizes to a 3D array of size x by y by z.
    void resize(int x, int y, int z) {
        data_.resize(x * y * z, T{});
        dim3d_x_ = x;
        dim3d_y_ = y;
        dim3d_z_ = z;
        shape_type_ = ShapeType::Dim3D;
    }

    /// Assigns n elements with a given value (1D).
    void assign(int n, const T& value) {
        data_.assign(n, value);
        shape_type_ = ShapeType::None;
    }

    /// Assigns x*y elements with a given value (2D).
    void assign(int x, int y, const T& value) {
        data_.assign(x * y, value);
        dim2d_x_ = x;
        dim2d_y_ = y;
        shape_type_ = ShapeType::Dim2D;
    }

    /// Assigns x*y*z elements with a given value (3D).
    void assign(int x, int y, int z, const T& value) {
        data_.assign(x * y * z, value);
        dim3d_x_ = x;
        dim3d_y_ = y;
        dim3d_z_ = z;
        shape_type_ = ShapeType::Dim3D;
    }

    /// Converts data buffer to 2D without realloc.
    void convert2D(int x, int y) {
        assert(static_cast<int>(data_.size()) == x * y);
        dim2d_x_ = x;
        dim2d_y_ = y;
        shape_type_ = ShapeType::Dim2D;
    }

    /// Converts data buffer to 3D without realloc.
    void convert3D(int x, int y, int z) {
        assert(static_cast<int>(data_.size()) == x * y * z);
        dim3d_x_ = x;
        dim3d_y_ = y;
        dim3d_z_ = z;
        shape_type_ = ShapeType::Dim3D;
    }

    /// Returns the current shape type.
    ShapeType getShapeType() const noexcept {
        return shape_type_;
    }

    /// Clears data and resets shape to None.
    void clear() noexcept {
        data_.clear();
        shape_type_ = ShapeType::None;
    }

    /// Returns a pointer to the internal data array.
    T* getData() noexcept { return data_.data(); }

    /// Returns a const pointer to the internal data array.
    const T* getData() const noexcept { return data_.data(); }

    /// Returns a reference to the internal std::vector.
    std::vector<T>& getVector() noexcept { return data_; }

    /// Returns a const reference to the internal std::vector.
    const std::vector<T>& getVector() const noexcept { return data_; }

    /// Returns the total number of elements.
    size_t getSize() const noexcept { return data_.size(); }

    /**
     * @brief Prints the 2D array to stdout for debugging.
     */
    void print2D() const {
        assert(shape_type_ == ShapeType::Dim2D);
        for (int i = 0; i < dim2d_x_; i++) {
            for (int j = 0; j < dim2d_y_; j++)
                std::cout << (*this)(i, j) << ' ';
            std::cout << '\n';
        }
    }
};