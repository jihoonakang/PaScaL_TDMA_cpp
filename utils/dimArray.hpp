#pragma once

#include <vector>
#include <iostream>
#include <cassert>

enum class ShapeType { None, Dim2D, Dim3D };

template <class T>
class dimArray {
    std::vector<T> data;
    int dim2d_x = 0, dim2d_y = 0;
    int dim3d_x = 0, dim3d_y = 0, dim3d_z = 0;
    ShapeType shape_type = ShapeType::None;

public:
    dimArray() = default;

    dimArray(int x) { resize(x); }
    dimArray(int x, int y) { resize(x, y); }
    dimArray(int x, int y, int z) { resize(x, y, z); }

    dimArray(const std::vector<T>& vec, int x, int y)
        : data(vec), dim2d_x(x), dim2d_y(y), shape_type(ShapeType::Dim2D) {
        assert(static_cast<int>(vec.size()) == x * y);
    }

    template <typename InputIterator>
    dimArray(InputIterator first, InputIterator last, int x, int y) :
        data(first, last), dim2d_x(x), dim2d_y(y), 
        shape_type(ShapeType::Dim2D) {
        assert(std::distance(first, last) == x * y);
    }

    template <typename InputIterator>
    dimArray(InputIterator first, InputIterator last, int x, int y, int z) :
        data(first, last), dim3d_x(x), dim3d_y(y), dim3d_z(z), 
        shape_type(ShapeType::Dim3D) {
        assert(std::distance(first, last) == x * y * z);
    }

    inline T& operator()(int x, int y) noexcept {
        assert(x < dim2d_x && y < dim2d_y);
        return data[x * dim2d_y + y];
    }

    inline const T& operator()(int x, int y) const noexcept {
        assert(x < dim2d_x && y < dim2d_y);
        return data[x * dim2d_y + y];
    }

    inline T& operator()(int x, int y, int z) noexcept {
        assert(x < dim3d_x && y < dim3d_y && z < dim3d_z);
        return data[x * dim3d_y * dim3d_z + y * dim3d_z + z];
    }

    inline const T& operator()(int x, int y, int z) const noexcept {
        assert(x < dim3d_x && y < dim3d_y && z < dim3d_z);
        return data[x * dim3d_y * dim3d_z + y * dim3d_z + z];
    }

    void resize(int x) {
        data.resize(x, T{});
        shape_type = ShapeType::None;
    }

    void resize(int x, int y) {
        data.resize(x * y, T{});
        dim2d_x = x;
        dim2d_y = y;
        shape_type = ShapeType::Dim2D;
    }

    void resize(int x, int y, int z) {
        data.resize(x * y * z, T{});
        dim3d_x = x;
        dim3d_y = y;
        dim3d_z = z;
        shape_type = ShapeType::Dim3D;
    }

    void assign(int n, const T& value) {
        data.assign(n, value);
        shape_type = ShapeType::None;
    }

    void assign(int x, int y, const T& value) {
        data.assign(x * y, value);
        dim2d_x = x;
        dim2d_y = y;
        shape_type = ShapeType::Dim2D;
    }

    void assign(int x, int y, int z, const T& value) {
        data.assign(x * y * z, value);
        dim3d_x = x;
        dim3d_y = y;
        dim3d_z = z;
        shape_type = ShapeType::Dim3D;
    }

    void convert2D(int x, int y) {
        assert(static_cast<int>(data.size()) == x * y);
        dim2d_x = x;
        dim2d_y = y;
        shape_type = ShapeType::Dim2D;
    }

    void convert3D(int x, int y, int z) {
        assert(static_cast<int>(data.size()) == x * y * z);
        dim3d_x = x;
        dim3d_y = y;
        dim3d_z = z;
        shape_type = ShapeType::Dim3D;
    }

    ShapeType getShapeType() const noexcept {
        return shape_type;
    }

    void clear() {
        data.clear();
        shape_type = ShapeType::None;
    }

    T* getData() noexcept { return data.data(); }
    auto& getVector() noexcept { return data; }
    size_t getSize() const noexcept { return data.size(); }

    void print2D() const {
        // assert(shape_type == ShapeType::Dim2D);
        for (int i = 0; i < dim2d_x; i++) {
            for (int j = 0; j < dim2d_y; j++)
                std::cout << (*this)(i, j) << ' ';
            std::cout << '\n';
        }
    }
};