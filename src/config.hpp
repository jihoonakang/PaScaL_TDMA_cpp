/**
 * @file cofig.hpp
 * @brief Define thread numbers for CUDA.
 *
 */

#pragma once

#ifdef CUDA
    #define THREADS_X 8
    #define THREADS_Y 8
    #define THREADS_Z 1
#endif