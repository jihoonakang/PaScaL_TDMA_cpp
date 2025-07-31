#include <vector>
#include <iostream>
#include <mpi.h>
#include "global_params.hpp"
#include "domain_layout_3d.hpp"
#include "comm_layout_3d.hpp"
#include "pascal_tdma.hpp"

class ConvectionSolver {
    public:
        static void solveThetaMany(dimArray<double>& theta,
                                   const DomainLayout3D& dom3D,
                                   const CommLayout3D& com3D,
                                   const GlobalParams& params);
};