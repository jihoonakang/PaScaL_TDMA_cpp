#include <vector>
#include <iostream>
#include <mpi.h>
#include "globalParams.hpp"
#include "domainLayout3D.hpp"
#include "commLayout3D.hpp"
#include "PaScaL_TDMA.hpp"

class ConvectionSolver {
    public:
        static void solveThetaMany(dimArray<double>& theta,
                                   const DomainLayout3D& dom3D,
                                   const CommLayout3D& com3D,
                                   const GlobalParams& params);
};