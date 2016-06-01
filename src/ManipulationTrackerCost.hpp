#ifndef MANIPULATION_TRACKER_COST_H
#define MANIPULATION_TRACKER_COST_H

#include <Eigen/Dense>
#include "ManipulationTracker.hpp"

// forward declaration
class ManipulationTracker;

class ManipulationTrackerCost {
  public:
    // handed references to Q, f matrices for the cost function to construct
    // returns a bool, which indicates whether this cost should be used this cycle
    virtual bool constructCost(ManipulationTracker * tracker, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& Q, Eigen::Matrix<double, Eigen::Dynamic, 1>& f, double& K) { return false; }
    
    // handed reference reference to a vector of this cost's extra decision variables
    // (but not robot state) and a square matrix of that width.
    // should populate the vector with updates to the state variables,
    // and populate the matrix with the jacobian thereof
    // returns bool, returning whether these updates should be used this cycle
    virtual bool constructPredictionMatrices(ManipulationTracker * tracker, Eigen::Matrix<double, Eigen::Dynamic, 1>& x, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& F) { return false; }

    int getNumExtraVars(void) { return num_extra_vars_; }
  private:
  	int num_extra_vars_ = 0; // extra decision variables needed by this cost. must be set in constructor
};

#endif