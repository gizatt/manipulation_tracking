
#include "ManipulationTracker.hpp"
#include "RobotStateCost.hpp"
#include "yaml-cpp/yaml.h"

using namespace std;
using namespace Eigen;

static double getUnixTime(void)
{
    struct timespec tv;

    if(clock_gettime(CLOCK_REALTIME, &tv) != 0) return 0;

    return (tv.tv_sec + (tv.tv_nsec / 1000000000.0));
}

int main(int argc, char** argv) {
  const char* drc_path = std::getenv("DRC_BASE");
  if (!drc_path) {
    throw std::runtime_error("environment variable DRC_BASE is not set");
  }

  if (argc != 2){
    printf("Use: runManipulationTrackerIRB140 <path to yaml config file>\n");
    return 0;
  }


  std::shared_ptr<lcm::LCM> lcm(new lcm::LCM());
  if (!lcm->good()) {
    throw std::runtime_error("LCM is not good");
  }

  string configFile(argv[1]);
  YAML::Node config = YAML::LoadFile(configFile);

  VectorXd x0_robot;
  std::shared_ptr<RigidBodyTree> robot = setupRobotFromConfig(config, x0_robot, string(drc_path), true);

  // initialize tracker itself
  ManipulationTracker estimator(robot, x0_robot, lcm, true);

  // and register all of the costs that we know how to handle
  if (config["costs"]){
    
    if (config["costs"]["RobotStateCost"]){
      std::shared_ptr<RobotStateCost> state_cost(new RobotStateCost(robot, lcm, config["costs"]["RobotStateCost"]));
      estimator.addCost(dynamic_pointer_cast<ManipulationTrackerCost, RobotStateCost>(state_cost));
    }

  }

  std::cout << "Manipulation Tracker Listening" << std::endl;

  double last_update_time = getUnixTime();
  double timestep = 0.01;
  if (config["timestep"])
    timestep = config["timestep"].as<double>();

  while(1){
    for (int i=0; i < 100; i++)
      lcm->handleTimeout(0);

    double dt = getUnixTime() - last_update_time;
    if (dt > timestep){
      last_update_time = getUnixTime();
      estimator.update();
      estimator.publish();
    }
  }

  return 0;
}