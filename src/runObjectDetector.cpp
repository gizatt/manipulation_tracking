#include <stdexcept>
#include <iostream>

#include "drake/systems/plants/RigidBodyTree.h"

#include <lcm/lcm-cpp.hpp>

#include "yaml-cpp/yaml.h"
#include "common/common.hpp"
#include "unistd.h"

using namespace std;
using namespace Eigen;

int main(int argc, char** argv) {
  if (argc != 2){
    printf("Use: runObjectDetector <optional path to yaml config file>\n");
  }

  std::shared_ptr<lcm::LCM> lcm(new lcm::LCM());
  if (!lcm->good()) {
    throw std::runtime_error("LCM is not good");
  }

  char * configFile = NULL;
  YAML::Node config;
  if (argc >= 2){
    configFile = argv[1];
    config = YAML::LoadFile(string(configFile)); 
  }

  std::cout << "Object Detector Listening" << std::endl;

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
      printf("Update!\n");
    } else {
      usleep(1000);
    }
  }

  return 0;
}