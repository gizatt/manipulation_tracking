#include <stdexcept>
#include <iostream>

#include "common/common.hpp"

#include "ObjectScan.hpp"

#include "drake/systems/plants/RigidBodyTree.h"

#include <lcm/lcm-cpp.hpp>

#include "yaml-cpp/yaml.h"
#include "common/common.hpp"
#include "unistd.h"
#include <mutex>

using namespace std;
using namespace Eigen;

int main(int argc, char** argv) {
  if (argc != 2){
    printf("Use: runObjectScanner <path to yaml config file or \"none\">\n");
    return -1;
  }

  std::shared_ptr<lcm::LCM> lcm(new lcm::LCM());
  if (!lcm->good()) {
    throw std::runtime_error("LCM is not good");
  }

  //
  // Load configuration
  //
  string configFile(argv[1]);

  if (configFile == "none")
    YAML::Node config;
  else
    YAML::Node config = YAML::LoadFile(configFile); 

  printf("\\O\\\n");
  usleep(1000*100);
  printf("/O/\n");
  usleep(1000*100);
  printf("\\O\\\n");
  usleep(1000*100);
  printf("/O/\n");
  usleep(1000*100);
  printf("\\O\\\n");
  usleep(1000*100);
  printf("/O/\n");
  usleep(1000*100);
  printf("\\O\\\n");
  usleep(1000*100);
  printf("/O/\n");
  usleep(1000*100);

  return 0;
}