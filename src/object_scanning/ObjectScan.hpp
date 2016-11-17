#ifndef OBJECT_SCAN_H
#define OBJECT_SCAN_H

#include <stdexcept>
#include <iostream>
#include <memory>
#include <mutex>
#include "yaml-cpp/yaml.h"

class ObjectScan {
public:
  ObjectScan(YAML::Node config);
  ~ObjectScan() {};
  void update();
private:
};

#endif