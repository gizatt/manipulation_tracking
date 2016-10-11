Real-time Object Tracking (for Manipulation)
======
This repo contains research code used to track objects in real-time,
using e.g. point cloud data, GelSight contact sensor surface geometry
measurements, and external joint state measurements or estimates.

## Overview
The core of the algorithm is the ManipulationTracker class itself,
which runs a modified EKF estimating some overall robot state. The process
update follows the regular EKF process update; the measurement update
is replaced by a direct optimization of the robot state with respect to
measurement probability, with the predicted state from the process update
brought in as a prior. See [Section IV of this paper](http://groups.csail.mit.edu/robotics-center/public_papers/Izatt16.pdf)
for a more complete overview. This optimization is formulated as an 
unconstrained QP, with its cost function being a sum over
a collection of contributing factors. These separate contributing factors are implemented as ManipulationTrackerCost objects. 

Such cost objects are available for:
* Point cloud data (KinectFrameCost)
* Tactile geometry map data (GelsightCost)
* Joint and complete robot state data (JointStateCost and RobotStateCost)
* AprilTag and Optotrak marker detections (AttachedApriltagCost and OptotrakMarkerCost)

Each of these cost objects is restricted by the contract that it must
be able to provide a second-order approximation to its measurement model (see
the template class, ManipulationTrackerCost). Each cost object usually
also contains interfaces to the rest of the system to capture point cloud
or robot state data from LCM messages, along with other debugging or bookkeeping
behavior.

## Using the tracker

A tracker is instantiated by the runManipulationTracker script, which reads
a tracker configuration file (in YAML format, under ```config```) to know
what robot to load, what channels to publish estimated state on, and what
kinds of costs to add with what configurations.

## Integration the code

In its current state, this code needs to be used as part of a distribution
that provides this project's dependencies. For those with access, the private
OpenHumanoids branch [here](https://github.com/oh-dev/oh-distro-private/tree/gizatt_irb140_workstation_icra_push)
works as an example. This project is brought in as a submodule as the 
```software/perception/manipulation_tracker``` subdirectory.

The current build system uses `pods` as the build rule: in short, it looks for
a ```build``` directory up to four directories up from the repository root,
and uses that as the build folder and possible dependency source. I haven't
begun trying to integrate this into projects outside of OH yet, as maintaining
compatibility with `pod-build` alongside a more traditional CMake build system
is something I'm still figuring out.