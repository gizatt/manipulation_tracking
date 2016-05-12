#ifndef MANIP_TRACKING_COMMON_H
#define MANIP_TRACKING_COMMON_H

#include <Eigen/Dense>

static double getUnixTime(void)
{
    struct timespec tv;

    if(clock_gettime(CLOCK_REALTIME, &tv) != 0) return 0;

    return (tv.tv_sec + (tv.tv_nsec / 1000000000.0));
}

// from https://forum.kde.org/viewtopic.php?f=74&t=91514
template<typename Derived>
inline bool is_finite(const Eigen::MatrixBase<Derived>& x)
{
   return ( (x - x).array() == (x - x).array()).all();
}
template<typename Derived>
inline bool is_nan(const Eigen::MatrixBase<Derived>& x)
{
   return ((x.array() == x.array())).all();
}

#endif