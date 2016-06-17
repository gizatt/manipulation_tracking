#!/usr/bin/python

import lcm
import drc
from bot_core import rigid_transform_t
import sys
import time, math
import numpy
from bot_core.robot_state_t import robot_state_t

_EPS = 1E-12

def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> numpy.allclose(quaternion_from_matrix(R, isprecise=False),
    ...                quaternion_from_matrix(R, isprecise=True))
    True

    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:4, :4]
    if isprecise:
        q = numpy.empty((4, ))
        t = numpy.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = numpy.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = numpy.linalg.eigh(K)
        q = V[[3, 0, 1, 2], numpy.argmax(w)]
    if q[0] < 0.0:
        numpy.negative(q, q)
    return q

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> numpy.allclose(M, numpy.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
    True

    """
    q = numpy.array(quaternion, dtype=numpy.float64, copy=True)
    n = numpy.dot(q, q)
    if n < _EPS:
        return numpy.identity(4)
    q *= math.sqrt(2.0 / n)
    q = numpy.outer(q, q)
    return numpy.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])


robot2local = numpy.identity(4)
robot2local[0:3, 3] = numpy.array([ -0.17, 0, 0.911 ])

kinect2robot = numpy.identity(4)
kinect2robot[0:3, 3] = numpy.array([-0.452, -0.191, 1.21])
kinect2robot[0:3, 0:3] = quaternion_matrix([0.471, 0.77, 0.36, -0.23])[0:3, 0:3]
lc = lcm.LCM()

gt_transform = numpy.identity(4)
have_gt_transform = False

def box_gt_handler(channel, data):
    global have_gt_transform
    global gt_transform
    latest_gt = robot_state_t.decode(data)

    gt_transform = numpy.identity(4)
    gt_transform[0:3, 3] = numpy.array([latest_gt.pose.translation.x,
                                        latest_gt.pose.translation.y,
                                        latest_gt.pose.translation.z])
    gt_transform[0:3, 0:3] = quaternion_matrix(numpy.array([latest_gt.pose.rotation.w,
                                                                    latest_gt.pose.rotation.x,
                                                                    latest_gt.pose.rotation.y,
                                                                    latest_gt.pose.rotation.z]))[0:3, 0:3]
    have_gt_transform = True

def box_state_handler(channel, data):
    if (have_gt_transform):
        latest_state = robot_state_t.decode(data)
        last_transform = numpy.identity(4)
        last_transform[0:3, 3] = numpy.array([latest_state.pose.translation.x,
                                        latest_state.pose.translation.y,
                                        latest_state.pose.translation.z])
        last_transform[0:3, 0:3] = quaternion_matrix(numpy.array([latest_state.pose.rotation.w,
                                                                    latest_state.pose.rotation.x,
                                                                    latest_state.pose.rotation.y,
                                                                    latest_state.pose.rotation.z]))[0:3, 0:3]

        kinect2local = numpy.dot(robot2local, numpy.linalg.inv(kinect2robot))
        local2kinect = numpy.linalg.inv(kinect2local)

        err_local = numpy.dot(numpy.linalg.inv(gt_transform), last_transform)

        # figure out corrections in local frame, then transfer to kinect frmae
        rotation_error = numpy.dot(gt_transform[0:3, 0:3], numpy.linalg.inv(last_transform[0:3, 0:3]))
        extra_correction = -numpy.dot(rotation_error, kinect2local[0:3, 3]) + kinect2local[0:3, 3]
        print extra_correction
        translation_error = gt_transform[0:3, 3] - last_transform[0:3, 3]
        print translation_error

        print numpy.dot(rotation_error, last_transform[0:3, 3]) - last_transform[0:3, 3]

        local_correction = numpy.identity(4)
        local_correction[0:3, 3] = translation_error
        local_correction[0:3, 0:3] = rotation_error

        print "local corr", local_correction

        correction_in_camera = numpy.identity(4)
        #correction_in_camera[0:3, 0:3] = numpy.linalg.inv(rotation_error)
        correction_in_camera[0:3, 3] = -numpy.dot(local2kinect[0:3, 0:3], local_correction[0:3, 3])
        print "camera corr", correction_in_camera



        new_local2kinect = numpy.dot(correction_in_camera, local2kinect)

        # undo each transform in sequence
        final_transform = numpy.dot(new_local2kinect, robot2local)
        print kinect2robot
        print final_transform

        print "\n\n\n"
        msg = rigid_transform_t();
        msg.utime = 0;
        msg.trans = final_transform[0:3, 3]
        msg.quat = quaternion_from_matrix(final_transform[0:3, 0:3])
        print msg.trans, msg.quat
        lc.publish("GT_CAMERA_OFFSET", msg.encode())

lc.subscribe("EST_MANIPULAND_STATE_optotrak_cube_GT", box_gt_handler)
lc.subscribe("EST_MANIPULAND_STATE_optotrak_cube", box_state_handler)
while (1):
    lc.handle()