#!/usr/bin/env python
# coding:utf-8

import sys
import rospy
import time
from sensor_msgs.msg import LaserScan
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2.cv as cv
import cv2
import threading
import time
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
import tf


sub_scan = None
pub_velocity = None
sub_odom = None

global_position = None

once_flag = True
insert_num = 5

COLOR_RED = [0, 0, 255]
COLOR_BLUE = [255, 0, 0]
COLOR_GREEN = [0, 255, 0]

LINE_LENGTH_MIN = 0.40   #meter
LINE_LENGTH_MAX = 0.50   #meter

RANGE_MAX = 2.00
RANGE_MIN = 0.05

RANGE_ANGLE_MIN = -0.49
RANGE_ANGLE_MAX = -0.42

line = []
line_buf = []

global_cnt = 0
sum_target_x = 0
sum_target_y = 0
sum_bisector_k = 0
sum_bisector_b = 0

def print_obj(obj):
    print obj.__dict__

class Position(object):
    def __init__(self):
        self.x = 0
        self.y = 0
        self.time = 0

class Bisector(object):
    def __init__(self):
        self.k = 0
        self.b = 0
        self.th = 0

target_position = Position()
target_bisector = Bisector()

def cal_angel(k):
    tan_k = (k[1] - k[0]) / (1 + k[1] * k [0])
    return np.arctan(tan_k)

def cal_point_of_intersection(_line):
    if len(_line) == 2:
        k1 = _line[0][0]
        k2 = _line[1][0]
        b1 = _line[0][1]
        b2 = _line[1][1]
        if k1 != k2:
            x = (b2 - b1) / (k1 - k2)
            y = (k1 * b2 - k2 * b1) / (k1 - k2)
            return [x, y]
        else:
            return []
    else:
        return []

def cal_bisector(k):
    if len(k) == 2:
        return ((k[0] * k[1] - 1) - ((1 + k[0] * k[1]) ** 2 + (k[0] + k[1]) ** 2) ** 0.5) / (k[0] + k[1])

def fit_line_test(_line):
    sum_xy = 0
    sum_x = 0
    sum_y = 0
    sum_xx = 0
    line_len = len(_line)
    if line_len > 0:
        for point in _line:
            sum_xy = sum_xy + point[0] * point[1]
            sum_x = sum_x + point[0]
            sum_y = sum_y + point[1]
            sum_xx = sum_xx + point[0] * point[0]
        k = (sum_xy * line_len - sum_x * sum_y) / (sum_xx * line_len - sum_x * sum_x)
        b = (sum_xx * sum_y - sum_xy * sum_x) / (sum_xx * line_len - sum_x * sum_x)
        return [k, b]
    else:
        return []

def fit_2_line_test(_line):
    sum_xy_1 = 0
    sum_x_1 = 0
    sum_y_1 = 0
    sum_xx_1 = 0
    sum_xy_2 = 0
    sum_x_2 = 0
    sum_y_2 = 0
    sum_xx_2 = 0
    line_len = len(_line) / 2
    ret = []
    if line_len > 0:
        for i in range(0, line_len):
            sum_xy_1 = sum_xy_1 + _line[i][0] * _line[i][1]
            sum_x_1 = sum_x_1 + _line[i][0]
            sum_y_1 = sum_y_1 + _line[i][1]
            sum_xx_1 = sum_xx_1 + _line[i][0] * _line[i][0]
        k_1 = (sum_xy_1 * line_len - sum_x_1 * sum_y_1) / (sum_xx_1 * line_len - sum_x_1 * sum_x_1)
        b_1 = (sum_xx_1 * sum_y_1 - sum_xy_1 * sum_x_1) / (sum_xx_1 * line_len - sum_x_1 * sum_x_1)
        ret.append([k_1, b_1])

        for i in range(line_len, len(_line)):
            sum_xy_2 = sum_xy_2 + _line[i][0] * _line[i][1]
            sum_x_2 = sum_x_2 + _line[i][0]
            sum_y_2 = sum_y_2 + _line[i][1]
            sum_xx_2 = sum_xx_2 + _line[i][0] * _line[i][0]
        line_len = len(_line) - line_len
        k_2 = (sum_xy_2 * line_len - sum_x_2 * sum_y_2) / (sum_xx_2 * line_len - sum_x_2 * sum_x_2)
        b_2 = (sum_xx_2 * sum_y_2 - sum_xy_2 * sum_x_2) / (sum_xx_2 * line_len - sum_x_2 * sum_x_2)
        ret.append([k_2, b_2])
    else:
        ret = []
    #print ret
    return ret

def cal_max_distance(_line):
    l = len(_line)
    if l > 0:
        return ((_line[l - 1][0] - _line[0][0]) ** 2 + (_line[l - 1][1] - _line[0][1]) ** 2) ** 0.5
    else:
        return 0

def laser_scan_callback(scan):
    global line_buf
    global line
    ranges = scan.ranges
    ranges_len = len(ranges)
    index = len(ranges) / 2
    angle_min = scan.angle_min
    angle_max = scan.angle_max
    angle_all = angle_max - angle_min
    delta_angle = 0.0
    cnt = 0
    x_array = np.array([])
    y_array = np.array([])
    xy_array = np.array([[0, 0], [0, 0]])
    #img = cv.CreateImage((1500, 1500), 8, 3)
    x_last = 0.00
    y_last = 0.00
    img = np.zeros((1000, 1500, 3), np.uint8)
    img_2 = np.zeros((1000, 1500, 3), np.uint8)
    get_flag = False
    for distance in ranges:
        cnt += 1

        delta_angle = angle_all * cnt / ranges_len
        x = distance * np.sin(angle_min + delta_angle)
        y = distance * np.cos(angle_min + delta_angle)
        x_array = np.append(x_array, x)
        y_array = np.append(y_array, y)
        xy_array = np.append(xy_array, [[x, y]], axis = 0)
        times = 150

#        if abs(x) >= 0.01 and abs(y) >= 0.01:
        if distance > RANGE_MIN:
            if distance < RANGE_MAX:
                img[750 - int((y) * times), 750 - int((x) * times)] = COLOR_GREEN

                if abs(x - x_last) < 0.02 and abs(y - y_last) < 0.02:
                    point = [x, y]
                    line.append(point)
                    #img[750 - int((y) * times), 750 - int((x) * times)] = COLOR_RED
                else:
                    if len(line) > 30:
                        line_buf.append(line)
                        line = []
                    else:
                        line  = []
            y_last = y
            x_last = x


    if len(line) > 30:
        line_buf.append(line)

    line = []
    #print line
#    del line[:]

    if len(line_buf) > 0:   #get

        for _line in line_buf:
            _len = cal_max_distance(_line)

            if _len > LINE_LENGTH_MIN and _len < LINE_LENGTH_MAX:

                [[k1, b1], [k2, b2]] = fit_2_line_test(_line)

                angel = cal_angel([k1, k2])

                if angel > RANGE_ANGLE_MIN and angel < RANGE_ANGLE_MAX:

                    cv2.line(img_2, (750 - int(_line[0][0] * times), 750 - int((_line[0][0] * k1 + b1) * times)),  \
                                    (750 - int(_line[len(_line) / 2 - 1][0] * times), 750 - int((_line[len(_line) / 2 - 1][0] * k1 + b1) * times)), (255, 255, 255), 1)
                    cv2.line(img_2, (750 - int(_line[len(_line) / 2][0] * times), 750 - int((_line[len(_line) / 2][0] * k2 + b2) * times)),  \
                                    (750 - int(_line[len(_line) - 1][0] * times), 750 - int((_line[len(_line) - 1][0] * k2 + b2) * times)), (255, 255, 255), 1)
                    [point_x, point_y] = cal_point_of_intersection([[k1, b1], [k2, b2]])
                    #print point_x, point_y
                    img_2[(750 - int((point_y) * times) - 3) : (750 - int((point_y) * times) + 3), (750 - int((point_x) * times) - 3) : (750 - int((point_x) * times) + 3)] = COLOR_BLUE
                    ########


                    ###### test test ####
                    #tmp_x = 0.30
                    #tmp_y = 1.04
                    #img_2[(750 - int((tmp_y) * times) - 3) : (750 - int((tmp_y) * times) + 3), (750 - int((tmp_x) * times) - 3) : (750 - int((tmp_x) * times) + 3)] = COLOR_BLUE
                    ##### #####
                    k_bisector = cal_bisector([k1, k2])
                    b_bisector = point_y - k_bisector * point_x
                    global global_cnt
                    global sum_target_x
                    global sum_target_y
                    global sum_bisector_k
                    global sum_bisector_b
                    sum_target_x += point_x
                    sum_target_y += point_y
                    sum_bisector_k += k_bisector
                    sum_bisector_b += b_bisector
                    global_cnt += 1
                    #print "global cnt", global_cnt
                    if global_cnt >= 5:
                        global target_position
                        global target_bisector
                        target_position.x = sum_target_x / global_cnt
                        target_position.y = sum_target_y / global_cnt
                        target_bisector.k = sum_bisector_k / global_cnt
                        target_bisector.b = sum_bisector_b / global_cnt


                        ######## test code ########
                        x0 = target_position.x
                        y0 = target_position.y
                        k = target_bisector.k
                        b = target_bisector.b
                        L = 0.15
                        tmp_x = 0
                        if k < 0:
                            tmp_x = ((x0 + k*y0 - k*b)   +   (( k*k*L*L -k*k*x0*x0 + 2*k*x0*y0 -2*k*b*x0 + 2*b*y0 -y0*y0 -b*b + L*L ) ** 0.5) ) / ( 1 + k*k)
                        else:
                            tmp_x = ((x0 + k*y0 - k*b)   -   (( k*k*L*L -k*k*x0*x0 + 2*k*x0*y0 -2*k*b*x0 + 2*b*y0 -y0*y0 -b*b + L*L ) ** 0.5) ) / ( 1 + k*k)
                        tmp_y = k*tmp_x + b
                        img_2[(750 - int((tmp_y) * times) - 3) : (750 - int((tmp_y) * times) + 3), (750 - int((tmp_x) * times) - 3) : (750 - int((tmp_x) * times) + 3)] = COLOR_BLUE
                        ########
                        sum_target_x = 0
                        sum_target_y = 0
                        sum_bisector_k = 0
                        sum_bisector_b = 0
                        global_cnt = 0
                    #print "bisector: k, b", [k_bisector, b_bisector]
                    cv2.line(img_2, (750 - int(point_x * times), 750 - int(point_y * times)),  \
                                    (750 - int((( - b_bisector) / k_bisector) * times), 750), (255, 255, 255), 1)
                    for x, y in _line:
                        img_2[750 - int((y) * times), 750 - int((x) * times)] = COLOR_RED

    line_buf = []
    #print line_buf
#    del line_buf[:]
    global once_flag
    if once_flag == True:
        img[747:753, 747:753] = COLOR_BLUE
        img_2[747:753, 747:753] = COLOR_BLUE
        cv2.imshow('img-test',img)
        cv2.imshow('img-test_2',img_2)

        cv2.waitKey(1)


def odom_callback(odom):
    global global_position
    global_position = odom.pose.pose

def pub_twist(linear = 0.00, angular = 0.00):
    global velocity_pub
    velocity = Twist()
    velocity.linear.x = linear
    velocity.angular.z = angular
    pub_velocity.publish(velocity)

def cal_target_point(x0, y0, k, b):
    L = 0.30 + 0.53
    target_x = 0
    target_y = 0
    if k < 0:
        target_x = ((x0 + k*y0 - k*b)   +   (( k*k*L*L -k*k*x0*x0 + 2*k*x0*y0 -2*k*b*x0 + 2*b*y0 -y0*y0 -b*b + L*L ) ** 0.5) ) / ( 1 + k*k)
    else:
        target_x = ((x0 + k*y0 - k*b)   -   (( k*k*L*L -k*k*x0*x0 + 2*k*x0*y0 -2*k*b*x0 + 2*b*y0 -y0*y0 -b*b + L*L ) ** 0.5) ) / ( 1 + k*k)
    target_y = k*target_x + b
    return [target_x, target_y]

def cal_target_odom_point(x, y, k, b):
    [x0, y0] = cal_target_point(x, y, k, b)
    x0 = x0 + global_position.position.x

def move_to_target(tmp):
    line_velocity = 0
    angular_velocity = 0
    step = 1
    pre_step = 0
    L = 0.15 + 0.53
    L_delta = 0.02
    th_odom = 0
    my_position = None
    global target_position
    global target_bisector
    global global_position
    time.sleep(3)
    while not rospy.is_shutdown():
        if 1:
            time.sleep(0.05)
            #print_obj(target_position)
            #print_obj(target_bisector)
            target_x = 0
            target_y = 0
            if step <= 2:                ##### move to the position 0.2m in front of charging pile
                                         ### calculate the posiont of 0.2m in front of charging pile ###
                if (target_position.x ** 2 + target_position.y ** 2) ** 0.5 >= 0.2:
                    ###
                    x0 = target_position.x
                    y0 = target_position.y
                    k = target_bisector.k
                    b = target_bisector.b
                    [target_x, target_y] = cal_target_point(x0, y0, k, b)
                    ### calculate th
                    th = np.arctan(target_x / target_y)
                    print "th:", th
                    print "target_x: ", target_x, "   target_y:", target_y
                    print step

            if step == 1:   ## xjbz
                if (target_position.x ** 2 + target_position.y ** 2) ** 0.5 >= 0.2:
                    ###
                    if pre_step != 1:
                        my_position = global_position
                        th_odom = th
                        pre_step = 1
                    print "step 1 x:", target_x, " y:", target_y
                    delta_th = 2*np.arcsin(global_position.orientation.z) - 2*np.arcsin(my_position.orientation.z) - th_odom
                    print "delta_th", delta_th
                    if abs(delta_th) <= 0.02:
                        step = 2
                        time.sleep(0.5)
                    else:
                        if delta_th < 0:
                            pub_twist(0, 0.02)
                        else:
                            pub_twist(0, -0.02)
                        ##TODO publish augular velocity
                        print "publish angular velocity"

            if step == 2:
                print "step = 2"
                #if abs(th) > 0.05:
                if 0:
                    #step = 1
                    pass
                else:
                    if target_x ** 2 + target_y ** 2 <= (L + L_delta / 2)**2:
                        step = 3
                        print "step = 3"
                    else:
                        ##TODO publish line velocity
                        print "publish line velocity"
                        pub_twist(0.02, 0)
            if step == 3:
                ### TODO: publish line velocity(forward)
                print "start step 3: just go forward"
        else:
            time.sleep(1)

def pub_tf_test(tmp):
    while not rospy.is_shutdown():
        br = tf.TransformBroadcaster()
        br.sendTransform((1, 0.2, 0),
                        tf.transformations.quaternion_from_euler(0, 0, 0),
                        rospy.Time.now(),
                        "in_front_of_charging_pile",
                        "laser"
                        )
        time.sleep(0.1)

def transform_test(tmp):
    listener = tf.TransformListener()
    while not rospy.is_shutdown():
        try:
            (trans, rot) = listener.lookupTransform('odom', 'in_front_of_charging_pile', rospy.Time(0))
            print 'trans:', trans
            print 'rot:', rot
            time.sleep(0.1)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue


def init():
    global pub_velocity
    sub_scan = rospy.Subscriber('scan_filtered', LaserScan, laser_scan_callback, None, 1)
    pub_velocity = rospy.Publisher("charge_pri_cmdvel", Twist, queue_size = 1)
    sub_odom = rospy.Subscriber('odom', Odometry, odom_callback, None, 1)

def de_init():
    sub_scan.unregister()
    sub_odom.unregister()

def main():
    rospy.init_node('dirver_test', anonymous=True)
    init()
    rate = rospy.Rate(10)
    time.sleep(1)
    #test_img = cv2.imread("test.jpg")
    #cv2.imshow('test img', test_img)
#    plt.ion()
#    plt.figure(1)
#    plt.add_subplot(111)
    thread_move_to_target = threading.Thread(target = move_to_target, args = (0,))
    thread_pub_tf_test = threading.Thread(target = pub_tf_test, args = (0,))
    thread_transform_test = threading.Thread(target = transform_test, args = (0,))
    thread_move_to_target.start()
    thread_pub_tf_test.start()
    thread_transform_test.start()
    rospy.spin()
    return

if __name__ == '__main__':
    try:
        main()
    except Exception:
        rospy.logerr(sys.exc_info())
        rospy.loginfo("lost connect")
        exit(1)



