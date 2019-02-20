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

global_odom = None

once_flag = True
insert_num = 5

COLOR_RED = [0, 0, 255]
COLOR_BLUE = [255, 0, 0]
COLOR_GREEN = [0, 255, 0]

LINE_LENGTH_MIN = 0.40   #meter
LINE_LENGTH_MAX = 0.50   #meter

RANGE_MAX = 1.50
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

intersector_position = Position()
cur_target_position = Position()
pre_target_position = Position()
in_front_of_charging_pile_position = Position()

cur_target_odom = None

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
                ########
                img[750 - int((y) * times), 750 - int((x) * times)] = COLOR_GREEN
                ########
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
                #print "angel: ", angel

                if angel > RANGE_ANGLE_MIN and angel < RANGE_ANGLE_MAX:

                    cv2.line(img_2, (750 - int(_line[0][0] * times), 750 - int((_line[0][0] * k1 + b1) * times)),  \
                                    (750 - int(_line[len(_line) / 2 - 1][0] * times), 750 - int((_line[len(_line) / 2 - 1][0] * k1 + b1) * times)), (255, 255, 255), 1)
                    cv2.line(img_2, (750 - int(_line[len(_line) / 2][0] * times), 750 - int((_line[len(_line) / 2][0] * k2 + b2) * times)),  \
                                    (750 - int(_line[len(_line) - 1][0] * times), 750 - int((_line[len(_line) - 1][0] * k2 + b2) * times)), (255, 255, 255), 1)
                    [point_x, point_y] = cal_point_of_intersection([[k1, b1], [k2, b2]])
                    #print "intersection point  x: ", point_x, "y: ", point_y
                    global intersector_position
                    intersector_position.x = point_y
                    intersector_position.y = point_x
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
                    if global_cnt >= 1:
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
                        L = 0.30 + 0.53
                        tmp_x = 0
                        if k < 0:
                            tmp_x = ((x0 + k*y0 - k*b)   +   (( k*k*L*L -k*k*x0*x0 + 2*k*x0*y0 -2*k*b*x0 + 2*b*y0 -y0*y0 -b*b + L*L ) ** 0.5) ) / ( 1 + k*k)
                        else:
                            tmp_x = ((x0 + k*y0 - k*b)   -   (( k*k*L*L -k*k*x0*x0 + 2*k*x0*y0 -2*k*b*x0 + 2*b*y0 -y0*y0 -b*b + L*L ) ** 0.5) ) / ( 1 + k*k)
                        tmp_y = k*tmp_x + b
                        #img_2[(750 - int((tmp_y) * times) - 3) : (750 - int((tmp_y) * times) + 3), (750 - int((tmp_x) * times) - 3) : (750 - int((tmp_x) * times) + 3)] = COLOR_BLUE
                        img_2[(750 - int((in_front_of_charging_pile_position.y) * times) - 3) : (750 - int((in_front_of_charging_pile_position.y) * times) + 3), (750 - int((in_front_of_charging_pile_position.x) * times) - 3) : (750 - int((in_front_of_charging_pile_position.x) * times) + 3)] = COLOR_BLUE

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
    global once_flag
    if once_flag == True:
        img[747:753, 747:753] = COLOR_BLUE
        img_2[747:753, 747:753] = COLOR_BLUE
        cv2.imshow('img-test',img)
        cv2.imshow('img-test_2',img_2)

        cv2.waitKey(1)


def odom_callback(odom):
    global global_odom
    global_odom = odom.pose.pose

def pub_twist(linear = 0.00, angular = 0.00):
    global velocity_pub
    velocity = Twist()
    velocity.linear.x = linear
    velocity.angular.z = angular
    pub_velocity.publish(velocity)

def cal_target_point(x0, y0, k, b, L):
    #L = 0.50 + 0.53
    target_x = 0
    target_y = 0
    if k < 0:
        target_x = ((x0 + k*y0 - k*b)   +   (( k*k*L*L -k*k*x0*x0 + 2*k*x0*y0 -2*k*b*x0 + 2*b*y0 -y0*y0 -b*b + L*L ) ** 0.5) ) / ( 1 + k*k)
    else:
        target_x = ((x0 + k*y0 - k*b)   -   (( k*k*L*L -k*k*x0*x0 + 2*k*x0*y0 -2*k*b*x0 + 2*b*y0 -y0*y0 -b*b + L*L ) ** 0.5) ) / ( 1 + k*k)
    target_y = k*target_x + b
    return [target_x, target_y]


def move_to_target(tmp):
    line_velocity = 0
    angular_velocity = 0
    step = 1
    pre_step = 0
    L = 0.45 + 0.53
    L_delta = 0.02
    th_odom = 0
    my_position = None
    vx = 0
    vth = 0

    k = target_bisector.k
    #### to target ####
    delta_th = 0
    delta_x = 0
    delta_y = 0
    distance = 0
    confirm_cnt = 0
    max_vx = 0.2
    max_vth = 0.2
    global target_position
    global target_bisector
    global global_odom

    global in_front_of_charging_pile_position
    global cur_target_position
    global cur_target_odom
    time.sleep(3)
    while not rospy.is_shutdown():
        if 1:
            time.sleep(0.03)
            #print_obj(target_position)
            #print_obj(target_bisector)
            target_x = 0
            target_y = 0
            if step <= 4:                ##### move to the position 0.2m in front of charging pile
                                         ### calculate the posiont of 0.2m in front of charging pile ###
                ###
                x0 = target_position.x
                y0 = target_position.y
                k = target_bisector.k
                b = target_bisector.b
                if step <= 3:
                    [target_x, target_y] = cal_target_point(x0, y0, k, b, 0.53 + 0.55)
                else:
                    [target_x, target_y] = cal_target_point(x0, y0, k, b, 0.53 + 0.16)
                in_front_of_charging_pile_position.x = target_x
                in_front_of_charging_pile_position.y = target_y
                ### calculate th
                if abs(target_x) >= 0.0001:
                    th = np.arctan(target_y / target_x)
                else:
                    th = np.pi / 2
                #print "th:", th
                #print "target_x: ", target_x, "   target_y:", target_y

                delta_x = cur_target_odom[0] - global_odom.position.x
                delta_y = cur_target_odom[1] - global_odom.position.y
                distance = (delta_x**2 + delta_y**2)**0.5
                arcsin = np.arcsin(delta_y / distance)
                arcsin_d = 0
                if delta_x >= 0:
                    if delta_y >= 0:
                        arcsin_d = arcsin
                    else:
                        arcsin_d = arcsin
                else:
                    if delta_y >= 0:
                        arcsin_d = np.pi - arcsin
                    else:
                        arcsin_d = -arcsin - np.pi
                delta_th = arcsin_d - 2*np.arcsin(global_odom.orientation.z)
                print "delta_th", delta_th
                print "distance: ", distance
                print "vth: ", vth
                print "vx: ", vx
                print "k: ", k
                print "step: ", step
                print 'cur_target_odom :', cur_target_odom
                print 'cur_odom x: ', global_odom.position.x, " y: ", global_odom.position.y
                print 'cur_target_position:', print_obj(cur_target_position)

            if step == 1:
                vx = 0
                if (target_position.x ** 2 + target_position.y ** 2) ** 0.5 >= 0.2:
                    ###
                    cur_target_position = in_front_of_charging_pile_position
                    #cur_target_position = target_position
                    if pre_step != 1:
                        my_position = global_odom
                        th_odom = th
                        pre_step = 1
                        time.sleep(0.3)
                    #print "delta_th", delta_th
                    if abs(delta_th) <= 0.015:
                        vx = 0
                        v_th = 0
                        confirm_cnt += 1
                        if confirm_cnt >= 3:
                            confirm_cnt = 0
                            step = 2
                            print "goto step 2"
                            time.sleep(0.5)
                    else:
                        _th = 0
                        #sign = 1
                        if abs(delta_th) > 0.10:
                            _th = delta_th
                        else:
                            if delta_th > 0:
                                _th = 0.03
                            else:
                                _th = -0.03
                        if delta_th > 0:
                            vth = min(max_vth, _th)
                            #sign = 1
                            #pub_twist(0, 0.05)
                        else:
                            #sign = -1
                            vth = max(-max_vth, _th)
                            #pub_twist(0, -0.05)

                        #_th = min(0.1, _th)
                        pub_twist(0, vth)
                        #print "publish angular velocity"

            elif step == 2:
                cur_target_position = in_front_of_charging_pile_position
                #cur_target_position = target_position
                if pre_step != 2:
                    pre_step = 2
                    time.sleep(0.3)
                #print "step = 2"
                #if abs(th) > 0.05:
                if 0:
                    #step = 1
                    pass
                else:
                    if 0:
                        pass
                    else:
                        _th = 0
                        if abs(delta_th) > 0.05:
                            _th = delta_th
                        else:
                            _th = 0.03
                        if delta_th > 0:
                            vth = min(max_vth, _th)
                        else:
                            vth = max(-max_vth, _th)

                    if distance >= 0.05:
                        vx = min(max_vx, distance / 2)
                    if distance < 0.025:
                        step = 3
                        print "goto step 3"
                        continue
                    else:
                        #print "publish line velocity"
                        pub_twist(vx, vth)
            elif step == 3:
                cur_target_position = target_position
                #cur_target_position = in_front_of_charging_pile_position
                #cur_target_position = target_position
                vx = 0
                if pre_step != 3:
                    pre_step = 3
                    time.sleep(0.3)
                if abs(delta_th) <= 0.015:
                    vx = 0
                    v_th = 0
                    confirm_cnt += 1
                    if confirm_cnt >= 3:
                        confirm_cnt = 0
                        step = 4
                        print "goto step 4"
                        time.sleep(0.5)
                else:
                    _th = 0
                    if abs(delta_th) > 0.10:
                        _th = delta_th
                    else:
                        if delta_th > 0:
                            _th = 0.03
                        else:
                            _th = -0.03
                    if delta_th > 0:
                        vth = min(max_vth / 1.5, _th)
                        #pub_twist(0, 0.05)
                    else:
                        vth = max(-max_vth / 1.5, _th)
                        #pub_twist(0, -0.05)

                    #_th = min(0.1, _th)
                    pub_twist(0, vth)
                    #print "publish angular velocity"

            elif step == 4:
                cur_target_position = target_position
                #cur_target_position = in_front_of_charging_pile_position
                if pre_step != 4:
                    pre_step = 4
                    time.sleep(0.5)

                if 0:
                    #step = 1
                    pass
                else:
                    #if abs(k) > 50.0:
                        #_th = 0
                    #else:
                        #_th = 1.0 / k
                    #if abs(delta_th) < 0.02:
                        #_th = 0
                    #else:
                        #if delta_th > 0:
                            #_th = 0.03
                        #else:
                            #_th = -0.03
                    _y = target_position.x
                    #_y = target_x
                    print "_y: ", _y
                    if abs(_y) < 0.015:
                        _th = 0
                    else:
                        #if target_position.y > 0:
                        _th = _y
                    #if k > 0:
                    #if delta_th > 0:
                    if _y > 0:
                        vth = min(max_vth / 2.0, _th)
                    else:
                        vth = max(-max_vth / 2.0, _th)

                    if distance >= 0.05:
                        vx = min(0.03, distance / 2)
                    if distance < 0.05:

                        print "done ! "
                        break
                    else:
                        #print "publish line velocity"
                        pub_twist(vx, vth)
        else:
            time.sleep(1)

def pub_tf_test(tmp):
    global intersector_position
    global cur_target_position
    while not rospy.is_shutdown():
        #print_obj(cur_target_position)
        y = cur_target_position.x
        x = cur_target_position.y
        cur_target_position
        br = tf.TransformBroadcaster()
        br.sendTransform((x, y, 0),
                        tf.transformations.quaternion_from_euler(0, 0, 0),
                        rospy.Time.now(),
                        "in_front_of_charging_pile",
                        "laser"
                        )
        time.sleep(0.05)

def transform_test(tmp):
    listener = tf.TransformListener()
    global cur_target_odom
    while not rospy.is_shutdown():
        try:
            (cur_target_odom, rot) = listener.lookupTransform('odom', 'in_front_of_charging_pile', rospy.Time(0))
            #print 'cur_target_odom:', cur_target_odom
            #print 'rot:', rot
            time.sleep(0.05)
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




