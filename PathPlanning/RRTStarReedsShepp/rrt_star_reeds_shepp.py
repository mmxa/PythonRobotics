"""

Path planning Sample Code with RRT with Reeds-Shepp path

author: AtsushiSakai(@Atsushi_twi)

"""
import copy
import math
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../ReedsSheppPath/")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../RRTStar/")

try:
    import reeds_shepp_path_planning
    from rrt_star import RRTStar
except ImportError:
    raise

show_animation = True


def multiply(v1, v2):   # 计算两个向量的叉积
    return v1.x * v2.y - v2.x * v1.y


class Point:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)


class Segment:
    def __init__(self, point1=Point(), point2=Point()):
        self.pt1 = point1
        self.pt2 = point2

    def straddle(self, another_segment):
        """
        :param another_segment: 另一条线段
        :return: 如果另一条线段跨立该线段，返回True；否则返回False
        """
        v1 = another_segment.pt1 - self.pt1
        v2 = another_segment.pt2 - self.pt2
        vm = self.pt2 - self.pt1
        if multiply(v1, vm) * multiply(v2, vm) <= 0:
            return True
        else:
            return False

    def is_cross(self, another_segment):
        """
        :param another_segment: 另一条线段
        :return: 如果两条线段相互跨立，则相交；否则不相交
        """
        if self.straddle(another_segment) and another_segment.straddle(self):
            # res_point = cross_point(self, another_segment)
            return True
        else:
            return False


class RRTStarReedsShepp(RRTStar):
    """
    Class for RRT star planning with Reeds Shepp path
    """

    class Node(RRTStar.Node):
        """
        RRT Node
        """

        def __init__(self, x, y, yaw):
            super().__init__(x, y)
            self.yaw = yaw
            self.path_yaw = []

    def __init__(self, start, goal, obstacle_list, rand_type,
                 max_iter=500,
                 connect_circle_dist=50.0
                 ):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]

        """
        self.start = self.Node(start[0], start[1], start[2])
        self.end = self.Node(goal[0], goal[1], goal[2])

        """
        # vertical parking
        self.min_rand_x = -3.0                      #rand_area[0]
        self.max_rand_x = 3.0                      #rand_area[1]
        self.min_rand_y = 2.0                       # 2.0
        self.max_rand_y = 8.0                       # 8.0
        """
        self.min_rand_x = -3.0  # rand_area[0]
        self.max_rand_x = 0.5  # rand_area[1]
        self.min_rand_y = -1.0  # 2.0
        self.max_rand_y = 4.5  # 8.0
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.connect_circle_dist = connect_circle_dist
        self.rand_type = rand_type
        self.curvature = 0.222  # R = 4.5m
        self.goal_yaw_th = np.deg2rad(10.0)
        self.goal_xy_th = 0.5

    def planning(self, animation=True, search_until_max_iter=True):
        """
        planning

        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            #if i == 1000:
            #    os.system("pause")
            print("Iter:", i, ", number of nodes:", len(self.node_list))
            rand_num = random.uniform(0, 1.0)
            if rand_num < 0.10:
                rnd = self.end
            elif 0.10<=rand_num<=0.20 and self.rand_type == 'V':
                rnd = self.get_random_node_x_axis()
            else:
                rnd = self.get_random_node()
            """if i < 15:      # driving reverse for a certain distance in first 15 iteration
                rnd = self.Node(self.start.x - i * 0.5 * math.cos(self.start.yaw),
                                self.start.y - i * 0.5 * math.sin(self.start.yaw),
                                self.start.yaw)"""
            if i < 1:
                rnd = self.end
            # RAND RADIUS
            # self.curvature = random.uniform(0.18, 0.222)
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd)

            if self.check_collision(new_node, self.obstacle_list):
                near_indexes = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_indexes)
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_indexes)
                    self.try_goal_path(new_node)

            if animation and i % 1 == 0:
                self.plot_start_goal_arrow()
                self.draw_graph(rnd)

            if (not search_until_max_iter) or (new_node and i>500):  # check reaching the goal
                last_index = self.search_best_goal_node()
                if last_index:
                    return self.generate_final_course(last_index)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index:
            return self.generate_final_course(last_index)
        else:
            print("Cannot find path")

        return None

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 +
                 10*(node.yaw - rnd_node.yaw)**2 for node in node_list]

        minind = dlist.index(min(dlist))

        return minind

    def try_goal_path(self, node):

        goal = self.Node(self.end.x, self.end.y, self.end.yaw)

        new_node = self.steer(node, goal)
        if new_node is None:
            return

        if self.check_collision(new_node, self.obstacle_list):
            self.node_list.append(new_node)

    @staticmethod
    def check_collision(node, obstacle_list):
        if node is None:
            return False
        # construct the vehicle profile
        front_length = 3.575
        rear_length = 0.875
        width = 1.800
        length2lf = math.hypot(front_length, width/2.0)
        length2lr = math.hypot(rear_length, width/2.0)
        for i in range(len(node.path_yaw)):
            phi1 = node.path_yaw[i] + math.atan((width/2.0) / front_length)
            phi2 = node.path_yaw[i] - math.atan((width/2.0) / front_length)
            phi3 = node.path_yaw[i] - math.pi/2 - math.atan(rear_length / (width/2.0))
            phi4 = node.path_yaw[i] + math.pi/2 + math.atan(rear_length / (width/2.0))
            point1 = Point(node.path_x[i] + math.cos(phi1) * length2lf,
                           node.path_y[i] + math.sin(phi1) * length2lf)
            point2 = Point(node.path_x[i] + math.cos(phi2) * length2lf,
                           node.path_y[i] + math.sin(phi2) * length2lf)
            point3 = Point(node.path_x[i] + math.cos(phi3) * length2lr,
                           node.path_y[i] + math.sin(phi3) * length2lr)
            point4 = Point(node.path_x[i] + math.cos(phi4) * length2lr,
                           node.path_y[i] + math.sin(phi4) * length2lr)
            seg_lfrf = Segment(point1, point2)
            seg_rfrr = Segment(point2, point3)
            seg_rrlr = Segment(point3, point4)
            seg_lrlf = Segment(point4, point1)

            for obs_seg in obstacle_list:
                if (obs_seg.is_cross(seg_lfrf) or obs_seg.is_cross(seg_rfrr) or
                                  obs_seg.is_cross(seg_rrlr) or obs_seg.is_cross(seg_lrlf)) is True:
                    return False
        return True


    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        """for (ox, oy, size) in self.obstacle_list:
            plt.plot(ox, oy, "ok", ms=30 * size)"""
        for obs_seg in self.obstacle_list:
            plt.plot([obs_seg.pt1.x, obs_seg.pt2.x], [obs_seg.pt1.y, obs_seg.pt2.y], '-b')
        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis([-13, 9, -8, 5])
        plt.grid(True)
        self.plot_start_goal_arrow()
        plt.pause(0.01)

    def plot_start_goal_arrow(self):
        reeds_shepp_path_planning.plot_arrow(
            self.start.x, self.start.y, self.start.yaw)
        reeds_shepp_path_planning.plot_arrow(
            self.end.x, self.end.y, self.end.yaw)

    def steer(self, from_node, to_node):
        # self.curvature = random.uniform(0.15, 0.222)
        px, py, pyaw, mode, course_lengths = reeds_shepp_path_planning.reeds_shepp_path_planning(
            from_node.x, from_node.y, from_node.yaw,
            to_node.x, to_node.y, to_node.yaw, self.curvature, step_size=0.2)

        if not px:
            return None

        new_node = copy.deepcopy(from_node)
        new_node.x = px[-1]
        new_node.y = py[-1]
        new_node.yaw = pyaw[-1]

        new_node.path_x = px
        new_node.path_y = py
        new_node.path_yaw = pyaw
        new_node.cost += sum([abs(l) for l in course_lengths])
        new_node.parent = from_node

        return new_node

    def calc_new_cost(self, from_node, to_node):
        _, _, _, _, course_lengths = reeds_shepp_path_planning.reeds_shepp_path_planning(
            from_node.x, from_node.y, from_node.yaw,
            to_node.x, to_node.y, to_node.yaw, self.curvature, step_size=0.2)
        if not course_lengths:
            return float("inf")

        return from_node.cost + sum([abs(l) for l in course_lengths])

    def get_random_node(self):
        temp_x = random.uniform(self.min_rand_x, self.max_rand_x)
        temp_y = random.uniform(self.min_rand_y, self.max_rand_y)
        temp_theta = math.atan2(temp_x, temp_y)
        temp_r = -math.hypot(temp_x, temp_y)
        """rnd = self.Node(self.end.x - temp_r*math.cos(self.end.yaw+temp_theta),
                        self.end.y + temp_r*math.sin(self.end.yaw+temp_theta),
                        random.uniform(-math.pi, math.pi)# -1/6*math.pi, math.pi/4)
                        )"""
        rnd = self.Node(self.end.x - temp_r * math.cos(self.end.yaw + temp_theta),
                        self.end.y + temp_r * math.sin(self.end.yaw + temp_theta),
                        random.uniform(math.pi, 5/4*math.pi)  # -1/6*math.pi, math.pi/4)
                        )

        return rnd

    def get_random_node_x_axis(self):
        temp_x = 0.0
        temp_y = random.uniform(self.min_rand_y, self.max_rand_y)
        temp_theta = math.atan(temp_x / temp_y)
        temp_r = math.hypot(temp_x, temp_y)
        rnd = self.Node(self.end.x + temp_r * math.cos(self.end.yaw + temp_theta),
                        self.end.y + temp_r * math.sin(self.end.yaw + temp_theta),
                        random.uniform(self.end.yaw-math.pi/3, self.end.yaw+math.pi/3)  # -1/6*math.pi, math.pi/4)
                        )
        return rnd

    def search_best_goal_node(self):

        goal_indexes = []
        for (i, node) in enumerate(self.node_list):
            if self.calc_dist_to_goal(node.x, node.y) <= self.goal_xy_th:
                goal_indexes.append(i)
        print("goal_indexes:", len(goal_indexes))

        # angle check
        final_goal_indexes = []
        for i in goal_indexes:
            if abs(self.node_list[i].yaw - self.end.yaw) <= self.goal_yaw_th:
                final_goal_indexes.append(i)

        print("final_goal_indexes:", len(final_goal_indexes))

        if not final_goal_indexes:
            return None

        min_cost = min([self.node_list[i].cost for i in final_goal_indexes])
        print("min_cost:", min_cost)
        for i in final_goal_indexes:
            if self.node_list[i].cost == min_cost:
                return i

        return None

    def generate_final_course(self, goal_index):
        path = [[self.end.x, self.end.y, self.end.yaw]]
        node = self.node_list[goal_index]
        while node.parent:
            for (ix, iy, iyaw) in zip(reversed(node.path_x), reversed(node.path_y), reversed(node.path_yaw)):
                path.append([ix, iy, iyaw])
            node = node.parent
        path.append([self.start.x, self.start.y, self.start.yaw])
        return path


def main(max_iter=1000):
    print("Start " + __file__)

    # ====Search Path with RRT====
    """obstacleList = [                                         # for vertical parking
        Segment(Point(0.0-6,12.0-1.3), Point(14.5-6, 12.0-1.3)),
        Segment(Point(1.0-6, 5.0-1.3), Point(4.8-6, 5.0-1.3)),
        Segment(Point(4.8-6, 5.0-1.3), Point(4.8-6, 0.0-1.3)),
        Segment(Point(4.8-6, 0.0-1.3), Point(7.0-6, 0.0-1.3)),
        Segment(Point(7.0-6, 0.0-1.3), Point(7.0-6, 5.0-1.3)),
        Segment(Point(7.0-6, 5.0-1.3), Point(14.0-6, 5.0-1.3)),

    ]  # [Segment(Point1(x,y),Point2(x,y)), ...]
    # for vertical parking
    start = [0.0-6, 8-1.3, np.deg2rad(195.0)]
    goal = [6.0-6, 1.3-1.3, np.deg2rad(90.0)] """

    obstacleList = [                                         # for parallel parking
            Segment(Point(-12.0, -0.8), Point(-5.0, -0.8)),
            Segment(Point(-5.0, -0.8), Point(-5.0, 1.2)),
            Segment(Point(-5.0, 1.2), Point(2.0, 1.2)),
            Segment(Point(2.0, 1.2), Point(2.0, -0.8)),
            Segment(Point(2.0, -0.8), Point(7.7, -0.8)),
            Segment(Point(-8.0, -5.8), Point(7.5, -5.8))
        ]  # [Segment(Point1(x,y),Point2(x,y)), ...]
    # Set Initial parameters
    # start = [0.0, 0.0, np.deg2rad(0.0)]
    # goal = [6.0, 7.0, np.deg2rad(90.0)]
    # for parallel parking
    start = [-8.4, -2.8, np.deg2rad(180.0)]
    goal = [0.0, 0.0, np.deg2rad(180.0)]                 # 6, 1.3

    rrt_star_reeds_shepp = RRTStarReedsShepp(start, goal,
                                             obstacleList,
                                             'P', max_iter=max_iter)
    path = rrt_star_reeds_shepp.planning(animation=show_animation)

    # Draw final path
    if path: # path and show_animation:  # pragma: no cover
        rrt_star_reeds_shepp.draw_graph()
        plt.plot([x for (x, y, yaw) in path], [y for (x, y, yaw) in path], '-r')
        plt.grid(True)
        plt.pause(0.001)
        plt.show()


if __name__ == '__main__':
    main()
