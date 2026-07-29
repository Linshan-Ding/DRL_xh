"""
定义订单、商品、储货位、拣货位、机器人和拣货员等类
"""
import random
import numpy as np
from environment.class_public import Config


# 订单类
class Order(Config):
    def __init__(self, order_id, items, arrive_time=0, due_time=None):
        """
        订单类
        :param order_id:  订单编号
        :param items:  订单中的商品列表
        :param arrive_time:  订单到达时间
        :param due_time:  订单交期
        """
        super().__init__()  # 调用父类的构造函数
        self.parameter = self.parameters["order"]  # 订单参数
        self.order_id = order_id  # 订单的编号
        self.items = items  # 订单中的商品列表
        self.arrive_time = arrive_time  # 订单到达时间
        self.due_time = None  # 订单交期
        self.complete_time = None  # 订单拣选完成时间
        # 订单中的未拣选完成的商品列表
        self.unpicked_items = items
        # 订单中的已拣选完成的商品列表
        self.picked_items = []
        # 订单交期
        self.due_time = due_time
        # 订单单位延期成本
        self.unit_delay_cost = self.parameter["unit_delay_cost"]

    # 订单延期总成本
    def total_delay_cost(self, current_time):
        """
        计算订单延期总成本
        :param current_time: 当前时间
        :return: 订单延期总成本
        """
        if self.complete_time is None:
            if current_time < self.due_time:
                return 0
            else:
                return (current_time - self.due_time) * self.unit_delay_cost
        else:
            if self.complete_time <= self.due_time:
                return 0
            else:
                return (self.complete_time - self.due_time) * self.unit_delay_cost


# 商品类
class Item(Config):
    def __init__(self, item_id, bin_id, position, area_id, pick_point_id):
        super().__init__()  # 调用父类的构造函数
        self.parameter = self.parameters["item"]  # 商品参数
        self.item_id = item_id  # 商品的编号
        self.bin_id = bin_id  # 商品所在的储货位编号
        self.position = position  # 商品所在的位置
        self.area_id = area_id  # 商品所在的区域编号
        self.pick_point_id = pick_point_id  # 商品所属拣货位的编号
        self.pick_time = self.parameter["pick_time"]  # 商品拣选时间
        self.pick_complete_time = 0  # 商品拣选完成时间


# 起始点类
class Depot:
    def __init__(self, position):
        self.position = position  # 起始点的位置


# 储货位类
class StorageBin:
    def __init__(self, bin_id, position, area_id, item_id, pick_point_id):
        self.bin_id = bin_id  # 储货位的编号
        self.position = position  # 储货位的位置
        self.item_id = item_id  # 储货位中的商品编号
        self.pick_point_id = pick_point_id  # 储货位所属拣货位的编号
        # 储货位所属区域的编号
        self.area_id = area_id
        # 当前储货位的机器人对象队列
        self.robot_queue = []
        # 当前储货位的拣货员对象
        self.picker = None


# 拣货位类
class PickPoint:
    def __init__(self, point_id, position, area_id, item_ids, storage_bin_ids):
        self.point_id = point_id  # 拣货位的编号
        self.position = position  # 拣货位的位置
        self.area_id = area_id  # 拣货位所属区域的编号
        self.item_ids = item_ids  # 拣货位中的商品编号列表
        self.storage_bin_ids = storage_bin_ids  # 拣货位对应的储货位编号列表
        self.robot_queue = []  # 当前拣货位的机器人对象队列
        self.picker = None  # 拣货员对象
        self.unpicked_items = []  # 拣货位的未拣货商品列表

    # 监测拣货位置是否待分配拣货员
    @property
    def is_idle(self):
        # 如果拣货位上未分配拣货员且机器人队列中有机器人，则返回True
        if len(self.robot_queue) > 0 and self.picker is None:
            return True
        # 如果拣货位上有拣货员，则返回False
        else:
            return False


#  -------------------------机器人类---------------------------
class Robot(Config):
    def __init__(self, position):
        super().__init__()
        self.parameter = self.parameters["robot"]  # 机器人参数
        self.position = position  # 机器人的位置
        self.pick_point = None  # 机器人当前拣货位
        self.order = None  # 机器人关联的订单
        self.item_pick_order = []  # 机器人剩余未拣选商品的对象列表
        self.state = 'idle'  # 机器人所处状态：'idle', 'busy'
        self.speed = self.parameter["robot_speed"]  # 机器人移动速度
        self.unit_time_cost = None  # 机器人单位运行成本
        self.pick_point_complete_time = 0  # 机器人在当前拣货位的拣货完成时间
        self.move_to_pick_point_time = 0  # 机器人移动到拣货位的时间
        self.move_to_depot_time = 0  # 机器人移动到depot_position的时间
        self.working_time = 0  # 机器人工作时间
        self.run_start_time = None  # 机器人运行开始时间
        self.run_end_time = None  # 机器人运行结束时间
        self.remove = False  # 机器人移除标识
        self.rent = None  # 'long' or 'short'
        # 仓库尺寸参数
        self.S_d = self.parameters["warehouse"]["entrance_width"]  # 仓库的出入口处的宽度
        self.S_b = self.parameters["warehouse"]["aisle_width"]  # 底部通道的宽度
        self.S_l = self.parameters["warehouse"]["shelf_length"]  # 储货位的长度
        self.N_l = self.parameters["warehouse"]["shelf_capacity"]  # 单个货架中储货位的数量
        # 拣货位选择规则标识符
        # 1：选择x坐标值最小的拣货位；2：选择距离最近的拣货位；3：选择排队机器人数量最少的拣货位；
        # 4：选择排队机器人数量最多的拣货位；5：选择待拣选商品最少的拣货位；6：选择待拣选商品最多的拣货位；
        # 7: 随机选择一个拣货位
        self.pick_point_selection_rule = 2

    def assign_order(self, order):
        """为机器人分配订单"""
        self.order = order
        self.plan_item_order()

    def plan_item_order(self):
        """订单中的商品对象拣选顺序规划"""
        if self.order is not None:
            self.item_pick_order = [item for item in self.order.items]
            # # 商品对象按照其拣货位的位置进行排序（按位置X坐标从小到大重排序，相同X坐标的按照Y坐标从小到大进行排序）
            # self.item_pick_order = sorted(self.item_pick_order, key=lambda x: (x.position[0], x.position[1]))
        else:
            self.item_pick_order = []

    # 返回机器人在当前拣货位拣货完成后的下一个拣货位
    def next_pick_point(self, pick_points):
        # 计算机器人的所有待拣选商品所属的拣货位列表
        pick_point_ids = [item.pick_point_id for item in self.item_pick_order]
        pick_point_ids = list(set(pick_point_ids))  # 去重
        # 根据拣货位选择规则选择下一个拣货位
        if self.pick_point_selection_rule == 1:  # 选择x坐标值最小的拣货位
            pick_point_ids_sorted = sorted(pick_point_ids, key=lambda x: (pick_points[x].position[0],
                                                                          pick_points[x].position[1]))
            next_pick_point_id = pick_point_ids_sorted[0]
        elif self.pick_point_selection_rule == 2:  # 选择距离机器人最近的拣货位
            distances = {point_id: self.distance_between_pick_points(self.position, pick_points[point_id].position)
                         for point_id in pick_point_ids}
            next_pick_point_id = min(distances, key=distances.get)
        elif self.pick_point_selection_rule == 3:  # 选择排队机器人数量最少的拣货位
            queue_lengths = {point_id: len(pick_points[point_id].robot_queue) for point_id in pick_point_ids}
            next_pick_point_id = min(queue_lengths, key=queue_lengths.get)
        elif self.pick_point_selection_rule == 4:  # 选择排队机器人数量最多的拣货位
            queue_lengths = {point_id: len(pick_points[point_id].robot_queue) for point_id in pick_point_ids}
            next_pick_point_id = max(queue_lengths, key=queue_lengths.get)
        elif self.pick_point_selection_rule == 5:  # 选择未拣选商品最少的拣货位
            unpicked_counts = {point_id: len(pick_points[point_id].unpicked_items) for point_id in pick_point_ids}
            next_pick_point_id = min(unpicked_counts, key=unpicked_counts.get)
        elif self.pick_point_selection_rule == 6:  # 选择未拣选商品最多的拣货位
            unpicked_counts = {point_id: len(pick_points[point_id].unpicked_items) for point_id in pick_point_ids}
            next_pick_point_id = max(unpicked_counts, key=unpicked_counts.get)
        elif self.pick_point_selection_rule == 7:  # 随机选择一个拣货位
            next_pick_point_id = random.choice(pick_point_ids)
        else:
            raise ValueError("拣货位选择规则标识符错误！")
        return pick_points[next_pick_point_id]

    def distance_between_pick_points(self, position1, position2):
        """两个拣货位之间的最短路径长度（若不在一个巷道，则需要从上部或下部绕过储货位）"""
        x1, y1 = position1
        x2, y2 = position2
        # 如果两个拣货位在同一巷道，则返回两个拣货位之间的直线路径长度
        if x1 == x2:
            return abs(y1 - y2)
        # 计算从上部绕过和从下部绕过的路径，选择最短路径，并返回路径长度
        else:
            path1 = abs(y1 - self.S_b / 2) + abs(y2 - self.S_b / 2) + abs(x1 - x2)
            path2 = (abs(y1 - (self.S_b * 1.5 + self.N_l * self.S_l)) + abs(y2 - (self.S_b * 1.5 + self.N_l * self.S_l))
                     + abs(x1 - x2))
            return min(path1, path2)

    # 当前时刻机器人总的运行成本
    def total_run_cost(self, current_time):
        if self.run_end_time is None:
            run_time = current_time - self.run_start_time
            total_cost = run_time * self.unit_time_cost
            return total_cost
        else:
            run_time = self.run_end_time - self.run_start_time
            total_cost = run_time * self.unit_time_cost
            return total_cost

    # 机器人关联订单中属于当前拣货位的商品列表
    @property
    def items(self):
        if self.order is not None:
            items = []  # 机器人关联订单中属于当前拣货位的商品列表
            for item in self.order.items:
                if item.pick_point_id == self.pick_point.point_id:
                    items.append(item)
            return items
        return None


# -------------------------拣货员类---------------------------
class Picker(Config):
    def __init__(self, area_id):
        super().__init__()  # 调用父类的构造函数
        self.parameter = self.parameters["picker"]  # 拣货员参数
        self.pick_point = None  # 拣货员当前拣货位
        self.position = None  # 拣货员的位置
        self.item = None  # 拣货员待拣选或正在拣选的商品
        self.state = 'idle'  # 拣货员状态：'idle', 'busy'
        self.speed = self.parameter["picker_speed"]  # 拣货员移动速度
        self.area_id = area_id  # 拣货员所在区域的编号
        self.unit_time_cost = None  # 拣货员单位时间雇佣成本
        self.storage_bins = []  # 拣货员负责的储货位列表
        self.pick_points = []  # 拣货员负责的拣货位列表
        self.working_time = 0  # 拣货员工作时间
        self.pick_start_time = 0  # 拣货员在当前拣货位拣货开始时间
        self.pick_end_time = 0  # 拣货员在当前拣货位拣货结束时间
        self.remove = False  # 拣货员移除标识
        self.unit_fire_cost = self.parameter["unit_fire_cost"]  # 拣货员单位辞退成本
        self.hire_time = None  # 拣货员聘用开始时间
        self.fire_time = None  # 拣货员解聘时间
        self.rent = None  # 拣货员长租或短租标识
        # 仓库尺寸参数
        self.S_d = self.parameters["warehouse"]["entrance_width"]  # 仓库的出入口处的宽度
        self.S_b = self.parameters["warehouse"]["aisle_width"]  # 底部通道的宽度
        self.S_l = self.parameters["warehouse"]["shelf_length"]  # 储货位的长度
        self.N_l = self.parameters["warehouse"]["shelf_capacity"]  # 单个货架中储货位的数量
        # 拣货位选择规则标识符
        # 1：选择x坐标值最小的拣货位；2：选择距离最近的拣货位；3：选择排队机器人数量最少的拣货位；
        # 4：选择排队机器人数量最多的拣货位；5：选择待拣选商品最少的拣货位；6：选择待拣选商品最多的拣货位；
        # 7: 随机选择一个拣货位
        self.pick_point_selection_rule = 6

    # 当前时刻拣货员总的雇佣成本
    def total_hire_cost(self, current_time):
        if self.fire_time is None:
            hire_time = current_time - self.hire_time
            total_cost = hire_time * self.unit_time_cost
            return total_cost
        else:
            hire_time = self.fire_time - self.hire_time
            total_cost = hire_time * self.unit_time_cost + self.unit_fire_cost
            return total_cost

    # 返回拣货员在当前拣货位拣货完成后的下一个拣货位
    def next_pick_point(self, idle_pick_points_in_area):
        """
        :param idle_pick_points_in_area:  区域内待分配拣货员的拣货位列表
        :return:  下一个拣货位对象
        """
        # 根据拣货位选择规则选择下一个拣货位
        if self.pick_point_selection_rule == 1:  # 选择x坐标值最小的拣货位
            pick_point_ids_sorted = sorted(idle_pick_points_in_area, key=lambda x: (x.position[0], x.position[1]))
            next_pick_point = pick_point_ids_sorted[0]
            next_pick_point_id = next_pick_point.point_id
        elif self.pick_point_selection_rule == 2:  # 选择距离拣货员最近的拣货位
            distances = {point.point_id: self.distance_between_pick_points(self.position, point.position)
                         for point in idle_pick_points_in_area}
            next_pick_point_id = min(distances, key=distances.get)
        elif self.pick_point_selection_rule == 3: # 选择排队机器人数量最少的拣货位
            queue_lengths = {point.point_id: len(point.robot_queue) for point in idle_pick_points_in_area}
            next_pick_point_id = min(queue_lengths, key=queue_lengths.get)
        elif self.pick_point_selection_rule == 4:  # 选择排队机器人数量最多的拣货位
            queue_lengths = {point.point_id: len(point.robot_queue) for point in idle_pick_points_in_area}
            next_pick_point_id = max(queue_lengths, key=queue_lengths.get)
        elif self.pick_point_selection_rule == 5:  # 选择未拣选商品最少的拣货位
            unpicked_counts = {point.point_id: len(point.unpicked_items) for point in idle_pick_points_in_area}
            next_pick_point_id = min(unpicked_counts, key=unpicked_counts.get)
        elif self.pick_point_selection_rule == 6:  # 选择未拣选商品最多的拣货位
            unpicked_counts = {point.point_id: len(point.unpicked_items) for point in idle_pick_points_in_area}
            next_pick_point_id = max(unpicked_counts, key=unpicked_counts.get)
        elif self.pick_point_selection_rule == 7:  # 随机选择一个拣货位
            next_pick_point = random.choice(idle_pick_points_in_area)
            next_pick_point_id = next_pick_point.point_id
        else:
            raise ValueError("拣货位选择规则标识符错误！")
        next_pick_point = [point for point in idle_pick_points_in_area if point.point_id == next_pick_point_id][0]
        return next_pick_point


    def distance_between_pick_points(self, position1, position2):
        """两个拣货位之间的最短路径长度（若不在一个巷道，则需要从上部或下部绕过储货位）"""
        x1, y1 = position1
        x2, y2 = position2
        # 如果两个拣货位在同一巷道，则返回两个拣货位之间的直线路径长度
        if x1 == x2:
            return abs(y1 - y2)
        # 计算从上部绕过和从下部绕过的路径，选择最短路径，并返回路径长度
        else:
            path1 = abs(y1 - self.S_b / 2) + abs(y2 - self.S_b / 2) + abs(x1 - x2)
            path2 = (abs(y1 - (self.S_b * 1.5 + self.N_l * self.S_l)) + abs(y2 - (self.S_b * 1.5 + self.N_l * self.S_l))
                     + abs(x1 - x2))
            return min(path1, path2)

    # 根据负责的拣货位列表中的拣货位的坐标计算拣货员的初始位置（取各拣货位的坐标均值）
    @property
    def initial_position(self):
        x = np.mean([point.position[0] for point in self.pick_points])
        y = np.mean([point.position[1] for point in self.pick_points])
        position = (x, y)
        return position