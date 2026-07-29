"""Warehouse human-robot collaborative picking simulation environment."""

import copy
import random

import gymnasium as gym
import numpy as np

from environment.class_public import Config
from environment.class_warehouse import Depot, Item, Order, Picker, PickPoint, Robot, StorageBin


class WarehouseEnv(gym.Env, Config):
    def __init__(self):
        super().__init__()
        self.parameter = self.parameters["warehouse"]
        self.N_l = self.parameter["shelf_capacity"]
        self.N_s = self.parameter["shelf_levels"]
        self.N_a = self.parameter["area_num"]
        self.N_ai = self.parameter["aisle_num"]
        self.N_w = self.parameter["area_num"] * self.parameter["aisle_num"]
        self.S_l = self.parameter["shelf_length"]
        self.S_w = self.parameter["shelf_width"]
        self.S_b = self.parameter["aisle_width"]
        self.S_d = self.parameter["entrance_width"]
        self.S_a = self.parameter["aisle_width"]
        self.area_dict = {'area{}'.format(i): self.N_ai for i in range(1, self.N_a + 1)}
        self.area_ids = list(self.area_dict.keys())
        self.depot_position = self.parameter["depot_position"]
        self.pack_time = self.parameters["order"]["pack_time"]
        self.total_time = None

        self.pick_points = {}
        self.storage_bins = {}
        self.items = {}
        self.pick_points_area = {area_id: [] for area_id in self.area_ids}
        self.depot_object = Depot(self.depot_position)
        self.create_warehouse_graph()

        self.adjust_pickers_dict = {area_id: None for area_id in self.area_ids}
        self.adjust_robots = None

        self.state = None
        self.action = None
        self.next_state = None
        self.reward = None
        self.done = False
        self.current_time = 0
        self.total_cost_current = 0
        self.total_cost_last = 0

        self.pickers = []
        self.pickers_area = {area_id: [] for area_id in self.area_ids}
        self.pickers_added = []
        self.robots = []
        self.robots_at_depot = []
        self.robots_assigned = []
        self.robots_added = []
        self.orders = []
        self.orders_not_arrived = []
        self.orders_unassigned = []
        self.orders_uncompleted = []
        self.orders_completed = []
        self.orders_arrived = []

    def area_id(self, nw):
        nw_sum = 0
        for area_id, area in self.area_dict.items():
            nw_sum += area
            if nw <= nw_sum:
                return area_id
        print(f"Error: Aisle number {nw} out of range!")
        return None

    def create_warehouse_graph(self):
        for nw in range(1, self.N_w + 1):
            area_id = self.area_id(nw)
            if area_id is None: continue
            for nl in range(1, self.N_l + 1):
                x = self.S_d + (2 * nw - 1) * self.S_w + (2 * nw - 1) / 2 * self.S_a
                y = self.S_b + (2 * nl - 1) / 2 * self.S_l
                position = (x, y)

                point_id = f"{nw}-{nl}"
                pick_point = PickPoint(point_id, position, area_id, [], [])
                self.pick_points[point_id] = pick_point
                self.pick_points_area[area_id].append(pick_point)

                for level in range(1, self.N_s + 1):
                    bin_id_left = f"{nw}-{nl}-{level}-left"
                    storage_bin = StorageBin(bin_id_left, position, area_id, None, None)
                    self.storage_bins[bin_id_left] = storage_bin
                    item_id_left = f"{nw}-{nl}-{level}-left-item"
                    item = Item(item_id_left, bin_id_left, position, area_id, None)
                    self.items[item_id_left] = item
                    storage_bin.item_id = item_id_left
                    pick_point.item_ids.append(item_id_left)
                    pick_point.storage_bin_ids.append(bin_id_left)

                    bin_id_right = f"{nw}-{nl}-{level}-right"
                    storage_bin = StorageBin(bin_id_right, position, area_id, None, None)
                    self.storage_bins[bin_id_right] = storage_bin
                    item_id_right = f"{nw}-{nl}-{level}-right-item"
                    item = Item(item_id_right, bin_id_right, position, area_id, None)
                    self.items[item_id_right] = item
                    storage_bin.item_id = item_id_right
                    pick_point.item_ids.append(item_id_right)
                    pick_point.storage_bin_ids.append(bin_id_right)

                    self.storage_bins[bin_id_left].pick_point_id = point_id
                    self.storage_bins[bin_id_right].pick_point_id = point_id
                    self.items[item_id_left].pick_point_id = point_id
                    self.items[item_id_right].pick_point_id = point_id

    def shortest_path_between_pick_points(self, point1, point2):
        x1, y1 = point1.position
        x2, y2 = point2.position
        if x1 == x2:
            return abs(y1 - y2)
        else:
            path1 = abs(y1 - self.S_b / 2) + abs(y2 - self.S_b / 2) + abs(x1 - x2)
            path2 = (abs(y1 - (self.S_b * 1.5 + self.N_l * self.S_l)) + abs(
                y2 - (self.S_b * 1.5 + self.N_l * self.S_l)) + abs(x1 - x2))
            return min(path1, path2)

    def adjust_robots_and_pickers(self, n_robots, n_pickers_dict, first_step=False):
        for area_id in self.area_ids:
            if n_pickers_dict[area_id] > 0:
                for i in range(n_pickers_dict[area_id]):
                    picker = Picker(area_id=area_id)
                    picker.pick_points = self.pick_points_area[area_id]
                    picker.position = picker.initial_position
                    picker.hire_time = self.current_time
                    self.pickers_area[area_id].append(picker)
                    self.pickers.append(picker)
                    self.pickers_added.append(picker)
                    if first_step:
                        picker.rent = 'long'
                        picker.unit_time_cost = picker.parameter["long_term_unit_time_cost"]
                    else:
                        picker.rent = 'short'
                        picker.unit_time_cost = picker.parameter["short_term_unit_time_cost"]
            elif n_pickers_dict[area_id] < 0:  # Corrected check from ==0 pass else to <0
                for i in range(abs(n_pickers_dict[area_id])):
                    false_short_pickers = [picker for picker in self.pickers_area[area_id] if
                                           picker.remove is False and picker.rent == 'short']
                    if len(false_short_pickers) <= 0:
                        break
                    else:
                        if len(self.idle_short_rent_pickers[area_id]) > 0:
                            picker = self.idle_short_rent_pickers[area_id].pop(0)
                            picker.remove = True
                            self.pickers_area[area_id].remove(picker)
                            self.pickers.remove(picker)
                            picker.fire_time = self.current_time
                        else:
                            picker = false_short_pickers[0]
                            picker.remove = True

        if n_robots > 0:
            for i in range(n_robots):
                robot = Robot(position=self.depot_position)
                self.robots.append(robot)
                self.robots_at_depot.append(robot)
                robot.run_start_time = self.current_time
                self.robots_added.append(robot)
                if first_step:
                    robot.rent = 'long'
                    robot.unit_time_cost = robot.parameter["long_term_unit_run_cost"]
                else:
                    robot.rent = 'short'
                    robot.unit_time_cost = robot.parameter["short_term_unit_run_cost"]
        elif n_robots < 0:  # Corrected logic
            for i in range(abs(n_robots)):
                false_short_robots = [robot for robot in self.robots if robot.remove is False and robot.rent == 'short']
                if len(false_short_robots) <= 0:
                    break
                else:
                    if len(self.idle_short_rent_robts) > 0:
                        robot = self.idle_short_rent_robts.pop(0)
                        robot.remove = True
                        self.robots.remove(robot)
                        self.robots_at_depot.remove(robot)
                        robot.run_end_time = self.current_time
                    else:
                        robot = false_short_robots[0]
                        robot.remove = True

    def reset(self, orders):
        self.pick_points = {}
        self.storage_bins = {}
        self.items = {}
        self.pick_points_area = {area_id: [] for area_id in self.area_ids}
        self.create_warehouse_graph()

        self.robots = []
        self.robots_at_depot = []
        self.robots_assigned = []
        self.robots_added = []

        self.pickers = []
        self.pickers_added = []
        self.pickers_area = {area_id: [] for area_id in self.area_ids}

        self.orders = copy.deepcopy(orders)
        for order in self.orders:
            order.unit_delay_cost = self.parameters["order"]["unit_delay_cost"]
            for item in order.items:
                item.pick_time = self.parameters["item"]["pick_time"]
        self.orders_not_arrived = copy.deepcopy(self.orders)
        self.orders_unassigned = []
        self.orders_uncompleted = []
        self.orders_completed = []
        self.orders_arrived = []

        self.current_time = 0
        self.total_cost_current = 0
        self.total_cost_last = 0

        self.done = False
        self.state = None
        self.action = None
        self.next_state = None
        self.reward = None

        self.state = self.state_extractor()
        return self.state

    def handle_events(self, current_time):
        """Process all discrete events scheduled at current_time."""
        while self.orders_not_arrived and self.orders_not_arrived[0].arrive_time <= current_time:
            order = self.orders_not_arrived.pop(0)
            self.orders_unassigned.append(order)
            self.orders_uncompleted.append(order)
            self.orders_arrived.append(order)

        for robot in self.robots:
            if abs(robot.move_to_pick_point_time - current_time) < 1e-5:
                robot.move_to_pick_point_time = float("inf")
                pick_point = robot.next_pick_point(self.pick_points)
                robot.position = pick_point.position
                robot.pick_point = pick_point
                pick_point.robot_queue.append(robot)

                if pick_point.picker is not None and pick_point.picker.state == "busy":
                    current_robot_pick_time = sum(item.pick_time for item in robot.items)
                    pick_point.picker.pick_end_time += current_robot_pick_time
                    robot.pick_point_complete_time = pick_point.picker.pick_end_time

        for robot in self.robots:
            if abs(robot.pick_point_complete_time - current_time) < 1e-5:
                robot.pick_point_complete_time = float("inf")
                pick_point = robot.pick_point

                if robot in pick_point.robot_queue:
                    pick_point.robot_queue.remove(robot)

                for item in robot.items:
                    robot.order.picked_items.append(item)
                    if item in robot.order.unpicked_items:
                        robot.order.unpicked_items.remove(item)
                    if item in robot.item_pick_order:
                        robot.item_pick_order.remove(item)

                if len(robot.item_pick_order) > 0:
                    next_pick_point = robot.next_pick_point(self.pick_points)
                    dist = self.shortest_path_between_pick_points(robot, next_pick_point)
                    move_time = dist / robot.speed
                    robot.move_to_pick_point_time = current_time + move_time
                else:
                    dist = self.shortest_path_between_pick_points(robot, self.depot_object)
                    move_time = dist / robot.speed
                    robot.move_to_depot_time = current_time + move_time + self.pack_time

        for picker in self.pickers[:]:
            if abs(picker.pick_end_time - current_time) < 1e-5:
                picker.pick_end_time = float("inf")
                picker.state = "idle"

                if picker.pick_point:
                    picker.pick_point.picker = None
                    picker.pick_point = None

                if picker.remove:
                    self.pickers_area[picker.area_id].remove(picker)
                    self.pickers.remove(picker)
                    picker.fire_time = current_time

        for robot in self.robots[:]:
            if abs(robot.move_to_depot_time - current_time) < 1e-5:
                robot.move_to_depot_time = float("inf")
                robot.state = "idle"
                robot.position = self.depot_position

                if robot.order:
                    if robot.order in self.orders_uncompleted:
                        self.orders_uncompleted.remove(robot.order)
                    self.orders_completed.append(robot.order)
                    robot.order.complete_time = current_time
                    robot.order = None

                if robot.remove:
                    self.robots.remove(robot)
                    if robot in self.robots_at_depot:
                        self.robots_at_depot.remove(robot)
                    robot.run_end_time = current_time
                else:
                    if robot not in self.robots_at_depot:
                        self.robots_at_depot.append(robot)

    def step(self, action, first_step=False, pattern=None):
        self.action = np.round(action).astype(int)
        self.adjust_robots = self.action[0]
        if len(self.robots) + self.adjust_robots <= 0:
            self.adjust_robots = -len(self.robots) + 1

        self.adjust_pickers_dict = {area_id: self.action[i] for i, area_id in enumerate(self.area_ids, start=1)}
        for area_id in self.area_ids:
            if len(self.pickers_area[area_id]) + self.adjust_pickers_dict[area_id] <= 0:
                self.adjust_pickers_dict[area_id] = -len(self.pickers_area[area_id]) + 1

        self.adjust_robots_and_pickers(self.adjust_robots, self.adjust_pickers_dict, first_step)
        if pattern == "long" and self.total_time is not None:
            epoch_time = max(0, self.total_time - self.current_time)
        else:
            epoch_time = 8 * 3600
        end_time = self.current_time + epoch_time

        while self.current_time < end_time:
            if len(self.orders_not_arrived) == 0 and len(self.orders_uncompleted) == 0:
                break

            self.assign_order_to_robot()
            for area_id in self.area_ids:
                self.assign_pick_point_to_picker(area_id)

            future_events = []
            if self.orders_not_arrived:
                future_events.append(self.orders_not_arrived[0].arrive_time)

            for picker in self.pickers:
                if picker.state == "busy":
                    future_events.append(picker.pick_end_time)
            for robot in self.robots:
                if robot.state == "busy":
                    times = [robot.move_to_pick_point_time, robot.pick_point_complete_time, robot.move_to_depot_time]
                    future_events.extend(
                        event_time
                        for event_time in times
                        if event_time != float("inf") and event_time > self.current_time
                    )

            valid_events = [event_time for event_time in future_events if event_time > self.current_time]
            if not valid_events:
                self.current_time = end_time
                break

            next_event_time = min(valid_events)
            if next_event_time > end_time:
                self.current_time = end_time
                break

            self.current_time = next_event_time
            self.handle_events(self.current_time)

        if self.current_time >= self.total_time:
            self.done = True

        self.state = self.state_extractor()
        total_delay_cost = sum(order.total_delay_cost(self.current_time) for order in self.orders_arrived)
        total_run_cost = sum(robot.total_run_cost(self.current_time) for robot in self.robots_added)
        total_hire_cost = sum(picker.total_hire_cost(self.current_time) for picker in self.pickers_added)
        self.total_cost_current = total_delay_cost + total_run_cost + total_hire_cost
        self.reward = self.compute_reward()
        self.total_cost_last = self.total_cost_current

        return self.state, self.reward, self.done

    def assign_order_to_robot(self):
        while len(self.orders_unassigned) > 0 and len(self.idle_robots) > 0:
            robot = self.idle_robots.pop(0)
            order = self.orders_unassigned.pop(0)
            robot.assign_order(order)
            next_pick_point = robot.next_pick_point(self.pick_points)
            shortest_path_length = self.shortest_path_between_pick_points(robot, next_pick_point)
            move_time = shortest_path_length / robot.speed
            robot.state = 'busy'
            robot.working_time += move_time
            robot.move_to_pick_point_time = self.current_time + move_time

    def assign_pick_point_to_picker(self, area_id):
        while len(self.idle_pickers[area_id]) > 0 and len(self.idle_pick_points[area_id]) > 0:
            idle_pickers_in_area = self.idle_pickers[area_id]
            idle_pick_points_in_area = self.idle_pick_points[area_id]
            picker = random.choice(idle_pickers_in_area)
            pick_point = picker.next_pick_point(idle_pick_points_in_area)

            picker.pick_point = pick_point
            pick_point.picker = picker

            shortest_path_length = self.shortest_path_between_pick_points(picker, pick_point)
            move_time = shortest_path_length / picker.speed

            picker.state = 'busy'
            picker.working_time += move_time
            picker.pick_start_time = self.current_time + move_time
            picker.pick_end_time = picker.pick_start_time

            # Sort robots in queue? Assuming FIFO
            for robot in pick_point.robot_queue:
                picker.pick_end_time += sum([i.pick_time for i in robot.items])
                robot.pick_point_complete_time = picker.pick_end_time

            picker.position = pick_point.position

    def state_extractor(self):
        robot_queue_list = [len(point.robot_queue) for point in self.pick_points.values()]
        robot_queue_list = np.array(robot_queue_list).reshape((self.N_w, self.N_l))
        picker_list = [0 if point.picker is None else 1 for point in self.pick_points.values()]
        picker_list = np.array(picker_list).reshape((self.N_w, self.N_l))
        unpicked_items_list = self.pick_point_unpicked_items
        unpicked_items_list = np.array(unpicked_items_list).reshape((self.N_w, self.N_l))
        n_robots = len([robot for robot in self.robots if robot.remove is False])
        n_pickers_area = [len(self.pickers_area[area_id]) for area_id in self.area_ids]

        self.state = {'robot_queue_list': robot_queue_list, 'picker_list': picker_list,
                      'unpicked_items_list': unpicked_items_list,
                      'n_robots': n_robots, 'n_pickers_area': n_pickers_area}
        return self.state

    def compute_reward(self):
        return self.total_cost_last - self.total_cost_current

    @property
    def idle_robots(self):
        return [robot for robot in self.robots if robot.state == 'idle']

    @property
    def idle_short_rent_robts(self):
        return [robot for robot in self.robots if robot.state == 'idle' and robot.rent == 'short']

    @property
    def idle_pickers(self):
        idle_pickers_area = {area_id: [picker for picker in self.pickers_area[area_id]
                                       if picker.state == 'idle'] for area_id in self.area_ids}
        return idle_pickers_area

    @property
    def idle_short_rent_pickers(self):
        idle_pickers_area = {area_id: [picker for picker in self.pickers_area[area_id]
                                       if picker.state == 'idle' and picker.rent == 'short'] for area_id in
                             self.area_ids}
        return idle_pickers_area

    @property
    def idle_pick_points(self):
        # 姣忎釜鍖哄煙寰呭垎閰嶆嫞璐у憳鐨勬嫞璐т綅鍒楄〃瀛楀吀
        # 淇锛氳繖閲屽繀椤讳娇鐢?self.pick_points_area (瀛樻斁 PickPoint 瀵硅薄)锛岃€屼笉鏄?self.pickers_area
        idle_pick_points_area = {
            area_id: [point for point in self.pick_points_area[area_id] if point.is_idle]
            for area_id in self.area_ids
        }
        return idle_pick_points_area

    @property
    def pick_point_unpicked_items(self):
        for point in self.pick_points.values():
            point.unpicked_items = []
        for order in self.orders_uncompleted:
            for item in order.unpicked_items:
                pick_point_id = item.pick_point_id
                self.pick_points[pick_point_id].unpicked_items.append(item)
        unpicked_items_list = [len(point.unpicked_items) for point in self.pick_points.values()]
        return unpicked_items_list


