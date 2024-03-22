import math
import time

from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
MAX_OF_GAME = 5000
MIN_OF_GAME = -5000

run_times = 0
def increment_run_times():
    global run_times
    run_times += 1

def regular_distance(pos1, pos2):
    return math.sqrt(pow(pos1[0]-pos2[0], 2) + pow(pos1[1]-pos2[1], 2))

def how_many_rightPickup_and_others(operations):
    num_of_ops = len(operations)
    num_of_special = 0
    for op, value in operations:
        if op == 'move east' or op == 'pick up':
            num_of_special += 1
    prob = 1/(num_of_ops+num_of_special)
    ans = 0
    for op, value in operations:
        if op == 'move east' or op == 'pick up':
            ans += (prob * 2 * value)
        else:
            ans += (prob * value)
    return ans

# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    other_robot = env.get_robot((robot_id + 1) % 2)
    unblocker = (regular_distance(robot.position, other_robot.position)/100)
    factor_to_raise_priority = 5 * robot.credit + unblocker
    if robot.package is not None:
        if robot.position == robot.package.destination:
            return 5 + factor_to_raise_priority
        return 3 + (1/manhattan_distance(robot.position, robot.package.destination)) + factor_to_raise_priority
    availablePackages = [p for p in env.packages if p.on_board]
    lengthsMe = []
    lengthsHis = []
    for av in availablePackages:
        lengthsMe.append(manhattan_distance(robot.position, av.position))
        lengthsHis.append(manhattan_distance(other_robot.position, av.position))
    if len(lengthsMe) == 2 and lengthsMe[0] > lengthsMe[1]:
        lengthsMe.reverse()
        lengthsHis.reverse()
    if (other_robot.package is not None) or lengthsMe[0] <= lengthsHis[0]:
        ans = (2 + (1/lengthsMe[0]) if lengthsMe[0] != 0 else 3)
        return ans + factor_to_raise_priority
    elif len(lengthsMe) == 2:
        ans = (2 + (1/lengthsMe[1]) if lengthsMe[1] != 0 else 3)
        return ans + factor_to_raise_priority
    else:
        go_to_charge = min(manhattan_distance(robot.position, env.charge_stations[0].position),
                     manhattan_distance(robot.position, env.charge_stations[1].position))
        ans = (1/go_to_charge if go_to_charge != 0 else 2)
        return ans + factor_to_raise_priority

class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)

def check_time_ended(start, lim):
    end = time.time()
    return (True if (end - start >= lim) else False)

class AgentMinimax(Agent):
    # TODO: section b : 1

    def minimax_impl(self, env: WarehouseEnv, agent_org, agent_id, depth, start, time_limit):
        if env.done():
            my_org = env.get_robot(agent_org)
            other_org = env.get_robot((agent_org + 1) % 2)
            if my_org.credit > other_org.credit:
                return MAX_OF_GAME
            elif my_org.credit < other_org.credit:
                return MIN_OF_GAME
            return 0
        if depth == 0:
            return smart_heuristic(env, agent_org)
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        lst_of_util = []
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
            if check_time_ended(start, time_limit):
                raise TimeoutError
            minimax_value = self.minimax_impl(child, agent_org, ((agent_id + 1) % 2), depth-1, start, time_limit)
            lst_of_util.append((op, minimax_value))

        jackpot = 0
        if agent_id == agent_org:
            jackpot = max(lst_of_util, key=lambda x: x[1])
        else:
            jackpot = min(lst_of_util, key=lambda x: x[1])
        return jackpot[1]

    def firstCall_minimax_impl(self, env: WarehouseEnv, agent_id, depth, start, time_limit):
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        lst_of_util = []
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
            if check_time_ended(start, time_limit):
                raise TimeoutError
            minimax_value = self.minimax_impl(child, agent_id, ((agent_id + 1) % 2), depth-1, start, time_limit)
            lst_of_util.append((op, minimax_value))
        res_value = max(lst_of_util, key=lambda x: x[1])
        return res_value[0]

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start = time.time()
        depth = 1
        time_limit *= 0.87
        last_value = self.firstCall_minimax_impl(env, agent_id, depth, start, time_limit)
        depth += 1
        while True:
            try:
                next_value = self.firstCall_minimax_impl(env, agent_id, depth, start, time_limit)
            except TimeoutError:
                return last_value
            depth += 1
            last_value = next_value



class AgentAlphaBeta(Agent):
    # TODO: section c : 1

    def minimax_alphabeta_impl(self, env: WarehouseEnv, agent_org, agent_id, depth, start, time_limit, alpha, beta):
        if env.done():
            my_org = env.get_robot(agent_org)
            other_org = env.get_robot((agent_org + 1) % 2)
            if my_org.credit > other_org.credit:
                return MAX_OF_GAME
            elif my_org.credit < other_org.credit:
                return MIN_OF_GAME
            return 0
        if depth == 0:
            return smart_heuristic(env, agent_org)
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        lst_of_util = []
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
            if check_time_ended(start, time_limit):
                raise TimeoutError
            minimax_value = self.minimax_alphabeta_impl(child, agent_org, ((agent_id + 1) % 2), depth - 1, start,
                                                        time_limit, alpha, beta)
            if agent_id == agent_org:
                alpha = max(alpha, minimax_value)
                if minimax_value >= beta:
                    return MAX_OF_GAME
            if agent_id != agent_org:
                beta = min(beta, minimax_value)
                if minimax_value <= alpha:
                    return MIN_OF_GAME
            lst_of_util.append((op, minimax_value))

        jackpot = 0
        if agent_id == agent_org:
            jackpot = max(lst_of_util, key=lambda x: x[1])
        else:
            jackpot = min(lst_of_util, key=lambda x: x[1])
        return jackpot[1]

    def firstCall_minimax_alphabeta_impl(self, env: WarehouseEnv, agent_id, depth, start, time_limit):
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        lst_of_util = []
        alpha = MIN_OF_GAME
        beta = MAX_OF_GAME
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
            if check_time_ended(start, time_limit):
                raise TimeoutError
            minimax_value = self.minimax_alphabeta_impl(child, agent_id, ((agent_id + 1) % 2), depth-1, start,
                                                        time_limit, alpha, beta)
            alpha = max(alpha, minimax_value)
            lst_of_util.append((op, minimax_value))
        res_value = max(lst_of_util, key=lambda x: x[1])
        return res_value[0]

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start = time.time()
        depth = 1
        time_limit *= 0.9
        last_value = self.firstCall_minimax_alphabeta_impl(env, agent_id, depth, start, time_limit)
        depth += 1
        while True:
            try:
                next_value = self.firstCall_minimax_alphabeta_impl(env, agent_id, depth, start, time_limit)
            except TimeoutError:
                return last_value
            depth += 1
            last_value = next_value

class AgentExpectimax(Agent):
    # TODO: section d : 1

    def expectimax_impl(self, env: WarehouseEnv, agent_org, agent_id, depth, start, time_limit):
        if env.done():
            my_org = env.get_robot(agent_org)
            other_org = env.get_robot((agent_org + 1) % 2)
            if my_org.credit > other_org.credit:
                return MAX_OF_GAME
            elif my_org.credit < other_org.credit:
                return MIN_OF_GAME
            return 0
        if depth == 0:
            return smart_heuristic(env, agent_org)
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        lst_of_util = []
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
            if check_time_ended(start, time_limit):
                raise TimeoutError
            minimax_value = self.expectimax_impl(child, agent_org, ((agent_id + 1) % 2), depth - 1, start, time_limit)
            lst_of_util.append((op, minimax_value))

        if agent_id == agent_org:
            return max(lst_of_util, key=lambda x: x[1])[1]
        else:
            return how_many_rightPickup_and_others(lst_of_util)

    def firstCall_expectimax_impl(self, env: WarehouseEnv, agent_id, depth, start, time_limit):
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        lst_of_util = []
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
            if check_time_ended(start, time_limit):
                raise TimeoutError
            minimax_value = self.expectimax_impl(child, agent_id, ((agent_id + 1) % 2), depth-1, start, time_limit)
            lst_of_util.append((op, minimax_value))
        res_value = max(lst_of_util, key=lambda x: x[1])
        return res_value[0]

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start = time.time()
        depth = 1
        time_limit *= 0.9
        last_value = self.firstCall_expectimax_impl(env, agent_id, depth, start, time_limit)
        depth += 1
        while True:
            try:
                next_value = self.firstCall_expectimax_impl(env, agent_id, depth, start, time_limit)
            except TimeoutError:
                return last_value
            depth += 1
            last_value = next_value


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)