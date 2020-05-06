'''
Authors :
Arpit Parwal <aparwal@usc.edu>
Yeon-soo Park <yeonsoop@usc.edu>

This code is part of assignment 7 of INF-552 Machine Learning for Data Science, Spring 2020 at Viterbi, USC
'''
import math
from collections import defaultdict

_STATES_MATRIX, _STATES_DIC = defaultdict(list), defaultdict(list)
_NOISY_MIN, _NOISY_MAX, _DEC_PRECISION = 0.7, 1.3, 1
_MAX_STEPS = 10
_INPUT_FILE = 'hmm-data.txt'
_PREV, _PROB = 'prev', 'prob'

TRANSITION_MATRIX = defaultdict(dict)
TRANSITION_PROB = defaultdict(int)
TRANS_PROB = defaultdict(dict)

def load_data(file_name):
    out1, out2, out3 = [], [], []

    fp = open(file_name)
    for i, line in enumerate(fp):
        if 2 <= i < 12:
            row = i - 2
            out1.extend([[int(row), int(col)] for col, _x in enumerate(line.split()) if _x == '1'])
        elif 16 <= i < 20:
            out2.append([int(_x) for _x in line.split()[2:]])  # skip two label words.
        elif 24 <= i < 35:
            out3.append([float(_x) for _x in line.split()])
    fp.close()

    return out1, out2, out3

def print_out(title, list):
    print(title)
    for i, value in enumerate(list):
        print(i+1, ":", value)

def find_moves(loc, grid_size):
    moves = []
    x, y = loc[0], loc[1]

    if x + 1 < grid_size: moves.append((x + 1, y))
    if x - 1 > 0: moves.append((x - 1, y))
    if y + 1 < grid_size: moves.append((x, y + 1))
    if y - 1 > 0: moves.append((x, y - 1))

    return moves

def tower_distance_noisy(grids, tower_loc):
    out = []
    for i, non_obstacle_cell in enumerate(grids):
        dist=[]
        for j, tower in enumerate(tower_loc):
            euclidean_dist = math.sqrt(pow(non_obstacle_cell[0]-tower[0],2) + pow(non_obstacle_cell[1]-tower[1],2))
            dist.append([round(euclidean_dist * _NOISY_MIN, _DEC_PRECISION), round(euclidean_dist * _NOISY_MAX, _DEC_PRECISION)])
        out.append(dist)
    return out

def get_next_step_prob(node, noisy_dist, noisy_dists):
    prob_states=[]
    num_of_noisy = len(noisy_dist)
    for i in range(len(node)):
        loc = node[i]
        count = 0
        for ii in range(len(noisy_dist)):
            if noisy_dists[i][ii][0] <= noisy_dist[ii] <= noisy_dists[i][ii][1]: count+=1
        if count == num_of_noisy: prob_states.append(loc)
    return prob_states


def transition_probability(states_dic, adj_moves):

    for cell in states_dic:
        TRANSITION_PROB[cell]= 0.0
        possible_states = states_dic[cell]
        adj_cells = adj_moves[cell]
        
        for state_t in possible_states:
            state_t += 1
            for _next in adj_cells:
                if _next in states_dic:
                    if state_t in states_dic[_next]:
                        if _next not in TRANSITION_MATRIX[cell]:
                            TRANSITION_MATRIX[cell][_next] = 0.0
                        TRANSITION_MATRIX[cell][_next] += 1.0
                        TRANSITION_PROB[cell] += 1.0

        for _next in TRANSITION_MATRIX[cell]:
            TRANS_PROB[cell][_next] = TRANSITION_MATRIX[cell][_next] / TRANSITION_PROB[cell]
    return TRANS_PROB

def HMM(noisy_dist,probable_states,trans_prob):
    state_t = 0

    paths = defaultdict(dict)
    paths[0] = defaultdict(dict)

    for element in probable_states[state_t]:
        tup = tuple(element)
        paths[state_t][tup] = dict()
        paths[state_t][tup][_PREV], paths[state_t][tup][_PROB] = '', (1.0 / len(probable_states[state_t]))
        
    for state_t in range(1, len(noisy_dist)):
        paths[state_t] = defaultdict(dict)
        for items in paths[state_t-1]:
            if items in trans_prob:
                for _next in trans_prob[items]:
                    if list(_next) in probable_states[state_t]:
                        crt, n_state = paths[state_t - 1][items][_PROB], trans_prob[items][_next]
                        present_prob = crt * n_state

                        if _next in paths[state_t] and present_prob > paths[state_t][_next][_PROB]:
                            paths[state_t][_next][_PREV] = items
                            paths[state_t][_next][_PROB] = present_prob

                        if _next not in paths[state_t]:
                            paths[state_t][_next] = dict()
                            paths[state_t][_next][_PREV] = items
                            paths[state_t][_next][_PROB] = present_prob
    return paths

def backtracking(paths, max_state=_MAX_STEPS):
    max_prob, result = 0.0, []
    temp_cell = None

    for c in paths[max_state]:
        if max_prob < paths[max_state][c][_PROB]:
            max_prob = paths[max_state][c][_PROB]
            temp_cell = c
    result.append(temp_cell)
    for max_state in range(_MAX_STEPS, 0, -1):
        parent_cell = paths[max_state][temp_cell][_PREV]
        result.append(parent_cell)
        temp_cell = parent_cell
    return result

if __name__ == "__main__":

    grid_non_obstacle_cells, tower_loc, noisy_dist = load_data(_INPUT_FILE)
    # The file hmm-data.txt contains a map of a 10 - by - 10 2D grid world.
    #
    # 1) grid_non_obstacle_cells: The non-obstacle cells are represented as '1', otherwise, '0'
    # 2) tower_loc: There are four towers, one in each of the four corners
    # 3) noisy_dist: Robot records a noisy measurement chosen uniformly at random from the set of numbers in the interval [0.7d, 1.3d] with one decimal place.
    #                These measurements for 11 time-steps are loaded from the hmm-data.txt data file.

    distance_noisy_matrix = tower_distance_noisy(grid_non_obstacle_cells, tower_loc)
    # The robot measures a true distance d (Euclidean distances), and records a noisy measurement based on noisy_dist

    # From grid_non_obstacle_cells and noisy_dist matrix, generate _STATES_MATRIX and TRANSITION_MATRIX through get_next_step_prob().
    for i in range(0, len(noisy_dist)):
        _STATES_MATRIX[i] = get_next_step_prob(grid_non_obstacle_cells, noisy_dist[i], distance_noisy_matrix)
        for cell in _STATES_MATRIX[i]:
            _STATES_DIC[tuple(cell)].append(i)

    # find possible moves from all cell node
    moves = defaultdict(list)
    for cell in _STATES_DIC:
        moves[cell] = find_moves(cell, _MAX_STEPS)

    # Use the function HMM based on observation_sequences (trans_matrix) which implementing the Viterbi algorighm to calculate the maximum probability of state in the last time-step,
    # then backtrack the state provides the condition for getting this probability, until the starting state is determined.
    trans_matrix = transition_probability(_STATES_DIC, moves)
    possible_paths = HMM(noisy_dist, _STATES_MATRIX, trans_matrix)

    print_out("=== PATH ===", backtracking(possible_paths)[::-1])