from pddlgym.parser import PDDLDomainParser, PDDLProblemParser
from pddlgym.structs import LiteralConjunction
import pddlgym
import os
import numpy as np
from itertools import count
np.random.seed(0)


PDDLDIR = os.path.join(os.path.dirname(pddlgym.__file__), "pddl")

I, G, W, P, X, H = range(6)

GRID1 = np.array([
    [I, P, P, P, P],
    [W, X, W, W, P],
    [X, X, X, W, H],
    [X, W, X, W, P],
    [W, X, X, W, P],
    [W, X, W, W, P],
    [G, P, P, H, P],
])

GRID2 = np.array([
    [P, P, I, X, X],
    [P, W, W, W, X],
    [P, W, W, X, X],
    [H, W, X, X, W],
    [P, W, X, X, X],
    [P, W, W, W, W],
    [P, P, G, W, W],
])

GRID3 = np.array([
    [I, P, P, P, P, P, P, P, P, P,],
    [X, X, W, W, X, X, X, W, W, P,],
    [X, X, X, W, W, X, X, W, W, P,],
    [W, X, X, W, W, X, X, X, W, P,],
    [W, X, X, W, W, X, W, X, W, P,],
    [W, X, X, W, W, X, W, X, W, P,],
    [X, X, X, X, X, X, W, X, X, P,],
    [X, X, X, W, W, X, W, W, X, P,],
    [W, X, W, W, W, X, W, W, W, P,],
    [W, X, X, W, W, X, W, W, W, P,],
    [W, X, X, W, W, X, G, P, P, P,],
])

GRID4 = np.array([
    [X, X, W, X, X, X, X, X, X, X, X, X, X, X, X, X],
    [X, X, W, W, X, X, X, X, X, X, W, W, X, X, W, W],
    [X, X, X, X, W, X, X, X, X, X, X, W, X, X, W, W],
    [X, X, X, X, W, W, W, X, X, X, X, X, X, X, X, X],
    [X, X, X, X, W, X, X, X, X, X, X, X, X, X, X, X],
    [X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X],
    [X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X],
    [P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P],
    [P, W, X, X, X, X, W, X, X, W, X, W, X, X, W, P],
    [P, W, W, X, X, X, W, X, X, W, X, W, X, X, W, P],
    [P, X, X, X, X, X, W, X, W, W, W, W, X, X, W, P],
    [I, X, X, X, X, W, W, W, W, W, W, W, X, X, W, G],
])

GRID5 = np.array([
    [G, P, P, P, W, W, W, W, W, W, X],
    [X, X, X, P, W, W, W, W, W, W, X],
    [X, X, X, P, W, W, W, W, W, W, X],
    [P, P, P, P, W, W, W, W, W, W, X],
    [P, X, X, X, W, W, W, W, W, W, X],
    [P, X, X, X, W, W, W, W, W, W, X],
    [P, X, X, X, W, W, W, W, W, W, X],
    [P, X, X, X, W, W, W, W, W, W, X],
    [P, X, X, X, W, W, W, W, W, W, X],
    [P, X, X, X, W, W, W, W, W, W, X],
    [P, X, X, X, W, W, W, W, W, W, X],
    [P, X, X, X, W, W, W, W, W, W, X],
    [I, X, X, X, W, W, W, W, W, W, X],
])

GRID6 = np.array([
    [I, P, P, P, X, X, X, W, W, G],
    [W, W, X, P, X, W, W, X, X, P],
    [W, W, X, P, X, X, W, X, X, P],
    [W, W, X, P, X, X, X, X, X, P],
    [W, W, X, P, X, W, W, X, X, P],
    [W, W, X, P, X, W, W, X, X, P],
    [W, W, X, P, X, X, W, X, X, P],
    [W, W, X, P, P, P, P, P, P, P],
])

TRAIN_GRIDS = [GRID1]
TEST_GRIDS = [GRID2, GRID3, GRID4, GRID5, GRID6]



def create_problem(grid, domain, problem_dir, problem_outfile):
    
    # Create location objects
    loc_type = domain.types['loc']
    objects = set()
    grid_locs = np.empty(grid.shape, dtype=object)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            obj = loc_type(f'r{r}_c{c}')
            objects.add(obj)
            grid_locs[r, c] = obj

    initial_state = set()

    # Add at, isWater, isHill, isGoal
    at = domain.predicates['at']
    isWater = domain.predicates['iswater']
    isHill = domain.predicates['ishill']
    isGoal = domain.predicates['isgoal']
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            obj = grid_locs[r, c]
            if grid[r, c] == I:
                initial_state.add(at(obj))
            elif grid[r, c] == W:
                initial_state.add(isWater(obj))
            elif grid[r, c] == H:
                initial_state.add(isHill(obj))
            elif grid[r, c] == G:
                initial_state.add(isGoal(obj))

    # Add adjacent
    adjacent = domain.predicates['adjacent']

    def get_neighbors(r, c):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                yield (nr, nc)

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):    
            obj = grid_locs[r, c]
            for (nr, nc) in get_neighbors(r, c):
                nobj = grid_locs[nr, nc]
                initial_state.add(adjacent(obj, nobj))

    # Add onMarkedPath
    onMarkedPath = domain.predicates['onmarkedpath']

    # Get the path
    path = []
    r, c = np.argwhere(grid == I)[0]
    while True:
        path.append((r, c))
        if grid[r, c] == G:
            break
        for (nr, nc) in get_neighbors(r, c):
            if (nr, nc) in path:
                continue
            if grid[nr, nc] in [P, G, H]:
                r, c = nr, nc
                break
        else:
            raise Exception("Should not happen")

    for (r, c), (nr, nc) in zip(path[:-1], path[1:]):
        obj = grid_locs[r, c]
        nobj = grid_locs[nr, nc]
        initial_state.add(onMarkedPath(obj, nobj))

    # Goal
    goal_rcs = np.argwhere(grid == G)
    assert len(goal_rcs) == 1
    goal_r, goal_c = goal_rcs[0]
    goal_obj = grid_locs[goal_r, goal_c]
    goal = LiteralConjunction([at(goal_obj)])

    filepath = os.path.join(PDDLDIR, problem_dir, problem_outfile)

    PDDLProblemParser.create_pddl_file(
        filepath,
        objects=objects,
        initial_state=initial_state,
        problem_name="hiking",
        domain_name=domain.domain_name,
        goal=goal,
        fast_downward_order=True,
    )
    print("Wrote out to {}.".format(filepath))

def generate_problems():
    domain = PDDLDomainParser(os.path.join(PDDLDIR, "hiking.pddl"),
        expect_action_preds=False,
        operators_as_actions=True)

    for problem_idx, grid in enumerate(TRAIN_GRIDS + TEST_GRIDS):
        if problem_idx < len(TRAIN_GRIDS):
            problem_dir = "hiking"
        else:
            problem_dir = "hiking_test"
        problem_outfile = "problem{}.pddl".format(problem_idx)

        create_problem(grid, domain, problem_dir, problem_outfile)

if __name__ == "__main__":
    generate_problems()
