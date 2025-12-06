from TspSolver import GaTspSolver

if __name__ == '__main__':
    city_data_path = "cityData.txt"
    intercity_data_path = "intercityDistance.txt"

    city_data = dict()
    intercity_data = dict()

    with open(city_data_path) as f:
        for key, coords in enumerate(f):
            vals = list(map(float, coords.split()))
            city_data[key] = vals

    with open(intercity_data_path) as f:
        for key, dists in enumerate(f):
            dists = list(map(float, dists.split()))
            intercity_data[key] = dists

    ga_solver = GaTspSolver(city_data, intercity_data)
    solutions = ga_solver.solve(5, 40, 0.8, 0.05, 10, 100)
    scores = [sol.get_cost() for sol in solutions]
    print(scores)
    print()
    print(solutions[0].get_path())
    print(len(set(solutions[0].get_path())))
    
