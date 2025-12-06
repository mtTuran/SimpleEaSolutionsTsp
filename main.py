from TspSolver import GaTspSolver, AcsTspSolver
import matplotlib.pyplot as plt

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

    starting_city = 22
    population_size = 40
    crossover_rate = 0.8
    mutation_rate = 0.05
    number_of_elits = 10
    max_iters = 1000

    '''
    ga_solver = GaTspSolver(city_data, intercity_data)
    solutions, avg_costs = ga_solver.solve(starting_city, population_size, crossover_rate, mutation_rate, number_of_elits, max_iters)
    
    print(f"\nAverage cost of the final solutions: {avg_costs[-1]}")
    print(f"\nCost of the best solution: {solutions[0].get_cost()}")
    print(f"\nPath of the best solution:\n{solutions[0].get_path()}")

    plt.scatter(range(1, len(avg_costs) + 1), avg_costs)
    plt.title("Average cost per iteration")
    plt.xlabel("Iterations")
    plt.ylabel("Average Cost")
    plt.show()
    '''

    starting_city = 22
    colony_size = 40
    alpha = 1
    beta = 3
    q0 = 0.8
    local_evop_rate = 0.1
    global_evop_rate = 0.1
    max_iters = 1000

    acs_solver = AcsTspSolver(city_data, intercity_data)
    solution = acs_solver.solve(starting_city, colony_size, alpha, beta, q0, local_evop_rate, global_evop_rate, max_iters)
    
    print(f"\nCost of the best solution: {solution.get_cost()}")
    print(f"\nPath of the best solution:\n{solution.get_path()}")

    
