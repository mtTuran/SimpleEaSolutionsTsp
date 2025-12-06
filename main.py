from TspSolver import GaTspSolver
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

    ga_solver = GaTspSolver(city_data, intercity_data)
    solutions, avg_costs = ga_solver.solve(22, 40, 0.8, 0.05, 10, 100)
    
    print(f"Average cost of the final solutions: {avg_costs[-1]}")
    print(f"Cost of the best solution: {solutions[0].get_cost()}")
    print(f"\nPath of the best solution:\n{solutions[0].get_path()}")

    plt.scatter(range(1, len(avg_costs) + 1), avg_costs)
    plt.title("Average cost per iteration")
    plt.xlabel("Iterations")
    plt.ylabel("Average Cost")
    plt.show()
    
