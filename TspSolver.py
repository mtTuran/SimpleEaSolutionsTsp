import random
from TspSolution import TspSolution

class GaTspSolver:
    def __init__(self, city_data, intercity_data):
        self.city_data = city_data
        self.intercity_data = intercity_data
        TspSolution.set_city_data(city_data)
        TspSolution.set_intercity_data(intercity_data)
        
    def solve(self, starting_city, population_size, crossover_rate, mutation_rate, number_of_elits, max_iters):
        avg_costs = []
        solutions = self.initialize_population(starting_city, population_size)
        costs = [sol.get_cost() for sol in solutions]
        avg_costs.append(sum(costs) / len(costs))
        for _ in range(max_iters):
            next_generation = self.elitism(solutions, number_of_elits)
            while len(next_generation) < population_size:
                if random.uniform(0, 1) < crossover_rate:
                    parent_solutions = self.roulette_select(solutions, 2)
                    children_solutions = self.mate(starting_city, parent_solutions[0], parent_solutions[1], 1)
                    next_generation = next_generation + children_solutions
                else:
                    survivor = self.roulette_select(solutions, 1)
                    next_generation = next_generation + survivor
            for i in range(number_of_elits, len(next_generation)):
                if random.uniform(0, 1) < mutation_rate:
                    mutated_survivor = self.mutate(next_generation[i])
                    next_generation[i] = mutated_survivor
            solutions = next_generation
            costs = [sol.get_cost() for sol in solutions]
            avg_costs.append(sum(costs) / len(costs))
        ordered_solutions = sorted(solutions, key=lambda solution: solution.get_cost())
        return ordered_solutions, avg_costs

    def initialize_population(self, starting_city, population_size):
        population = []
        travel_path = [city for city in self.city_data.keys()]
        travel_path.remove(starting_city)
        for _ in range(population_size):
            random.shuffle(travel_path)
            new_solution_path = [starting_city] + travel_path
            new_solution = TspSolution(new_solution_path)
            population.append(new_solution)
        return population
    
    def roulette_select(self, population, selection_size):
        population_weights = [1 / solution.get_cost() for solution in population]
        return random.choices(population, weights=population_weights, k=selection_size)
    
    def mate(self, starting_city, solution_1, solution_2, number_of_children):
        solution_path_1, solution_path_2 = solution_1.get_path(), solution_2.get_path()
        children = []
        for _ in range(number_of_children):
            first_div_spot = random.randint(1, len(solution_path_1))
            sec_div_spot = random.randint(1, len(solution_path_1))
            first_div_spot, sec_div_spot = min(first_div_spot, sec_div_spot), max(first_div_spot, sec_div_spot)
            middle_slice = solution_path_1[first_div_spot : sec_div_spot]
            left_slice = [city for city in solution_path_2[1:first_div_spot] if city not in middle_slice]
            right_slice = [city for city in solution_path_2[first_div_spot:] if city not in middle_slice]
            new_solution_path = [starting_city] + left_slice + middle_slice + right_slice
            
            children.append(TspSolution(new_solution_path))

            solution_path_1, solution_path_2 = solution_path_2, solution_path_1

        return children
    
    def mutate(self, solution):
        solution_path = solution.get_path()
        exchange_inds = range(1, len(solution_path))
        gene_1_ind, gene_2_ind = random.choices(exchange_inds, k=2)
        mutated_solution_path = solution_path[:gene_1_ind] + [solution_path[gene_2_ind]] + solution_path[gene_1_ind + 1:gene_2_ind] + [solution_path[gene_1_ind]] + solution_path[gene_2_ind + 1:]
        mutated_solution = TspSolution(mutated_solution_path)
        return mutated_solution
    
    def elitism(self, population, number_of_elits):
        ordered_solutions = sorted(population, key=lambda solution: solution.get_cost())
        return ordered_solutions[:number_of_elits]           
    

class AcsTspSolver:
    def __init__(self, city_data, intercity_data):
        self.city_data = city_data
        self.intercity_data = intercity_data
        TspSolution.set_city_data(city_data)
        TspSolution.set_intercity_data(intercity_data)

    def solve(self, starting_city, colony_size, alpha, beta, q0, local_evop_rate, global_evop_rate, max_iters):
        tau0 = 0.01 
        pheromons = self.initialize_pheromone_table(tau0)
        
        global_best_solution = None
        global_best_distance = float('inf')

        for _ in range(max_iters):
            current_iteration_solutions = []          
            for ant in range(colony_size):
                ant_path = [starting_city]        
                while len(ant_path) < len(self.city_data):
                    next_city = self.select_next_city(ant_path, pheromons, alpha, beta, q0)                    
                    curr_node = ant_path[-1]
                    current_val = pheromons[curr_node][next_city]                    
                    new_val = (1 - local_evop_rate) * current_val + (local_evop_rate * tau0)
                    
                    pheromons[curr_node][next_city] = new_val
                    pheromons[next_city][curr_node] = new_val # Symmetric TSP        
                    ant_path.append(next_city)
            
                solution = TspSolution(ant_path) 
                current_iteration_solutions.append(solution)
            
            current_iteration_solutions.sort(key=lambda s: s.get_cost())
            iter_best_sol = current_iteration_solutions[0]
            
            if iter_best_sol.get_cost() < global_best_distance:
                global_best_solution = iter_best_sol
                global_best_distance = iter_best_sol.get_cost()

            best_path = global_best_solution.get_path()
            pheromon_addition = self.compute_released_pheromons(global_best_distance)
            
            for i in range(len(best_path)):
                u = best_path[i]
                v = best_path[(i + 1) % len(best_path)] # Wrap around to start
                
                current_t = pheromons[u][v]
                pheromons[u][v] = (1 - global_evop_rate) * current_t + (global_evop_rate * pheromon_addition)
                pheromons[v][u] = pheromons[u][v] # Symmetric

        return global_best_solution

    def initialize_pheromone_table(self, initial_val):
        pheromons = dict()
        for city, dists in self.intercity_data.items():
            pheromons[city] = [initial_val for _ in dists]
        return pheromons

    def compute_released_pheromons(self, path_length):
        # Q / L
        return 1.0 / path_length

    def select_next_city(self, path, pheromon_table, alpha, beta, q0):
        current_city = path[-1]
        candidates = {}     
        unvisited = [i for i in range(len(self.city_data)) if i not in path]

        for i in unvisited:
            dist = self.intercity_data[current_city][i]
            if dist != 0:
                tau = pheromon_table[current_city][i] ** alpha
                eta = (1 / dist) ** beta
                candidates[i] = tau * eta

        if random.random() < q0:
            # Exploitation (Best)
            return max(candidates, key=candidates.get)
        else:
            # Exploration (Weighted Random)
            city_indices = list(candidates.keys())
            weights = list(candidates.values())
            return random.choices(city_indices, weights=weights)[0]
