import random
from TspSolution import TspSolution

class GaTspSolver:
    def __init__(self, city_data, intercity_data):
        self.city_data = city_data
        self.intercity_data = intercity_data
        TspSolution.set_city_data(city_data)
        TspSolution.set_intercity_data(intercity_data)
        
    def solve(self, starting_city, population_size, crossover_rate, mutation_rate, number_of_elits, max_iters):
        solutions = self.initialize_population(starting_city, population_size)
        for _ in range(max_iters):
            next_generation = self.elitism(solutions, number_of_elits)
            while len(next_generation) < population_size:
                if random.uniform(0, 1) < crossover_rate:
                    parent_solutions = self.roulette_select(solutions, 2)
                    children_solutions = self.mate(starting_city, parent_solutions[0], parent_solutions[1], 2)
                    next_generation = next_generation + children_solutions
                else:
                    survivor = self.roulette_select(solutions, 1)
                    next_generation = next_generation + survivor
            for i in range(number_of_elits, len(next_generation)):
                if random.uniform(0, 1) < mutation_rate:
                    mutated_survivor = self.mutate(next_generation[i])
                    next_generation[i] = mutated_survivor
        ordered_solutions = sorted(solutions, key=lambda solution: solution.get_cost())
        return ordered_solutions

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
            first_parent_slice = solution_path_1[first_div_spot : sec_div_spot + 1]
            new_solution_path = [starting_city]
            i = 1
            j = 1
            while i < first_div_spot:
                if solution_path_2[j] not in first_parent_slice:
                    new_solution_path.append(solution_path_2[j])
                    i += 1
                j += 1
            new_solution_path = new_solution_path + first_parent_slice
            i = sec_div_spot + 1
            j = 1
            while i < len(solution_path_1):
                if solution_path_2[j] not in new_solution_path:
                    new_solution_path.append(solution_path_2[j])
                    i += 1
                j += 1
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