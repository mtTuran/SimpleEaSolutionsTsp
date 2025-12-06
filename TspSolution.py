class TspSolution:

    city_data = dict()
    intercity_data = dict()

    def __init__(self, solution_path):
        self.solution_path = solution_path
        self.cost = self.evaluate_cost()

    def evaluate_cost(self):
        if len(TspSolution.intercity_data) == 0:
            print("Set the intercity data for the TspSolution class!")
        cost = 0
        for i in range(len(self.solution_path) - 1):
            cost = cost + TspSolution.intercity_data[self.solution_path[i]][self.solution_path[i + 1]]
        cost = cost + TspSolution.intercity_data[self.solution_path[-1]][self.solution_path[0]]
        return cost
    
    def get_path(self):
        return self.solution_path
    
    def get_cost(self):
        return self.cost
    
    @staticmethod
    def set_city_data(city_data):
        TspSolution.city_data = city_data
    
    @staticmethod
    def set_intercity_data(intercity_data):
        TspSolution.intercity_data = intercity_data