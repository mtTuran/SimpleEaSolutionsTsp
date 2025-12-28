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
    
    # ==================== GENETIC ALGORITHM ====================
    print("=" * 60)
    print("Running GENETIC ALGORITHM...")
    print("=" * 60)
    
    population_size = 40
    crossover_rate = 0.8
    mutation_rate = 0.05
    number_of_elits = 10
    max_iters = 1000
    
    ga_solver = GaTspSolver(city_data, intercity_data)
    ga_solutions, ga_avg_costs = ga_solver.solve(starting_city, population_size, crossover_rate, 
                                                   mutation_rate, number_of_elits, max_iters)
    
    print(f"✓ GA Complete!")
    print(f"  Average cost of final solutions: {ga_avg_costs[-1]:.2f}")
    print(f"  Cost of best solution: {ga_solutions[0].get_cost():.2f}")
    
    # ==================== ANT COLONY SYSTEM ====================
    print("\n" + "=" * 60)
    print("Running ANT COLONY SYSTEM...")
    print("=" * 60)
    
    colony_size = 40
    alpha = 1
    beta = 3
    q0 = 0.8
    local_evop_rate = 0.1
    global_evop_rate = 0.1
    max_iters = 1000
    
    acs_solver = AcsTspSolver(city_data, intercity_data)
    acs_solution = acs_solver.solve(starting_city, colony_size, alpha, beta, q0, 
                                     local_evop_rate, global_evop_rate, max_iters)
    
    print(f"✓ ACS Complete!")
    print(f"  Cost of best solution: {acs_solution.get_cost():.2f}")
    
    # ==================== COMPARISON ====================
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    ga_cost = ga_solutions[0].get_cost()
    acs_cost = acs_solution.get_cost()
    print(f"GA Best Cost:  {ga_cost:.2f}")
    print(f"ACS Best Cost: {acs_cost:.2f}")
    print(f"Difference:    {abs(ga_cost - acs_cost):.2f}")
    if ga_cost < acs_cost:
        print(f"Winner: GA (by {((acs_cost - ga_cost) / acs_cost * 100):.2f}%)")
    else:
        print(f"Winner: ACS (by {((ga_cost - acs_cost) / ga_cost * 100):.2f}%)")
    
    # ==================== PLOTTING ALL RESULTS ====================
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)
    
    fig = plt.figure(figsize=(16, 10))
    
    # ----- GA Convergence Plot -----
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(range(1, len(ga_avg_costs) + 1), ga_avg_costs, linewidth=2, color='#2E86AB')
    ax1.set_title("GA: Convergence Curve", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Iteration", fontsize=11)
    ax1.set_ylabel("Average Cost", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    ax1.text(0.98, 0.97, f'Final Avg: {ga_avg_costs[-1]:.2f}', 
             transform=ax1.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ----- GA Best Path -----
    ax2 = plt.subplot(2, 3, 2)
    ga_path = ga_solutions[0].get_path()
    ga_x = [city_data[city][0] for city in ga_path]
    ga_y = [city_data[city][1] for city in ga_path]
    
    ax2.plot(ga_x, ga_y, 'o-', linewidth=2, markersize=6, color='#2E86AB', alpha=0.7)
    ax2.plot(ga_x[0], ga_y[0], 'r*', markersize=15, label=f'Start (City {starting_city})')
    
    for i, city in enumerate(ga_path[:-1]):
        ax2.annotate(str(city), (ga_x[i], ga_y[i]), 
                    textcoords="offset points", xytext=(0, 5), ha='center', fontsize=7)
    
    ax2.set_title(f"GA: Best Tour (Cost: {ga_cost:.2f})", fontsize=13, fontweight='bold')
    ax2.set_xlabel("X Coordinate", fontsize=11)
    ax2.set_ylabel("Y Coordinate", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    
    # ----- GA Statistics -----
    ax3 = plt.subplot(2, 3, 3)
    ax3.axis('off')
    
    ga_stats = f"""
    GENETIC ALGORITHM RESULTS
    {'=' * 35}
    
    Best Cost:        {ga_cost:.2f}
    Final Avg Cost:   {ga_avg_costs[-1]:.2f}
    Initial Avg Cost: {ga_avg_costs[0]:.2f}
    Improvement:      {ga_avg_costs[0] - ga_avg_costs[-1]:.2f}
    
    Parameters:
    • Population Size:  {population_size}
    • Crossover Rate:   {crossover_rate}
    • Mutation Rate:    {mutation_rate}
    • Number of Elites: {number_of_elits}
    • Max Iterations:   {max_iters}
    
    Path Length: {len(ga_path)} cities
    """
    
    ax3.text(0.1, 0.5, ga_stats, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='#E8F4F8', alpha=0.8))
    
    # ----- ACS Best Path -----
    ax4 = plt.subplot(2, 3, 5)
    acs_path = acs_solution.get_path()
    acs_x = [city_data[city][0] for city in acs_path]
    acs_y = [city_data[city][1] for city in acs_path]
    
    ax4.plot(acs_x, acs_y, 'o-', linewidth=2, markersize=6, color='#06A77D', alpha=0.7)
    ax4.plot(acs_x[0], acs_y[0], 'r*', markersize=15, label=f'Start (City {starting_city})')
    
    for i, city in enumerate(acs_path[:-1]):
        ax4.annotate(str(city), (acs_x[i], acs_y[i]), 
                    textcoords="offset points", xytext=(0, 5), ha='center', fontsize=7)
    
    ax4.set_title(f"ACS: Best Tour (Cost: {acs_cost:.2f})", fontsize=13, fontweight='bold')
    ax4.set_xlabel("X Coordinate", fontsize=11)
    ax4.set_ylabel("Y Coordinate", fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(left=0)
    ax4.set_ylim(bottom=0)
    
    # ----- ACS Statistics -----
    ax5 = plt.subplot(2, 3, 6)
    ax5.axis('off')
    
    acs_stats = f"""
    ANT COLONY SYSTEM RESULTS
    {'=' * 35}
    
    Best Cost: {acs_cost:.2f}
    
    Parameters:
    • Colony Size:          {colony_size}
    • Alpha (pheromone):    {alpha}
    • Beta (heuristic):     {beta}
    • q0 (exploitation):    {q0}
    • Local Evap Rate:      {local_evop_rate}
    • Global Evap Rate:     {global_evop_rate}
    • Max Iterations:       {max_iters}
    
    Path Length: {len(acs_path)} cities
    """
    
    ax5.text(0.1, 0.5, acs_stats, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='#E8F8F5', alpha=0.8))
    
    # ----- Comparison Panel -----
    ax6 = plt.subplot(2, 3, 4)
    
    algorithms = ['GA', 'ACS']
    costs = [ga_cost, acs_cost]
    colors = ['#2E86AB', '#06A77D']
    
    bars = ax6.bar(algorithms, costs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{cost:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax6.set_title("Algorithm Comparison", fontsize=13, fontweight='bold')
    ax6.set_ylabel("Total Tour Cost", fontsize=11)
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_ylim(bottom=0)
    
    # Add winner annotation
    winner_idx = costs.index(min(costs))
    ax6.text(winner_idx, costs[winner_idx] * 0.5, '★ WINNER ★', 
             ha='center', fontsize=12, fontweight='bold', color='gold',
             bbox=dict(boxstyle='round', facecolor='darkgreen', alpha=0.7))
    
    plt.suptitle(f'TSP Solutions Comparison - Starting City: {starting_city}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig('tsp_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved as 'tsp_comparison.png'")
    plt.show()
    
    print("\nDone! Close the plot window to exit.")