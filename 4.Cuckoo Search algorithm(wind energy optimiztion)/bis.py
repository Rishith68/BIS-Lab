import numpy as np

# Wind Farm Parameters
N_TURBINES = 4 # Number of turbines to optimize
WIND_DIR = np.array([1.0, 0.0]) # Wind blowing from West to East

# Let's define an objective function that we want to MINIMIZE.
# We want to maximize total power output, which is equivalent to minimizing negative power output.
def wind_farm_neg_power(positions_flat):
    # Reshape to (N_TURBINES, 2) representing (x, y) coordinates of each turbine
    positions = positions_flat.reshape((N_TURBINES, 2))
    total_power = 0.0
    
    # Base power output of a single turbine
    base_power = 1.0 
    
    for i in range(N_TURBINES):
        wake_loss = 0.0
        for j in range(N_TURBINES):
            if i == j: 
                continue
            # Vector from turbine j to turbine i
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            
            # Distance project along wind direction
            dist_along_wind = dx * WIND_DIR[0] + dy * WIND_DIR[1]
            # Cross-wind distance
            dist_cross_wind = np.abs(-dx * WIND_DIR[1] + dy * WIND_DIR[0])
            
            # If turbine i is downwind of turbine j, it suffers wake losses
            if dist_along_wind > 0:
                # Simplified wake model
                wake_width = 1.0 + 0.1 * dist_along_wind
                if dist_cross_wind < wake_width:
                    # Loss decreases with crosswind and downwind distance
                    loss = (1.0 / (1.0 + 0.5 * dist_along_wind)) * (1.0 - (dist_cross_wind / wake_width))
                    wake_loss = max(wake_loss, loss)
        
        turbine_power = base_power * (1.0 - 0.5 * wake_loss) # Max 50% power loss
        total_power += turbine_power
        
    # Add a penalty for turbines being placed too close to each other overall (safety distance)
    penalty = 0.0
    min_safety_dist = 2.0
    for i in range(N_TURBINES):
        for j in range(i + 1, N_TURBINES):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < min_safety_dist:
                penalty += 10.0 * (min_safety_dist - dist)**2
                
    return -(total_power) + penalty

import numpy as np
import scipy.special as special

# Helper function to generate Levy Flight steps using Mantegna's algorithm
def levy_flight(beta, dim):
    num = special.gamma(1 + beta) * np.sin(np.pi * beta / 2)
    den = special.gamma((1 + beta) / 2) * beta * (2**((beta - 1) / 2))
    sigma = (num / den)**(1 / beta)
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)
    return u / (np.abs(v)**(1 / beta))

# Cuckoo Search Algorithm
def cuckoo_search(obj_func, dim, n_nests=25, max_iter=100, Pa=0.25, lb=-5.12, ub=5.12):
    nests = np.random.uniform(lb, ub, (n_nests, dim))
    fitness = np.array([obj_func(nest) for nest in nests])
    
    best_idx = np.argmin(fitness)
    best_nest = nests[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    beta = 1.5
    for iteration in range(max_iter):
        for i in range(n_nests):
            step = levy_flight(beta, dim)
            step_size = 0.01 * step * (nests[i] - best_nest)
            new_solution = nests[i] + step_size * np.random.normal(0, 1, dim)
            new_solution = np.clip(new_solution, lb, ub)
            
            new_fitness = obj_func(new_solution)
            random_idx = np.random.randint(0, n_nests)
            if new_fitness < fitness[random_idx]:
                nests[random_idx] = new_solution
                fitness[random_idx] = new_fitness
        
        num_abandon = int(Pa * n_nests)
        worst_indices = np.argsort(fitness)[-num_abandon:]
        for idx in worst_indices:
            r1, r2 = np.random.randint(0, n_nests, 2)
            step_size = np.random.rand() * (nests[r1] - nests[r2])
            new_nest = nests[idx] + step_size
            new_nest = np.clip(new_nest, lb, ub)
            nests[idx] = new_nest
            fitness[idx] = obj_func(new_nest)
            
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx]
            best_nest = nests[current_best_idx].copy()
            
    return best_nest, best_fitness

# Solve wind energy optimization
dim = N_TURBINES * 2 # Coordinate x and y for each turbine
lb, ub = 0.0, 10.0 # Wind farm boundaries (10x10 km grid)

best_layout_flat, min_neg_power = cuckoo_search(
    wind_farm_neg_power,
    dim=dim,
    n_nests=30,
    max_iter=300,
    Pa=0.25,
    lb=lb,
    ub=ub
)

best_layout = best_layout_flat.reshape((N_TURBINES, 2))
max_power_achieved = -min_neg_power

print(f"Optimized Turbine Coordinates (x, y):\n{best_layout}\n")
print(f"Estimated Maximum Power Output: {max_power_achieved:.4f} MW")