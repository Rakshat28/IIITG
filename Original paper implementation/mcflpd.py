import numpy as np
import heapq
from ortools.linear_solver import pywraplp
import random
from typing import List, Tuple, Dict, Set

class MCFLPD_Solver:
    """
    A solver class for the Maximum Coverage Capacitated Facility Location Problem
    with Range Constrained Drones (MCFLPD).

    This class implements two heuristic approaches:
    1. A Greedy Heuristic
    2. A Three-Stage Heuristic (3SH)
    """

    def __init__(self,
                 p: int,
                 K: int,
                 U: float,
                 B: float,
                 d: List[float],
                 b: List[List[float]],
                 w: List[float]):
        """
        Initialize the MCFLPD solver with problem parameters.

        Args:
            p: Maximum number of facilities to open.
            K: Total number of drones available across all facilities.
            U: Maximum capacity of each facility (e.g., total demand it can serve).
            B: Maximum battery capacity of a single drone (e.g., total battery consumption
               for a round trip to demand points).
            d: List of demands for each demand point 'i'. d[i] is the demand at point i.
            b: 2D list of battery consumption. b[i][j] is the battery consumed by a drone
               for a round trip between demand point 'i' and facility 'j'.
            w: List of weights (or priorities) for each demand point 'i'. w[i] represents
               the value of serving demand point i. The objective is to maximize the sum of w[i]
               for all served demand points.
        """
        self.p = p
        self.K = K
        self.U = U
        self.B = B
        self.d = d
        self.b = b
        self.w = w
        self.num_demand_points = len(d)
        # Determine the number of potential facility locations.
        # It's assumed that b[i][j] implies 'j' is a facility location.
        self.num_facility_locations = len(b[0]) if self.num_demand_points > 0 else 0

    def greedy_heuristic(self) -> Tuple[float, List[int], Dict[int, List[List[int]]]]:
        """
        Implements a greedy heuristic for the MCFLPD.
        This heuristic prioritizes assigning demand points to facilities based on a
        weight-to-battery-consumption ratio, then allocates drones.

        Returns:
            Tuple containing:
            - total_demand_served: The sum of demands (or weights, depending on 'w')
                                   of all demand points successfully served.
            - open_facilities: A list of indices of facilities that are opened.
            - assignments: A dictionary where keys are facility indices (j) and values
                           are lists of drone assignments for that facility. Each drone
                           assignment is a list of demand point indices (i) served by that drone.
        """
        # Step 1: Create and sort a list of potential assignments.
        # Each item is (negative_weight_ratio, demand_point_idx, facility_idx).
        # We use negative weight_ratio to simulate a max-heap with heapq (which is a min-heap).
        # Only consider assignments where a drone can physically reach the demand point (battery constraint).
        weight_matrix = []
        for i in range(self.num_demand_points):
            for j in range(self.num_facility_locations):
                if self.b[i][j] <= self.B:  # Check drone battery range constraint
                    # Calculate the ratio of demand point weight to battery consumption.
                    # A higher ratio means more "value" per unit of battery.
                    # Handle division by zero for battery consumption if it's possible.
                    weight_ratio = self.w[i] / self.b[i][j] if self.b[i][j] > 0 else float('inf')
                    weight_matrix.append((-weight_ratio, i, j))

        # Convert the list into a min-heap for efficient extraction of the highest ratio.
        heapq.heapify(weight_matrix)

        # Initialize solution variables
        open_facilities_set: Set[int] = set()  # Use a set for efficient checking of open facilities
        facility_current_demand: Dict[int, float] = {}  # Tracks total demand assigned to each facility
        # Maps demand point 'i' to its assigned facility 'j'.
        demand_to_facility_assignment: Dict[int, int] = {}
        # Tracks remaining capacity at each open facility.
        facility_remaining_capacity: Dict[int, float] = {}

        # Step 2: Demand allocation based on sorted weight ratios.
        # Iterate while there are potential assignments and we haven't assigned all demands.
        while weight_matrix and len(demand_to_facility_assignment) < self.num_demand_points:
            neg_weight_ratio, i, j = heapq.heappop(weight_matrix)
            weight_ratio = -neg_weight_ratio

            # Skip if the demand point is already assigned or if the ratio is non-positive (invalid).
            if i in demand_to_facility_assignment or weight_ratio <= 0:
                continue

            # Check if facility 'j' is already open.
            if j in open_facilities_set:
                # If facility 'j' is open, try to assign demand 'i' to it.
                if facility_remaining_capacity[j] >= self.d[i]:
                    demand_to_facility_assignment[i] = j
                    facility_current_demand[j] += self.d[i]
                    facility_remaining_capacity[j] -= self.d[i]
            else:
                # If facility 'j' is not open, try to open it if allowed by 'p' and capacity.
                if len(open_facilities_set) < self.p and self.d[i] <= self.U:
                    open_facilities_set.add(j)
                    demand_to_facility_assignment[i] = j
                    facility_current_demand[j] = self.d[i]
                    facility_remaining_capacity[j] = self.U - self.d[i]
                else:
                    # If facility 'j' cannot be opened (p limit reached or demand too large for new facility),
                    # try to assign demand 'i' to an already open facility with the lowest battery consumption.
                    best_j_for_i = None
                    min_battery_consumption = float('inf')
                    for open_j in open_facilities_set:
                        if (self.b[i][open_j] <= self.B and
                                facility_remaining_capacity[open_j] >= self.d[i] and
                                self.b[i][open_j] < min_battery_consumption):
                            best_j_for_i = open_j
                            min_battery_consumption = self.b[i][open_j]

                    if best_j_for_i is not None:
                        demand_to_facility_assignment[i] = best_j_for_i
                        facility_current_demand[best_j_for_i] += self.d[i]
                        facility_remaining_capacity[best_j_for_i] -= self.d[i]

        open_facilities = list(open_facilities_set) # Convert set to list for consistent return type

        # Step 3: Drone allocation to facilities.
        # Estimate the number of drones needed per facility based on total battery consumption
        # for assigned demands.
        drones_estimated_needed = {j: 0 for j in open_facilities}
        for j in open_facilities:
            total_battery_for_facility = sum(self.b[i][j] for i, assigned_j in demand_to_facility_assignment.items() if assigned_j == j)
            # Each drone can carry B battery units.
            drones_estimated_needed[j] = int(np.ceil(total_battery_for_facility / self.B))

        # Allocate actual drones (up to K total) to facilities.
        # Prioritize facilities that need more drones.
        drone_actual_allocation = {j: 0 for j in open_facilities}
        remaining_total_drones = self.K

        # Create a priority queue to distribute drones, prioritizing facilities with higher estimated needs.
        # Store (-estimated_drones_needed, facility_idx) to use min-heap as max-heap.
        drone_priority_queue = []
        for j in open_facilities:
            if drones_estimated_needed[j] > 0: # Only add facilities that need drones
                heapq.heappush(drone_priority_queue, (-drones_estimated_needed[j], j))

        while remaining_total_drones > 0 and drone_priority_queue:
            neg_estimated_drones, j = heapq.heappop(drone_priority_queue)
            estimated_drones = -neg_estimated_drones

            # Allocate one drone to this facility if it still needs more than it has.
            if drone_actual_allocation[j] < estimated_drones:
                drone_actual_allocation[j] += 1
                remaining_total_drones -= 1

                # If the facility still needs more drones, put it back in the queue with updated priority.
                if drone_actual_allocation[j] < estimated_drones:
                    heapq.heappush(drone_priority_queue, (-(estimated_drones - drone_actual_allocation[j]), j))

        # Step 4: Assign demand points to individual drones within each facility.
        # This uses a "first-fit decreasing" like approach for each drone's battery capacity.
        # Each drone assignment is a list of demand point indices.
        assignments: Dict[int, List[List[int]]] = {j: [[] for _ in range(drone_actual_allocation[j])] for j in open_facilities}

        # Keep track of demands that were initially assigned to a facility but couldn't be served by a drone.
        unserved_demands_due_to_drone_capacity = set()

        for j in open_facilities:
            # Get all demand points assigned to this facility 'j' from Step 2.
            facility_assigned_demands = [i for i, assigned_j in demand_to_facility_assignment.items() if assigned_j == j]

            # Sort these demand points by their battery consumption to facility 'j' in ascending order.
            # This helps in fitting more demands into a drone's battery capacity.
            facility_assigned_demands.sort(key=lambda i: self.b[i][j])

            drone_idx_counter = 0 # To cycle through available drones for this facility
            for i in facility_assigned_demands:
                assigned_to_drone = False
                # Iterate through the drones allocated to facility 'j'
                # Start from the current drone_idx_counter to distribute load.
                for _ in range(drone_actual_allocation[j]): # Try all drones at this facility
                    current_drone_idx = drone_idx_counter % drone_actual_allocation[j] if drone_actual_allocation[j] > 0 else -1

                    if current_drone_idx == -1: # No drones allocated to this facility
                        break

                    # Calculate current battery load on this specific drone.
                    current_load_on_drone = sum(self.b[di][j] for di in assignments[j][current_drone_idx])

                    # Check if adding demand 'i' to this drone would exceed its battery capacity.
                    if current_load_on_drone + self.b[i][j] <= self.B:
                        assignments[j][current_drone_idx].append(i)
                        assigned_to_drone = True
                        break # Demand 'i' is assigned, move to next demand point.
                    
                    drone_idx_counter += 1 # Try next drone for the current demand if current one is full.

                if not assigned_to_drone:
                    # If demand 'i' could not be assigned to any drone at facility 'j',
                    # it means it cannot be served. Mark it as unserved.
                    unserved_demands_due_to_drone_capacity.add(i)

        # Calculate total demand served based on final assignments.
        # Only count demands that were successfully assigned to a drone.
        total_demand_served = 0.0
        for j_assignments in assignments.values():
            for drone_path in j_assignments:
                for i in drone_path:
                    total_demand_served += self.w[i] # Sum of weights for covered demands

        return total_demand_served, open_facilities, assignments

    def three_stage_heuristic(self, r: int = 1, max_iter: int = 100) -> Tuple[float, List[int], Dict[int, List[List[int]]]]:
        """
        Implements the Three-Stage Heuristic (3SH) for MCFLPD.
        This heuristic combines an initial facility location/allocation, a repeated
        knapsack for drone assignment, and a local r-exchange search.

        Args:
            r: Number of facilities to exchange in the local search (r-exchange).
            max_iter: Maximum number of iterations for the local search phase.

        Returns:
            Tuple containing:
            - total_demand_served: The sum of demands (or weights) of all demand points
                                   successfully served by the best found solution.
            - open_facilities: A list of indices of facilities opened in the best solution.
            - assignments: A dictionary mapping facility index to list of drone assignments
                           (each drone assignment is a list of demand point indices)
                           for the best found solution.
        """
        best_solution_assignments = None
        best_demand_served = -1.0 # Initialize with a value lower than any possible demand
        best_open_facilities = []

        # Stage 1: Initial Facility Location and Allocation (Random Initialization)
        # Start with a random set of 'p' open facilities.
        if self.num_facility_locations < self.p:
            # Handle case where there are fewer potential locations than facilities to open
            initial_open_facilities = list(range(self.num_facility_locations))
        else:
            initial_open_facilities = random.sample(range(self.num_facility_locations), self.p)

        current_open_facilities = initial_open_facilities.copy()

        for iteration in range(max_iter):
            # Stage 1 (within iteration): Solve the Facility Allocation Problem (FLAP)
            # This assigns demand points to the *current* set of open facilities,
            # maximizing the weighted demand served, respecting facility capacity and drone range.
            # Note: This ILP assumes a drone can reach the facility from the demand point.
            # The actual drone capacity (number of trips) is handled in Stage 2.
            demand_to_facility_assignment, current_total_demand_from_allocation = \
                self._solve_allocation_problem(current_open_facilities)

            # Stage 2: Drone Assignment using Repeated Knapsack
            # For the demands assigned in Stage 1, this stage determines how demands are grouped
            # onto individual drones at each facility, respecting drone battery capacity and
            # the total number of available drones (K).
            current_assignments, current_demand_served_by_drones = \
                self._drone_assignment_knapsack(current_open_facilities, demand_to_facility_assignment)

            # Update best solution if current iteration yields a better result.
            if current_demand_served_by_drones > best_demand_served:
                best_demand_served = current_demand_served_by_drones
                best_solution_assignments = current_assignments
                best_open_facilities = current_open_facilities.copy()

            # Stage 3: Local Search (r-exchange heuristic)
            # This step attempts to improve the set of open facilities by exchanging
            # 'r' currently open facilities with 'r' currently closed facilities.

            # If no facilities are open or cannot perform exchange, break.
            if not current_open_facilities or len(current_open_facilities) < r:
                break

            # Calculate demand served by each currently open facility (for selecting facilities to remove).
            facility_demand_contribution = {j: 0.0 for j in current_open_facilities}
            for i, assigned_j in demand_to_facility_assignment.items():
                if assigned_j in facility_demand_contribution: # Ensure the facility is still in current_open_facilities
                    facility_demand_contribution[assigned_j] += self.w[i] # Use weight 'w' as contribution

            # Identify 'r' facilities with the lowest demand contribution to be removed.
            # Sort facilities by their contribution in ascending order.
            sorted_facilities_by_contribution = sorted(current_open_facilities,
                                                       key=lambda j: facility_demand_contribution.get(j, 0.0))
            facilities_to_remove = sorted_facilities_by_contribution[:r]

            # Select 'r' new facilities from the remaining (closed) locations.
            remaining_potential_locations = [j for j in range(self.num_facility_locations)
                                             if j not in current_open_facilities]

            if len(remaining_potential_locations) < r:
                # If not enough new facilities to exchange, stop local search.
                break

            # Randomly pick 'r' new facilities to open.
            new_facilities_to_add = random.sample(remaining_potential_locations, r)

            # Create the new set of open facilities for the next iteration.
            current_open_facilities = [j for j in current_open_facilities if j not in facilities_to_remove] + new_facilities_to_add

        # Return the best solution found over all iterations.
        return best_demand_served, best_open_facilities, best_solution_assignments

    def _solve_allocation_problem(self, open_facilities: List[int]) -> Tuple[Dict[int, int], float]:
        """
        Solves the facility allocation subproblem for a given set of open facilities.
        This is formulated as an Integer Linear Program (ILP) to maximize the total
        weighted demand served, subject to facility capacity and drone reachability.

        Args:
            open_facilities: A list of indices of facilities that are currently considered open.

        Returns:
            Tuple containing:
            - demand_assignment: A dictionary mapping demand point index 'i' to its
                                 assigned facility index 'j'. Only includes assigned demands.
            - total_weighted_demand: The sum of weights of all demand points successfully assigned.
        """
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            print("Error: SCIP solver not available.")
            return {}, 0.0

        # Decision variables: x_ij = 1 if demand point 'i' is assigned to facility 'j', 0 otherwise.
        x = {}
        for i in range(self.num_demand_points):
            for j in open_facilities:
                # A demand point 'i' can only be assigned to facility 'j' if a drone
                # can make the trip (battery consumption <= drone battery capacity).
                if self.b[i][j] <= self.B:
                    x[i, j] = solver.IntVar(0, 1, f'x_{i}_{j}')

        # Objective: Maximize the total weighted demand served.
        # The coefficient for x_ij should be w[i] (the weight/value of serving demand i).
        objective = solver.Objective()
        for (i, j), var in x.items():
            objective.SetCoefficient(var, self.w[i]) # Maximize the sum of weights of covered demands
        objective.SetMaximization()

        # Constraint 1: Each demand point can be assigned to at most one facility.
        # Sum of x_ij over all facilities 'j' for a given demand 'i' must be <= 1.
        for i in range(self.num_demand_points):
            constraint = solver.Constraint(0, 1, f'demand_assignment_limit_{i}')
            for j in open_facilities:
                if (i, j) in x:
                    constraint.SetCoefficient(x[i, j], 1)

        # Constraint 2: Facility capacity constraint.
        # The total demand assigned to any facility 'j' cannot exceed its capacity 'U'.
        for j in open_facilities:
            constraint = solver.Constraint(0, self.U, f'facility_capacity_{j}')
            for i in range(self.num_demand_points):
                if (i, j) in x:
                    constraint.SetCoefficient(x[i, j], self.d[i]) # Sum of demands assigned to facility j

        # Solve the ILP problem.
        status = solver.Solve()

        # Extract the solution.
        demand_assignment = {}
        total_weighted_demand_served = 0.0

        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            for i in range(self.num_demand_points):
                for j in open_facilities:
                    if (i, j) in x and x[i, j].solution_value() > 0.5: # Check if x_ij is essentially 1
                        demand_assignment[i] = j
                        total_weighted_demand_served += self.w[i]
                        break # Move to the next demand point, as it's assigned.
        else:
            print(f"Warning: Allocation problem did not find an optimal or feasible solution. Status: {status}")

        return demand_assignment, total_weighted_demand_served

    def _drone_assignment_knapsack(self,
                                   open_facilities: List[int],
                                   demand_to_facility_assignment: Dict[int, int]) \
                                   -> Tuple[Dict[int, List[List[int]]], float]:
        """
        Solves the drone assignment problem using a repeated knapsack approach.
        This function distributes the total available drones (K) among the open facilities
        and then assigns demand points to individual drones at each facility, maximizing
        the total weighted demand served while respecting drone battery capacity.

        Args:
            open_facilities: A list of indices of facilities that are currently open.
            demand_to_facility_assignment: A dictionary mapping demand point index 'i' to its
                                           assigned facility index 'j' (from Stage 1).

        Returns:
            Tuple containing:
            - assignments: A dictionary where keys are facility indices (j) and values
                           are lists of drone assignments for that facility. Each drone
                           assignment is a list of demand point indices (i) served by that drone.
            - total_weighted_demand_served: The total sum of weights of demands successfully
                                            assigned to drones.
        """
        # Initialize the structure to store drone assignments for each facility.
        # Each inner list represents a single drone's path/assignments.
        assignments: Dict[int, List[List[int]]] = {j: [] for j in open_facilities}
        total_weighted_demand_served = 0.0
        remaining_total_drones = self.K

        # Group demands by their assigned facility.
        # Store as (demand_id, battery_consumption_to_facility, demand_weight)
        facility_demands_for_knapsack: Dict[int, List[Tuple[int, float, float]]] = \
            {j: [] for j in open_facilities}
        for i, j in demand_to_facility_assignment.items():
            facility_demands_for_knapsack[j].append((i, self.b[i][j], self.w[i]))

        # Iteratively assign drones until all drones are used or no more demands can be served.
        while remaining_total_drones > 0:
            knapsack_results_for_iteration = [] # Stores (value, facility_idx, selected_items) for this iteration

            # For each open facility, solve a knapsack problem to find the best set of demands
            # that can be served by one additional drone from that facility.
            for j in open_facilities:
                if not facility_demands_for_knapsack[j]:
                    continue # No remaining demands for this facility

                # Solve a knapsack problem for the demands assigned to facility 'j'
                # to find the maximum weighted demand that can be served by a single drone (capacity B).
                selected_items_for_drone, value_from_drone = \
                    self._solve_knapsack(facility_demands_for_knapsack[j])

                if value_from_drone > 0: # Only consider positive value assignments
                    knapsack_results_for_iteration.append((value_from_drone, j, selected_items_for_drone))

            if not knapsack_results_for_iteration:
                # No more demands can be assigned to any drone across all facilities.
                break

            # Select the knapsack solution (i.e., the drone assignment) that yields the maximum value.
            # This is a greedy choice: use a drone where it can serve the most weighted demand.
            knapsack_results_for_iteration.sort(key=lambda x: x[0], reverse=True)
            best_value_this_drone, best_j_for_drone, best_selected_items_for_drone = \
                knapsack_results_for_iteration[0]

            # Assign this drone's path to the facility.
            assignments[best_j_for_drone].append([i for i, _, _ in best_selected_items_for_drone])
            total_weighted_demand_served += best_value_this_drone
            remaining_total_drones -= 1

            # Remove the assigned demands from the pool of available demands for that facility,
            # as they are now served by the newly allocated drone.
            selected_demand_ids = set(i for i, _, _ in best_selected_items_for_drone)
            facility_demands_for_knapsack[best_j_for_drone] = [
                item for item in facility_demands_for_knapsack[best_j_for_drone]
                if item[0] not in selected_demand_ids
            ]

        return assignments, total_weighted_demand_served

    def _solve_knapsack(self, items: List[Tuple[int, float, float]]) -> Tuple[List[Tuple[int, float, float]], float]:
        """
        Solves a 0/1 knapsack problem for a single drone.
        Given a list of items (demand points), each with a battery consumption (weight)
        and a demand weight (profit), select items to maximize total profit without
        exceeding the drone's battery capacity (knapsack capacity).

        Args:
            items: A list of tuples, where each tuple represents a demand point:
                   (demand_point_id, battery_consumption, demand_weight).

        Returns:
            Tuple containing:
            - selected_items: A list of tuples representing the demand points selected for this drone.
                              (same format as input items).
            - total_profit: The total demand weight (profit) of the selected items.
        """
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            print("Error: SCIP solver not available for knapsack.")
            return [], 0.0

        # Decision variables: x_idx = 1 if item at 'idx' in the 'items' list is selected, 0 otherwise.
        x = {}
        for idx, (i, bij, di) in enumerate(items):
            x[idx] = solver.IntVar(0, 1, f'x_item_{i}')

        # Constraint: Total battery consumption (weight) must not exceed drone's battery capacity (B).
        knapsack_capacity_constraint = solver.Constraint(0, self.B, 'drone_battery_capacity')
        for idx, (i, bij, di) in enumerate(items):
            knapsack_capacity_constraint.SetCoefficient(x[idx], bij) # bij is the weight for the knapsack

        # Objective: Maximize the total demand weight (profit) of selected items.
        objective = solver.Objective()
        for idx, (i, bij, di) in enumerate(items):
            objective.SetCoefficient(x[idx], di) # di is the profit for the knapsack
        objective.SetMaximization()

        # Solve the knapsack problem.
        status = solver.Solve()

        # Extract the solution.
        selected_items = []
        total_profit = 0.0

        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            for idx, item in enumerate(items):
                if x[idx].solution_value() > 0.5: # Check if item is selected
                    selected_items.append(item)
                    total_profit += item[2] # Add the demand weight (profit)
        else:
            print(f"Warning: Knapsack problem did not find an optimal or feasible solution. Status: {status}")

        return selected_items, total_profit

# Example usage
if __name__ == "__main__":
    # Example problem parameters
    p = 3  # Maximum number of facilities to open
    K = 5  # Total number of drones available
    U = 100.0  # Facility capacity (e.g., total demand it can serve)
    B = 50.0  # Drone battery capacity (e.g., max battery consumption for a trip)

    # Example data (small problem)
    num_demand_points = 10
    num_facility_locations = 5

    # Random data generation for demonstration
    np.random.seed(42) # For reproducibility
    d = np.random.randint(1, 20, size=num_demand_points).tolist()  # Demands at each point
    w = [float(val) for val in d] # Using demand as weight for simplicity, can be different
    # Battery consumption between demand point i and facility j
    b = np.random.randint(1, 30, size=(num_demand_points, num_facility_locations)).tolist()

    # Create solver instance
    solver = MCFLPD_Solver(p, K, U, B, d, b, w)

    print("Running Greedy Heuristic...")
    greedy_demand, greedy_facilities, greedy_assignments = solver.greedy_heuristic()
    print(f"\nGreedy Heuristic Results:")
    print(f"  Total weighted demand served: {greedy_demand:.2f}")
    print(f"  Open facilities: {greedy_facilities}")
    print(f"  Assignments (Facility: [[Drone1_demands], [Drone2_demands]]):")
    for facility, drone_paths in greedy_assignments.items():
        if drone_paths: # Only print if there are actual drone assignments
            print(f"    Facility {facility}:")
            for idx, path in enumerate(drone_paths):
                if path: # Only print non-empty drone paths
                    path_battery_sum = sum(solver.b[i][facility] for i in path)
                    path_demand_sum = sum(solver.d[i] for i in path)
                    path_weight_sum = sum(solver.w[i] for i in path)
                    print(f"      Drone {idx}: Demands {path} (Battery: {path_battery_sum:.2f}/{solver.B}, Demand: {path_demand_sum:.2f}, Weight: {path_weight_sum:.2f})")
                else:
                    print(f"      Drone {idx}: No demands assigned (Idle)")
        else:
            print(f"    Facility {facility}: No drones allocated or no demands assigned.")


    print("\n" + "="*50 + "\n")

    print("Running Three-Stage Heuristic...")
    # r=1 means exchange 1 facility at each iteration. max_iter can be tuned.
    tsh_demand, tsh_facilities, tsh_assignments = solver.three_stage_heuristic(r=1, max_iter=50)
    print(f"\nThree-Stage Heuristic Results:")
    print(f"  Total weighted demand served: {tsh_demand:.2f}")
    print(f"  Open facilities: {tsh_facilities}")
    print(f"  Assignments (Facility: [[Drone1_demands], [Drone2_demands]]):")
    for facility, drone_paths in tsh_assignments.items():
        if drone_paths: # Only print if there are actual drone assignments
            print(f"    Facility {facility}:")
            for idx, path in enumerate(drone_paths):
                if path: # Only print non-empty drone paths
                    path_battery_sum = sum(solver.b[i][facility] for i in path)
                    path_demand_sum = sum(solver.d[i] for i in path)
                    path_weight_sum = sum(solver.w[i] for i in path)
                    print(f"      Drone {idx}: Demands {path} (Battery: {path_battery_sum:.2f}/{solver.B}, Demand: {path_demand_sum:.2f}, Weight: {path_weight_sum:.2f})")
                else:
                    print(f"      Drone {idx}: No demands assigned (Idle)")
        else:
            print(f"    Facility {facility}: No drones allocated or no demands assigned.")

