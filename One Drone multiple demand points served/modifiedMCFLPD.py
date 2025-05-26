import numpy as np
import heapq
from ortools.linear_solver import pywraplp
import random
from typing import List, Tuple, Dict, Set
import math # For infinity and floor/ceil

class MCFLPD_Solver:
    """
    A solver class for the Maximum Coverage Capacitated Facility Location Problem
    with Range Constrained Drones (MCFLPD).

    This class implements two heuristic approaches, modified to allow
    drones to deliver to multiple demand points on a single route:
    1. A Greedy Heuristic (modified for multi-stop routes)
    2. A Three-Stage Heuristic (3SH - modified for multi-stop routes)
    """

    def __init__(self,
                 p: int,
                 K: int,
                 U: float,
                 B: float,
                 d: List[float],
                 w: List[float],
                 facility_to_demand_battery: List[List[float]],
                 demand_to_demand_battery: List[List[float]] = None): # New parameter for inter-demand battery
        """
        Initialize the MCFLPD solver with problem parameters.

        Args:
            p: Maximum number of facilities to open.
            K: Total number of drones available across all facilities.
            U: Maximum capacity of each facility (e.g., total demand it can serve).
            B: Maximum battery capacity of a single drone (e.g., total battery consumption
               for a round trip for a route).
            d: List of demands for each demand point 'i'. d[i] is the demand at point i.
            w: List of weights (or priorities) for each demand point 'i'. w[i] represents
               the value of serving demand point i. The objective is to maximize the sum of w[i]
               for all served demand points.
            facility_to_demand_battery: 2D list. facility_to_demand_battery[i][j] is the battery consumed by a drone
                                        for a ONE-WAY trip between demand point 'i' and facility 'j'.
                                        (This is a crucial re-interpretation from the original 'b')
            demand_to_demand_battery: Optional 2D list. demand_to_demand_battery[i1][i2] is the battery consumed
                                      for a ONE-WAY trip between demand point i1 and demand point i2.
                                      If not provided, a simplified Euclidean-like proxy will be used for demo.
        """
        self.p = p
        self.K = K
        self.U = U
        self.B = B
        self.d = d
        self.w = w
        self.facility_to_demand_battery = facility_to_demand_battery # f_i_j
        self.num_demand_points = len(d)
        self.num_facility_locations = len(facility_to_demand_battery[0]) if self.num_demand_points > 0 else 0

        # Pre-compute or use provided demand-to-demand battery
        if demand_to_demand_battery:
            self.demand_to_demand_battery = demand_to_demand_battery
        else:
            # If not provided, create a dummy one for demonstration.
            # In a real scenario, this would come from geo-coordinates or a proper distance matrix.
            self.demand_to_demand_battery = [[0.0 for _ in range(self.num_demand_points)] for _ in range(self.num_demand_points)]
            # Simple Euclidean-like proxy for demonstration:
            # Assuming demand points are on a 2D plane for a rough estimation
            # This is a very rough approximation, in reality, you'd need actual coordinates or a distance matrix.
            # Here, I'll just use facility_to_demand_battery to simulate some variability.
            for i1 in range(self.num_demand_points):
                for i2 in range(self.num_demand_points):
                    if i1 == i2:
                        self.demand_to_demand_battery[i1][i2] = 0.0
                    else:
                        # This is a very rough proxy. In real-world, you'd have true point-to-point distances.
                        # For example, using distances from a dummy facility 0.
                        # This part needs concrete definition for a real problem.
                        self.demand_to_demand_battery[i1][i2] = abs(self.facility_to_demand_battery[i1][0] - self.facility_to_demand_battery[i2][0]) + 0.1 # Ensure >0

        # A drone can physically reach demand point 'i' from facility 'j'
        # if the round trip for just that point is <= B
        self.reachable = [[False for _ in range(self.num_facility_locations)] for _ in range(self.num_demand_points)]
        for i in range(self.num_demand_points):
            for j in range(self.num_facility_locations):
                if (self.facility_to_demand_battery[i][j] * 2) <= self.B: # Check round trip for single point
                    self.reachable[i][j] = True

    def _calculate_route_battery_cost(self, facility_idx: int, route_points: List[int]) -> float:
        """
        Calculates the total battery cost for a multi-stop route.
        Assumes starting from facility, visiting all points in order, and returning to facility.

        Args:
            facility_idx: The index of the facility where the route originates.
            route_points: A list of demand point indices in the order they are visited.

        Returns:
            The total battery consumed for the route. Returns float('inf') if route is invalid.
        """
        if not route_points:
            return 0.0

        total_battery = 0.0
        current_location_is_facility = True
        last_point_visited = -1 # Dummy value

        for i, current_point in enumerate(route_points):
            if current_location_is_facility:
                # Facility to first point
                total_battery += self.facility_to_demand_battery[current_point][facility_idx]
                current_location_is_facility = False
            else:
                # Point to next point
                # Check if this segment is valid
                if self.demand_to_demand_battery[last_point_visited][current_point] > 0: # Ensure valid connection
                    total_battery += self.demand_to_demand_battery[last_point_visited][current_point]
                else:
                    return float('inf') # Invalid segment, should not happen with proper matrix
            last_point_visited = current_point

        # Return from last point to facility
        total_battery += self.facility_to_demand_battery[last_point_visited][facility_idx]

        return total_battery

    def greedy_heuristic(self) -> Tuple[float, List[int], Dict[int, List[List[int]]]]:
        """
        Implements a greedy heuristic for the MCFLPD, supporting multi-stop routes.
        This heuristic prioritizes assigning demand points to facilities, then builds
        multi-stop routes for drones.

        Returns:
            Tuple containing:
            - total_demand_served: The sum of demands (or weights, depending on 'w')
                                   of all demand points successfully served.
            - open_facilities: A list of indices of facilities that are opened.
            - assignments: A dictionary where keys are facility indices (j) and values
                           are lists of drone assignments for that facility. Each drone
                           assignment is a list of demand point indices (i) served by that drone.
        """
        # Step 1: Initial facility selection (simplified greedy, could be random)
        # Select 'p' facilities based on some initial heuristic, e.g., facilities with highest
        # potential reach, or simply randomly for a starting point.
        # For a simple greedy, let's just use the first 'p' facilities as candidates to start.
        if self.num_facility_locations == 0: return 0.0, [], {}
        
        # A more robust greedy might evaluate based on sum of reachable weights
        initial_candidate_facilities = list(range(self.num_facility_locations))
        if self.num_facility_locations > self.p:
             # Sort facilities by potential total weighted demand reachable (simplified)
            facility_potential_weights = []
            for j in range(self.num_facility_locations):
                total_reachable_weight = 0
                for i in range(self.num_demand_points):
                    if self.reachable[i][j]:
                        total_reachable_weight += self.w[i]
                facility_potential_weights.append((total_reachable_weight, j))
            facility_potential_weights.sort(key=lambda x: x[0], reverse=True)
            open_facilities_set = set(j for _, j in facility_potential_weights[:self.p])
        else:
            open_facilities_set = set(range(self.num_facility_locations)) # Open all available if less than p

        open_facilities = list(open_facilities_set)

        # Initialize solution variables
        assignments: Dict[int, List[List[int]]] = {j: [] for j in open_facilities}
        served_demand_points: Set[int] = set()
        facility_current_demand: Dict[int, float] = {j: 0.0 for j in open_facilities}

        # Step 2: Iteratively build routes and assign drones until K drones are used
        # or no more demands can be served.
        remaining_total_drones = self.K

        while remaining_total_drones > 0 and len(served_demand_points) < self.num_demand_points:
            best_route_value = -1.0
            best_route_details = None # (facility_idx, [route_points], route_battery_cost, route_demand_sum)

            # For each open facility, try to find the best next route
            for j in open_facilities:
                # Find unserved demands reachable from this facility
                unserved_reachable_demands = [i for i in range(self.num_demand_points)
                                              if i not in served_demand_points and self.reachable[i][j]]

                if not unserved_reachable_demands:
                    continue

                # Build a greedy route: start with the highest weighted unserved demand,
                # then add nearest unserved demands until battery or capacity limit is hit.
                # Sort by weight descending for greedy selection
                unserved_reachable_demands.sort(key=lambda i: self.w[i], reverse=True)

                current_route = []
                current_route_battery = 0.0
                current_route_demand_sum = 0.0
                current_route_weighted_sum = 0.0

                temp_route_points = []
                temp_current_battery = 0.0
                temp_current_demand = 0.0
                temp_current_weight = 0.0
                last_point_on_route = -1 # Represents the last demand point added

                for next_dp in unserved_reachable_demands:
                    # Calculate cost of adding next_dp to current_route
                    if not temp_route_points: # First point in the route
                        cost_to_add = self.facility_to_demand_battery[next_dp][j] # To first point
                        cost_return = self.facility_to_demand_battery[next_dp][j] # From first point
                    else: # Subsequent point in the route
                        cost_to_add = self.demand_to_demand_battery[last_point_on_route][next_dp]
                        cost_return = self.facility_to_demand_battery[next_dp][j] # From last point of temp_route

                    # Check battery constraint for adding this point
                    # We are recalculating the full route battery for every addition to ensure validity
                    hypothetical_route_points = temp_route_points + [next_dp]
                    hypothetical_total_battery = self._calculate_route_battery_cost(j, hypothetical_route_points)

                    # Check facility capacity for adding this point
                    if (hypothetical_total_battery <= self.B and
                            facility_current_demand[j] + self.d[next_dp] <= self.U):
                        
                        temp_route_points.append(next_dp)
                        temp_current_demand += self.d[next_dp]
                        temp_current_weight += self.w[next_dp]
                        last_point_on_route = next_dp
                    # If we can't add this point, maybe the next one is better?
                    # For a simple greedy, we just try the next best demand point.
                    # A more sophisticated approach would be to find the "best fit" among remaining.

                if temp_route_points: # If a valid route was formed for this facility
                    # Use the weight of the demands as the value of the route
                    if temp_current_weight > best_route_value:
                        best_route_value = temp_current_weight
                        best_route_details = (j, temp_route_points, hypothetical_total_battery, temp_current_demand)

            if best_route_details:
                facility_idx, route_points, route_battery_cost, route_demand_sum = best_route_details

                assignments[facility_idx].append(route_points)
                for dp_idx in route_points:
                    served_demand_points.add(dp_idx)
                facility_current_demand[facility_idx] += route_demand_sum
                remaining_total_drones -= 1
            else:
                # No more routes can be formed that satisfy constraints
                break

        total_demand_served = sum(self.w[i] for i in served_demand_points)
        return total_demand_served, open_facilities, assignments

    def three_stage_heuristic(self, r: int = 1, max_iter: int = 100) -> Tuple[float, List[int], Dict[int, List[List[int]]]]:
        """
        Implements the Three-Stage Heuristic (3SH) for MCFLPD, modified for multi-stop routes.
        This heuristic combines an initial facility location/allocation, a repeated
        routing heuristic for drone assignment, and a local r-exchange search.

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
        best_demand_served = -1.0
        best_open_facilities = []

        # Stage 1: Initial Facility Location (Random Initialization)
        if self.num_facility_locations < self.p:
            initial_open_facilities = list(range(self.num_facility_locations))
        else:
            initial_open_facilities = random.sample(range(self.num_facility_locations), self.p)

        current_open_facilities = initial_open_facilities.copy()

        for iteration in range(max_iter):
            # Stage 1 (within iteration): Solve the Facility Allocation Problem (FLAP)
            # This assigns demand points to the *current* set of open facilities,
            # maximizing the weighted demand served, respecting facility capacity and drone reachability.
            demand_to_facility_assignment, current_total_demand_from_allocation = \
                self._solve_allocation_problem(current_open_facilities)

            # Stage 2: Drone Routing and Assignment using a repeated heuristic
            # For the demands assigned in Stage 1, this stage determines how demands are grouped
            # onto individual drones at each facility to form routes, respecting drone battery capacity
            # and the total number of available drones (K).
            current_assignments, current_demand_served_by_drones = \
                self._drone_routing_heuristic(current_open_facilities, demand_to_facility_assignment)

            # Update best solution if current iteration yields a better result.
            if current_demand_served_by_drones > best_demand_served:
                best_demand_served = current_demand_served_by_drones
                best_solution_assignments = current_assignments
                best_open_facilities = current_open_facilities.copy()

            # Stage 3: Local Search (r-exchange heuristic)
            if not current_open_facilities or len(current_open_facilities) < r:
                break # Cannot perform exchange

            # Calculate demand served by each currently open facility (for selecting facilities to remove).
            # This is based on the actual routed demands for this iteration.
            facility_demand_contribution = {j: 0.0 for j in current_open_facilities}
            for j, routes in current_assignments.items():
                for route in routes:
                    for dp_idx in route:
                        facility_demand_contribution[j] += self.w[dp_idx]

            # Identify 'r' facilities with the lowest demand contribution to be removed.
            sorted_facilities_by_contribution = sorted(current_open_facilities,
                                                       key=lambda j: facility_demand_contribution.get(j, 0.0))
            facilities_to_remove = sorted_facilities_by_contribution[:r]

            # Select 'r' new facilities from the remaining (closed) locations.
            remaining_potential_locations = [j for j in range(self.num_facility_locations)
                                             if j not in current_open_facilities]

            if len(remaining_potential_locations) < r:
                break # Not enough new facilities to exchange

            new_facilities_to_add = random.sample(remaining_potential_locations, r)

            current_open_facilities = [j for j in current_open_facilities if j not in facilities_to_remove] + new_facilities_to_add

        return best_demand_served, best_open_facilities, best_solution_assignments

    def _solve_allocation_problem(self, open_facilities: List[int]) -> Tuple[Dict[int, int], float]:
        """
        Solves the facility allocation subproblem for a given set of open facilities.
        This is formulated as an Integer Linear Program (ILP) to maximize the total
        weighted demand served, subject to facility capacity and drone reachability
        (for a single direct trip, as multi-stop routes are handled in the next stage).

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
                # can make a round trip for just that point (battery consumption <= drone battery capacity).
                # This ensures basic reachability for allocation. Multi-stop routing is handled later.
                if self.reachable[i][j]: # Use the pre-calculated reachability based on single round trip
                    x[i, j] = solver.IntVar(0, 1, f'x_{i}_{j}')

        # Objective: Maximize the total weighted demand served.
        objective = solver.Objective()
        for (i, j), var in x.items():
            objective.SetCoefficient(var, self.w[i])
        objective.SetMaximization()

        # Constraint 1: Each demand point can be assigned to at most one facility.
        for i in range(self.num_demand_points):
            constraint = solver.Constraint(0, 1, f'demand_assignment_limit_{i}')
            for j in open_facilities:
                if (i, j) in x:
                    constraint.SetCoefficient(x[i, j], 1)

        # Constraint 2: Facility capacity constraint.
        for j in open_facilities:
            constraint = solver.Constraint(0, self.U, f'facility_capacity_{j}')
            for i in range(self.num_demand_points):
                if (i, j) in x:
                    constraint.SetCoefficient(x[i, j], self.d[i])

        status = solver.Solve()

        demand_assignment = {}
        total_weighted_demand_served = 0.0

        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            for i in range(self.num_demand_points):
                for j in open_facilities:
                    if (i, j) in x and x[i, j].solution_value() > 0.5:
                        demand_assignment[i] = j
                        total_weighted_demand_served += self.w[i]
                        break
        else:
            print(f"Warning: Allocation problem did not find an optimal or feasible solution. Status: {status}")

        return demand_assignment, total_weighted_demand_served

    def _drone_routing_heuristic(self,
                                 open_facilities: List[int],
                                 demand_to_facility_assignment: Dict[int, int]) \
                                 -> Tuple[Dict[int, List[List[int]]], float]:
        """
        Performs drone routing and assignment for multi-stop routes using a greedy approach.
        This function groups demand points assigned to a facility into routes, respecting
        drone battery capacity (B) and the total number of available drones (K).

        Args:
            open_facilities: A list of indices of facilities that are currently open.
            demand_to_facility_assignment: A dictionary mapping demand point index 'i' to its
                                           assigned facility index 'j' (from Stage 1).

        Returns:
            Tuple containing:
            - assignments: A dictionary where keys are facility indices (j) and values
                           are lists of drone assignments for that facility. Each drone
                           assignment is a list of demand point indices (i) representing a route.
            - total_weighted_demand_served: The total sum of weights of demands successfully
                                            assigned to drones via routes.
        """
        assignments: Dict[int, List[List[int]]] = {j: [] for j in open_facilities}
        total_weighted_demand_served = 0.0
        remaining_total_drones = self.K

        # Group demands by their assigned facility.
        facility_assigned_demands: Dict[int, List[int]] = \
            {j: [] for j in open_facilities}
        for i, j in demand_to_facility_assignment.items():
            facility_assigned_demands[j].append(i)

        # Track served demands for this routing stage to avoid double counting
        served_in_routing: Set[int] = set()

        # Iteratively create and assign routes
        while remaining_total_drones > 0 and len(served_in_routing) < self.num_demand_points:
            best_route_value = -1.0
            best_route_details = None # (facility_idx, [route_points], route_battery_cost, route_demand_sum)

            for j in open_facilities:
                unserved_demands_for_facility = [i for i in facility_assigned_demands[j] if i not in served_in_routing]

                if not unserved_demands_for_facility:
                    continue

                # Sort unserved demands by weight in descending order to prioritize high-value points
                unserved_demands_for_facility.sort(key=lambda i: self.w[i], reverse=True)

                # Simple greedy route building for this facility:
                # Start with the highest weighted unserved demand, then add nearest unserved demands.
                current_route_candidate = []
                current_route_battery_cost = 0.0
                current_route_demand_sum = 0.0
                current_route_weight_sum = 0.0

                temp_served_for_candidate_route = set() # To track points for this specific candidate route

                # Try to build a route starting with the highest weighted point
                for start_dp in unserved_demands_for_facility:
                    if start_dp in temp_served_for_candidate_route: # Already in this candidate route
                        continue

                    # Start a new route with this point
                    route_points_attempt = [start_dp]
                    battery_attempt = self._calculate_route_battery_cost(j, route_points_attempt)
                    demand_attempt = self.d[start_dp]
                    weight_attempt = self.w[start_dp]

                    if battery_attempt > self.B: # Single point already too expensive
                        continue # Try next starting point

                    # Greedily add nearest unserved points to this route
                    last_point_in_route = start_dp
                    potential_additions = [dp for dp in unserved_demands_for_facility if dp not in temp_served_for_candidate_route and dp != start_dp]

                    # Sort potential additions by proximity to the last point in the current route
                    # (This is a very simple nearest neighbor-like addition)
                    potential_additions.sort(key=lambda dp: self.demand_to_demand_battery[last_point_in_route][dp])

                    for next_dp in potential_additions:
                        hypothetical_route = route_points_attempt + [next_dp]
                        hypothetical_battery_cost = self._calculate_route_battery_cost(j, hypothetical_route)
                        hypothetical_demand_sum = demand_attempt + self.d[next_dp]

                        if hypothetical_battery_cost <= self.B and hypothetical_demand_sum <= self.U: # Assuming U applies to total demand on a route too
                            route_points_attempt = hypothetical_route
                            battery_attempt = hypothetical_battery_cost
                            demand_attempt = hypothetical_demand_sum
                            weight_attempt += self.w[next_dp]
                            last_point_in_route = next_dp
                            temp_served_for_candidate_route.add(next_dp) # Mark as considered for THIS candidate route
                        else:
                            # Cannot add this point, try next.
                            # For simple greedy, we don't backtrack.
                            pass

                    # After attempting to build a route from start_dp, compare its value
                    if weight_attempt > best_route_value:
                        best_route_value = weight_attempt
                        best_route_details = (j, route_points_attempt, battery_attempt, demand_attempt)

            if best_route_details:
                facility_idx, route_points, route_battery_cost, route_demand_sum = best_route_details

                assignments[facility_idx].append(route_points)
                for dp_idx in route_points:
                    served_in_routing.add(dp_idx) # Mark points as served in this stage
                    # Remove from current facility's pool so they aren't reconsidered for other routes
                    if dp_idx in facility_assigned_demands[facility_idx]:
                        facility_assigned_demands[facility_idx].remove(dp_idx)
                
                total_weighted_demand_served += best_route_value
                remaining_total_drones -= 1
            else:
                # No more routes can be formed
                break

        return assignments, total_weighted_demand_served


# Example usage (updated for new parameters)
if __name__ == "__main__":
    # Example problem parameters
    p = 3  # Maximum number of facilities to open
    K = 5  # Total number of drones available
    U = 100.0  # Facility capacity (e.g., total demand it can serve)
    B = 50.0  # Drone battery capacity (e.g., max battery consumption for a route)

    # Example data (small problem)
    num_demand_points = 10
    num_facility_locations = 5

    np.random.seed(42) # For reproducibility

    d = np.random.randint(1, 10, size=num_demand_points).tolist()  # Demands at each point
    w = [float(val) for val in d] # Using demand as weight for simplicity

    # New: facility_to_demand_battery (one-way trip)
    # This matrix should ideally come from distances from facilities to demand points.
    # For demo, using random values, ensuring they are not too high
    facility_to_demand_battery = np.random.uniform(5, 15, size=(num_demand_points, num_facility_locations)).tolist()

    # New: demand_to_demand_battery (one-way trip between demand points)
    # This matrix should come from inter-point distances.
    demand_to_demand_battery = np.random.uniform(1, 5, size=(num_demand_points, num_demand_points)).tolist()
    # Make symmetric and zero diagonal
    for i in range(num_demand_points):
        demand_to_demand_battery[i][i] = 0.0
        for j in range(i + 1, num_demand_points):
            demand_to_demand_battery[j][i] = demand_to_demand_battery[i][j]


    # Create solver instance
    solver = MCFLPD_Solver(p, K, U, B, d, w, facility_to_demand_battery, demand_to_demand_battery)

    print("Running Greedy Heuristic (Multi-Stop Routes)...")
    greedy_demand, greedy_facilities, greedy_assignments = solver.greedy_heuristic()
    print(f"\nGreedy Heuristic Results:")
    print(f"  Total weighted demand served: {greedy_demand:.2f}")
    print(f"  Open facilities: {greedy_facilities}")
    print(f"  Assignments (Facility: [[Drone1_route_demands], [Drone2_route_demands]]):")
    for facility, drone_routes in greedy_assignments.items():
        if drone_routes:
            print(f"    Facility {facility}:")
            for idx, route_path in enumerate(drone_routes):
                if route_path:
                    route_battery_sum = solver._calculate_route_battery_cost(facility, route_path)
                    route_demand_sum = sum(solver.d[i] for i in route_path)
                    route_weight_sum = sum(solver.w[i] for i in route_path)
                    print(f"      Drone {idx}: Route {route_path} (Battery: {route_battery_sum:.2f}/{solver.B}, Demand: {route_demand_sum:.2f}, Weight: {route_weight_sum:.2f})")
                else:
                    print(f"      Drone {idx}: No demands assigned (Idle)")
        else:
            print(f"    Facility {facility}: No drones allocated or no demands assigned.")

    print("\n" + "="*50 + "\n")

    print("Running Three-Stage Heuristic (Multi-Stop Routes)...")
    tsh_demand, tsh_facilities, tsh_assignments = solver.three_stage_heuristic(r=1, max_iter=50)
    print(f"\nThree-Stage Heuristic Results:")
    print(f"  Total weighted demand served: {tsh_demand:.2f}")
    print(f"  Open facilities: {tsh_facilities}")
    print(f"  Assignments (Facility: [[Drone1_route_demands], [Drone2_route_demands]]):")
    for facility, drone_routes in tsh_assignments.items():
        if drone_routes:
            print(f"    Facility {facility}:")
            for idx, route_path in enumerate(drone_routes):
                if route_path:
                    route_battery_sum = solver._calculate_route_battery_cost(facility, route_path)
                    route_demand_sum = sum(solver.d[i] for i in route_path)
                    route_weight_sum = sum(solver.w[i] for i in route_path)
                    print(f"      Drone {idx}: Route {route_path} (Battery: {route_battery_sum:.2f}/{solver.B}, Demand: {route_demand_sum:.2f}, Weight: {route_weight_sum:.2f})")
                else:
                    print(f"      Drone {idx}: No demands assigned (Idle)")
        else:
            print(f"    Facility {facility}: No drones allocated or no demands assigned.")