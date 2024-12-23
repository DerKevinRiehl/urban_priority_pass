###############################################################################
### Author: Kevin Riehl <kriehl@ethz.ch>
### Date: 01.12.2024
### Organization: ETH Zürich, Institute for Transport Planning and Systems (IVT)
### Project: Urban Priority Pass - Fair Intersection Management
###############################################################################
### This file ...
###############################################################################




###############################################################################
############################## Urgency Model with 10,000 Individuals and 10 Urgency Levels
###############################################################################

from Data_NewYorkSalaryDistribution import salary_distribution
from Data_RouteDistribution import route_durations, route_distances

import numpy as np
np.random.seed(42)

def getUrgencyLevelProcess(p):
    urgency_dist = []
    urgency_level = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    urgency_dist = [p*np.power(1-p, k-1) for k in urgency_level]
    urgency_dist = urgency_dist/sum(urgency_dist)
    return urgency_dist, urgency_level

urgency_scenarios = [0.6, 0.5, 0.4]

urgency_distribution = {}
ctr = 1

pop_size = 10000
pop_salary = np.random.choice(salary_distribution["hour_salary"], pop_size , p=np.asarray(salary_distribution["share_population"])/sum(salary_distribution["share_population"]))
routes = list(route_durations.keys())
pop_routes = np.random.choice(routes, pop_size)
pop_route_durations = [route_durations[route] for route in pop_routes]
pop_route_distances = [route_distances[route] for route in pop_routes]

for scenario in urgency_scenarios:
    urgency_dist, urgency_level = getUrgencyLevelProcess(p=scenario)
    pop_urgency_levels = np.random.choice(urgency_level, pop_size, p=urgency_dist)
    pop_urgency_vots = np.asarray(pop_salary)*np.asarray(pop_urgency_levels)
    
    urgency_distribution[ctr] = {
        "scenario": scenario,
        "pop_salary": pop_salary,
        "pop_urgency_levels": pop_urgency_levels,
        "pop_urgency_vots": pop_urgency_vots,
        "pop_route_durations": pop_route_durations,
        "pop_route_distances": pop_route_distances
    }
    ctr += 1
