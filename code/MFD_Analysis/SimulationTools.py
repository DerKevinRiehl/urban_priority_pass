###############################################################################
### Author: Kevin Riehl <kriehl@ethz.ch>
### Date: 01.12.2024
### Organization: ETH Zürich, Institute for Transport Planning and Systems (IVT)
### Project: Urban Priority Pass - Fair Intersection Management
###############################################################################
### This file contains tools generate controller settings for simulations.
###############################################################################




###############################################################################
############################## IMPORTS
###############################################################################

import numpy as np




###############################################################################
############################## GENERATE CONTROLLER SETTINGS BY TYPE
###############################################################################

def generateSpawnEntranceProbabilities(spawn_entrances_probabilities, flow):
    out = {}
    for key in spawn_entrances_probabilities:
        out[key] = flow/60/60
    return out




###############################################################################
############################## GENERATE CONTROLLER SETTINGS BY TYPE
###############################################################################

def genFixProgrammeController(phase_durations, time_delay):
    return {
        "type": "fixed_programme",
        "transition_duration": 3,
        "time_delay": time_delay,
        "phase_durations": phase_durations
    }

def genMaxPressureController(min_green, auct_sus):
    return {
        "type": "max_pressure",
        "transition_duration": 3,
        "bidding_strategy": "phase_queue_length",
        "auction_winner": "highest_bid",
        "min_green_duration": min_green,
        "max_green_duration": 60,
        "auction_suspend_duration": auct_sus,
    }

def genPriorityPassController(tau, min_green, auct_sus):
    return {
        "type": "priority_pass",
        "transition_duration": 3,
        "bidding_strategy": "phase_queue_length",
        "auction_winner": "highest_bid",
        "min_green_duration": min_green,
        "max_green_duration": 60,
        "auction_suspend_duration": auct_sus,
        "trade_off": tau,
    }




###############################################################################
############################## ANALYSIS FUNCTIONS
###############################################################################

def analyse_experiment_delays(settings, recorder):
    # Vehicle Statistics
    completed_vehicle_ids = recorder.getVehiclePopulationCompleted()
    population_delay_time = recorder.getVehiclePopulationDelayTime(completed_vehicle_ids, vot=False)
    population_routes = recorder.getVehiclePopulationRoute(completed_vehicle_ids)
    population_lengths_km = [settings.route_length[route]/1000 for route in population_routes]
    # population_delay_time_per_intersection = np.asarray(population_delay_time) / np.asarray(population_intersections)
    population_delay_time_per_distance = np.asarray(population_delay_time) / np.asarray(population_lengths_km)
    veh_av_delay_time = np.nanmean(population_delay_time_per_distance)
    veh_md_delay_time = np.nanmedian(population_delay_time_per_distance)
    veh_st_delay_time = np.nanstd(population_delay_time_per_distance)
    return veh_av_delay_time, veh_md_delay_time, veh_st_delay_time

def analyse_experiment(settings, recorder):
    # Intersection Statistics
    int_total_throughput = 0
    int_total_av_queue_length = 0
    for intersection in settings.tl_control:
        int_total_throughput += recorder.getIntersectionVehicleThroughput(intersection)
        int_total_av_queue_length += recorder.getIntersectionAverageQueueLength(intersection)
    # Vehicle Statistics
    all_vehicle_ids = list(recorder.vehicles.route.keys())
    population_time_spent = recorder.getVehiclePopulationTravelTime(all_vehicle_ids)
    completed_vehicle_ids = recorder.getVehiclePopulationCompleted()
    int_num_vehicles_completed = len(completed_vehicle_ids)
    population_delay_time = recorder.getVehiclePopulationDelayTime(completed_vehicle_ids, vot=False)
    population_intersections = recorder.getVehiclePopulationIntersections(completed_vehicle_ids)
    population_routes = recorder.getVehiclePopulationRoute(completed_vehicle_ids)
    population_lengths_km = [settings.route_length[route]/1000 for route in population_routes]
    # population_delay_time_per_intersection = np.asarray(population_delay_time) / np.asarray(population_intersections)
    population_delay_time_per_distance = np.asarray(population_delay_time) / np.asarray(population_lengths_km)
    veh_av_delay_time = np.nanmean(population_delay_time_per_distance)
    veh_md_delay_time = np.nanmedian(population_delay_time_per_distance)
    veh_st_delay_time = np.nanstd(population_delay_time_per_distance)
    results = {
        "int_total_throughput": int_total_throughput, 
        "int_total_av_queue_length": int_total_av_queue_length, 
        "int_num_vehicles_completed": int_num_vehicles_completed, 
        "sum(population_intersections)": sum(population_intersections), 
        "len(population_time_spent)": len(population_time_spent), 
        "sum(population_time_spent)": sum(population_time_spent), 
        "veh_av_delay_time": veh_av_delay_time, 
        "veh_md_delay_time": veh_md_delay_time, 
        "veh_st_delay_time": veh_st_delay_time
        }
    return results

def _getAverageDelays_DifferentGroups(settings, recorder):
    completed_vehicle_ids = recorder.getVehiclePopulationCompleted()
    upp_vehicles = recorder.getVehiclePopulationUPP(completed_vehicle_ids)
    del_vehicles = recorder.getVehiclePopulationDelayTime(completed_vehicle_ids, vot=False)
    # int_vehicles = recorder.getVehiclePopulationIntersections(completed_vehicle_ids)
    population_routes = recorder.getVehiclePopulationRoute(completed_vehicle_ids)
    population_lengths_km = [settings.route_length[route]/1000 for route in population_routes]
    del_upp = []
    del_nupp = []
    dis_upp = []
    dis_nupp = []
    for x in range(0, len(completed_vehicle_ids)):
        if upp_vehicles[x]==1:
            del_upp.append(del_vehicles[x])
            dis_upp.append(population_lengths_km[x])
        else:
            del_nupp.append(del_vehicles[x])
            dis_nupp.append(population_lengths_km[x])
    del_upp = np.asarray(del_upp) / np.asarray(dis_upp)
    del_nupp = np.asarray(del_nupp) / np.asarray(dis_nupp)
    upp_veh_av_delay_time = np.nanmean(del_upp)
    upp_veh_md_delay_time = np.nanmedian(del_upp)
    upp_veh_st_delay_time = np.nanstd(del_upp)
    nupp_veh_av_delay_time = np.nanmean(del_nupp)
    nupp_veh_md_delay_time = np.nanmedian(del_nupp)
    nupp_veh_st_delay_time = np.nanstd(del_nupp)
    return upp_veh_av_delay_time, upp_veh_md_delay_time, upp_veh_st_delay_time, nupp_veh_av_delay_time, nupp_veh_md_delay_time, nupp_veh_st_delay_time

def analyse_experiment_PriorityPass(settings, recorder):
    # Intersection Statistics
    int_total_throughput = 0
    int_total_av_queue_length = 0
    for intersection in settings.tl_control:
        int_total_throughput += recorder.getIntersectionVehicleThroughput(intersection)
        int_total_av_queue_length += recorder.getIntersectionAverageQueueLength(intersection)
    # Vehicle Statistics
    all_vehicle_ids = list(recorder.vehicles.route.keys())
    population_time_spent = recorder.getVehiclePopulationTravelTime(all_vehicle_ids)
    completed_vehicle_ids = recorder.getVehiclePopulationCompleted()
    int_num_vehicles_completed = len(completed_vehicle_ids)
    population_delay_time = recorder.getVehiclePopulationDelayTime(completed_vehicle_ids, vot=False)
    population_intersections = recorder.getVehiclePopulationIntersections(completed_vehicle_ids)
    population_routes = recorder.getVehiclePopulationRoute(completed_vehicle_ids)
    population_lengths_km = [settings.route_length[route]/1000 for route in population_routes]
    # population_delay_time_per_intersection = np.asarray(population_delay_time) / np.asarray(population_intersections)
    population_delay_time_per_distance = np.asarray(population_delay_time) / np.asarray(population_lengths_km)
    veh_av_delay_time = np.nanmean(population_delay_time_per_distance)
    veh_md_delay_time = np.nanmedian(population_delay_time_per_distance)
    veh_st_delay_time = np.nanstd(population_delay_time_per_distance)
    # UPP statistics
    upp_veh_av_delay_time, upp_veh_md_delay_time, upp_veh_st_delay_time, nupp_veh_av_delay_time, nupp_veh_md_delay_time, nupp_veh_st_delay_time = _getAverageDelays_DifferentGroups(settings, recorder)
    
    results = {
        "int_total_throughput": int_total_throughput, 
        "int_total_av_queue_length": int_total_av_queue_length, 
        "int_num_vehicles_completed": int_num_vehicles_completed, 
        "sum(population_intersections)": sum(population_intersections), 
        "len(population_time_spent)": len(population_time_spent), 
        "sum(population_time_spent)": sum(population_time_spent), 
        "veh_av_delay_time": veh_av_delay_time, 
        "veh_md_delay_time": veh_md_delay_time, 
        "veh_st_delay_time": veh_st_delay_time, 
        "upp_veh_av_delay_time": upp_veh_av_delay_time, 
        "upp_veh_md_delay_time": upp_veh_md_delay_time, 
        "upp_veh_st_delay_time": upp_veh_st_delay_time, 
        "nupp_veh_av_delay_time": nupp_veh_av_delay_time, 
        "nupp_veh_md_delay_time": nupp_veh_md_delay_time, 
        "nupp_veh_st_delay_time": nupp_veh_st_delay_time
    }
    return results
