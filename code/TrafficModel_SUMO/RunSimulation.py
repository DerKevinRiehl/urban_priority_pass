# Signalized Intersection With Auction Control, Copyrights Kevin Riehl 2023, <kriehl@ethz.ch>




# Imports
from Settings import Settings
from Simulation import Simulation
import numpy as np
import json
import sys


# Methods
def do_simulation(settings, results, wait=0.0):
    label = str(np.random.random())
    simulation = Simulation(settings, label) 
    simulation.open_simulation()
    simulation.run_simulation_loop(wait=wait)
    simulation.close_simulation()
    results.append(analyse_experiment(settings, simulation.recorder))

def do_simulation_UPP(settings, results, wait=0.0):
    label = str(np.random.random())
    simulation = Simulation(settings, label) 
    simulation.open_simulation()
    simulation.run_simulation_loop(wait=wait)
    simulation.close_simulation()
    results.append(analyse_experiment_UPP(settings, simulation.recorder))
    
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
    return [int_total_throughput, int_total_av_queue_length, int_num_vehicles_completed, sum(population_intersections), 
            len(population_time_spent), sum(population_time_spent), 
            veh_av_delay_time, veh_md_delay_time, veh_st_delay_time]

def getUPPAverageDelays(recorder):
    completed_vehicle_ids = recorder.getVehiclePopulationCompleted()
    upp_vehicles = recorder.getVehiclePopulationUPP(completed_vehicle_ids)
    del_vehicles = recorder.getVehiclePopulationDelayTime(completed_vehicle_ids)
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

def analyse_experiment_UPP(settings, recorder):
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
    upp_veh_av_delay_time, upp_veh_md_delay_time, upp_veh_st_delay_time, nupp_veh_av_delay_time, nupp_veh_md_delay_time, nupp_veh_st_delay_time = getUPPAverageDelays(recorder)
    return [int_total_throughput, int_total_av_queue_length, int_num_vehicles_completed, sum(population_intersections), 
            len(population_time_spent), sum(population_time_spent), 
            veh_av_delay_time, veh_md_delay_time, veh_st_delay_time, 
            upp_veh_av_delay_time, upp_veh_md_delay_time, upp_veh_st_delay_time, 
            nupp_veh_av_delay_time, nupp_veh_md_delay_time, nupp_veh_st_delay_time]

def do_experiment(settings, n_reps=5):
    results = []
    for it in range(0, n_reps):
        do_simulation(settings, results)
        
    return len(results), np.nanmean(results, axis=0), np.nanmedian(results, axis=0), np.nanstd(results, axis=0)
    
def do_experiment_UPP(settings, n_reps=5):
    results = []
    for it in range(0, n_reps):
        do_simulation_UPP(settings, results)
    return len(results), np.nanmean(results, axis=0), np.nanmedian(results, axis=0), np.nanstd(results, axis=0)
    

def writeToLogFile(file, line):
    f = open(file, "a")
    f.write(line)
    f.write("\n")
    f.close()
    
def logResult(file, parameters, n, resultMean, resultMedian, resultStd):
    line = "Parameters: \n"
    for p in parameters:
        line += str(p)+" "
    line += "\nN: "+str(n)+"\n"
    line += "\nResultMean: \n"
    for r in resultMean:
        line += str(r)+" "
    line += "\nResultMedian: \n"
    for r in resultMedian:
        line += str(r)+" "
    line += "\nResultStd: \n"
    for r in resultStd:
        line += str(r)+" "
    writeToLogFile(file, line)
    
    
    
    
# SETUP Settings
settings = Settings()
    # SUMO SPECIFIC
# settings.sumo_location = "sumo" # LINUX
settings.sumo_location = "C:/Users/kriehl/AppData/Local/sumo-1.19.0/bin/sumo.exe"#"sumo.exe"#"sumo-gui.exe"
settings.sumo_config_file = "../../models/Manhattan3x3/Configuration.sumocfg"
# NETWORK SPECIFIC
    # route_specific
settings.spawn_entrances_probabilities = {"0": 0.1, "1": 0.1, "2": 0.1, "3": 0.1, "4": 0.1, "5": 0.1, "6": 0.1, "7": 0.1, "8": 0.1, "9": 0.1, "10": 0.1, "11": 0.1}
settings.spawn_entrances_routes_probabilities = json.load(open("Maps/Manhattan3x3/Route_Probabilities.json", "r"))
settings.route_min_possible_travel_time = json.load(open("Maps/Manhattan3x3/Route_Durations.json", "r"))
settings.route_recording_start_edge = json.load(open("Maps/Manhattan3x3/Route_StartEdges.json", "r"))
settings.route_recording_completion_edge = json.load(open("Maps/Manhattan3x3/Route_EndEdges.json", "r"))
settings.route_length = json.load(open("Maps/Manhattan3x3/Route_Distances.json", "r"))
def genSpawnEntrProbs(spawn_entrances_probabilities, flow):
    out = {}
    for key in spawn_entrances_probabilities:
        out[key] = flow/60/60
    return out
    # intersection specific
settings.phase_bidder_lanes = json.load(open("Maps/Manhattan3x3/Phase_BidderLanes.json", "r")) 
settings.phase_leaver_lanes = json.load(open("Maps/Manhattan3x3/Phase_ExitLanes.json", "r"))
# CONTROL SPECIFIC
def genFixProgramme(phase_durations, time_delay):
    return {
        "type": "fixed_programme",
        "transition_duration": 3,
        "time_delay": time_delay,
        "phase_durations": phase_durations
    }
def genAuctionProgrammeQueueLength(min_green, auct_sus):
    return {
        "type": "phase_auction",
        "transition_duration": 3,
        "bidding_strategy": "phase_queue_length",
        "auction_winner": "highest_bid",
        "min_green_duration": min_green,
        "max_green_duration": 60,
        "auction_suspend_duration": auct_sus,
    }
def genAuctionProgrammeQueueLength_UPP(alpha, tau, min_green, auct_sus):
    return {
        "type": "phase_auction_upp",
        "transition_duration": 3,
        "bidding_strategy": "phase_queue_length",
        "auction_winner": "highest_bid",
        "min_green_duration": min_green,
        "max_green_duration": 60,
        "auction_suspend_duration": auct_sus,
        "trade_off": tau,
    }

traffic_flows = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]




# # Determine Best "FIXED_PROGRAMME" Parameters
# # Parse Run Args
# flow = int(sys.argv[1])
# parameter1 = int(sys.argv[2])
# parameter2 = int(sys.argv[3])
# repe = int(sys.argv[4])
# logfile = "Logs/Manhattan3x3/log_fixed_programme_"+str(flow)+"_"+str(parameter1)+"_"+str(parameter2)+"_"+str(repe)+".txt"
# results = []
# traffic_flows = [flow]
# for flow in traffic_flows:
#     settings.spawn_entrances_probabilities = genSpawnEntrProbs(settings.spawn_entrances_probabilities, flow)
#     print(">>>>",flow)
#     writeToLogFile(logfile, ">>>> "+str(flow)+", "+str(settings.spawn_entrances_probabilities))
#     # for parameter1 in range(1, 40, 1):
#     # for parameter2 in range(1, 40, 1):
#     settings.tl_control = {
#         "J25": genFixProgramme([parameter1, parameter2, parameter1, parameter2], 0),
#         "J26": genFixProgramme([parameter1, parameter2, parameter1, parameter2], int((parameter1+parameter2)/2) ),
#         "J27": genFixProgramme([parameter1, parameter2, parameter1, parameter2], 0),
#         "J28": genFixProgramme([parameter1, parameter2, parameter1, parameter2], int((parameter1+parameter2)/2) ),
#         "J29": genFixProgramme([parameter1, parameter2, parameter1, parameter2], 0),
#         "J30": genFixProgramme([parameter1, parameter2, parameter1, parameter2], 0),
#         "J31": genFixProgramme([parameter1, parameter2, parameter1, parameter2], int((parameter1+parameter2)/2) ),
#         "J32": genFixProgramme([parameter1, parameter2, parameter1, parameter2], int((parameter1+parameter2)/2) ),
#         "J33": genFixProgramme([parameter1, parameter2, parameter1, parameter2], 0)
#     }
#     # results = []
#     # simulation = do_simulation(settings, results, wait=0.0)
#     n, resultMean, resultMedian, resultStd = do_experiment(settings, n_reps=1)
#     logResult(logfile, [parameter1], n, resultMean, resultMedian, resultStd)
#     print(parameter1, *resultMean)




# # Determine Best "PHASE_AUCTION" Parameters
# results = []
# # Parse Run Args
# flow = int(sys.argv[1])
# parameter1 = int(sys.argv[2])
# parameter2 = int(sys.argv[3])
# repe = int(sys.argv[4])
# logfile = "Logs/Manhattan3x3/log_auction_queue_"+str(flow)+"_"+str(parameter1)+"_"+str(parameter2)+"_"+str(repe)+".txt"
#     # CONTROL SPECIFIC
# traffic_flows = [flow]
# for flow in traffic_flows:
#     settings.spawn_entrances_probabilities = genSpawnEntrProbs(settings.spawn_entrances_probabilities, flow)
#     print(">>>>",flow)
#     writeToLogFile(logfile, ">>>> "+str(flow)+", "+str(settings.spawn_entrances_probabilities))
#     # for parameter1 in range(1, 40, 1):
#     # for parameter2 in range(1, 40, 1):
#     settings.tl_control = {
#         "J25": genAuctionProgrammeQueueLength(min_green=parameter1, auct_sus=parameter2),
#         "J26": genAuctionProgrammeQueueLength(min_green=parameter1, auct_sus=parameter2),
#         "J27": genAuctionProgrammeQueueLength(min_green=parameter1, auct_sus=parameter2),
#         "J28": genAuctionProgrammeQueueLength(min_green=parameter1, auct_sus=parameter2),
#         "J29": genAuctionProgrammeQueueLength(min_green=parameter1, auct_sus=parameter2),
#         "J30": genAuctionProgrammeQueueLength(min_green=parameter1, auct_sus=parameter2),
#         "J31": genAuctionProgrammeQueueLength(min_green=parameter1, auct_sus=parameter2),
#         "J32": genAuctionProgrammeQueueLength(min_green=parameter1, auct_sus=parameter2),
#         "J33": genAuctionProgrammeQueueLength(min_green=parameter1, auct_sus=parameter2),
#     }
#     # results = []
#     # simulation = do_simulation(settings, results, wait=0.1)            
#     n, resultMean, resultMedian, resultStd = do_experiment(settings, n_reps=1)
#     logResult(logfile, [parameter1, parameter2], n, resultMean, resultMedian, resultStd)
#     print(parameter1, *resultMean)





# Determine Best "PHASE_AUCTION_UPP" Parameters
results = []
# Parse Run Args
flow = int(sys.argv[1])
parameter1 = float(sys.argv[2])
parameter2 = float(sys.argv[3])
repe = int(sys.argv[4])
logfile = "Logs/Manhattan3x3/log_auction_queue_UPP_"+str(flow)+"_"+str(parameter1)+"_"+str(parameter2)+"_"+str(repe)+".txt"
    # CONTROL SPECIFIC
traffic_flows = [flow]
# traffic_flows=[300]
# parameter1 = 0.1 
# parameter2 = 0.8
# Opt for total travel time
auction_min_green = {
    50:         3,
    100:        3,
    150:        4,
    200:        3,
    250:        5,
    300:        6,
    350:        8,
    400:        10,
    450:        2,
    500:        1,
    550:        2
}
auction_auc_sus = {
    50:         3,
    100:        5,
    150:        6,
    200:        4,
    250:        5,
    300:        5,
    350:        9,
    400:        9,
    450:        40,
    500:        39,
    550:        38
}
#      Flow  Parameter1  Parameter2
# 0    50.0         3.0         3.0
# 1   100.0         3.0         5.0
# 2   150.0         4.0         6.0
# 3   200.0         3.0         4.0
# 4   250.0         5.0         5.0
# 5   300.0         6.0         5.0
# 6   350.0         8.0         9.0
# 7   400.0        10.0         9.0
# 8   450.0         2.0        40.0
# 9   500.0         1.0        39.0
# 10  550.0         2.0        38.0


for flow in traffic_flows:
    settings.spawn_entrances_probabilities = genSpawnEntrProbs(settings.spawn_entrances_probabilities, flow)
    settings.vot_spawn_probabilities = {0: 1-parameter1, 1: parameter1}
    settings.vot_upp_spawn_probabilities = {0: 0.0, 1: 1.0}
    print(">>>>",flow)
    writeToLogFile(logfile, ">>>> "+str(flow)+", "+str(settings.spawn_entrances_probabilities))
    # for parameter1 in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    # for parameter2 in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    settings.tl_control = {
        "J25": genAuctionProgrammeQueueLength_UPP(alpha=parameter1, tau=parameter2, min_green=auction_min_green[flow], auct_sus=auction_auc_sus[flow]),
        "J26": genAuctionProgrammeQueueLength_UPP(alpha=parameter1, tau=parameter2, min_green=auction_min_green[flow], auct_sus=auction_auc_sus[flow]),
        "J27": genAuctionProgrammeQueueLength_UPP(alpha=parameter1, tau=parameter2, min_green=auction_min_green[flow], auct_sus=auction_auc_sus[flow]),
        "J28": genAuctionProgrammeQueueLength_UPP(alpha=parameter1, tau=parameter2, min_green=auction_min_green[flow], auct_sus=auction_auc_sus[flow]),
        "J29": genAuctionProgrammeQueueLength_UPP(alpha=parameter1, tau=parameter2, min_green=auction_min_green[flow], auct_sus=auction_auc_sus[flow]),
        "J30": genAuctionProgrammeQueueLength_UPP(alpha=parameter1, tau=parameter2, min_green=auction_min_green[flow], auct_sus=auction_auc_sus[flow]),
        "J31": genAuctionProgrammeQueueLength_UPP(alpha=parameter1, tau=parameter2, min_green=auction_min_green[flow], auct_sus=auction_auc_sus[flow]),
        "J32": genAuctionProgrammeQueueLength_UPP(alpha=parameter1, tau=parameter2, min_green=auction_min_green[flow], auct_sus=auction_auc_sus[flow]),
        "J33": genAuctionProgrammeQueueLength_UPP(alpha=parameter1, tau=parameter2, min_green=auction_min_green[flow], auct_sus=auction_auc_sus[flow]),
    }
    # results = []
    # simulation = do_simulation_UPP(settings, results, wait=0.1)            
    n, resultMean, resultMedian, resultStd = do_experiment_UPP(settings, n_reps=1)
    logResult(logfile, [parameter1, parameter2], n, resultMean, resultMedian, resultStd)
    print(parameter1, *resultMean)