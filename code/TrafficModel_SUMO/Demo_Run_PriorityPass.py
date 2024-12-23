###############################################################################
### Author: Kevin Riehl <kriehl@ethz.ch>
### Date: 01.12.2024
### Organization: ETH Zürich, Institute for Transport Planning and Systems (IVT)
### Project: Urban Priority Pass - Fair Intersection Management
###############################################################################
### This file demonstrates (visually) how a Max-Pressure traffic light programme
### can run using our simulation environment.
###############################################################################




###############################################################################
############################## IMPORTS
###############################################################################

from SimulationTools import generateSpawnEntranceProbabilities, analyse_experiment_PriorityPass, genPriorityPassController
from Settings import Settings
from Simulator import Simulator
import json




###############################################################################
############################## DEFINE SIMULATION SETTINGS
###############################################################################

settings = Settings()

# SIMULATION MODEL SPECIFIC
simulation_model_path = "../../models/Manhattan3x3/"

# SUMO SPECIFIC
# settings.sumo_location = "sumo" # LINUX
settings.sumo_location = "C:/Users/kriehl/AppData/Local/sumo-1.19.0/bin/sumo-gui.exe"#"sumo.exe"#"sumo-gui.exe"
settings.sumo_config_file = simulation_model_path+"Configuration.sumocfg"

# NETWORK SPECIFIC
    # route_specific
settings.spawn_entrances_probabilities = {"0": 0.1, "1": 0.1, "2": 0.1, "3": 0.1, "4": 0.1, "5": 0.1, "6": 0.1, "7": 0.1, "8": 0.1, "9": 0.1, "10": 0.1, "11": 0.1}
settings.spawn_entrances_routes_probabilities = json.load(open(simulation_model_path+"Route_Probabilities.json", "r"))
settings.route_min_possible_travel_time = json.load(open(simulation_model_path+"Route_Durations.json", "r"))
settings.route_recording_start_edge = json.load(open(simulation_model_path+"Route_StartEdges.json", "r"))
settings.route_recording_completion_edge = json.load(open(simulation_model_path+"Route_EndEdges.json", "r"))
settings.route_length = json.load(open(simulation_model_path+"Route_Distances.json", "r"))
    # intersection specific
settings.phase_bidder_lanes = json.load(open(simulation_model_path+"Phase_BidderLanes.json", "r")) 
settings.phase_leaver_lanes = json.load(open(simulation_model_path+"Phase_ExitLanes.json", "r"))

# DEMAND SPECIFIC
flow = 200 # [veh/h]
settings.spawn_entrances_probabilities = generateSpawnEntranceProbabilities(settings.spawn_entrances_probabilities, flow)

# CONTROL SPECIFIC
parameter1 = 3
parameter2 = 4
tau = 0.4
gamma = 0.20

settings.vot_spawn_probabilities = {0: 1-gamma, 1: gamma}
settings.vot_upp_spawn_probabilities = {0: 0.0, 1: 1.0}

settings.tl_control = {
    "J25": genPriorityPassController(tau=tau, min_green=parameter1, auct_sus=parameter2),
    "J26": genPriorityPassController(tau=tau, min_green=parameter1, auct_sus=parameter2),
    "J27": genPriorityPassController(tau=tau, min_green=parameter1, auct_sus=parameter2),
    "J28": genPriorityPassController(tau=tau, min_green=parameter1, auct_sus=parameter2),
    "J29": genPriorityPassController(tau=tau, min_green=parameter1, auct_sus=parameter2),
    "J30": genPriorityPassController(tau=tau, min_green=parameter1, auct_sus=parameter2),
    "J31": genPriorityPassController(tau=tau, min_green=parameter1, auct_sus=parameter2),
    "J32": genPriorityPassController(tau=tau, min_green=parameter1, auct_sus=parameter2),
    "J33": genPriorityPassController(tau=tau, min_green=parameter1, auct_sus=parameter2),
}
     



###############################################################################
############################## DEFINE SIMULATION SETTINGS
###############################################################################

# Waiting Time To See
wait = 0.0 # [s]

# Run Simulation
simulator = Simulator(settings, label="demo_priority_pass") 
simulator.open_simulation()
simulator.run_simulation_loop(wait=wait)
simulator.close_simulation()
results = analyse_experiment_PriorityPass(settings, simulator.recorder)
print(results)

