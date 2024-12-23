###############################################################################
### Author: Kevin Riehl <kriehl@ethz.ch>
### Date: 01.12.2024
### Organization: ETH Zürich, Institute for Transport Planning and Systems (IVT)
### Project: Urban Priority Pass - Fair Intersection Management
###############################################################################
### This file demonstrates (visually) how a fixed-cycle traffic light programme
### can run using our simulation environment.
###############################################################################




###############################################################################
############################## IMPORTS
###############################################################################

from SimulationTools import genFixProgrammeController, analyse_experiment_delays, generateSpawnEntranceProbabilities
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





###############################################################################
############################## SIMULATION WITH 200
###############################################################################

# DEMAND SPECIFIC
flow = 200 # [veh/h]
settings.spawn_entrances_probabilities = generateSpawnEntranceProbabilities(settings.spawn_entrances_probabilities, flow)

# CONTROL SPECIFIC
parameter1 = 29
parameter2 = 21
phase_durations = [parameter1, parameter2, parameter1, parameter2]
settings.tl_control = {
    "J25": genFixProgrammeController(phase_durations, time_delay=0),
    "J26": genFixProgrammeController(phase_durations, time_delay=int((parameter1+parameter2)/2) ),
    "J27": genFixProgrammeController(phase_durations, time_delay=0),
    "J28": genFixProgrammeController(phase_durations, time_delay=int((parameter1+parameter2)/2) ),
    "J29": genFixProgrammeController(phase_durations, time_delay=0),
    "J30": genFixProgrammeController(phase_durations, time_delay=0),
    "J31": genFixProgrammeController(phase_durations, time_delay=int((parameter1+parameter2)/2) ),
    "J32": genFixProgrammeController(phase_durations, time_delay=int((parameter1+parameter2)/2) ),
    "J33": genFixProgrammeController(phase_durations, time_delay=0)
}

# Run Simulation
simulator = Simulator(settings, label="demo_fixed_programme") 
simulator.open_simulation()

ctr = 0
res = []
while not simulator._criterion_to_abort():
    simulator.run_simulation_step(0)
    ctr += 1
    if ctr%10==0:
        a,b,c = analyse_experiment_delays(settings, simulator.recorder)
        res.append([ctr, simulator.connection.simulation.getTime(), a, b, c])

simulator.close_simulation()

import numpy as np
res = np.asarray(res)
data_200 = res.copy()


###############################################################################
############################## SIMULATION WITH 400
###############################################################################

# DEMAND SPECIFIC
flow = 400 # [veh/h]
settings.spawn_entrances_probabilities = generateSpawnEntranceProbabilities(settings.spawn_entrances_probabilities, flow)

# CONTROL SPECIFIC
parameter1 = 9
parameter2 = 4
phase_durations = [parameter1, parameter2, parameter1, parameter2]
settings.tl_control = {
    "J25": genFixProgrammeController(phase_durations, time_delay=0),
    "J26": genFixProgrammeController(phase_durations, time_delay=int((parameter1+parameter2)/2) ),
    "J27": genFixProgrammeController(phase_durations, time_delay=0),
    "J28": genFixProgrammeController(phase_durations, time_delay=int((parameter1+parameter2)/2) ),
    "J29": genFixProgrammeController(phase_durations, time_delay=0),
    "J30": genFixProgrammeController(phase_durations, time_delay=0),
    "J31": genFixProgrammeController(phase_durations, time_delay=int((parameter1+parameter2)/2) ),
    "J32": genFixProgrammeController(phase_durations, time_delay=int((parameter1+parameter2)/2) ),
    "J33": genFixProgrammeController(phase_durations, time_delay=0)
}

# Run Simulation
simulator = Simulator(settings, label="demo_fixed_programme") 
simulator.open_simulation()

ctr = 0
res = []
while not simulator._criterion_to_abort():
    simulator.run_simulation_step(0)
    ctr += 1
    if ctr%10==0:
        a,b,c = analyse_experiment_delays(settings, simulator.recorder)
        res.append([ctr, simulator.connection.simulation.getTime(), a, b, c])

simulator.close_simulation()

import numpy as np
res = np.asarray(res)
data_400 = res.copy()

import matplotlib.pyplot as plt
plt.plot(data_200[:, 1], data_200[:, 2], label="200 veh/h")
plt.plot(data_400[:, 1], data_400[:, 2], label="400 veh/h")
plt.legend()