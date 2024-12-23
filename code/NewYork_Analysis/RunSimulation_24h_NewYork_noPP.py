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

from SimulationTools import genPriorityPassController, generateSpawnEntranceProbabilities
from Settings import Settings
from Simulator import Simulator
import json
import os


###############################################################################
############################## DEFINE SIMULATION SETTINGS
###############################################################################

settings = Settings()

# SIMULATION MODEL SPECIFIC
simulation_model_path = "../../models/Manhattan3x3/"

# SUMO SPECIFIC
# settings.sumo_location = "sumo" # LINUX
settings.sumo_location = "C:/Users/kriehl/AppData/Local/sumo-1.19.0/bin/sumo-gui.exe"#"sumo.exe"#"sumo-gui.exe"
settings.sumo_config_file = "Configuration.sumocfg"

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

# TIME_SPECIFIC
settings.spawn_horizon = 86400 

# RECORDING SPECIFIC
settings.recording_settings = {
    "recording_interval": [86400,86400], # after warmup time
    "phase_wait_time": False, 
    "phase_queue_length": False,
    "emissions": False,
    "phase_throughput": False,
    "vehicle_travel_time": False
}

# Demand Specific

def generateTrafficDemandNewYork():
    xlabels = ["00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00", "08:00", "09:00", "10:00", 
                "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"]
    yvalues = [400, 420, 440, 430, 460, 530, 650, # 06:00
                920, 1400, 1250, 1200, 1120, 1100, # 12:00
                1000, 1050, 1200, 1450, 1500, 1400, # 18:00
                1320, 1250, 1100, 600, 500]
    
    max_flow = max(yvalues)    
    max_inflow_per_lane = 300 # 0 - 320
    n_intersections = 12 # 12 (3x4)
    scale_factor = max_flow/n_intersections/max_inflow_per_lane
    
    yvalues2 = [y/n_intersections/scale_factor for y in yvalues]

    return xlabels, yvalues2
times, flows = generateTrafficDemandNewYork()

# CONTROL SPECIFIC
def getOptimalParameter(flow):
    if flow<100:
        return 3.0, 3.0, 0.60, 0.30
    elif flow<150:
        return 3.0, 5.0, 0.90, 0.40
    elif flow<200:
        return 4.0, 6.0, 0.80, 0.30
    elif flow<250:
        return 3.0, 4.0, 0.90, 0.30
    elif flow<300:
        return 5.0, 5.0, 0.80, 0.20
    elif flow<350:
        return 6.0, 5.0, 0.70, 0.10
    elif flow<400:
        return 8.0, 9.0, 0.60, 0.05
    elif flow<450:
        return 10.0, 9.0, 0.0, 0.00
    elif flow<500:
        return 2.0, 40.0, 0.0, 0.00
    elif flow<550:
        return 1.0, 39.0, 0.0, 0.00
    else:
        return 2.0, 38.0, 0.0, 0.00
    
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
    
    # flow,  tau,   gamma
    # ---------------------
    #  50,   0.60,  0.30
    # 100,   0.90,  0.40
    # 150    0.80,  0.30
    # 200,   0.90,  0.30
    # 250,   0.80,  0.20
    # 300,   0.70,  0.10
    

###############################################################################
############################## DEFINE SIMULATION SETTINGS
###############################################################################

# Init Demand
flow = flows[0]

# Init Controller & Demand
parameter1, parameter2, tau, gamma = getOptimalParameter(flow)
tau = 0
gamma = 0
settings.spawn_entrances_probabilities = generateSpawnEntranceProbabilities(settings.spawn_entrances_probabilities, flow)
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

# Run Simulation
simulator = Simulator(settings, label="demo_priority_pass") 
simulator.open_simulation()

for it in range(0, len(times)):
    # update settings
    # UPDATE DEMAND
    flow = flows[it] 
    settings.spawn_entrances_probabilities = generateSpawnEntranceProbabilities(settings.spawn_entrances_probabilities, flow)
    # UPDATE CONTROL
    parameter1, parameter2, tau, gamma = getOptimalParameter(flow)
    tau = 0.0
    gamma = 0.0
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
    simulator.controllers = simulator._setup_controllers()
    print(times[it], flow, parameter1, parameter2, tau, gamma)
    # UPDATE ENTITLEMENT
    settings.vot_spawn_probabilities = {0: 1-gamma, 1: gamma}
    settings.vot_upp_spawn_probabilities = {0: 0.0, 1: 1.0}
    
    # run for 1h with the settings    
    for step in range(0,3600):
        simulator.run_simulation_step()
    
# Finish Simulation
simulator.close_simulation()
