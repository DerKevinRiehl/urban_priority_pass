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
settings.spawn_horizon = 30000 

# RECORDING SPECIFIC
settings.recording_settings = {
    "recording_interval": [30000,30000], # after warmup time
    "phase_wait_time": False, 
    "phase_queue_length": False,
    "emissions": False,
    "phase_throughput": False,
    "vehicle_travel_time": False
}

# CONTROL SPECIFIC
parameter1 = 29
parameter2 = 21
tau = 0.60
gamma = 0.30
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
    
    # 10  550.0         2.0        38.0

###############################################################################
############################## DEFINE SIMULATION SETTINGS
###############################################################################

INIT_FLOW = 50
FACTOR = 1.086357 # (600/50)^(1/30), so that flow 50 grows to 600 after 30,000 seconds


traffic_lights = ["J25", "J26", "J27", "J28", "J29", "J30", "J31", "J32", "J33"]

traffic_flows = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
traffic_flows = [50, 100, 150, 200, 250, 300, ]
for flow in traffic_flows:
    fLogger = open("logs/log_tls_states_priority_pass_"+str(flow)+".xml", "w+")
    for tl in traffic_lights:
        fLogger.write(tl)
        fLogger.write("\t")
    fLogger.write("\n")
        
    # Init Demand
    flow = flow
    print(flow)
    # Init Controller
    parameter1, parameter2, tau, gamma = getOptimalParameter(flow)
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
    for step in range(0,5000):
        # Run step
        simulator.run_simulation_step()
        
        # Log Traffic Lights
        for tl in traffic_lights:
            tl_signal = simulator.connection.trafficlight.getRedYellowGreenState(tl)
            fLogger.write(tl_signal)
            fLogger.write("\t")
        fLogger.write("\n")
        
    # Finish Simulation
    simulator.close_simulation()
    fLogger.close()