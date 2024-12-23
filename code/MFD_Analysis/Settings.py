###############################################################################
### Author: Kevin Riehl <kriehl@ethz.ch>
### Date: 01.12.2024
### Organization: ETH Zürich, Institute for Transport Planning and Systems (IVT)
### Project: Urban Priority Pass - Fair Intersection Management
###############################################################################
### This file contains the settings class and stores the information about
### the simulation's settings, including different commented-out examples for controllers.
###############################################################################




###############################################################################
############################## IMPORTS
###############################################################################
import json




###############################################################################
############################## SETTINGS CLASS 
###############################################################################

class Settings:
    def __init__(self):
        
        # SUMO SPECIFIC
        self.sumo_location = "C:/Users/kriehl/AppData/Local/sumo-1.19.0/bin/sumo.exe"#"sumo.exe"#"sumo-gui.exe"
        self.sumo_config_file = "IntersectionA/Configuration.sumocfg"
        
        # RANDOM SPECIFIC
        self.random_seed = 42
        
        # TIME_SPECIFIC
        self.spawn_horizon = 600+3600*1 # 1h Spawning + 10mins warmup
        
        # RECORDING SPECIFIC
        self.recording_settings = {
            "recording_interval": [600,600+3600*1], # after warmup time
            "phase_wait_time": True, 
            "phase_queue_length": True,
            "emissions": False,
            "phase_throughput": True,
            "vehicle_travel_time": True
        }
        
        # NETWORK SPECIFIC
            # route_specific
        self.spawn_entrances_probabilities = {"0": 0.1, "1": 0.1, "2": 0.1, "3": 0.1, "4": 0.1, "5": 0.1, "6": 0.1, "7": 0.1, "8": 0.1, "9": 0.1, "10": 0.1, "11": 0.1}
        self.spawn_entrances_routes_probabilities = {} # Load from File "Route_Probabilities.json"
        self.vot_spawn_probabilities = {0: 1.0}
        self.vot_upp_spawn_probabilities = {0: 0.0}
        self.route_min_possible_travel_time = {} # Load from File "Route_Durations.json"
        self.route_recording_start_edge = {} # Load from File "Route_StartEdges.json"
        self.route_recording_completion_edge = {} # Load from File "Route_EndEdges.json"
        self.route_length = {} # Load from File "Route_Distances.json"
            # intersection specific
        self.phase_bidder_lanes = {} # Load from File "Phase_BidderLanes.json"
        self.phase_leaver_lanes = {} # Load from File "Phase_ExitLanes.json"
        
        # SENSOR SPECIFIC
        self.sensor = {
            "max_distance_from_intersection": 100,
        }
        
        # VISUALIZATION SPECIFIC
        self.color_upp = (30, 111, 192, 255)
        self.color_npp = (110, 110, 110, 255)

        # CONTROL SPECIFIC
        self.tl_control = {
            # Example Fixed Programme
            # "J9": {
            #     "type": "fixed_programme",
            #     "transition_duration": 3,
            #     "time_delay": 0,
            #     "phase_durations": [50, 50]
            # },
            
            # Example Max Pressure (Queue Lenght)
            # "J9": {
            #     "type": "max_pressure",
            #     "transition_duration": 3,
            #     "bidding_strategy": "phase_queue_length",
            #     "auction_winner": "max_pressure",
            #     "min_green_duration": 20,
            #     "max_green_duration": 120,
            #     "auction_suspend_duration": 5,
            # },
            
            # Example Max Pressure (Weighted Queue Position)
            # "J9": {
            #     "type": "max_pressure",
            #     "transition_duration": 3,
            #     "bidding_strategy": "phase_weigthed_queue_position",
            #     "position_weights": [0.2, 0.3, 0.3, 0.2],
            #     "auction_winner": "max_pressure",
            #     "min_green_duration": 20,
            #     "max_green_duration": 120,
            #     "auction_suspend_duration": 5,
            # }
            
            # Example Max Pressure (Weighted Vehicle Position)
            # "J9": {
            #     "type": "max_pressure",
            #     "transition_duration": 3,
            #     "bidding_strategy": "phase_weigthed_vehicle_position",
            #     "position_weights": [0.2, 0.3, 0.3, 0.2],
            #     "position_distance_to_intersection": [10, 20, 30],
            #     "auction_winner": "max_pressure",
            #     "min_green_duration": 20,
            #     "max_green_duration": 120,
            #     "auction_suspend_duration": 5,
            # }
            
            # Example Priority Pass (Queue Length)
            # "J9": {
            #     "type": "priority_pass",
            #     "transition_duration": 3,
            #     "bidding_strategy": "phase_queue_length",
            #     "auction_winner": "max_pressure",
            #     "min_green_duration": 20,
            #     "max_green_duration": 120,
            #     "auction_suspend_duration": 5,
            #     "trade_off": 0.1,
            # },
            
            # Example Priority Pass (Weighted Vehicle Position)
            # "J9": {
            #     "type": "priority_pass",
            #     "transition_duration": 3,
            #     "bidding_strategy": "phase_weigthed_vehicle_position",
            #     "position_weights": [0.2, 0.3, 0.3, 0.2],
            #     "position_distance_to_intersection": [10, 20, 30],
            #     "auction_winner": "max_pressure",
            #     "min_green_duration": 20,
            #     "max_green_duration": 120,
            #     "auction_suspend_duration": 5,
            # }
        }

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)
