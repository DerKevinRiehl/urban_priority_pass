###############################################################################
### Author: Kevin Riehl <kriehl@ethz.ch>
### Date: 01.12.2024
### Organization: ETH Zürich, Institute for Transport Planning and Systems (IVT)
### Project: Urban Priority Pass - Fair Intersection Management
###############################################################################
### This file contains the simulation class, which maintains the connection to
### SUMO traffic simulation environment, controller, recorder, and spawns vehicles.
###############################################################################




###############################################################################
############################## IMPORTS
###############################################################################
import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci
import numpy as np
import time

from Recorder import Recorder
from Controller import Controller




###############################################################################
############################## SIMULATION CLASS
###############################################################################
class Simulator:
    def __init__(self, settings, label):
        self.settings = settings
        self.label = label
        
    def open_simulation(self):
        sumoCmd = [self.settings.sumo_location, "-c", self.settings.sumo_config_file, "--start", "--quit-on-end", "--time-to-teleport", "-1"]
        traci.start(sumoCmd, label=self.label)
        self.connection = traci.getConnection(self.label)
        self._crawl_batch_initial_traci()
        self.recorder = Recorder(self.settings, self.connection, self)
        self.controllers = self._setup_controllers()
        
    def _setup_controllers(self):
        controllers = {}
        for tl_id in self.settings.tl_control:
            controllers[tl_id] = Controller(tl_id, self.settings, self.connection, self.recorder, self)
        return controllers
    
    def run_simulation_loop(self, wait=0.0):
        while not self._criterion_to_abort():
            self.run_simulation_step(wait)
                
    def _crawl_batch_initial_traci(self):
        self.lanes = self.connection.lane.getIDList()
        self.lane_lengths = {}
        for lane in self.lanes:
            self.lane_lengths[lane] = self.connection.lane.getLength(lane)
        self.edges = self.connection.edge.getIDList()
                
    def _crawl_batch_information_traci(self):
        self.time = self.connection.simulation.getTime()
        self.vehicle_ids = self.connection.vehicle.getIDList()
        self.vehicle_lanes = {}
        self.vehicle_edges = {}
        for v_id in self.vehicle_ids:
            lane_id = self.connection.vehicle.getLaneID(v_id)
            self.vehicle_lanes[v_id] = lane_id
            self.vehicle_edges[v_id] = "_".join(lane_id.split("_")[:-1])
        self.vehicle_lane_positions = {}
        for v_id in self.vehicle_ids:
            self.vehicle_lane_positions[v_id] = self.connection.vehicle.getLanePosition(v_id)
            
    def run_simulation_step(self, wait=0.0):
        self._spawn_vehicles()
        self._crawl_batch_information_traci()
        for controller in self.controllers:
            self.controllers[controller].execute()
        self.recorder.record()
        self.connection.simulationStep()
        time.sleep(wait)
        
    def close_simulation(self):
        self.connection.close()
    
    def _criterion_to_abort(self):
        return (self.connection.simulation.getTime() > self.settings.spawn_horizon) and (self.connection.vehicle.getIDCount() == 0) or (self.connection.simulation.getTime() > self.settings.recording_settings["recording_interval"][1])
        
    def _spawn_vehicles(self):
        if self.connection.simulation.getTime() < self.settings.spawn_horizon:
            # For each entrance
            for entrance in self.settings.spawn_entrances_probabilities:
                # Rand Variable: Exponential Distribution with Lambda=5 (so it is between 0 and 1.0 almost)
                rand_exp_variable = min(1, np.random.exponential(1/5))
                t = self.settings.spawn_entrances_probabilities[entrance]
                if rand_exp_variable <= -np.log(1-t)/5: # CDF of exp. variable CDF(x) = 1-e^(-lambda*x)
                    # Choose Route
                    available_routes = list(self.settings.spawn_entrances_routes_probabilities[entrance].keys())
                    route_probabilities = list(self.settings.spawn_entrances_routes_probabilities[entrance].values())
                    route_probabilities = [p/sum(route_probabilities) for p in route_probabilities]
                    route = np.random.choice(a=available_routes, p=route_probabilities)
                    # Choose VOT
                    vot = np.random.choice(a=list(self.settings.vot_spawn_probabilities.keys()), 
                                           p=list(self.settings.vot_spawn_probabilities.values()))
                    # Choose UPP
                    if np.random.random() < self.settings.vot_upp_spawn_probabilities[vot]:
                        upp = 1
                    else:
                        upp = 0
                    # Spawn Car
                    v_id = self.recorder.vehicles.add_vehicle(route, vot, upp)
                    self.connection.vehicle.add(v_id, route)
                    if upp==1:
                        self.connection.vehicle.setColor(v_id, color=self.settings.color_upp)
                    else:
                        self.connection.vehicle.setColor(v_id, color=self.settings.color_npp)
                            