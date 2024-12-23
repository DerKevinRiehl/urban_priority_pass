###############################################################################
### Author: Kevin Riehl <kriehl@ethz.ch>
### Date: 01.12.2024
### Organization: ETH Zürich, Institute for Transport Planning and Systems (IVT)
### Project: Urban Priority Pass - Fair Intersection Management
###############################################################################
### This file contains the controller class, which controlls the traffic lights
### based on three possible algorithm types: fixed_programme, max_pressure, 
### and priority_pass.
###############################################################################




###############################################################################
############################## IMPORTS
###############################################################################
import numpy as np




###############################################################################
############################## CONSTANTS
###############################################################################
STATE_READY_FOR_AUCTION = 0
STATE_CHANGING_SIGNAL = 1
STATE_WAIT_MIN_GREEN_TIME = 2
STATE_WAIT_FOR_NEXT_AUCTION = 3

TYPE_FIXED_PROGRAMME = "fixed_programme"
TYPE_MAX_PRESSURE = "max_pressure"
TYPE_PRIORITY_PASS = "priority_pass"

BIDDING_STRATEGY_PHASE_QUEUE_LENGTH = "phase_queue_length"
BIDDING_STRATEGY_PHASE_WEIGHTED_VEHICLE_POSITION = "phase_weigthed_vehicle_position"

AUCTION_WINNER_HIGHEST_BID = "highest_bid"




###############################################################################
############################## CONTROLLER CLASS
###############################################################################
class Controller:
    def __init__(self, tl_id, settings, connection, recorder, simulator):
        self.settings = settings.tl_control[tl_id]
        self.sensor_distance = settings.sensor["max_distance_from_intersection"]
        self.connection = connection
        self.recorder = recorder
        self.simulator = simulator
        
        self.type = settings.tl_control[tl_id]["type"]
        self.tl_id = tl_id
        if settings.tl_control[tl_id]["type"] == TYPE_FIXED_PROGRAMME:
            self.timer = self.settings["time_delay"]
            self._set_tl_fixed_programme()
        elif settings.tl_control[tl_id]["type"] == TYPE_MAX_PRESSURE or TYPE_PRIORITY_PASS:            
            self.phase_bidder_lanes = settings.phase_bidder_lanes[tl_id]
            self.state = STATE_READY_FOR_AUCTION
            self.number_phases = len(self.connection.trafficlight.getAllProgramLogics(tl_id)[0].phases)
            self.phase = 0
            self.phase_signal = 0
            self.timer = 0              
            self.current_phase_timer = 0
            
    def _set_tl_fixed_programme(self):      
        ft_logic = self.connection.trafficlight.getAllProgramLogics(self.tl_id)[0]
        counter = 0
        for ph_id in range(0,len(ft_logic.phases)):
            if ph_id % 2 == 0:
                # actual phase
                ft_logic.phases[ph_id].minDur = self.settings["phase_durations"][counter]
                ft_logic.phases[ph_id].maxDur = self.settings["phase_durations"][counter]
                ft_logic.phases[ph_id].duration = self.settings["phase_durations"][counter]
                # yellow transition
                ft_logic.phases[ph_id+1].minDur = self.settings["transition_duration"]
                ft_logic.phases[ph_id+1].maxDur = self.settings["transition_duration"]
                ft_logic.phases[ph_id+1].duration = self.settings["transition_duration"]
                counter += 1
        self.connection.trafficlight.setProgramLogic(self.tl_id, ft_logic)
        self.connection.trafficlight.setPhase(self.tl_id, 0)
        
    def _get_phase_queue_lengths(self, phase_lanes):
        result = []
        phases = {}
        for phase in phase_lanes:
            for lane in phase_lanes[phase]:
                phases[lane] = int(phase)
            result.append(0)
        for v_id in self.simulator.vehicle_ids:
            lane = self.simulator.vehicle_lanes[v_id]
            if lane in phases:
                vehicle_position = self.simulator.vehicle_lane_positions[v_id]
                lane_length = self.simulator.lane_lengths[lane] 
                vehicle_distance_to_intersection = lane_length - vehicle_position
                if vehicle_distance_to_intersection <= self.sensor_distance:
                    result[phases[lane]] += 1
        return result
    
    def _get_phase_weighted_vehicle_positions(self, phase_lanes):
        weights = self.settings["position_weights"]
        distances = self.settings["position_distance_to_intersection"]
        result = []
        phases = {}
        for phase in phase_lanes:
            for lane in phase_lanes[phase]:
                phases[lane] = int(phase)
            result.append(0)
        for v_id in self.simulator.vehicle_ids:
            lane = self.simulator.vehicle_lanes[v_id]
            if lane in phases:
                vehicle_position = self.simulator.vehicle_lane_positions[v_id]
                lane_length = self.simulator.lane_lengths[lane] 
                vehicle_distance_to_intersection = lane_length - vehicle_position
                if vehicle_distance_to_intersection <= self.sensor_distance:
                    fdistances = distances.copy()
                    if vehicle_distance_to_intersection>max(fdistances):
                        fdistances.append(vehicle_distance_to_intersection+1)
                    vehicle_distance_class = next(idx for idx in range(0, len(fdistances)) if fdistances[idx]>vehicle_distance_to_intersection)
                    result[phases[lane]] += 1*weights[vehicle_distance_class]
        return result
    
    def _get_phase_bids(self):
        if self.settings["bidding_strategy"]==BIDDING_STRATEGY_PHASE_QUEUE_LENGTH:
            bids = self._get_phase_queue_lengths(self.phase_bidder_lanes)
        elif self.settings["bidding_strategy"]==BIDDING_STRATEGY_PHASE_WEIGHTED_VEHICLE_POSITION:
            bids = self._get_phase_weighted_vehicle_positions(self.phase_bidder_lanes)
        return bids
    
    def _get_phase_upp_bids(self):
        result = []
        phases = {}
        for phase in self.phase_bidder_lanes:
            for lane in self.phase_bidder_lanes[phase]:
                phases[lane] = int(phase)
            result.append(0)
        for v_id in self.simulator.vehicle_ids:
            lane = self.simulator.vehicle_lanes[v_id]
            if lane in phases:
                vehicle_position = self.simulator.vehicle_lane_positions[v_id]
                lane_length = self.simulator.lane_lengths[lane] 
                vehicle_distance_to_intersection = lane_length - vehicle_position
                if vehicle_distance_to_intersection <= self.sensor_distance:
                    result[phases[lane]] += self.recorder.vehicles.upp_entitlement[v_id]
        return result
    
    def _determine_auction_winner_phase(self, bids):
        if self.settings["auction_winner"]==AUCTION_WINNER_HIGHEST_BID:
            # highest bid wins, tie braker: random choice
            bids = np.asarray(bids)
            return np.random.choice(np.flatnonzero(bids == bids.max()))
        return -1
    
    def execute(self):
        if self.type == TYPE_FIXED_PROGRAMME:
            self._phase_fixed_programme_logic()
        elif self.type == TYPE_MAX_PRESSURE:
            self._phase_auction_logic(upp=False)
        elif self.type == TYPE_PRIORITY_PASS:
            self._phase_auction_logic(upp=True)
            
    def _phase_fixed_programme_logic(self):
        if self.timer > 0:
            self.timer -= 1
        else:
            if self.timer==0:
                self._set_tl_fixed_programme()
                self.timer = -1
            
    def _phase_auction_logic(self, upp=False):
        if self.state == STATE_READY_FOR_AUCTION:
            self.current_phase_timer += 1
            phase_bids = self._get_phase_bids()
            if upp:
                tau = self.settings["trade_off"]
                upp_bids = self._get_phase_upp_bids()
                bids = [(1-tau)*phase_bids[x] + tau*upp_bids[x] for x in range(0, len(phase_bids))]
            else:
                bids = phase_bids
            if self.current_phase_timer > self.settings["max_green_duration"]:
                bids[self.phase] = -10000
            winner_phase = self._determine_auction_winner_phase(bids)
            if winner_phase == self.phase: # same phase continues
                self.state = STATE_WAIT_FOR_NEXT_AUCTION
                self.timer = self.settings["auction_suspend_duration"] - 1
            else: # first transition, then next phase
                self.phase = winner_phase
                self.phase_signal = self.phase_signal+1
                self.state = STATE_CHANGING_SIGNAL
                self.timer = self.settings["transition_duration"] - 1
        elif self.state == STATE_CHANGING_SIGNAL:
            if self.timer > 0:
                self.timer -= 1
            else:
                self.phase_signal = self.phase*2
                self.state = STATE_WAIT_MIN_GREEN_TIME
                self.timer = self.settings["min_green_duration"] - 1                
                self.current_phase_timer = 0
        elif self.state == STATE_WAIT_MIN_GREEN_TIME:
            self.current_phase_timer += 1
            if self.timer > 0:
                self.timer -= 1
            else:
                self.state = STATE_READY_FOR_AUCTION               
        elif self.state == STATE_WAIT_FOR_NEXT_AUCTION:
            self.current_phase_timer += 1
            if self.timer > 0:
                self.timer -= 1
            else:
                self.state = STATE_READY_FOR_AUCTION
        self.connection.trafficlight.setPhase(self.tl_id, self.phase_signal)
        