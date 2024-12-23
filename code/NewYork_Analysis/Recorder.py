###############################################################################
### Author: Kevin Riehl <kriehl@ethz.ch>
### Date: 01.12.2024
### Organization: ETH Zürich, Institute for Transport Planning and Systems (IVT)
### Project: Urban Priority Pass - Fair Intersection Management
###############################################################################
### This file contains the recorder class, which records information during the
### runtime of the simulation for analysis and control purposes.
###############################################################################




###############################################################################
############################## IMPORTS
###############################################################################
import numpy as np




###############################################################################
############################## RECORDER, VehicleInfos, PhaseInfos Class
###############################################################################
class VehicleInfos:
    def __init__(self):
        self.route = {}
        self.vot_field = {}
        self.upp_entitlement = {}
        self.started = {}
        self.completed = {}
        self.travel_time = {}
        self.num_intersections_passed = {}
        
    def add_vehicle(self, route, vot_field, upp_entitlement):
        v_id = "v_"+str(len(self.route))+"_"+str(upp_entitlement)
        self.route[v_id] = route
        self.vot_field[v_id] = vot_field
        self.upp_entitlement[v_id] = upp_entitlement
        self.started[v_id] = -1
        self.completed[v_id] = False
        self.travel_time[v_id] = -1
        self.num_intersections_passed[v_id] = 0
        return v_id
    
class PhaseInfos:
    def __init__(self):
        self.vehicles_passed_per_phase = {}
        self.waiting_time = []
        self.queue_length = []
        
class Recorder:
    def __init__(self, settings, connection, simulator):
        self.settings = settings
        self.connection = connection
        self.simulator = simulator
        self.vehicles = VehicleInfos()
        self.times = []
        self.emissions_CO2 = []
        # self.emissions_CO_ = []
        # self.emissions_HC_ = []
        # self.emissions_NOX = []
        # self.emissions_PMX = []
        self.emissions_NOI = []
        self.phases = {}
        for tl_id in settings.tl_control:
            self.phases[tl_id] = PhaseInfos()
    
    def _criterion_to_record(self):
        cond1 = (self.simulator.time >= self.settings.recording_settings["recording_interval"][0]) 
        cond2 = (self.simulator.time <= self.settings.recording_settings["recording_interval"][1])
        return cond1 and cond2

    def record(self):
        if self._criterion_to_record():
            self.times.append(self.simulator.time)
            for tl_id in self.settings.tl_control:
                self._record_phase_information(tl_id)
            self._record_vehicle_information()
            if self.settings.recording_settings["emissions"]:
                self._record_emission_information()
                
    def _record_emission_information(self):
        self.emissions_CO2.append(self._record_emission_information_metric(self.simulator.connection.edge.getCO2Emission))
        # self.emissions_CO_.append(self._record_emission_information_metric(self.simulator.connection.edge.getCOEmission))
        # self.emissions_HC_.append(self._record_emission_information_metric(self.simulator.connection.edge.getHCEmission))
        # self.emissions_NOX.append(self._record_emission_information_metric(self.simulator.connection.edge.getNOxEmission))
        # self.emissions_PMX.append(self._record_emission_information_metric(self.simulator.connection.edge.getPMxEmission))
        self.emissions_NOI.append(self._record_emission_information_metric(self.simulator.connection.edge.getNoiseEmission))
        
    def _record_emission_information_metric(self, func):
        sum_emission = 0
        for edge in self.simulator.edges:
            sum_emission += func(edge)
        return sum_emission
    
    def _record_vehicle_information(self):
        cond_record_travel_time = self.settings.recording_settings["vehicle_travel_time"]
        currentTime = self.simulator.time
        if cond_record_travel_time:
            for v_id in self.simulator.vehicle_ids:
                edge = self.simulator.vehicle_edges[v_id]
                route = self.vehicles.route[v_id]
                # Record Started
                if edge == self.settings.route_recording_start_edge[route] and self.vehicles.started[v_id]==-1:
                    self.vehicles.started[v_id] = currentTime
                # Record Travel Time
                if self.settings.recording_settings["vehicle_travel_time"]:
                    if not self.vehicles.started[v_id]==-1:
                        self.vehicles.travel_time[v_id] = currentTime - self.vehicles.started[v_id]
                # Record Completed
                if edge == self.settings.route_recording_completion_edge[route]:
                    if not self.vehicles.started[v_id]==-1:
                        if self.vehicles.travel_time[v_id] >= self.settings.route_min_possible_travel_time[route]: 
                            # sumo sometimes teleports vehicles therefore we 
                            # need to check for that as well and exclude otherwise
                            self.vehicles.completed[v_id] = True
                        
    def _record_phase_information(self, tl_id):
        if self.settings.recording_settings["phase_wait_time"]:
            self.phases[tl_id].waiting_time.append(self._get_phase_waiting_time(self.settings.phase_bidder_lanes[tl_id]))
        if self.settings.recording_settings["phase_queue_length"]:
            self.phases[tl_id].queue_length.append(self._get_phase_queue_lengths(self.settings.phase_bidder_lanes[tl_id]))
        if self.settings.recording_settings["phase_throughput"]:
            self._record_phase_throughput(tl_id)
        
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
                result[phases[lane]] += 1
        return result
    
    def _get_phase_waiting_time(self, phase_lanes):
        result = []
        for phase in phase_lanes:
            waitingTime = 0
            for lane in phase_lanes[phase]:
                waitingTime += self.connection.lane.getWaitingTime(lane)
            result.append(waitingTime)
        return result
    
    def _record_phase_throughput(self, tl_id):
        phases_lane_dict = {}
        for phase in self.settings.phase_leaver_lanes[tl_id]:
            for lane in self.settings.phase_leaver_lanes[tl_id][phase]:
                phases_lane_dict[lane] = phase
            if phase not in self.phases[tl_id].vehicles_passed_per_phase:
                self.phases[tl_id].vehicles_passed_per_phase[phase] = []
        for v_id in self.simulator.vehicle_ids:
            lane = self.simulator.vehicle_lanes[v_id]
            if lane in phases_lane_dict:
                lane_phase = phases_lane_dict[lane]
                if v_id not in self.phases[tl_id].vehicles_passed_per_phase[lane_phase]:
                    self.phases[tl_id].vehicles_passed_per_phase[lane_phase].append(v_id)
                    self.vehicles.num_intersections_passed[v_id] += 1

    # Vehicle Statistics That Completed their journey
    def getVehiclePopulationCompleted(self):
        lst_v_id = []
        for v_id in self.vehicles.completed:
            if self.vehicles.completed[v_id]:
                lst_v_id.append(v_id)
        return lst_v_id
    
    def getVehiclePopulationRoute(self, lst_v_id):
        routes = []
        for v_id in lst_v_id:
            routes.append(self.vehicles.route[v_id])
        return routes
    
    def getVehiclePopulationVOT(self, lst_v_id):
        vots = []
        for v_id in lst_v_id:
            vots.append(self.vehicles.vot_field[v_id])
        return vots
    
    def getVehiclePopulationUPP(self, lst_v_id):
        vots = []
        for v_id in lst_v_id:
            vots.append(self.vehicles.upp_entitlement[v_id])
        return vots
    
    def getVehiclePopulationTravelTime(self, lst_v_id, vot=False):
        times = []
        for v_id in lst_v_id:
            time = self.vehicles.travel_time[v_id]
            if vot:
                time = time * self.vehicles.vot_field[v_id]
            times.append(time)
        return times
    
    def getVehiclePopulationDelayTime(self, lst_v_id, vot=False):
        delays = []
        for v_id in lst_v_id:
            route = self.vehicles.route[v_id]
            min_possible_time = self.settings.route_min_possible_travel_time[route]
            delay_time = self.vehicles.travel_time[v_id] - min_possible_time
            if vot:
                delay_time = delay_time * self.vehicles.vot_field[v_id]
            delays.append(delay_time)
        return delays
    
    def getVehiclePopulationIntersections(self, lst_v_id):
        intersections = []
        for v_id in lst_v_id:
            intersections.append(self.vehicles.num_intersections_passed[v_id])
        return intersections
    
    def getIntersectionVehicleThroughput(self, tl_id):
        """
        This function returns the number of vehicles that passed the intersection
        during the recording interval in veh/h.

        Parameters
        ----------
        tl_id : str
            The intersection (traffic light) id.

        Returns
        -------
        intersection_throughput: float
            intersection throughput in veh/h.

        """
        vehicles = 0
        for phase in self.phases[tl_id].vehicles_passed_per_phase:
            vehicles += len(self.phases[tl_id].vehicles_passed_per_phase[phase])
        duration = (self.settings.recording_settings["recording_interval"][1]- self.settings.recording_settings["recording_interval"][0]) / (60 * 60)
        return vehicles/duration    

    def getIntersectionAverageQueueLength(self, tl_id):
        lengths = np.asarray(self.phases[tl_id].queue_length)
        return np.average(np.sum(lengths, axis=1))
    
    def getIntersectionAverageWaitingTime(self, tl_id):
        lengths = np.asarray(self.phases[tl_id].waiting_time)
        return np.average(np.sum(lengths, axis=1))