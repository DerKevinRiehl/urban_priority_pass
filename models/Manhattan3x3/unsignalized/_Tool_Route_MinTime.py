# Signalized Intersection With Auction Control, Copyrights Kevin Riehl 2023, <kriehl@ethz.ch>




# Imports
import traci
    
    
# SETUP Settings
    # SUMO SPECIFIC
sumo_location = "C:/Users/kriehl/AppData/Local/sumo-1.19.0/bin/sumo.exe"#"sumo.exe"#"sumo-gui.exe"
sumo_config_file = "Configuration_nolight.sumocfg"
# settings.sumo_config_file = "Maps/zurich_network_relevant/Configuration_nolight.sumocfg"



# Get all Routes
sumoCmd = [sumo_location, "-c", sumo_config_file, "--start", "--quit-on-end", "--time-to-teleport", "-1"]
traci.start(sumoCmd, label="test")
routes = traci.route.getIDList()
lanes = traci.lane.getIDList()
traci.close()



# Measure Time for Each Route
durations = {}
for route in routes:      
    traci.start(sumoCmd, label="test")
    traci.vehicle.add("v_id", route)
    traci.simulationStep()
    vehicles = traci.vehicle.getIDList()
    while len(vehicles)>0:
        # time.sleep(0.1)
        traci.simulationStep()
        vehicles = traci.vehicle.getIDList()
    route_duration = traci.simulation.getTime()
    traci.close()
    print(route, route_duration)
    durations[route] = route_duration
