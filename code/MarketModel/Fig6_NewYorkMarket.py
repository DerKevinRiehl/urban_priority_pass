# =============================================================================
# ================   Imports   ================================================
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from Data_UrgencyDistribution import urgency_distribution
import json
import warnings
warnings.filterwarnings("ignore")



# =============================================================================
# ================   Paths & Parameter  =======================================
# =============================================================================
log_priority_pass   = "../../data/log_priority_pass.csv"
log_max_pressure = "../DataAnalysis/0_optim_benchmark_controller/optim_max_pressure_totaltraveltime.csv"

flows = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
tau_controls = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
gamma_shares = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

metrics = metrics = [
    'Total_Throuput.1', 'Total_AvQueueLength.1', 'NumCompletedVeh.1',
    'NumVehIntersectionPassages.1', 'PopTimeSpent.1', 'VehAvDelay.1', 
    'VehMdDelay.1', 'VehStDelay.1', 'Total_Throuput.2', 'Total_AvQueueLength.2', 
    'NumCompletedVeh.2', 'NumVehIntersectionPassages.2', 'PopTimeSpent.2', 
    'VehAvDelay.2', 'VehMdDelay.2', 'VehStDelay.2'
    ]

MINIMUM_SALARY_THRESHOLD = 15.0

NETWORK_LENGTH = 19477.68000000002/1000

scenario = 1
blue_color = (30/255, 111/255, 192/255)

# https://www.nyc.gov/html/dot/html/infrastructure/signals.shtml
# According to the search results, Manhattan has 2,862 intersections with traffic signals1
INTERSECTIONS = 2862
BLOCK_VALUE_FACTOR = 2862/9
INTERVAL = 300
STD_FAC = 40.0





# =============================================================================
# ================   Methods   ================================================
# =============================================================================

def process_summary(file_path, interval):
    tree = ET.parse(file_path)
    root = tree.getroot()
    data = []
    last_ended = 0
    for stepinfo in root.findall('step'):
        # depart = float(tripinfo.get('depart'))
        # distance = float(tripinfo.get('routeLength'))
        av_traveltime =  float(stepinfo.get('meanTravelTime'))
        n_cars = float(stepinfo.get('running'))
        n_ended =  float(stepinfo.get('ended'))
        # n_cars += n_ended - last_ended
        n_flow = (n_ended-last_ended)/interval*3600
        # n_cars = float(stepinfo.get("running"))
        ttt = n_cars * av_traveltime
        av_speed = float(stepinfo.get('meanSpeed'))
        time = float(stepinfo.get('time'))
        if av_speed != -1:
            data.append([time, n_cars, ttt, av_speed*3.6, n_flow])
        last_ended = n_ended
    return data

def generateTrafficDemandNewYork():
    xlabels = ["00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00", "08:00", "09:00", "10:00", 
                "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"]
    yvalues = [400, 420, 440, 430, 460, 530, 650, # 06:00
                920, 1400, 1250, 1200, 1120, 1100, # 12:00
                1000, 1050, 1200, 1450, 1500, 1400, # 18:00
                1320, 1250, 1100, 600, 500]
    max_flow = max(yvalues)    
    max_inflow_per_lane = 350 # 0 - 320
    n_intersections = 12 # 12 (3x4)
    scale_factor = max_flow/n_intersections/max_inflow_per_lane
    
    yvalues2 = [y/n_intersections/scale_factor for y in yvalues]
    
    return xlabels, yvalues, yvalues2
simulation_times, real_flows, simulation_flows = generateTrafficDemandNewYork()

def load_tables(file):
    table = pd.read_csv(file, index_col=False, skiprows=1, sep=",")
    tables = {}
    for flow in  flows:
        table_filt = table[table["Flow"]==flow]
        tables[flow] = table_filt.copy()
    return tables

def smooth_filter(matrix):
    matrix_smooth = matrix.copy()
    for x in range(0, matrix.shape[0]-1):
        for y in range(0, matrix.shape[1]-1):
            matrix_smooth[x][y] = (matrix_smooth[x-1][y] + matrix_smooth[x][y-1] + matrix_smooth[x][y] + matrix_smooth[x+1][y] + matrix_smooth[x][y+1])/5
    for x in [0]:
        for y in range(0, matrix.shape[1]-1):
            matrix_smooth[x][y] = (matrix_smooth[x][y-1] + matrix_smooth[x][y] + matrix_smooth[x+1][y] + matrix_smooth[x][y+1])/4
    for x in range(0, matrix.shape[0]-1):
        for y in [0]:
            matrix_smooth[x][y] = (matrix_smooth[x-1][y] + matrix_smooth[x][y] + matrix_smooth[x+1][y] + matrix_smooth[x][y+1])/4
    for x in [0]:
        for y in [0]:
            matrix_smooth[x][y] = (matrix_smooth[x][y] + matrix_smooth[x+1][y] + matrix_smooth[x][y+1])/3
    for x in [matrix.shape[0]-1]:
        for y in range(0, matrix.shape[1]-1):
            matrix_smooth[x][y] = (matrix_smooth[x][y-1] + matrix_smooth[x][y] + matrix_smooth[x-1][y] + matrix_smooth[x][y+1])/4
    for x in range(0, matrix.shape[0]-1):
        for y in [matrix.shape[1]-1]:
            matrix_smooth[x][y] = (matrix_smooth[x-1][y] + matrix_smooth[x][y] + matrix_smooth[x+1][y] + matrix_smooth[x][y-1])/4
    for x in [matrix.shape[0]-1]:
        for y in [matrix.shape[1]-1]:
            matrix_smooth[x][y] = (matrix_smooth[x][y] + matrix_smooth[x-1][y] + matrix_smooth[x][y-1])/3
    return matrix_smooth

def calculateDelayChanges(tables, benchmark, vot_upp, vot_npp):
    for flow in  flows:
        tables[flow]["benchmark_throughput"] = benchmark[benchmark["Flow"]==flow]["Throughput"].iloc[0]
        tables[flow]["change_throughput"] = tables[flow]["Total_Throuput"] - tables[flow]["benchmark_throughput"]
        
        tables[flow]["benchmark_queuelength"] = benchmark[benchmark["Flow"]==flow]["AvQueueLength"].iloc[0]
        tables[flow]["change_queuelength"] = tables[flow]['Total_AvQueueLength'] - tables[flow]["benchmark_queuelength"]
        
        if flow==350:
            tables[flow]["benchmark_delay"] = 65.8792-0.9
        else:
            tables[flow]["benchmark_delay"] = benchmark[benchmark["Flow"]==flow]["VehMdDelayPerInter"].iloc[0]
        
        tables[flow]["benchmark_delay_STD"] = benchmark[benchmark["Flow"]==flow]["VehStdDelayPerInter"].iloc[0]        
        tables[flow]["change_delay_avg"] = tables[flow]["VehMdDelay"] - tables[flow]["benchmark_delay"]

        tables[flow]["benchmark_TTT"] = benchmark[benchmark["Flow"]==flow]["PopTimeSpent"].iloc[0]
        tables[flow]["change_TTT"] = tables[flow]['TTT'] - tables[flow]["benchmark_TTT"]
        
        tables[flow]["change_delay_upp"] = tables[flow]["UPPMedDelay"] - tables[flow]["benchmark_delay"]
        tables[flow]["change_delay_npp"] = tables[flow]["NPPMedDelay"] - tables[flow]["benchmark_delay"]
        tables[flow]["change_delay_upp_STD"] = (tables[flow]["UPPStDelay"]+tables[flow]["benchmark_delay_STD"])/2
        tables[flow]["change_delay_npp_STD"] = (tables[flow]["NPPStDelay"]+tables[flow]["benchmark_delay_STD"])/2
        
        tables[flow]["change_delay_upp_rel"] = 100*(tables[flow]["change_delay_upp"])/tables[flow]["benchmark_delay"]
        tables[flow]["change_delay_npp_rel"] = 100*(tables[flow]["change_delay_npp"])/tables[flow]["benchmark_delay"]
        
        tables[flow]["change_delay_upp_rel_STD"] = 100*(tables[flow]["change_delay_upp_STD"])/tables[flow]["benchmark_delay"] -100
        tables[flow]["change_delay_npp_rel_STD"] = 100*(tables[flow]["change_delay_npp_STD"])/tables[flow]["benchmark_delay"] -100
              
        thr = 2.0
        tables[flow]["change_delay_upp_rel_STD"] = tables[flow]["change_delay_upp_rel_STD"].where( abs(tables[flow]["change_delay_upp_rel_STD"]) > thr, thr)
        tables[flow]["change_delay_npp_rel_STD"] = tables[flow]["change_delay_npp_rel_STD"].where( abs(tables[flow]["change_delay_npp_rel_STD"]) > thr, thr)

def smootje(data, n=2):
    data = np.array(data)  # Ensure data is a numpy array
    for _ in range(n):
        padded = np.pad(data, (1, 1), mode='edge')
        data = padded[1:-1] * 0.25 + data * 0.5 + padded[2:] * 0.25
    return data


def calculateOptimalPrice(tablesUPP, flow, taus, gammas, df_population):
    upp_relevantX = tablesUPP[flow]
    upp_relevantX = upp_relevantX[["Parameter1", "Parameter2", 
                                 'change_delay_upp', 'change_delay_npp', 'change_delay_upp_STD', 'change_delay_npp_STD',]]
    data = [] # matrix benefit cr
    data2 = [] # matrix price
    data3 = [] # matrix benefit Cr
    for tau in taus:
        rows = []
        rows2 = []
        rows3 = []
        for gamma in gammas:    
            upp_relevant = upp_relevantX[(upp_relevantX["Parameter1"]==gamma) & (upp_relevantX["Parameter2"]==tau)].copy()
            df_population["upp_exp_delay"] = (df_population["route_distance"]/1000) * upp_relevant["change_delay_upp"].iloc[0]
            df_population["npp_exp_delay"] = (df_population["route_distance"]/1000) * upp_relevant["change_delay_npp"].iloc[0]
            df_population["exp_time_save"] = df_population["npp_exp_delay"] - df_population["upp_exp_delay"]
            df_population["exp_cost_save"] = df_population["exp_time_save"] * df_population["vot"]/3600
            
            # Determine optimal price
            df_sorted = df_population.sort_values(by='exp_cost_save', ascending=False)
            df_sorted = df_sorted.reset_index()
            df_sorted = df_sorted.sort_values(by='exp_cost_save', ascending=False)
            optimal_price = df_sorted["exp_cost_save"].iloc[int(gamma*len(df_population))]
            
            df_population["buy_decision"] = df_population["exp_cost_save"] > optimal_price
            df_population['actual_delay'] = df_population['upp_exp_delay'].where(df_population['buy_decision'], df_population['npp_exp_delay'])
            df_population['actual_price'] = np.where(df_population['buy_decision'], optimal_price, 0)
            received = (gamma*len(df_population)*optimal_price) / ((1-gamma)*len(df_population))
            df_population['actual_reciv'] = np.where(df_population['buy_decision'], 0, received)
            df_population['actual_time_costs'] = -df_population['actual_delay'] * df_population["vot"]/3600 - df_population['actual_price'] + df_population['actual_reciv']
            
            benefit = np.mean(df_population['actual_time_costs'])
            
            table = tablesUPP[flow]
            throughput = table[table["Parameter1"]==gamma][table["Parameter2"]==tau]["Total_Throuput"].iloc[0]
            benefitSys = benefit * throughput * NETWORK_LENGTH

            rows.append(benefit)
            rows2.append(optimal_price)
            rows3.append(benefitSys)
        data.append(rows)
        data2.append(rows2)
        data3.append(rows3)
    data = np.asarray(data)
    data = smooth_filter(data)
    data = smooth_filter(data)
    
    data2 = np.asarray(data2)
    data2 = smooth_filter(data2)
    data2 = smooth_filter(data2)
    
    data3 = np.asarray(data3)
    data3 = smooth_filter(data3)
    data3 = smooth_filter(data3)
    return data, data2, data3

def generateParametersOverFlows(scenario):
    df_population = pd.DataFrame(np.asarray([urgency_distribution[scenario]["pop_route_distances"], 
                                  urgency_distribution[scenario]["pop_route_durations"], 
                                  urgency_distribution[scenario]["pop_salary"],
                                  urgency_distribution[scenario]["pop_urgency_levels"],
                                  urgency_distribution[scenario]["pop_urgency_vots"]]).T, 
                                 columns=["route_distance", "route_duration", "salary", "urgency_level", "vot"])
    df_population = df_population[df_population["salary"]>MINIMUM_SALARY_THRESHOLD]

    data = []
    for flow in flows:
        matrix, prices, matrixSys = calculateOptimalPrice(tablesUPP, flow, tau_controls, gamma_shares, df_population)
        max_row, max_col = np.unravel_index(np.argmax(matrixSys), matrixSys.shape)
        tau_optim = tau_controls[max_row]
        gamma_optim = gamma_shares[max_col]
        price = prices[max_row][max_col]
        benefit = matrix[max_row][max_col]
        benefitSys = matrixSys[max_row][max_col]
        
        data.append([flow, scenario, tau_optim, gamma_optim, price, benefit, benefitSys])
    return np.asarray(data)

def process_tripinfos(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    data = []
    for tripinfo in root.findall('tripinfo'):    
        startTime = float(tripinfo.get("depart"))
        endTime = float(tripinfo.get("arrival"))
        duration = float(tripinfo.get("duration"))
        priority_pass = tripinfo.get("id").split("_")[-1]
        route = tripinfo.get("departLane").split("_")[0].replace("-", "").replace("E","") + "_" + tripinfo.get("arrivalLane").split("_")[0].replace("-", "").replace("E","")
        routeDuration = route_durations[route]
        routeDistance = route_distances[route]/1000
        delay = duration-routeDuration
        if delay<0:
            delay = 0
        delayPD = delay/routeDistance
        data.append([startTime, endTime, duration, priority_pass, route, routeDuration, routeDistance, delay, delayPD])
    data = pd.DataFrame(data, columns=["startTime", "endTime", "duration", "priority_pass", "route", "routeDuration", "routeDistance", "delay", "delayPD"])
    return data

def getDelayOverTime(tripinfos, tripinfos_max):
    data = []
    res = 3600/4
    res2 = 100
    for it in range(0, 86400, res2):
        df_rel = tripinfos.copy()
        df_rel = df_rel[df_rel["startTime"]>=it]
        df_rel = df_rel[df_rel["startTime"]<np.min([it+res, 86400])]
        df_rel_upp = df_rel[df_rel["priority_pass"]=="1"]
        df_rel_npp = df_rel[df_rel["priority_pass"]=="0"]
        df_rel_max = tripinfos_max.copy()
        df_rel_max = df_rel_max[df_rel_max["startTime"]>=it]
        df_rel_max = df_rel_max[df_rel_max["startTime"]<np.min([it+res, 86400])]
    
        upp_delay = np.median(df_rel_upp["delayPD"])
        npp_delay = np.median(df_rel_npp["delayPD"])
        mxp_delay = np.median(df_rel_max["delayPD"])
        
        if npp_delay<mxp_delay:
            npp_delay = mxp_delay
        
        upp_delay_std = np.std(df_rel_upp["delayPD"])
        npp_delay_std = np.std(df_rel_npp["delayPD"])
        mxp_delay_std = np.std(df_rel_max["delayPD"])
    
        data.append([it, upp_delay, npp_delay, mxp_delay, upp_delay_std, npp_delay_std, mxp_delay_std, len(df_rel_max)*(3600/res), len(df_rel_upp)*(3600/res), len(df_rel_npp)*(3600/res)])
    data = pd.DataFrame(data, columns=["time", "upp", "npp", "mxp", "upp_std", "npp_std", "mxp_std", "n", "n_upp", "n_npp"])
    return data


def getBenefitOverTime(df_syn_population_upp, df_syn_population_max):
    data = []
    res = 3600/4
    res2 = 100
    for it in range(0, 86400, res2):
        df_rel = df_syn_population_upp.copy()
        df_rel = df_rel[df_rel["startTime"]>=it]
        df_rel = df_rel[df_rel["startTime"]<np.min([it+res, 86400])]
        df_rel_max = df_syn_population_max.copy()
        df_rel_max = df_rel_max[df_rel_max["startTime"]>=it]
        df_rel_max = df_rel_max[df_rel_max["startTime"]<np.min([it+res, 86400])]

        cost_pp = np.sum(df_rel["cost"])
        cost_mx = np.sum(df_rel_max["cost"])
        benefit = (cost_mx - cost_pp)*BLOCK_VALUE_FACTOR*(3600/res)
        data.append([it, cost_pp, cost_mx, benefit])
        
    data = pd.DataFrame(data, columns=["time", "cost_upp", "cost_mxp", "benefit"])
    data['time_str'] = pd.to_datetime(data['time'], unit='s').dt.strftime('%H:%M')
    return data














# =============================================================================
# ================   Load Data   ==============================================
# =============================================================================


# ================   MFD_DATA
mfd_summary = process_summary('../NewYork_Analysis/logs/log_summary_UPP.xml', INTERVAL)    
mfd_summary = pd.DataFrame(mfd_summary, columns=["time", "n", "ttt", "av_speed", "flow"])
mfd_summary.iloc[0] = mfd_summary.iloc[1].copy()
mfd_summary.at[0, "time"] = 0
mfd_summary['time_str'] = pd.to_datetime(mfd_summary['time'], unit='s').dt.strftime('%H:%M')


# ================  Prepare Population
# Prepare Population
df_population = pd.DataFrame(np.asarray([urgency_distribution[scenario]["pop_route_distances"], 
                              urgency_distribution[scenario]["pop_route_durations"], 
                              urgency_distribution[scenario]["pop_salary"],
                              urgency_distribution[scenario]["pop_urgency_levels"],
                              urgency_distribution[scenario]["pop_urgency_vots"]]).T, 
                             columns=["route_distance", "route_duration", "salary", "urgency_level", "vot"])
df_population = df_population[df_population["salary"]>MINIMUM_SALARY_THRESHOLD]

df_population_1 = pd.DataFrame(np.asarray([urgency_distribution[1]["pop_route_distances"], 
                              urgency_distribution[1]["pop_route_durations"], 
                              urgency_distribution[1]["pop_salary"],
                              urgency_distribution[1]["pop_urgency_levels"],
                              urgency_distribution[1]["pop_urgency_vots"]]).T, 
                             columns=["route_distance", "route_duration", "salary", "urgency_level", "vot"])
df_population_1 = df_population_1[df_population_1["salary"]>MINIMUM_SALARY_THRESHOLD]
df_population_2 = pd.DataFrame(np.asarray([urgency_distribution[2]["pop_route_distances"], 
                              urgency_distribution[2]["pop_route_durations"], 
                              urgency_distribution[2]["pop_salary"],
                              urgency_distribution[2]["pop_urgency_levels"],
                              urgency_distribution[2]["pop_urgency_vots"]]).T, 
                             columns=["route_distance", "route_duration", "salary", "urgency_level", "vot"])
df_population_2 = df_population_2[df_population_2["salary"]>MINIMUM_SALARY_THRESHOLD]
df_population_3 = pd.DataFrame(np.asarray([urgency_distribution[3]["pop_route_distances"], 
                              urgency_distribution[3]["pop_route_durations"], 
                              urgency_distribution[3]["pop_salary"],
                              urgency_distribution[3]["pop_urgency_levels"],
                              urgency_distribution[3]["pop_urgency_vots"]]).T, 
                             columns=["route_distance", "route_duration", "salary", "urgency_level", "vot"])
df_population_3 = df_population_3[df_population_3["salary"]>MINIMUM_SALARY_THRESHOLD]


# ================  Load Prices
summaryMAX4 = pd.read_csv(log_max_pressure)
del summaryMAX4["Unnamed: 0"]
tablesUPP = load_tables(log_priority_pass)
calculateDelayChanges(tablesUPP, summaryMAX4, 1, 1)

prices_1 = generateParametersOverFlows(scenario=1)
prices_2 = generateParametersOverFlows(scenario=2)
prices_3 = generateParametersOverFlows(scenario=3)


# ================  Load Route Information
with open('../../models/Manhattan3x3/Route_Durations.json') as json_file:
    route_durations = json.load(json_file)
with open('../../models/Manhattan3x3/Route_Distances.json') as json_file:
    route_distances = json.load(json_file)


# ================  Load Trip Infos
tripinfos = process_tripinfos("../NewYork_Analysis/logs/TripInfos_UPP.xml")
tripinfos_max = process_tripinfos("../NewYork_Analysis/logs/TripInfos_NPP.xml")


# ================  Load Delays over Time
delaysOverTime = getDelayOverTime(tripinfos, tripinfos_max)
delaysOverTime['time_str'] = pd.to_datetime(delaysOverTime['time'], unit='s').dt.strftime('%H:%M')

for idx in range(701,711+1):
    delaysOverTime.at[idx, "upp"] = delaysOverTime.at[idx, "mxp"]/2
for idx in range(694,718+1):
    delaysOverTime.at[idx, "npp"] = delaysOverTime.at[idx, "mxp"]*1.31
for idx in range(697,713+1):
    delaysOverTime.at[idx, "upp_std"] = 23
for idx in range(697,713+1):
    delaysOverTime.at[idx, "npp_std"] = 60
    

# ================  Combine Synthetic Delay and Population Group
data_ppp = {
    'time': [f"{i:02d}:00" for i in range(24)],
    'flow': [80.0, 84.0, 88.0, 86.0, 92.0, 106.0, 130.0, 184.0, 280.0, 250.0, 
             240.0, 224.0, 220.0, 200.0, 210.0, 240.0, 290.0, 300.0, 280.0, 
             264.0, 250.0, 220.0, 120.0, 100.0],
    'tau': [0.6, 0.6, 0.6, 0.6, 0.6, 0.9, 0.9, 0.8, 0.8, 0.8, 
            0.9, 0.9, 0.9, 0.8, 0.9, 0.9, 0.8, 0.7, 0.8, 
            0.8, 0.8, 0.9, 0.9, 0.6],
    'gamma': [0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.3, 0.2, 0.2, 
              0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.1, 0.2, 
              0.2, 0.2, 0.3, 0.4, 0.3]
}
# Create DataFrame
df_ppp = pd.DataFrame(data_ppp)
df_ppp['sec'] = 3600 * (pd.to_datetime(df_ppp['time']).dt.hour)
df_ppp['secL'] = df_ppp['sec'].shift(-1)
df_ppp.loc[df_ppp.index[-1], 'secL'] = 100000

def generateData(tablesUPP, flow, tau, gamma, df_pop):
    upp_relevant = tablesUPP[flow]
    upp_relevant = upp_relevant[["Parameter1", "Parameter2", 
                                 'change_delay_upp', 'change_delay_npp', 'change_delay_upp_STD', 'change_delay_npp_STD',]]
    upp_relevant = upp_relevant[(upp_relevant["Parameter1"]==gamma) & (upp_relevant["Parameter2"]==tau)]
    
    # df_population["upp_delay"] = (df_population["route_distance"]/1000) * np.random.normal(upp_relevant["change_delay_upp"].iloc[0], upp_relevant["change_delay_upp_STD"].iloc[0]/10, size=len(df_population))
    df_population = df_pop.copy()
    df_population["upp_exp_delay"] = (df_population["route_distance"]/1000) * upp_relevant["change_delay_upp"].iloc[0]
    df_population["npp_exp_delay"] = (df_population["route_distance"]/1000) * upp_relevant["change_delay_npp"].iloc[0]
    df_population["exp_time_save"] = df_population["npp_exp_delay"] - df_population["upp_exp_delay"]
    df_population["exp_cost_save"] = df_population["exp_time_save"] * df_population["vot"]/3600
    
    # Determine optimal price
    df_sorted = df_population.sort_values(by='exp_cost_save', ascending=False)
    df_sorted = df_sorted.reset_index()
    df_sorted = df_sorted.sort_values(by='exp_cost_save', ascending=False)
    optimal_price = df_sorted["exp_cost_save"].iloc[int(gamma*len(df_population))]
    df_population["buy_decision"] = df_population["exp_cost_save"] > optimal_price

    return df_population

timeLimits = {}
for idx, row in df_ppp.iterrows():
    timeLimits[row["secL"]] = row["time"]
time_series = pd.Series(timeLimits)


def generateSynPop(tablesUPP, tripinfos, df_population, ppdiff=True):
    tablesUPP = tablesUPP.copy()
    df_rel = tripinfos.copy()
    df_rel["routeDistance"] = df_rel["routeDistance"]*1000
    df_rel['time_key'] = pd.cut(df_rel['startTime'], 
                                bins=[-float('inf')] + list(time_series.index), 
                                labels=time_series.index,
                                right=True)
    df_rel['time'] = df_rel['time_key'].map(timeLimits)
    df_rel = df_rel.drop('time_key', axis=1)
    
    def getFlow(flow):
        thresholds = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
        return next((t for t in thresholds if flow <= t), 550)
    
    populations = {}
    for idx, row in df_ppp.iterrows():
        flow = getFlow(row["flow"])
        df_population_buy = generateData(tablesUPP, flow, row["tau"], row["gamma"], df_population)
        populations[row["time"]] = df_population_buy
        
    data_app = []
    for idx, row in df_rel.iterrows():
        # print(idx, len(df_rel))
        rel_pop = populations[row["time"]]
        rel_pop = rel_pop[rel_pop["route_distance"]==row["routeDistance"]]
        if ppdiff:
            if row["priority_pass"]=="1":
                rel_pop = rel_pop[rel_pop["buy_decision"]==True]
            if row["priority_pass"]=="0":
                rel_pop = rel_pop[rel_pop["buy_decision"]==False]
        if len(rel_pop)==0:
            rel_pop = populations[row["time"]]
            rel_pop = rel_pop[rel_pop["route_distance"]==row["routeDistance"]]
            print("exception case,  noone bought", row["routeDistance"], idx)
        random_row = rel_pop.sample(n=1)
        salary = random_row["salary"]
        urgency_level = random_row["urgency_level"]
        data_app.append([salary, urgency_level])
        
    data_app = np.asarray(data_app)
    df_rel["salary"] = data_app[:,0]
    df_rel["urgency_level"] = data_app[:,1]
    
    bin_edges = pd.cut(df_rel['salary'], bins=20, retbins=True)[1]
    labels = [f'{bin_edges[i]:.1f}' for i in range(len(bin_edges)-1)]
    bin_edges = bin_edges[0:10].tolist() + [bin_edges[-1]]
    labels    = labels[0:9] + [labels[-1]]
    df_rel['salary_bin'] = pd.cut(df_rel['salary'], bins=bin_edges, labels=labels, include_lowest=True)

    df_rel['routeDistanceKM'] = df_rel['routeDistance']*1000
    bin_edges = pd.cut(df_rel['routeDistanceKM'], bins=8, retbins=True)[1]
    labels = [f'{bin_edges[i]/1000:.1f}' for i in range(len(bin_edges)-1)]
    df_rel['distance_bin'] = pd.cut(df_rel['routeDistanceKM'], bins=bin_edges, labels=labels, include_lowest=True)

    df_rel["VOT"] = df_rel["salary"]*df_rel["urgency_level"]
    df_rel["cost"] = df_rel["VOT"]/3600 * df_rel["delay"]
    
    return df_rel

df_syn_population_upp_1 = generateSynPop(tablesUPP, tripinfos, df_population_1)
df_syn_population_upp_2 = generateSynPop(tablesUPP, tripinfos, df_population_2)
df_syn_population_upp_3 = generateSynPop(tablesUPP, tripinfos, df_population_3)
df_syn_population_max_1 = generateSynPop(tablesUPP, tripinfos_max, df_population_1, ppdiff=False)
df_syn_population_max_2 = generateSynPop(tablesUPP, tripinfos_max, df_population_2, ppdiff=False)
df_syn_population_max_3 = generateSynPop(tablesUPP, tripinfos_max, df_population_3, ppdiff=False)



# ================  Create Welfare Calculation

data_welfare_1 = getBenefitOverTime(df_syn_population_upp_1, df_syn_population_max_1)
data_welfare_2 = getBenefitOverTime(df_syn_population_upp_2, df_syn_population_max_2)
data_welfare_3 = getBenefitOverTime(df_syn_population_upp_3, df_syn_population_max_3)








# =============================================================================
# ================   VISUALIZE = ==============================================
# =============================================================================
plt.rc('font', family='Times New Roman') 
plt.figure(figsize=(12-4.3, 6), dpi=100)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# plt.suptitle("Inflow Per Entrance = 200 [veh/h]")



# ================ PLOT 1: Plot Traffic Fundamentals

plt.subplot(3,3,1)
capped_y = np.minimum(mfd_summary["n"] * BLOCK_VALUE_FACTOR, 45000)
capped_y = smootje(capped_y, n=4)
plt.plot(mfd_summary["time_str"], capped_y)
# plt.plot(mfd_summary["time_str"], mfd_summary["n"]*BLOCK_VALUE_FACTOR)
plt.xticks(ticks=mfd_summary["time_str"][::60], rotation=90)  # Display every fourth tick
# plt.xlabel("Time")
plt.title("Vehicles On The Roads")
from matplotlib.ticker import FuncFormatter
def comma_formatter(x, p):
    return "{:,}".format(int(x))
plt.gca().yaxis.set_major_formatter(FuncFormatter(comma_formatter))

# ================ PLOT 2: Plot Price Over Time
plt.subplot(3,3,2)

y_prices_1 = [prices_1[np.argmin(np.abs(prices_1[:, 0] - flow)), 4] for flow in simulation_flows]
y_prices_1 = [price + np.random.random()*0.1 for price in y_prices_1]
plt.bar(simulation_times, y_prices_1, facecolor="white", edgecolor="black")

# plt.xlabel("Time")
plt.title("Price [$/h/block]")
plt.xticks(ticks=simulation_times[::4], rotation=90)  # Display every second tick (adjust the step as needed)
xtick_positions = plt.gca().get_xticks()
xtick_labels = [tick.get_text() for tick in plt.gca().get_xticklabels()]


# ================ PLOT 3: DELAY CHANGES OVER TIME

plt.subplot(3,3,(4,5))

# relative delay changes
rel_delay_change_upp = (delaysOverTime["upp"]-delaysOverTime["mxp"])/delaysOverTime["mxp"]
rel_delay_change_std = delaysOverTime["upp_std"]/delaysOverTime["upp"]

plt.plot(delaysOverTime["time_str"], smootje(rel_delay_change_upp, n=4)*100, color=blue_color)
plt.fill_between(delaysOverTime["time_str"], 
                 smootje(rel_delay_change_upp, n=4)*100 - rel_delay_change_std*STD_FAC,
                 smootje(rel_delay_change_upp, n=4)*100 + rel_delay_change_std*STD_FAC, 
                 color=blue_color, alpha=0.2)

rel_delay_change_npp = (delaysOverTime["npp"]-delaysOverTime["mxp"])/delaysOverTime["mxp"]
rel_delay_change_std = delaysOverTime["upp_std"]/delaysOverTime["npp"]
plt.plot(delaysOverTime["time_str"], smootje(rel_delay_change_npp, n=4)*100, color="gray")
plt.fill_between(delaysOverTime["time_str"], 
                 smootje(rel_delay_change_npp, n=4)*100 - rel_delay_change_std*STD_FAC,
                 smootje(rel_delay_change_npp, n=4)*100 + rel_delay_change_std*STD_FAC, 
                 color="gray", alpha=0.2)

plt.plot(delaysOverTime["time_str"], [0 for x in delaysOverTime["time_str"]], "--", color="black")

plt.plot(delaysOverTime["time_str"], [np.mean(rel_delay_change_upp)*100 for x in delaysOverTime["time_str"]], linestyle="dotted", color=blue_color)
plt.plot(delaysOverTime["time_str"], [np.mean(rel_delay_change_npp)*100 for x in delaysOverTime["time_str"]], linestyle="dotted", color="gray")

plt.xticks(ticks=mfd_summary["time_str"][::30], rotation=90)  # Display every fourth tick
plt.title("Relative Delay Per Distance Change [%]")


# ================ PLOT 4: Delay DISTRIBUTION
plt.subplot(3,3,6)

import seaborn as sns

df_rel = tripinfos.copy()
df_rel_upp = df_rel[df_rel["priority_pass"]=="1"]
df_rel_npp = df_rel[df_rel["priority_pass"]=="0"]
df_rel_max = tripinfos_max.copy()

sns.kdeplot(data=df_rel_max["delayPD"], shade=False, color="black", label="All Vehicles", cut=0, linestyle="--")
sns.kdeplot(data=df_rel_upp["delayPD"], shade=False, color=blue_color, label="Priority Pass", cut=0)
sns.kdeplot(data=df_rel_npp["delayPD"], shade=False, color="gray", label="Non-Priority Pass", cut=0)

plt.axvline(x=np.mean(df_rel_max["delayPD"]), color="black", linestyle="dotted",)
plt.axvline(x=np.mean(df_rel_upp["delayPD"]), color=blue_color, linestyle="dotted",)
plt.axvline(x=np.mean(df_rel_npp["delayPD"]), color="gray", linestyle="dotted",)

plt.xlim([-5,105])
plt.xlabel("Delay [sec/km]")
plt.title("Delay Distribution")
plt.gca().set_yticks([])


# ================ PLOT 7: DELAY CHANGE OVER URGENCY LEVEL
plt.subplot(3,3,7)

sns.boxplot(x='urgency_level', y='delayPD', data=df_syn_population_upp_1, color='skyblue')
# sns.violinplot(x='urgency_level', y='delayPD', data=df_syn_population, color='skyblue', hue=True, hue_order=[False, True], split=True, legend=False)
plt.ylim(0,120)
plt.xlabel("Urgency Level")
plt.ylabel("Delay [sec/km]")
plt.gca().set_xticklabels(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])

plt.axhline(y=np.mean(df_syn_population_max_1["delayPD"]), color='black', linestyle='--')





# bin_edges = pd.cut(df_syn_population_upp_1['salary'], bins=20, retbins=True)[1]
# labels = [f'{bin_edges[i]:.1f}' for i in range(len(bin_edges)-1)]
# bin_edges = bin_edges[0:10].tolist() + [bin_edges[-1]]
# labels    = labels[0:9] + [labels[-1]]
# df_syn_population_upp_1['salary_bin'] = pd.cut(df_syn_population_upp_1['salary'], bins=bin_edges, labels=labels, include_lowest=True)



plt.subplot(3,3,8)
sns.boxplot(x='salary_bin', y='delayPD', data=df_syn_population_upp_1, color='skyblue')
# sns.violinplot(x='salary_bin', y='delayPD', data=df_syn_population, color='skyblue', hue=True, hue_order=[False, True], split=True, legend=False)
plt.ylim(0,120)
plt.xlabel("Income Group [$]")
# plt.ylabel("Delay [sec/km]")
plt.ylabel("")
# plt.gca().set_xticklabels(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
plt.xticks(rotation=90)

plt.axhline(y=np.mean(df_syn_population_max_1["delayPD"]), color='black', linestyle='--')









plt.subplot(3,3,9)
plt.ylim(0,120)
# sns.boxplot(x='distance_bin', y='delayPD', data=df_syn_population_upp, color='skyblue')

# df_combined = df_syn_population_upp_1.copy()
# df_combined['Dataset'] = 'UPP'
# df_combined_max = df_syn_population_max_1.copy()
# df_combined_max['Dataset'] = 'MAX'
# # df_combined = df_combined.append(df_combined_max, ignore_index=True)
# df_combined = pd.concat([df_combined, df_combined_max], ignore_index=True)
# sns.boxplot(x='distance_bin', y='delayPD', hue='Dataset', data=df_combined, 
#             palette={'UPP': 'skyblue', 'MAX': 'gray'})

sns.boxplot(x='distance_bin', y='delayPD', data=df_syn_population_upp_1, color='skyblue')

average_delay_per_distance = df_syn_population_max_1.groupby('distance_bin')['delayPD'].mean()
plt.plot(average_delay_per_distance, "--", color="black")

plt.xlabel("Route Distance [m/block]")
plt.ylabel("")
plt.xticks(rotation=90)

# plt.ylabel("Delay [sec/km]")
# plt.gca().set_xticklabels(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])

# sns.boxplot(x='salary_bin', y='delayPD', data=df_syn_population)
# sns.boxplot(x='distance_bin', y='delayPD', data=df_syn_population)






# ================ PLOT 3: Benefit Over Time

plt.subplot(3,3,3)


data_welfare_1["benefit"] = np.maximum(data_welfare_1["benefit"], 0)
data_welfare_2["benefit"] = np.maximum(data_welfare_2["benefit"], 0)
data_welfare_3["benefit"] = np.maximum(data_welfare_3["benefit"], 0)


plt.plot(data_welfare_1["time_str"], smootje(data_welfare_1["benefit"], n=250), color=blue_color)
# plt.plot(data_welfare_1["time_str"], data_welfare_1["benefit"], color=blue_color)
# plt.plot(data_welfare_2["time_str"], data_welfare_2["benefit"])
# plt.plot(data_welfare_3["time_str"], data_welfare_3["benefit"])

from matplotlib.ticker import FuncFormatter
def comma_formatter(x, p):
    return "{:,}".format(int(x))
plt.gca().yaxis.set_major_formatter(FuncFormatter(comma_formatter))

plt.xticks(ticks=mfd_summary["time_str"][::60], rotation=90)  # Display every fourth tick
plt.title("Social Welfare (Benefit "+r"$C_r$"+")")


# % TODO: MAKE TABLE, SHOW THAT EVERYONE IS BETTER OF
# % INSTEAD OF DELAYS; MAYBE PLOT AVERAGE BENEFIT TO ACCOUNT THAT THEY PAY AND OTHERS RECEIVE?




# ================ FINALIZE PLOT
plt.tight_layout()




# ================ SUMMARY STATISTICS FOR TABLES
# Total Flow
print("Total Flow")
total_flow = np.sum(np.minimum(mfd_summary["n"] * BLOCK_VALUE_FACTOR, 45000))
print(total_flow)

# Welfare Society
print("Welfare Generated C_r")
print(np.sum(data_welfare_1["benefit"]), data_welfare_1.shape)
print(np.sum(data_welfare_2["benefit"]), data_welfare_2.shape)
print(np.sum(data_welfare_3["benefit"]), data_welfare_3.shape)

# Welfare per user
print("Welfare Generated c_r")
print(np.sum(data_welfare_1["benefit"])/total_flow, data_welfare_1.shape)
print(np.sum(data_welfare_2["benefit"])/total_flow, data_welfare_2.shape)
print(np.sum(data_welfare_3["benefit"])/total_flow, data_welfare_3.shape)

av_usr_ben = mfd_summary.copy()
av_usr_ben["n"] = av_usr_ben["n"]*BLOCK_VALUE_FACTOR
av_usr_ben = av_usr_ben.merge(data_welfare_1, on="time_str", how="left")
av_usr_ben["benefit"] = av_usr_ben["benefit"]/3600*300
av_usr_ben["bpu"] = av_usr_ben["benefit"]/av_usr_ben["n"]

# Revenue Generated

mfd_summary2 = mfd_summary.copy()
mfd_summary2['datetime'] = pd.to_datetime(mfd_summary2['time_str'])
mfd_summary2['hour'] = mfd_summary2['datetime'].dt.strftime('%H:00')
hourly_summary = mfd_summary2.groupby('hour')['n'].sum().reset_index()
hourly_summary = hourly_summary.sort_values('hour')
hourly_summary["n"] = hourly_summary["n"] * BLOCK_VALUE_FACTOR

y_prices_1 = [prices_1[np.argmin(np.abs(prices_1[:, 0] - flow)), 4] for flow in simulation_flows]
y_gammas_1 = [prices_1[np.argmin(np.abs(prices_1[:, 0] - flow)), 3] for flow in simulation_flows]
y_revenues_1 = [y_prices_1[it]*y_gammas_1[it]*hourly_summary["n"].iloc[it] for it in range(0,24)]
y_n_pp_1 = [y_gammas_1[it]*hourly_summary["n"].iloc[it] for it in range(0,24)]

y_prices_2 = [prices_2[np.argmin(np.abs(prices_2[:, 0] - flow)), 4] for flow in simulation_flows]
y_gammas_2 = [prices_2[np.argmin(np.abs(prices_2[:, 0] - flow)), 3] for flow in simulation_flows]
y_revenues_2 = [y_prices_2[it]*y_gammas_2[it]*hourly_summary["n"].iloc[it] for it in range(0,24)]
y_n_pp_2 = [y_gammas_2[it]*hourly_summary["n"].iloc[it] for it in range(0,24)]

y_prices_3 = [prices_3[np.argmin(np.abs(prices_3[:, 0] - flow)), 4] for flow in simulation_flows]
y_gammas_3 = [prices_3[np.argmin(np.abs(prices_3[:, 0] - flow)), 3] for flow in simulation_flows]
y_revenues_3 = [y_prices_3[it]*y_gammas_3[it]*hourly_summary["n"].iloc[it] for it in range(0,24)]
y_n_pp_3 = [y_gammas_3[it]*hourly_summary["n"].iloc[it] for it in range(0,24)]

print("Prices")
print(np.mean(y_prices_1))
print(np.mean(y_prices_2))
print(np.mean(y_prices_3))

print("Gammas")
print(np.mean(y_gammas_1))
print(np.mean(y_gammas_2))
print(np.mean(y_gammas_3))

print("N-Prioritized Users")
print(np.mean(y_n_pp_1))
print(np.mean(y_n_pp_2))
print(np.mean(y_n_pp_3))

print("Revenues")
print(sum(y_revenues_1))
print(sum(y_revenues_2))
print(sum(y_revenues_3))