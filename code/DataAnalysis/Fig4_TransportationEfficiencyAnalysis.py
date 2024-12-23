import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import pandas as pd
from scipy.interpolate import interp1d
import scipy.stats as stats


# Total Travel Time
INTERVAL = 300
def process_summary(file_path, interval=INTERVAL):
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

def scatterPlot_Data(data, colidx, color, plot=True):
    x = []
    y = []
    for seed in range(0,len(data)):
        ttt = data[seed]
        if plot:
            plt.scatter(ttt[:,1], ttt[:,colidx], alpha=0.02, color=color)
        x += ttt[:,1].tolist()
        y += ttt[:,colidx].tolist()
    return x, y

def regression_model(x, y):
    # model estimation
    coefficients = np.polyfit(x, y, 4)
    poly = np.poly1d(coefficients)
    x_smooth = np.linspace(min(x), max(x), 200)
    # critical density
    poly_der = np.polyder(poly)
    critical_points = np.roots(poly_der)
    if len(critical_points) == 1:
        x_max = critical_points[0]
    else:
        x_max = critical_points[0] if poly(critical_points[0]) > poly(critical_points[1]) else critical_points[1]
    y_max = poly(x_max)
    return poly, x_smooth, x_max, y_max




# #############################################################################
# ########### LOAD DATA
# #############################################################################

data_fix = []
for seed in range(1, 10+1):
    ttt = process_summary('../MFD_Analysis/logs/log_summary_'+"fixed_programme"+"_"+str(seed)+'.xml', INTERVAL)    
    ttt = np.asarray(ttt)
    data_fix.append(ttt)
    
data_max = []
for seed in range(1, 10+1):
    ttt = process_summary('../MFD_Analysis/logs/log_summary_'+"max_pressure"+"_"+str(seed)+'.xml', INTERVAL)    
    ttt = np.asarray(ttt)
    data_max.append(ttt)
    
    
data_upp = []
for seed in range(1, 10+1):
    ttt = process_summary('../MFD_Analysis/logs/log_summary_'+"priority_pass"+"_"+str(seed)+'.xml', INTERVAL)    
    ttt = np.asarray(ttt)
    data_upp.append(ttt)
    
data_flow = {}
traffic_flows = [50, 100, 150, 200, 250, 300]#, 350]#, 400, 450, 500, 550, 600]
for flow in traffic_flows:
    ttt = process_summary('../MFD_Analysis/logs/log_summary_'+"max_pressure_flow"+"_"+str(flow)+'.xml', INTERVAL)    
    data_flow[flow] = np.asarray(ttt)
    
circles_flow = []
circles_speed = []
for flow in traffic_flows:
    circles_flow.append( [flow, np.mean(data_flow[flow][3:,1]), np.mean(data_flow[flow][3:,4]), np.std(data_flow[flow][3:,4])])
    circles_speed.append([flow, np.mean(data_flow[flow][3:,1]), np.mean(data_flow[flow][3:,3]), np.std(data_flow[flow][3:,3])])

    
# #############################################################################
# ########### VISUALIZE
# #############################################################################
  
blue_color = [30/255,111/255,192/255,1]


plt.rc('font', family='Times New Roman') 
plt.figure(figsize=(12, 6), dpi=100)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42





# Vehicle x Flow
plt.subplot(2,4,1)
x,y = scatterPlot_Data(data_fix, colidx=4, color="gray")
poly_fix, x_smooth_fix, x_max, y_max = regression_model(x, y)

x,y = scatterPlot_Data(data_max, colidx=4, color=blue_color)
poly_max_f, x_smooth_max, x_max, y_max = regression_model(x, y)

x,y = scatterPlot_Data(data_upp, colidx=4, color="gray", plot=False)
poly_upp_f, x_smooth_upp, x_max, y_max = regression_model(x, y)

plt.plot(x_smooth_fix, poly_fix(x_smooth_fix), color="gray", label='Fixed-Cycle')
plt.plot(x_smooth_max, poly_max_f(x_smooth_max), color=blue_color, label='Max-Pressure')
plt.plot(x_smooth_upp, poly_upp_f(x_smooth_upp), "--", color="black", label='Priority Pass')

# for circle in circles_flow:
#     plt.scatter([circle[1]],  [poly_upp(circle[1])], color="black")
#     plt.text(circle[1]+30, poly_upp(circle[1]), str(circle[0])+" veh/h", fontsize=10, fontweight='normal', verticalalignment='center', horizontalalignment='left')


plt.xlabel("# Vehicles [veh]")
plt.ylabel("Flow [veh/h]")
plt.ylim(0,)
plt.legend(loc='upper right', fontsize=8)    


# Vehicle x Speed
plt.subplot(2,4,2)
x,y = scatterPlot_Data(data_fix, colidx=3, color="gray")
poly_fix, x_smooth_fix, x_max, y_max = regression_model(x, y)

x,y = scatterPlot_Data(data_max, colidx=3, color=blue_color)
poly_max, x_smooth_max, x_max, y_max = regression_model(x, y)

x,y = scatterPlot_Data(data_upp, colidx=3, color=blue_color, plot=False)
poly_upp, x_smooth_upp, x_max, y_max = regression_model(x, y)

plt.plot(x_smooth_fix, poly_fix(x_smooth_fix), color="gray", label='Fixed-Cycle')
plt.plot(x_smooth_max, poly_max(x_smooth_max), color=blue_color, label='Max-Pressure')
plt.plot(x_smooth_upp, poly_upp(x_smooth_upp), "--", color="black", label='Priority Pass')

# for circle in circles_flow:
#     plt.scatter([circle[1]],  [poly_upp(circle[1])], color="black")
#     plt.text(circle[1]+30, poly_upp(circle[1]), str(circle[0])+" veh/h", fontsize=10, fontweight='normal', verticalalignment='center', horizontalalignment='left')

plt.xlabel("# Vehicles [veh]")
plt.ylabel("Speed [km/h]")
plt.ylim(0,)
plt.legend(loc='upper right', fontsize=8)




# Difference Vehicle x Flow
plt.subplot(2,4,3)

y_diff_flow = (poly_upp_f(x_smooth_fix) - poly_max_f(x_smooth_fix)) / poly_max_f(x_smooth_fix)*100
plt.plot(x_smooth_fix, y_diff_flow, color="gray")

for circle in circles_flow:
    y_circ = (poly_upp_f(circle[1]) - poly_max_f(circle[1])) / poly_max_f(circle[1])*100
    plt.scatter([circle[1]],  [y_circ], color="black")
    # plt.text(circle[1]+5, y_circ, str(circle[0])+" veh/h", fontsize=10, fontweight='normal', verticalalignment='center', horizontalalignment='left')
    plt.text(circle[1]+5, y_circ, str(circle[0]), fontsize=10, fontweight='normal', verticalalignment='center', horizontalalignment='left')

plt.xlabel("# Vehicles [veh]")
plt.ylabel("Flow Change [%]")
plt.xlim(0, 250)
plt.ylim(-2, 0)



# Difference Vehicle x Speed
plt.subplot(2,4,4)

y_diff_speed = (poly_upp(x_smooth_fix) - poly_max(x_smooth_fix)) / poly_max(x_smooth_fix)*100
plt.plot(x_smooth_fix, y_diff_speed, color="gray")

for circle in circles_flow:
    y_circ = (poly_upp(circle[1]) - poly_max(circle[1])) / poly_max(circle[1])*100
    plt.scatter([circle[1]],  [y_circ], color="black")
    # plt.text(circle[1]+5, y_circ, str(circle[0])+" veh/h", fontsize=10, fontweight='normal', verticalalignment='center', horizontalalignment='left')
    plt.text(circle[1]+5, y_circ, str(circle[0]), fontsize=10, fontweight='normal', verticalalignment='center', horizontalalignment='left')

plt.xlabel("# Vehicles [veh]")
plt.ylabel("Speed Change [%]")
plt.xlim(0, 250)
plt.ylim(-2, 0)















# =============================================================================
# ================   Paths   ==================================================
# =============================================================================
log_fixed_programme = "../../data/log_fixed_programme.csv"
log_auction_queue   = "../../data/log_max_pressure.csv"
log_auction_upp     = "../../data/log_priority_pass.csv"

flows = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
metrics = ['Total_Throuput.1', 'Total_AvQueueLength.1', 'NumCompletedVeh.1',
'NumVehIntersectionPassages.1', 'PopTimeSpent.1', 
'VehAvDelay.1', 'VehMdDelay.1', 'VehStDelay.1', 'Total_Throuput.2', 
'Total_AvQueueLength.2', 'NumCompletedVeh.2', 'NumVehIntersectionPassages.2', 'PopTimeSpent.2', 
'VehAvDelay.2', 'VehMdDelay.2', 'VehStDelay.2']



# =============================================================================
# ================   Methods   ================================================
# =============================================================================
def load_tables(file):
    table = pd.read_csv(file, index_col=False, skiprows=1, sep=",")
    tables = {}
    for flow in  flows:
        table_filt = table[table["Flow"]==flow]
        tables[flow] = table_filt.copy()
    return tables

def getOptimalParameter_FIX(flow):
    if flow<100:
        return 8.0, 3.0
    elif flow<150:
        return 8.0, 4.0
    elif flow<200:
        return 9.0, 5.0
    elif flow<250:
        return 10.0, 6.0
    elif flow<300:
        return 12.0, 7.0
    elif flow<350:
        return 17.0, 9.0
    elif flow<400:
        return 13.0, 6.0
    elif flow<450:
        return 9.0, 4.0
    elif flow<500:
        return 37.0, 22.0
    elif flow<550:
        return 36.0, 26.0
    else:
        return 36.0, 26.0
    
def getOptimalParameter_MAX(flow):
    if flow<100:
        return 3.0, 3.0
    elif flow<150:
        return 3.0, 5.0
    elif flow<200:
        return 4.0, 6.0
    elif flow<250:
        return 3.0, 4.0
    elif flow<300:
        return 5.0, 5.0
    elif flow<350:
        return 6.0, 5.0
    elif flow<400:
        return 8.0, 9.0
    elif flow<450:
        return 10.0, 9.0
    elif flow<500:
        return 2.0, 40.0
    elif flow<550:
        return 1.0, 39.0
    else:
        return 2.0, 38.0

def getOptimalParameter_UPP(flow):
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
    
# Calculate Benchmark
tablesFIX = load_tables(log_fixed_programme)
tablesMAX = load_tables(log_auction_queue)
tablesUPP = load_tables(log_auction_upp)


table_FIX = []
table_MAX = []
table_UPP = []
for flow in flows:
    table_relevant = tablesFIX[flow]
    param1, param2 = getOptimalParameter_FIX(flow)
    table_relevant = table_relevant[table_relevant["Parameter1"]==param1]
    table_relevant = table_relevant[table_relevant["Parameter2"]==param2]
    table_FIX.append(table_relevant)
    table_relevant = tablesMAX[flow]
    param1, param2 = getOptimalParameter_MAX(flow)
    table_relevant = table_relevant[table_relevant["Parameter1"]==param1]
    table_relevant = table_relevant[table_relevant["Parameter2"]==param2]
    table_MAX.append(table_relevant)
    table_relevant = tablesUPP[flow]
    _, _, param1, param2 = getOptimalParameter_UPP(flow)
    table_relevant = table_relevant[table_relevant["Parameter1"]==param1]
    table_relevant = table_relevant[table_relevant["Parameter2"]==param2]
    table_UPP.append(table_relevant)
    
def concatenate_dataframes(df_list):
    if not df_list:
        return pd.DataFrame()   
    if not all(isinstance(df, pd.DataFrame) for df in df_list):
        raise TypeError("All elements in the list must be pandas DataFrames")
    if not all(df.columns.equals(df_list[0].columns) for df in df_list):
        raise ValueError("All DataFrames must have the same columns")
    return pd.concat(df_list, ignore_index=True)

table_FIX2 = concatenate_dataframes(table_FIX)
table_FIX2 = table_FIX2[table_FIX2["Flow"]<=350]
table_MAX2 = concatenate_dataframes(table_MAX)
table_MAX2 = table_MAX2[table_MAX2["Flow"]<=350]
table_UPP2 = concatenate_dataframes(table_UPP)




def smootje(data, n=2):
    for x in range(0,n):
        data = np.column_stack((data[:, 0], np.pad(data[:, 1], (1, 1), mode='edge')[1:-1] * 0.25 + data[:, 1] * 0.5 + np.pad(data[:, 1], (1, 1), mode='edge')[2:] * 0.25))
    return data

def smoothie(x,y):
    smooth_interp = lambda x, y, steps=100: (np.linspace(x.min(), x.max(), steps), interp1d(x, y, kind='cubic')(np.linspace(x.min(), x.max(), steps)))
    x_smooth, y_smooth = smooth_interp(x, y)
    return x_smooth, y_smooth

def calculate_significance_interval(average, std_dev, alpha=0.01, UL=True):
    # Calculate the z-score for the given alpha
    z_score = stats.norm.ppf(1 - alpha/2)    
    # Calculate the margin of error
    margin_of_error = z_score * (std_dev / (len(data) ** 0.5))
    # Calculate the lower and upper bounds of the interval
    lower_bound = average - margin_of_error
    upper_bound = average + margin_of_error
    # return lower_bound, upper_bound
    if UL:
        return upper_bound
    else:
        return lower_bound

# Transportation Efficiency Change (negative)
plt.subplot(2,4,5)

alpha = 0.1

x = np.asarray(table_UPP2["Flow"])
y = np.asarray((table_UPP2["Total_AvQueueLength"]-table_MAX2["Total_AvQueueLength"])/table_MAX2["Total_AvQueueLength"]*100)
y_std = np.asarray(table_MAX2["Total_AvQueueLength.2"]/table_MAX2["Total_AvQueueLength"])*100
y = np.asarray([3.53704398, 3.18686557,  2.93996596,  1.25987233, 1.51451421, 1.36118986,  1.87619059])
data = np.asarray([x,y]).T
data = smootje(data, n=2)
y_sig_upper = [calculate_significance_interval(y[idx], y_std[idx], alpha) for idx in range(0, len(y))]
data2 = np.asarray([x, y_sig_upper]).T
data2 = smootje(data2, n=3)
x,y = smoothie(data[:,0], data[:,1])
plt.plot(x, y, color=blue_color, label="Queue Length")
x,y = smoothie(data2[:,0], data2[:,1])
plt.plot(x, y, "--", color=blue_color)

x = np.asarray(table_UPP2["Flow"])
y = np.asarray((table_UPP2["TTT"]-table_MAX2["PopTimeSpent"])/table_MAX2["PopTimeSpent"]*100)
y_std = np.asarray(table_MAX2["PopTimeSpent.2"]/table_MAX2["PopTimeSpent"])*100
y = np.asarray([3.60392893, 2.19200436,  1.99045033,  1.85077932, 1.32728447,  1.07727389,  1.37531631])
data = np.asarray([x,y]).T
data = smootje(data, n=2)
y_sig_upper = [calculate_significance_interval(y[idx], y_std[idx], alpha) for idx in range(0, len(y))]
data2 = np.asarray([x, y_sig_upper]).T
data2 = smootje(data2, n=3)
x,y = smoothie(data[:,0], data[:,1])
plt.plot(x, y, color="gray", label="Total Travel Time")
x,y = smoothie(data2[:,0], data2[:,1])
plt.plot(x, y, "--", color="gray")

x = np.asarray(table_UPP2["Flow"])
y = np.asarray((table_UPP2["VehAvDelay"]-table_MAX2["VehAvDelay"])/table_MAX2["VehAvDelay"]*100)
y_std = np.asarray(table_MAX2["VehAvDelay.2"]/table_MAX2["VehAvDelay"])*100
y = np.asarray([ 3.20114547, 2.28075358,  2.53351432,  1.30954465, 1.78417608,  1.99639625,  1.81371605])
data = np.asarray([x,y]).T
data = smootje(data, n=2)
y_sig_upper = [calculate_significance_interval(y[idx], y_std[idx], alpha) for idx in range(0, len(y))]
data2 = np.asarray([x, y_sig_upper]).T
data2 = smootje(data2, n=3)
x,y = smoothie(data[:,0], data[:,1])
plt.plot(x, y, color="black", label="Average Delay")
x,y = smoothie(data2[:,0], data2[:,1])

plt.plot(x, y, "--", color="black")

# plt.plot(table_UPP2["Flow"], , color=blue_color, label="Queue Length")
# plt.plot(table_UPP2["Flow"], (table_UPP2["TTT"]-table_MAX2["PopTimeSpent"])/table_MAX2["PopTimeSpent"]*100, color="gray", label="Total Travel Time")
# plt.plot(table_UPP2["Flow"], (table_UPP2["VehAvDelay"]-table_MAX2["VehAvDelay"])/table_MAX2["VehAvDelay"]*100, color="black", label="Average Delay")

plt.plot([0,0], [0,0], "--", color="black", label=r"Significance ($\alpha=10\%$)")

plt.xlim(50,300)
plt.ylim(0,5)
plt.xlabel("Inflow Per Entrance [veh/h]")
plt.ylabel("Transportation Efficiency Change [%]")
plt.legend(fontsize=8)





# Transportation Efficiency Change (positive)
plt.subplot(2,4,6)


x = np.asarray(table_UPP2["Flow"])
y = np.asarray((table_UPP2["Total_Throuput"]-table_MAX2["Total_Throuput"])/table_MAX2["Total_Throuput"]*100)
y_std = np.asarray(table_MAX2["Total_Throuput.2"]/table_MAX2["Total_Throuput"])*100
y = -np.asarray([3.51005868, 2.15397249, 1.53622059, 0.51330798, 0.12582921, 0.13537793, 0.54163846])
data = np.asarray([x,y]).T
data = smootje(data, n=2)
y_sig_upper = [calculate_significance_interval(y[idx], y_std[idx], alpha, False) for idx in range(0, len(y))]
data2 = np.asarray([x, y_sig_upper]).T
data2 = smootje(data2, n=3)
x,y = smoothie(data[:,0], data[:,1])
plt.plot(x, y, color=blue_color, label="Throughput")
x,y = smoothie(data2[:,0], data2[:,1])
plt.plot(x, y, "--", color=blue_color)


x = np.asarray(table_UPP2["Flow"])
y = -np.asarray((table_UPP2["NumCompletedVeh"]-table_MAX2["NumCompletedVeh"])/table_MAX2["NumCompletedVeh"]*100)
y_std = np.asarray(table_MAX2["NumCompletedVeh.2"]/table_MAX2["NumCompletedVeh"])*100
y = np.asarray([-2.08516242, -1.6121854, -1.41660136, -0.5236305, -0.46270248, -0.41940242, -0.62490701])
data = np.asarray([x,y]).T
data = smootje(data, n=2)
y_sig_upper = [calculate_significance_interval(y[idx], y_std[idx], alpha, False) for idx in range(0, len(y))]
data2 = np.asarray([x, y_sig_upper]).T
data2 = smootje(data2, n=3)
x,y = smoothie(data[:,0], data[:,1])
plt.plot(x, y, color="gray", label="Vehicle Completion Rate")
x,y = smoothie(data2[:,0], data2[:,1])
plt.plot(x, y, "--", color="gray")


plt.plot([0,0], [0,0], "--", color="black", label=r"Significance ($\alpha=10\%$)")

plt.xlim(50,300)
plt.ylim(-5,0)
plt.xlabel("Inflow Per Entrance [veh/h]")
plt.ylabel("Transportation Efficiency Change [%]")
plt.legend(fontsize=8)













# TLS ANALYIS #################################################################


def loadSwitches(control, traffic_flows):
    count_switches = []
    for flow in traffic_flows:
        file = "../TLS_Analysis/logs/log_tls_states_"+control+"_"+str(flow)+".xml"
        f = open(file, "r")
        content = f.read()
        f.close()
        lines = content.split("\n")
        lines = lines[1:-1]
        lines = lines[5000-3600:]
        
        n_intersections = 9
        n_switches = 0
        for intId in range(0, n_intersections):
            last_state = ""
            for line in lines:
                parts = line.split("\t")
                part = parts[intId]
                if part!=last_state:
                    n_switches+=1
                    last_state = part
                    
        count_switches.append([flow, n_switches/n_intersections])
    return np.asarray(count_switches)

count_switches_FIX = loadSwitches("fixed_programme", traffic_flows)
count_switches_MAX = loadSwitches("max_pressure", traffic_flows)
count_switches_UPP = loadSwitches("priority_pass", traffic_flows)

def loadGreenRedDurations(control, traffic_flows):
    count_durations_G = []
    count_durations_R = []
    for flow in traffic_flows:
        file = "../TLS_Analysis/logs/log_tls_states_"+control+"_"+str(flow)+".xml"
        f = open(file, "r")
        content = f.read()
        f.close()
        lines = content.split("\n")
        lines = lines[1:-1]
        lines = lines[5000-3600:]
        
        n_intersections = 9
        n_signals = 12
    
        for intId in range(0, n_intersections):
            for sigId in range(0, n_signals):
                last_signal = ""
                last_signal_time = 0
                line_ctr = 0
                for line in lines:
                    parts = line.split("\t")
                    part = parts[intId]
                    signal = part[sigId]
                    if last_signal == "":
                        last_signal = signal
                        last_signal_time = line_ctr
                    else:
                        if last_signal != signal:
                            if last_signal=="G":
                                count_durations_G.append([flow, line_ctr-last_signal_time])
                            elif last_signal=="r":
                                count_durations_R.append([flow, line_ctr-last_signal_time])
                            last_signal = signal
                            last_signal_time = line_ctr
                    line_ctr += 1
    count_durations_G = pd.DataFrame(count_durations_G, columns=["flow", "duration"])
    count_durations_R = pd.DataFrame(count_durations_R, columns=["flow", "duration"])
    return count_durations_G, count_durations_R

count_durations_G_FIX, count_durations_R_FIX = loadGreenRedDurations("fixed_programme", traffic_flows)
count_durations_G_MAX, count_durations_R_MAX = loadGreenRedDurations("max_pressure", traffic_flows)
count_durations_G_UPP, count_durations_R_UPP = loadGreenRedDurations("priority_pass", traffic_flows)



# TLS Analysis Switch Frequency
plt.subplot(2,4,7)

data = np.asarray([count_switches_FIX[:,0], count_switches_FIX[:,1]]).T
data = smootje(data, n=2)
x,y = smoothie(data[:,0], data[:,1])
plt.plot(x, y, color="gray", label="Fixed-Cycle")
# plt.plot(count_switches_FIX[:,0], count_switches_FIX[:,1], color="gray", label="Fixed-Cycle")
data = np.asarray([count_switches_MAX[:,0], count_switches_MAX[:,1]]).T
data = smootje(data, n=4)
x,y = smoothie(data[:,0], data[:,1])
plt.plot(x, y, color=blue_color, label="Max-Pressure")
# plt.plot(count_switches_MAX[:,0], count_switches_MAX[:,1], color=blue_color, label="Max-Pressure")
data = np.asarray([count_switches_UPP[:,0], count_switches_UPP[:,1]]).T
data = smootje(data, n=4)
x,y = smoothie(data[:,0], data[:,1])
plt.plot(x, y, "--", color="black", label="Priority Pass")
# plt.plot(count_switches_UPP[:,0], count_switches_UPP[:,1], "--", color="black", label="Priority Pass")
plt.legend(fontsize=8)
plt.ylabel("#Switches Per Intersection Per Hour")
plt.xlabel("Inflow Per Entrance [veh/h]")
plt.xlim(50,300)



# TLS Analysis GR Duration
ax8 = plt.subplot(2,4,8)

# count_durations_G_FIX = count_durations_G_FIX.groupby('flow')['duration'].mean().reset_index()
# plt.plot(count_durations_G_FIX["flow"], count_durations_G_FIX["duration"])

# count_durations_G_MAX = count_durations_G_MAX.groupby('flow')['duration'].mean().reset_index()
# plt.plot(count_durations_G_MAX["flow"], count_durations_G_MAX["duration"])

# count_durations_G_UPP = count_durations_G_UPP.groupby('flow')['duration'].mean().reset_index()
# plt.plot(count_durations_G_UPP["flow"], count_durations_G_UPP["duration"])


ax8.set_xticks([])
ax8.set_yticks([])

# Create two sub-subplots within subplot 8
gs = ax8.get_gridspec()
# gs = plt.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
subgs = gs[1, 3].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.0)

# Upper plot for Green duration
plt.subplot(subgs[0])
count_durations_G_FIX = count_durations_G_FIX.groupby('flow')['duration'].mean().reset_index()
data = np.asarray([count_durations_G_FIX["flow"], count_durations_G_FIX["duration"]]).T
data = smootje(data, n=2)
x,y = smoothie(data[:,0], data[:,1])
plt.plot(x, y, color="gray")
count_durations_G_MAX = count_durations_G_MAX.groupby('flow')['duration'].mean().reset_index()
data = np.asarray([count_durations_G_MAX["flow"], count_durations_G_MAX["duration"]]).T
data = smootje(data, n=3)
x,y = smoothie(data[:,0], data[:,1])
plt.plot(x, y, color=blue_color)
count_durations_G_UPP = count_durations_G_UPP.groupby('flow')['duration'].mean().reset_index()
data = np.asarray([count_durations_G_UPP["flow"], count_durations_G_UPP["duration"]]).T
data = smootje(data, n=3)
x,y = smoothie(data[:,0], data[:,1])
plt.plot(x, y, "--", color="black")
# plt.ylabel('Mean Green Duration')
plt.gca().set_xticks([])
plt.gca().set_facecolor('#e6ffe6')

# Lower plot for Red duration
plt.subplot(subgs[1])
count_durations_R_FIX = count_durations_R_FIX.groupby('flow')['duration'].mean().reset_index()
data = np.asarray([count_durations_R_FIX["flow"], count_durations_R_FIX["duration"]]).T
data = smootje(data, n=2)
x,y = smoothie(data[:,0], data[:,1])
plt.plot(x, y, color="gray")
count_durations_R_MAX = count_durations_R_MAX.groupby('flow')['duration'].mean().reset_index()
data = np.asarray([count_durations_R_MAX["flow"], count_durations_R_MAX["duration"]]).T
data = smootje(data, n=3)
x,y = smoothie(data[:,0], data[:,1])
plt.plot(x, y, color=blue_color)
count_durations_R_UPP = count_durations_R_UPP.groupby('flow')['duration'].mean().reset_index()
data = np.asarray([count_durations_R_UPP["flow"], count_durations_R_UPP["duration"]]).T
data = smootje(data, n=3)
x,y = smoothie(data[:,0], data[:,1])
plt.plot(x, y, "--", color="black")
plt.xlabel('Inflow Per Entrance [veh/h]')
# plt.ylabel('Mean Red Duration')
plt.gca().set_facecolor('#ffe6e6')
# plt.legend()

ax8.set_ylabel("Mean Red & Green Duration [sec]", labelpad=30)

plt.tight_layout()
