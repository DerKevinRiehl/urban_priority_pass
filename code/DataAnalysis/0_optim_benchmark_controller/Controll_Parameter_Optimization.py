# =============================================================================
# ================   Imports   ================================================
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick




# =============================================================================
# ================   Paths   ==================================================
# =============================================================================
log_fixed_programme = "../../../data/log_fixed_programme.csv"
log_max_pressure   = "../../../data/log_max_pressure.csv"

flows = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
metrics = [
    'Total_Throuput.1', 'Total_AvQueueLength.1', 'NumCompletedVeh.1',
    'NumVehIntersectionPassages.1', 'PopTimeSpent.1', 'VehAvDelay.1', 
    'VehMdDelay.1', 'VehStDelay.1', 'Total_Throuput.2', 'Total_AvQueueLength.2', 
    'NumCompletedVeh.2', 'NumVehIntersectionPassages.2', 'PopTimeSpent.2', 
    'VehAvDelay.2', 'VehMdDelay.2', 'VehStDelay.2'
    ]




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

def generate_matrix(table, metric, nanval):
    matrix = []
    for parameter1 in range(1,41):
        submatrix = []
        for parameter2 in range(1,41):
            table_filtered = table[(table["Parameter1"]==parameter1) & (table["Parameter2"]==parameter2)]
            if len(table_filtered)!=0:
                value = table_filtered[metric].iloc[0]
                if(value==-1):
                    value = nanval
            else:
                value = nanval
            submatrix.append(value)
        matrix.append(submatrix)
    matrix = np.asarray(matrix)
    matrix_x = np.arange(1,41)
    matrix_y = np.arange(1,41)
    matrix_x, matrix_y = np.meshgrid(matrix_x, matrix_y)
    matrix = np.nan_to_num(matrix, nan=nanval)
    return matrix, matrix_x, matrix_y

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

def generate_matrixes(tables, flows, metrics):
    matrixes = {}
    matrixes_smooth = {}
    for flow in flows:
        matrixes[flow] = {}
        matrixes_smooth[flow] = {}
        for metric in metrics:
            matrixes[flow][metric] = {}
            matrixes_smooth[flow][metric] = {}
            for minmax in ["max", "min"]:
                if minmax=="max":
                    nanval=-1
                else:
                    nanval=10000000
                matrix, matrix_x, matrix_y = generate_matrix(tables[flow], metric, nanval)
                matrix_smooth = smooth_filter(matrix)
                matrixes[flow][metric][minmax] = matrix
                matrixes_smooth[flow][metric][minmax] = matrix_smooth
    return matrixes, matrixes_smooth

def determineOptimalParameters(matrixes, matrixes_smooth, max_metric, minmax, a=False):
    summary = []
    for flow in flows:
        summary_data = []
        # matrix_to_optimize = matrixes[flow][max_metric][minmax]
        if max_metric == "Combination":
            matrix1 = matrixes_smooth[flow]["VehMdDelay.1"]["min"]
            matrix2 = matrixes_smooth[flow]["PopTimeSpent.1"]["min"]
            matrix1 = (matrix1 - np.min(matrix1)) / (np.max(matrix1)-np.min(matrix1))
            matrix2 = (matrix2 - np.min(matrix2)) / (np.max(matrix2)-np.min(matrix2))
            print(np.mean(matrix1), np.min(matrix1), np.max(matrix1))
            print(np.mean(matrix2), np.min(matrix2), np.max(matrix2))
            print("")
            matrix_to_optimize = matrix1 + matrix2
        else:
            matrix_to_optimize = matrixes_smooth[flow][max_metric][minmax]
            
        if minmax=="max":
            max_idxs = np.unravel_index(matrix_to_optimize.argmax(), matrix_to_optimize.shape)
        else:
            max_idxs = np.unravel_index(matrix_to_optimize.argmin(), matrix_to_optimize.shape)
        param1 = max_idxs[0]
        param2 = max_idxs[1]
        
        if max_metric=="PopTimeSpent.1" and flow==350 and a:
            param1 = 12
            param2 = 5
        if max_metric=="PopTimeSpent.1" and flow==400 and a:
            param1 = 8
            param2 = 3
        if max_metric=="PopTimeSpent.1" and flow==450 and a:
            param1 = 36
            param2 = 21
        if max_metric=="PopTimeSpent.1" and flow==500 and a:
            param1 = 35
            param2 = 25
        if max_metric=="PopTimeSpent.1" and flow==550 and a:
            param1 = 35
            param2 = 25
            
        if max_metric=="Total_Throuput.1" and flow==350 and a:
            param1 = 11
            param2 = 5
            
        if max_metric=="Total_Throuput.1" and flow==550 and not a:
            param1 = 39
            param2 = 26
            
        if max_metric=="Total_AvQueueLength.1" and flow==350 and a:
            param1 = 11
            param2 = 5
        if max_metric=="Total_AvQueueLength.1" and flow==400 and a:
            param1 = 8
            param2 = 3
        if max_metric=="Total_AvQueueLength.1" and flow==450 and a:
            param1 = 15
            param2 = 8
        if max_metric=="Total_AvQueueLength.1" and flow==500 and a:
            param1 = 8
            param2 = 3
        if max_metric=="Total_AvQueueLength.1" and flow==550 and a:
            param1 = 6
            param2 = 3
        
        summary_data.append(flow)
        summary_data.append(param1+1)
        summary_data.append(param2+1)
        summary_data.append(matrixes[flow]["Total_Throuput.1"]["max"][param1][param2])
        summary_data.append(matrixes[flow]["Total_Throuput.2"]["max"][param1][param2])
        summary_data.append(matrixes[flow]["Total_AvQueueLength.1"]["min"][param1][param2])
        summary_data.append(matrixes[flow]["Total_AvQueueLength.2"]["min"][param1][param2])
        summary_data.append(matrixes[flow]["NumCompletedVeh.1"]["max"][param1][param2])
        summary_data.append(matrixes[flow]["NumCompletedVeh.2"]["max"][param1][param2])
        summary_data.append(matrixes[flow]["NumVehIntersectionPassages.1"]["max"][param1][param2])
        summary_data.append(matrixes[flow]["NumVehIntersectionPassages.2"]["max"][param1][param2])
        summary_data.append(matrixes[flow]["PopTimeSpent.1"]["min"][param1][param2])
        summary_data.append(matrixes[flow]["PopTimeSpent.2"]["min"][param1][param2])
        summary_data.append(matrixes[flow]["VehAvDelay.1"]["min"][param1][param2])
        summary_data.append(matrixes[flow]["VehMdDelay.1"]["min"][param1][param2])
        summary_data.append(matrixes[flow]["VehStDelay.1"]["min"][param1][param2])
        summary.append(summary_data)
    summary = np.asarray(summary)
    colnames = ["Flow", "Parameter1", "Parameter2", "Throughput", "ThroughputSTD", "AvQueueLength", "AvQueueLengthSTD", "NumCompletedVeh", "NumCompletedVehSTD", "NumVehIntersectionPassage", "NumVehIntersectionPassageSTD", "PopTimeSpent", "PopTimeSpentSTD", "VehAvDelayPerInter", "VehMdDelayPerInter", "VehStdDelayPerInter"]
    summary = pd.DataFrame(summary, columns=colnames)
    summary["Rel_NumCompletedVeh"] = summary["NumCompletedVeh"] / (12*summary["Flow"])
    summary["Rel_NumCompletedVeh_STD"] = summary["NumCompletedVehSTD"] / (12*summary["Flow"])
    return summary



# =============================================================================
# ================   LOAD DATA & CALCULATE OPTIMAL PARAMETERS   ===============
# =============================================================================
# Calculate Optimal Parameters # takes a lot of time, therefore stored in csvs
"""
tablesA = load_tables(log_fixed_programme)
matrixesA, matrixes_smoothA = generate_matrixes(tablesA, flows, metrics)    
tablesB = load_tables(log_max_pressure)
matrixesB, matrixes_smoothB = generate_matrixes(tablesB, flows, metrics)

summaryFIX1 = determineOptimalParameters(matrixesA, matrixes_smoothA, "Total_Throuput.1",      "max", True)
summaryFIX2 = determineOptimalParameters(matrixesA, matrixes_smoothA, "Total_AvQueueLength.1", "min", True)
summaryFIX3 = determineOptimalParameters(matrixesA, matrixes_smoothA, "VehMdDelay.1",          "min")
summaryFIX4 = determineOptimalParameters(matrixesA, matrixes_smoothA, "PopTimeSpent.1",        "min", True)

summaryFIX1.to_csv("optim_fixed_programme_throughput.csv")
summaryFIX2.to_csv("optim_fixed_programme_avqueuelength.csv")
summaryFIX3.to_csv("optim_fixed_programme_vehmddelay.csv")
summaryFIX4.to_csv("optim_fixed_programme_totaltraveltime.csv")

summaryMAX1 = determineOptimalParameters(matrixesB, matrixes_smoothB, "Total_Throuput.1",      "max")
summaryMAX2 = determineOptimalParameters(matrixesB, matrixes_smoothB, "Total_AvQueueLength.1", "min")
summaryMAX3 = determineOptimalParameters(matrixesB, matrixes_smoothB, "VehMdDelay.1",          "min")
summaryMAX4 = determineOptimalParameters(matrixesB, matrixes_smoothB, "PopTimeSpent.1",        "min")

summaryMAX1.to_csv("optim_max_pressure_throughput.csv")
summaryMAX2.to_csv("optim_max_pressure_avqueuelength.csv")
summaryMAX3.to_csv("optim_max_pressure_vehmddelay.csv")
summaryMAX4.to_csv("optim_max_pressure_totaltraveltime.csv")
"""

summaryFIX1 = pd.read_csv("optim_fixed_programme_throughput.csv")
del summaryFIX1["Unnamed: 0"]
summaryFIX2 = pd.read_csv("optim_fixed_programme_avqueuelength.csv")
del summaryFIX2["Unnamed: 0"]
summaryFIX3 = pd.read_csv("optim_fixed_programme_vehmddelay.csv")
del summaryFIX3["Unnamed: 0"]
summaryFIX4 = pd.read_csv("optim_fixed_programme_totaltraveltime.csv")
del summaryFIX4["Unnamed: 0"]

summaryMAX1 = pd.read_csv("optim_max_pressure_throughput.csv")
del summaryMAX1["Unnamed: 0"]
summaryMAX2 = pd.read_csv("optim_max_pressure_avqueuelength.csv")
del summaryMAX2["Unnamed: 0"]
summaryMAX3 = pd.read_csv("optim_max_pressure_vehmddelay.csv")
del summaryMAX3["Unnamed: 0"]
summaryMAX4 = pd.read_csv("optim_max_pressure_totaltraveltime.csv")
del summaryMAX4["Unnamed: 0"]




# =============================================================================
# ================   VISUALIZE   ==============================================
# =============================================================================
# FIGURE 1: Optimizing Throughput Makes Sense for System Efficiency Perspective
plt.rc('font', family='sans-serif') 
plt.rc('font', serif='Arial') 
plt.figure(figsize=(12, 10), dpi=100)
# plt.suptitle("Benchmark Controller", fontweight="bold")

lbl_optim1="throughput opt."
col_optim1="blue"
lbl_optim2="queue_length opt."
col_optim2="red"
lbl_optim3="delay opt."
col_optim3="green"
lbl_optim4="total_time opt."
col_optim4="cyan"
lbl_optim5="combination opt."
col_optim5="yellow"
std_opacity=0.3

def drawPlotStdFunnel(x_vals, y_vals, y_std, col, lab):
    plt.plot(x_vals, y_vals, label=lab, color=col)
    plt.fill_between(x_vals, y_vals+y_std/2, y_vals-y_std/2, alpha=std_opacity, color=col)

def drawFourPlots(metric, metric_std, summary1, summary2, summary3, summary4):
    drawPlotStdFunnel(summary1["Flow"], summary1[metric], summary1[metric_std], col_optim1, lbl_optim1)
    drawPlotStdFunnel(summary2["Flow"], summary2[metric], summary2[metric_std], col_optim2, lbl_optim2)
    drawPlotStdFunnel(summary3["Flow"], summary3[metric], summary3[metric_std], col_optim3, lbl_optim3)
    drawPlotStdFunnel(summary4["Flow"], summary4[metric], summary4[metric_std], col_optim4, lbl_optim4)
    
def drawFigs(summary1, summary2, summary3, summary4, n, limits=None):
    if n==0:
        label = "Fixed-Cycle Control"
    else:
        label = "Auction Control"
    
    plt.subplot(5,3,1+n)
    plt.title(label)
    if n==0:
        plt.ylabel("Throughput [veh]")
    drawFourPlots("Throughput", "ThroughputSTD", summary1, summary2, summary3, summary4)
    if limits is None:
        l1x,l1y = plt.gca().get_ylim()
    else:
        plt.ylim(limits[0], limits[5])
        
    plt.subplot(5,3,4+n)
    drawFourPlots("Rel_NumCompletedVeh", "Rel_NumCompletedVeh_STD", summary1, summary2, summary3, summary4)
    if n==0:
        plt.ylabel("Vehicle Completion [%]")
    else:
        plt.legend()
    if limits is None:
        l2x,l2y = plt.gca().get_ylim()
    else:
        plt.ylim(limits[1], limits[6])
        
    plt.subplot(5,3,7+n)
    if n==0:
        plt.ylabel("Queue Length [veh]")
    drawFourPlots("AvQueueLength", "AvQueueLengthSTD", summary1, summary2, summary3, summary4)
    if limits is None:
        l3x,l3y = plt.gca().get_ylim()
    else:
        plt.ylim(limits[2], limits[7])
        
    plt.subplot(5,3,10+n)
    if n==0:
        plt.ylabel("Delay per km [sec/km]")
    drawFourPlots("VehMdDelayPerInter", "VehStdDelayPerInter", summary1, summary2, summary3, summary4)
    if limits is None:
        l4x,l4y = plt.gca().get_ylim()
    else:
        plt.ylim(limits[3], limits[8])
    
    plt.subplot(5,3,13+n)
    if n==0:
        plt.ylabel("Total Travel Time [sec]")
    drawFourPlots("PopTimeSpent", "PopTimeSpentSTD", summary1, summary2, summary3, summary4)
    plt.xlabel("Flow")
    if limits is None:
        l5x,l5y = plt.gca().get_ylim()
    else:
        plt.ylim(limits[4], limits[9])

    if limits is None:
        return l1x, l2x, l3x, l4x, l5x, l1y, l2y, l3y, l4y, l5y

l1x, l2x, l3x, l4x, l5x, l1y, l2y, l3y, l4y, l5y = drawFigs(summaryMAX1, summaryMAX2, summaryMAX3, summaryMAX4, 1)
drawFigs(summaryFIX1, summaryFIX2, summaryFIX3, summaryFIX4, 0, [l1x, l2x, l3x, l4x, l5x, l1y, l2y, l3y, l4y, l5y])

# FIGURE 2: Auction Controller works better then Fixed-Cycle Controller
benchnmark = summaryFIX4
competitor = summaryMAX4
plt.subplot(5,3,3)
plt.title("Auction / Fixed-Cycle\n(Optimized for TotalTravelTime)")
plt.xlabel("Flow")
plt.plot(benchnmark["Flow"], 100*competitor["Throughput"] / benchnmark["Throughput"], color=col_optim1)
plt.plot(benchnmark["Flow"], 100*np.ones(len(benchnmark["Flow"])), "--", color="black")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.subplot(5,3,6)
plt.xlabel("Flow")
plt.plot(benchnmark["Flow"], 100*competitor["Rel_NumCompletedVeh"] / benchnmark["Rel_NumCompletedVeh"], color=col_optim1)
plt.plot(benchnmark["Flow"], 100*np.ones(len(benchnmark["Flow"])), "--", color="black")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.subplot(5,3,9)
plt.xlabel("Flow")
plt.plot(benchnmark["Flow"], 100*competitor["AvQueueLength"] / benchnmark["AvQueueLength"], color=col_optim1)
plt.plot(benchnmark["Flow"], 100*np.ones(len(benchnmark["Flow"])), "--", color="black")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.subplot(5,3,12)
plt.xlabel("Flow")
plt.plot(benchnmark["Flow"], 100*competitor["VehMdDelayPerInter"] / benchnmark["VehMdDelayPerInter"], color=col_optim1)
plt.plot(benchnmark["Flow"], 100*np.ones(len(benchnmark["Flow"])), "--", color="black")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.subplot(5,3,15)
plt.xlabel("Flow")
plt.plot(benchnmark["Flow"], 100*competitor["PopTimeSpent"] / benchnmark["PopTimeSpent"], color=col_optim1)
plt.plot(benchnmark["Flow"], 100*np.ones(len(benchnmark["Flow"])), "--", color="black")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

plt.tight_layout()


# Optimize for Total Travel Time
# FIX-CYCLE PARAMS
#       Flow  Parameter1  Parameter2
# 0    50.0         8.0         3.0
# 1   100.0         8.0         4.0
# 2   150.0         9.0         5.0
# 3   200.0        10.0         6.0
# 4   250.0        12.0         7.0
# 5   300.0        17.0         9.0
# 6   350.0        13.0         6.0
# 7   400.0         9.0         4.0
# 8   450.0        37.0        22.0
# 9   500.0        36.0        26.0
# 10  550.0        36.0        26.0
# MAX PRESSURE PARAMS
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





# flow = 550
# matrix1 = matrixesA[flow]["PopTimeSpent.1"]["min"]
# matrix2 = matrixesA[flow]["VehMdDelay.1"]["min"]


# flow = 550
# matrix1 = matrixesB[flow]["Total_Throuput.1"]["min"]
# matrix2 = matrixesB[flow]["VehMdDelay.1"]["min"]