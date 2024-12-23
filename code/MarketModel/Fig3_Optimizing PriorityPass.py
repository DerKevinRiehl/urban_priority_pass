# =============================================================================
# ================   Imports   ================================================
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from Data_UrgencyDistribution import urgency_distribution
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import interp1d




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
    for x in range(0,n):
        data = np.column_stack((data[:, 0], np.pad(data[:, 1], (1, 1), mode='edge')[1:-1] * 0.25 + data[:, 1] * 0.5 + np.pad(data[:, 1], (1, 1), mode='edge')[2:] * 0.25))
    return data


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



# =============================================================================
# ================   Load Data   ==============================================
# =============================================================================

blue_color = (30/255, 111/255, 192/255)

# Load Priority Pass Experiment Results
summaryMAX4 = pd.read_csv(log_max_pressure)
del summaryMAX4["Unnamed: 0"]
tablesUPP = load_tables(log_priority_pass)
calculateDelayChanges(tablesUPP, summaryMAX4, 1, 1)

# Prepare Population
df_population = pd.DataFrame(np.asarray([urgency_distribution[scenario]["pop_route_distances"], 
                              urgency_distribution[scenario]["pop_route_durations"], 
                              urgency_distribution[scenario]["pop_salary"],
                              urgency_distribution[scenario]["pop_urgency_levels"],
                              urgency_distribution[scenario]["pop_urgency_vots"]]).T, 
                             columns=["route_distance", "route_duration", "salary", "urgency_level", "vot"])
df_population = df_population[df_population["salary"]>MINIMUM_SALARY_THRESHOLD]




# =============================================================================
# ================   VISUALIZE = ==============================================
# =============================================================================
plt.rc('font', family='Times New Roman') 
plt.figure(figsize=(12, 6), dpi=100)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# plt.suptitle("Inflow Per Entrance = 200 [veh/h]")


# =============================================================================
# ================  FIGURE 1: PRICE - DEMAND - CURVE [Flow=150] ===============
# =============================================================================
plt.subplot(2,4,1)

def generateData(tablesUPP, flow, tau, gamma):
    upp_relevant = tablesUPP[flow]
    upp_relevant = upp_relevant[["Parameter1", "Parameter2", 
                                 'change_delay_upp', 'change_delay_npp', 'change_delay_upp_STD', 'change_delay_npp_STD',]]
    upp_relevant = upp_relevant[(upp_relevant["Parameter1"]==gamma) & (upp_relevant["Parameter2"]==tau)]
    
    # df_population["upp_delay"] = (df_population["route_distance"]/1000) * np.random.normal(upp_relevant["change_delay_upp"].iloc[0], upp_relevant["change_delay_upp_STD"].iloc[0]/10, size=len(df_population))
    df_population["upp_exp_delay"] = (df_population["route_distance"]/1000) * upp_relevant["change_delay_upp"].iloc[0]
    df_population["npp_exp_delay"] = (df_population["route_distance"]/1000) * upp_relevant["change_delay_npp"].iloc[0]
    df_population["exp_time_save"] = df_population["npp_exp_delay"] - df_population["upp_exp_delay"]
    df_population["exp_cost_save"] = df_population["exp_time_save"] * df_population["vot"]/3600
    
    # Determine optimal price
    df_sorted = df_population.sort_values(by='exp_cost_save', ascending=False)
    df_sorted = df_sorted.reset_index()
    df_sorted = df_sorted.sort_values(by='exp_cost_save', ascending=False)
    optimal_price = df_sorted["exp_cost_save"].iloc[int(gamma*len(df_population))]
    
    # Determine price demand curve
    price_demand_curve = []
    for pit in range(0, 100):
        price = 1/100*pit
        df_population["buy_decision"] = df_population["exp_cost_save"] > price
        share_willing_to_buy = sum(df_population["buy_decision"])/len(df_population["buy_decision"])
        price_demand_curve.append([price, share_willing_to_buy])
    price_demand_curve = np.asarray(price_demand_curve)
    return price_demand_curve, optimal_price, gamma

flow = 250
tau = 0.5
gamma = 0.2

price_demand_curveA, optimal_priceA, gammaA = generateData(tablesUPP, flow, tau, gamma)
plt.plot(price_demand_curveA[:,0], price_demand_curveA[:,1]*100, color=blue_color)
plt.plot([optimal_priceA, optimal_priceA], [0, gammaA*100], "--", color="black")
plt.plot([0, optimal_priceA], [gammaA*100, gammaA*100], "--", color="black")
plt.text(optimal_priceA, gammaA*100,r"(p*, $\gamma$)", fontsize=10, fontweight='normal', verticalalignment='bottom', horizontalalignment='left')
# plt.text(0.15, 90, "(Scenario 1, In-Flow="+str(flow)+", "+r"$\tau$=0.5, $\gamma$=0.2)", fontsize=10, fontweight='normal', verticalalignment='bottom', horizontalalignment='left')

plt.title("Price-Demand-Curve")
plt.xlabel("Priority Pass Price [$]")
plt.ylabel("Population's Demand [%]")



# =============================================================================
# ================  FIGURE 2: price heatmap ===================================
# =============================================================================
plt.subplot(2,4,2)

def generateData3(tablesUPP, flow, taus, gammas):
    upp_relevantX = tablesUPP[flow]
    upp_relevantX = upp_relevantX[["Parameter1", "Parameter2", 
                                 'change_delay_upp', 'change_delay_npp', 'change_delay_upp_STD', 'change_delay_npp_STD',]]
    data = []
    for tau in taus:
        rows = []
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
            
            rows.append(optimal_price)
        data.append(rows)
    data = np.asarray(data)
    data = smooth_filter(data)
    data = smooth_filter(data)
    return data

matrix = generateData3(tablesUPP, flow, tau_controls, gamma_shares)
plt.imshow(np.flip(matrix, axis=0), cmap="Grays")
cbar = plt.colorbar(location="top", orientation="horizontal", fraction=0.05, shrink=0.8, pad=0.12)
cbar.ax.tick_params(rotation=90)
#formatting
plt.gca().set_yticks(np.arange(len(tau_controls)), labels=np.asarray(np.flip(tau_controls)*100, dtype=int))
plt.gca().set_xticks(np.arange(len(gamma_shares)), labels=np.asarray(np.asarray(gamma_shares)*100, dtype=int))

plt.xlabel("Entitled Share $\gamma$ [%]")
plt.ylabel(r"Threshold $\tau$ [%]")
plt.title("Optimal Priority Pass Price [$]")


# =============================================================================
# ================  FIGURE 3: cr benefit x tau ================================
# =============================================================================
plt.subplot(2,4,3)

def generateData2(taus, tablesUPP, flow, gamma, pay=True, redist=False):
    upp_relevantX = tablesUPP[flow]
    upp_relevantX = upp_relevantX[["Parameter1", "Parameter2", 
                                 'change_delay_upp', 'change_delay_npp', 'change_delay_upp_STD', 'change_delay_npp_STD',]]
    
    data = []
    for tau in taus:
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
        
        if pay:
            if not redist:
                df_population['actual_time_costs'] = -df_population['actual_delay'] * df_population["vot"]/3600 - df_population['actual_price']
            else:
                df_population['actual_time_costs'] = -df_population['actual_delay'] * df_population["vot"]/3600 - df_population['actual_price'] + df_population['actual_reciv']
        else:
            df_population['actual_time_costs'] = -df_population['actual_delay'] * df_population["vot"]/3600 
        benefit = np.mean(df_population['actual_time_costs'])
        data.append([tau, benefit])
        
    data = np.asarray(data)
    data = np.column_stack((data[:, 0], np.pad(data[:, 1], (1, 1), mode='edge')[1:-1] * 0.25 + data[:, 1] * 0.5 + np.pad(data[:, 1], (1, 1), mode='edge')[2:] * 0.25))
    data = np.column_stack((data[:, 0], np.pad(data[:, 1], (1, 1), mode='edge')[1:-1] * 0.25 + data[:, 1] * 0.5 + np.pad(data[:, 1], (1, 1), mode='edge')[2:] * 0.25))
    data = np.column_stack((data[:, 0], np.pad(data[:, 1], (1, 1), mode='edge')[1:-1] * 0.25 + data[:, 1] * 0.5 + np.pad(data[:, 1], (1, 1), mode='edge')[2:] * 0.25))
    data = np.column_stack((data[:, 0], np.pad(data[:, 1], (1, 1), mode='edge')[1:-1] * 0.25 + data[:, 1] * 0.5 + np.pad(data[:, 1], (1, 1), mode='edge')[2:] * 0.25))
    data = np.column_stack((data[:, 0], np.pad(data[:, 1], (1, 1), mode='edge')[1:-1] * 0.25 + data[:, 1] * 0.5 + np.pad(data[:, 1], (1, 1), mode='edge')[2:] * 0.25))

    return np.asarray(data)
"""
benefit_curveA = generateData2(tau_controls, tablesUPP, flow="+str(flow)+", gamma=0.1)
benefit_curveB = generateData2(tau_controls, tablesUPP, flow="+str(flow)+", gamma=0.2)
benefit_curveC = generateData2(tau_controls, tablesUPP, flow="+str(flow)+", gamma=0.5)
plt.plot(benefit_curveA[:,0], benefit_curveA[:,1], color=blue_color)
plt.plot(benefit_curveB[:,0], benefit_curveB[:,1], color="gray")
plt.plot(benefit_curveC[:,0], benefit_curveC[:,1], color="black")
"""

benefit_curveA = generateData2(tau_controls, tablesUPP, flow, gamma=0.2, pay=False)
benefit_curveB = generateData2(tau_controls, tablesUPP, flow, gamma=0.2, pay=True)
benefit_curveC = generateData2(tau_controls, tablesUPP, flow, gamma=0.2, pay=True, redist=True)
plt.plot(benefit_curveA[:,0], benefit_curveA[:,1], color="gray", label="no payment")
plt.plot(benefit_curveB[:,0], benefit_curveB[:,1], color=blue_color, label="payment")
plt.plot(benefit_curveC[:,0], benefit_curveC[:,1], "--", color="black", label="payment redistributed*")
# plt.text(0.05, 0.1375, r"(Scenario 1, In-Flow="+str(flow)+", $\gamma$=0.2)", fontsize=10, fontweight='normal', verticalalignment='bottom', horizontalalignment='left')

plt.legend()
plt.ylim(-0.1,0.17)
plt.title("Average User Benefit ($\gamma$=0.2)")
plt.xlabel(r"Threshold $\tau$ [%]")
plt.ylabel("c r [$/km]")


# =============================================================================
# ================  FIGURE 4: benefit heatmap  ================================
# =============================================================================
plt.subplot(2,4,4)

def generateData3(tablesUPP, flow, taus, gammas, df_population):
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

matrix, prices, matrixSys = generateData3(tablesUPP, flow, tau_controls, gamma_shares, df_population)
plt.imshow(np.flip(matrix, axis=0), cmap="Grays")
cbar = plt.colorbar(location="top", orientation="horizontal", fraction=0.05, shrink=0.8, pad=0.12)
cbar.ax.tick_params(rotation=90)
max_row, max_col = np.unravel_index(np.argmax(matrix), matrix.shape)
plt.scatter(max_col, matrix.shape[0] - 1 - max_row, color=blue_color, s=50, zorder=2)
#formatting
plt.gca().set_yticks(np.arange(len(tau_controls)), labels=np.asarray(np.flip(tau_controls)*100, dtype=int))
plt.gca().set_xticks(np.arange(len(gamma_shares)), labels=np.asarray(np.asarray(gamma_shares)*100, dtype=int))

plt.xlabel("Entitled Share $\gamma$ [%]")
plt.ylabel(r"Threshold $\tau$ [%]")
plt.title("Average User Benefit [$/km]")


plt.subplot(2,4,8)

plt.imshow(np.flip(matrixSys, axis=0), cmap="Grays")
cbar = plt.colorbar(location="top", orientation="horizontal", fraction=0.05, shrink=0.8, pad=0.12)
cbar.ax.tick_params(rotation=90)
max_row, max_col = np.unravel_index(np.argmax(matrixSys), matrix.shape)
plt.scatter(max_col, matrix.shape[0] - 1 - max_row, color=blue_color, s=50, zorder=2)
plt.gca().set_yticks(np.arange(len(tau_controls)), labels=np.asarray(np.flip(tau_controls)*100, dtype=int))
plt.gca().set_xticks(np.arange(len(gamma_shares)), labels=np.asarray(np.asarray(gamma_shares)*100, dtype=int))

plt.xlabel("Entitled Share $\gamma$ [%]")
plt.ylabel(r"Threshold $\tau$ [%]")
plt.title("System Benefit [$/h]")


# =============================================================================
# ================  FIGURE 4: optimization over traffic & market scenarios ====
# =============================================================================

flows = [50, 100, 150, 200, 250, 300]

def generateData4(scenario):
    df_population = pd.DataFrame(np.asarray([urgency_distribution[scenario]["pop_route_distances"], 
                                  urgency_distribution[scenario]["pop_route_durations"], 
                                  urgency_distribution[scenario]["pop_salary"],
                                  urgency_distribution[scenario]["pop_urgency_levels"],
                                  urgency_distribution[scenario]["pop_urgency_vots"]]).T, 
                                 columns=["route_distance", "route_duration", "salary", "urgency_level", "vot"])
    df_population = df_population[df_population["salary"]>MINIMUM_SALARY_THRESHOLD]

    data = []
    for flow in flows:
        matrix, prices, matrixSys = generateData3(tablesUPP, flow, tau_controls, gamma_shares, df_population)
        max_row, max_col = np.unravel_index(np.argmax(matrixSys), matrixSys.shape)
        tau_optim = tau_controls[max_row]
        gamma_optim = gamma_shares[max_col]
        price = prices[max_row][max_col]
        benefit = matrix[max_row][max_col]
        benefitSys = matrixSys[max_row][max_col]
        
        data.append([flow, scenario, tau_optim, gamma_optim, price, benefit, benefitSys])
    return np.asarray(data)

optim_1 = generateData4(1)
optim_2 = generateData4(2)
optim_3 = generateData4(3)

plt.subplot(2,4,5)

def format_with_commas(x, p):
    return f"{x:,.0f}"
plt.title("Optimal System Benefit [$/h]")

def smoothie(x,y):
    smooth_interp = lambda x, y, steps=100: (np.linspace(x.min(), x.max(), steps), interp1d(x, y, kind='cubic')(np.linspace(x.min(), x.max(), steps)))
    x_smooth, y_smooth = smooth_interp(x, y)
    return x_smooth, y_smooth

x_smooth, y_smooth = smoothie(optim_1[:,0], smootje(optim_1[:,[0,6]], n=3)[:,1])
plt.plot(x_smooth, y_smooth, color=blue_color, label="Scenario 1")
# plt.plot(optim_1[:,0], smootje(optim_1[:,[0,6]], n=3)[:,1], color=blue_color, label="Scenario 1")
x_smooth, y_smooth = smoothie(optim_2[:,0], smootje(optim_2[:,[0,6]], n=3)[:,1])
plt.plot(x_smooth, y_smooth, color="gray", label="Scenario 2")
# plt.plot(optim_2[:,0], smootje(optim_2[:,[0,6]], n=3)[:,1], color="gray", label="Scenario 2")
x_smooth, y_smooth = smoothie(optim_3[:,0], smootje(optim_3[:,[0,6]], n=3)[:,1])
plt.plot(x_smooth, y_smooth, color="black", label="Scenario 3")
# plt.plot(optim_3[:,0], smootje(optim_3[:,[0,6]], n=3)[:,1], color="black", label="Scenario 3")
plt.xlabel("Inflow Per Entrance [veh/h]")
plt.ylabel("System Benefit [$/h]")
plt.legend()
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_with_commas))

plt.subplot(2,4,6)

plt.title("Optimal Priority Pass Price")
x_smooth, y_smooth = smoothie(optim_1[:,0], smootje(optim_1[:,[0,4]], n=3)[:,1])
plt.plot(x_smooth, y_smooth, color=blue_color, label="Scenario 1")
# plt.plot(optim_1[:,0], smootje(optim_1[:,[0,4]], n=3)[:,1], color=blue_color, label="Scenario 1")
x_smooth, y_smooth = smoothie(optim_2[:,0], smootje(optim_2[:,[0,4]], n=3)[:,1])
plt.plot(x_smooth, y_smooth, color="gray", label="Scenario 2")
# plt.plot(optim_2[:,0], smootje(optim_2[:,[0,4]], n=3)[:,1], color="gray", label="Scenario 2")
x_smooth, y_smooth = smoothie(optim_3[:,0], smootje(optim_3[:,[0,4]], n=3)[:,1])
plt.plot(x_smooth, y_smooth, color="black", label="Scenario 3")
# plt.plot(optim_3[:,0], smootje(optim_3[:,[0,4]], n=3)[:,1], color="black", label="Scenario 3")
plt.xlabel("Inflow Per Entrance [veh/h]")
plt.ylabel("Price [$]")
plt.legend()
plt.legend()

plt.subplot(2,4,7)
plt.title("Optimal Priority Pass Parameter")
x_smooth, y_smooth = smoothie(optim_1[:,0], smootje(optim_1[:,[0,2]], n=3)[:,1])
plt.plot(x_smooth, y_smooth*100, color=blue_color, label=r"$\tau$")
# plt.plot(optim_1[:,0], smootje(optim_1[:,[0,2]], n=3)[:,1]*100, color=blue_color, label=r"$\tau$")
x_smooth, y_smooth = smoothie(optim_1[:,0], smootje(optim_1[:,[0,3]], n=3)[:,1])
plt.plot(x_smooth, y_smooth*100, color="gray", label=r"$\gamma$")
# plt.plot(optim_1[:,0], smootje(optim_1[:,[0,3]], n=3)[:,1]*100, color="gray", label=r"$\gamma$")
plt.xlabel("Inflow Per Entrance [veh/h]")
plt.ylabel("Priority Pass Parameter [%]")
plt.legend()
# plt.grid()

# array([[5.0e+01, 6.0e-01, 3.0e-01],
#        [1.0e+02, 9.0e-01, 4.0e-01],
#        [1.5e+02, 8.0e-01, 3.0e-01],
#        [2.0e+02, 9.0e-01, 3.0e-01],
#        [2.5e+02, 8.0e-01, 2.0e-01],
#        [3.0e+02, 7.0e-01, 1.0e-01]])

# flow,  tau,   gamma
# ---------------------
#  50,   0.60,  0.30
# 100,   0.90,  0.40
# 150    0.80,  0.30
# 200,   0.90,  0.30
# 250,   0.80,  0.20
# 300,   0.70,  0.10


# =============================================================================
# ================  FINALIZE PLOT  ============================================
# =============================================================================

plt.tight_layout()
