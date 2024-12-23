# =============================================================================
# ================   Imports   ================================================
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")




# =============================================================================
# ================   Paths   ==================================================
# =============================================================================
log_fixed_programme = "../../data/log_fixed_programme.csv"
log_max_pressure    = "../../data/log_max_pressure.csv"
log_priority_pass   = "../../data/log_priority_pass.csv"

flows = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
metrics = metrics = [
    'Total_Throuput.1', 'Total_AvQueueLength.1', 'NumCompletedVeh.1',
    'NumVehIntersectionPassages.1', 'PopTimeSpent.1', 'VehAvDelay.1', 
    'VehMdDelay.1', 'VehStDelay.1', 'Total_Throuput.2', 'Total_AvQueueLength.2', 
    'NumCompletedVeh.2', 'NumVehIntersectionPassages.2', 'PopTimeSpent.2', 
    'VehAvDelay.2', 'VehMdDelay.2', 'VehStDelay.2'
    ]


blue_color = (30/255, 111/255, 192/255)


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
        
        tables[flow]["delay_benefit_per_vehicle"] = -tables[flow]["Parameter1"]*tables[flow]["change_delay_upp"]*vot_upp - (1-tables[flow]["Parameter1"])*tables[flow]["change_delay_npp"]*vot_npp
        # tables[flow]["total_benefit"] = tables[flow]["delay_benefit"] * (tables[flow]["change_throughput"]+tables[flow]["benchmark_throughput"])/tables[flow]["benchmark_throughput"]
    
        
# Calculate Benchmark
summaryFIX4 = pd.read_csv("0_optim_benchmark_controller/optim_fixed_programme_totaltraveltime.csv")
del summaryFIX4["Unnamed: 0"]
summaryMAX4 = pd.read_csv("0_optim_benchmark_controller/optim_max_pressure_totaltraveltime.csv")
del summaryMAX4["Unnamed: 0"]

# Calculate UPP
tablesUPP = load_tables(log_priority_pass)
calculateDelayChanges(tablesUPP, summaryMAX4, 1, 1)



# *****************************************************************************
# ************ Figure 1
# *****************************************************************************

plt.rc('font', family='Times New Roman') 
plt.figure(figsize=(12, 6), dpi=100)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42



selected_flow = 100

plt.subplot(2,4,1)
# data
table = tablesUPP[selected_flow]
cols_to_average = ['change_delay_upp_rel', 'change_delay_npp_rel',
'change_delay_upp_rel_STD', 'change_delay_npp_rel_STD',]
table = table[["Parameter1", "Parameter2", *cols_to_average ]]
table = table.groupby('Parameter1')[cols_to_average].mean()
table = table.reset_index()
# entitled
color = [30/255,111/255,192/255,1]
label = "Entitled"
xm = table["Parameter1"]
ym = table["change_delay_upp_rel"]
ys = np.abs(table["change_delay_upp_rel_STD"])
plt.plot(xm*100, ym, color=color, label=label)
plt.fill_between(xm*100, (ym-ys), (ym+ys), facecolor=color, alpha=0.1)
# not-entitled
color = "gray"
label = "Not-Entitled"
xm = table["Parameter1"]
ym = table["change_delay_npp_rel"]
ys = np.abs(table["change_delay_npp_rel_STD"])
plt.plot(xm*100,ym, color=color, label=label)
plt.fill_between(xm*100, (ym-ys), (ym+ys), facecolor=color, alpha=0.1)
# formatting
plt.title("Scenario 1: In-Flow "+str(selected_flow)+" [veh/h]", fontweight="bold")
plt.plot([0, 95], [0, 0], "--", color="black")
plt.xlabel(r"Entitled Share $\gamma$ [%]")
plt.ylabel("Delay Change [%]")
plt.legend(loc="upper left")
plt.ylim(-50, 60)
plt.grid()


plt.subplot(2,4,2)
# data
table = tablesUPP[selected_flow]
cols_to_average = ['change_delay_upp_rel', 'change_delay_npp_rel',
'change_delay_upp_rel_STD', 'change_delay_npp_rel_STD',]
table = table[["Parameter1", "Parameter2", *cols_to_average ]]
table = table.groupby('Parameter2')[cols_to_average].mean()
table = table.reset_index()
# entitled
color = [30/255,111/255,192/255,1]
label = "Entitled"
xm = table["Parameter2"]
ym = table["change_delay_upp_rel"]
ys = np.abs(table["change_delay_upp_rel_STD"])
plt.plot(xm*100, ym, color=color, label=label)
plt.fill_between(xm*100, (ym-ys), (ym+ys), facecolor=color, alpha=0.1)
# not-entitled
color = "gray"
label = "Not-Entitled"
xm = table["Parameter2"]
ym = table["change_delay_npp_rel"]
ys = np.abs(table["change_delay_npp_rel_STD"])
plt.plot(xm*100,ym, color=color, label=label)
plt.fill_between(xm*100, (ym-ys), (ym+ys), facecolor=color, alpha=0.1)
# formatting
plt.plot([0, 95], [0, 0], "--", color="black")
plt.xlabel(r"Threshold $\tau$ [%]")
plt.ylabel("Delay Change [%]")
plt.legend(loc="upper left")
plt.ylim(-50, 60)
plt.grid()

plt.subplot(2,4,3)
# data
table = tablesUPP[selected_flow]
cols_to_average = ['change_delay_upp', 'change_delay_npp']
table = table[["Parameter1", "Parameter2", *cols_to_average ]]
vot_upp = 2
vot_npp = 1
table["benefit"] = table["Parameter1"]*(-table["change_delay_upp"])*vot_upp + (1-table["Parameter1"])*(-table["change_delay_npp"])*vot_npp
matrix = table.pivot(index='Parameter2', columns='Parameter1', values='benefit')
matrix = matrix.sort_index().sort_index(axis=1)
matrix = np.asarray(matrix)
matrix = smooth_filter(matrix)
share_upps = (table["Parameter1"].unique()*100).tolist()
thresholds = (table["Parameter2"].unique()*100).tolist()
plt.imshow(np.flip(matrix, axis=0), cmap="Grays")
#formatting
plt.gca().set_yticks(np.arange(len(thresholds)), labels=np.asarray(np.flip(thresholds), dtype=int))
plt.gca().set_xticks(np.arange(len(share_upps)), labels=np.asarray(np.asarray(share_upps), dtype=int))
cbar = plt.colorbar(location="top", orientation="horizontal", fraction=0.05, shrink=0.8, pad=0.12)
cbar.ax.tick_params(rotation=90)
plt.xlabel("Entitled Share $\gamma$ [%]")
plt.ylabel(r"Threshold $\tau$ [%]")
plt.text(0.05, 0.95, '2x', transform=plt.gca().transAxes, 
         color=blue_color, fontsize=24, fontweight='bold', 
         verticalalignment='top', horizontalalignment='left')
plt.title("Average Driver Benefit $c_r$ ")

plt.subplot(2,4,4)
# data
table = tablesUPP[selected_flow]
cols_to_average = ['change_delay_upp', 'change_delay_npp']
table = table[["Parameter1", "Parameter2", *cols_to_average ]]
vot_upp = 4
vot_npp = 1
table["benefit"] = table["Parameter1"]*(-table["change_delay_upp"])*vot_upp + (1-table["Parameter1"])*(-table["change_delay_npp"])*vot_npp
matrix = table.pivot(index='Parameter2', columns='Parameter1', values='benefit')
matrix = matrix.sort_index().sort_index(axis=1)
matrix = np.asarray(matrix)
matrix = smooth_filter(matrix)
share_upps = (table["Parameter1"].unique()*100).tolist()
thresholds = (table["Parameter2"].unique()*100).tolist()
plt.imshow(np.flip(matrix, axis=0), cmap="Grays")
#formatting
plt.gca().set_yticks(np.arange(len(thresholds)), labels=np.asarray(np.flip(thresholds), dtype=int))
plt.gca().set_xticks(np.arange(len(share_upps)), labels=np.asarray(np.asarray(share_upps), dtype=int))
cbar = plt.colorbar(location="top", orientation="horizontal", fraction=0.05, shrink=0.8, pad=0.12)
cbar.ax.tick_params(rotation=90)
plt.xlabel("Entitled Share $\gamma$ [%]")
plt.ylabel(r"Threshold $\tau$ [%]")
plt.text(0.05, 0.95, '4x', transform=plt.gca().transAxes, 
         color=blue_color, fontsize=24, fontweight='bold', 
         verticalalignment='top', horizontalalignment='left')
plt.title("Average Driver Benefit $c_r$ ")

selected_flow = 250

plt.subplot(2,4,5)
# data
table = tablesUPP[selected_flow]
cols_to_average = ['change_delay_upp_rel', 'change_delay_npp_rel',
'change_delay_upp_rel_STD', 'change_delay_npp_rel_STD',]
table = table[["Parameter1", "Parameter2", *cols_to_average ]]
table = table.groupby('Parameter1')[cols_to_average].mean()
table = table.reset_index()
# entitled
color = [30/255,111/255,192/255,1]
label = "Entitled"
xm = table["Parameter1"]
ym = table["change_delay_upp_rel"]
ys = np.abs(table["change_delay_upp_rel_STD"])
plt.plot(xm*100, ym, color=color, label=label)
plt.fill_between(xm*100, (ym-ys), (ym+ys), facecolor=color, alpha=0.1)
# not-entitled
color = "gray"
label = "Not-Entitled"
xm = table["Parameter1"]
ym = table["change_delay_npp_rel"]
ys = np.abs(table["change_delay_npp_rel_STD"])
plt.plot(xm*100,ym, color=color, label=label)
plt.fill_between(xm*100, (ym-ys), (ym+ys), facecolor=color, alpha=0.1)
# formatting
plt.title("Scenario 2: In-Flow "+str(selected_flow)+" [veh/h]", fontweight="bold")
plt.plot([0, 95], [0, 0], "--", color="black")
plt.xlabel(r"Entitled Share $\gamma$ [%]")
plt.ylabel("Delay Change [%]")
plt.legend(loc="upper left")
plt.ylim(-50, 60)
plt.grid()


plt.subplot(2,4,6)
# data
table = tablesUPP[selected_flow]
cols_to_average = ['change_delay_upp_rel', 'change_delay_npp_rel',
'change_delay_upp_rel_STD', 'change_delay_npp_rel_STD',]
table = table[["Parameter1", "Parameter2", *cols_to_average ]]
table = table.groupby('Parameter2')[cols_to_average].mean()
table = table.reset_index()
# entitled
color = [30/255,111/255,192/255,1]
label = "Entitled"
xm = table["Parameter2"]
ym = table["change_delay_upp_rel"]
ys = np.abs(table["change_delay_upp_rel_STD"])
plt.plot(xm*100, ym, color=color, label=label)
plt.fill_between(xm*100, (ym-ys), (ym+ys), facecolor=color, alpha=0.1)
# not-entitled
color = "gray"
label = "Not-Entitled"
xm = table["Parameter2"]
ym = table["change_delay_npp_rel"]
ys = np.abs(table["change_delay_npp_rel_STD"])
plt.plot(xm*100,ym, color=color, label=label)
plt.fill_between(xm*100, (ym-ys), (ym+ys), facecolor=color, alpha=0.1)
# formatting
plt.plot([0, 95], [0, 0], "--", color="black")
plt.xlabel(r"Threshold $\tau$ [%]")
plt.ylabel("Delay Change [%]")
plt.legend(loc="upper left")
plt.ylim(-50, 60)
plt.grid()

plt.subplot(2,4,7)
# data
table = tablesUPP[selected_flow]
cols_to_average = ['change_delay_upp', 'change_delay_npp']
table = table[["Parameter1", "Parameter2", *cols_to_average ]]
vot_upp = 2
vot_npp = 1
table["benefit"] = table["Parameter1"]*(-table["change_delay_upp"])*vot_upp + (1-table["Parameter1"])*(-table["change_delay_npp"])*vot_npp
matrix = table.pivot(index='Parameter2', columns='Parameter1', values='benefit')
matrix = matrix.sort_index().sort_index(axis=1)
matrix = np.asarray(matrix)
matrix = smooth_filter(matrix)
share_upps = (table["Parameter1"].unique()*100).tolist()
thresholds = (table["Parameter2"].unique()*100).tolist()
plt.imshow(np.flip(matrix, axis=0), cmap="Grays")
#formatting
plt.gca().set_yticks(np.arange(len(thresholds)), labels=np.asarray(np.flip(thresholds), dtype=int))
plt.gca().set_xticks(np.arange(len(share_upps)), labels=np.asarray(np.asarray(share_upps), dtype=int))
cbar = plt.colorbar(location="top", orientation="horizontal", fraction=0.05, shrink=0.8, pad=0.12)
cbar.ax.tick_params(rotation=90)
plt.xlabel("Entitled Share $\gamma$ [%]")
plt.ylabel(r"Threshold $\tau$ [%]")
plt.text(0.05, 0.95, '2x', transform=plt.gca().transAxes, 
         color=blue_color, fontsize=24, fontweight='bold', 
         verticalalignment='top', horizontalalignment='left')
plt.title("Average Driver Benefit $c_r$ ")

plt.subplot(2,4,8)
# data
table = tablesUPP[selected_flow]
cols_to_average = ['change_delay_upp', 'change_delay_npp']
table = table[["Parameter1", "Parameter2", *cols_to_average ]]
vot_upp = 4
vot_npp = 1
table["benefit"] = table["Parameter1"]*(-table["change_delay_upp"])*vot_upp + (1-table["Parameter1"])*(-table["change_delay_npp"])*vot_npp
matrix = table.pivot(index='Parameter2', columns='Parameter1', values='benefit')
matrix = matrix.sort_index().sort_index(axis=1)
matrix = np.asarray(matrix)
matrix = smooth_filter(matrix)
share_upps = (table["Parameter1"].unique()*100).tolist()
thresholds = (table["Parameter2"].unique()*100).tolist()
plt.imshow(np.flip(matrix, axis=0), cmap="Grays")
#formatting
plt.gca().set_yticks(np.arange(len(thresholds)), labels=np.asarray(np.flip(thresholds), dtype=int))
plt.gca().set_xticks(np.arange(len(share_upps)), labels=np.asarray(np.asarray(share_upps), dtype=int))
cbar = plt.colorbar(location="top", orientation="horizontal", fraction=0.05, shrink=0.8, pad=0.12)
cbar.ax.tick_params(rotation=90)
plt.xlabel("Entitled Share $\gamma$ [%]")
plt.ylabel(r"Threshold $\tau$ [%]")
plt.text(0.05, 0.95, '4x', transform=plt.gca().transAxes, 
         color=blue_color, fontsize=24, fontweight='bold', 
         verticalalignment='top', horizontalalignment='left')
plt.title("Average Driver Benefit $c_r$ ")


plt.tight_layout()



import sys
sys.exit(0)


# *****************************************************************************
# ************ Figure 1
# *****************************************************************************
plt.rc('font', family='sans-serif') 
plt.rc('font', serif='Arial') 
# plt.figure(figsize=(3, 3), dpi=100)
# plt.suptitle("(A) Applicability of Priority Pass", fontweight="bold")

flow=50
fig, axes = plt.subplots(nrows=2, ncols=1)
fig.set_size_inches(3, 3)
counter = 0
for ax in axes.flat:
    counter += 1
    
    if counter==1:
        # plt.subplot(2,1,1)
        ax.set_title("Delay Change at Flow="+str(flow)+" [veh/h]")
        # plt.scatter(tablesUPP[flow]["change_delay_upp"],tablesUPP[flow]["change_delay_npp"], c=[(0,0,n) for n in tablesUPP[flow]["Parameter1"]])
        sc = ax.scatter(tablesUPP[flow]["change_delay_upp"],tablesUPP[flow]["change_delay_npp"], c=tablesUPP[flow]["Parameter1"], cmap='viridis')
        ax.set_ylabel("for NPP\n[sec/km]")
        ax.text(-4, 4, r"$\gamma$")
        # plt.xlabel("Delay Change for PP [sec]")
        plt.colorbar(sc)
        
    if counter==2:
        # plt.subplot(2,1,2)
        # plt.title("Flow="+str(flow)+" [veh/h]")
        # plt.scatter(tablesUPP[flow]["change_delay_upp"],tablesUPP[flow]["change_delay_npp"], c=[(0,0,n) for n in tablesUPP[flow]["Parameter2"]])
        sc = ax.scatter(tablesUPP[flow]["change_delay_upp"],tablesUPP[flow]["change_delay_npp"], c=tablesUPP[flow]["Parameter2"], cmap='viridis')
        ax.set_ylabel("for NPP\n[sec/km]")
        ax.text(-4, 4, r"$\tau$")
        ax.set_xlabel("for PP [sec/km]")
        plt.colorbar(sc)
        
# plt.colorbar(sc, ax=axes.ravel().tolist())
# plt.colorbar(sc)
plt.tight_layout()



import sys
sys.exit(0)



# plt.figure(figsize=(6, 3), dpi=100)
fig, ax = plt.subplots(1,2, gridspec_kw={'width_ratios': [1, 1.5]})
fig.set_size_inches(6, 3)

x = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
y = [sum(tablesUPP[flow]["change_delay_upp"]<0)/len(tablesUPP[flow]["change_delay_upp"])*100 for flow in x]
# y = [sum(tablesUPP[flow]["delay_benefit_per_vehicle"]>0)/len(tablesUPP[flow]["change_delay_upp"])*100 for flow in x]
ax[0].plot(x,y)
ax[0].set_xlabel("Flow [veh/h]")
ax[0].set_ylabel("Share of parameter space [%]")
ax[0].set_title("% Benefit for Entitled Vehicles   \n (Delay Change < 0)")
ax[0].scatter(x[7],y[7], color="red")
ax[0].plot([x[8], x[10]], [y[7], 90], "--", color="lightgray")
ax[0].plot([x[8], x[10]], [y[7], 10], "--", color="lightgray")

# plt.figure(figsize=(3, 3), dpi=100)

# plt.subplot(1,2,2, gridspec_kw={'width_ratios': [1, 2]})
flow=400

parameter1_labels = list(set(tablesUPP[flow]["Parameter1"].tolist()))
parameter1_labels.sort()
parameter2_labels = list(set(tablesUPP[flow]["Parameter2"].tolist()))
parameter2_labels.sort()
matrix = [[]]
coordsX = []
coordsY = []
for p1 in parameter1_labels:
    row = []
    for p2 in parameter2_labels:
        upp_benefit = tablesUPP[flow][tablesUPP[flow]["Parameter1"]==p1][tablesUPP[flow]["Parameter2"]==p2]["change_delay_upp"].iloc[0]
        row.append(upp_benefit)
        if upp_benefit<0:
            coordsX.append(parameter1_labels.index(p1))
            coordsY.append(parameter2_labels.index(p2))
    matrix.append(np.asarray(row))
matrix = np.asarray(matrix[1:])
matrix = smooth_filter(matrix)
matidx = np.where(matrix==np.nanmax(matrix))
thr_idx = matidx[0][0]
sha_idx = matrix.shape[1]-matidx[1][0]-1
# sha_opt = parameter1_labels[sha_idx]
# thr_opt = parameter2_labels[thr_idx]    
res = ax[1].imshow(np.flip(matrix.transpose(), axis=0), cmap='viridis', interpolation='nearest')
parameter1_labels = [int(x*100) for x in parameter1_labels]
parameter2_labels = [int(x*100) for x in parameter2_labels]
ax[1].set_xticks(np.arange(len(parameter1_labels)), parameter1_labels)
plt.xticks(rotation = 90)
ax[1].set_yticks(np.arange(len(parameter2_labels)), np.flip(parameter2_labels))
ax[1].set_xlabel(r"Entitled Share $\gamma$ [%]")
ax[1].set_ylabel(r"Threshold $\tau$ [%]")
ax[1].scatter(coordsX, coordsY, s=6, marker="x", color="red")
# ax[1].colorbar(orientation="horizontal", location='bottom', shrink=0.8, aspect=20, pad=0.3)
# fig.colorbar(s, ax=ax, cax=ax)
cbar = plt.colorbar(res, orientation="vertical")
cbar.ax.set_ylabel('Delay Change at Flow = 400', rotation=90)

ax[1].set_title("for PP [sec/km]")
plt.tight_layout()



import sys
sys.exit(0)

# *****************************************************************************
# ************ Figure 2
# *****************************************************************************
plt.rc('font', family='sans-serif') 
plt.rc('font', serif='Arial') 
plt.figure(figsize=(12, 6), dpi=100)
# plt.suptitle("(B) Mechanics of Priority Pass", fontweight="bold")

plt.subplot(2,3,1)
nd = 550
for flow in [50, 100, 150, 200, 250, 300, 350, 400]:#,450, 500, 550]:
    color = (1.0/nd*(nd-flow), 0, 1.0/nd*flow)
    subtable = tablesUPP[flow][["change_delay_upp_rel", "change_delay_npp_rel", "Parameter1", "Parameter2"]]
    subtable = subtable[subtable["change_delay_upp_rel"]<0]
    gamma = subtable.groupby("Parameter1").mean().reset_index()
    tau = subtable.groupby("Parameter2").mean().reset_index()    
    plt.plot(gamma["Parameter1"]*100, gamma["change_delay_upp_rel"], color=color, label=str(flow)+" veh/h")
    # plt.scatter(gamma["Parameter1"]*100, gamma["change_delay_upp_rel"], color=color)
    plt.ylabel(r"Variation of Entitled Share $\gamma$"+"\n"+r"(Average across all thresholds "+r"$\tau$)"+"\n\n\nDelay Change for PP [%]")
    plt.xlabel(r"Entitled Share $\gamma$ [%]")
plt.legend()

plt.subplot(2,3,2)
nd = 550
for flow in [50, 100, 150, 200, 250, 300, 350, 400]:#,450, 500, 550]:
    color = (1.0/nd*(nd-flow), 0, 1.0/nd*flow)
    subtable = tablesUPP[flow][["change_delay_upp_rel", "change_delay_npp_rel", "Parameter1", "Parameter2"]]
    subtable = subtable[subtable["change_delay_upp_rel"]<0]
    gamma = subtable.groupby("Parameter1").mean().reset_index()
    tau = subtable.groupby("Parameter2").mean().reset_index()    
    plt.plot(gamma["Parameter1"]*100, gamma["change_delay_npp_rel"], color=color, label=str(flow))
    # plt.scatter(gamma["Parameter1"]*100, gamma["change_delay_upp_rel"], color=color)
    plt.ylabel("\nDelay Change for NPP [%]")
    plt.xlabel(r"Entitled Share $\gamma$ [%]")

plt.subplot(2,3,3)
for flow in [50, 100, 150, 200, 250, 300, 350, 400]:#,450, 500, 550]:
    # flow = 50
    color = (1.0/nd*(nd-flow), 0, 1.0/nd*flow)
    subtable = tablesUPP[flow][["change_delay_upp_rel", "change_delay_npp_rel", "Parameter1", "Parameter2"]]
    subtable = subtable[subtable["change_delay_upp_rel"]<0]
    gamma = subtable.groupby("Parameter1").mean().reset_index()
    tau = subtable.groupby("Parameter2").mean().reset_index()    
    plt.plot(gamma["change_delay_npp_rel"], gamma["change_delay_upp_rel"], color=color, label=str(flow))
    plt.scatter(gamma["change_delay_npp_rel"], gamma["change_delay_upp_rel"], color=color)
    plt.xlabel("Delay Change for NPP [%]")
    plt.ylabel("Delay Change for PP [%]")
    
plt.subplot(2,3,4)
nd = 550
for flow in [50, 100, 150, 200, 250, 300, 350, 400]:#,450, 500, 550]:
    color = (1.0/nd*(nd-flow), 0, 1.0/nd*flow)
    subtable = tablesUPP[flow][["change_delay_upp_rel", "change_delay_npp_rel", "Parameter1", "Parameter2"]]
    subtable = subtable[subtable["change_delay_upp_rel"]<0]
    gamma = subtable.groupby("Parameter1").mean().reset_index()
    tau = subtable.groupby("Parameter2").mean().reset_index()    
    plt.plot(tau["Parameter2"]*100, tau["change_delay_upp_rel"], color=color, label=str(flow))
    # plt.scatter(gamma["Parameter1"]*100, gamma["change_delay_upp_rel"], color=color)
    plt.ylabel(r"Variation of Threshold $\tau$"+"\n"+"(Average across all thresholds $\gamma$)\n\n\nDelay Change for PP [%]")
    plt.xlabel("Threshold "+r"$\tau$"+" [%]")

plt.subplot(2,3,5)
nd = 550
for flow in [50, 100, 150, 200, 250, 300, 350, 400]:#,450, 500, 550]:
    color = (1.0/nd*(nd-flow), 0, 1.0/nd*flow)
    subtable = tablesUPP[flow][["change_delay_upp_rel", "change_delay_npp_rel", "Parameter1", "Parameter2"]]
    subtable = subtable[subtable["change_delay_upp_rel"]<0]
    gamma = subtable.groupby("Parameter1").mean().reset_index()
    tau = subtable.groupby("Parameter2").mean().reset_index()    
    plt.plot(tau["Parameter2"]*100, tau["change_delay_npp_rel"], color=color, label=str(flow))
    # plt.scatter(gamma["Parameter1"]*100, gamma["change_delay_upp_rel"], color=color)
    plt.ylabel("Delay Change for NPP [%]")
    plt.xlabel("Threshold "+r"$\tau$"+" [%]")

plt.subplot(2,3,6)
for flow in [50, 100, 150, 200, 250, 300, 350, 400]:#,450, 500, 550]:
    # flow = 50
    color = (1.0/nd*(nd-flow), 0, 1.0/nd*flow)
    subtable = tablesUPP[flow][["change_delay_upp_rel", "change_delay_npp_rel", "Parameter1", "Parameter2"]]
    subtable = subtable[subtable["change_delay_upp_rel"]<0]
    gamma = subtable.groupby("Parameter1").mean().reset_index()
    tau = subtable.groupby("Parameter2").mean().reset_index()    
    plt.plot(tau["change_delay_npp_rel"], tau["change_delay_upp_rel"], color=color, label=str(flow))
    plt.scatter(tau["change_delay_npp_rel"], tau["change_delay_upp_rel"], color=color)
    plt.xlabel("Delay Change for NPP [%]")
    plt.ylabel("Delay Change for PP [%]")
    
plt.tight_layout()





# *****************************************************************************
# ************ Figure 3
# *****************************************************************************

plt.rc('font', family='sans-serif') 
plt.rc('font', serif='Arial') 
plt.figure(figsize=(12, 12), dpi=100)
# plt.suptitle("User benefit and parameters of Priority Pass (cents/km)\n", fontweight="bold")

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

def drawHeatmap(vot_upp, flow, optim="per_vehicle"):
    tablesUPP = load_tables(log_auction_upp)
    calculateDelayChanges(tablesUPP, summaryMAX4, vot_upp, 1)
    parameter1_labels = list(set(tablesUPP[flow]["Parameter1"].tolist()))
    parameter1_labels.sort()
    parameter2_labels = list(set(tablesUPP[flow]["Parameter2"].tolist()))
    parameter2_labels.sort()
    matrix = [[]]
    for p1 in parameter1_labels:
        row = []
        for p2 in parameter2_labels:
            if optim=="per_vehicle":
                benefit = tablesUPP[flow][tablesUPP[flow]["Parameter1"]==p1][tablesUPP[flow]["Parameter2"]==p2]["delay_benefit_per_vehicle"].iloc[0]
            else:
                benefit_per_vehicle = tablesUPP[flow][tablesUPP[flow]["Parameter1"]==p1][tablesUPP[flow]["Parameter2"]==p2]["delay_benefit_per_vehicle"].iloc[0]
                vehicles = tablesUPP[flow][tablesUPP[flow]["Parameter1"]==p1][tablesUPP[flow]["Parameter2"]==p2]["NumCompletedVeh.1"].iloc[0]
                benefit = benefit_per_vehicle*vehicles
            row.append(benefit)
        matrix.append(np.asarray(row))
    matrix = np.asarray(matrix[1:])
    matrix = smooth_filter(matrix)
    matidx = np.where(matrix==np.nanmax(matrix))
    thr_idx = matidx[0][0]
    sha_idx = matrix.shape[1]-matidx[1][0]-1
    # sha_opt = parameter1_labels[sha_idx]
    # thr_opt = parameter2_labels[thr_idx]    
    plt.imshow(np.flip(matrix.transpose(), axis=0), cmap='Blues_r', interpolation='nearest')
    parameter1_labels = [int(x*100) for x in parameter1_labels]
    parameter2_labels = [int(x*100) for x in parameter2_labels]
    plt.gca().set_xticks(np.arange(len(parameter1_labels)), parameter1_labels)
    plt.gca().set_yticks(np.arange(len(parameter2_labels)), np.flip(parameter2_labels))
    plt.xlabel(r"Entitled Share $\gamma$ [%]")
    plt.ylabel(r"Threshold $\tau$ [%]")
    plt.colorbar(orientation="horizontal", shrink=0.25, aspect=10)
    # plt.scatter(thr_idx, sha_idx, color="red")
    
plt.subplot(3,3,1)
plt.title("Flow=100, VOT=1x")
drawHeatmap(vot_upp = 1, flow = 100)
plt.subplot(3,3,2)
plt.title("Flow=250, VOT=1x")
drawHeatmap(vot_upp = 1, flow = 250)
plt.subplot(3,3,3)
plt.title("Flow=400, VOT=1x")
drawHeatmap(vot_upp = 1, flow = 400)

plt.subplot(3,3,4)
plt.title("Flow=100, VOT=2x")
drawHeatmap(vot_upp = 2, flow = 100)
plt.subplot(3,3,5)
plt.title("Flow=250, VOT=2x")
drawHeatmap(vot_upp = 2, flow = 250)
plt.subplot(3,3,6)
plt.title("Flow=400, VOT=2x")
drawHeatmap(vot_upp = 2, flow = 400)

plt.subplot(3,3,7)
plt.title("Flow=100, VOT=5x")
drawHeatmap(vot_upp = 5, flow = 100)
plt.subplot(3,3,8)
plt.title("Flow=250, VOT=5x")
drawHeatmap(vot_upp = 5, flow = 250)
plt.subplot(3,3,9)
plt.title("Flow=400, VOT=5x")
drawHeatmap(vot_upp = 5, flow = 400)

plt.tight_layout()




# *****************************************************************************
# ************ Figure 4
# *****************************************************************************

plt.rc('font', family='sans-serif') 
plt.rc('font', serif='Arial') 
plt.figure(figsize=(12, 8), dpi=100)
# plt.suptitle("Benefit of Priority Pass\n\n", fontweight="bold")

def smooth_filter(matrix):
    matrix_smooth = matrix.copy()
    for x in range(0, matrix.shape[0]-1):
        for y in range(0, matrix.shape[1]-1):
            matrix_smooth[x][y] = np.nanmean([matrix_smooth[x-1][y], matrix_smooth[x][y-1], matrix_smooth[x][y], matrix_smooth[x+1][y], matrix_smooth[x][y+1]])
    for x in [0]:
        for y in range(0, matrix.shape[1]-1):
            matrix_smooth[x][y] = np.nanmean([matrix_smooth[x][y-1], matrix_smooth[x][y], matrix_smooth[x+1][y], matrix_smooth[x][y+1]])
    for x in range(0, matrix.shape[0]-1):
        for y in [0]:
            matrix_smooth[x][y] = np.nanmean([matrix_smooth[x-1][y], matrix_smooth[x][y], matrix_smooth[x+1][y], matrix_smooth[x][y+1]])
    for x in [0]:
        for y in [0]:
            matrix_smooth[x][y] = np.nanmean([matrix_smooth[x][y], matrix_smooth[x+1][y], matrix_smooth[x][y+1]])
    for x in [matrix.shape[0]-1]:
        for y in range(0, matrix.shape[1]-1):
            matrix_smooth[x][y] = np.nanmean([matrix_smooth[x][y-1], matrix_smooth[x][y], matrix_smooth[x-1][y], matrix_smooth[x][y+1]])
    for x in range(0, matrix.shape[0]-1):
        for y in [matrix.shape[1]-1]:
            matrix_smooth[x][y] = np.nanmean([matrix_smooth[x-1][y], matrix_smooth[x][y], matrix_smooth[x+1][y], matrix_smooth[x][y-1]])
    for x in [matrix.shape[0]-1]:
        for y in [matrix.shape[1]-1]:
            matrix_smooth[x][y] = np.nanmean([matrix_smooth[x][y], matrix_smooth[x-1][y], matrix_smooth[x][y-1]])
    return matrix_smooth

def getFlowSeries(vot_upp, opt="per_vehicle"):
    tablesUPP = load_tables(log_auction_upp)
    calculateDelayChanges(tablesUPP, summaryMAX4, vot_upp, 1)
    data = []
    for flow in [50, 100, 150, 200, 250, 300, 350, 400]:#, 450, 500, 550]:
        parameter1_labels = list(set(tablesUPP[flow]["Parameter1"].tolist()))
        parameter1_labels.sort()
        parameter2_labels = list(set(tablesUPP[flow]["Parameter2"].tolist()))
        parameter2_labels.sort()
        matrix = [[]]
        for p1 in parameter1_labels:
            row = []
            for p2 in parameter2_labels:
                if opt=="per_vehicle":
                    benefit = tablesUPP[flow][tablesUPP[flow]["Parameter1"]==p1][tablesUPP[flow]["Parameter2"]==p2]["delay_benefit_per_vehicle"].iloc[0]
                else:
                    benefit_per_vehicle = tablesUPP[flow][tablesUPP[flow]["Parameter1"]==p1][tablesUPP[flow]["Parameter2"]==p2]["delay_benefit_per_vehicle"].iloc[0]
                    vehicles = tablesUPP[flow][tablesUPP[flow]["Parameter1"]==p1][tablesUPP[flow]["Parameter2"]==p2]["NumCompletedVeh.1"].iloc[0]
                    benefit = benefit_per_vehicle*vehicles
                row.append(benefit/100) # convert cents to eur
            matrix.append(np.asarray(row))
        matrix = np.asarray(matrix[1:])
        # matrix = smooth_filter(matrix)
        matidx = np.where(matrix==np.nanmax(matrix))
        thr_idx = matidx[0][0]
        sha_idx = matidx[1][0]
        gamma_opt = parameter1_labels[thr_idx] 
        tau_opt = parameter2_labels[sha_idx]
        delay_upp = tablesUPP[flow][tablesUPP[flow]["Parameter1"]==gamma_opt][tablesUPP[flow]["Parameter2"]==tau_opt]["change_delay_upp"].iloc[0]
        delay_npp = tablesUPP[flow][tablesUPP[flow]["Parameter1"]==gamma_opt][tablesUPP[flow]["Parameter2"]==tau_opt]["change_delay_npp"].iloc[0]
        delay_upp_rel = tablesUPP[flow][tablesUPP[flow]["Parameter1"]==gamma_opt][tablesUPP[flow]["Parameter2"]==tau_opt]["change_delay_upp_rel"].iloc[0]
        delay_npp_rel = tablesUPP[flow][tablesUPP[flow]["Parameter1"]==gamma_opt][tablesUPP[flow]["Parameter2"]==tau_opt]["change_delay_npp_rel"].iloc[0]

        benefit_per_vehicle = tablesUPP[flow][tablesUPP[flow]["Parameter1"]==gamma_opt][tablesUPP[flow]["Parameter2"]==tau_opt]["delay_benefit_per_vehicle"].iloc[0]
        vehicles = tablesUPP[flow][tablesUPP[flow]["Parameter1"]==gamma_opt][tablesUPP[flow]["Parameter2"]==tau_opt]["NumCompletedVeh.1"].iloc[0]
        benefit = benefit_per_vehicle*vehicles

        delay_upp_rel_std = tablesUPP[flow][tablesUPP[flow]["Parameter1"]==gamma_opt][tablesUPP[flow]["Parameter2"]==tau_opt]["change_delay_upp_rel_STD"].iloc[0]
        delay_npp_rel_std = tablesUPP[flow][tablesUPP[flow]["Parameter1"]==gamma_opt][tablesUPP[flow]["Parameter2"]==tau_opt]["change_delay_npp_rel_STD"].iloc[0]
        
        data.append([flow, gamma_opt, tau_opt, benefit_per_vehicle/100, benefit/100*0.7667, delay_upp, delay_npp, delay_upp_rel, delay_npp_rel, delay_upp_rel_std, delay_npp_rel_std])
    data = pd.DataFrame(data, columns=["Flow", "Gamma", "Tau", "Benefit_per_vehicle", "Benefit", "DelUPP", "DelNPP", "DelUPPrel", "DelNPPrel", "DelUPPrelSTD", "DelNPPrelSTD"])
    data = data[data["Flow"]<=400]
    return data


data = getFlowSeries(vot_upp=1, opt="per_vehicle")
data2 = getFlowSeries(vot_upp=1, opt="total")
plt.subplot(4,3,1)
plt.title("VOT=1x")
plt.plot(data["Flow"], data["Gamma"]*100, label=r"Entitled share $\gamma$", color="blue")
plt.plot(data["Flow"], data["Tau"]*100, label=r"Threshold $\tau$", color="orange")
plt.plot(data2["Flow"], data2["Gamma"]*100, color="blue", linestyle='dashed')
plt.plot(data2["Flow"], data2["Tau"]*100, color="orange", linestyle='dashed')
plt.legend()
plt.ylabel("Parameters [%]")
plt.subplot(4,3,4)
plt.plot(data["Flow"], data["DelUPPrel"], label="PP", color="blue")
plt.plot(data["Flow"], data["DelNPPrel"], label="Non-PP", color="orange")
plt.plot(data2["Flow"], data2["DelUPPrel"], color="blue", linestyle='dashed')
plt.plot(data2["Flow"], data2["DelNPPrel"], color="orange", linestyle='dashed')
plt.plot(data["Flow"], np.zeros(len(data["Flow"])), color="black", linestyle='dashed')
plt.fill_between(data["Flow"], data["DelUPPrel"]+data["DelUPPrelSTD"], data["DelUPPrel"]-data["DelUPPrelSTD"], color="blue", alpha=0.3)
plt.fill_between(data["Flow"], data["DelNPPrel"]+data["DelNPPrelSTD"], data["DelNPPrel"]-data["DelNPPrelSTD"], color="orange", alpha=0.3)  
plt.ylabel("Delay Change [%]")
plt.legend()
plt.subplot(4,3,7)
plt.plot(data["Flow"], np.asarray(data["Benefit_per_vehicle"])*100, label="optim. per vehicle", color="black")
plt.plot(data2["Flow"], np.asarray(data2["Benefit_per_vehicle"])*100, label="optim. total", color="black", linestyle='dashed')
plt.ylabel("User Benefit [cents/km]")
# plt.xlabel("Flow [veh]")
plt.yscale("log")
# plt.legend()
plt.subplot(4,3,10)
plt.plot(data["Flow"], data["Benefit"], label="optim. per vehicle", color="black")
plt.plot(data2["Flow"], data2["Benefit"], label="optim. total", color="black", linestyle='dashed')
plt.ylabel("System Benefit [€/h]")
plt.xlabel("Flow [veh/h]")
# plt.legend()

data = getFlowSeries(vot_upp=2, opt="per_vehicle")
data2 = getFlowSeries(vot_upp=2, opt="total")
plt.subplot(4,3,2)
plt.title("VOT=2x")
plt.plot(data["Flow"], data["Gamma"]*100, label="Gamma", color="blue")
plt.plot(data["Flow"], data["Tau"]*100, label="Tau", color="orange")
plt.plot(data2["Flow"], data2["Gamma"]*100, color="blue", linestyle='dashed')
plt.plot(data2["Flow"], data2["Tau"]*100, color="orange", linestyle='dashed')
# plt.ylabel("Parameters [%]")
plt.subplot(4,3,5)
plt.plot(data["Flow"], data["DelUPPrel"], label="PP", color="blue")
plt.plot(data["Flow"], data["DelNPPrel"], label="Non-PP", color="orange")
plt.plot(data2["Flow"], data2["DelUPPrel"], color="blue", linestyle='dashed')
plt.plot(data2["Flow"], data2["DelNPPrel"], color="orange", linestyle='dashed')
plt.plot(data["Flow"], np.zeros(len(data["Flow"])), color="black", linestyle='dashed')
plt.fill_between(data["Flow"], data["DelUPPrel"]+data["DelUPPrelSTD"], data["DelUPPrel"]-data["DelUPPrelSTD"], color="blue", alpha=0.3)
plt.fill_between(data["Flow"], data["DelNPPrel"]+data["DelNPPrelSTD"], data["DelNPPrel"]-data["DelNPPrelSTD"], color="orange", alpha=0.3)  
# plt.ylabel("Delay Change [%]")
plt.subplot(4,3,8)
plt.plot(data["Flow"], np.asarray(data["Benefit_per_vehicle"])*100, label="optim. per vehicle", color="black")
plt.plot(data2["Flow"], np.asarray(data2["Benefit_per_vehicle"])*100, label="optim. total", color="black", linestyle='dashed')
plt.yscale("log")
# plt.ylabel("Benefit [€]")
# plt.xlabel("Flow [veh]")
plt.subplot(4,3,11)
plt.plot(data["Flow"], data["Benefit"], label="optim. per vehicle", color="black")
plt.plot(data2["Flow"], data2["Benefit"], label="optim. total", color="black", linestyle='dashed')
# plt.ylabel("Total Benefit [€]")
plt.xlabel("Flow [veh/h]")

data = getFlowSeries(vot_upp=5, opt="per_vehicle")
data2 = getFlowSeries(vot_upp=5, opt="total")
plt.subplot(4,3,3)
plt.title("VOT=5x")
plt.plot(data["Flow"], data["Gamma"]*100, label="Gamma", color="blue")
plt.plot(data["Flow"], data["Tau"]*100, label="Tau", color="orange")
plt.plot(data2["Flow"], data2["Gamma"]*100, color="blue", linestyle='dashed')
plt.plot(data2["Flow"], data2["Tau"]*100, color="orange", linestyle='dashed')
# plt.ylabel("Parameters [%]")
plt.subplot(4,3,6)
plt.plot(data["Flow"], data["DelUPPrel"], label="PP", color="blue")
plt.plot(data["Flow"], data["DelNPPrel"], label="Non-PP", color="orange")
plt.plot(data2["Flow"], data2["DelUPPrel"], color="blue", linestyle='dashed')
plt.plot(data2["Flow"], data2["DelNPPrel"], color="orange", linestyle='dashed')
plt.plot(data["Flow"], np.zeros(len(data["Flow"])), color="black", linestyle='dashed')
plt.fill_between(data["Flow"], data["DelUPPrel"]+data["DelUPPrelSTD"], data["DelUPPrel"]-data["DelUPPrelSTD"], color="blue", alpha=0.3)
plt.fill_between(data["Flow"], data["DelNPPrel"]+data["DelNPPrelSTD"], data["DelNPPrel"]-data["DelNPPrelSTD"], color="orange", alpha=0.3)  
    
# plt.ylabel("Delay Change [%]")
plt.subplot(4,3,9)
plt.plot(data["Flow"], np.asarray(data["Benefit_per_vehicle"])*100, label="optim. per vehicle", color="black")
plt.plot(data2["Flow"], np.asarray(data2["Benefit_per_vehicle"])*100, label="optim. total", color="black", linestyle='dashed')
plt.yscale("log")
# plt.ylabel("Benefit [€]")
# plt.xlabel("Flow [veh]")
plt.subplot(4,3,12)
plt.plot(data["Flow"], data["Benefit"], label="optim. per vehicle", color="black")
plt.plot(data2["Flow"], data2["Benefit"], label="optim. total", color="black", linestyle='dashed')
# plt.ylabel("Total Benefit [€]")
plt.xlabel("Flow [veh/h]")


plt.tight_layout()