import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import seaborn as sns

    
# #############################################################################
# ########### VISUALIZE
# #############################################################################
  
blue_color = [30/255,111/255,192/255,1]


plt.rc('font', family='Times New Roman') 
plt.figure(figsize=(12, 3), dpi=100)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42




traffic_flows = [50, 100, 150, 200, 250, 300]#, 350]#, 400, 450, 500, 550, 600]


def smootje(data, n=2):
    for x in range(0,n):
        data = np.column_stack((data[:, 0], np.pad(data[:, 1], (1, 1), mode='edge')[1:-1] * 0.25 + data[:, 1] * 0.5 + np.pad(data[:, 1], (1, 1), mode='edge')[2:] * 0.25))
    return data

def smoothie(x,y):
    smooth_interp = lambda x, y, steps=100: (np.linspace(x.min(), x.max(), steps), interp1d(x, y, kind='cubic')(np.linspace(x.min(), x.max(), steps)))
    x_smooth, y_smooth = smooth_interp(x, y)
    return x_smooth, y_smooth





# TLS ANALYIS #################################################################

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



plt.subplot(1,2,1)
plt.gca().set_facecolor('#e6ffe6')
plt.xlim(0,30)

ax1 = sns.violinplot(
    data=count_durations_G_MAX, x="duration", y="flow", hue=True, hue_order=[True, False], split=True, orient="h", inner=None,
    palette=[blue_color, blue_color], linewidth=0)
ax1.legend_ = None
ax1.set_xlabel('')
ax1.set_ylabel('')

ax2 = sns.violinplot(
    data=count_durations_G_UPP, x="duration", y="flow", hue=True, hue_order=[False, True], split=True, orient="h", inner=None,
    palette=["black", 'black'], linewidth=0)
ax2.legend_ = None
ax2.set_xlabel('')
ax2.set_ylabel('')

ax3 = sns.violinplot(
    data=count_durations_G_FIX, x="duration", y="flow", hue=True, hue_order=[True, False], split=True, orient="h", inner=None,
    palette=["gray", 'gray'], linewidth=0)
ax3.legend_ = None
ax3.set_xlabel('')
ax3.set_ylabel('')
        
plt.xlabel('Green Signal Duration [sec]')
plt.ylabel("Inflow Per Entrance [veh/h]")
        

plt.subplot(1,2,2)
plt.gca().set_facecolor('#ffe6e6')
plt.xlim(0,60)

ax = sns.violinplot(
    data=count_durations_R_MAX, x="duration", y="flow", hue=True, hue_order=[True, False], split=True, orient="h", inner=None,
    palette=[blue_color, blue_color], linewidth=0)
ax.legend_ = None
ax.set_xlabel('')
ax.set_ylabel('')

ax = sns.violinplot(
    data=count_durations_R_UPP, x="duration", y="flow", hue=True, hue_order=[False, True], split=True, orient="h", inner=None,
    palette=["black", 'black'], linewidth=0)
ax.legend_ = None
ax.set_xlabel('')
ax.set_ylabel('')

ax = sns.violinplot(
    data=count_durations_R_FIX, x="duration", y="flow", hue=True, hue_order=[True, False], split=True, orient="h", inner=None,
    palette=["gray", 'gray'], linewidth=0)
ax.legend_ = None
ax.set_xlabel('')
ax.set_ylabel('')

plt.xlabel('Red Signal Duration [sec]')
plt.ylabel("Inflow Per Entrance [veh/h]")





# plt.subplot(2,2,2)
# plt.gca().set_facecolor('#e6ffe6')
# plt.xlim(0,20)

# # Define offset
# offset = 0.5

# # # Function to create boxplot with offset
# # def create_boxplot(data, palette):
# #     ax = sns.boxplot(
# #         data=data, x="duration", y="flow", hue=True, 
# #         hue_order=[True, False], orient="h",
# #         palette=palette, width=0.5)
# #     return ax

# # # Create boxplots with offsets
# # count_durations_G_MAX["flow"] = count_durations_G_MAX["flow"]-20
# # ax1 = create_boxplot(count_durations_G_MAX, [blue_color, blue_color])
# # # ax2 = create_boxplot(count_durations_G_UPP, ['black', 'black'])
# # count_durations_G_FIX["flow"] = count_durations_G_MAX["flow"]+20
# # ax3 = create_boxplot(count_durations_G_FIX, ['gray', 'gray'])

# # # Remove legends and labels
# # for ax in [ax1, ax2, ax3]:
# #     if ax.legend_:
# #         ax.legend_.remove()
# #     ax.set_xlabel('')
# #     ax.set_ylabel('')

# for flow in traffic_flows:
#     plt.gca().boxplot(count_durations_G_FIX[count_durations_G_FIX["flow"]==flow]["duration"], positions=[flow-10], widths=10.0, showfliers=False, patch_artist=True, vert=False)
#     plt.gca().boxplot(count_durations_G_MAX[count_durations_G_MAX["flow"]==flow]["duration"], positions=[flow],    widths=10.0, showfliers=False, patch_artist=True, vert=False)
#     plt.gca().boxplot(count_durations_G_UPP[count_durations_G_UPP["flow"]==flow]["duration"], positions=[flow+10], widths=10.0, showfliers=False, patch_artist=True, vert=False)

# plt.gca().set_yticklabels([])
# plt.yticks(np.arange(50, 349, 50))
# plt.gca().set_yticklabels([str(x) for x in np.arange(50, 349, 50)])



# # Add the legend to the figure
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='gray', edgecolor='none', label='Fixed-Cycle'),
    Patch(facecolor=blue_color, edgecolor='none', label='Max-Pressure'),
    Patch(facecolor='black', edgecolor='none', label='Priority Pass')
]
plt.figlegend(
    handles=legend_elements, 
    loc='upper center', 
    ncol=3, 
    bbox_to_anchor=(0.5, 1.0)
)

plt.tight_layout()
plt.subplots_adjust(top=0.87)  # Adjust this value to make room for the legend
