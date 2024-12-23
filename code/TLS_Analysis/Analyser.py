import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


traffic_flows = [50, 100, 150, 200, 250, 300, ]







def loadSwitches(control, traffic_flows):
    count_switches = []
    for flow in traffic_flows:
        file = "logs/log_tls_states_"+control+"_"+str(flow)+".xml"
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



plt.subplot(1,2,1)
plt.plot(count_switches_FIX[:,0], count_switches_FIX[:,1], label="Fixed Programme")
plt.plot(count_switches_MAX[:,0], count_switches_MAX[:,1], label="Max Pressure")
plt.plot(count_switches_UPP[:,0], count_switches_UPP[:,1], label="Priority Pass")
plt.legend()
plt.ylabel("#SignalSwitches Per Intersection Per Hour")

import sys
sys.exit(0)


def loadGreenRedDurations(control, traffic_flows):
    count_durations_G = []
    count_durations_R = []
    for flow in traffic_flows:
        file = "logs/log_tls_states_"+control+"_"+str(flow)+".xml"
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
    
plt.subplot(1,2,2)
# violinparts  = plt.gca().violinplot(count_durations_G, showmeans=True, showmedians=True, side="right")
# for pc in violinparts ['bodies']:
#     pc.set_facecolor('green')
#     pc.set_edgecolor('green')
#     pc.set_alpha(0.7)
# violinparts['cmedians'].set_edgecolor('green')
# violinparts['cmeans'].set_edgecolor('green')
# violinparts['cmaxes'].set_edgecolor('green')
# violinparts['cmins'].set_edgecolor('green')
# violinparts['cbars'].set_edgecolor('green')

# sns.violinplot(data=count_durations_G, inner="quartile", side="right")



plt.subplot(1,3,1)

ax = sns.violinplot(
    data=count_durations_G_FIX,
    x="flow", y="duration", hue=True,
    hue_order=[True, False], split=True,
    color="green"
)
ax.legend_ = None
ax = sns.violinplot(
    data=count_durations_R_FIX,
    x="flow", y="duration", hue=True,
    hue_order=[False, True], split=True,
    color="red"
)
ax.legend_ = None
plt.ylim(0,600)


plt.subplot(1,3,2)

ax = sns.violinplot(
    data=count_durations_G_MAX,
    x="flow", y="duration", hue=True,
    hue_order=[True, False], split=True,
    color="green"
)
ax.legend_ = None
ax = sns.violinplot(
    data=count_durations_R_MAX,
    x="flow", y="duration", hue=True,
    hue_order=[False, True], split=True,
    color="red"
)
ax.legend_ = None
plt.ylim(0,600)

plt.subplot(1,3,3)

ax = sns.violinplot(
    data=count_durations_G_UPP,
    x="flow", y="duration", hue=True,
    hue_order=[True, False], split=True,
    color="green"
)
ax.legend_ = None
ax = sns.violinplot(
    data=count_durations_R_UPP,
    x="flow", y="duration", hue=True,
    hue_order=[False, True], split=True,
    color="red"
)
ax.legend_ = None
plt.ylim(0,600)





# plt.figure(figsize=(15, 6))

# # Left subplot (Green duration)
# plt.subplot(1, 2, 1)

# for data, label in [(count_durations_G_FIX, 'FIX'), 
#                     (count_durations_G_MAX, 'MAX'), 
#                     (count_durations_G_UPP, 'UPP')]:
#     mean = data.groupby('flow')['duration'].mean()
#     std = data.groupby('flow')['duration'].std()
    
#     plt.plot(mean.index, mean.values, label=label)
#     plt.fill_between(mean.index, mean.values - std.values, mean.values + std.values, alpha=0.2)

# plt.xlabel('Flow')
# plt.ylabel('Mean Green Duration')
# plt.title('Green Duration vs Flow')
# plt.legend()

# # Right subplot (Red duration)
# plt.subplot(1, 2, 2)

# for data, label in [(count_durations_R_FIX, 'FIX'), 
#                     (count_durations_R_MAX, 'MAX'), 
#                     (count_durations_R_UPP, 'UPP')]:
#     mean = data.groupby('flow')['duration'].mean()
#     std = data.groupby('flow')['duration'].std()
    
#     plt.plot(mean.index, mean.values, label=label)
#     plt.fill_between(mean.index, mean.values - std.values, mean.values + std.values, alpha=0.2)

# plt.xlabel('Flow')
# plt.ylabel('Mean Red Duration')
# plt.title('Red Duration vs Flow')
# plt.legend()

# plt.tight_layout()
# plt.show()









# plt.subplot(1,2,1)

# count_durations_G_FIX = count_durations_G_FIX.groupby('flow')['duration'].mean().reset_index()
# plt.plot(count_durations_G_FIX["flow"], count_durations_G_FIX["duration"])

# count_durations_G_MAX = count_durations_G_MAX.groupby('flow')['duration'].mean().reset_index()
# plt.plot(count_durations_G_MAX["flow"], count_durations_G_MAX["duration"])

# count_durations_G_UPP = count_durations_G_UPP.groupby('flow')['duration'].mean().reset_index()
# plt.plot(count_durations_G_UPP["flow"], count_durations_G_UPP["duration"])


# plt.subplot(1,2,2)

# count_durations_R_FIX = count_durations_R_FIX.groupby('flow')['duration'].mean().reset_index()
# plt.plot(count_durations_R_FIX["flow"], count_durations_R_FIX["duration"])

# count_durations_R_MAX = count_durations_R_MAX.groupby('flow')['duration'].mean().reset_index()
# plt.plot(count_durations_R_MAX["flow"], count_durations_R_MAX["duration"])

# count_durations_R_UPP = count_durations_R_UPP.groupby('flow')['duration'].mean().reset_index()
# plt.plot(count_durations_R_UPP["flow"], count_durations_R_UPP["duration"])