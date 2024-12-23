# IMPORTS
from Data_UrgencyDistribution import urgency_distribution, getUrgencyLevelProcess
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde



blue_color = (30/255, 111/255, 192/255)
gray_color = (110/255, 110/255, 110/255)




plt.rc('font', family='sans-serif') 
plt.rc('font', serif='Arial') 
plt.figure(figsize=(12/2, 6), dpi=100)


# Plot Salary Distribution
plt.subplot(2,1,1)

    # Determine Gaussian Kernel Density Estimation (KDE) For Salary Distribution
salaries = urgency_distribution[1]["pop_salary"]
kde = gaussian_kde(salaries)
x_range = np.linspace(0, salaries.max(), 1000)  # Start from 0
y_values = kde(x_range)
y_values_percentage = y_values * 100
minimum_wage = 15.00
minimum_wage_mask = x_range <= minimum_wage

plt.plot(x_range, y_values_percentage, linewidth=2, color="black")
plt.fill_between(x_range, y_values_percentage, color=blue_color, label="above minimum wage \n[included]")
plt.fill_between(x_range[minimum_wage_mask], y_values_percentage[minimum_wage_mask], color=gray_color, label="below minimum wage \n[excluded]")

plt.xlabel('Salary [UDS/h]')
plt.ylabel('Share Of Population [%]')
plt.title('Salary Distribution')
plt.legend()

plt.xlim(0, None)  # Set x-axis to start at 0
plt.ylim(0, None)  # Set y-axis to start at 0


# Urgency Level Distribution
plt.subplot(2,1,2)
colors = ["black", blue_color, gray_color]
for scenario in [1,2,3]:
    urgency_dist, urgency_level = getUrgencyLevelProcess(p=urgency_distribution[scenario]["scenario"])
    plt.bar(np.asarray(urgency_level)-0.3*(3-scenario+1)+0.3, urgency_dist*100, width=0.3, color=colors[scenario-1], label="Scenario "+str(scenario))
    
plt.title("Urgency Level Distribution")
plt.xlabel('Urgency Level [Times Salary]')
plt.ylabel('Share Of Population [%]')
plt.xticks(range(1, 11))
plt.legend()


# # Urgency VOT Distribution
# plt.subplot(1,3,3)

# plt.title("Value Of Time (VOT) Distribution")
# plt.xlabel("Value of Time [€/h]")
# plt.ylabel("Share of population [%]")

# for scenario in [1,2,3]:
#     # Determine Cumulative Gaussian Kernel Density Estimation (KDE) For Salary Distribution
#     salaries = urgency_distribution[scenario]["pop_urgency_vots"]
#     kde = gaussian_kde(salaries)
#     x_range = np.linspace(0, salaries.max(), 1000)  # Start from 0
#     y_values = kde(x_range)
#     y_values_percentage = y_values * 100
#     cumulative_y = np.cumsum(y_values) / np.sum(y_values) * 100
#     minimum_wage = 15.00
#     minimum_wage_mask = x_range <= minimum_wage
    
#     plt.plot(x_range, cumulative_y, linewidth=2, color="black")
#     plt.fill_between(x_range, cumulative_y, color=colors[scenario-1], label="Scenario "+str(scenario), alpha=0.3)

# plt.xlabel('Salary [USD/h]')
# plt.ylabel('Share Of Population [%]')
# plt.title('Cumulative VOT Distribution')

# plt.grid(True, linestyle='--', alpha=0.7)
# plt.xlim(0, None)  # Set x-axis to start at 0
# plt.ylim(0, None)  # Set y-axis to start at 0




plt.tight_layout()

