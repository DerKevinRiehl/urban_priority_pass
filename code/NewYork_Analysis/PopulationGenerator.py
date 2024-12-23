import matplotlib.pyplot as plt


plt.title("Daily Traffic Demand")


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

    return xlabels, yvalues2

xlabels, yvalues = generateTrafficDemandNewYork()
plt.bar(xlabels, yvalues)

for it in range(0, len(xlabels)):
    for x in range(0, 12):
        for y in range(0, 12):
            route = str(x)+"_"+str(y)
            flowLabel = route+"_"+xlabels[it].replace(":","")
            prob = yvalues[it]/3600
            begin = 3600*it
            end = 3600*(it+1)
            flow_string = '<flow id="'+flowLabel+'" probability="'+str(prob)+'" route="'+route+'" begin="'+str(begin)+'" end="'+str(end)+'" > </flow>'
            print(flow_string)
# <flow id="flow_main_left_car_1" probability="0.2" route="0_0" begin="0" end="3600" > </flow>

