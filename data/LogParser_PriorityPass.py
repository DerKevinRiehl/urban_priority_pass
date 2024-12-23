# Imports
import os
import numpy as np




# Folder Parameter
folder = "Logs/manhattan_priority_pass/"
outputfile = "log_priority_pass.csv"



# Methods
def parseFile(file):
    f = open(file, "r")
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    relevant_line = f.readline()
    if len(relevant_line)!=0:
        mean = relevant_line.replace("\n", "").strip()
        f.close()
        try:
            parts = mean.split(" ")
            parts = [float(p) for p in parts]
        except:
            parts = -1# [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        return parts
    else:
        return -1


# Files
files = os.listdir(folder)

# traffic_flows = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
traffic_flows = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
fW = open(outputfile, "w+")
fW.write("Flow,Parameter1,Parameter2,Errors,Mean,Mean,Mean,Mean,Mean,Mean,Mean,Mean,Mean,Mean,Mean,Mean,Mean,Median,Median,Median,Median,Median,Median,Median,Median,Median,Median,Median,Median,Median,Std,Std,Std,Std,Std,Std,Std,Std,Std,Std,Std,Std,Std,\n")
fW.write("Flow,Parameter1,Parameter2,Errors,Total_Throuput,Total_AvQueueLength,NumCompletedVeh,NumVehIntersectionPassages,VehAvDelay,VehMdDelay,VehStDelay,UPPAvDelay,UPPMedDelay,UPPStDelay,NPPAvDelay,NPPMedDelay,NPPStDelay,Total_Throuput,Total_AvQueueLength,NumCompletedVeh,NumVehIntersectionPassages,VehAvDelay,VehMdDelay,VehStDelay,UPPAvDelay,UPPMedDelay,UPPStDelay,NPPAvDelay,NPPMedDelay,NPPStDelay,Total_Throuput,Total_AvQueueLength,NumCompletedVeh,NumVehIntersectionPassages,VehAvDelay,VehMdDelay,VehStDelay,UPPAvDelay,UPPMedDelay,UPPStDelay,NPPAvDelay,NPPMedDelay,NPPStDelay,\n")
for flow in traffic_flows:
    for parameter1 in  [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        for parameter2 in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            print(flow, parameter1, parameter2)
            rel_files = []
            for file in files:
                if file.startswith("log_auction_queue_UPP_"+str(flow)+"_"+str(parameter1)+"_"+str(parameter2)+"_"):
                    rel_files.append(file)
            results = []
            count_errors = 0
            for file in rel_files:
                res = parseFile(folder+"/"+file)
                if not res==-1:
                    if np.nan in res:
                        count_errors += 1
                    results.append(res)
            if len(results)==0:
                count_errors = 10
                mean = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
                medi = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
                std =  [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            else:
                try:
                    results = np.asarray(results)
                    mean = np.nanmean(results, axis=0)
                    medi = np.nanmedian(results, axis=0)
                    std = np.nanstd(results, axis=0)
                except:
                    mean = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
                    medi = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
                    std =  [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            fW.write(str(flow))
            fW.write(",")
            fW.write(str(parameter1))
            fW.write(",")
            fW.write(str(parameter2))
            fW.write(",")
            fW.write(str(count_errors))
            fW.write(",")
            for r in mean:
                fW.write(str(r))
                fW.write(",")
            for r in medi:
                fW.write(str(r))
                fW.write(",")
            for r in std:
                fW.write(str(r))
                fW.write(",")
            fW.write("\n")
fW.close()