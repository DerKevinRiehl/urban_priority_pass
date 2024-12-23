
f = open("Network.net.xml", "r")
content = f.read()
f.close()

lines = content.split("\n")

lenght = 0
for line in lines:
    if "length=\"" in line:
        lenght += float(line.split("length=\"")[1].split("\"")[0])
        
print("total length", lenght)