import pandas as pd
import matplotlib.pyplot as plt
import json


# # CUT EDGES GRAPH AND METRICS GENERATION
# plt.style.use('ggplot')
data_json = json.load(open("10KPlansFinal.json"))
# cm = plt.cm.get_cmap('Purples')
# compact = data_json["CompactnessScores"]
# compact = pd.Series(compact)
# print(compact.describe())
# print(compact.unique())
# n, bins, patches = plt.hist(compact, bins="auto", color='purple')
# bin_centers = 0.5 * (bins[:-1] + bins[1:])
# col = bin_centers - min(bin_centers)
# col /= max(col)
# for c, p in zip(col, patches):
#     plt.setp(p, 'facecolor', cm(c))
# plt.title("Cut Edges Score Distribution for Training Data")
# plt.ylabel("Instances")
# plt.xlabel("Cut Edges")
# plt.show()

# MEAN - MEDIAN GRAPH AND METRICS GENERATION
plt.style.use('ggplot')
cm = plt.cm.get_cmap('Purples')
MM = data_json["MeanMedianArray"]
MM = pd.Series(MM)
print(MM.describe())
print(MM.unique())
n, bins, patches = plt.hist(MM, bins="auto", color='purple')
bin_centers = 0.5 * (bins[:-1] + bins[1:])
col = bin_centers - min(bin_centers)
col /= max(col)
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))
plt.title("Mean Median Distribution for Training Data")
plt.ylabel("Instances")
plt.xlabel("Mean Median")
plt.show()

#plt.savefig(dpi=1500)