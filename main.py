#Shreeyan Pampana
#import pandas as pd

#planets = pd.read_csv("planets.csv")
#print(planets.to_string())
#planets.sort_values(by = "Temperature_C", inplace = True)
#planets.loc[4]
#print(planets.loc[4, "Name"])
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns

# Load the planet data
df = pd.read_csv("data2_converted.csv")
df2 = pd.read_csv("exoplanets.csv")

#plt.figure(figsize = (10, 6))
#plt.bar(df2["orbital_period"], df2["radius"], color = "skyblue")
plt.show()

filtered_df = df2[(df2["orbital_period"] >= 300) & (df2["orbital_period"] <= 400)]

plt.figure(figsize = (10, 6))
plt.hist(filtered_df["orbital_period"].dropna(), bins = 50, color = "skyblue", edgecolor = "black")
plt.title("Bar Graph of Orbital Periods (300-400 days)")
plt.xlabel("Orbital Period (days)")
plt.ylabel("Number of Planets")
plt.grid(True)
plt.savefig("orbital_period_distribution.png")

plt.scatter(df2["radius"], df2["mass"])
plt.title("Mass vs Radius")
plt.xlabel("Mass")
plt.xlim(0,50)
plt.ylabel("Radius")
plt.savefig("mass_vs_radius.png")

correlation = df2["mass"].corr(df2["radius"])
print("Correlation between mass and radius in a linear model: ", correlation)
print("----------------------------------------")


columns = ["mass", "radius", "orbital_period"] #Select only useful columns
df2_cluster = df2[columns].dropna().copy()

scaler = StandardScaler() #Standardize the data using StandardScaler
scaled_data = scaler.fit_transform(df2_cluster)
kmeans = KMeans(n_clusters = 3, random_state = 0)  #K-Means Clustering
df2_cluster.loc[:,"Cluster"] = kmeans.fit_predict(scaled_data)

for cluster in sorted(df2_cluster["Cluster"].unique()):
  subset = df2_cluster[df2_cluster["Cluster"] == cluster]
  desc = subset[["mass", "radius", "orbital_period"]].describe()
  mean_mass = desc.loc["mean", "mass"]
  mean_radius = desc.loc["mean", "radius"]
  mean_orbital_period = desc.loc["mean", "orbital_period"]
  mean_ratio = mean_mass / mean_radius

  mass_type = (
    "low mass" if mean_mass < 1 else 
    "medium mass" if 1 < mean_mass < 5 else
    "high mass")

  radius_type = (
    "low radius" if mean_radius < 1 else
    "medium radius" if 1 < mean_radius < 5 else
    "high radius")

  orbital_period_type = (
    "short orbital period" if mean_orbital_period < 100 else
    "medium orbital period" if 100 < mean_orbital_period < 500 else
    "long orbital period")
  
  ratio_type = ("low density" if mean_ratio < 2.5 else 
                "medium density" if 2.5 <= mean_ratio <= 5.5 else 
                "high density")
  print(f"Cluster {cluster}:")
  print(f"Mean mass: {mean_mass:.2f} ({mass_type})")
  print(f"Mean radius: {mean_radius:.2f} ({radius_type})")
  print(f"Mean orbital period: {mean_orbital_period:.2f} ({orbital_period_type})")
  print("This cluster generally has planets with a ", mass_type, ", ", radius_type, ", and ", orbital_period_type, ", and a ", ratio_type, ".")
  print("----------------------------------------")

#Visualize clusters (2D plot using first two features)
plt.hist2d(df2_cluster["mass"], df2_cluster["radius"], bins = 20, cmap = "viridis")
plt.title("Mass vs Radius with Clusters")
plt.xlabel("Mass")
plt.xlim(0,10)
plt.ylabel("Radius")
plt.savefig("mass_vs_radius_clusters.png")
plt.colorbar(label = "Cluster")

earth_ratio = 9.374 * 10**17
df2["Ratio"] = df2["mass"]/df2["radius"]

df2.dropna(subset = "radius", inplace = True)
    
earth_like = df2[abs(df2["Ratio"] - earth_ratio) < 1e14]
if not earth_like.empty:
    print("A planet close to the size of Earth has been found:")
else:
    print("No planet close to the size of Earth has been found.")
#Clean the data
df2_clean = df2.dropna(subset = ["mass", "radius"])
df2_clean.loc[:,"Ratio"] = df2_clean["mass"] / df2_clean["radius"]

#Find the mean and standard deviation of the ratio
number = 0
mean_ratio = df2_clean["Ratio"].mean()
std_ratio = df2_clean["Ratio"].std()
anomalies = df2_clean[abs(df2_clean["Ratio"] - mean_ratio) > 3 * std_ratio] #Define anomalies

print("Number of planets with a very high mass to radius ratio: ", number)
print("Number of planets with a very different mass to radius ratio: ", len(anomalies))

plt.figure(figsize = (10, 6))
plt.hist(df2_clean["Ratio"], bins = 50, color = "skyblue", edgecolor = "black")
plt.title("Ratio of Mass to Radius")
plt.xlabel("Ratio")
plt.xlim(0, 30)
plt.ylabel("Number of Planets")
plt.grid(True)
plt.savefig("ratio_distribution.png")

habitable = df2[
    (df2["orbital_period"] >= 200) & (df2["orbital_period"] <= 700) &
    (df2["mass"] >= 0.5) & (df2["mass"] <= 5) &
    (df2["radius"] >= 0.8) & (df2["radius"] <= 1.5) &
    (df2["star_distance"] >= 0.95) & (df2["star_distance"] <= 1.37)
]
print("Number of possible habitable planets in exoplanets.csv: ", len(habitable))


filtered_df = df2_cluster[df2_cluster["orbital_period"] <= 30]
order = (
  filtered_df.groupby("Cluster")["orbital_period"]
  .median()
  .sort_values()
  .index
)

plt.figure(figsize = (10, 6))
sns.boxplot(x = "Cluster", y = "orbital_period", data = filtered_df, order = order)
plt.title("Orbital Period Distribution by Cluster")
plt.savefig("orbital_period_by_cluster.png")

possible_habitable = df[
(df["pl_orbper"] >= 200) & (df["pl_orbper"] <= 10000) &
(df["pl_bmassj"] >= 0) & (df["pl_bmassj"] <= 1000) & 
(df["st_dist"] <= 0.50) &
(df["pl_dens"] >= 3) & (df["pl_dens"] <= 8) &
(df["st_mass"] >= 0) & (df["st_mass"] <= 10) &
(df["st_rad"] >= 0) & (df["st_rad"] <= 10)
]

total_planets = len(df) + len(df2)
print("Number of possible habitable planets in data2_converted.csv: ", len(possible_habitable))
total_habitable = len(habitable) + len(possible_habitable)
print("There are a total of ", total_habitable, " possible habitable planets out of ", total_planets, "planets.")