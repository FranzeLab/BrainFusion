import matplotlib.pyplot as plt
from numpy import pi

n = 16
# create a n x n square with a marker at each point as dummy data
x_data = []
y_data = []
for x in range(n):
    for y in range(n):
        x_data.append(x)
        y_data.append(y)

# open figure
fig,ax = plt.subplots(figsize=[7,7])
# set limits BEFORE plotting
ax.set_xlim((0,n-1))
ax.set_ylim((0,n-1))
# radius in data coordinates:
r = 0.5 # units
# radius in display coordinates:
r_ = ax.transData.transform([r,0])[0] - ax.transData.transform([0,0])[0] # points
# marker size as the area of a circle
marker_size = pi * r_**2
# plot
ax.scatter(x_data, y_data, s=marker_size, edgecolors='black')

plt.show()

# plot with invisible color
ax.scatter(x_data, y_data, s=marker_size, color=(0,0,0,0))
# calculate scaling
scl = ax.get_xlim()[1] - ax.get_xlim()[0]
# plot correctly (with color)
ax.scatter(x_data, y_data, s=marker_size/scl**2, edgecolors='blue',color='red')

# open figure
fig,ax = plt.subplots(figsize=[7,7])
# setting the limits
ax.set_xlim((0,n-1))
ax.set_ylim((0,n-1))
# radius in data coordinates:
r = 0.5 # units
# radius in display coordinates:
r_ = ax.transData.transform([r,0])[0] - ax.transData.transform([0,0])[0] # points
# marker size as the area of a circle
marker_size = (2*r_)**2
# plot
ax.scatter(x_data, y_data, s=marker_size,linewidths=1)
ax.plot(x_data, y_data, "o",markersize=2*r_)
plt.show()