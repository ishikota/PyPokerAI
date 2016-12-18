import matplotlib
matplotlib.use("tkAgg")
import matplotlib.pyplot as plt
import csv

with open("initial_value_transition.csv", "rb") as f:
    vals = csv.reader(f).next()
NB_PLOT = 5000
assert len(vals) >= NB_PLOT
slimmed = vals[::len(vals)/NB_PLOT]
X = range(len(slimmed))
plt.plot(X, slimmed)
plt.savefig("initial_value_transition.png")

