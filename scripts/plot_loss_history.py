import matplotlib
matplotlib.use("tkAgg")
import matplotlib.pyplot as plt
import csv

SLIM_MODE = False  # thin out plot to make number of plot equals NB_PLOT
FILE_NAME = "loss_history.png"
X_LABEL = "episodes"
Y_LAVEL = "loss"

with open("loss_history.csv", "rb") as f:
    vals = csv.reader(f).next()

if SLIM_MODE:
    NB_PLOT = 5000
    assert len(vals) >= NB_PLOT
    vals = vals[::len(vals)/NB_PLOT]
    FILE_NAME = "loss_history_slimmed.png"
    X_LABEL = "episodes (scaled to make number of plot becomes %d)" % NB_PLOT

X = range(len(vals))
plt.plot(X, vals)
plt.xlabel(X_LABEL)
plt.ylabel(Y_LAVEL)
plt.savefig(FILE_NAME)

