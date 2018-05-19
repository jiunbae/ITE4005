from matplotlib import pyplot as plt

def cluster_plot(path, x, y, c, bold=1):
    plt.scatter(x, y, bold, c)
    plt.savefig(path)
