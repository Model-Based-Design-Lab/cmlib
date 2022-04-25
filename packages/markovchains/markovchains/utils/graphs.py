import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO


def plotSvg(data, plots):

    df = pd.DataFrame(data)
    fig, ax = plt.subplots()
    for p in plots:
        ax.plot('k', p, data=df, marker='o', markersize=12, linewidth=4)
    ax.set_title("Markov State Distributions")
    ax.set_xlabel("Steps")

    plt.legend()
    buf = BytesIO()
    plt.savefig(buf, format="svg")
    buf.seek(0)    
    byte_str = buf.read()
    return byte_str.decode('UTF-8')
    


