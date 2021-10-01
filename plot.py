import os
import matplotlib.pyplot as plt

def plot(dat, nash, player, model, option='individual'):
    plt.figure(figsize=(20, 10))

    if option == 'average':
        plt.title(player + " Convergence -- 50 episodes, all top strategies added,"
                           " strategy set limit: 100 (with half random)")
        plt.ylabel("Average Utility")
        filename = player[0] + "O_avg.png"
    elif option == 'individual':
        plt.title(player + " Utilities -- select Nash equilibrium for max Defender utility")
        plt.ylabel("Utility")
        filename = player[0] + "O.png"

    plt.xlabel("Iteration")
    plt.plot(dat, label=player + "Oracle: " + model)
    plt.plot(nash, label="Nash Eq")
    plt.legend()
    if os.path.isfile(filename):
        os.remove(filename)
    plt.savefig(filename)
    plt.close()