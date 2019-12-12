import numpy as np
import matplotlib.pyplot as plt

titles = ["OpenCV", "CppFlow", "TfLite (1 Thread)", "TfLite (4 Threads)"]
semSegMean = [12.6762, 13.4546, 21.2997, 18.0461]
semSegStd = [3.24769, 0.526951, 0.285547, 1.55253]
classMean = [5.54286, 6.14033, 186.294, 92.1016]
classStd = [0.827205, 0.485032, 21.5911, 6.11409]


def plot(mean, std, name, file):
    fig, ax = plt.subplots()
    x_pos = np.arange(len(titles))
    ax.bar(x_pos, mean, yerr=std, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel("Time (ms")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(titles)
    ax.set_title("Inference time of the " + name)
    ax.yaxis.grid(True)

    plt.tight_layout()
    plt.savefig(file)
    plt.show()


def main():
    plot(semSegMean, semSegStd, "Semantic Segmentation", "semSegPlt.svg")
    plot(classMean, classStd, "Classification", "classPlt.svg")


if __name__ == '__main__':
    main()
