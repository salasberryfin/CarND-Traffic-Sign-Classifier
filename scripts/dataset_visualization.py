import matplotlib.pyplot as plt
import numpy as np
import csv
import random
import data_preproc as preproc

CSV_FILE = "signnames.csv"


def show_sample(dt):
    index = random.randint(0, len(dt.X_train))
    image = dt.X_train[index]
    plt.figure(figsize=(1, 1))
    plt.imshow(image)
    plt.show()


def load_csv():
    with open(CSV_FILE) as csvfile:
        id_name_map = {}
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            id_name_map[row[0]] = row[1]

    return id_name_map


def map_plot_signs(dt, sign_map):
    images = []
    labels = []
    y, indices = np.unique(dt.y_train, return_index=True)
    for i, _ in enumerate(y):
        images.append(dt.X_train[indices[i]])
        labels.append(y[i])
            
    n_col = 5
    n_rows = int(len(images)/n_col) + 1
    fig, axs = plt.subplots(n_rows, n_col)
    fig.subplots_adjust(hspace=.4, wspace=.3)
    axs = axs.ravel()
    for i in range(n_col * n_rows):
        axs[i].axis('off')
        if i < len(images):
            image = images[i]
            axs[i].axis('off')
            axs[i].imshow(image)
            axs[i].set_title(f"{labels[i]} {sign_map[str(labels[i])]:.30}")
    plt.show()


if __name__ == "__main__":
    dataset = preproc.Dataset()
    # show_sample(dataset)
    id_name_map = load_csv()
    map_plot_signs(dataset, id_name_map)
