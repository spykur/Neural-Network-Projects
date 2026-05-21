import numpy as np
import os
import matplotlib.pyplot as plt
import tkinter as tk


digits = [[] for _ in range(10)]


def digit_load(path):

    filename = os.path.basename(path)
    name = os.path.splitext(filename)[0]
    label = int(name.split('_')[0])

    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    data = []

    for line in lines:
        for c in line:
            value = 1 if c == '1' else -1
            data.append(value)

    return label, data


def load_all_digits(folder_path):

    for file in os.listdir(folder_path):

        path = os.path.join(folder_path, file)

        label, data = digit_load(path)

        digits[label].append(data)

    return digits


def digits_to_numpy(digits):

    X = []
    y = []

    for label, samples in enumerate(digits):

        for sample in samples:

            X.append(sample)
            y.append(label)

    return np.array(X), np.array(y)


def show_digit_from_file(path):

    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    img = np.array([[int(c) for c in line] for line in lines])

    plt.figure(figsize=(2, 3))
    plt.imshow(img, cmap="gray_r")
    plt.xticks([])
    plt.yticks([])
    plt.title(path)
    plt.show()

#show_digit_from_file("data_training/1_1.txt")

def draw_digit_gui():

    rows, cols = 7, 5

    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    buttons = [[0 for _ in range(cols)] for _ in range(rows)]

    root = tk.Tk()
    root.title("Draw digit 5x7")

    def toggle(r, c):

        grid[r][c] = 1 - grid[r][c]

        color = "black" if grid[r][c] == 1 else "white"
        buttons[r][c].config(bg=color)

    for r in range(rows):
        for c in range(cols):

            btn = tk.Button(
                root,
                width=4,
                height=2,
                bg="white",
                command=lambda r=r, c=c: toggle(r, c)
            )

            btn.grid(row=r, column=c, padx=1, pady=1)
            buttons[r][c] = btn

    result = {"matrix": None, "vector": None}

    def finish():

        result["matrix"] = [row[:] for row in grid]
        result["vector"] = np.array(grid).reshape(-1).tolist()

        root.destroy()

    done_btn = tk.Button(root, text="Gotowe", command=finish)
    done_btn.grid(row=rows, column=0, columnspan=cols, sticky="we")

    root.mainloop()

    return result["matrix"], result["vector"]


def add_noise(sample, p=0.05):

    noisy = sample.copy()

    for i in range(len(noisy)):

        if np.random.rand() < p:
            noisy[i] = -noisy[i]

    return noisy


def perceptron_predict(w, b, x):

    s = np.dot(w, x) + b
    if s >= 0:
        return 1
    else:
        return -1


def train_perceptron(X, y, n_inputs=35, lr=0.1, epochs=200):

    w = np.zeros(n_inputs)
    b = 0

    best_w = w.copy()
    best_b = b
    best_life = 0

    life = 0

    for _ in range(epochs):

        for i in range(len(X)):

            x = add_noise(X[i])
            target = y[i]

            pred = perceptron_predict(w, b, x)

            if pred == target:

                life += 1

                if life > best_life:
                    best_life = life
                    best_w = w.copy()
                    best_b = b

            else:

                life = 0

                w += lr * target * x
                b += lr * target

    return best_w, best_b


def train_all_perceptrons(X, y):

    
    weights = []
    biases = []

    for digit in range(10):

        print("Trening perceptronu:", digit)

        y_binary = np.where(y == digit, 1, -1)

        w, b = train_perceptron(X, y_binary)

        weights.append(w)
        biases.append(b)

    return weights, biases


def classify_digit(weights, biases, x):

    scores = []

    for w, b in zip(weights, biases):

        s = np.dot(w, x) + b
        scores.append(s)

    return np.argmax(scores)


def classify_drawn_digit(weights, biases):

    matrix, vec = draw_digit_gui()

    vec = np.array(vec)
    vec[vec == 0] = -1

    digit = classify_digit(weights, biases, vec)

    print("Rozpoznana cyfra:", digit)

    plt.imshow(matrix, cmap="gray_r")
    plt.title(f"Prediction: {digit}")
    plt.xticks([])
    plt.yticks([])
    plt.show()


def confusion_matrix(weights, biases, X, y):

    matrix = np.zeros((10,10), dtype=int)

    for i in range(len(X)):

        true_label = y[i]
        pred = classify_digit(weights, biases, X[i])

        matrix[true_label][pred] += 1

    return matrix


def show_confusion_matrix(matrix):

    plt.figure(figsize=(6,6))
    plt.imshow(matrix, cmap="Blues")

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.xticks(range(10))
    plt.yticks(range(10))

    for i in range(10):
        for j in range(10):
            plt.text(j, i, matrix[i,j], ha="center", va="center")

    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.show()


def save_model(weights, biases, path="perceptron_weights.npz"):

    np.savez(path, weights=weights, biases=biases)

    print("Model zapisany:", path)


def load_model(path="perceptron_weights.npz"):

    data = np.load(path, allow_pickle=True)

    weights = data["weights"]
    biases = data["biases"]

    print("Model wczytany")

    return weights, biases


def train_test_split(digits, train_ratio=8/12):

    train_X = []
    train_y = []

    test_X = []
    test_y = []

    for label, samples in enumerate(digits):

        samples = np.array(samples)

        np.random.shuffle(samples)

        split = int(len(samples) * train_ratio)

        train = samples[:split]
        test = samples[split:]

        for i in train:
            train_X.append(i)
            train_y.append(label)

        for i in test:
            test_X.append(i)
            test_y.append(label)

    return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)


mode = input("Tryb (train/load): ")

digits = load_all_digits("data_training")

X_train, y_train, X_test, y_test = train_test_split(digits)


if mode == "train":

    weights, biases = train_all_perceptrons(X_train, y_train)

    save_model(weights, biases)

else:

    weights, biases = load_model()


matrix = confusion_matrix(weights, biases, X_test, y_test)

print("Macierz pomyłek:")
print(matrix)

show_confusion_matrix(matrix)

classify_drawn_digit(weights, biases)