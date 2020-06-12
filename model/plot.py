import matplotlib.pyplot as plt


def plot_history(history):
    plt.figure(figsize=(10, 7))
    plt.plot(np.arange(1, len(history["train_loss"]) + 1), history["train_loss"], label="training loss")
    plt.plot(np.arange(1, len(history["train_loss"]) + 1), history["valid_loss"], label="validation loss")
    plt.legend(loc="best")
    plt.title("Training and Validation Losses")
    plt.show()
