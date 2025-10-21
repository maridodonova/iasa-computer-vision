import matplotlib.pyplot as plt


def imshow(image, title: str | None = None) -> None:
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def plot_history(history):
    n_plots = len(history)
    plt.figure(figsize=(5 * n_plots, 3))

    for i, (metric_name, values) in enumerate(history.items()):
        metric_name = metric_name.capitalize()
        plt.subplot(1, n_plots, i+1)
        plt.plot(values["train"], label=f"Train {metric_name}")
        plt.plot(values["val"], label=f"Validation {metric_name}")
        plt.title(f"{metric_name} Curve")
        plt.xlabel("Epochs")
        plt.ylabel(metric_name)
        plt.legend()

    plt.show()
