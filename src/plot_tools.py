import matplotlib.pyplot as plt


def imshow(image, title=None):
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()
