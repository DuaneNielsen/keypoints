import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import torchvision.transforms as tvt
from PIL import Image
import matplotlib

def precision(confusion):
    correct = confusion * torch.eye(confusion.shape[0])
    incorrect = confusion - correct
    correct = correct.sum(0)
    incorrect = incorrect.sum(0)
    precision = correct / (correct + incorrect)
    total_correct = correct.sum().item()
    total_incorrect = incorrect.sum().item()
    percent_correct = total_correct / (total_correct + total_incorrect)
    return precision, percent_correct

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def to_numpyRGB(image, invert_color=False):
    """
    Universal method to detect and convert an image to numpy RGB format
    :params image: the output image
    :params invert_color: perform RGB -> BGR convert
    :return: the output image
    """
    if type(image) == torch.Tensor:
        image = image.cpu().detach().numpy()
    # remove batch dimension
    if len(image.shape) == 4:
        image = image[0]
    smallest_index = None
    if len(image.shape) == 3:
        smallest = min(image.shape[0], image.shape[1], image.shape[2])
        smallest_index = image.shape.index(smallest)
    elif len(image.shape) == 2:
        smallest = 0
    else:
        raise Exception(f'too many dimensions, I got {len(image.shape)} dimensions, give me less dimensions')
    if smallest == 3:
        if smallest_index == 2:
            pass
        elif smallest_index == 0:
            image = np.transpose(image, [1, 2, 0])
        elif smallest_index == 1:
            # unlikely
            raise Exception(f'Is this a color image?')
        if invert_color:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif smallest == 1:
        image = np.squeeze(image)
    elif smallest == 0:
        # greyscale
        pass
    elif smallest == 4:
        # I guess its probably the 32-bit RGBA format
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        raise Exception(f'dont know how to display color of dimension {smallest}')
    return image


class UniImageViewer:
    def __init__(self, title='title', screen_resolution=(640, 480), format=None, channels=None, invert_color=True):
        self.C = None
        self.title = title
        self.screen_resolution = screen_resolution
        self.format = format
        self.channels = channels
        self.invert_color = invert_color

    def render(self, image, block=False):

        image = to_numpyRGB(image, self.invert_color)

        image = cv2.resize(image, self.screen_resolution)

        # Display the resulting frame
        cv2.imshow(self.title, image)
        if block:
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)

    def view_input(self, model, input, output):
        image = input[0] if isinstance(input, tuple) else input
        self.render(image)

    def view_output(self, model, input, output):
        image = output[0] if isinstance(output, tuple) else output
        self.render(image)

    def update(self, image):
        self.render(image)


def plot_keypoints_on_image(k, image_tensor, batch_index=torch.tensor([0], dtype=torch.long)):

    height, width = image_tensor.size(2), image_tensor.size(3)
    image_tensor = torch.cat(torch.unbind(image_tensor[batch_index], 0), dim=2)
    x, y = k
    x, y = x[batch_index].detach().squeeze().cpu().numpy(), y[batch_index].detach().squeeze().cpu().numpy()

    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    ax.axis('off')
    ax.margins(0)
    ax.set_xlim(0.0, width)
    ax.set_ylim(height, 0.0)
    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
    fig.tight_layout()
    ax.imshow(image_tensor.permute(1, 2, 0).cpu(), zorder=1)
    cluster = list(Line2D.filled_markers)[:x.shape[0]]
    offset = 0
    for i in range(batch_index.size(0)):
        for xp, yp, m in zip(x, y, cluster):
            ax.scatter(xp * width + offset, yp * height, marker=m, zorder=2)
        canvas.draw()
        offset += width

    s = canvas.tostring_rgb()

    # return PIL image.
    return Image.frombytes("RGB", fig.canvas.get_width_height(), s)
