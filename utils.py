import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
import matplotlib
import statistics as stats
import torchvision.transforms as transforms
from colorama import Fore, Style
import logging

colormap = ["FF355E",
            "8ffe09",
            "1d5dec",
            "FF9933",
            "FFFF66",
            "CCFF00",
            "AAF0D1",
            "FF6EFF",
            "FF00CC",
            "299617",
            "AF6E4D"]


def color_map():
    s = ''
    for color in colormap:
        s += color
    b = bytes.fromhex(s)
    cm = np.frombuffer(b, np.uint8)
    cm = cm.reshape(len(colormap), 3)
    return cm


class ResultsLogger(object):
    def __init__(self, title='title', logfile='keypoints.log'):
        super().__init__()
        self.ll = []
        self.scale = 4
        self.viewer = UniImageViewer(title, screen_resolution=(128 * 5 * self.scale, 128 * self.scale))
        self.logging = logging
        self.best_loss = 1e10
        logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s',
                            level=logging.DEBUG, handlers=[logging.FileHandler(logfile, 'a', 'utf-8')])

    def header(self, run_id, model_name, run_type, batch_size, lr, opt_level, dataset, num_keypoints):
        self.logging.debug(f'STARTING RUN: {run_id}, model_name: {model_name}, run_type: {run_type}, '
                           f'batch_size: {batch_size}, lr {lr}, '
                           f'opt_level: {opt_level}, dataset: {dataset}, keypoints: {num_keypoints}')

    def display(self, x, x_, x_t, k, *images, blocking=False):
        key_x, key_y = k
        kp_image = plot_keypoints_on_image((key_x.float(), key_y.float()), x_.float())
        kp_image_t = transforms.ToTensor()(kp_image).to(x.device)
        panel = [x[0].float(), x_[0].float(), x_t[0].float(), kp_image_t]
        for i in images:
            panel.append(i[0].float())
        panel = torch.cat(panel, dim=2)
        self.viewer.render(panel, blocking)

    def log(self, tqdm, epoch, batch_i, loss, optim, x, x_, x_t, k, type, depth=None):
        self.ll.append(loss.item())
        if depth is not None and len(self.ll) > depth:
            self.ll.pop(0)
        tqdm.set_description(f'Epoch: {epoch} LR: {get_lr(optim)} {type} Loss: {stats.mean(self.ll)}')

        if not batch_i % 8:
            self.display(x, x_, x_t, k)
            # display(x, x_, k, loss_image, loss_mask.expand(-1, 3, -1, -1))

    def end_epoch(self, epoch, optim):

        """ check improvement """
        if len(self.ll) > 0:
            ave_loss = stats.mean(self.ll)
            self.best_loss = ave_loss if ave_loss <= self.best_loss else self.best_loss
            mesg = f'{Fore.GREEN}EPOCH {epoch} LR: {get_lr(optim)} {Fore.CYAN}ave loss: {ave_loss} {Fore.LIGHTBLUE_EX}best loss: {self.best_loss} {Style.RESET_ALL}'
            print(mesg)
            self.logging.debug(mesg)

            return ave_loss, self.best_loss


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


def plot_keypoints_on_image(k, image_tensor, radius=2, thickness=1, batch_index=torch.tensor([0], dtype=torch.long)):
    height, width = image_tensor.size(2), image_tensor.size(3)
    image_tensor = torch.cat(torch.unbind(image_tensor[batch_index], 0), dim=2)

    def to_npy(x):
        return x.detach().squeeze().cpu().numpy()

    x, y = k
    x, y = to_npy(x[batch_index]), to_npy(y[batch_index])
    x, y = np.floor(x * width), np.floor(y * height)
    img = transforms.ToPILImage()(image_tensor.cpu())
    img = np.array(img)

    cm = color_map()[:len(x)].astype(int)

    for x_, y_, color in zip(x, y, cm):
        c = color.item(0), color.item(1), color.item(2)
        cv2.circle(img, (x_, y_), radius, c, thickness)

    # You may need to convert the color.
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)

    return img_pil


# def plot_keypoints_on_image(k, image_tensor, batch_index=torch.tensor([0], dtype=torch.long)):
#     height, width = image_tensor.size(2), image_tensor.size(3)
#     image_tensor = torch.cat(torch.unbind(image_tensor[batch_index], 0), dim=2)
#     x, y = k
#     x, y = x[batch_index].detach().squeeze().cpu().numpy(), y[batch_index].detach().squeeze().cpu().numpy()
#
#     fig = Figure()
#     canvas = FigureCanvas(fig)
#     ax = fig.gca()
#
#     ax.axis('off')
#     ax.margins(0)
#     ax.set_xlim(0.0, width)
#     ax.set_ylim(height, 0.0)
#     ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
#     ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     plt.autoscale(tight=True)
#     fig.tight_layout()
#     ax.imshow(image_tensor.permute(1, 2, 0).cpu(), zorder=1)
#     cluster = list(Line2D.filled_markers)[:x.shape[0]]
#     offset = 0
#     for i in range(batch_index.size(0)):
#         for xp, yp, m in zip(x, y, cluster):
#             ax.scatter(xp * width + offset, yp * height, marker=m, zorder=2)
#         canvas.draw()
#         offset += width
#
#     s = canvas.tostring_rgb()
#
#     # return PIL image.
#     return Image.frombytes("RGB", fig.canvas.get_width_height(), s)
