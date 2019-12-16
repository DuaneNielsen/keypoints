import matplotlib.figure
import torch
import cv2
import numpy as np
from PIL import Image
import statistics as stats
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import math

from colorama import Fore, Style
import logging

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.tensorboard import SummaryWriter
from math import floor
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


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


def resize2D(tensor, size, interpolation=Image.BILINEAR):
    pic = F.to_pil_image(tensor.cpu())
    pic = F.resize(pic, size, interpolation)
    return F.to_tensor(pic).to(tensor.device)


class ResultsLogger(object):
    def __init__(self, model_name, run_id, num_keypoints,
                 comment='', title='title', logfile='keypoints.log',
                 visuals=True, image_capture_freq=8, kp_rows=4):
        super().__init__()
        self.ll = []
        self.scale = 2
        self.viewer = UniImageViewer(title, screen_resolution=(128 * 5 * self.scale, 128 * self.scale))
        matplotlib_scale = 200
        kp_columns = math.ceil(num_keypoints / kp_rows)
        self.kp_viewer = UniImageViewer('bottleneck',
                                        screen_resolution=(2 * kp_columns * matplotlib_scale, kp_rows * matplotlib_scale))
        self.logging = logging
        self.best_loss = 1e10
        self.visuals = visuals
        logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s',
                            level=logging.DEBUG,
                            handlers=[logging.FileHandler(logfile, 'a', 'utf-8')])
        self.step = 0
        self.name = model_name
        log_dir = f'data/models/{self.name}/run_{run_id}'
        self.tb = SummaryWriter(log_dir=log_dir, comment=comment)
        self.image_capture_freq = image_capture_freq
        self.kp_rows = kp_rows
        self.debug_view = UniImageViewer()

    def header(self, args):

        mesg = ''
        for name, value in args.__dict__.items():
            mesg += f'{name}: {value} '
        self.logging.debug(mesg)
        print(mesg)
        if self.tb:
            self.tb.add_text(mesg, 'Config', global_step=0)

    # def build_panel(self, *images):
    #     panel = []
    #     for img in images:
    #         panel.append(to_numpyRGB(img))
    #
    #     return np.concatenate(panel, axis=1)

    def display(self, panel, blocking=False):
        self.viewer.render(panel, blocking)

    def log(self, tqdm, epoch, batch_i, loss, optim, x, x_, x_t, hm, k, m, p, loss_mask, type, depth=None):
        self.ll.append(loss.item())
        if depth is not None and len(self.ll) > depth:
            self.ll.pop(0)
        tqdm.set_description(f'Epoch: {epoch} LR: {get_lr(optim)} {type} Loss: {stats.mean(self.ll)}')

        if self.tb:
            self.tb.add_scalar(f'{type}_loss', loss.item(), global_step=self.step)

        if not batch_i % self.image_capture_freq:
            kp_image = plot_keypoints_on_image(k[0], x_[0])
            panel = torch.cat([x[0], x_[0], x_t[0], loss_mask[0], F.to_tensor(kp_image)], dim=2)
            bottleneck_image = plot_bottleneck_layer(hm=hm, p=p, k=k, g=m, rows=self.kp_rows)
            bottleneck_image = cv2.cvtColor(bottleneck_image, cv2.COLOR_RGBA2RGB)

            if self.visuals:
                self.display(panel)
                self.kp_viewer.render(bottleneck_image)
            if self.tb:
                scale = 2
                panel = resize2D(panel, (panel.size(1) * scale, panel.size(2) * scale))
                self.tb.add_image(f'{type}_panel', panel, global_step=self.step)
                self.tb.add_image(f'{type}_kp', F.to_tensor(bottleneck_image), global_step=self.step)

        self.step += 1

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

    # if type(image) == Image.Image:
    #     img = image.convert("RGB")
    #     img = np.array(img)
    #     return img

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


def plot_keypoints_on_image(k, image_tensor, radius=1, thickness=1):
    height, width = image_tensor.size(1), image_tensor.size(2)
    num_keypoints = k.size(0)

    if len(k.shape) != 2:
        raise Exception('Individual images and keypoints, not batches')

    k = k.clone()
    k[:, 0] = k[:, 0] * height
    k[:, 1] = k[:, 1] * width
    k.floor_()
    k = k.detach().cpu().numpy()

    img = transforms.ToPILImage()(image_tensor.cpu())

    img = np.array(img)
    cm = color_map()[:num_keypoints].astype(int)

    for co_ord, color in zip(k, cm):
        c = color.item(0), color.item(1), color.item(2)
        co_ord = co_ord.squeeze()
        cv2.circle(img, (co_ord[1], co_ord[0]), radius, c, thickness)

    #img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    #img = np.transpose(img, (2, 0, 1))
    #img = Image.fromarray(img)

    return img


def plot_joint(img, x_marginal, y_marginal, k, color):
    w, h = matplotlib.figure.figaspect(1.0)
    fig = plt.figure(figsize=(h, w))
    canvas = FigureCanvas(fig)

    gs = GridSpec(4, 4)

    top = gs[0, 0:3]
    side = gs[1:4, 3]
    joint = gs[1:4, 0:3]

    ax_joint = fig.add_subplot(joint)
    ax_marg_top = fig.add_subplot(top)
    ax_marg_top_kp = fig.add_subplot(top)

    ax_marg_side = fig.add_subplot(side)
    ax_marg_side_kp = fig.add_subplot(side)

    ax_joint.imshow(img, cmap='gray', vmin=0, vmax=img.max())

    width = x_marginal.shape[0]
    ax_marg_top.bar(np.arange(width), x_marginal, color='#949494')
    xbins = np.zeros(width)
    k_w = floor(k[1].item()  * width)
    # dirty hack for now
    if k_w > width - 1:
        k_w = width -1
    xbins[k_w] = x_marginal.max()
    ax_marg_top_kp.bar(np.arange(width), xbins, color=color)

    height = y_marginal.shape[0]
    ax_marg_side.barh(np.arange(height), y_marginal, color='#949494')
    ax_marg_side.set_ylim(height, 0)
    ybins = np.zeros(height)
    k_h = floor(k[0].item() * height)
    if k_h > height - 1:
        k_h = height - 1
    ybins[k_h] = y_marginal.max()
    ax_marg_side_kp.barh(np.arange(height), ybins, color=color)

    # Turn off tick labels on marginals
    plt.setp(ax_joint.get_xticklabels(), visible=False)
    plt.setp(ax_marg_top.get_xticklabels(), visible=False)
    plt.setp(ax_marg_top_kp.get_xticklabels(), visible=False)
    plt.setp(ax_marg_side.get_yticklabels(), visible=False)
    plt.setp(ax_marg_side_kp.get_yticklabels(), visible=False)

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    cnvs = np.fromstring(s, np.uint8).reshape((height, width, 4))
    plt.close(fig)
    return cnvs


def plot_bottleneck_layer(hm, p, k, g, rows):
    img = []
    for index in range(k.size(1)):
        color = '#' + colormap[index]
        img.append(plot_single_bottleneck(index, hm, p, k, g, color))

    pads = - k.size(1) % rows
    for i in range(pads):
        img.append(np.ones_like(img[0]) * 255)

    columns = []
    for i in range(rows):
            start = i * rows
            if start >= len(img):
                break
            end = min(start + rows, len(img))
            columns.append(np.concatenate(img[start:end], axis=0))
    return np.concatenate(columns, axis=1)


def plot_single_bottleneck(index, hm, p, k, g, color):
    hm_plot = plot_joint(hm[0, index].cpu().detach().numpy(),
                         p[1][0, index].cpu().detach().numpy().squeeze(),
                         p[0][0, index].cpu().detach().numpy().squeeze(),
                         k[0, index].cpu().detach().numpy(),
                         color
                         )
    g_plot = plot_joint(g[0, index].cpu().detach().numpy(),
                        p[1][0, index].cpu().detach().numpy().squeeze(),
                        p[0][0, index].cpu().detach().numpy().squeeze(),
                        k[0, index].cpu().detach().numpy(),
                        color
                        )
    image = np.concatenate((hm_plot, g_plot), axis=1)
    return image
