import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataloader import default_collate
import sys
from config import Config
from model.darkJNN import DarkJNN
from model.mobilev2JNN import mobilev2JNN
import numbers

def judge_tensor_is_zero(input_, dev="cuda:0"):
    is_zero = False
    tmp = torch.zeros(input_.shape, device=dev)
    '''判断两个tensor是否相等'''
    is_zero = torch.equal(tmp, input_)
    return is_zero

"""
@:param handel_choice [None -> read from Config.network_type]
"""


def judge_pillow_image_is_wrong(image_input):
    is_wrong = False
    try:
        if image_input.size[0] == 0 or (image_input.size[1] == 0):
            is_wrong = True
            print("[data] image({}) is ({}, {}), jump.".format(type(image_input), image_input.size[0], image_input.size[1]))
    except:
        print("[ERR][data] input less size object")
        is_wrong = True
    return is_wrong


def network_choice(handle_choice=None):
    # if using default None ,will using the Config.network_type
    net = DarkJNN()
    net_name = "darknet"
    handle_choice = Config.network_type if handle_choice is None else handle_choice
    if handle_choice == "mobile_net_v2":
        net = mobilev2JNN()
        net_name = handle_choice
    print("choose net :{}".format(net_name))
    return net


class Utils:

    def custom_collate_fn(batch):
        """
            Collate data of different batch, it is because the boxes and gt_classes have changeable length.
            This function will pad the boxes and gt_classes with zero. (https://github.com/tztztztztz)

            Arguments:
            batch -- list of tuple (im, boxes, gt_classes)

            im_data -- tensor of shape (3, H, W)
            boxes -- tensor of shape (N, 4)
            num_obj -- tensor of shape (1)

            Returns:

            tuple
            1) tensor of shape (batch_size, 3, H, W)
            1) tensor of shape (batch_size, 3, H, W)
            2) tensor of shape (batch_size, N, 4)
            4) tensor of shape (batch_size, 1)
        """

        # kind of hack, this will break down a list of tuple into
        # individual list
        bsize = len(batch)
        im_dataq, im_datat, boxes, num_obj = zip(*batch)
        max_num_obj = max([x.item() for x in num_obj])
        padded_boxes = torch.zeros((bsize, max_num_obj, 4))

        for i in range(bsize):
            padded_boxes[i, :num_obj[i], :] = boxes[i]

        return torch.stack(im_dataq, 0), torch.stack(im_datat, 0), padded_boxes, torch.stack(num_obj, 0)


class ConsoleLogger(object):

    def __init__(self, filename="log/log.txt"):
        self.terminal = sys.stdout
        # self.err = sys.stderr
        self.log = open(filename, "w")

    def write(self, message):
        self.log.write(message)
        self.terminal.write(message)
        # self.err.write(message)
        self.log.flush()  # 缓冲区的内容及时更新到log文件中

    def flush(self):
        pass


def logg_init_obj(filename="log.txt"):
    sys.stdout = ConsoleLogger(filename=filename)


def augment_img(img, boxes):
    """
    Apply data augmentation. (https://github.com/tztztztztz)
    1. convert color to HSV
    2. adjust hue(.1), saturation(1.5), exposure(1.5)
    3. convert color to RGB
    4. random scale (up to 20%)
    5. translation (up to 20%)
    6. resize to given input size.

    Arguments:
    img -- PIL.Image object
    boxes -- numpy array of shape (N, 4) N is number of boxes, (x1, y1, x2, y2)

    Returns:
    au_img -- numpy array of shape (H, W, 3)
    au_boxes -- numpy array of shape (N, 4) N is number of boxes, (x1, y1, x2, y2)
    """

    # img = np.array(img).astype(np.float32)
    boxes = np.copy(boxes).astype(np.float32)

    for i in range(5):
        img_t, boxes_t = random_scale_translation(img.copy(), boxes.copy(), jitter=Config.jitter)
        keep = (boxes_t[:, 0] != boxes_t[:, 2]) & (boxes_t[:, 1] != boxes_t[:, 3])
        boxes_t = boxes_t[keep, :]
        if boxes_t.shape[0] > 0:
            img = img_t
            boxes = boxes_t
            break

    img = random_distort(img, Config.hue, Config.saturation, Config.exposure)
    return img, boxes


def random_scale_translation(img, boxes, jitter=0.2):
    """

    Arguments:
    img -- PIL.Image
    boxes -- numpy array of shape (N, 4) N is number of boxes
    factor -- max scale size

    Returns:
    im_data -- numpy.ndarray
    boxes -- numpy array of shape (N, 4)
    """

    w, h = img.size

    dw = int(w*jitter)
    dh = int(h*jitter)

    pl = np.random.randint(-dw, dw)
    pr = np.random.randint(-dw, dw)
    pt = np.random.randint(-dh, dh)
    pb = np.random.randint(-dh, dh)

    # scaled width, scaled height
    sw = w - pl - pr
    sh = h - pt - pb

    cropped = img.crop((pl, pt, pl + sw - 1, pt + sh - 1))

    # update boxes accordingly
    #print("boxes: ", boxes)
    boxes[:, 0::2] -= pl
    boxes[:, 1::2] -= pt

    # clamp boxes
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, sw-1)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, sh-1)

    # if flip
    if np.random.randint(2):
        cropped = cropped.transpose(Image.FLIP_LEFT_RIGHT)
        boxes[:, 0::2] = (sw-1) - boxes[:, 2::-2]

    return cropped, boxes


def convert_color(img, source, dest):
    """
    Convert color

    Arguments:
    img -- numpy.ndarray
    source -- str, original color space
    dest -- str, target color space.

    Returns:
    img -- numpy.ndarray
    """

    if source == 'RGB' and dest == 'HSV':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif source == 'HSV' and dest == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img

def our_center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    w, h = img.size
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, i, j, th, tw)


def our_resize(img, size, interpolation=Image.BILINEAR):
    r"""Resize the input PIL Image to the given size.

    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    Returns:
        PIL Image: Resized image.
    """
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def crop(img, i, j, h, w):
    """Crop the given PIL Image.

    Args:
        img (PIL Image): Image to be cropped.
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped image.
        w (int): Width of the cropped image.

    Returns:
        PIL Image: Cropped image.
    """
    return img.crop((j, i, j + w, i + h))



def resized_crop(img, i, j, h, w, size, interpolation=Image.BILINEAR):
    """Crop the given PIL Image and resize it to desired size.

    Notably used in :class:`~torchvision.transforms.RandomResizedCrop`.

    Args:
        img (PIL Image): Image to be cropped.
        i (int): i in (i,j) i.e coordinates of the upper left corner
        j (int): j in (i,j) i.e coordinates of the upper left corner
        h (int): Height of the cropped image.
        w (int): Width of the cropped image.
        size (sequence or int): Desired output size. Same semantics as ``resize``.
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``.
    Returns:
        PIL Image: Cropped image.
    """
    img = crop(img, i, j, h, w)
    img = our_resize(img, size, interpolation)
    return img


def letterbox_image(image, size, use_letterbox=True):
    w, h = size
    iw, ih = image.size
    if use_letterbox:
        '''resize image with unchanged aspect ratio using padding'''
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (0, 0, 0))  # 128, 128, 128
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        if h == w:
            new_image = our_resize(image, h)
        else:
            new_image = our_resize(image, [h, w])
        new_image = our_center_crop(new_image, [h, w])
    return new_image


def rand_scale(s):
    scale = np.random.uniform(1, s)
    if np.random.randint(1, 10000) % 2:
        return scale
    return 1./scale


def random_distort(img, hue=.1, sat=1.5, val=1.5):

    hue = np.random.uniform(-hue, hue)
    sat = rand_scale(sat)
    val = rand_scale(val)

    img = img.convert('HSV')
    cs = list(img.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)

    def change_hue(x):
        x += hue * 255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x

    cs[0] = cs[0].point(change_hue)
    img = Image.merge(img.mode, tuple(cs))

    img = img.convert('RGB')
    return img


def random_hue(img, rate=.1):
    """
    adjust hue
    Arguments:
    img -- numpy.ndarray
    rate -- float, factor used to adjust hue
    Returns:
    img -- numpy.ndarray
    """

    delta = rate * 360.0 / 2

    if np.random.randint(2):
        img[:, :, 0] += np.random.uniform(-delta, delta)
        img[:, :, 0] = np.clip(img[:, :, 0], a_min=0.0, a_max=360.0)

    return img


def random_saturation(img, rate=1.5):
    """
    adjust saturation

    Arguments:
    img -- numpy.ndarray
    rate -- float, factor used to adjust hue

    Returns:
    img -- numpy.ndarray
    """

    lower = 0.5  # hard code
    upper = rate

    if np.random.randint(2):
        img[:, :, 1] *= np.random.uniform(lower, upper)
        img[:, :, 1] = np.clip(img[:, :, 1], a_min=0.0, a_max=1.0)

    return img


def random_exposure(img, rate=1.5):
    """
    adjust exposure (In fact, this function change V (HSV))

    Arguments:
    img -- numpy.ndarray
    rate -- float, factor used to adjust hue

    Returns:
    img -- numpy.ndarray
    """

    lower = 0.5  # hard code
    upper = rate

    if np.random.randint(2):
        img[:, :, 2] *= np.random.uniform(lower, upper)
        img[:, :, 2] = np.clip(img[:, :, 2], a_min=0.0, a_max=255.0)

    return img

class WeightLoader(object):
    """ https://github.com/tztztztztz/yolov2.pytorch """
    def __init__(self):
        super(WeightLoader, self).__init__()
        self.start = 0
        self.buf = None

    def load_conv_bn(self, conv_model, bn_model):
        num_w = conv_model.weight.numel()
        num_b = bn_model.bias.numel()
        bn_model.bias.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))
        self.start = self.start + num_b
        bn_model.weight.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))
        self.start = self.start + num_b
        bn_model.running_mean.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))
        self.start = self.start + num_b
        bn_model.running_var.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))
        self.start = self.start + num_b
        conv_model.weight.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_w]), conv_model.weight.size()))
        self.start = self.start + num_w

    def load_conv(self, conv_model):
        num_w = conv_model.weight.numel()
        num_b = conv_model.bias.numel()
        conv_model.bias.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), conv_model.bias.size()))
        self.start = self.start + num_b
        conv_model.weight.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_w]), conv_model.weight.size()))
        self.start = self.start + num_w

    def dfs(self, m):
        children = list(m.children())
        for i, c in enumerate(children):
            if isinstance(c, torch.nn.Sequential):
                self.dfs(c)
            elif isinstance(c, torch.nn.Conv2d):
                if c.bias is not None:
                    self.load_conv(c)
                else:
                    self.load_conv_bn(c, children[i + 1])

    def load(self, model, weights_file):
        self.start = 0
        fp = open(weights_file, 'rb')
        header = np.fromfile(fp, count=4, dtype=np.int32)
        self.buf = np.fromfile(fp, dtype=np.float32)
        fp.close()
        size = self.buf.size
        self.dfs(model)

        # make sure the loaded weight is right
        assert size == self.start
