import os
import random
import numpy as np
from PIL import Image, ImageEnhance

import jittor as jt
from jittor.dataset import Dataset
from jittor import transform

_class_labels_TR_sorted = 'Airplane, Ant, Antenna, Archery, Axe, BabyCarriage, Bag, BalanceBeam, Balcony, Balloon, Basket, BasketballHoop, Beatle, Bed, Bee, Bench, Bicycle, BicycleFrame, BicycleStand, Boat, Bonsai, BoomLift, Bridge, BunkBed, Butterfly, Button, Cable, CableLift, Cage, Camcorder, Cannon, Canoe, Car, CarParkDropArm, Carriage, Cart, Caterpillar, CeilingLamp, Centipede, Chair, Clip, Clock, Clothes, CoatHanger, Comb, ConcretePumpTruck, Crack, Crane, Cup, DentalChair, Desk, DeskChair, Diagram, DishRack, DoorHandle, Dragonfish, Dragonfly, Drum, Earphone, Easel, ElectricIron, Excavator, Eyeglasses, Fan, Fence, Fencing, FerrisWheel, FireExtinguisher, Fishing, Flag, FloorLamp, Forklift, GasStation, Gate, Gear, Goal, Golf, GymEquipment, Hammock, Handcart, Handcraft, Handrail, HangGlider, Harp, Harvester, Headset, Helicopter, Helmet, Hook, HorizontalBar, Hydrovalve, IroningTable, Jewelry, Key, KidsPlayground, Kitchenware, Kite, Knife, Ladder, LaundryRack, Lightning, Lobster, Locust, Machine, MachineGun, MagazineRack, Mantis, Medal, MemorialArchway, Microphone, Missile, MobileHolder, Monitor, Mosquito, Motorcycle, MovingTrolley, Mower, MusicPlayer, MusicStand, ObservationTower, Octopus, OilWell, OlympicLogo, OperatingTable, OutdoorFitnessEquipment, Parachute, Pavilion, Piano, Pipe, PlowHarrow, PoleVault, Punchbag, Rack, Racket, Rifle, Ring, Robot, RockClimbing, Rope, Sailboat, Satellite, Scaffold, Scale, Scissor, Scooter, Sculpture, Seadragon, Seahorse, Seal, SewingMachine, Ship, Shoe, ShoppingCart, ShoppingTrolley, Shower, Shrimp, Signboard, Skateboarding, Skeleton, Skiing, Spade, SpeedBoat, Spider, Spoon, Stair, Stand, Stationary, SteeringWheel, Stethoscope, Stool, Stove, StreetLamp, SweetStand, Swing, Sword, TV, Table, TableChair, TableLamp, TableTennis, Tank, Tapeline, Teapot, Telescope, Tent, TobaccoPipe, Toy, Tractor, TrafficLight, TrafficSign, Trampoline, TransmissionTower, Tree, Tricycle, TrimmerCover, Tripod, Trombone, Truck, Trumpet, Tuba, UAV, Umbrella, UnevenBars, UtilityPole, VacuumCleaner, Violin, Wakesurfing, Watch, WaterTower, WateringPot, Well, WellLid, Wheel, Wheelchair, WindTurbine, Windmill, WineGlass, WireWhisk, Yacht'
class_labels_TR_sorted = _class_labels_TR_sorted.split(', ')


# 随机翻转
def cv_random_flip(img, label, trunk=None, struct=None):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        if trunk is not None:
            trunk = trunk.transpose(Image.FLIP_LEFT_RIGHT)
            struct = struct.transpose(Image.FLIP_LEFT_RIGHT)
    if trunk is None:
        return img, label
    else:
        return img, label, trunk, struct


# 随机裁剪
def randomCrop(image, label, trunk=None, struct=None):
    image_width = image.size[0]
    image_height = image.size[1]
    border = image_height / 8
    crop_win_width = np.random.randint((image_width - border), image_width)
    crop_win_height = np.random.randint((image_height - border), image_height)
    random_region = (((image_width - crop_win_width) >> 1), ((image_height - crop_win_height) >> 1),
                     ((image_width + crop_win_width) >> 1), ((image_height + crop_win_height) >> 1))
    if trunk is None:
        return image.crop(random_region), label.crop(random_region)
    else:
        return image.crop(random_region), label.crop(random_region), trunk.crop(random_region), struct.crop(
            random_region)


# 随机旋转
def randomRotation(image, label, grad):
    mode = Image.BICUBIC
    if (random.random() > 0.8):
        random_angle = np.random.randint((- 15), 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        grad = grad.rotate(random_angle, mode)
    return (image, label, grad)


# 颜色增强
def colorEnhance(image):
    bright_intensity = (random.randint(5, 15) / 10.0)
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = (random.randint(5, 15) / 10.0)
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = (random.randint(0, 20) / 10.0)
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = (random.randint(0, 30) / 10.0)
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


# 随机高斯噪声
def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    (width, height) = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


# 另一种噪声
def randomPeper(img):
    img = np.array(img)
    noiseNum = int(((0.0015 * img.shape[0]) * img.shape[1]))
    for i in range(noiseNum):
        randX = random.randint(0, (img.shape[0] - 1))
        randY = random.randint(0, (img.shape[1] - 1))
        if (random.randint(0, 1) == 0):
            img[(randX, randY)] = 0
        else:
            img[(randX, randY)] = 255
    return Image.fromarray(img)


class TrainDataset(Dataset):

    def __init__(self, image_root, gt_root, trainsize, is_train=True):
        super().__init__()
        self.trainsize = trainsize
        self.images = [(image_root + f) for f in os.listdir(image_root) if (f.endswith('.jpg') or f.endswith('.png'))]
        self.gts = [(gt_root + f) for f in os.listdir(gt_root) if (f.endswith('.jpg') or f.endswith('.png'))]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.filter_files()
        self.img_transform = jt.transform.Compose([
            jt.transform.Resize((self.trainsize, self.trainsize)),
            jt.transform.ImageNormalize([0.5, 0.5, 0.5], [1, 1, 1]),
            jt.transform.ToTensor()
        ])
        self.gt_transform = jt.transform.Compose([
            jt.transform.Resize((self.trainsize, self.trainsize)),
            jt.transform.ToTensor()
        ])
        self.size = len(self.images)
        self.is_train = is_train

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        if self.is_train:
            (image, gt) = cv_random_flip(image, gt)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        if self.is_train:
            return image, gt
        else:
            return image, self.gts[index]

    def filter_files(self):
        assert ((len(self.images) == len(self.gts)) and (len(self.gts) == len(self.images)))
        images = []
        gts = []
        for (img_path, gt_path) in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if (img.size == gt.size):
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

    def batch_len(self):
        return int(self.size / self.batch_size) if self.size % self.batch_size == 0 else int(
            self.size / self.batch_size + 1)


def get_train_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, is_train=True):
    dataset = TrainDataset(image_root, gt_root, trainsize, is_train).set_attrs(batch_size=batchsize, shuffle=shuffle)

    return dataset


class GTDataset(Dataset):

    def __init__(self, gt_root, trainsize, is_train=True):
        super().__init__()
        self.trainsize = trainsize
        self.images = [(gt_root + '/' + f) for f in os.listdir(gt_root) if (f.endswith('.jpg') or f.endswith('.png'))]
        self.gts = [(gt_root + '/' + f) for f in os.listdir(gt_root) if (f.endswith('.jpg') or f.endswith('.png'))]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.filter_files()
        self.img_transform = jt.transform.Compose([
            jt.transform.Resize((self.trainsize, self.trainsize)),
            jt.transform.ToTensor()
        ])
        self.gt_transform = jt.transform.Compose([
            jt.transform.Resize((self.trainsize, self.trainsize)),
            jt.transform.ToTensor()
        ])
        self.size = len(self.images)
        self.is_train = is_train

    def __getitem__(self, index):
        image = self.binary_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        image = jt.divide(image, 255)
        gt = jt.divide(gt, 255)
        if self.is_train:
            (image, gt) = cv_random_flip(image, gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        return image, gt

    def filter_files(self):
        assert ((len(self.images) == len(self.gts)) and (len(self.gts) == len(self.images)))
        images = []
        gts = []
        for (img_path, gt_path) in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if (img.size == gt.size):
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


def get_gt_loader(gt_root, batchsize, trainsize, shuffle=True, is_train=True):
    dataset = GTDataset(gt_root, trainsize, is_train).set_attrs(batch_size=batchsize, shuffle=shuffle)

    return dataset


class UDUNDataset(Dataset):

    def __init__(self, image_root, gt_root, trainsize, trunk_root=None, struct_root=None, is_train=True):
        super().__init__()
        self.trainsize = trainsize
        self.images = [(image_root + f) for f in os.listdir(image_root) if (f.endswith('.jpg') or f.endswith('.png'))]
        self.gts = [(gt_root + f) for f in os.listdir(gt_root) if (f.endswith('.jpg') or f.endswith('.png'))]

        if is_train:
            self.trunks = [(trunk_root + f) for f in os.listdir(image_root) if
                           (f.endswith('.jpg') or f.endswith('.png'))]
            self.structs = [(struct_root + f) for f in os.listdir(gt_root) if
                            (f.endswith('.jpg') or f.endswith('.png'))]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        if is_train:
            self.trunks = sorted(self.trunks)
            self.structs = sorted(self.structs)

        self.filter_files()
        self.img_transform = jt.transform.Compose([
            jt.transform.Resize((self.trainsize, self.trainsize)),
            jt.transform.ImageNormalize([124.55 / 255.0, 118.90 / 255.0, 102.94 / 255.0],
                                        [56.77 / 255.0, 55.97 / 255.0, 57.50 / 255.0]),
            jt.transform.ToTensor()
        ])
        if is_train:
            self.gt_transform = jt.transform.Compose([
                jt.transform.Resize((self.trainsize, self.trainsize)),
                jt.transform.ToTensor()
            ])
            self.trunk_transform = jt.transform.Compose([
                jt.transform.Resize((self.trainsize, self.trainsize)),
                jt.transform.ToTensor()
            ])
            self.struct_transform = jt.transform.Compose([
                jt.transform.Resize((self.trainsize, self.trainsize)),
                jt.transform.ToTensor()
            ])
        self.size = len(self.images)
        self.is_train = is_train

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        image = self.img_transform(image)
        if self.is_train:
            gt = self.binary_loader(self.gts[index])
            trunk = self.binary_loader(self.trunks[index])
            struct = self.binary_loader(self.structs[index])

            gt = self.gt_transform(gt)
            trunk = self.gt_transform(trunk)
            struct = self.gt_transform(struct)

            (image, gt, trunk, struct) = cv_random_flip(image, gt, trunk, struct)

            return image, gt, trunk, struct

        return image, self.gts[index]

    def filter_files(self):
        assert ((len(self.images) == len(self.gts)) and (len(self.gts) == len(self.images)))
        images = []
        gts = []
        for (img_path, gt_path) in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if (img.size == gt.size):
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

    def batch_len(self):
        return int(self.size / self.batch_size) if self.size % self.batch_size == 0 else int(
            self.size / self.batch_size + 1)


def get_udun_loader(image_root, gt_root, batchsize, trainsize, trunk_root=None, struct_root=None, shuffle=True,
                    is_train=True):
    dataset = UDUNDataset(image_root, gt_root, trainsize, trunk_root, struct_root, is_train).set_attrs(
        batch_size=batchsize, shuffle=shuffle)

    return dataset


class BiRefNetDataset(Dataset):

    def __init__(self, image_root, gt_root, trainsize, is_train=True):
        super().__init__()
        self.trainsize = trainsize
        self.images = [(image_root + f) for f in os.listdir(image_root) if (f.endswith('.jpg') or f.endswith('.png'))]
        self.gts = [(gt_root + f) for f in os.listdir(gt_root) if (f.endswith('.jpg') or f.endswith('.png'))]
        self.labels = []
        self.is_train = is_train

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        if self.is_train:
            self.cls_name2id = {_name: _id for _id, _name in enumerate(class_labels_TR_sorted)}

        for p in self.images:
            for ext in ['.png', '.jpg']:
                ## 'im' and 'gt' may need modifying
                p_gt = p.replace('/im/', '/gt/').replace('.'+p.split('.')[-1], ext)
                if os.path.exists(p_gt):
                    self.labels.append(p_gt)
                    break

        self.filter_files()
        self.img_transform = jt.transform.Compose([
            jt.transform.Resize((self.trainsize, self.trainsize)),
            jt.transform.ImageNormalize([0.5, 0.5, 0.5], [1, 1, 1]),
            jt.transform.ToTensor()
        ])
        self.gt_transform = jt.transform.Compose([
            jt.transform.Resize((self.trainsize, self.trainsize)),
            jt.transform.ToTensor()
        ])
        self.size = len(self.images)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        if self.is_train:
            (image, gt) = cv_random_flip(image, gt)
            class_label = self.cls_name2id[self.labels[index].split('/')[-1].split('#')[3]] if self.is_train else -1

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        if self.is_train:
            return image, gt
        else:
            return image, self.gts[index]

    def filter_files(self):
        assert ((len(self.images) == len(self.gts)) and (len(self.gts) == len(self.images)))
        images = []
        gts = []
        for (img_path, gt_path) in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if (img.size == gt.size):
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

    def batch_len(self):
        return int(self.size / self.batch_size) if self.size % self.batch_size == 0 else int(
            self.size / self.batch_size + 1)


def get_birefnet_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, is_train=True):
    dataset = BiRefNetDataset(image_root, gt_root, trainsize, is_train).set_attrs(batch_size=batchsize, shuffle=shuffle)

    return dataset
