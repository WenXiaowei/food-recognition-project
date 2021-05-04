from pycocotools.coco import COCO
import numpy as np
import random
import os
import cv2
import skimage.io as io
### For visualizing the outputs ###
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


# % matplotlib
# inline


def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == classID:
            return cats[i]['name']
    return None


def getImage(imageObj, img_folder, input_image_size):
    # Read and normalize an image
    train_img = io.imread(img_folder + '/' + imageObj['file_name']) / 255.0
    # Resize
    train_img = cv2.resize(train_img, input_image_size)
    if (len(train_img.shape) == 3 and train_img.shape[2] == 3):  # If it is a RGB 3 channel image
        return train_img
    else:  # To handle a black and white image, increase dimensions to 3
        stacked_img = np.stack((train_img,) * 3, axis=-1)
        return stacked_img


def getCatAllias(catAlliasList, key):
    for al in catAlliasList:
        if key in al:
            return al[key]

    raise NameError('No such key in list')


def createCatAllias(catIDs):
    l = []
    i = 1
    for id in catIDs:
        l.append({id: i})
        i += 1
    return l


def getMasksWithCats(imageObj, coco, catIDs, catAllias, input_image_size):
    annIds = coco.getAnnIds(imageObj['id'], catIds=catIDs, iscrowd=None)
    anns = coco.loadAnns(annIds)
    cats = []
    train_mask = np.zeros((input_image_size[0], input_image_size[0], len(anns)))
    for a in range(len(anns)):
        ann = anns[a]
        cat_value = getCatAllias(catAllias, ann['category_id'])
        mask = cv2.resize(coco.annToMask(ann), input_image_size)
        train_mask[:, :, a] = mask
        cats.append(cat_value)

    return train_mask, cats


def cocoDataGeneratorWithAug(images, coco, folder, input_image_size, catAllias, mode='train', batch_size=16):
    seed = 32
    augGeneratorArgs = dict(featurewise_center=False,
                            samplewise_center=False,
                            rotation_range=5,
                            width_shift_range=0.01,
                            height_shift_range=0.01,
                            brightness_range=(0.8, 1.2),
                            shear_range=0.01,
                            zoom_range=[1, 1.25],
                            horizontal_flip=True,
                            vertical_flip=False,
                            fill_mode='reflect',
                            data_format='channels_last')
    augGeneratorArgs_mask = augGeneratorArgs.copy()
    _ = augGeneratorArgs_mask.pop('brightness_range', None)
    # Initialize the mask data generator with modified args
    image_gen = ImageDataGenerator(**augGeneratorArgs)
    mask_gen = ImageDataGenerator(**augGeneratorArgs_mask)
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    idx = 0
    img_folder = '{}/images/{}'.format(folder, mode)
    dataset_size = len(images)
    catIds = coco.getCatIds()
    n_classes = len(catIds)
    while True:
        imageObj = images[idx]
        img = getImage(imageObj, img_folder, input_image_size)
        im_masks, im_cats = getMasksWithCats(imageObj, coco, catIDs, catAllias, input_image_size)
        n_masks = im_masks.shape[2]

        # print(im_cats)
        X = np.zeros((batch_size, input_image_size[0], input_image_size[1], 3)).astype('float')
        y = np.zeros((batch_size, input_image_size[0], input_image_size[1], n_classes)).astype('float')

        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation of the images
        # will end up different from the augmentation of the masks
        g_x = image_gen.flow(img.reshape((1,) + img.shape),
                             batch_size=batch_size,
                             seed=seed)

        # print((mm.reshape((1,)+mm.shape+(1,)).shape))
        g_ys = [mask_gen.flow(im_masks[None, :, :, mn, None],
                              batch_size=batch_size,
                              seed=seed) for mn in range(n_masks)]

        for batch_num in range(batch_size):
            X[batch_num] = g_x.next() / 255.0
            for i in range(n_masks):
                y[batch_num, :, :, im_cats[i]] = g_ys[i].next()[:, :, :, 0]

        yield X, y

        if idx >= dataset_size:
            idx = 0
        else:
            idx += 1


ann_dir = os.path.join(os.getcwd(), "./annotations")

ann_file_train = ann_dir + "/train.json"
ann_file_val = ann_dir + "/val.json"
coco_train = COCO(ann_file_train)
coco_val = COCO(ann_file_val)

# locate all categories
catIDs = np.sort(coco_train.getCatIds())
cats = coco_train.loadCats(catIDs)

# training coco images
imgIds_train = coco_train.getImgIds()
images_train = coco_train.loadImgs(imgIds_train)
n_train_imgs = len(images_train)

# validation coco images
imgIds_val = coco_val.getImgIds()
images_val = coco_val.loadImgs(imgIds_val)
n_val_imgs = len(images_val)

# data generator parameters
batch_size = 16
img_height = images_train[0]['height']
img_width = images_train[0]['width']
imgs_size = (img_width, img_height)
cat_allias = createCatAllias(catIDs)


# training data generator\
def get_train_data_generator():
    dg_train = cocoDataGeneratorWithAug(images_train, coco_train, os.getcwd(), imgs_size, cat_allias,
                                        batch_size=batch_size)
    return dg_train


# validation data generator
def get_val_data_generator():
    dg_val = cocoDataGeneratorWithAug(images_val, coco_val, os.getcwd(), imgs_size, cat_allias, batch_size=batch_size)
    return dg_val
