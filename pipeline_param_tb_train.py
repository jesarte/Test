from __future__ import print_function
import os, time, cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import argparse
import random
import os, sys
import subprocess
import SimpleITK as sitk
import yaml

# use 'Agg' on matplotlib so that plots could be generated even without Xserver
# running
import matplotlib

matplotlib.use('Agg')

from utils import utils, helpers
from builders import model_builder


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



import matplotlib.pyplot as plt

print('Checking for previous training settings')
with open('./SampleYaml/input_arguments.yaml') as file:
    input_parameters = yaml.load(file, Loader=yaml.FullLoader)

parser = argparse.ArgumentParser()
parser.add_argument('--test_name', type=str, default=input_parameters['test_name'], help='Name of the test, which will be used to create a results folder')
parser.add_argument('--prior_test_name', type=str, default=input_parameters['test_name'], help='Name of the test which will be used to load the checkpoints')
parser.add_argument('--num_epochs', type=int, default=input_parameters['num_epochs'], help='Number of epochs to train for')
parser.add_argument('--epoch_start_i', type=int, default=input_parameters['epoch_start_i'], help='Start counting epochs from this number')
parser.add_argument('--checkpoint_step', type=int, default=input_parameters['checkpoint_step'], help='How often to save checkpoints (epochs)')
parser.add_argument('--validation_step', type=int, default=input_parameters['validation_step'], help='How often to perform validation (epochs)')
parser.add_argument('--image', type=str, default=input_parameters['image'],
                    help='The image you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--continue_training', type=str2bool, default=input_parameters['continue_training'],
                    help='Whether to continue training from a checkpoint')
parser.add_argument('--data_augmentation', type=str2bool, default=input_parameters['data_augmentation'],
                    help='Whether to perform data augmentation on the images')
parser.add_argument('--dataset', type=str, default=input_parameters['dataset'], help='Dataset you are using.')
parser.add_argument('--crop_height', type=int, default=input_parameters['crop_height'], help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=input_parameters['crop_width'], help='Width of cropped input image to network')
parser.add_argument('--save_step', type=int, default=input_parameters['save_step'], help='Number of epochs between saving validation images')
parser.add_argument('--batch_size', type=int, default=input_parameters['batch_size'], help='Number of images in each batch')
parser.add_argument('--input_channels', type=int, default=input_parameters['input_channels'], help='Number of input channels')
parser.add_argument('--num_val_images', type=int, default=input_parameters['num_val_images'], help='The number of images to used for validations')
parser.add_argument('--h_flip', type=str2bool, default=input_parameters['h_flip'],
                    help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=input_parameters['v_flip'],
                    help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--brightness', type=float, default=input_parameters['brightness'],
                    help='Whether to randomly change the image brightness for data augmentation. Specifies the max brightness change as a factor between 0.0 and 1.0. For example, 0.1 represents a max brightness change of 10%% (+-).')
parser.add_argument('--rotation', type=float, default=input_parameters['rotation'],
                    help='Whether to randomly rotate the image for data augmentation. Specifies the max rotation angle in degrees.')
parser.add_argument('--model', type=str, default=input_parameters['model'],
                    help='The model you are using. See model_builder.py for supported models')
parser.add_argument('--frontend', type=str, default=input_parameters['frontend'],
                    help='The frontend you are using. See frontend_builder.py for supported models')
parser.add_argument('--noise', type=str2bool, default=input_parameters['noise'],
                    help='Whether to add gaussian noise or not')
parser.add_argument('--mean', type=float, default=input_parameters['mean'],
                    help='The mean value for the Gaussian noise')
parser.add_argument('--sigma', type=float, default=input_parameters['sigma'],
                    help='The standard deviation for the Gaussian noise')
parser.add_argument('--learning_rate', type=float, default=input_parameters['learning_rate'],
                    help='Learning rate')
parser.add_argument('--decay', type=float, default=input_parameters['decay'],
                    help='Decay')
parser.add_argument('--save_images', type=str2bool, default=input_parameters['save_images'],
                    help='Whether to save validation and training images or not')
args = parser.parse_args()

#Update the input_arguments dictionary with the parsed arguments

input_parameters['test_name'] = args.test_name

input_parameters['prior_test_name'] = args.prior_test_name

input_parameters['num_epochs'] = args.num_epochs

input_parameters['epoch_start_i'] = args.epoch_start_i

input_parameters['checkpoint_step'] = args.checkpoint_step

input_parameters['validation_step'] = args.validation_step

input_parameters['image'] = args.image

input_parameters['continue_training'] = args.continue_training

input_parameters['data_augmentation'] = args.data_augmentation

input_parameters['dataset'] = args.dataset

input_parameters['crop_height'] = args.crop_height

input_parameters['crop_width'] = args.crop_width

input_parameters['save_step'] = args.save_step

input_parameters['batch_size'] = args.batch_size

input_parameters['input_channels'] = args.input_channels

input_parameters['num_val_images'] = args.num_val_images

input_parameters['h_flip'] = args.h_flip

input_parameters['v_flip'] = args.v_flip

input_parameters['brightness'] = args.brightness

input_parameters['rotation'] = args.rotation

input_parameters['model'] = args.model

input_parameters['frontend'] = args.frontend

input_parameters['noise'] = args.noise

input_parameters['mean'] = args.mean

input_parameters['sigma'] = args.sigma

input_parameters['learning_rate'] = args.learning_rate

input_parameters['decay'] = args.decay

input_parameters['save_images'] = args.save_images


# Create the checkpoints directory
if not os.path.isdir("%s" % (input_parameters['test_name'] + "_checkpoints")):
    os.makedirs("%s" % (input_parameters['test_name'] + "_checkpoints"), 0o777)


with open(input_parameters['test_name'] + '_checkpoints/' + input_parameters['test_name'] + '_config.yaml', 'w+') as file:
    documents = yaml.dump(input_parameters, file)


def data_augmentation(input_image, output_image):
    # Data augmentation
    #input_image, output_image = utils.random_crop(input_image, output_image, input_parameters['crop_height'],
    #                                              input_parameters['crop_width'])

    if input_parameters['h_flip'] and random.randint(0, 1):
        input_image = cv2.flip(input_image, 1)
        input_image = np.expand_dims(input_image, axis=3)
        output_image = cv2.flip(output_image, 1)
        output_image = np.expand_dims(output_image, axis=3)
    if input_parameters['v_flip'] and random.randint(0, 1):
        input_image = cv2.flip(input_image, 0)
        input_image = np.expand_dims(input_image, axis=3)
        output_image = cv2.flip(output_image, 0)
        output_image = np.expand_dims(output_image, axis=3)
    if input_parameters['brightness']:
        factor = 1.0 + random.uniform(-1.0 * input_parameters['brightness'], input_parameters['brightness'])
        table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        input_image = cv2.LUT(input_image, table)
    if input_parameters['rotation']:
        angle = random.uniform(-1 * input_parameters['rotation'], input_parameters['rotation'])

    if input_parameters['rotation']:
        M = cv2.getRotationMatrix2D((input_image.shape[1] // 2, input_image.shape[0] // 2), angle, 1.0)
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]),
                                     flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=-3024)
        input_image = np.expand_dims(input_image, axis=3)
        output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]),
                                      flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=-3024)
        output_image = np.expand_dims(output_image, axis=3)

    if input_parameters['noise'] and random.randint(0, 1):
        mean = 0
        sigma = 10000
        gauss = np.zeros(input_image.shape, np.uint8)
        cv2.randn(gauss, mean, sigma)
        input_image = input_image + gauss

    return input_image, output_image


# Get the names of the classes so we can record the evaluation results
class_names_list, label_values = helpers.get_label_info(os.path.join(input_parameters['dataset'], "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
input_channels = input_parameters['input_channels']

# Compute your softmax cross entropy loss
net_input = tf.placeholder(tf.float32, shape=[None, None, None, input_channels])
net_output = tf.placeholder(tf.float32, shape=[None, None, None, num_classes])

network, init_fn = model_builder.build_model(model_name=input_parameters['model'],
                                             frontend=input_parameters['frontend'], net_input=net_input,
                                             num_classes=num_classes, crop_width=input_parameters['crop_width'],
                                             crop_height=input_parameters['crop_height'], is_training=True)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output))

#TODO meter aqui las imagenes para tensorboard

#TODO cambiar el loss (en el futuro)

#Create the TF summary to save the loss for Tensorboard
tf.summary.scalar('loss', loss)

# loss = tf.reduce_mean()

opt = tf.train.RMSPropOptimizer(input_parameters['learning_rate'], input_parameters['decay']).minimize(loss, var_list=[var for var in
                                                                                            tf.trainable_variables()])

saver = tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

#Create the writers for the Tensorboard summaries, one for the training loss, and the other one for the validation loss
train_writer = tf.summary.FileWriter('./' + input_parameters['test_name'] + '_checkpoints/logs/1/train', sess.graph)
val_writer = tf.summary.FileWriter('./' + input_parameters['test_name'] + '_checkpoints/logs/1/val', sess.graph)
haus_writer = tf.summary.FileWriter('./' + input_parameters['test_name'] + '_checkpoints/logs/1/haus', sess.graph)
jacc_writer = tf.summary.FileWriter('./' + input_parameters['test_name'] + '_checkpoints/logs/1/jacc', sess.graph)

utils.count_params()

# If a pre-trained ResNet is required, load the weights.
# This must be done AFTER the variables are initialized with sess.run(tf.global_variables_initializer())
if init_fn is not None:
    init_fn(sess)

# Set the initial epoch, to be able to overwrite it when loading the last checkpoint
start_epoch = input_parameters['epoch_start_i']


# Load a previous checkpoint if desired
if input_parameters['continue_training'] and os.path.exists(input_parameters['prior_test_name'] + "_checkpoints/epoch.txt"):
    model_checkpoint_name = input_parameters['prior_test_name'] + "_checkpoints/latest_model_" + input_parameters['model'] + "_" + \
                        input_parameters['dataset'] + ".ckpt"
    print('Loaded latest model checkpoint')
    saver.restore(sess, model_checkpoint_name)

    # Recover the previously saved epoch
    fepoch = open(input_parameters['prior_test_name'] + '_checkpoints/epoch.txt')
    strepoch = fepoch.read()
    start_epoch = int(strepoch)

#Change back the model_checkpoint name to save the checkpoints properly
model_checkpoint_name = input_parameters['test_name'] + "_checkpoints/latest_model_" + input_parameters['model'] + "_" + \
                        input_parameters['dataset'] + ".ckpt"

# Load the data
print("Loading the data ...")
train_input_names, train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = utils.prepare_data(
    dataset_dir=input_parameters['dataset'])

# Create the csv to save the acc, iou and loss after every epoch
targetacc = open(input_parameters['test_name'] + '_checkpoints/avg_val_scores.csv', 'w')
targetacc.write("epoch, average accuracy\n")
targetiou = open(input_parameters['test_name'] + '_checkpoints/avg_iou_scores.csv', 'w')
targetiou.write("epoch, average iou\n")
targethausdorff = open(input_parameters['test_name'] + '_checkpoints/avg_hausdorff_scores.csv', 'w')
targethausdorff.write("epoch, average hausdorff\n")
targetloss = open(input_parameters['test_name'] + '_checkpoints/avg_loss_scores.csv', 'w')
targetloss.write("epoch, average loss\n")
targetvalloss = open(input_parameters['test_name'] + '_checkpoints/avg_val_loss_scores.csv', 'w')
targetvalloss.write("epoch, average val_loss\n")
targetacc.close
targetiou.close
targethausdorff.close
targetloss.close
targetvalloss.close


print("\n***** Begin training *****")
print("Dataset -->", input_parameters['dataset'])
print("Model -->", input_parameters['model'])
print("Crop Height -->", input_parameters['crop_height'])
print("Crop Width -->", input_parameters['crop_width'])
print("Num Epochs -->", input_parameters['num_epochs'])
print("Batch Size -->", input_parameters['batch_size'])
print("Num Classes -->", num_classes)

print("Data Augmentation:")
print("\tVertical Flip -->", input_parameters['v_flip'])
print("\tHorizontal Flip -->", input_parameters['h_flip'])
print("\tBrightness Alteration -->", input_parameters['brightness'])
print("\tRotation -->", input_parameters['rotation'])
print("\tNoise -->", input_parameters['noise'])
print("")

avg_loss_per_epoch = []
avg_scores_per_epoch = []
avg_iou_per_epoch = []

# Which validation images do we want
val_indices = []
num_vals = min(input_parameters['num_val_images'], len(val_input_names))

# Set random seed to make sure models are validated on the same validation images.
# So you can compare the results of different models more intuitively.
random.seed(16)
val_indices = random.sample(range(0, len(val_input_names)), num_vals)

# Do the training here
imcount = 0
save_ON = 0
train_save_ON = 0
if args.save_images == True:
    save_ON = 1
    train_save_ON = 1
save_count = 0
epoch_step = input_parameters['save_step']

tfcnt = 0
valcnt = 0

for epoch in range(start_epoch, input_parameters['num_epochs']):

    current_losses = []

    cnt = 0
    trainsavecount = 0

    # Equivalent to shuffling
    id_list = np.random.permutation(len(train_input_names))

    num_iters = int(np.floor(len(id_list) / input_parameters['batch_size']))
    st = time.time()
    epoch_st = time.time()
    for i in range(num_iters):
        # st=time.time()

        input_image_batch = []
        output_image_batch = []

        # Collect a batch of images00
        for j in range(input_parameters['batch_size']):
            index = i * input_parameters['batch_size'] + j
            id = id_list[index]
            input_image = utils.load_image(train_input_names[id])
            # print('INPUT IMAGE SHAPE',input_image.shape)
            output_image = utils.load_image(train_output_names[id])

            with tf.device('/cpu:0'):
                if (input_parameters['data_augmentation'] == True):
                    input_image, output_image = data_augmentation(input_image, output_image)

                # Prep the data. Make sure the labels are in one-hot format
                input_image = np.float32(input_image) / 3024.  # confirmar que es el maximo del valor absoluto #######
                output_image = np.float32(helpers.one_hot_it(label=output_image, label_values=label_values))

                input_image_batch.append(np.expand_dims(input_image, axis=0))
                # print('INPUT IMAGE SHAPE BATCH',type(input_image_batch[0]))
                output_image_batch.append(np.expand_dims(output_image, axis=0))

        if input_parameters['batch_size'] == 1:
            input_image_batch = input_image_batch[0]
            output_image_batch = output_image_batch[0]
        else:
            input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
            output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))

            input_image_batch = np.expand_dims(input_image_batch, axis=3)

        # Do the training
        trainim, current = sess.run([network, loss], feed_dict={net_input: input_image_batch, net_output: output_image_batch})

        print(trainim.shape)

        #Save the images obtained from the validation to Tensorboard
        if train_save_ON == True:

            #Save the training image for Tensorboard using trainim


            #TODO save the appropiate input image and mask too. But first, we must pipeline the network input and preprocessing

            #Stop saving images after certain number of them have been saved
            if trainsavecount >= 10:
                train_save_ON = False

            trainsavecount = trainsavecount + 1

        #Save the loss for TensorBoard
        trainloss = tf.Summary()
        trainloss.value.add(tag='Loss', simple_value=current)
        train_writer.add_summary(trainloss, tfcnt)
        train_writer.flush()


        current_losses.append(current)
        cnt = cnt + input_parameters['batch_size']
        tfcnt = tfcnt + input_parameters['batch_size']
        if cnt % 20 == 0:
            string_print = "Epoch = %d Count = %d Current_Loss = %.4f Time = %.2f" % (
                epoch, cnt, current, time.time() - st)
            utils.LOG(string_print)
            st = time.time()

    # TODO añadir el loss o lo que sea [No se muy bien que había que hacer aquí]

    mean_loss = np.mean(current_losses)
    avg_loss_per_epoch.append(mean_loss)

    # Create directories if needed
    if not os.path.isdir("%s/%d" % (input_parameters['test_name'] + "_checkpoints", epoch)):
        os.makedirs("%s/%d" % (input_parameters['test_name'] + "_checkpoints", epoch), 0o777)

    # Save latest checkpoint to same file name
    print("Saving latest checkpoint")
    saver.save(sess, model_checkpoint_name)

    if val_indices != 0 and epoch % input_parameters['checkpoint_step'] == 0:
        print("Saving checkpoint for this epoch")
        saver.save(sess, "%s/%d/model.ckpt" % (input_parameters['test_name'] + "_checkpoints", epoch))

        # Create a txt to save the checkpoint epoch
        targetepoch = open(input_parameters['test_name'] + "_checkpoints/epoch.txt", 'w')
        targetepoch.write(str(epoch))
        targetepoch.close

    if epoch % input_parameters['validation_step'] == 0:
        print("Performing validation")
        target = open("%s/%d/val_scores.csv" % (input_parameters['test_name'] + "_checkpoints", epoch), 'w')
        target.write(
            "val_name, avg_accuracy, precision, recall, f1 score, mean iou, hausdorff, %s\n" % (class_names_string))


        scores_list = []
        class_scores_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        iou_list = []
        hausdorff_list = []

        save_count = save_count + 1
        print('The current count =' + str(save_count))

        # Do the validation on a small set of validation images
        current_floss = []
        for ind in val_indices:

            input_image = np.expand_dims(
                np.float32(utils.load_image(val_input_names[ind])[:input_parameters['crop_height'],
                           :input_parameters['crop_width']]), axis=0) / 255.0

            gt = utils.load_image(val_output_names[ind])[:input_parameters['crop_height'],
                 :input_parameters['crop_width']]
            gt_onehot = helpers.one_hot_it(gt, label_values)
            gt_onehot = np.expand_dims(gt_onehot, axis=0)
            prefloss, output_image = sess.run([loss, network],
                                              feed_dict={net_input: input_image, net_output: gt_onehot})


            #Write the Validation Loss to TensorBoard
            valloss = tf.Summary()
            valloss.value.add(tag='Loss', simple_value=prefloss)
            val_writer.add_summary(valloss, valcnt)
            val_writer.flush()

            valcnt = valcnt + 1

            gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))
            current_floss.append(prefloss)

            # output_image,currentvaloss = sess.run([opt,loss],network,feed_dict={net_input:input_image})
            # current_losses.append(currentvaloss)

            output_image = np.array(output_image[0, :, :, :])
            output_image = helpers.reverse_one_hot(output_image)
            out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
            out_vis_image = out_vis_image.astype(int)
            output_img_tosave = out_vis_image[:, :, 0]
            output_img_tosave = sitk.GetImageFromArray(output_img_tosave)
            output_img_tosave = sitk.Cast(output_img_tosave, sitk.sitkFloat32)

            accuracy, class_accuracies, prec, rec, f1, iou, hausdorff, floss = utils.evaluate_segmentation(
                pred=output_image, label=gt, pfloss=prefloss, num_classes=num_classes)


            #Write the Hausdorff Distance to TensorBoard
            haus = tf.Summary()
            haus.value.add(tag='Hausdorff Distance', simple_value=hausdorff)
            val_writer.add_summary(haus, valcnt)
            val_writer.flush()

            #Write the Jaccard Index to TensorBoard
            jacc = tf.Summary()
            jacc.value.add(tag='Jaccard Index', simple_value=iou)
            val_writer.add_summary(jacc, valcnt)
            val_writer.flush()

            file_name = utils.filepath_to_name(val_input_names[ind])
            target.write("%s, %f, %f, %f, %f, %f, %f, %f" % (file_name, accuracy, prec, rec, f1, iou, hausdorff, floss))
            for item in class_accuracies:
                target.write(", %f" % (item))
            # target.write(", %f"%(floss))
            target.write("\n")

            scores_list.append(accuracy)
            class_scores_list.append(class_accuracies)
            precision_list.append(prec)
            recall_list.append(rec)
            f1_list.append(f1)
            iou_list.append(iou)
            hausdorff_list.append(hausdorff)

            gt = helpers.colour_code_segmentation(gt, label_values)
            gt = gt.astype(int)
            gt = gt[:, :, 0]
            gt = sitk.GetImageFromArray(gt)
            gt_tosave = sitk.Cast(gt, sitk.sitkFloat32)

            file_name = os.path.basename(val_input_names[ind])
            file_name = os.path.splitext(file_name)[0]

            if (save_ON == 1):
                # cv2.imwrite("%s/%04d/%s_pred.png"%("checkpoints",epoch, file_name),np.uint8(out_vis_image))
                # cv2.imwrite("%s/%04d/%s_gt.png"%("checkpoints",epoch, file_name),np.uint8(gt))
                sitk.WriteImage(output_img_tosave, input_parameters['test_name'] + '_checkpoints/' + str(
                    epoch) + '/' + file_name + '_pred.mhd')
                sitk.WriteImage(gt_tosave, input_parameters['test_name'] + '_checkpoints/' + str(
                    epoch) + '/' + file_name + '_gt.mhd')

                if imcount >= 15:
                    save_ON = 0

                imcount = imcount + 1


        save_count = 0
        imcount = 0
        if args.save_images == True:
            save_ON = 1

        target.close()

        avg_score = np.mean(scores_list)
        avg_val_loss = np.mean(current_floss)
        class_avg_scores = np.mean(class_scores_list, axis=0)
        avg_scores_per_epoch.append(avg_score)
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_f1 = np.mean(f1_list)
        avg_iou = np.mean(iou_list)
        avg_iou_per_epoch.append(avg_iou)
        avg_hausdorff = np.mean(hausdorff_list)

        print("\nAverage validation accuracy for epoch # %04d = %f . Average loss = %f" % (
            epoch, avg_score, avg_val_loss))
        print("Average per class validation accuracies for epoch # %04d:" % (epoch))
        for index, item in enumerate(class_avg_scores):
            print("%s = %f" % (class_names_list[index], item))
        print("Validation precision = ", avg_precision)
        print("Validation recall = ", avg_recall)
        print("Validation F1 score = ", avg_f1)
        print("Validation IoU score = ", avg_iou)

    epoch_time = time.time() - epoch_st
    remain_time = epoch_time * (input_parameters['num_epochs'] - 1 - epoch)
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    if s != 0:
        train_time = "Remaining training time = %d hours %d minutes %d seconds\n" % (h, m, s)
    else:
        train_time = "Remaining training time : Training completed.\n"
    utils.LOG(train_time)
    scores_list = []

    # Create the total avg accuracies/iou/ csv

    targetacc = open(input_parameters['test_name'] + '_checkpoints/avg_val_scores.csv', 'a')

    targetiou = open(input_parameters['test_name'] + '_checkpoints/avg_iou_scores.csv', 'a')

    targetloss = open(input_parameters['test_name'] + '_checkpoints/avg_loss_scores.csv', 'a')

    targetvalloss = open(input_parameters['test_name'] + '_checkpoints/avg_val_loss_scores.csv', 'a')

    targethausdorff = open(input_parameters['test_name'] + '_checkpoints/avg_hausdorff_scores.csv', 'a')

    targetacc.write("%f , %f\n" % (epoch, avg_score))
    targetiou.write("%f , %f\n" % (epoch, avg_iou))
    targetloss.write("%f , %f\n" % (epoch, mean_loss))
    targetvalloss.write("%f , %f\n" % (epoch, avg_val_loss))
    targetacc.close
    targetiou.close
    targetloss.close
    targetvalloss.close
