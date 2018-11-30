# coding=utf-8
import tensorflow as tf
from nets.mobilenet import mobilenet_v2
import numpy as np
from config import R
import os, cv2


def read_img(img_path):
    if not os.path.exists(img_path):
        raise NotADirectoryError(img_path + " doesn't exist.")

    img_list = list()
    for root, dirnames, _ in os.walk(img_path):  # root path
        for subDirName in dirnames:
            fullDirPath = os.path.join(root, subDirName)  # sub path
            for _, _, filenames in os.walk(fullDirPath):
                for filename in filenames:
                    if filename.endswith('.jpg'):
                        img_list.append(os.path.join(fullDirPath, filename))
    return img_list


def test_model_from_img(img_path):
    img_list = read_img(img_path)
    # placeholder holds an input tensor for classification
    input_images = tf.placeholder(dtype=tf.float32, shape=[None, R.resize_height, R.resize_width, R.depths],
                                  name='input')

    out, end_points = mobilenet_v2.mobilenet(input_tensor=input_images, num_classes=R.num_classes,
                                             depth_multiplier=R.depth_multiplier, is_training=False)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # restore variables that have been trained.
        saver.restore(sess, R.CKPT)

        label = str()
        for s in R.labelsStr:
            label += s + ' '
        print(label)
        acc = 0
        for img_name in img_list:
            dirname, filename = os.path.split(img_name)
            dirname = dirname.split(os.sep)[-1]
            img_raw_data = tf.gfile.FastGFile(img_name, 'rb').read()
            img_data = tf.image.decode_jpeg(img_raw_data)
            # img_data = tf.image.per_image_standardization(img_data)
            img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
            # elements are in [0,1)
            resized_img = tf.image.resize_images(img_data, size=[R.resize_height, R.resize_width], method=0)

            # decode an image
            img = resized_img.eval(session=sess)
            img.resize([1, R.resize_height, R.resize_width, R.depths])

            # input an image array and inference to get predictions and set normal format
            predictions = end_points['Predictions'].eval(session=sess, feed_dict={input_images: img})
            predictions.resize([R.num_classes])
            np.set_printoptions(precision=4, suppress=True)
            index = np.argmax(predictions)
            print('Predict[{: ^7s}] is {}.'.format(dirname, str(predictions)))
            if R.labelsDict[dirname] == index:
                acc += 1
            else:
                print(" --- Wrong: Mistake " + dirname + " for " + R.labelsStr[index])
                # wrong[dirname] = value
        print("The accuracy of this test is {:.3f} - {}/{}".format(acc/len(img_list), acc, len(img_list)))
        '''
        for key, val in wrong.items():
            img = cv2.imread(key, 0)
            cv2.imshow(val, img)
            k = cv2.waitKey(0)
            if k == 27:  # wait for ESC key to exit
                cv2.destroyAllWindows()
                break
            cv2.destroyAllWindows()
        '''
    print('Bye.')


if __name__ == '__main__':
    test_model_from_img(R.test_image_path)
