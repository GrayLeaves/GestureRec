# coding=utf-8

import datetime
import tensorflow.contrib.slim as slim
from generateTfrecords import *
from nets.mobilenet import mobilenet_v2
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


# define log_mode
dat = datetime.datetime.now()  # 获取当前时间以便定义日志格式
logPathName = R.log_path + '{:2d}{:2d}_{:2d}h_{}l2.log'.format(dat.month, dat.day, dat.hour, R.max_steps)
logging.basicConfig(level=logging.INFO, filename=logPathName, datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)
param = '\n### dropout = ' + str(R.dropout) + '\n### base_size = ' + str(R.batch_size) \
        + '\n### base_lr = ' + str(R.base_lr) + '\n### max_steps = ' + str(R.max_steps) \
        + '\n### depth_multiplier = ' + str(R.depth_multiplier) \
        + '\n### save_min_accu = ' + str(R.save_min_accu)
logger.info(param)

# 定义input_images为图片数
input_images = tf.placeholder(dtype=tf.float32, shape=[None, R.resize_height, R.resize_width, R.depths], name='input')
# 定义input_labels为labels数据
input_labels = tf.placeholder(dtype=tf.int32, shape=[None, R.num_classes], name='label')
# 定义dropout的概
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
is_training = tf.placeholder(tf.bool, name='is_training')


def resultpic(log_step, losses, accuracy):
    '''
    show and save the final result picture
    '''
    (train_log_step, val_log_step) = log_step
    (train_losses, val_losses) = losses
    (train_accuracy, val_accuracy) = accuracy
    eval_train = range(0, R.max_steps + 1, train_log_step)
    eval_val = range(0, R.max_steps + 1, val_log_step)
    plt.figure(1)
    # plot loss over time
    plt.subplot(211)
    plt.plot(eval_train, train_losses, 'k-')
    plt.plot(eval_val, val_losses, 'b--')
    plt.title('Softmax Loss Per Generation')
    plt.legend(('Train', 'Test'))
    plt.xlabel('Generation')
    plt.ylabel('Softmax Loss')
    plt.grid(True)
    # plot accuracy over time
    plt.subplot(212)
    plt.plot(eval_train, train_accuracy, 'k-')
    plt.plot(eval_val, val_accuracy, 'b--')
    plt.title('Accuracy Per Generation')
    plt.legend(('Train', 'Test'))
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(R.picName)
    plt.show()


def net_evaluation(sess, loss, accuracy, val_images_batch, val_labels_batch, val_nums):
    val_max_steps = int(val_nums / R.batch_size)
    val_losses, val_accs = [], []
    for _ in range(val_max_steps):
        val_x, val_y = sess.run([val_images_batch, val_labels_batch])
        # val_loss = sess.run(loss, feed_dict={x: val_x, y: val_y, keep_prob: 1.0})
        # val_acc = sess.run(accuracy, feed_dict={x: val_x, y: val_y, keep_prob: 1.0})
        val_loss, val_acc = sess.run([loss, accuracy],
                                     feed_dict={input_images: val_x, input_labels: val_y, is_training: False})
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    mean_loss = np.array(val_losses, dtype=np.float32).mean()
    mean_acc = np.array(val_accs, dtype=np.float32).mean()
    return mean_loss, mean_acc


def step_train(train_op, loss, accuracy, train_images_batch, train_labels_batch, max_steps, train_log_step,
               val_images_batch, val_labels_batch, val_nums, val_log_step, snapshot_prefix, snapshot, start=0):
    '''
    循环迭代训练过程
    :param train_op: 训练op
    :param loss:     loss函数
    :param accuracy: 准确率函�?
    :param train_images_batch: 训练images数据
    :param train_labels_batch: 训练labels数据
    :param train_nums:         总训练数�?
    :param train_log_step:     训练log显示间隔
    :param val_images_batch:   验证images数据
    :param val_labels_batch:   验证labels数据
    :param val_nums:           总验证数�?
    :param val_log_step:       验证log显示间隔
    :param snapshot_prefix:    模型保存的路�?
    :param snapshot:           模型保存间隔
    :return: None
    '''
    saver = tf.train.Saver()
    max_acc = 0.0
    train_losses, val_losses = [], []
    train_accuracy, val_accuracy = [], []
    with tf.Session() as sess:
        # find whether checkpoint exists so as to continue training
        ckpt = tf.train.get_checkpoint_state(R.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Restore from previous checkpoint.")
        else:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

        # 启动多线�?
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(start, max_steps + 1):  # generation times
            batch_input_images, batch_input_labels = sess.run([train_images_batch, train_labels_batch])
            _, train_loss = sess.run([train_op, loss], feed_dict={input_images: batch_input_images,
                                                                  input_labels: batch_input_labels,
                                                                  is_training: True})
            # train测试(仅测试训练集1-batch)
            if i % train_log_step == 0:
                train_acc = sess.run(accuracy, feed_dict={input_images: batch_input_images,
                                                          input_labels: batch_input_labels,
                                                          is_training: False})
                train_losses.append(train_loss)
                train_accuracy.append(train_acc)
                msg = "Step [%d]  train Loss : %f, training   accuracy :  %g" % (i, train_loss, train_acc)
                print('%s ' % datetime.datetime.now() + msg)
                logger.info(msg)

            # val测试(测试全部val数据)
            if i % val_log_step == 0:
                mean_loss, mean_acc = net_evaluation(sess, loss, accuracy, val_images_batch, val_labels_batch, val_nums)
                val_losses.append(mean_loss)
                val_accuracy.append(mean_acc)
                msg = "Step [%d]  val   Loss : %f, validation accuracy :  %g" % (i, mean_loss, mean_acc)
                print('%s ' % datetime.datetime.now() + msg)
                logger.info(msg)

            # 模型保存:每迭代snapshot次或者最后一次保存
            if (i % snapshot == 0 and i > 0) or i == R.max_steps:
                print("Save model as " + snapshot_prefix.split(os.sep)[-1] + " !" * 3)
                saver.save(sess, snapshot_prefix, global_step=i)
            # 保存val准确率最高的模型
            if mean_acc > max_acc and mean_acc > R.save_min_accu:
                max_acc = mean_acc
                path = os.path.dirname(snapshot_prefix)
                best_models = os.path.join(path, 'best_models_{}_{:.4f}.ckpt'.format(i, max_acc))
                saver.save(sess, best_models)
        # 关闭多线
        coord.request_stop()
        coord.join(threads)
        resultpic((train_log_step, val_log_step), (train_losses, val_losses), (train_accuracy, val_accuracy))


def train(train_record_file, train_log_step, train_param, val_record_file,
          val_log_step, num_classes, data_shape, snapshot, snapshot_prefix):
    '''
    :param train_record_file: 训练的tfrecord文件
    :param train_log_step:    显示训练过程log信息间隔
    :param train_param:       train参数
    :param val_record_file:   验证的tfrecord文件
    :param val_log_step:      显示验证过程log信息间隔
    :param val_param:         val参数
    :param num_classes:       labels�?
    :param data_shape:        输入数据shape
    :param snapshot:          保存模型间隔
    :param snapshot_prefix:   保存模型文件的前缀�?
    :return:
    '''
    [base_lr, max_steps] = train_param
    [batch_size, resize_height, resize_width, depths] = data_shape

    # 获得训练和测试的样本
    train_nums = get_example_nums(train_record_file)
    val_nums = get_example_nums(val_record_file)
    print('Train-nums : %d and Validation-nums : %d' % (train_nums, val_nums))

    # 从record中读取图片和labels数据
    # train数据,训练数据一般要求打乱顺序shuffle=True
    train_images, train_labels = read_records(train_record_file, resize_height, resize_width, type='normalization',
                                              is_train=None)
    train_images_batch, train_labels_batch = get_batch_images(train_images, train_labels, batch_size=batch_size,
                                                              labels_nums=num_classes, one_hot=True, shuffle=True)
    # val数据,验证数据可以不需要打乱数
    val_images, val_labels = read_records(val_record_file, resize_height, resize_width, type='normalization',
                                          is_train=None)
    val_images_batch, val_labels_batch = get_batch_images(val_images, val_labels, batch_size=batch_size,
                                                          labels_nums=num_classes, one_hot=True, shuffle=False)
    # ============================================================================================================
    # Define the model: [core]
    with slim.arg_scope(mobilenet_v2.training_scope(dropout_keep_prob=R.dropout)):
        out, end_points = mobilenet_v2.mobilenet(input_tensor=input_images, num_classes=num_classes,
                                                 depth_multiplier=R.depth_multiplier, is_training=is_training)

    # Specify the loss function: tf.losses定义的loss函数都会自动添加到loss函数, 无需 # slim.losses.add_loss(my_loss)
    tf.losses.softmax_cross_entropy(onehot_labels=input_labels, logits=out)  # 添加交叉熵损失loss=1.6
    loss = tf.losses.get_total_loss(add_regularization_losses=True)  # 添加正则化损失loss=2.2
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(input_labels, 1)), tf.float32))

    # Specify the optimization scheme:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=base_lr)
    '''
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.05, global_step, 150, 0.9) 
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    train_tensor = optimizer.minimize(loss, global_step)
    train_op = slim.learning.create_train_op(loss, optimizer, global_step=global_step)
    '''
    # 在定义训练的时�? 注意到我们使用了`batch_norm`层时,需要更新每一层的`average`和`variance`参数,
    # 更新的过程不包含在正常的训练过程�? 需要我们去手动像下面这样更�?    # 通过`tf.get_collection`获得所有需要更新的`op`
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # 使用`tensorflow`的控制流, 先执行更新算�? 再执行训�?
    # with tf.control_dependencies(update_ops):
    # create_train_op that ensures that when we evaluate it to get the loss,
    # the update_ops are done and the gradient updates are computed.
    train_op = slim.learning.create_train_op(total_loss=loss, optimizer=optimizer)

    # 循环迭代过程
    step_train(train_op, loss, accuracy, train_images_batch, train_labels_batch, max_steps, train_log_step,
               val_images_batch, val_labels_batch, val_nums, val_log_step, snapshot_prefix, snapshot)
    # ================================================================================================================


def train_kfold(record_file, train_log_step, train_param, val_log_step, num_classes, data_shape, snapshot, snapshot_prefix):
    [base_lr, max_steps] = train_param
    [batch_size, resize_height, resize_width, depths] = data_shape
    # ============================================================================================================
    # Define the model: [core]
    with slim.arg_scope(mobilenet_v2.training_scope(dropout_keep_prob=R.dropout)):
        out, end_points = mobilenet_v2.mobilenet(input_tensor=input_images, num_classes=num_classes,
                                                 depth_multiplier=R.depth_multiplier, is_training=is_training)

    # Specify the loss function: tf.losses定义的loss函数都会自动添加到loss函数, 无需 # slim.losses.add_loss(my_loss)
    tf.losses.softmax_cross_entropy(onehot_labels=input_labels, logits=out)  # 添加交叉熵损失loss=1.6
    loss = tf.losses.get_total_loss(add_regularization_losses=True)  # 添加正则化损失loss=2.2
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(input_labels, 1)), tf.float32))

    # Specify the optimization scheme:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=base_lr)
    '''
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.05, global_step, 150, 0.9) 
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    train_tensor = optimizer.minimize(loss, global_step)
    train_op = slim.learning.create_train_op(loss, optimizer, global_step=global_step)
    '''
    # 在定义训练的时�? 注意到我们使用了`batch_norm`层时,需要更新每一层的`average`和`variance`参数,
    # 更新的过程不包含在正常的训练过程�? 需要我们去手动像下面这样更�?    # 通过`tf.get_collection`获得所有需要更新的`op`
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # 使用`tensorflow`的控制流, 先执行更新算�? 再执行训�?
    # with tf.control_dependencies(update_ops):
    # create_train_op that ensures that when we evaluate it to get the loss,
    # the update_ops are done and the gradient updates are computed.
    train_op = slim.learning.create_train_op(total_loss=loss, optimizer=optimizer)
    # ================================================================================================================
    # 从record中读取图片和labels数据
    all_nums = get_example_nums(record_file)
    all_images, all_labels = read_records(record_file, resize_height, resize_width, type='normalization', is_train=None)
    all_images_batch, all_labels_batch = get_batch_images(all_images, all_labels, batch_size=batch_size,
                                                          labels_nums=num_classes, one_hot=True, shuffle=True)


if __name__ == '__main__':
    print('run_train package:', __package__, 'name:', __name__)
    data_shape = [R.batch_size, R.resize_height, R.resize_width, R.depths]
    train_param = [R.base_lr, R.max_steps]
    '''
    train_kfold(record_file=R.record_file,
                train_log_step=R.train_log_step,
                train_param=train_param,
                val_log_step=R.val_log_step,
                num_classes=R.num_classes,
                data_shape=data_shape,
                snapshot=R.snapshot,
                snapshot_prefix=R.snapshot_prefix)
    '''
    train(train_record_file=R.train_record_file,
          train_log_step=R.train_log_step,
          train_param=train_param,
          val_record_file=R.val_record_file,
          val_log_step=R.val_log_step,
          num_classes=R.num_classes,
          data_shape=data_shape,
          snapshot=R.snapshot,
          snapshot_prefix=R.snapshot_prefix)
