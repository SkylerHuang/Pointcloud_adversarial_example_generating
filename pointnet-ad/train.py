import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=3, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--need_label', type=int, default=5, help='original label')
parser.add_argument('--target_label', type=int, default=0, help='target_label')
parser.add_argument('--pretrain_model', default='log/pretrained_model/model.ckpt', help='pretrain_model dir [default:log/pretrained_model/model.ckpt]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=101, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
PRETRAIN_MODEL = FLAGS.pretrain_model
NEED_LABEL = FLAGS.need_label
TARGET_LABEL = FLAGS.target_label

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 2048
NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train(need_label,target_label):
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)


            batch = tf.get_variable('batch',[],initializer=tf.constant_initializer(0),trainable=False)

            #batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred, pert_vec ,end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss ,pert_loss = MODEL.get_adversarial_loss(pred, labels_pl, pert_vec)
            print(loss.shape)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            #add new train parameter
            var_list = tf.contrib.framework.get_variables('epsilon')

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            #Only train the perturbation parameter
            train_op = optimizer.minimize(loss, global_step=batch,var_list = var_list)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        #Parameters loaded from the pre-trained model
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['epsilon','batch'])
        saver_old = tf.train.Saver(variables_to_restore)

        # Init variables
        init = tf.global_variables_initializer()

        sess.run(init, {is_training_pl: True})
        saver_old.restore(sess, PRETRAIN_MODEL)
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'pert_vec': pert_vec,
               'loss': loss,
               'pert_loss': pert_loss,
               'train_op': train_op,
               'merged': merged,
               'lr':learning_rate,
               'step': batch}
        pert = []
        raw_pc = []
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            pc, res = train_adversarial_one_epoch(sess, ops, train_writer, need_label, target_label)
            #eval_one_epoch(sess, ops, test_writer)
            pert.append(res)
            # Save the variables to disk.
            if epoch % 50 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)
        raw_pc.append(pc)
        np.savez('pert.npz', pert = pert, pc = raw_pc)

def train_adversarial_one_epoch(sess, ops, train_writer, need_label, target_label):
    is_training = False
    # Merge all train file
    #train_file_idxs = np.arange(0,len(TRAIN_FILES))
    all_train_data, all_train_label = provider.loadDataFile(TRAIN_FILES[0])#None, None

    #for fn in range(len(TRAIN_FILES)):
    #    tmp_data, tmp_label = provider.loadDataFile(TRAIN_FILES[fn])
    #    all_train_data = merge(all_train_data,tmp_data)
    #    all_train_label = merge(all_train_label,tmp_label)

    need_data = all_train_data[np.reshape(all_train_label == need_label,(-1)),...]
    need_data = need_data[:BATCH_SIZE,:NUM_POINT,:]

    target_label = np.tile(np.array([target_label]),[BATCH_SIZE,1])
    target_label = np.squeeze(target_label)

    loss_sum = 0
    feed_dict = {ops['pointclouds_pl']: need_data,
                 ops['labels_pl']: target_label,
                 ops['is_training_pl']: is_training,}

    summary, step, _, loss_val, pert_loss_val, pred_val, pert_vec ,lr_val= sess.run([ops['merged'], ops['step'],ops['train_op'], ops['loss'], ops['pert_loss'], ops['pred'], ops['pert_vec'], ops['lr']], feed_dict=feed_dict)
    train_writer.add_summary(summary, step)
    pred_val = np.argmax(pred_val, 1)
    correct = np.sum(pred_val == target_label)
    loss_sum += loss_val

    log_string('mean loss: %f' % (loss_sum / 1.0))
    log_string('pert loss: %f' % (pert_loss_val / 1.0))
    log_string('accuracy: %f' % (correct / float(BATCH_SIZE)))
    log_string(' lr: %f' % (lr_val / 1.0))
    log_string(' batch: %f' % (step / 1.0))
    return need_data, pert_vec

def merge(main_data, attach_data):
    if main_data is None:
        main_data = attach_data
    else:
        main_data = np.concatenate((main_data,attach_data), axis = 0)
    return main_data


if __name__ == "__main__":
    train(NEED_LABEL,TARGET_LABEL)
    LOG_FOUT.close()
