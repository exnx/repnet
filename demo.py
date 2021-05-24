import tensorflow.compat.v2 as tf
from model import ResnetPeriodEstimator
import numpy as np
import base64
import io
import os
import time

from PIL import Image
import cv2
from scipy.signal import medfilt
# ! pip install youtube_dl
import youtube_dl

import matplotlib
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

from IPython.display import display
from IPython.display import HTML
from IPython.display import Javascript




def get_repnet_model(logdir):
  """Returns a trained RepNet model.

  Args:
    logdir (string): Path to directory where checkpoint will be downloaded.

  Returns:
    model (Keras model): Trained RepNet model.
  """
  # Check if we are in eager mode.
  assert tf.executing_eagerly()

  # Models will be called in eval mode.
  tf.keras.backend.set_learning_phase(0)

  # Define RepNet model.
  model = ResnetPeriodEstimator()
  # tf.function for speed.
  model.call = tf.function(model.call)

  # Define checkpoint and checkpoint manager.
  ckpt = tf.train.Checkpoint(model=model)
  ckpt_manager = tf.train.CheckpointManager(
      ckpt, directory=logdir, max_to_keep=10)
  latest_ckpt = ckpt_manager.latest_checkpoint
  print('Loading from: ', latest_ckpt)
  if not latest_ckpt:
    raise ValueError('Path does not have a checkpoint to load.')
  # Restore weights.
  ckpt.restore(latest_ckpt).expect_partial()

  # Pass dummy frames to build graph.
  model(tf.random.uniform((1, 64, 112, 112, 3)))
  return model


def unnorm(query_frame):
  min_v = query_frame.min()
  max_v = query_frame.max()
  query_frame = (query_frame - min_v) / max(1e-7, (max_v - min_v))
  return query_frame


def create_count_video(frames,
                       per_frame_counts,
                       within_period,
                       score,
                       fps,
                       output_file,
                       delay,
                       plot_count=True,
                       plot_within_period=False,
                       plot_score=False):
  """Creates video with running count and within period predictions.

  Args:
    frames (List): List of images in form of NumPy arrays.
    per_frame_counts (List): List of floats indicating repetition count for
      each frame. This is the rate of repetition for that particular frame.
      Summing this list up gives count over entire video.
    within_period (List): List of floats indicating score between 0 and 1 if the
      frame is inside the periodic/repeating portion of a video or not.
    score (float): Score between 0 and 1 indicating the confidence of the
      RepNet model's count predictions.
    fps (int): Frames per second of the input video. Used to scale the
      repetition rate predictions to Hz.
    output_file (string): Path of the output video.
    delay (integer): Delay between each frame in the output video.
    plot_count (boolean): if True plots the count in the output video.
    plot_within_period (boolean): if True plots the per-frame within period
      scores.
    plot_score (boolean): if True plots the confidence of the model along with
      count ot within_period scores.
  """
  if output_file[-4:] not in ['.mp4', '.gif']:
    raise ValueError('Output format can only be mp4 or gif')
  num_frames = len(frames)

  running_counts = np.cumsum(per_frame_counts)
  final_count = running_counts[-1]

  def count(idx):
    return int(np.round(running_counts[idx]))

  def rate(idx):
    return per_frame_counts[idx] * fps

  if plot_count and not plot_within_period:
    fig = plt.figure(figsize=(10, 12), tight_layout=True)
    im = plt.imshow(unnorm(frames[0]))
    if plot_score:
      plt.suptitle('Pred Count: %d, '
                   'Prob: %0.1f' % (int(np.around(final_count)), score),
                   fontsize=24)

    plt.title('Count 0, Rate: 0', fontsize=24)
    plt.axis('off')
    plt.grid(b=None)
    def update_count_plot(i):
      """Updates the count plot."""
      im.set_data(unnorm(frames[i]))
      plt.title('Count %d, Rate: %0.4f Hz' % (count(i), rate(i)), fontsize=24)

    anim = FuncAnimation(
        fig,
        update_count_plot,
        frames=np.arange(1, num_frames),
        interval=delay,
        blit=False)
    if output_file[-3:] == 'mp4':
      anim.save(output_file, dpi=100, fps=24)
    elif output_file[-3:] == 'gif':
      anim.save(output_file, writer='imagemagick', fps=24, dpi=100)

  elif plot_within_period:
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    im = axs[0].imshow(unnorm(frames[0]))
    axs[1].plot(0, within_period[0])
    axs[1].set_xlim((0, len(frames)))
    axs[1].set_ylim((0, 1))

    if plot_score:
      plt.suptitle('Pred Count: %d, '
                   'Prob: %0.1f' % (int(np.around(final_count)), score),
                   fontsize=24)

    if plot_count:
      axs[0].set_title('Count 0, Rate: 0', fontsize=20)

    plt.axis('off')
    plt.grid(b=None)

    def update_within_period_plot(i):
      """Updates the within period plot along with count."""
      im.set_data(unnorm(frames[i]))
      axs[0].set_xticks([])
      axs[0].set_yticks([])
      xs = []
      ys = []
      if plot_count:
        axs[0].set_title('Count %d, Rate: %0.4f Hz' % (count(i), rate(i)),
                         fontsize=20)
      for idx in range(i):
        xs.append(idx)
        ys.append(within_period[int(idx * len(within_period) / num_frames)])
      axs[1].clear()
      axs[1].set_title('Within Period or Not', fontsize=20)
      axs[1].set_xlim((0, num_frames))
      axs[1].set_ylim((-0.05, 1.05))
      axs[1].plot(xs, ys)

    anim = FuncAnimation(
        fig,
        update_within_period_plot,
        frames=np.arange(1, num_frames),
        interval=delay,
        blit=False,
    )
    if output_file[-3:] == 'mp4':
      anim.save(output_file, dpi=100, fps=24)
    elif output_file[-3:] == 'gif':
      anim.save(output_file, writer='imagemagick', fps=24, dpi=100)

  plt.close()


def show_video(video_path):
  mp4 = open(video_path, 'rb').read()
  data_url = 'data:video/mp4;base64,' + base64.b64encode(mp4).decode()
  return HTML("""<video width=600 controls>
      <source src="%s" type="video/mp4"></video>
  """ % data_url)



def read_video(video_filename, width=224, height=224):
  """Read video from file."""
  cap = cv2.VideoCapture(video_filename)
  fps = cap.get(cv2.CAP_PROP_FPS)
  frames = []
  if cap.isOpened():
    while True:
      success, frame_bgr = cap.read()
      if not success:
        break
      frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
      frame_rgb = cv2.resize(frame_rgb, (width, height))
      frames.append(frame_rgb)
  frames = np.asarray(frames)
  return frames, fps



def get_score(period_score, within_period_score):
  """Combine the period and periodicity scores."""
  within_period_score = tf.nn.sigmoid(within_period_score)[:, 0]
  per_frame_periods = tf.argmax(period_score, axis=-1) + 1
  pred_period_conf = tf.reduce_max(
      tf.nn.softmax(period_score, axis=-1), axis=-1)
  pred_period_conf = tf.where(
      tf.math.less(per_frame_periods, 3), 0.0, pred_period_conf)
  within_period_score *= pred_period_conf
  within_period_score = np.sqrt(within_period_score)
  pred_score = tf.reduce_mean(within_period_score)
  return pred_score, within_period_score


def get_counts(model, frames, strides, batch_size,
               threshold,
               within_period_threshold,
               constant_speed=False,
               median_filter=False,
               fully_periodic=False):
  """Pass frames through model and conver period predictions to count."""
  seq_len = len(frames)
  raw_scores_list = []
  scores = []
  within_period_scores_list = []

  if fully_periodic:
    within_period_threshold = 0.0

  frames = model.preprocess(frames)

  for stride in strides:
    num_batches = int(np.ceil(seq_len/model.num_frames/stride/batch_size))
    raw_scores_per_stride = []
    within_period_score_stride = []
    for batch_idx in range(num_batches):
      idxes = tf.range(batch_idx*batch_size*model.num_frames*stride,
                       (batch_idx+1)*batch_size*model.num_frames*stride,
                       stride)
      idxes = tf.clip_by_value(idxes, 0, seq_len-1)
      curr_frames = tf.gather(frames, idxes)
      curr_frames = tf.reshape(
          curr_frames,
          [batch_size, model.num_frames, model.image_size, model.image_size, 3])

      raw_scores, within_period_scores, _ = model(curr_frames)
      raw_scores_per_stride.append(np.reshape(raw_scores.numpy(),
                                              [-1, model.num_frames//2]))
      within_period_score_stride.append(np.reshape(within_period_scores.numpy(),
                                                   [-1, 1]))
    raw_scores_per_stride = np.concatenate(raw_scores_per_stride, axis=0)
    raw_scores_list.append(raw_scores_per_stride)
    within_period_score_stride = np.concatenate(
        within_period_score_stride, axis=0)
    pred_score, within_period_score_stride = get_score(
        raw_scores_per_stride, within_period_score_stride)
    scores.append(pred_score)
    within_period_scores_list.append(within_period_score_stride)

  # Stride chooser
  argmax_strides = np.argmax(scores)
  chosen_stride = strides[argmax_strides]
  raw_scores = np.repeat(
      raw_scores_list[argmax_strides], chosen_stride, axis=0)[:seq_len]
  within_period = np.repeat(
      within_period_scores_list[argmax_strides], chosen_stride,
      axis=0)[:seq_len]
  within_period_binary = np.asarray(within_period > within_period_threshold)
  if median_filter:
    within_period_binary = medfilt(within_period_binary, 5)

  # Select Periodic frames
  periodic_idxes = np.where(within_period_binary)[0]

  if constant_speed:
    # Count by averaging predictions. Smoother but
    # assumes constant speed.
    scores = tf.reduce_mean(
        tf.nn.softmax(raw_scores[periodic_idxes], axis=-1), axis=0)
    max_period = np.argmax(scores)
    pred_score = scores[max_period]
    pred_period = chosen_stride * (max_period + 1)
    per_frame_counts = (
        np.asarray(seq_len * [1. / pred_period]) *
        np.asarray(within_period_binary))
  else:
    # Count each frame. More noisy but adapts to changes in speed.
    pred_score = tf.reduce_mean(within_period)  # an avg
    per_frame_periods = tf.argmax(raw_scores, axis=-1) + 1  # I WANT THIS!!!
    per_frame_counts = tf.where(tf.math.less(per_frame_periods, 3), 0.0, tf.math.divide(1.0, tf.cast(chosen_stride * per_frame_periods, tf.float32)),)
    if median_filter:
      per_frame_counts = medfilt(per_frame_counts, 5)

    per_frame_counts *= np.asarray(within_period_binary)  # only uses counts from frames within period binary
    # per_frame_period *= np.asarray(within_period_binary)  # only uses periods from frames within period binary 
    # per_frame_period *= 2  # then this divided by the fps gives you the cpm rate!

    pred_period = seq_len/np.sum(per_frame_counts)  # i think this is avg period 

  if pred_score < threshold:
    print('No repetitions detected in video as score '
          '%0.2f is less than threshold %0.2f.'%(pred_score, threshold))
    per_frame_counts = np.asarray(len(per_frame_counts) * [0.])

  return (pred_period, pred_score, within_period,
          per_frame_counts, chosen_stride)