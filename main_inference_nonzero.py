import argparse
import json
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from functools import singledispatch

from demo import get_counts, read_video, get_repnet_model
from sklearn.metrics import precision_recall_fscore_support as score

# Threshold to consider periodicity in entire video.
THRESHOLD = 0.1#@param {type:"number"}

# Threshold to consider periodicity for individual frames in video.
WITHIN_PERIOD_THRESHOLD = 0.1#@param {type:"number"}

# Use this setting for better results when it is
# known action is repeating at constant speed.
CONSTANT_SPEED = False#@param {type:"boolean"}

# Use median filtering in time to ignore noisy frames.
MEDIAN_FILTER = True#@param {type:"boolean"}

# Use this setting for better results when it is
# known the entire video is periodic/reapeating and
# has no aperiodic frames.
FULLY_PERIODIC = True#@param {type:"boolean"}

# Plot score in visualization video.
PLOT_SCORE = False#@param {type:"boolean"}

# Visualization video's FPS.
VIZ_FPS = 30#@param {type:"integer"}

def read_csv_as_df(path):
    df = pd.read_csv(path)
    return df

def read_json(path):
    with open(path) as f:
        data_dict = json.load(f)
    return data_dict

def write_json(data_dict, path, indent=4):

    '''

    :param data_dict: dict to write to json
    :param path: path to json
    :param indent: for easy viewing.  Use None if you want to save a lot of space
    :return:
    '''

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=indent, default=to_serializable)


@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)

@to_serializable.register(np.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return np.float64(val)


def convert_label(rate_np):
    low_bar = 100
    high_bar = 120

    rate_by_class = np.where(rate_np < low_bar, 0, rate_np)
    rate_by_class = np.where(rate_by_class > high_bar, 2, rate_by_class)
    rate_by_class = np.where((rate_by_class <= high_bar) & (rate_by_class >= low_bar), 1, rate_by_class)

    return rate_by_class


def run_classification(per_frame_rate, rate_label):

    rate_np = np.zeros(len(per_frame_rate))
    rate_np[:] = rate_label

    # constant = np.zeros(len(per_frame_rate))
    # constant[:] = 111.9

    y_test = convert_label(rate_label)
    y_pred = convert_label(per_frame_rate)

    precision, recall, fscore, support = score(y_test, y_pred)

    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))

    # import pdb; pdb.set_trace()

    # print(f1_score(y_test, y_pred, average="macro"))
    # print(precision_score(y_test, y_pred, average="macro"))
    # print(recall_score(y_test, y_pred, average="macro"))

    return (precision, recall, fscore, support)


def main(args):

    model_dir = args.model_dir
    model = get_repnet_model(model_dir)

    video_dir = args.video_dir
    out_dir = args.out_dir
    rate_labels_dir = args.rate_labels
    remove_zeros = args.remove_zeros

    os.makedirs(out_dir, exist_ok=True)

    # new for using labels
    rate_labels_df = read_csv_as_df(rate_labels_dir)
    rate_video_ids = rate_labels_df['video_id'].tolist()
    rate_labels = rate_labels_df['rate'].tolist()
    count_labels = rate_labels_df['count'].tolist()

    mae = 0
    avg_count_diff = 0
    num_videos = len(rate_video_ids)
    curr_num_video_count = 0

    all_results = {}

    # need to aggregate these for all videos
    all_per_frame_rate = None
    all_rate_label = None

    for i, video_id in enumerate(rate_video_ids):
        video_result = {}

        video_name = video_id + ".mp4"  # get path to video
        video_path = os.path.join(video_dir, video_name)

        imgs, vid_fps = read_video(video_path)  # figure out how they get frames

        orig_num_frames = len(imgs)

        if orig_num_frames < 300:
            continue

        print('Running RepNet on {}...'.format(video_id))
        (pred_period, pred_score, within_period,
            per_frame_counts, chosen_stride) = get_counts(
                model,
                imgs,
                strides=[1,2,3,4],
                batch_size=20,
                threshold=THRESHOLD,
                within_period_threshold=WITHIN_PERIOD_THRESHOLD,
                constant_speed=CONSTANT_SPEED,
                median_filter=MEDIAN_FILTER,
                fully_periodic=FULLY_PERIODIC)

        if remove_zeros:
            # need to remove all non zero elements from repnet output
            per_frame_counts = per_frame_counts[per_frame_counts != 0]
            num_frames = len(per_frame_counts)
        else:
            num_frames = orig_num_frames

        rate_label = rate_labels[i]
        per_frame_rate = per_frame_counts * VIZ_FPS * 60

        # import pdb; pdb.set_trace()
        rate_avg_pred = np.average(per_frame_rate)
        video_mae_rate = sum(np.absolute(per_frame_rate-rate_label)) / num_frames
        mae += video_mae_rate

        count_pred = sum(per_frame_counts)
        count_label = count_labels[i]
        count_diff = abs(count_label - count_pred)
        avg_count_diff += count_diff

        print('video_id {}, num frames {}'.format(video_id, num_frames))
        print('video MAE rate {:.1f}'.format(video_mae_rate))
        print('pred count {:.1f}'.format(count_pred))
        print('diff count {:.1f}'.format(count_label - count_pred))

        video_result['per_frame_rate'] = per_frame_rate.tolist()
        # we removed zero elements, so saving the original num here
        video_result['orig_num_frames'] = num_frames
        video_result['count_pred'] = count_pred
        video_result['count_label'] = count_label
        video_result['count_diff'] = count_diff
        video_result['rate_label'] = rate_label
        video_result['rate_avg_pred'] = rate_avg_pred
        video_result['rate_mae'] = video_mae_rate

        all_results[video_id] = video_result

        rate_label_np = np.zeros(len(per_frame_rate))
        rate_label_np[:] = rate_label

        if all_per_frame_rate is None:
            all_per_frame_rate = per_frame_rate
            all_rate_label = rate_label_np
        else:

            try:
                all_per_frame_rate = np.concatenate((all_per_frame_rate, per_frame_rate))
                all_rate_label = np.concatenate((all_rate_label, rate_label_np))
            except:
                import pdb; pdb.set_trace()

        curr_num_video_count += 1

    # run 3 level classification
    precision, recall, fscore, support = run_classification(all_per_frame_rate, all_rate_label)
    classification_results = {'precision': precision, 'recall': recall, 'fscore': fscore, 'support': support}

    mae /= curr_num_video_count
    avg_count_diff /= curr_num_video_count

    print('\n')
    print('num videos:', curr_num_video_count)
    print('MAE: {:.2f}'.format(mae))
    print('avg count diff: {:.2f}'.format(avg_count_diff))

    results = {'video_results': all_results, 'avg_count_diff': avg_count_diff, 'mae': mae, 'classification': classification_results}

    # out_path = os.path.join(out_dir, 'test_results.json')
    # write_json(results, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video-dir', help='root dir to all frames')
    parser.add_argument('-rl', '--rate-labels', default=None, help='path to rate labels')
    parser.add_argument('-o', '--out-dir', help='path to the output dir')
    parser.add_argument('-m', '--model-dir', help='path to the output dir')
    parser.add_argument('-rz', '--remove-zeros', action='store_true', help='remove zero elements from metrics')

    args = parser.parse_args()

    main(args)

'''

python main_inference_nonzero.py \
--video-dir /vision2/u/enguyen/mini_cba/clipped_videos_fps30 \
--rate-labels /vision2/u/enguyen/cpr-detection/post_processing_code/data/432/rate_labels_corrected.csv \
--out-dir /vision2/u/enguyen/demos/rate_pred/repnet/inference7 \
--model-dir /vision2/u/enguyen/pretrained_models/repnet/ \
--remove-zeros

'''




