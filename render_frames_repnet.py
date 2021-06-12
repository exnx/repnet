import argparse
import json
import os
import pandas as pd
import numpy as np
import cv2



def read_json(path):
    try:
        with open(path) as f:
            data_dict = json.load(f)

    except Exception as e:
        print(e)
        raise Exception

    return data_dict


def save_frames(frames, prefix, frame_names):

    for i in range(len(frames)):
        frame_path = os.path.join(prefix, frame_names[i])

        cv2.imwrite(frame_path, frames[i])

    return


def write_on_frames(frames,
                    prev_rep_count,
                    count_per_frame,
                    rate_pred_per_frame,
                    rate_pred_avg,
                    rate_label,
                    count_label,
                    fps=16,
                    base_rate=109):

    '''

    :param frames: list of frames
    :param output: float, is the speed factor
    :param rate_label: float, ground truth rate
    :return:
    '''

    width = 640
    height = 360
    font = cv2.FONT_HERSHEY_SIMPLEX  # font

    rate_label_loc = (25, 50)
    rate_avg_loc = (25, 75)
    rate_pred_loc = (25, 100)
    # output_loc = (25, 125)
    count_loc = (25, 125)
    # count_label_loc = (50, 175)

    fontScale = 1  # fontScale
    thickness = 2  # Line thickness of 2 px

    green = (0, 255, 0)

    frames_with_text = []

    rate_label_text = 'Avg label: {:.1f}'.format(rate_label)
    rate_avg_text = 'Avg pred: {:.1f}'.format(rate_pred_avg)

    font_color = green

    curr_rep_count = prev_rep_count

    for i, frame in enumerate(frames):
        rate_pred = rate_pred_per_frame[i]
        rate_pred_text = 'Inst. pred: {}'.format(round(rate_pred))
        # calculate current count for frame, round down
        curr_rep_count += count_per_frame[i]

        # import pdb; pdb.set_trace()

        count_text = 'count: {}/{}'.format(int(curr_rep_count), int(count_label))

        resized = cv2.resize(frame, (width, height))

        cv2.putText(resized, rate_pred_text, rate_pred_loc, font,
                          fontScale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(resized, rate_avg_text, rate_avg_loc, font,
                          fontScale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(resized, rate_label_text, rate_label_loc, font,
                          fontScale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(resized, count_text, count_loc, font,
                          fontScale, font_color, thickness, cv2.LINE_AA)

        frames_with_text.append(resized)

    return frames_with_text


def get_frames(prefix, segment):
    ext = ".jpeg"

    images = []
    img_names = []

    for i in range(segment[0], segment[1]):
        name = str(i).zfill(5) + ext
        file_path = os.path.join(prefix, name)

        try:
            img = cv2.imread(file_path)
            if img is not None:
                images.append(img)
                img_names.append(name)
            else:
                print('img is not', file_path)

        except Exception as e:
            print('cannot open img:'.format(file_path))
            print(e)

    return images, img_names


def main(args):
    '''
    loop thru videos
        loop thru pred/target
            get frame list
            retrieve frames
            write target/labels on frames
            save frames to path
    '''


    results_dir = args.results_dir
    results_non_zero = args.results_non_zero
    out_dir = args.out_dir
    fps = args.fps
    window = args.window
    meta_dir = args.meta_dir

    results_json = read_json(results_dir)['video_results']
    meta_json = read_json(meta_dir)

    if results_non_zero is not None:
        results_non_zero_json = read_json(results_non_zero)['video_results']

    # loop thru video
    for video_id, result_json in results_json.items():

        print('processing frames:', video_id)

        prefix = os.path.join(out_dir, video_id)
        os.makedirs(prefix, exist_ok=True)  # make a new dir for each video
        frame_dir = os.path.join(args.frame_dir, video_id)

        count_label = result_json['count_label']
        rate_label = result_json['rate_label']  # for whole clip

        if results_non_zero is not None:
            video_rate_pred_avg = results_non_zero_json[video_id]['rate_avg_pred']
        else:
            video_rate_pred_avg = result_json['rate_avg_pred']

        per_frame_rate = result_json['per_frame_rate']
        per_frame_count = np.asarray(per_frame_rate) / (fps * 60)

        # need to get either prediction len or meta, and get the min
        num_frames = min(len(per_frame_rate), meta_json[video_id]['num_new_frames'])

        curr_rep_count = 0

        # write on frames by window size batches
        for i in range(0, num_frames, window):
            # retrieve frames
            start = i
            end = min(i + window, num_frames)
            frames, frame_names = get_frames(frame_dir, [start, end])
            rate_pred_per_frame = per_frame_rate[start:end]
            count_per_frame = per_frame_count[start:end]

            # write on frames (all with same text)
            frames_with_text = write_on_frames(frames, curr_rep_count, count_per_frame, rate_pred_per_frame,
                                               video_rate_pred_avg, rate_label, count_label, fps)

            save_frames(frames_with_text, prefix, frame_names)

            # import pdb; pdb.set_trace()

            # make sure to do this after writing/saving frames
            curr_rep_count = curr_rep_count + sum(count_per_frame)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--results-dir', help='dir to the inference results file')
    parser.add_argument('-rn', '--results-non-zero', default=None, help='dir non results file, used for more accurate avgs')
    parser.add_argument('-v', '--frame-dir', help='root dir to all frames')
    parser.add_argument('-m', '--meta-dir', help='path to metadata')
    parser.add_argument('-o', '--out-dir', help='path to the output dir')
    parser.add_argument('-f', '--fps', default=16, type=int, help='fps')
    parser.add_argument('-w', '--window', default=24, type=int, help='sliding window size')

    args = parser.parse_args()

    main(args)

'''

python render_frames_repnet.py \
--results-dir /vision2/u/enguyen/demos/rate_pred/repnet/inference9_all/test_results.json \
--frame-dir /scr-ssd/enguyen/normal_1.0x/frames_fps30 \
--out-dir /vision2/u/enguyen/demos/rate_pred/repnet_frames/v2 \
--results-non-zero /vision2/u/enguyen/demos/rate_pred/repnet/inference7_nonzero/test_results.json \
--meta-dir /scr-ssd/enguyen/normal_1.0x/frames_fps30/meta/video_metadata.json \
--window 30 \
--fps 30

'''

