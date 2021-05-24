from demo import get_counts, create_count_video, show_video, download_video_from_url, read_video

##@title 

# FPS while recording video from webcam.
WEBCAM_FPS = 16#@param {type:"integer"}

# Time in seconds to record video on webcam. 
RECORDING_TIME_IN_SECONDS = 8. #@param {type:"number"}

# Threshold to consider periodicity in entire video.
THRESHOLD = 0.2#@param {type:"number"}

# Threshold to consider periodicity for individual frames in video.
WITHIN_PERIOD_THRESHOLD = 0.5#@param {type:"number"}

# Use this setting for better results when it is 
# known action is repeating at constant speed.
CONSTANT_SPEED = False#@param {type:"boolean"}

# Use median filtering in time to ignore noisy frames.
MEDIAN_FILTER = True#@param {type:"boolean"}

# Use this setting for better results when it is 
# known the entire video is periodic/reapeating and
# has no aperiodic frames.
FULLY_PERIODIC = False#@param {type:"boolean"}

# Plot score in visualization video.
PLOT_SCORE = False#@param {type:"boolean"}

# Visualization video's FPS.
VIZ_FPS = 30#@param {type:"integer"}


def main():


     PATH_TO_CKPT = '/tmp/repnet_ckpt/'
     !mkdir $PATH_TO_CKPT
     !wget -nc -P $PATH_TO_CKPT https://storage.googleapis.com/repnet_ckpt/checkpoint
     !wget -nc -P $PATH_TO_CKPT https://storage.googleapis.com/repnet_ckpt/ckpt-88.data-00000-of-00002
     !wget -nc -P $PATH_TO_CKPT https://storage.googleapis.com/repnet_ckpt/ckpt-88.data-00001-of-00002
     !wget -nc -P $PATH_TO_CKPT https://storage.googleapis.com/repnet_ckpt/ckpt-88.index

     model = get_repnet_model(PATH_TO_CKPT)

     # from youtube
     VIDEO_URL = 'https://youtu.be/mP9TutWoUh0'

     # Cheetah running.
     # VIDEO_URL = 'https://www.reddit.com/r/gifs/comments/4qfif6/cheetah_running_at_63_mph_102_kph/'

     # Exercise repetition counting.
     # VIDEO_URL = 'https://www.youtube.com/watch?v=5g1T-ff07kM'

     # Kitchen activities repetition counting. Tough example with many starts and
     # stops and varying speeds of action.
     # VIDEO_URL = 'https://www.youtube.com/watch?v=5EYY2J3nb5c'

     download_video_from_url(VIDEO_URL)
     imgs, vid_fps = read_video("/tmp/video.mp4")  # figure out how they get frames


     print('Running RepNet...')
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

     print('Visualizing results...') 
     viz_reps(imgs, per_frame_counts, pred_score, interval=1000/VIZ_FPS,
              plot_score=PLOT_SCORE)


     # Debugging video showing scores, per-frame frequency prediction and 
     # within_period scores.
     create_count_video(imgs,
                        per_frame_counts,
                        within_period,
                        score=pred_score,
                        fps=vid_fps,
                        output_file='/tmp/debug_video.mp4',
                        delay=1000/VIZ_FPS,
                        plot_count=True,
                        plot_within_period=True,
                        plot_score=True)
     show_video('/tmp/debug_video.mp4')




if __name__ == '__main__':
     main()
