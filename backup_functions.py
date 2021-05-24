

def get_webcam_video(img_b64):
  """Populates global variable imgs by converting image URI to Numpy array."""
  image = data_uri_to_img(img_b64)
  imgs.append(image)


def download_video_from_url(url_to_video,
                            path_to_video='/tmp/video.mp4'):
  if os.path.exists(path_to_video):
    os.remove(path_to_video)
  ydl_opts = {
      'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
      'outtmpl': str(path_to_video),
  }
  with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url_to_video])


def record_video(interval_in_ms, num_frames, quality=0.8):
  """Capture video from webcam."""
  # https://colab.research.google.com/notebooks/snippets/advanced_outputs.ipynb.

  # Give warning before recording.
  for i in range(0, 3):
    print('Opening webcam in %d seconds'%(3-i))
    time.sleep(1)
    output.clear('status_text')

  js = Javascript('''
    async function recordVideo(interval_in_ms, num_frames, quality) {
      const div = document.createElement('div');
      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      // show the video in the HTML element
      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      google.colab.output.setIframeHeight(document.documentElement.scrollHeight,
        true);

      for (let i = 0; i < num_frames; i++) {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        img = canvas.toDataURL('image/jpeg', quality);
        google.colab.kernel.invokeFunction(
        'notebook.get_webcam_video', [img], {});
        await new Promise(resolve => setTimeout(resolve, interval_in_ms));
      }
      stream.getVideoTracks()[0].stop();
      div.remove();
    }
    ''')
  display(js)
  eval_js('recordVideo({},{},{})'.format(interval_in_ms, num_frames, quality))


def data_uri_to_img(uri):
  """Convert base64image to Numpy array."""
  image = base64.b64decode(uri.split(',')[1], validate=True)
  # Binary string to PIL image.
  image = Image.open(io.BytesIO(image))
  image = image.resize((224, 224))
  # PIL to Numpy array.
  image = np.array(np.array(image, dtype=np.uint8), np.float32)
  return image





def viz_reps(frames,
             count,
             score,
             alpha=1.0,
             pichart=True,
             colormap=plt.cm.PuBu,
             num_frames=None,
             interval=30,
             plot_score=True):
  """Visualize repetitions."""
  if isinstance(count, list):
    counts = len(frames) * [count/len(frames)]
  else:
    counts = count
  sum_counts = np.cumsum(counts)
  tmp_path = '/tmp/output.mp4'
  fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5),
                         tight_layout=True,)

  h, w, _ = np.shape(frames[0])
  wedge_x = 95 / 112 * w
  wedge_y = 17 / 112 * h
  wedge_r = 15 / 112 * h
  txt_x = 95 / 112 * w
  txt_y = 19 / 112 * h
  otxt_size = 62 / 112 * h

  if plot_score:
    plt.title('Score:%.2f' % score, fontsize=20)
  im0 = ax.imshow(unnorm(frames[0]))

  if not num_frames:
    num_frames = len(frames)

  if pichart:
    wedge1 = matplotlib.patches.Wedge(
        center=(wedge_x, wedge_y),
        r=wedge_r,
        theta1=0,
        theta2=0,
        color=colormap(1.),
        alpha=alpha)
    wedge2 = matplotlib.patches.Wedge(
        center=(wedge_x, wedge_y),
        r=wedge_r,
        theta1=0,
        theta2=0,
        color=colormap(0.5),
        alpha=alpha)

    ax.add_patch(wedge1)
    ax.add_patch(wedge2)
    txt = ax.text(
        txt_x,
        txt_y,
        '0',
        size=35,
        ha='center',
        va='center',
        alpha=0.9,
        color='white',
    )

  else:
    txt = ax.text(
        txt_x,
        txt_y,
        '0',
        size=otxt_size,
        ha='center',
        va='center',
        alpha=0.8,
        color=colormap(0.4),
    )

  def update(i):
    """Update plot with next frame."""
    im0.set_data(unnorm(frames[i]))
    ctr = int(sum_counts[i])
    if pichart:
      if ctr%2 == 0:
        wedge1.set_color(colormap(1.0))
        wedge2.set_color(colormap(0.5))
      else:
        wedge1.set_color(colormap(0.5))
        wedge2.set_color(colormap(1.0))

      wedge1.set_theta1(-90)
      wedge1.set_theta2(-90 - 360 * (1 - sum_counts[i] % 1.0))
      wedge2.set_theta1(-90 - 360 * (1 - sum_counts[i] % 1.0))
      wedge2.set_theta2(-90)

    txt.set_text(int(sum_counts[i]))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

  anim = FuncAnimation(
      fig,
      update,
      frames=num_frames,
      interval=interval,
      blit=False)
  anim.save(tmp_path, dpi=80)
  plt.close()
  return show_video(tmp_path)

