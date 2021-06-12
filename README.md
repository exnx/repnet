# RepNet code

This code contains inference code for RepNet, currently supporting rate prediction, eval, frame rendering.


```
main_inference.py
```
Runs inference, saves results.

```
main_inference_nonzero.py
```
Runs inference, but removes the zero predictions, improving results.

```
render_frames_repnet.py
```
Will render frames with pred/labels.

Note: for code to frames to videos, see repo for ResNet3D.