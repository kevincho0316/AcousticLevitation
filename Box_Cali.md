# Box Calibration

Box calibration is the most critical and fragile part of this pipeline — it needs to be handled very carefully.

Initial RMS is very important. Get it as low as possible. Experimentally, bundle adjustment passes cannot recover from high initial RMS.

This is a step-by-step guide for getting usable data.


## Capture with a High-Resolution Camera

Input quality is critical. Use the highest resolution camera available.

We used 3 webcams to detect the levitating ball, each at 1080p. This resolution was not enough — it led to high initial RMS, which is bad.

In our case, we used a phone camera. This approach should work for most people.

### Capturing with a Phone

In our case, the camera calibration result looked like this:

camera_matrix:
- - 3088.9321918709943
  - 0.0
  - 2019.0811998544625
- - 0.0
  - 3109.0744926332295
  - 1501.0563994606648
- - 0.0
  - 0.0
  - 1.0
distortion_coefficients:
- 0.26729805613843555
- -1.3473021544979038
- 0.00028953053254239736
- 0.0018386094646656987
- 2.178494251727113

Reading these values, here is what to watch out for:

For those unfamiliar with graphics or 3D camera concepts:

**Camera matrix** — contains focal length and principal point data. This does not mean you need to fix the focus manually.

**Distortion coefficients** — describe lens distortion.

However, many phone cameras have auto-rotation enabled — if the phone tilts, the image rotates too. This is problematic. Use a third-party camera app that locks orientation, or be mindful of this.

Fix the camera position and move the box by hand when capturing. This worked significantly better.

Keep markers as visible and crisp as possible.
