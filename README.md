# BodyPoseAngleEstimation

## Using OpenPose to estimate body angles.

## :cinema: Video:
* https://youtu.be/OoNDu9B3CjQ

## :grey_question: What is it?
* Uses CMU's OpenPose to detect key body angles. For me these were the arms, but I have included legs too.
* To detect angles, I use the law of cosines (see video)
## :zap: Features:
* Detects angles on images!
* Detects angles on videos!

## :package: Modules / Packages:
* numpy: https://numpy.org/devdocs/user/quickstart.html
* opencv: https://pypi.org/project/opencv-python/
* math: https://docs.python.org/3/library/math.html
* os: https://docs.python.org/3/library/os.html

## The original work:
* This idea is based on the work from:
https://github.com/CMU-Perceptual-Computing-Lab/openpose

###### :hammer: To do:
* Compare the pose angles of multiple images, and tell if it is the same pose!
