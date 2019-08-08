Code for navigating a robot arm to specific locations in the world.

Currently, to do point tracking, you'll need to install the following
dependencies:

- Scikit learn (currently used only for a useful weighted linear
  regression function): **pip install scikit-learn --user**
- Scikit image (used to draw lines into a numpy array): **pip install
  scikit-image --user**
- [ros_numpy](https://github.com/eric-wieser/ros_numpy): Used for
  efficient transformation of a point cloud into a numpy array (it's
  suuuper slow without it)
  