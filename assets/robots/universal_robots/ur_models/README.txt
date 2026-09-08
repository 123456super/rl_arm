Universal Robots offline URDF package contents:

- ur5.urdf: ROS2 Humble ur_description generated UR5 model.
- meshes/ur5/: visual and collision meshes required by the UR5 URDF file.

This URDF file was exported in a ROS2 environment and copied into this
project for PyBullet simulation. The URDF uses relative mesh paths, so it
can be loaded without ROS as long as the directory layout in this package is
kept.

Example PyBullet usage:

  p.loadURDF("assets/robots/universal_robots/ur_models/ur5.urdf", useFixedBase=True)
