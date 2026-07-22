Universal Robots offline URDF package contents:

- ur5.urdf: ROS2 Humble ur_description generated UR5 model.
- ur5e.urdf: ROS2 Humble ur_description generated UR5e model.
- meshes/: visual and collision meshes required by the URDF files.

These URDF files were exported in a ROS2 environment and copied into this
project for PyBullet simulation. The URDF files use relative mesh paths, so they
can be loaded without ROS as long as the directory layout in this package is
kept.

Example PyBullet usage:

  p.loadURDF("assets/robots/universal_robots/ur_models/ur5.urdf", useFixedBase=True)
  p.loadURDF("assets/robots/universal_robots/ur_models/ur5e.urdf", useFixedBase=True)
