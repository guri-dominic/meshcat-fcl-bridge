import numpy as np
from spatialmath import SE3

from MFB import GeometryManager


def _test_initialization_simple_collisions():
    import time

    zmq_url = "tcp://127.0.0.1:6002"
    gm = GeometryManager(zmq_url=zmq_url)

    pose = SE3.Tz(1.1)
    radius = 0.5
    gm.add_sphere("ball", radius=radius, pose=pose)

    gm.add_box("block", size=(1, 1, 2 * radius), pose=SE3.Tz(0))
    # vis["block"].delete()

    for z in np.linspace(-1.5, 1.5, 16):
        correct = np.abs(z) - radius < radius
        T = SE3.Tz(z)
        gm.update("ball", T)
        collision = gm.in_collision("ball", "block")
        print(f"(z={z:.1f}) \tCollision detected: {collision}|{correct}")
        time.sleep(0.1)

    gm.clear()


def _test_cylinder():
    print("Cylinder Test")
    import time

    zmq_url = "tcp://127.0.0.1:6002"
    gm = GeometryManager(zmq_url=zmq_url)
    gm._vis.delete()

    radius, length = 0.05, 0.5
    pose = SE3.Tz(0.2)
    pose = SE3.Tz(radius + 0.5 * length + 0.01)
    key1 = "ball"
    key2 = "cylinder"
    gm.add_sphere(key1, radius=radius, pose=pose)

    gm.add_cylinder(key2, radius=radius, length=length, pose=SE3())
    # vis["block"].delete()

    collision = gm.in_collision(key2, key1)
    # correct = radius + 0.5 * length + 0.01
    correct = pose.t[2] < (radius + 0.5 * length)
    print(f"() \tCollision detected: {collision}|{correct}")

    for z in np.linspace(-0.7 * length, 0.7 * length, 16):
        T = SE3.Tz(z)
        correct = T.t[2] < (radius + 0.5 * length)
        gm.update(key1, T)
        collision = gm.in_collision(key2, key1)
        print(f"(z={z:.1f}) \tCollision detected: {collision}|{correct}")
        time.sleep(0.1)

    T = SE3.Tz(0.45 * length)
    gm.update(key1, T)
    for z in np.linspace(0, 2 * np.pi, 16):
        T = SE3.Ry(z)
        gm.update(key2, T)
        correct = None
        collision = gm.in_collision(key2, key1)
        print(f"(z={z:.1f}) \tCollision detected: {collision}|{correct}")
        time.sleep(0.1)

    # gm.clear()


def _test_capsules():
    print("Capsule Test")
    import time

    zmq_url = "tcp://127.0.0.1:6002"
    gm = GeometryManager(zmq_url=zmq_url)
    gm.clear()
    gm._vis.delete()

    radius, length = 0.05, 0.5
    pose = SE3.Tz(0.2)
    pose = SE3.Tz(radius + 0.5 * length + 0.01)
    key1 = "ball"
    key2 = "capsule"
    gm.add_sphere(key1, radius=radius, pose=pose)
    gm.add_capsule(key2, radius=radius, length=length, pose=SE3())

    collision = gm.in_collision(key2, key1)
    correct = pose.t[2] < (radius + 0.5 * length)
    print(f"() \tCollision detected: {collision}|{correct}")

    """
    for z in np.linspace(-0.8 * length, 0.8 * length, 16):
        T = SE3.Tz(z)
        correct = int(T.t[2] < (2 * radius + 0.5 * length))
        gm.update(key1, T)
        collision = int(gm.in_collision(key2, key1))
        print(f"(z={z:.1f}) \tCollision detected: {collision}|{correct}")
        time.sleep(0.1)
    """
    # return
    T = SE3.Tz(0.69 * length)
    gm.update(key1, T)
    for z in np.linspace(0, 2 * np.pi, 180 // 3 + 1):
        T = SE3.Ry(z)
        gm.update(key2, T)
        correct = None
        collision = gm.in_collision(key2, key1)
        print(f"(z={z:.1f}) \tCollision detected: {collision}|{correct}")
        time.sleep(0.05)
    # gm.clear()


def _test_kinematic_chains():
    print("Capsule Test")
    import time

    zmq_url = "tcp://127.0.0.1:6002"
    gm = GeometryManager(zmq_url=zmq_url)
    # gm.clear()
    gm._vis.delete()
    # return
    time.sleep(0.001)
    robot_key = "robot"
    radius = 0.025
    chain = np.array(
        [
            [0, 0, 0],  # ==
            [0, 0, 0.1],
            [-0.15, 0, 0.4],
            [0.35, 0, 0.6],
            [0.4, 0, 0.5],
        ]
    )
    gm.add_chain(key_prefix=robot_key, chain=chain, radius=radius)


# Example usage:
if __name__ == "__main__":
    # _test_initialization_simple_collisions()
    # _test_cylinder()
    # _test_capsules()
    _test_kinematic_chains()
