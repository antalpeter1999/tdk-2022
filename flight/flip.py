import mujoco
import glfw
import os
import numpy as np
from ctrl.GeomControl import GeomControl
import time
from assets.splines.spline import BSpline, BSplineBasis
from assets.util import sync


def main():
    # Reading model data
    print(f'Working directory:  {os.getcwd()}\n')
    model = mujoco.MjModel.from_xml_path("../assets/cf2.xml")
    data = mujoco.MjData(model)

    # Initialize the library
    if not glfw.init():
        return

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(1280, 720, "Crazyflie in MuJoCo", None, None)
    if not window:
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)

    # initialize visualization data structures
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation = 170, -30
    cam.lookat, cam.distance = [0, 0, 0.3], 2

    pert = mujoco.MjvPerturb()
    opt = mujoco.MjvOption()
    scn = mujoco.MjvScene(model, maxgeom=30)
    con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

    ## To obtain inertia matrix
    mujoco.mj_step(model, data)
    ### Controller
    controller = GeomControl(model, data)

    ### Trajectory
    flip_spline_params = np.loadtxt("../assets/pos_spline.csv", delimiter=',')
    flip_traj = [BSpline(BSplineBasis(flip_spline_params[i, 1:18], int(flip_spline_params[i, 0])), flip_spline_params[i, 18:]) for i in range(3)]
    flip_vel = [flip_traj[i].derivative(1) for i in range(3)]
    flip_acc = [flip_traj[i].derivative(2) for i in range(3)]

    simtime = 0.0
    i = 0  # loop variable for syncing
    timestep = 0.005
    start = time.time()

    while simtime < 4:
        simtime = data.time

        if simtime < 2:
            target_pos = np.array([0, 0, 0.3])
            pos = data.qpos[0:3]
            quat = data.qpos[3:7]
            vel = data.qvel[0:3]
            ang_vel = data.qvel[3:6]
            data.ctrl = controller.compute_pos_control(pos, quat, vel, ang_vel, target_pos)
        elif simtime < 2.9:
            eval_time = (simtime - 2) / 0.9
            target_pos = np.array([flip_traj[0](eval_time)[0], 0, flip_traj[1](eval_time)[0]])
            target_pos[2] = target_pos[2] + 0.3
            target_vel = np.array([flip_vel[0](eval_time)[0], 0, flip_vel[1](eval_time)[0]])
            target_acc = np.array([flip_acc[0](eval_time)[0], 0, flip_acc[1](eval_time)[0]])
            q0 = flip_traj[2](eval_time)[0]
            q2 = np.sqrt(1 - q0**2)
            target_quat = np.array([q0, 0, q2, 0])
            dq0 = flip_vel[2](eval_time)[0]
            dq2 = - dq0 * q0 / q2
            target_quat_vel = np.array([dq0, 0, dq2, 0])

            pos = data.qpos[0:3]
            quat = data.qpos[3:7]
            vel = data.qvel[0:3]
            ang_vel = data.qvel[3:6]
            data.ctrl = controller.compute_att_control(pos, quat, vel, ang_vel, target_pos, target_vel, target_acc,
                                                       target_quat=target_quat, target_quat_vel=target_quat_vel)
        else:
            target_pos = np.array([0, 0, 0.3])
            pos = data.qpos[0:3]
            quat = data.qpos[3:7]
            vel = data.qvel[0:3]
            ang_vel = data.qvel[3:6]
            data.ctrl = controller.compute_pos_control(pos, quat, vel, ang_vel, target_pos)

        while data.time - simtime < timestep:
            mujoco.mj_step(model, data)
        viewport = mujoco.MjrRect(0, 0, 0, 0)
        viewport.width, viewport.height = glfw.get_framebuffer_size(window)
        mujoco.mjv_updateScene(model, data, opt, pert=None, cam=cam, catmask=mujoco.mjtCatBit.mjCAT_ALL, scn=scn)
        mujoco.mjr_render(viewport, scn, con)

        glfw.swap_buffers(window)
        glfw.poll_events()

        # sync with wall-clock time
        sync(i, start, timestep)
        i = i + 1

    glfw.terminate()


if __name__ == '__main__':
    main()
