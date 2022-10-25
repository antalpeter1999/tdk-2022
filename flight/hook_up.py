import pickle

import mujoco
import glfw
import os
import numpy as np
from ctrl.GeomControl import GeomControl
from ctrl.RobustGeomControl import RobustGeomControl
from ctrl.PlanarLQRControl import PlanarLQRControl
import time
from assets.util import sync
from scipy.spatial.transform import Rotation
from assets.logger import Logger
from matplotlib import pyplot as plt


def main():
    traj_mode = 'optimal_traj'  # One of 'feedforward', 'dummy_traj', or 'optimal_traj'
    # Reading model data
    print(f'Working directory:  {os.getcwd()}\n')
    model = mujoco.MjModel.from_xml_path("../hook_up_scenario/hook_scenario.xml")
    data = mujoco.MjData(model)

    # Initialize the library
    if not glfw.init():
        return

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(1920, 1080, "Crazyflie in MuJoCo", None, None)
    if not window:
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)

    # initialize visualization data structures
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation = 60, -30
    cam.lookat, cam.distance = [0, -2, 2], 1

    pert = mujoco.MjvPerturb()
    opt = mujoco.MjvOption()
    scn = mujoco.MjvScene(model, maxgeom=100)
    con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

    if traj_mode == 'dummy_traj' or traj_mode == 'optimal_traj':
        ## To obtain inertia matrix
        mujoco.mj_step(model, data)
        ### Controller
        controller = RobustGeomControl(model, data, drone_type='large_quad')
        controller.delta_r = 0
        mass = controller.mass
        controller_lqr = PlanarLQRControl(model)

    L = 0.4
    simtime = 0.0
    simulation_step = 0.001
    control_step = 0.01
    graphics_step = 0.02

    if traj_mode == 'dummy_traj':
        with open('../assets/splines/dummy_traj.pickle', 'rb') as file:
            traj = pickle.load(file)
        episode_length = 10
    if traj_mode == 'feedforward':
        with open('../hook_up_scenario/pickle/inputs.pickle', 'rb') as file:
            datas = pickle.load(file)
            # t = sp.symbols('t')
            F = np.squeeze(datas[0])
            tau = datas[1]
            r0 = datas[2]
            v0 = datas[3]
            R0 = datas[4]
            om0 = datas[5]
            alpha0 = datas[6]
            dalpha0 = datas[7]
        episode_length = 4

        # Initial conditions
        q0 = np.roll(Rotation.from_matrix(R0).as_quat(), 1)
        data.qpos[0:8] = np.hstack((np.squeeze(r0), q0, alpha0))
        data.qvel[0:7] = np.hstack((np.squeeze(v0), np.squeeze(om0), dalpha0))
    if traj_mode == 'optimal_traj':
        load_pos = np.array([1, 0, 0.76])
        with open('../hook_up_scenario/pickle/optimal_trajectory.pickle', 'rb') as file:
            ref = pickle.load(file)
            pos_ref = ref[0]
            vel_ref = ref[1]
            yaw_ref = ref[2]
            ctrl_type = ref[3]
            q0 = np.roll(Rotation.from_euler('xyz', [0, 0, yaw_ref[0]]).as_quat(), 1)
            data.qpos[0:8] = np.hstack((pos_ref[0, :] + load_pos, q0, 0))
            episode_length = (pos_ref.shape[0] - 1) * control_step + 2.5

    logger = Logger(episode_length, control_step)
    start = time.time()

    for i in range(int(episode_length / control_step)):
        # Get time and states
        simtime = data.time
        pos = data.qpos[0:3]
        quat = data.xquat[1, :]
        vel = data.qvel[0:3]
        ang_vel = data.sensordata[0:3]
        if traj_mode == 'dummy_traj':
            if simtime < 1:
                target_pos = np.array([traj_(0) for traj_ in traj])
                data.ctrl = controller.compute_pos_control(pos, quat, vel, ang_vel, target_pos)
            else:
                t = simtime - 1
                target_pos = np.array([traj_(t) for traj_ in traj])
                target_vel = np.array([traj_.derivative(1)(t) for traj_ in traj])
                data.ctrl = controller.compute_pos_control(pos, quat, vel, ang_vel, target_pos, target_vel=target_vel)
        if traj_mode == 'feedforward':
            data.ctrl = np.hstack((F[i], tau[i].T))
            target_pos = np.zeros(3)
        if traj_mode == 'optimal_traj':
            if simtime < 1:
                target_pos = pos_ref[0, :] + load_pos
                target_rpy = np.array([0, 0, yaw_ref[0]])
                data.ctrl = controller.compute_pos_control(pos, quat, vel, ang_vel, target_pos, target_rpy=target_rpy)
            else:
                # if i < 600:
                #     data.xfrc_applied[1, 0:3] = np.array([0, 0.5, 0.2])
                #     controller.delta_r = 1
                # else:
                #     data.xfrc_applied[1, 0:3] = np.array([0, 0, 0])
                #     controller.delta_r = 0
                # print(Rotation.from_quat(np.roll(data.qpos[3:7], -1)).as_euler('xyz')*180/np.pi)
                i_ = i - int(1/control_step)
                # alpha = data.qpos[7]
                # hook_pos = pos + L * np.array([np.sin(alpha), 0, -np.cos(alpha)])
                # cur_load_pos = data.qpos[8:11] + np.array([0, 0, 0.3])
                # if i_ == 123:
                #     controller.reset()
                #     controller.k_i = 100
                if i_ < pos_ref.shape[0]:
                    # Add the load mass to the feedforward term of geometric ctrl
                    # if np.linalg.norm(hook_pos - cur_load_pos) < 0.25:
                    #     # print('high mass')
                    #     controller.mass = mass + 0.05
                    # if np.linalg.norm(hook_pos - cur_load_pos) > 0.35:
                    #     # print('low mass')
                    #     controller.mass = mass
                    target_pos = pos_ref[i_, :] + load_pos
                    target_vel = vel_ref[i_, :]
                    target_rpy = np.array([0, 0, yaw_ref[i_]])
                    if ctrl_type[i_] == 'lqr':
                        data.ctrl = controller.compute_pos_control(pos, quat, vel, ang_vel, target_pos,
                                                                   target_vel=target_vel, target_rpy=target_rpy)
                        alpha = data.qpos[7]
                        dalpha = data.qvel[6]
                        pos_ = pos.copy()
                        vel_ = vel.copy()
                        R_plane = np.array([[np.cos(yaw_ref[i_]), -np.sin(yaw_ref[i_])],
                                            [np.sin(yaw_ref[i_]), np.cos(yaw_ref[i_])]])
                        pos_[0:2] = R_plane.T @ pos_[0:2]
                        vel_[0:2] = R_plane.T @ vel_[0:2]
                        hook_pos = pos_ + L * np.array([-np.sin(alpha), 0, -np.cos(alpha)])
                        hook_vel = vel_ + L * dalpha * np.array([-np.cos(alpha), 0, np.sin(alpha)])
                        hook_pos = np.take(hook_pos, [0, 2])
                        hook_vel = np.take(hook_vel, [0, 2])
                        phi_Q = Rotation.from_quat(np.roll(quat, -1)).as_euler('xyz')[1]
                        dphi_Q = ang_vel[1]
                        target_pos_ = target_pos.copy()
                        target_pos_[0:2] = R_plane.T @ target_pos_[0:2]
                        target_pos_load = np.take(target_pos_, [0, 2]) - np.array([0, L])
                        lqr_ctrl = controller_lqr.compute_control(hook_pos,
                                                                    hook_vel,
                                                                    alpha,
                                                                    dalpha,
                                                                    phi_Q,
                                                                    dphi_Q,
                                                                    target_pos_load)
                        data.ctrl[0] = lqr_ctrl[0]
                        data.ctrl[2] = lqr_ctrl[2]
                    else:
                        data.ctrl = controller.compute_pos_control(pos, quat, vel, ang_vel, target_pos,
                                                                   target_vel=target_vel, target_rpy=target_rpy)
                else:
                    target_pos = pos_ref[-1, :] + load_pos
                    target_vel = np.zeros(3)
                    target_rpy = np.array([0, 0, yaw_ref[-1]])
                    data.ctrl = controller.compute_pos_control(pos, quat, vel, ang_vel, target_pos,
                                                               target_vel=target_vel, target_rpy=target_rpy)
        for _ in range(int(control_step / simulation_step)):
            mujoco.mj_step(model, data, 1)
        state = np.hstack([target_pos - pos, Rotation.from_quat(quat).as_euler('xyz'), np.zeros(7)])
        logger.log(timestamp=simtime, state=state, control=data.ctrl)

        if i % (graphics_step / control_step) == 0:
            viewport = mujoco.MjrRect(0, 0, 0, 0)
            viewport.width, viewport.height = glfw.get_framebuffer_size(window)
            mujoco.mjv_updateScene(model, data, opt, pert=None, cam=cam, catmask=mujoco.mjtCatBit.mjCAT_ALL, scn=scn)
            mujoco.mjr_render(viewport, scn, con)

            glfw.swap_buffers(window)
            glfw.poll_events()

            # rgb = np.zeros((viewport.height, viewport.width, 3), dtype=np.uint8)
            # depth = np.zeros((viewport.height, viewport.width, 1))
            # mujoco.mjr_readPixels(rgb, depth, viewport=viewport, con=con)
            # rgb = np.flipud(rgb)
            # plt.imsave('../hook_up_scenario/videos/temp/vid_'+str(i)+'.png', rgb)
            # sync with wall-clock time
            sync(i, start, control_step)

            if glfw.window_should_close(window):
                break
    # print('Load distance from target: x: ' + "{:.2f}".format(2.5 - cur_load_pos[0]) +
    #       ' m; y: ' + "{:.2f}".format(1.5 - cur_load_pos[1]) + ' m')
    glfw.terminate()
    logger.plot()

if __name__ == '__main__':
    main()
