import matplotlib.pyplot as plt
import numpy as np
from controllers.adrc_flc_controller import ADRFLController
from trajectory_generators.constant_torque import ConstantTorque
from trajectory_generators.sinusonidal import Sinusoidal
from trajectory_generators.poly3 import Poly3
from utils.simulation import simulate

Tp = 0.001
end = 5

# traj_gen = ConstantTorque(np.array([0., 1.0])[:, np.newaxis])
traj_gen = Sinusoidal(np.array([0., 1.]), np.array([2., 2.]), np.array([0., 0.]))
# traj_gen = Poly3(np.array([0., 0.]), np.array([pi/4, pi/6]), end)

b_est_1 = 12
b_est_2 = 12
kp_est_1 = 53
kp_est_2 = 53
kd_est_1 = 53
kd_est_2 = 53
p1 = 9
p2 = 18

q0, qdot0, _ = traj_gen.generate(0.)
q1_0 = np.array([q0[0], qdot0[0]])
q2_0 = np.array([q0[1], qdot0[1]])

Kp = np.diag([kp_est_1, kp_est_2])
Kd = np.diag([kd_est_1, kd_est_2])
p = np.array([p1, p2])

controller = ADRFLController(Tp, np.concatenate([q0, qdot0]), Kp, Kd, p)


Q, Q_d, u, T = simulate("PYBULLET", traj_gen, controller, Tp, end)

eso = np.array(controller.eso.states)

plt.subplot(221)
plt.plot(T, eso[:, 0])
plt.plot(T, Q[:, 0], 'r')
plt.subplot(222)
plt.plot(T, eso[:, 2])
plt.plot(T, Q[:, 2], 'r')
plt.subplot(223)
plt.plot(T, eso[:, 1])
plt.plot(T, Q[:, 1], 'r')
plt.subplot(224)
plt.plot(T, eso[:, 3])
plt.plot(T, Q[:, 3], 'r')
plt.show()

plt.subplot(221)
plt.plot(T, Q[:, 0], 'r')
plt.plot(T, Q_d[:, 0], 'b')
plt.subplot(222)
plt.plot(T, Q[:, 1], 'r')
plt.plot(T, Q_d[:, 1], 'b')
plt.subplot(223)
plt.plot(T, u[:, 0], 'r')
plt.plot(T, u[:, 1], 'b')
plt.show()