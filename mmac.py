import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy.integrate import odeint

from controllers.dummy_controller import DummyController
from controllers.feedback_linearization_controller import FeedbackLinearizationController
from controllers.mma_controller import MMAController
from trajectory_generators.constant_torque import ConstantTorque
from trajectory_generators.sinusonidal import Sinusoidal
from trajectory_generators.poly3 import Poly3
from utils.simulation import simulate

"""http://www.gipsa-lab.fr/~ioandore.landau/adaptivecontrol/Transparents/Courses/AdaptiveCourse5GRK.pdf"""

Tp = 0.01
end = 10.


# TODO: Switch to MMAC as soon as you implement it
controller = MMAController(Tp)
# controller = FeedbackLinearizationController(Tp)
# controller = DummyController(Tp)

"""
Here you have some trajectory generators. You can use them to check your implementations.
"""
# traj_gen = ConstantTorque(np.array([0., 1.0])[:, np.newaxis])
traj_gen = Sinusoidal(np.array([0., 1.]), np.array([2., 2.]), np.array([0., 0.]))
#traj_gen = Poly3(np.array([0., 0.]), np.array([pi/4, pi/6]), end)


Q, Q_d, u, T = simulate("PYBULLET", traj_gen, controller, Tp, end, multimodel=True)

plt.subplot(221)
plt.plot(T, Q[:, 0], 'r', label="q_1")
plt.plot(T, Q_d[:, 0], 'b', label="qd_1")
plt.legend()
plt.subplot(222)
plt.plot(T, Q[:, 1], 'r', label="q_2")
plt.plot(T, Q_d[:, 1], 'b', label="qd_2")
plt.legend()
plt.subplot(223)
plt.plot(T, u[:, 0], 'r', label="u_1")
plt.plot(T, u[:, 1], 'b', label="u_2")
plt.legend()
plt.show()