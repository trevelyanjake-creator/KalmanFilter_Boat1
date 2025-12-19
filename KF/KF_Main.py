import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from Boat_KF import Boat
from Est_KF import Est

plt.ion()

real_pos = []
GPS_pos = []
Dyn_pos = []
est_pos = []

t = 0
N = 0
dt = 0.1

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_facecolor('blue')

boat = Boat(ax)
complete_state = np.array([boat.BoatState[0], boat.BoatState[1], 0, 0])
est = Est(complete_state, ax, dt)

ax.set_xlim(0, 750)
ax.set_ylim(0, 750)

OG_currX = np.random.default_rng().uniform(low = -1.5, high = 1.5)
OG_currY = np.random.default_rng().uniform(low = -1.5, high = 1.5)

current = np.array([OG_currX, OG_currY])

current_disp = np.linalg.norm(current)
if current[0] >= 0 and current[1] >= 0:
    current_dir = 'NE'
elif current[0] >= 0 and current[1] < 0:
    current_dir = 'SE'
elif current[0] < 0 and current[1] < 0:
    current_dir = 'SW'
elif current[0] < 0 and current[1] >= 0:
    current_dir = 'NW'

current_change = np.random.randint(low = 200, high = 400)

gps_on = True

def on_key(event):
    global gps_on
    if event.key == 'g':
        gps_on = not gps_on
        est.gps_lastError = np.array(real_pos[-1][0:2]) - np.array(GPS_pos[-1][0:2])

fig.canvas.mpl_connect('key_press_event', on_key)


while t < 80:
    ax.cla()
    ax.set_xlim(0, 750)
    ax.set_ylim(0, 750)
    boat.move(dt, current, real_pos, t)
    boat.plot_boat(ax)

    if N % 10 == 0 and gps_on:
        est.dynamic_predict()
        est.get_Vmeas(current, boat.Boat_VEng, Dyn_pos, t, dt)
        est.update_vel()
        est.getGPS(boat.BoatState, GPS_pos, t)
        est.update_est(est_pos, t)
    else:
        if N % 10 == 0 and not gps_on:
            GPS_pos.append([(real_pos[-1][0] - est.gps_lastError[0]), (real_pos[-1][1] - est.gps_lastError[1]), t])
        est.dyn_only_update(t, est_pos)
        est.get_Vmeas(current, boat.Boat_VEng, Dyn_pos, t, dt)
        est.update_vel()
    ax.plot(est.EstState[0], est.EstState[1], marker = 'o', markersize = 4, color = 'black')

    if N % current_change == 0:
        currX = np.random.default_rng().uniform(low = -1.5, high = 1.5)
        currY = np.random.default_rng().uniform(low = -1.5, high = 1.5)

        current[0] = currX
        current[1] = currY

        if current[0] >= 0 and current[1] >= 0:
            current_dir = 'NE'
        elif current[0] >= 0 and current[1] < 0:
            current_dir = 'SE'
        elif current[0] < 0 and current[1] < 0:
            current_dir = 'SW'
        elif current[0] < 0 and current[1] >= 0:
            current_dir = 'NW'

        current_disp = np.linalg.norm(current)

        current_change = np.random.randint(low = 200, high = 400)
    
    if gps_on:
        ax.set_title(f"Current = {current_disp:.2f}m/s {current_dir}, GPS is on", color='black', fontsize=10)
    else:
        ax.set_title(f"Current = {current_disp:.2f}m/s {current_dir}, GPS is off", color='black', fontsize=10)

    plt.draw()
    plt.pause(0.01)

    t += dt
    N += 1

plt.ioff()
plt.show()

real_pos_arr = np.array(real_pos)
est_pos_arr = np.array(est_pos)
GPS_pos_arr = np.array(GPS_pos)
Dyn_pos_arr = np.array(Dyn_pos)

t_arr = real_pos_arr[:, 2]
real_arr = real_pos_arr[:, 0:2]
est_arr  = est_pos_arr[:, 0:2]
GPS_arr  = GPS_pos_arr[:, 0:2]
t_GPS_arr = GPS_pos_arr[:, 2]
Dyn_arr  = Dyn_pos_arr[:, 0:2]

est_err = est_arr - real_arr
est_err_plot = np.linalg.norm(est_err, axis=1)
Dyn_err = Dyn_arr - real_arr
Dyn_err_plot = np.linalg.norm(Dyn_err, axis=1)

GPS_err = np.zeros((GPS_arr.shape[0], 2))
i = 0
while i < GPS_arr.shape[0]:
    GPS_err[i][0] = GPS_arr[i][0] - real_arr[10*i][0]
    GPS_err[i][1] = GPS_arr[i][1] - real_arr[10*i][1]
    i += 1
GPS_err_plot = np.linalg.norm(GPS_err, axis=1)

plt.plot(t_arr, est_err_plot, label="KF estimate")
plt.plot(t_GPS_arr, GPS_err_plot, label="GPS estimate")
plt.plot(t_arr, Dyn_err_plot, label="Velocity input estimate")

plt.xlabel("Time (ds)")
plt.ylabel("Position error magnitude")
plt.legend()
plt.grid(True)

plt.show()