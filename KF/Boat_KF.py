import numpy as np
from matplotlib.patches import Polygon
from PIL import Image

class Boat:
    def __init__(self, ax):
        OG_posX = 250
        OG_posY = 250

        OG_Xv = 3
        OG_Yv = 3

        self.BoatState = np.array([OG_posX, OG_posY], dtype = float)
        self.BoatVState = np.array([OG_Xv, OG_Yv], dtype = float)
        self.Boat_VEng = np.array([OG_Xv, OG_Yv], dtype = float)

    def position(self, points):
        heading = np.arctan2(self.BoatVState[1], self.BoatVState[0])
        R = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])

        return (R @ points.T).T + np.array([self.BoatState[0], self.BoatState[1]])

    def plot_boat(self, ax):
        L = 25
        W = 10
        
        hull_outline = np.array([[-L/2, -W], [L/2, -W], [L/2, W], [-L/2, W]])
        bow_outline = np.array([[L/2, -W], [L/2 + W, 0], [L/2,  W]])

        hull = self.position(hull_outline)
        bow = self.position(bow_outline)

        hull_obj = Polygon(hull, closed = True, color = 'grey', alpha = 0.8)
        bow_obj = Polygon(bow, closed = True, color = 'grey', alpha = 0.8)

        ax.add_patch(hull_obj)
        ax.add_patch(bow_obj)

    def move(self, dt, current, pos, t):
        enforce = False
        if self.BoatState[0] > 650:
            self.Boat_VEng[0] += -5 * (self.BoatState[0] - 650)/100
            enforce = True
        elif self.BoatState[0] < 100:
            self.Boat_VEng[0] += 5 * (100 - self.BoatState[0])/100
            enforce = True
        if self.BoatState[1] > 650:
            self.Boat_VEng[1] += -5 * (self.BoatState[1] - 650)/100
            enforce = True
        elif self.BoatState[1] < 100:
            self.Boat_VEng[1] += 5 * (100 - self.BoatState[1])/100
            enforce = True
        
        if enforce == False:
            axisChanging = np.random.randint(0, 2)
            Delta = np.random.default_rng().uniform(low=-0.5, high=0.5)

            self.Boat_VEng[axisChanging] += Delta

        v_norm = np.linalg.norm(self.Boat_VEng)
        v_max = 12
        if v_norm > v_max and v_norm > 0:
            self.Boat_VEng *= v_max / v_norm   

        self.BoatVState = self.Boat_VEng + current

        self.BoatState += self.BoatVState * dt

        pos.append([self.BoatState[0], self.BoatState[1], t])



