# Demo 4h. Preparation of 2D power topography

""" (1) Class for 2D interpolation """
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
class bi_interp2:
  def __init__(self, x, y, z, xb, yb, xi, yi, method='linear', m_directory=m_directory):
    self.x = x
    self.y = y
    self.z = z
    self.xb = xb
    self.yb = yb
    self.xi = xi
    self.yi = yi
    self.x_new, self.y_new = np.meshgrid(xi, yi)
    self.id_out = np.zeros([len(self.xi), len(self.xi)], dtype='bool')
    self.x_up, self.y_up, self.x_dn, self.y_dn = [], [], [], []
    self.interp_method = method
    self.z_new = []

  def __call__(self):
    self.find_boundary()
    self.interp2d()
    return self.x_new, self.y_new, self.z_new

  def find_boundary(self):
    self.divide_plane()
    # sort x value
    idup = self.sort_arr(self.x_up)
    iddn = self.sort_arr(self.x_dn)
    self.x_up = self.x_up[idup]
    self.y_up = self.y_up[idup]
    self.x_dn = self.x_dn[iddn]
    self.y_dn = self.y_dn[iddn]
    self.remove_overlap()
    # find outline, use monotone cubic interpolation
    ybnew_up = self.interp1d(self.x_up, self.y_up, self.xi)
    ybnew_dn = self.interp1d(self.x_dn, self.y_dn, self.xi)
    for i in range(len(self.xi)):
        idt1 = self.y_new[:, i] > ybnew_up[i]
        idt2 = self.y_new[:, i] < ybnew_dn[i]
        self.id_out[idt1, i] = True
        self.id_out[idt2, i] = True
    # expand data points
    self.x = np.concatenate((self.x, self.x_new[self.id_out].flatten(), self.xb))
    self.y = np.concatenate((self.y, self.y_new[self.id_out].flatten(), self.yb))
    self.z = np.concatenate((self.z, np.zeros(np.sum(self.id_out) + len(self.xb))))

  def interp2d(self):
    pts = np.concatenate((self.x.reshape([-1, 1]), self.y.reshape([-1, 1])), axis=1)
    self.z_new = interpolate.griddata(pts, self.z, (self.x_new, self.y_new), method=self.interp_method)
    self.z_new[self.id_out] = np.nan
    
  def remove_overlap(self):
    id1 = self.find_val(np.diff(self.x_up) == 0, None)
    id2 = self.find_val(np.diff(self.x_dn) == 0, None)
    for i in id1:
      temp = (self.y_up[i] + self.y_up[i+1]) / 2
      self.y_up[i+1] = temp
      self.x_up = np.delete(self.x_up, i)
      self.y_up = np.delete(self.y_up, i)
    for i in id2:
      temp = (self.y_dn[i] + self.y_dn[i + 1]) / 2
      self.y_dn[i+1] = temp
      self.x_dn = np.delete(self.x_dn, i)
      self.y_dn = np.delete(self.y_dn, i)

  def divide_plane(self):
    ix1 = self.find_val(self.xb == min(self.xb), 1)
    ix2 = self.find_val(self.xb == max(self.xb), 1)
    iy1 = self.find_val(self.yb == min(self.yb), 1)
    iy2 = self.find_val(self.yb == max(self.yb), 1)
    # divide the plane with Quadrant
    qd = np.zeros([self.xb.shape[0], 4], dtype='bool')
    qd[:, 0] = (self.xb > self.xb[iy2]) & (self.yb > self.yb[ix2])
    qd[:, 1] = (self.xb > self.xb[iy1]) & (self.yb < self.yb[ix2])
    qd[:, 2] = (self.xb < self.xb[iy1]) & (self.yb < self.yb[ix1])
    qd[:, 3] = (self.xb < self.yb[iy2]) & (self.yb > self.yb[ix1])
    # divide the array with y axis
    self.x_up = self.xb[qd[:, 0] | qd[:, 3]]
    self.y_up = self.yb[qd[:, 0] | qd[:, 3]]
    self.x_dn = self.xb[qd[:, 1] | qd[:, 2]]
    self.y_dn = self.yb[qd[:, 1] | qd[:, 2]]

  def find_val(self, condition, num_of_returns):
    # find the value that satisfy the condition
    ind = np.where(condition == 1)
    return ind[:num_of_returns]

  def sort_arr(self, arr):
    # return sorting index
    return sorted(range(len(arr)), key=lambda i: arr[i])

  def interp1d(self, xx, yy, xxi):
    # find the boundary line
    interp_obj = interpolate.PchipInterpolator(xx, yy)
    return interp_obj(xxi)

""" (2) Function for Topography plot """
from pandas import read_csv
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
def plot_topo2d(data, clim=(-15,25), montage_file = m_directory+'montage.csv', plot_opt = True):

  # Zero-padding
  short = 38-len(data)
  if short: data=np.concatenate((data, np.tile(.00000001, short)), axis=0)

  # Boundary of mouse skull
  boundary = np.array([
      -4.400, 0.030, -4.180, 0.609, -3.960, 1.148, -3.740, 1.646, -3.520, 2.105, -3.300, 2.525, -3.080, 2.908, -2.860, 3.255,
      -2.640, 3.566, -2.420, 3.843, -2.200, 4.086, -1.980, 4.298, -1.760, 4.4799, -1.540, 4.6321, -1.320, 4.7567, -1.100, 4.8553,
      -0.880, 4.9298, -0.660, 4.9822, -0.440, 5.0150, -0.220, 5.0312,0, 5.035, 0.220, 5.0312, 0.440, 5.0150, 0.660, 4.9822,
      0.880, 4.9298, 1.100, 4.8553, 1.320, 4.7567, 1.540, 4.6321,1.760, 4.4799, 1.980, 4.2986, 2.200, 4.0867, 2.420, 3.8430,
      2.640, 3.5662, 2.860, 3.2551, 3.080, 2.9087, 3.300, 2.5258,3.520, 2.1054, 3.740, 1.6466, 3.960, 1.1484, 4.180, 0.6099,
      4.400, 0.0302, 4.400, 0.0302, 4.467, -0.1597, 4.5268, -0.3497,4.5799, -0.5397, 4.6266, -0.7297, 4.6673, -0.9197, 4.7025, -1.1097,
      4.7326, -1.2997, 4.7579, -1.4897, 4.7789, -1.6797, 4.7960, -1.8697,4.8095, -2.0597, 4.8199, -2.2497, 4.8277, -2.4397, 4.8331, -2.6297,
      4.8366, -2.8197, 4.8387, -3.0097, 4.8396, -3.1997, 4.8399, -3.3897,4.8384, -3.5797, 4.8177, -3.7697, 4.7776, -3.9597, 4.7237, -4.1497,
      4.6620, -4.3397, 4.5958, -4.5297, 4.5021, -4.7197, 4.400, -4.8937,4.1800, -5.1191, 3.9600, -5.3285, 3.7400, -5.5223, 3.5200, -5.7007,
      3.3000, -5.8642, 3.0800, -6.0131, 2.8600, -6.1478, 2.6400, -6.2688,2.4200, -6.3764, 2.2000, -6.4712, 1.9800, -6.5536, 1.7600, -6.6241,
      1.5400, -6.6833, 1.3200, -6.7317, 1.1000, -6.7701, 0.8800, -6.7991,0.6600, -6.8194, 0.4400, -6.8322, 0.2200, -6.8385, 0, -6.840,
      -0.220, -6.8385, -0.440, -6.8322, -0.660, -6.8194, -0.880, -6.7991,-1.100, -6.7701, -1.320, -6.7317, -1.540, -6.6833, -1.760, -6.6241,
      -1.980, -6.5536, -2.200, -6.4712, -2.420, -6.3764, -2.640, -6.2688,-2.860, -6.1478, -3.080, -6.0131, -3.300, -5.8642, -3.520, -5.7007,
      -3.740, -5.5223, -3.960, -5.3285, -4.180, -5.1191, -4.400, -4.89370,-4.5021, -4.7197, -4.5958, -4.5297, -4.6620, -4.3397, -4.7237, -4.1497,
      -4.7776, -3.9597, -4.8177, -3.7697, -4.8384, -3.5797, -4.8399, -3.3897,-4.8397, -3.1997, -4.8387, -3.0097, -4.8367, -2.8197, -4.8331, -2.6297,
      -4.8277, -2.4397, -4.8200, -2.2497, -4.8095, -2.0597, -4.7960, -1.8697,-4.7789, -1.6797, -4.7579, -1.4897, -4.7326, -1.2997, -4.7025, -1.1097,
      -4.6673, -0.9197, -4.6266, -0.7297, -4.5799, -0.5397, -4.5268, -0.3497,-4.4670, -0.1597, -4.4000, 0.03025]).reshape(-1, 2)

  montage_table = read_csv(montage_file)
  x, y = np.array(montage_table['X_ML']), np.array(montage_table['Y_AP'])
  xb, yb = boundary[:, 0], boundary[:, 1]
  xi, yi = np.linspace(min(xb), max(xb), 500),np.linspace(min(yb), max(yb), 500)
  xx, yy, topo_data = bi_interp2(x, y, data, xb, yb, xi, yi)()

  if plot_opt:
    topo_to_draw = topo_data.copy()
    topo_to_draw[np.where(topo_data>clim[1])] = clim[1]
    topo_to_draw[np.where(topo_data<clim[0])] = clim[0]
    plt.contourf(xx, yy, topo_to_draw, cmap=cm.jet, levels = np.linspace(clim[0],clim[1],50))
    plt.grid(False)
    plt.gca().set_aspect('equal','box')
    plt.xlabel('ML coordinate (mm)', fontsize=15);
    plt.ylabel('AP coordinate (mm)', fontsize=15);
    plt.text(0, 0.0, 'BP', color='w', fontsize=10, weight='bold', ha='center',va='center');
    plt.text(0,-4.2, 'LP', color='w', fontsize=10, weight='bold', ha='center',va='center');
    if clim is not None: plt.clim(clim)
    plt.plot(montage_table['X_ML'][0:36],montage_table['Y_AP'][0:36], 'w.')    
    plt.axis( (-5.5, 5.5, -7, 5.2) ) 
    plt.gca().set_facecolor((1,1,1))

  return xx, yy, topo_data

""" (3) Function for obtaining band-power from given time-series data """
def bandpower(x, targetBand, Fs=2000):
  # x : time series data
  # targetBand: 1 * 2 array_like
  # Fs (default:2000Hz) : sampling frequency
  nx, ldata = x.shape
  cutind = int(np.ceil(ldata / 2))
  freq = (np.linspace(0, (ldata - 1) / ldata * Fs, ldata))[:cutind]
  boolf = np.where( (freq > targetBand[0]) & (freq <= targetBand[1]))
  power = np.zeros(nx)
  for n in range(nx):
    temp = (np.fft.fft(x[n, :])) / ldata * 2
    temp = temp[:cutind]
    power[n] = np.mean(abs(temp[boolf]) ** 2)
  return power
