import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from sympy import *
import scipy.optimize as opt

w1 = symbols('w1')
w2 = symbols('w2')
w3 = symbols('w3')
b = symbols('b')

f1 = 1*w1+2*w2+3*w3+b-1
f2 = 4*w1+1*w2+2*w3+b-1
f3 = (-1)*w1+2*w2+(-1)*w3+b+1

res = solve([f1,f2,f3],[w1,w2,w3,b])
print(res)
res = list(res.values())

####################################################################
# w1^2 + w2^2 + w3^2
def fun(b):
  return ((1 / 13 - 2 * b / 13) ** 2 + (-7 * b / 13 - 3 / 13) ** 2 + (b / 13 + 6 / 13) ** 2) / 2

# (w^2)/2 -> 0
mymin = opt.minimize(fun, 0, method='BFGS')
print(f'b:{mymin.x}')

b = mymin.x[0]
f1 = 1*w1+2*w2+3*w3+b-1
f2 = 4*w1+1*w2+2*w3+b-1
f3 = (-1)*w1+2*w2+(-1)*w3+b+1
res = solve([f1,f2,f3],[w1,w2,w3,b])
res = list(res.values())
for i in range(len(res)):
    res[i] = float(format(res[i], '.4f'))
print(f'w:{res}')


n = 100
x, y = np.meshgrid(np.linspace(-15, 15, n),
                   np.linspace(-8, 8, n))

#核心函数: w1*x + w2*y + w3*z + b = 0
z = -(float(res[0])*x + float(res[1])*y + b) / float(res[2])
z1 = -(float(res[0])*x + float(res[1])*y + b + 1) / float(res[2])
z2 = -(float(res[0])*x + float(res[1])*y + b - 1) / float(res[2])

# 绘制图片
fig = plt.figure()
ax3d = plt.axes(projection='3d')
# 设置坐标轴名称
ax3d.set_xlabel("X")
ax3d.set_ylabel("Y")
ax3d.set_zlabel("Z")
# 设置坐标标识大小
plt.tick_params(labelsize=8)

#绘图
ax3d.plot_surface(x, y, z)
ax3d.plot_surface(x, y, z1, color='lightyellow')
ax3d.plot_surface(x, y, z2, color='lightsalmon')
ax3d.plot3D(1, 2, 3, "rs", label="positive")
ax3d.plot3D(4, 1, 2, "rs", label="positive")
ax3d.plot3D(-1, 2, -1, "bo", label="negative")
ax3d.text(1.5, 2.5, 3.5, "(1, 2, 3)")
ax3d.text(13, 0.5, 1.5, "(4, 1, 2)")
ax3d.text(-1.5, 2.5, -1.8, "(-1, 2, -1)")
ax3d.text(15, 12, 11, f'b:{format(float(mymin.x), ".4f")}\nw:{res}',
          color = "purple", style = "normal", weight = "light", verticalalignment='center',rotation=90)

#改变视角,elev：沿着y轴旋转,azim：沿着z轴旋转
ax3d.view_init(elev=13, azim=125)
plt.legend()
plt.tight_layout()
plt.savefig('res.jpg',dpi=400)
plt.show()


