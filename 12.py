from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import serial
import datetime

seri = serial.Serial(port='COM4', baudrate=115200)
seri_2 = serial.Serial(port='COM3', baudrate=115200)
''', parity=serial.PARITY_NONE,
                     stopbits=serial.STOPBITS_ONE,
                     bytesize=serial.EIGHTBITS)'''

fig = plt.figure()
ax = plt.subplot(121, xlim=(0, 50), ylim=(0, 200))
ax_2 = plt.subplot(122, xlim=(0, 50), ylim=(0, 200))

max_points = 50
line, = ax.plot(np.arange(max_points),
                np.ones(max_points, dtype=float)*np.nan,
                lw=1, ms=1)
line_2, = ax_2.plot(np.arange(max_points),
                    np.ones(max_points, dtype=float)*np.nan,
                    lw=1, ms=1)


def init():
    return line

def init_2():
    return line_2


def decode(A):
    A = A.decode()
    A = str(A)
    if A[0]=='d':
        ard=int(A[1:4])
        return ard


def getDataFunc():
    if seri.readable():
        ret = seri.readline()
        code = decode(ret)
        return code
    return -1


def getDataFunc_2():
    if seri_2.readable():
        ret = seri_2.readline()
        code = decode(ret)
        return code
    return -1


def rv():
    y = float(getDataFunc())
    old_y = line.get_ydata()
    new_y = np.r_[old_y[1:], y]
    line.set_ydata(new_y)
    rv_L = (new_y[49] - old_y[49]) * 3.6

    y_2 = float(getDataFunc_2())
    old_y_2 = line_2.get_ydata()
    new_y_2 = np.r_[old_y_2[1:], y_2]
    line_2.set_ydata(new_y_2)
    rv_R = (new_y_2[49] - old_y_2[49]) * 3.6

    now = datetime.datetime.now()

    if(-50<rv_L<-5):
        print(now.strftime("%H:%M:%S"), ":LLL accela more", -rv_L, "km/h")
        return 1, rv_L
    elif(-50<rv_R<-5):
        print(now.strftime("%H:%M:%S"), ":RRR accela more", -rv_R, "km/h")
        return 1, rv_R


def animate(i):
    y = float(getDataFunc())
    old_y = line.get_ydata()
    new_y = np.r_[old_y[1:], y]
    line.set_ydata(new_y)
    rv_L = (new_y[49] - old_y[49]) * 3.6

    # if(abs(rv_L)<50):
    #    print(rv_L)
    return line


def animate_2(i):
    y_2 = float(getDataFunc_2())
    old_y_2 = line_2.get_ydata()
    new_y_2 = np.r_[old_y_2[1:], y_2]
    line_2.set_ydata(new_y_2)
    rv_R = (new_y_2[49] - old_y_2[49]) * 3.6

    # if (abs(rv_R) < 50):
    #    print(rv_R)
    return line_2

'''
anim = animation.FuncAnimation(fig, animate  , init_func= init, frames=300, interval=20, blit=False)
anim_2 = animation.FuncAnimation(fig, animate_2  , init_func= init_2, frames=300, interval=20, blit=False)

plt.show()
'''


while True:
    rv()




