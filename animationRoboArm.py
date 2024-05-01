"""
animationRoboArm.py
Authors: Ethan Senatore & Patrick Cettina
Date: 04/30/2024

This program is able to take in two postions and simulate the movement of a robotic arm from the initial to final point.
It contains various functions that are used to simulate the movement and ensure accurate calculations of the arm itself.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation 

#Initialize figure and plot
fig = plt.figure()
plot = fig.add_subplot(111)

def coefCalc(a,b):
    """
    coefCalc(a,b)

    Input: (a,b) ~ position

    Output: [s1,c2,s2,c2] ~ grobner basis coefficients 
    """

    if a<0:
        #To ensure the elbow does not "bend backwards" we flip the s1 coefficient from - to +
        s1 = (b/2) + ((1/2)*np.sqrt(a**2 * ((4- a**2 - b**2)/(a**2 + b**2))))
        c1 = (-b/a)*s1 + ((a**2 + b**2 )/(2*a))

    elif a == 0:
        #Special case if a = 0 that is not handled in the literature
        #We manually set the horizontal component of the arm to 0
        s1 = (b/2) + ((1/2)*np.sqrt(a**2 * ((4- a**2 - b**2)/(a**2 + b**2))))
        c1= 0

    else:
        s1 = (b/2) - ((1/2)*np.sqrt(a**2 * ((4- a**2 - b**2)/(a**2 + b**2))))
        c1 = (-b/a)*s1 + ((a**2 + b**2 )/(2*a))

    s2 = b*c1 - a*s1
    c2 = a*c1 + b*s1 - 1
    
    return [s1,c1,s2,c2]

def vectorCalc(a1,a2):
    """
    vectorCalc(a1,a2)

    Input: (a1,a2) ~ Arm 1 and Arm2 

    Ouput: r ~ position vector for hand of the robot
    """

    r = np.dot(a1,a2)
    r = np.dot(r,np.vstack([0,0,1]))
    return r

def angle(theta, phi):
    """
    angle(theta, phi)

    Input: (theta, phi) ~ theta is the angle that arm 1 makes with the x-axis
                          phi is the angle that arm 2 makes with the x-axis (parallel to Arm 1)

    Output: arm1, arm2 ~ arm1 and arm2 are the respective arm matrices to be used by vectorCalc
    """
    arm1_1 = [np.cos(theta), -np.sin(theta), np.cos(theta)]
    arm1_2 = [np.sin(theta), np.cos(theta), np.sin(theta)]
    arm1_3 = [0,0,1]

    arm1 = np.array([arm1_1, arm1_2, arm1_3])

    arm2_1 = [np.cos(phi), -np.sin(phi), np.cos(phi)]
    arm2_2 = [np.sin(phi), np.cos(phi), np.sin(phi)]
    arm2_3 = [0,0,1]

    arm2 = np.array([arm2_1, arm2_2, arm2_3])

    return arm1, arm2

def animate(i):
    pos1 = [.25,1]
    pos2 = [1,1]

    s1,c1,s2,c2 = coefCalc(pos1[0] ,pos1[1])
    if(s1 <0):
        initTheta = np.arcsin(s1)

    else:
        initTheta = np.arccos(c1)
    initPhi = np.arccos(c2)
     
    s1,c1,s2,c2 = coefCalc(pos2[0] ,pos2[1])
    if(s1 <0):
        finalTheta = np.arcsin(s1)

    else:
        finalTheta = np.arccos(c1)

    finalPhi = np.arccos(c2)
    
    
    theta = np.linspace(initTheta,finalTheta,100)
    phi = np.linspace(initPhi, finalPhi,100)

    arm1, arm2 =  angle(theta[i],phi[i])
    r = vectorCalc(arm1,arm2)
    plot.clear()
    plot.set_xlim([-2.5,2.5])
    plot.set_ylim([-2.5,2.5])
    plot.invert_yaxis()
    plot.plot(pos1[0],pos1[1],'ko') 
    plot.plot(pos2[0], pos2[1], 'ko')
    plot.plot([0, np.cos(theta[i])],[0, np.sin(theta[i])],'r')
    plot.plot([np.cos(theta[i]),r[0][0]],[np.sin(theta[i]),r[1][0]], 'b-')


def main():

    anim = FuncAnimation(fig, animate, interval=100, frames= 100)
    
    writer = animation.PillowWriter(fps=15,
                                 metadata=dict(artist='Me'),
                                 bitrate=1800)
    anim.save('handRaise.gif', writer=writer)


main()


 