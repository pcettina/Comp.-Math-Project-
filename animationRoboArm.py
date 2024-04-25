import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 


fig = plt.figure()
plot = fig.add_subplot(111)

def coefCalc(a,b):
    if a<0:
        s1 = (b/2) + ((1/2)*np.sqrt(a**2 * ((4- a**2 - b**2)/(a**2 + b**2))))

    else:
        s1 = (b/2) - ((1/2)*np.sqrt(a**2 * ((4- a**2 - b**2)/(a**2 + b**2))))


    c1 = (-b/a)*s1 + ((a**2 + b**2 )/(2*a))

    s2 = b*c1 - a*s1
    c2 = a*c1 + b*s1 - 1
    
    return [s1,c1,s2,c2]

def posFunction(s1, c1, s2, c2,l1,l2):
    r1 = l2*c1*c2 - l2*s1*s2 +l1*c1
    r2 = l2*s1*c2 - l2*c1*s2 +l1*c1
    r3 = 1

    return np.vstack(np.array(r1,r2,r3))

def translation(s2,c2,l1,theta1):
    c2 += l1
    c2 = c2*np.cos(theta1) - s2*np.sin(theta1)
    s2 = c2*np.sin(theta1) + s2*np.cos(theta1)

    return [s2,c2]

def mapper(s,c,l):
    r1 = [c, -s, c*l]
    r2 = [s, c ,s*l]
    r3 = [0, 0 ,1]

    return np.array([r1,r2,r3])

def vectorCalc(a1,a2):
    r = np.dot(a1,a2)
    r = np.dot(r,np.vstack([0,0,1]))
    return r

def angle(theta, phi):
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
    pos1 = [1.5,0.7]

    s1,c1,s2,c2 = coefCalc(pos1[0] ,pos1[1])

    if(s1 <0):
        initTheta = np.arcsin(s1)

    else:
        initTheta = np.arccos(c1)
    initPhi = np.arccos(c2)
    

    pos2 = [-1,1.5]
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
    plot.set_xlim([-2,2])
    plot.set_ylim([-2,2])
    plot.plot([0, np.cos(theta[i])],[0, np.sin(theta[i])],'r')
    plot.plot([np.cos(theta[i]),r[0][0]],[np.sin(theta[i]),r[1][0]], 'b-')



def main():
    l1 = 1
    l2 = 1

    #check if s1 is negative and if so then take the arcsin of s1/ if pos take arccos of c1
    
    
    

    """
    pos1 = [1.5,0.7]

    s1,c1,s2,c2 = coefCalc(pos1[0] ,pos1[1])

    if(s1 <0):
        initTheta = np.arcsin(s1)

    else:
        initTheta = np.arccos(c1)
    initPhi = np.arccos(c2)
    

    pos2 = [1,1.5]
    s1,c1,s2,c2 = coefCalc(pos2[0] ,pos2[1])
    if(s1 <0):
        finalTheta = np.arcsin(s1)

    else:
        finalTheta = np.arccos(c1)
    finalPhi = np.arccos(c2)
    
    

    theta = np.linspace(initTheta,finalTheta,10)
    phi = np.linspace(initPhi, finalPhi,10)
    
    for i in range(10):
        arm1, arm2 =  angle(theta[i],phi[i])
        r = vectorCalc(arm1,arm2)
        plot.plot([0, np.cos(theta[i])],[0, np.sin(theta[i])],'r')
        plot.plot([np.cos(theta[i]),r[0][0]],[np.sin(theta[i]),r[1][0]], 'b-')
    
    """

    plot.set_xlim([-5,5])
    plot.set_ylim([-5,5])
    anim = FuncAnimation(fig, animate, interval=100, frames = 100)
    
    plt.show()


main()



"""
for thetas,phis in zip(theta,phi):
    arm1, arm2 =  angle(thetas,phis)
    r = vectorCalc(arm1,arm2)
    plot.plot([0, np.cos(thetas)],[0, np.sin(thetas)],'r')
    plot.plot([np.cos(thetas),r[0][0]],[np.sin(thetas),r[1][0]], 'b-')

#a1 = mapper(s1,c1,l1)
#a2 = mapper(s2,c2,l2)
#r = vectorCalc(a1,a2)

#plot.plot([0, np.cos(np.radians(30.651))],[0, np.sin(np.radians(30.651))],'r')
#plot.plot([np.cos(np.radians(30.651)),r[0][0]],[np.sin(np.radians(30.651)),r[1][0]], 'b-')

"""
"""
s1,c1,s2,c2 = coefCalc(pos2[0] ,pos2[1])
a1 = mapper(s1,c1,l1)
a2 = mapper(s2,c2,l2)
r = vectorCalc(a1,a2)

#plot.plot([0, c1],[0, s1],'r')
# plot.plot([c1,r[0][0]],[s1,r[1][0]], 'b-')


arm1, arm2 =  angle(finalTheta,finalPhi)
r = vectorCalc(arm1,arm2)
print(r)
finalTheta = np.arccos(c1)
finalPhi = np.arcsin(s2)
#plot.plot([0, np.cos(finalTheta)],[0, np.sin(finalTheta)],'k')
#plot.plot([np.cos(finalTheta),r[0][0]],[np.sin(finalTheta),r[1][0]], 'k-')
"""