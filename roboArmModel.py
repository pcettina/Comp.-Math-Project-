import numpy as np
import matplotlib.pyplot as plt

def coefCalc(a,b):
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


def main():
    l1 = 1
    l2 = 1

    
    pos1 = [1,1.5]
    pos2 = [-1 ,1.5]

    s1,c1,s2,c2 = coefCalc(pos1[0] ,pos1[1])
    a1 = mapper(s1,c1,l1)
    a2 = mapper(s2,c2,l2)
    r = vectorCalc(a1,a2)

    fig = plt.figure()
    plot = fig.add_subplot(111)

    plot.plot([0, c1],[0, s1],'r')
    plot.plot([c1,r[0][0]],[s1,r[1][0]], 'b-')
    plot.set_xlim([-1,1])
    plot.set_ylim([0,2])

    
    plt.show()


main()