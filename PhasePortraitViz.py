#Importing necessary libraries
import numpy as np
import sympy as sp
import pandas as pd
import sympy
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#Create the main tkinter window
root = tk.Tk()
root.title("Phase Portrait Vizualizer")
root.geometry("1600x1000")
#Create label and entry fields for matrix input and variables x,y, and z
label_l = tk.Label(root, text="Enter matrix below as comma seperated list")
label_l.place(x=215, y=50)

entry_xnl = tk.Entry(root)
entry_xnl.place(x=1150, y=12)
label_xnl = tk.Label(root, text="dx:")
label_xnl.place(x=1125, y=14)
entry_ynl = tk.Entry(root)
entry_ynl.place(x=1150, y=42)
label_ynl = tk.Label(root, text="dy:")
label_ynl.place(x=1125, y=44)
entry_znl = tk.Entry(root)
entry_znl.place(x=1150, y=72)
label_znl = tk.Label(root, text="dz:")
label_znl.place(x=1125, y=74)

entry_l = tk.Entry(root)
entry_l.place(x=250, y=72)

#Create a figure and add 2 subplots 
fig = Figure(figsize=(14, 7), dpi=100)
#Embed the figure in a Tkinter frame

ax = fig.add_subplot(121)
ax3d = fig.add_subplot(122, projection='3d')

orbit_label = tk.Label(root, text="Enter number of desired orbits")
orbit_entry = tk.Entry()


def recieve_process_l():
    #Initialize input_recieved flag
    input_recieved = False
    while(not input_recieved):
        #Get the matrix input from the entry field
        raw_input = entry_l.get()
        #If the number of orbits is provided in the orbit_entry field, convert to an integer
        if orbit_entry.get():
            num_orbits = int(orbit_entry.get())
        else :
            num_orbits = 5
        #Convert input to lowercase and check for quit command
        raw_input.lower()
        if raw_input == 'quit':
            return root.destroy()
        #Remove brackets from input and split into a list of strings
        raw_input = raw_input.translate({ord(i): '' for i in '[]'})
        str_matrix = raw_input.split(',')
        #Remove whitespace from each string and convert to integers
        matrix = []
        for i, s in enumerate(str_matrix):
            str_matrix[i] = s.replace(" ", "")
            matrix.append(int(str_matrix[i]))
        matrix = np.asarray(matrix)
        #Check if matrix is 2x2 or 3x3 and set input_recieved flag accordingly
        check = False
        if (len(matrix) == 4):
            matrix = matrix.reshape(2, 2)
            check = True
        elif (len(matrix) == 9):
            matrix = matrix.reshape(3, 3)
            check = True
        input_recieved = check
    #Plot the phase portrait based on the dimensions of the matrix
    if (matrix.shape == (2,2)):
        plot_phase_portrait2D(matrix, num_orbits)
    elif(matrix.shape == (3,3)):
        plot_phase_portrait3D(matrix, num_orbits)
def recieve_process_nl():
    #Get the matrix input from the entry field
    raw_input_x = entry_xnl.get()
    raw_input_y = entry_ynl.get()
    #If the number of orbits is provided in the orbit_entry field, convert to an integer
    if orbit_entry.get():
            num_orbits = int(orbit_entry.get())
    else :
        num_orbits = 5
    #Convert input to lowercase and check for quit command
    raw_input_x.lower()
    raw_input_y.lower()
    #Use sympy library to get appropriate symbols from equations
    #interprate equations into sympy equation
    x, y, z = sympy.symbols('x y z')
    raw_dx = sympy.sympify(raw_input_x)
    raw_dy = sympy.sympify(raw_input_y)
    dx = lambda x, t: raw_dx.subs({x[0]: x, t: t})
    dy = lambda x, t: raw_dy.subs({x[0]: x, t: t})
    #If dz entry conatins an equation and if so repeat equation interpretation 
    if entry_znl.get():
        raw_input_z = entry_xnl.get()
        raw_input_z.lower()
        raw_dz = sympy.sympify(raw_input_z)
        dz = lambda x, t: raw_dy.subs({x[0]: x, t: t})
    #Plot the phase portrait based on the dimensions of the matrix
        plot_nl_phase_portrait3D(dx, dy, dz)
    else:
        plot_nl_phase_portrait2D(dx, dy)

def plot_nl_phase_portrait2D(dx, dy, x_range = 1, y_range = 1):
    #Clear the current plot
    ax.clear()
    #Set up the system of equations
    eqs = dx, dy
    #Generate evenly spaces points within range, (ticks on graph)
    x = np.linspace(-x_range, x_range, 20)
    y = np.linspace(-y_range, y_range, 20)
    X, Y = np.meshgrid(x, y)
    #Plotting the arrows - setting up variables
    t = 0
    u, v = np.zeros_like(X), np.zeros_like(Y)
    NI, NJ = X.shape
    #Compute directional vectors
    for i in range(NI):
        for j in range(NJ):
            x_val = X[i, j]
            y_val = Y[i, j]
            #Evaluate the derivatives of current point
            prime = [eq([x_val, y_val], t) for eq in eqs]
            #Extract x and y components of derivatives
            u[i, j] = prime[0].subs({'x': x_val, 'y':y_val})
            v[i, j] = prime[1].subs({'x': x_val, 'y':y_val})
    #Plot the vector field
    ax.quiver(X, Y, u, v, color='red')
    #Set up the plot labels and limits and update canvas
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim([-1*x_range, x_range])
    ax.set_ylim([-1*y_range, y_range])
    canvas.draw()
def plot_nl_phase_portrait3D(dx, dy, dz, x_range = 1, y_range = 1, z_range = 1):
    #Clear the current plot
    ax3d.clear()
    #Set up the system of equations
    eqs = dx, dy, dz
    #Generate evenly spaces points within range, (ticks on graph)
    x = np.linspace(-x_range, x_range, 5)
    y = np.linspace(-y_range, y_range, 5)
    z = np.linspace(-z_range, z_range, 5)
    X, Y, Z = np.meshgrid(x, y, z)
    #Plotting the arrows - setting up variables
    t = 0
    u, v, w = np.zeros_like(X), np.zeros_like(Y), np.zeros_like(Z)
    NI, NJ, NK = X.shape
    #Compute directional vectors
    for i in range(NI):
        for j in range(NJ):
            for k in range(NK):
                x_val = X[i, j, k]
                y_val = Y[i, j, k]
                z_val = Z[i, j, k]
                #Evaluate the derivatives of current point
                prime = [eq([x_val, y_val, z_val], t) for eq in eqs]
                #Extract x, y and z components of derivatives
                u[i, j] = prime[0].subs({'x': x_val, 'y':y_val, 'z':z_val})
                v[i, j] = prime[1].subs({'x': x_val, 'y':y_val, 'z':z_val})
                w[i, j] = prime[2].subs({'x': x_val, 'y':y_val, 'z':z_val})
    #Plot the vector field
    ax3d.quiver(X, Y, Z, u, v, w, length=0.1, normalize=False, color='red')
    #Set up the plot labels and limits and update canvas
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_xlim([-1*x_range, x_range])
    ax3d.set_ylim([-1*y_range, y_range])
    ax3d.set_zlim([-1*z_range, z_range])
    canvas.draw()

def plot_phase_portrait2D(matrix, num_orbits, x_range = 1, y_range = 1):
    #Clear the current plot
    ax.clear()
    #Generate evenly spaces points within range, (ticks on graph)
    x = np.linspace(-1*x_range, x_range, 20)
    y = np.linspace(-1*y_range, y_range, 20)
    X, Y = np.meshgrid(x, y)
    #Set up the system of equations
    def system(x, y):
        return matrix @ np.array([x, y])
    u, v = np.zeros_like(X), np.zeros_like(Y)
    #Compute directional vectors
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            dxu, dyv= system(X[i, j], Y[i, j])
            u[i, j] = dxu
            v[i, j] = dyv
    #Plot the vector field
    ax.quiver(X, Y, u, v, color='red')
    #Plotting the orbits/trajectories - find eigenvalues
    eigs, eig_vects = np.linalg.eig(matrix)
    eig1, eig2 = eigs[0], eigs[1]
    #Calculate eigenvectors
    eig_vect1, eig_vect2 = [eig_vects[0, 0], eig_vects[1, 0]], [eig_vects[0, 1], eig_vects[1, 1]]
    #Create x values for trajectories, final parameter determines number of segments in trajectory
    t = np.linspace(-4*x_range, 4*x_range, 50)
    #Calculate flow of each variable
    flow_x = (np.e ** (eig1 * t)) * eig_vect1[0] + (np.e ** (eig2 * t)) * eig_vect2[0]
    flow_y = (np.e ** (eig1 * t)) * eig_vect1[1] + (np.e ** (eig2 * t)) * eig_vect2[1]
    #Find all orbits
    alpha, beta = np.linspace(-1*x_range, x_range, num=num_orbits), np.linspace(-1*y_range, y_range, num=num_orbits)
    for i in range(num_orbits):
        ax.plot(alpha[i] * flow_x, beta[i] * flow_y, color='blue')
    #Set up the plot labels and limits and update canvas
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim([-1*x_range, x_range])
    ax.set_ylim([-1*y_range, y_range])
    canvas.draw()
def plot_phase_portrait3D(matrix, num_orbits):
    #Clear the current plot
    ax3d.clear()
    #Generate evenly spaces points within range, (ticks on graph)
    x_range = matrix.max()
    y_range = matrix.max()
    z_range = matrix.max()
    x = np.linspace(-1*x_range, x_range, 5)
    y = np.linspace(-1*y_range, y_range, 5)
    z = np.linspace(-1*z_range, z_range, 5)
    X, Y, Z = np.meshgrid(x, y, z)
    #Set up the system of equations
    def system(x, y, z):
        return matrix @ np.array([x, y, z])
    u, v, w = np.zeros_like(X), np.zeros_like(Y), np.zeros_like(Z)
    #Compute directional vectors
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            for k in range(Z.shape[2]):
                dxu, dyv, dzw = system(X[i, j, k], Y[i, j, k], Z[i, j, k])
                u[i, j, k] = dxu
                v[i, j, k] = dyv
                w[i, j, k] = dzw
    ax3d.quiver(X, Y, Z, u, v, w, length=0.1, normalize=False, color='red')
    #Plotting the orbits/trajectories - find eigenvalues
    t = np.linspace(-4*x_range, 4*x_range, 50)
    eigs, eig_vects = np.linalg.eig(matrix)
    eig1, eig2, eig3 = eigs[0], eigs[1], eigs[2]
    #Calculate eigenvectors
    eig_vect1, eig_vect2, eig_vect3 = [eig_vects[0][0], eig_vects[1][0], eig_vects[2][0]], [eig_vects[0][1], eig_vects[1][1], eig_vects[2][1]], [eig_vects[0][2], eig_vects[1][2], eig_vects[2][2]]
    #Calculate flow of each variable
    flow_x = (np.e ** (eig1 * t)) * eig_vect1[0] + (np.e ** (eig2 * t)) * eig_vect2[0] + (np.e ** (eig3 * t)) * eig_vect3[0]
    flow_y = (np.e ** (eig1 * t)) * eig_vect1[1] + (np.e ** (eig2 * t)) * eig_vect2[1] + (np.e ** (eig3 * t)) * eig_vect3[1]
    flow_z = (np.e ** (eig1 * t)) * eig_vect1[2] + (np.e ** (eig2 * t)) * eig_vect2[2] + (np.e ** (eig3 * t)) * eig_vect3[2]
    #Find all orbits
    alpha, beta, gamma = np.linspace(-1*x_range, x_range, num=num_orbits), np.linspace(-1*y_range, y_range, num=num_orbits), np.linspace(-1*z_range, z_range, num=num_orbits)
    for i in range(len(alpha)):
        ax3d.plot(alpha[i] * flow_x, beta[i] * flow_y, gamma[i] * flow_z, color='blue')
    #Set up the plot labels and limits and update canvas
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_xlim([-1*x_range, x_range])
    ax3d.set_ylim([-1*y_range, y_range])
    ax3d.set_zlim([-1*z_range, z_range])
    canvas.draw()

#Place tkinter (User Interface) elements
button_l = tk.Button(root, text="Submit", command=recieve_process_l)
button_l.place(x=300, y=100)

button_nl = tk.Button(root, text="Submit", command=recieve_process_nl)
button_nl.place(x=1208, y=100)

#Set up tkinter canvas
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().place(x=100, y=150)
orbit_label.place(x=100, y = 862)
orbit_entry.place(x=100, y = 884)

root.mainloop()
