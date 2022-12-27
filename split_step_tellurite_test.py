import numpy as np
import math
from plotly.offline import plot
import plotly.graph_objs as go
import matplotlib.pyplot as plt

#rp-photonics.com/solitons
#w=2pif
#10 длин Ld, шаг 1/10 Ld

SMILE = 10*np.sqrt(0.1)  # Коэф изменения значений оси Z при изменении Po в 10 раз
FIBER_LEN = 0.5 #0.5
CALC_STEP = 0.005 # при t0=100е-15 лучше ставить 0.005
h = 0.001 #step size при t0=100е-15 лучше ставить 0.001
TAU_AMPLITUDE = 10000 #4096

alpha = 1000
alph = alpha/(4343) # dB/km (1.2.4)
b2 = 1.478e-25 #second order disperion s^2/m
gamma = 23.53  # 1/W*m
C = 0 # chirp parameter

Po = 0.01125 #Watt
Ao = np.sqrt(Po)
to = 200e-15 # initial pulse width in sec
dt = 1e-15

tau0 = np.arange(-TAU_AMPLITUDE, TAU_AMPLITUDE, 1)
tau = np.arange(-TAU_AMPLITUDE*dt, TAU_AMPLITUDE*dt, dt) # dt=t/t0 (3.1.2)
#tau = np.arange(-3020e-15, 3020e-15, dt)

Ld = (to**2)/np.absolute(b2) #dispertion lenght 0,27
#Ln = 3,77

distance = np.arange(0.1, FIBER_LEN, CALC_STEP)

rel_error = 1e-8



n = np.arange(0,290,1)

op_pulse = [[0 for y in tau] for x in np.arange(0.1, FIBER_LEN, CALC_STEP)]
U = Ao*np.exp(-((1+1j*(-C))/2.0)*(tau/to)**2)
op_pulse_w = op_pulse[:]



_ = np.arange(0.1, FIBER_LEN, CALC_STEP)
for index, ii in enumerate(_): 
    u = U[:]
    l = np.max(u.shape)
    dw = 1.0/float(l)/dt*2.0*np.pi
    
    w = dw*np.arange(-1*l/2.0, l/2.0, 1)
    w = np.asarray(w)
    w = np.fft.fftshift(w)
    
    u = np.asarray(u)
    u = np.fft.fftshift(u)
    
    spectrum = np.fft.fft(np.fft.fftshift(u))

    for jj in np.arange(h, ii+0.00001, h):
        spectrum = spectrum*np.exp(-alph*(h/4.0)+1j*b2/2.0*(np.power(w, 2))*(h/2.0))
        f = np.fft.ifft(spectrum)
        f = f*np.exp(1j*gamma*np.power(np.absolute(f), 2)*(h))
        spectrum = np.fft.fft(f)
        spectrum = spectrum*np.exp(-alph*(h/4.0)+1j*b2/2.0*(np.power(w, 2)*(h/2.0)))
        
    f = np.fft.ifft(spectrum)
    spectrum = np.fft.fft(f)
    op_pulse_w[index] = np.absolute(spectrum)**2
    op_pulse[index] = np.absolute(f*SMILE**5)**2
    
    print("\r> Computing... ({} of {})".format(index, _.shape[0]), end="")
    index += 1
    
print("\r> Computing complete! ({} of {})".format(index, _.shape[0]))

print(len(op_pulse_w))

#------------------------------------------------------------
# FREQUENCY SPACE
wf = 2*math.pi*10**12

def plot_pulse_evo_w():
    plt.plot(w/wf,op_pulse_w[79], color='red')
    plt.plot(w/wf,op_pulse_w[39], color='blue')
    plt.axis([-10,10,0,2900])
    plt.show()
    trace_pulse_evolution = go.Surface(x=w/wf,y=distance,z=op_pulse_w, colorscale='Jet')
    layout_pulse_evolution = go.Layout(
        autosize = False,
        width=800,
        height=800,
        title='Pulse Evolution',
        scene=go.layout.Scene(
            xaxis=go.layout.scene.XAxis(title='f, TГц', range=[-10,10]),
            yaxis=go.layout.scene.YAxis(title='z, м'),
            zaxis=go.layout.scene.ZAxis(title='|U|^2, Ватт*с'))
    )
    pulse_evolution = go.Figure(data=[trace_pulse_evolution], layout=layout_pulse_evolution)
    
    plot(pulse_evolution, filename='./pulse_evolution_w.html')
    

#-----------------------------------------------------------------------
# BEGINNING

def plot_input_pulse():
    trace_input_pulse = go.Scatter(x=tau0,y=np.absolute(U*SMILE**5))
    layout_input_pulse = go.Layout(
        autosize = False,
        width=500,
        height=400,
        title='Input Pulse',
        xaxis=dict(
            title='Time',
            range=[-3000,3000],
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        ),
        yaxis=dict(
            title='Amplitude',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )
    )                   
    input_pulse = go.Figure(data=[trace_input_pulse], layout=layout_input_pulse)
    
    plot(input_pulse, filename='./input_pulse.html')

#-----------------------------------------------------------------------
# TIME SPACE

def plot_pulse_evo():
    trace_pulse_evolution = go.Surface(x=tau0,y=distance,z=op_pulse, colorscale='Jet')
    layout_pulse_evolution = go.Layout(
        autosize = False,
        width=800,
        height=800,
        title='Pulse Evolution',
        scene=go.layout.Scene(
            xaxis=go.layout.scene.XAxis(title='t, фс', range=[-3000, 3000]),
            yaxis=go.layout.scene.YAxis(title='z, м'),
            zaxis=go.layout.scene.ZAxis(title='|U|^2, Ватт'))
    )
    pulse_evolution = go.Figure(data=[trace_pulse_evolution], layout=layout_pulse_evolution)
    
    plot(pulse_evolution, filename='./pulse_evolution.html')
    
    
# Plots
print("> Plotting...", end="")
plot_input_pulse()
plot_pulse_evo()
plot_pulse_evo_w()
print("\r> Plotting complete!")