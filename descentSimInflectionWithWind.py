# -*- coding: utf-8 -*-

"""
Created on Tue Feb  9 11:08:50 2021

@author: stw2nf
"""

import numpy as np
import matplotlib.pyplot as plt
import math as mt
import pandas as pd
from tqdm import tqdm

# %% Determining Air Density
def rho(alt):
    alt = alt/m2ft #Convert to meters
    C1 = -3.9142e-14
    C2 = 3.6272e-9
    C3 = -1.1357e-4
    C4 = 1.2204
    rho_poly = (C1*alt**3 + C2*alt**2 + C3*alt + C4)*0.062428 #convert to (lb/ft^3)
    return rho_poly

# %% Simulating altitude response
def z_sim(startAlt, endAlt, dt, Mt, CdA, descentRateMean):

    z = np.ones(1) # Initializing altitude array, will append as sim runs
    zd = np.ones(1) # Init. z dot array
    zdd = np.ones(1) # Init. z double dot array
    t = np.ones(1) # Initializing time array, will append as sim runs
    t[0] = 0
    z[0] = startAlt # Start sim at release altitude
    if (descentRateMean < -50):
        zd[0] = -0.1 # Starting with 0 descent velocity (may have forward velocity though)
    else: 
        zd[0] = descentRateMean
        
    zdd[0] = 0 # 0 descent acceleration for first time step
    i = 0
    
    while (z[i] > endAlt): # Looping until altitude is below the ground
        i = i + 1
        if (descentRateMean < -50):
            zdd = np.append(zdd, g - np.sign(zd[i-1])*0.5/Mt*rho(z[i-1])*CdA*zd[i-1]**2)
            # z accel is weight and drag only (no other external force/propulsion)
            zd = np.append(zd, zd[i-1] + zdd[i]*dt)
            # z velocity is simple kinematic integration, vel = vel_prev + accel*d
            z = np.append(z, z[i-1] + zd[i]*dt + 0.5*zdd[i]*dt**2)
        
        else:
            
            z = np.append(z, (z[i-1] + descentRateMean*dt))
            zd = np.append(zd, descentRateMean) # Init. z dot array
            zdd = np.append(zdd, 0) # Init. z double dot array
            
        t = np.append(t, t[i-1] + dt) # Simple, but sticking in here for convenience
    return z,zd,zdd,t

# %% XY Translational Simulation
def xy_descent_sim(z, zd, t, dt, xGoal, yGoal, descentRateMean, wind_alt, wind_x, wind_y, pwind_alt, pwind_x, pwind_y):
    
    errThresh = 0.1 #prevents discontinuities for tangent calculation
    x = np.zeros(len(z))
    xd = np.zeros(len(z))
    y = np.zeros(len(z))
    yd = np.zeros(len(z))
    xError = np.zeros(len(z))
    yError = np.zeros(len(z))
    desAngle= np.zeros(len(z))
    thetaGoal = np.zeros(len(z))
    thetaGoalWind = np.zeros(len(z))
    
    # Setting initial conditions
    AoA = np.linspace(0,30,num=4)
    slowGlide = np.linspace(0, 10.25, num=4)
    medGlide = np.linspace(0, 4.98, num=4)
    fastGlide = np.linspace(0, 4.32, num=4)
    invGlide = np.linspace(0, 0.12, num=4)
    
    if descentRateMean == -58:
        xp = invGlide
    if descentRateMean == -5:
        xp = slowGlide
    if descentRateMean == -10:
        xp = medGlide
    if descentRateMean == -15:
        xp = fastGlide
    fp = AoA
     
    for i in range(1,len(x)):
        
        wind_pred_x = np.interp(z[i], pwind_alt, pwind_x)
        wind_pred_y = np.interp(z[i], pwind_alt, pwind_y)
        
        d = (((xGoal-x[i-1])**2 + (yGoal-y[i-1])**2))**(1/2) #How far from goal am I?
        
        #Prevents division by zero on final time step
        if t[-1] == t[i]:
            vt_goal = d/(dt)
        else:
            vt_goal = d/(t[-1]-t[i]) #I want to close distance to reach goal when I reach ground if possible
        
        
        #Populate x and y error
        xError[i] = xGoal - x[i-1]
        yError[i] = yGoal - y[i-1]
        
        #Prevents discontinuity in arctan calulation
        if abs(xError[i]) < errThresh:
            xError[i] = np.sign(xError[i])*errThresh
        
        #Where is my goal relative to me
        thetaGoal[i] = np.arctan2(yError[i], xError[i])

        vx_goal = vt_goal*np.cos(thetaGoal[i]) - wind_pred_x
        vy_goal = vt_goal*np.sin(thetaGoal[i]) - wind_pred_y
        vt_des = (vx_goal**2 + vy_goal**2)**(1/2)
        thetaGoalWind[i] =  np.arctan2(vy_goal, vx_goal)
        
        #Find glide necessary to achieve target velocity
        desGlide = vt_des/-zd[i]
    
        #If I cant achieve that glide pick the biggest one I can achieve
        if desGlide > max(xp):
                desGlide = max(xp)
        
        #Find angle to achieve glide setpoint
        desAngle[i] = np.interp(desGlide, xp,fp)
        
        wind_act_x = np.interp(z[i], wind_alt, wind_x)
        wind_act_y = np.interp(z[i], wind_alt, wind_y)
        
        #Calculate velocity and update new position
        xd[i] = desGlide*-zd[i]*np.cos(thetaGoalWind[i]) + wind_act_x #Add noise from wind here
        yd[i] = desGlide*-zd[i]*np.sin(thetaGoalWind[i]) + wind_act_y#Add noise from wind here
        
        x[i] = x[i-1]+xd[i]*dt
        y[i] = y[i-1]+yd[i]*dt
        
    return x, y, desAngle

# %% Power/Energy Consumption Simulation
def energy_sim(z, t, dt, descentRateMean, actualAoA):
    energy = np.ones(1) # Init X position array
    # Setting initial conditions
    energy[0] = 0 # Initial position from aircraft deployment (0,0 is "perfect")
    
    AoA = np.linspace(0,30,num=4)
    slowPwr = [136.44, 69.64, 138.32, 184.07]
    medPwr = [92.43, 66.43, 401.61, 167.34]
    fastPwr = [106.05, 52.92, 461.13, 96.25]
    invPwr = [221.5, 133.11, 142.07, 143.63]
    
    xp = AoA
    if descentRateMean == -58:
        fp = invPwr
    if descentRateMean == -5:
        fp = slowPwr
    if descentRateMean == -10:
        fp = medPwr
    if descentRateMean == -15:
        fp = fastPwr
            
    for i in range(1,len(actualAoA)):
        
        powerCons = np.interp(actualAoA[i], xp, fp)

        if descentRateMean>-50:
            pwrMultAir = mt.sqrt(rho(z[i])/rho(endAlt))
            pwrMultWeight = ((Mt*-g)**(3/2))/((Mact*-g)**(3/2))
        else:
            pwrMultAir = 1
            pwrMultWeight = 1
            
        energy = np.append(energy, energy[i-1] + (powerCons*pwrMultAir*pwrMultWeight*dt/3600))
        
    if descentRateMean<-50:
        flipMult = (((1.5*Mt*-g)**(3/2))/((1.5*Mact*-g)**(3/2)))
        
        energy[-1] = energy[-1] + flipEnergy*flipMult
    return energy

# %%
#Initialize parameters
descentRateMean = np.array([-58.0, -15.0, -10.0, -5.0]) #Slow, Medium, Fast and Inverted Descent Rates (ft/s)
legendValue = ["Inverted", "Fast", "Medium", "Slow"]
plotColors = ['b-', 'r-', 'g-', 'c-']
flipEnergy = 0.33 #Energy required to rescue fip (Whr)
totalEnergy = np.zeros(len(descentRateMean))
t2Grnd = np.zeros(len(descentRateMean))
d2Grnd = np.zeros(len(descentRateMean))
totalDescentEnergy = np.zeros(len(descentRateMean))
dfinal = np.zeros(len(descentRateMean))
transPower = 421.69 #Power to translate (Watts)
transSpeed = 65.62 #PX4 parameter defining max XY translational speed (ft/s)
m2ft = 3.28 #meter to foot conversion

startAlt = 14500 #Starting altitude (ft)
endAlt = 4500 #Ending altitude (ft)
start_x = 0.0 # Starting X location 
start_y = 0.0 # Starting Y location
dt = .1 # time step in seconds
g = -32.2 #gravity (ft^2/s)

# Drone Parameters #
CdA = 0.398 # Drone drag coefficient*Area (found experimentally)
Md = 2.19 # mass of drone (lbs)
Mb = 0.47 #mass of battery
Mp = 0#np.linspace(0,5,num=20)#0 #mass of payload

Mact = 2.66 #mass of actual drone used for testing
battEnergyMax = 34.04 #Maximum battery capacity (Wh)
ft2mile = 5280 #conversion for foot to miles

startx = 0
starty = 0

xGoalArray = np.linspace(1,14000,num=20)
yGoal = 0.01
eInv = []
eSlow = []
eMedium = []
eFast = []

airdensity = rho(909)
# %%
####################################
##### Main Simulation/Program ######
####################################
# fig1 = plt.figure(1)
# ax = fig1.add_subplot(111,projection='3d')

location =  r'C:\Users\Sean\Box\Scripts\Balloon Wind Files\8_matlab_asc.csv'
real_wind = pd.read_csv(location)
real_wind = real_wind.to_numpy()
wind_alt = real_wind[:,0]*m2ft #MSL altitude (ft)
wind_x = real_wind[:,1]*m2ft #ft/s
wind_y = real_wind[:,2]*m2ft #ft/s

location =  r'C:\Users\Sean\Box\Scripts\Balloon Wind Files\9_matlab_asc.csv'
pred_wind = pd.read_csv(location)
pred_wind = pred_wind.to_numpy()
pwind_alt = pred_wind[:,0]*m2ft #MSL altitude (ft)
pwind_x = pred_wind[:,1]*m2ft #ft/s
pwind_y = pred_wind[:,2]*m2ft #ft/s

for k in tqdm(range(len(xGoalArray))):
    Mt = Md + Mb + Mp #total mass with battery (lbs)
    xGoal = xGoalArray[k]
    looksGood = 0
    # fig1 = plt.figure(1)
    # ax = fig1.add_subplot(111,projection='3d')
    for i in range(len(descentRateMean)):
        #print(descentRateMean[i])
        z,zd,zdd,t = z_sim(startAlt, endAlt, dt, Mt, CdA, descentRateMean[i]) # Run z (altitude) simulation
        x_sim, y_sim, desAngle = xy_descent_sim(z, zd, t, dt, xGoal, yGoal, descentRateMean[i], wind_alt, wind_x, wind_y, pwind_alt, pwind_x, pwind_y)
        energyCons = energy_sim(z, t, dt, descentRateMean[i], desAngle)
        
        totalDescentEnergy[i] = energyCons[-1] #Energy consumed during descent (Whr)
        d2Grnd[i] = (x_sim[-1]**2 + y_sim[-1]**2)**(1/2)/ft2mile #Distance travelled in descent (ft)
        t2Grnd[i] = t[-1]/60 #Time to ground (mins)
        totalEnergy[i] = energyCons[-1]
        d2Goal = ((xGoal-x_sim[-1])**2 + (yGoal-y_sim[-1])**2)**(1/2)
        
        while ( d2Goal > (transSpeed*dt)) :
            
            thetaGoal = np.arctan2((yGoal-y_sim[-1]), (xGoal-x_sim[-1]))
        
            transSpeed_x = transSpeed*np.cos(thetaGoal)
            transSpeed_y = transSpeed*np.sin(thetaGoal)
            
            wind_act_x = np.interp(z[-1], wind_alt, wind_x)
            wind_act_y = np.interp(z[-1], wind_alt, wind_y)
        
            x_sim = np.append(x_sim, (x_sim[-1] + ((transSpeed_x + wind_act_x)*dt)))
            y_sim = np.append(y_sim, (y_sim[-1] + ((transSpeed_y + wind_act_y)*dt)))
            
            d2Goal = ((xGoal-x_sim[-1])**2 + (yGoal-y_sim[-1])**2)**(1/2)
            
            #print(distance[-1])
            t = np.append(t, (t[-1] + dt) )# Simple, but sticking in here for convenience
            z = np.append(z, endAlt)
            zd = np.append(zd, 0)
            zdd = np.append(zdd, 0)
            
            pwrMultWeight = ((Mt*-g)**(3/2))/((Mact*-g)**(3/2))
            energyCons = np.append(energyCons, (energyCons[-1] + (pwrMultWeight*transPower*dt/3600)))
            
        totalEnergy[i] = energyCons[-1]       
        
        if (descentRateMean[i] == -5):
            eSlow = np.append(eSlow,energyCons[-1])
            
        if (descentRateMean[i] == -10):
            eMedium = np.append(eMedium,energyCons[-1])
            
        if (descentRateMean[i] == -15):
            eFast = np.append(eFast,energyCons[-1])
            
        if (descentRateMean[i] == -58):
            eInv = np.append(eInv,energyCons[-1])
    
    #     x_sim = np.append(x_sim, xGoal)
    #     y_sim = np.append(y_sim, yGoal)
    #     ax.plot3D(x_sim, y_sim, z, alpha=0.5, linewidth=1, label=legendValue[i])
    #     ax.set_xlabel('X Distance (ft)')
    #     ax.set_ylabel('Y Distance (ft)')
    #     ax.set_zlabel('Altitude (ft)')
    #     ax.set_title('Method of Descent Comparison')
    
    # print('Enter any key to move on')
    # while looksGood == 0:
    #     looksGood = input()
    
    # plt.close()
        
    if (eInv[-1]>eSlow[-1]) or (eInv[-1]>eMedium[-1]) or (eInv[-1]>eFast[-1]):
            break
#     # %% Plotting Results
#     for j in range(len(angle)):
        
#         if j == 0:
#             ax.plot3D(np.cos(angle[j])*distance/ft2mile, np.sin(angle[j])*distance/ft2mile, z/ft2mile, plotColors[i], alpha=0.5, linewidth=1, label=legendValue[i])
#         else:
#             ax.plot3D(np.cos(angle[j])*distance/ft2mile, np.sin(angle[j])*distance/ft2mile, z/ft2mile, plotColors[i], alpha=0.5, linewidth=1)
            
#         ax.set_xlabel('X Distance (miles)')
#         ax.set_ylabel('Y Distance (miles)')
#         ax.set_zlabel('Altitude (miles)')
#         ax.set_title('Method of Descent Comparison')

fig1 = plt.figure(1)
plt.plot(xGoalArray[0:len(eInv)], eInv, linewidth=1, label=legendValue[0])
plt.plot(xGoalArray[0:len(eSlow)], eSlow, linewidth=1, label=legendValue[3])
plt.plot(xGoalArray[0:len(eMedium)], eMedium, linewidth=1, label=legendValue[2])
plt.plot(xGoalArray[0:len(eFast)], eFast, linewidth=1, label=legendValue[1])
plt.hlines(battEnergyMax, 0, max(xGoalArray), linestyle='dashed', label='Battery Capacity')
plt.xlabel('Standoff Distance (ft)')
plt.ylabel('Energy (Wh)')
plt.title('Method of Descent Comparison\n(10,000 AGL Drop)')
plt.grid(which='both', axis='both', color='k', linestyle='-', linewidth=.2)
fig1.legend()

# fig1 = plt.figure(1)
# plt.plot(Mp[0:len(eInv)], eInv, linewidth=1, label=legendValue[0])
# plt.plot(Mp[0:len(eSlow)], eSlow, linewidth=1, label=legendValue[3])
# plt.plot(Mp[0:len(eMedium)], eMedium, linewidth=1, label=legendValue[2])
# plt.plot(Mp[0:len(eFast)], eFast, linewidth=1, label=legendValue[1])
# plt.hlines(battEnergyMax, 0, max(Mp), linestyle='dashed', label='Battery Capacity')
# plt.xlabel('Weight of Payload (lbs)')
# plt.ylabel('Energy (Wh)')
# plt.title('Method of Descent Comparison\n(10,000 AGL Drop, 0 ft Standoff)')
# plt.grid(which='both', axis='both', color='k', linestyle='-', linewidth=.2)
# fig1.legend()
# fig1 = plt.figure(1)
# plt.plot(xGoal[0:len(eInv)], eInv, linewidth=1, label=legendValue[0])
# plt.plot(xGoal[0:len(eSlow)], eSlow, linewidth=1, label=legendValue[1])
# plt.plot(xGoal[0:len(eMedium)], eMedium, linewidth=1, label=legendValue[2])
# plt.plot(xGoal[0:len(eFast)], eFast, linewidth=1, label=legendValue[3])
# plt.xlabel('Standoff Distance (ft)')
# plt.ylabel('Energy (Wh)')
# plt.title('Method of Descent Comparison')
# plt.grid(which='both', axis='both', color='k', linestyle='-', linewidth=.2)
# fig1.legend() 

#     fig3 = plt.figure(3)
#     plt.plot(t,zd, plotColors[i], alpha=0.9, linewidth=1, label=legendValue[i])
#     plt.xlabel('Time, s')
#     plt.ylabel('Descent Rate, ft/s')
#     plt.title('Descent Rate Comparison')
#     plt.grid(which='both', axis='both', color='k', linestyle='-', linewidth=.2)
    
# for i in range(len(dfinal)):
#     fig4 = plt.figure(4)
#     rad = dfinal[i]
#     theta = np.linspace(0, 2*np.pi, 100)
#     a = rad*np.cos(theta)
#     b = rad*np.sin(theta)
#     plt.plot(a,b,plotColors[i],alpha=0.9,linewidth=1,label=legendValue[i])
#     plt.xlabel('X (miles)')
#     plt.ylabel('Y (miles)')
#     plt.title('Maximum Operating Window')
#     plt.grid(which='both', axis='both', color='k', linestyle='-', linewidth=.2)
    
# fig1.legend()
# fig2.legend()
# fig3.legend()
# fig4.legend()

#%% Formatting Results for Tables
# descentParameters = np.array([legendValue, np.around(descentRateMean,2), np.around(powerCons,2), np.around(glideRatio,2)]).T
# environmentParameters = np.array([startAlt, endAlt, transSpeed, transPower, Mt, CdA, flipEnergy]).T
# simResults = np.array([legendValue, np.around(t2Grnd,2), np.around(d2Grnd,2), np.around(totalDescentEnergy,2), np.around(dfinal,2)]).T
