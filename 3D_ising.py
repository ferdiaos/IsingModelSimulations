# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 16:58:41 2022

@author: ferdi
"""
import numpy as np
import matplotlib.pyplot as plt

N=15
#create function to generate a lattice of randomly generated spins, roughly 75% up and 25% down
def random_spins_p(N):
    """This function generates a random NxNxN lattice of spins with roughly 75% of them spin up and
    the other 25% spin down, randomly distributed throught it
    
    N represents the the length of one of the dimensions of the lattice. Must be given as an integer
    value
    
    Returns lattice of random spins"""
    #first randomly generate an NxN lattice with random integers between 1 and N**2
    init_lattice = np.random.random(size=(N,N,N))
    #create for loop where for if the number generated is greater than 1/4 of N**2, its spin will be up, else 
    #spin down
    for i in range(0,N):
        for j in range(0,N):
            for k in range(0,N):
                if init_lattice[i,j,k] < 0.25:
                    init_lattice[i,j,k]=1
                else:
                    init_lattice[i,j,k]=-1
    return init_lattice

init_lattice=random_spins_p(N)

def neighbour_spin_total(initial_states,N):
    """""This function finds the energy of nearest neighbours in an Ising model by multiplying each state with its 6 nearest neighbours
    for a cube 3D model. For states on the edge or corner of the lattice, as a lattice is repeated, it is multiplied by its  
    nearest neighbours in the lattice and the states that it would be next to if the lattice had been repeated
    
    initial_states is the numpy array for the lattice of spin states
    N is the integer length of one of the dimensions of the lattice, as it is a cube
    
    Returns the total sum of all neighbouring energies"""
    #take initial energy as 0
    neighbour_spin_total=0
    #create loop for over all of the lattice
    for x in range(0,N):
        for y in range(0,N):
            for z in range(0,N):
                neighbour_spin_total+=initial_states[x,y,z]*(initial_states[np.mod((x+1),N),y,z]+initial_states[x-1,y,z]+initial_states[x,np.mod(y+1,N),z]+initial_states[x,y-1,z]+initial_states[x,y,np.mod((z+1),N)]+initial_states[x,y,z-1])
                
    
    #energy returned as negative
    return -neighbour_spin_total

init_energy=neighbour_spin_total(init_lattice,N)
k=1.38*10**-23

def monte_carlo(iterations,lattice,J,T,energy):
    """"This function takes an initial lattice os spin states, and applies the Metropolis Monte Carlo algorithm on it for a given number
    of iterations, to find the equilibrium energy of the system, the state it requires, and also returns the net total of spin of the 
    system. The probability chosen for the system for if to keep a change in spin, if the overall energy increase is positive, is given
    by assuming detailed balance and thus the probability has to be e^(-delta_E*B*J), where B=1/kT

    interations is the number of times the algorithm is used on the lattice
    lattice is the initial state that wants to be programmed
    BJ is the exponential factor, or the ratio of J/kT
    Energy is the nearest neighbour energy of the initial lattice
    
    Returns an array of the sum of all the spins after an iteration of the method
    Returns the total nearest neighbour energy as an array after each iteration
    Returns the lattice for the equilbrium state"""
    #generate lattices for the spins and energies
    lattice=lattice.copy()
    total_spins=np.zeros(iterations-1)
    total_energy=np.zeros(iterations-1)
    #find probabilities for positive energy changes by finding the exponential constants
    diff_4=np.exp(-4*J/(k*T))
    diff_8=np.exp(-8*J/(k*T))
    diff_12=np.exp(-12*J/(k*T))
    #start Monte Carlo loop for the given number of iterations
    for i in range(iterations-1):
        #find a random point in the lattice to flip the spin
        x=np.random.randint(0,N)
        y=np.random.randint(0,N)
        z=np.random.randint(0,N)
        #flip the spin
        spin_i=lattice[x,y,z]
        spin_f=-lattice[x,y,z]
        
        #find the difference in energies between the two
        #sum the neighbours around the spin flipped
        nearest_neighbours=(lattice[np.mod((x+1),N),y,z]+lattice[x-1,y,z]+lattice[x,np.mod(y+1,N),z]+lattice[x,y-1,z]+lattice[x,y,np.mod((z+1),N)]+lattice[x,y,z-1])
        #multiply by the spin initially and after
        E_i=-spin_i*nearest_neighbours
        E_f=-spin_f*nearest_neighbours
        
        #find the difference between the two energies
        E_diff=E_f-E_i
        #for when there is an increase in energy, only accept with a given probability
        if E_diff>0:
            #maximum total energy difference of 12, at intervals of 4, so calculate for each case
            if E_diff==4:
                if np.random.random() < diff_4:
                    lattice[x,y,z]=spin_f
                    energy+=E_diff
            if E_diff==8:
                if np.random.random() < diff_8:
                    lattice[x,y,z]=spin_f
                    energy+=E_diff
            if E_diff==12:
                if np.random.random() < diff_12:
                    lattice[x,y,z]=spin_f
                    energy+=E_diff
        #for when the change in energy is 0 or less, accept regardless
        else:
            lattice[x,y,z]=spin_f
            energy+=E_diff
        
        #find the sum of the spin and energy and add them to the lattice
        total_spins[i]+=np.sum(lattice)
        total_energy[i]+=energy
    
    return total_spins, total_energy, lattice

    
iterations=100000
J=1*10**-21
T_low=20
T_high=20000
times=np.arange(iterations-1)
spins_low,energies_low, final_lattice=monte_carlo(iterations,init_lattice,J,T_low,init_energy)
spins_high,energies_high, final_lattice=monte_carlo(iterations,init_lattice,J,T_high,init_energy)
fig1=plt.figure()
plt.plot(times,spins_low, label=('T=20K'))
plt.plot(times,spins_high, label=('T=20000K'))
plt.title('Change in spin over iterations of Monte Carlo Algorithm')
plt.xlabel('Iterations')
plt.ylabel('Total spin')
plt.legend()
fig2=plt.figure()
plt.plot(times,energies_low, label=('T=20K'))
plt.plot(times,energies_high, label=('T=20000K'))  
plt.title('Change in energy (E/J) over iterations of Monte Carlo Algorithm')
plt.xlabel('Iterations')
plt.ylabel('Nearest neighbour energy')
plt.legend()

def mean_and_variance(iterations, init_lattice, min_temp, max_temp, J, energy):
    """This function computes the mean and variance of the spins and energies for the Ising model over varying temperatures, by looping
    over the Monte Carlo Algorithm over a given number of iterations for a given temperature and inital lattice state, to find the spins 
    and energies of that model, computing their respective means and variances, then repeating for the next temperature up.
    
    iterations is the number of time you wish to iterate over the Monte Carlo algorithm
    init_lattice is the initial lattice state being computer
    min_temp is the starting temperature, must be given as an integer greater than 0
    max_temp is the ending temperature, must be given as an integer greater than min_temp
    J is a positive constant between 0 and 1 that acts as the interaction force between spins
    energy is the initial nearest neighbour spin energy of the lattice
    
    Returns as an array spin mean, spin variance, energy mean, energy variance, energy given in terms of E/J"""
    #create an array for the incrementing temperature
    n=0
    temps=np.arange(min_temp, max_temp, 10)
    #create arrays for all means/variances
    spin_mean=np.zeros(len(temps))
    spin_variance=np.zeros(len(temps))
    energy_mean=np.zeros(len(temps))
    energy_variance=np.zeros(len(temps))
    #loop over all temperatures the Monte Carlo algorithm to find the spins and energies of each lattice for each temperature
    for i in range(len(temps)):
        spins, energies, final_lattice=monte_carlo(iterations, init_lattice, J, temps[i], energy)
        #compute the mean and variances and add them to each of their respective arrays
        spin_mean[i]+=np.mean(spins)/(N**3)
        spin_variance[i]+=np.var(spins)/(N**3)
        energy_mean[i]+=np.mean(energies)
        energy_variance[i]+=np.var(energies)
        n+=1
        print(n)
        
    return spin_mean, spin_variance, energy_mean, energy_variance

iterations=10000
min_temp=50
max_temp=1000
J=10**-21
temp_range=np.arange(min_temp,max_temp, 10)
sm, sv, em, ev=mean_and_variance(iterations, init_lattice, min_temp, max_temp, J, init_energy)

spec_heat=ev/k
chi=sv/k

fig3=plt.figure()
plt.plot(temp_range,sm)
plt.title('Mean spin of equilibrium lattice for varying temperatures')
plt.xlabel('Temperature (K)')
plt.ylabel('Mean spin of system')

fig4=plt.figure()
plt.plot(temp_range,em)
plt.title('Mean nearest neighbour energy of equilibrium lattice for varying temperatures')
plt.xlabel('Temperature (K)')
plt.ylabel('Mean nearest neighbour energy (E/J)')

fig5=plt.figure()
plt.plot(temp_range,spec_heat)
plt.title('Specific heat capacity of equilibrium lattice for varying temperatures')
plt.xlabel('Temperature (K)')
plt.ylabel('Specific heat capacity multiplied by square of temperature (ET/KJ)')

fig6=plt.figure()
plt.plot(temp_range,chi)
plt.title('Magnetic susceptibility of equilibrium lattice for varying temperatres')
plt.xlabel('Temperatire (K)')
plt.ylabel('Magnetic susceptibility multiplied by temperature (T/K)')