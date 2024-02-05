import ctypes
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

# Load the C library
c_lib = ctypes.CDLL('./libRepressilator_ring.so')
num_points = 1000
cyc=4
# Define the function signature in Python
c_lib.simulate_repressilator.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # params_array
                                         np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # initial_conditions
                                         np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # time_span
                                         ctypes.c_double,
                                         ctypes.c_int]  # num_points
c_lib.simulate_repressilator.restype = ctypes.POINTER(ctypes.c_double * (2*cyc+1) * num_points)

# Prepare data for the C function
params_array = np.array([1, 1, 1, 0.01, 10, 2, cyc], dtype=np.float64)  # deg_m, deg_p,alpha, alpha_0, beta, n
initial_conditions = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0,4.0], dtype=np.float64)  # m1, m2, m3, p1, p2, p3 ,
time_span = np.array([0, 50], dtype=np.float64)  # Start time, end time

# Call the C function
result_ptr = c_lib.simulate_repressilator(params_array, initial_conditions, time_span, num_points,cyc)

# Convert the results from C to a Python format
results = np.ctypeslib.as_array(result_ptr.contents)


# Assuming the number of time steps equals the number of rows in the data
#time_steps = np.linspace(np.min(time_span),np.max(time_span), num=num_points)

# Plotting
plt.figure(figsize=(12, 6))
for z1 in range(1,cyc+1):
    plt.plot(results[:,0], results[:, z1], label="mRNA "+str(z1), alpha=0.7)
for z2 in range(cyc+1,2*cyc+1):
    plt.plot(results[:,0], results[:, z2], label="Protein "+str(z2-cyc), alpha=0.7)

plt.xlabel("Time")
plt.ylabel("Output")
plt.title("mRNA and Proteins Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.show()
plt.savefig("repressilator_variables_ring.jpg", bbox_inches="tight", dpi=300)
plt.savefig("repressilator_variables_ring.png", bbox_inches="tight", dpi=300, transparent=True)

# Don't forget to free the allocated memory in C
c_lib.free(result_ptr)  # Assuming you have a 'free' function in your C library to free the memory
