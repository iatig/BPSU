#!/usr/bin/python3
#----------------------------------------------------------
#
#                  example-BP-SU.py
#             =======================
#
#  This program uses the BP-SU algorithm to:
#
#  1. Run BP
#  2. Use the converged messages to calculate <obs>_11 --- the expectation
#     value of the obs := |+><+| on site 11
#
#  3. Use the messages to calculate the Vidal gauge
#
#  4. Calculate <obs>_11 using the Vidal gauge
# 
#  5. Apply a 2-local gate (C-NOT) on sites 11,9 using the Vidal gauge
#
#  6. Absorb the Vidal weights into the TN
#  
#  7. Re-run the BP
#
#  8. Use the converged BP messages to re-calculate <obs>_11
#
#
# History:
# ---------
# 
# 28-Jun-2024   Initial version
#
# 18-Aug-2024   Added truncation error to apply_2local_gate
#
# 31-Aug-2024   Change the normalization after update to L_2 
#               normalization. Accordingly, when eps is given
#               use the L_2 norm to choose the truncation index.
#
#----------------------------------------------------------
#


import numpy as np

from numpy.linalg import norm

from numpy import zeros, ones, array, tensordot, sqrt, diag, dot, \
	reshape, transpose, conj, eye, trace, pi
	

from qbp import qbp, calc_e_dict

from BPSU import apply_2local_gate, BP_gauging, merge_SU_weights


#
# ------------------------- create_example  ---------------------------
#

def create_example_TN(d, D):
	
	"""
	 We create a random 12 spins PEPS according to the example in 
	 the document.
	 
	 Input Parameters:
	 ------------------
	 d --- Physical bond dimension
	 D --- Logical bond dimension
	 	 
	 	 
	 Output:
	 -------
	 
	 The T_list, e_list of the random TN
	 

	"""
	
	
	
	#           T0        T1       T2       T3          T4
	e_list = [ [1,2], [1,3,19,6], [5,3], [2,5,4,14], [19,4,7,8,20], 
	#   T5           T6        T7          T8               9
	[6,7,10], [8,10,9,11], [9,12], [11,20,15,18,17], [16,14,15],
	#  10            11
	[13,17,12], [16,18,13]]
	
	
	# The list of the original PEPS tensors. First index in each tensor
	# is the physical leg.
	T_list = []


	#
	# Create the local tensors as random tensors
	#
	
	for eL in e_list:
		k = len(eL)
		sh = [d] + [D]*k
		
		T = np.random.normal(size=sh) + 1j*np.random.normal(size=sh)
		T = T/norm(T)
		
		T_list.append(T)
		
	return T_list, e_list


#
# -------------------------  calc_rho_i  -------------------------------
#

def calc_rho_i(T, in_messages):
	
	"""
	
	Given a PEPS tensor T a corresponding list of incoming BP messages, 
	calculate the BP approximation of the reduced density matrix on that
	site
	
	
	"""
	
	m = len(in_messages)
	
	T_ket = T.copy()
	
	#
	# Contract the incoming messages to T_ket
	#
	
	for i in range(m):
		T_ket = tensordot(T_ket, in_messages[i], axes=([1],[0]))
		
	legs = list(range(1,m+1))
	
	rho = tensordot(T_ket, conj(T), axes=(legs, legs))
	
	rho = rho/trace(rho)
	
	return rho
	
	


#
#
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#
#                             M  A  I  N
#
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#
#


def main():
	
	print("\n\n\n")
	print("   EXAMPLE OF BP-SU: ")
	print("   =================\n")
	print("  Run BP on a random PEPS, use it to move to the Vidal gauge,")
	print("  calculate local expectation value, and apply a 2-local gate\n\n")
	
	np.random.seed(5)
	
	d = 2  # physical dimension
	D = 5  # PEPS bond dimension
	
	#
	# BP parameters
	#
	BP_MAX_ITER  = 50
	BP_DELTA     = 1e-6
	BP_INITIAL_M = 'U'
	BP_DAMPING   = 0
	
	#
	# The observable for which we are going to calculate the expectation
	# value
	#
	obs = array([[1,1],[1,1]])/2  # = |+><+|
	
	#
	# Create the TN from the example with random tensors
	#
	T_list, e_list = create_example_TN(d, D)
	e_dict = calc_e_dict(e_list)
	
	#
	# =======================   STEP 1: Run BP   =========================
	#
	print("\n")
	print("Step 1: Running BP")
	print("------------------------------------------------\n")
	
	m_list, err, iter_no = qbp(T_list, e_list, e_dict, \
		initial_m=BP_INITIAL_M, max_iter=BP_MAX_ITER, \
		delta=BP_DELTA, damping=BP_DAMPING)
		
	print(f"Done with {iter_no} iterations and convergence err {err:.6g}")
		
	#
	# =====================   STEP 2: Calc <obs>   =======================
	#
	
	#
	# Calc the RDM rho_i for i=11. Vertex 11 has edges [16,18,13], which 
	# correspond to neighboring vertices [9,8,10]
	#
	print("\n")
	print("Step 2: Calc <obs>_11 using the converged BP messages")
	print("----------------------------------------------------\n")
	
	T = T_list[11]
	
	#
	# Define a list of incoming messages to vertex 11
	#
	in_messages = [m_list[9][11], m_list[8][11], m_list[10][11]]
	
	rho_BP = calc_rho_i(T, in_messages)
	
	#
	# Once we have the RDM, we can easily calculate the local expectation
	# value
	#
	BP_av = trace(rho_BP@obs)
	BP_av = BP_av.real
	
	print(f"BP average: <obs>_11 = {BP_av:6g}")
	
	#
	# ==============  STEP 3: Calculate the Vidal Gauge   ================
	#
	
	#
	# The BP messages are used to calculate the Vidal gauge, which gives
	# us canonical tensors + edge weights
	#
	
	print("\n")
	print("Step 3: Calc the Vidal gauge")
	print("------------------------------\n")
	
	
	gauged_T_list, w_dict = BP_gauging(T_list, e_dict, m_list)

	#
	# ==========  STEP 4: Use the Vidal Gauge to calc <obs>  =============
	#

	#
	# In the Vidal gauge the effective messages are diagonal with 
	# the square of the weights on the diagonal. The edges of vertex 11
	# are [16,18,13], so we use their weights.

	print("\n")
	print("Step 4: Use the Vidal gauge to calc <obs>_11")
	print("--------------------------------------------\n")

	
	T = gauged_T_list[11]
	in_messages = [diag(w_dict[16]**2), diag(w_dict[18]**2), \
		diag(w_dict[13]**2) ]
	rho_BP = calc_rho_i(T, in_messages)
	
	Vidal_av = trace(rho_BP@obs)
	Vidal_av = Vidal_av.real
	
	print(f"Vidal gauge average: <obs>_11 = {Vidal_av:6g}")
	
	#
	# ==========  STEP 5: apply C-NOT on (9,11)  =============
	#

	print("\n")
	print("Step 5: Use the Vidal gauge to apply C-NOT(9,11)")
	print("------------------------------------------------\n")

	
	#
	# first, create the 2-local tensor that represents the CNOT
	#
	X = array([[0,1],[1,0]])
	ketbra00 = array([[1,0],[0,0]])
	ketbra11 = array([[0,0],[0,1]])
	CNOT = tensordot(ketbra00, eye(2), 0) + tensordot(ketbra11, X, 0) 
	
	#
	# The ID of the (9,11) edge is 16, and since 9<11, then 9 is the 
	# first qubit and 11 is the second qubit. So 9 will be the control
	# and 11 will be the target in the C-NOT gate.
	#
	
	gauged_T_list, w_dict, err = apply_2local_gate(gauged_T_list, e_list, \
		e_dict, w_dict, g=CNOT, e=16, Dmax=D)
	
	print(f" => *Hueristic* relative truncation error: {err:.6g}")
	
	#
	# Now gauged_T_list is the updated tensors with an approx Vidal gauge
	#
	
	#
	# ==========  STEP 6: merge the Vidal weights back into the TN =======
	#

	print("\n")
	print("Step 6: merge the Vidal weights back into the TN")
	print("---------------------------------------------------\n")

	T_list = merge_SU_weights(gauged_T_list, e_dict, w_dict)
	
	#
	# ==========  STEP 7: re-run BP =======
	#
	print("\n")
	print("Step 7: re-run BP")
	print("---------------------------------------------------\n")
	
	m_list, err, iter_no = qbp(T_list, e_list, e_dict, \
		initial_m=BP_INITIAL_M, max_iter=BP_MAX_ITER, \
		delta=BP_DELTA, damping=0)
		
	print(f"Done with {iter_no} iterations and convergence err {err:.6g}")
	
	#
	# ===============  STEP 8: Re-calculate <obs>_11   =================
	#
	print("\n")
	print("Step 8: Recalculate <obs>_11")
	print("------------------------------\n")
	
	T = T_list[11]
	in_messages = [m_list[9][11], m_list[8][11], m_list[10][11]]
	
	rho_BP = calc_rho_i(T, in_messages)
	
	BP_av = trace(rho_BP@obs)
	BP_av = BP_av.real
	
	print(f"BP average: <obs>_11 = {BP_av:6g}")
	
	
main()
