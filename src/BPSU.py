#-----------------------------------------------------------------------
#
#               BPSU.py            VERSION 0.1
#    ==================================================
#
#  Functions to perform Imaginary/Real Time Evolution (ITE) using
#  Simple Update (SU), with a BP gauge fixing.
#
#
# History
# =========================
#
# 18-Aug-2024: Itai  apply_2local_gate: added a the relative truncation 
#                    error as an output parameter to 
#
#
#=======================================================================



import numpy as np
import scipy

from numpy.linalg import norm, svd, qr

from numpy import zeros, ones, array, tensordot, sqrt, diag, conj, \
	eye, trace, pi, exp, isnan


HERMICITY_ERR = 1e-4
PINV_THRESH = 1e-12
ROBUST_THRESH = 1e8


#
# ---------------------------- sqrt_message  ---------------------------
#

def sqrt_message(m):
	
	"""
	
	Given a message m (which is a PSD matrix), calculate m^{1/2}, m^{-1/2}.
	
	Do that in a robust way by first diagonalizing, and then removing
	parts of the spectrum that are smaller than some threshold.
	
	"""
	
	#
	# Diagonalize
	#
	evals, U = np.linalg.eigh(m)
	
	#
	# The eigenvalues threshold: ignore the space of eigenvalues smaller 
	# than that.
	#
	
	thresh = evals[-1]*PINV_THRESH
	i = np.where(evals>thresh)[0][0]
	evals_red = evals[i:]
	U_red = U[:,i:]
	
	# 
	# Calculate m^{1/2}, and m^{-1/2}
	#
	M_sq = U_red@diag(sqrt(evals_red))@conj(U_red.T)
	Minv_sq = U_red@diag(evals_red**(-0.5))@conj(U_red.T)
	

	return M_sq, Minv_sq




#
# ------------------------ contract_leg  -------------------------------
#

def contract_leg(T, g, leg):
	
	newT = tensordot(T, g, axes=([leg+1],[0]))
	L = len(T.shape)
	perm = list(range(leg+1)) + [L-1] + list(range(leg+1,L-1))
	newT = newT.transpose(perm)
	
	return newT
	

#
# ------------------------  edge_BP_gauging  ---------------------------
#

def edge_BP_gauging(T1, leg1, T2, leg2, m12, m21):
	r"""
	
	Given two neighboring tensors, T1, T2, with a common edge, together
	with the two incoming/outgoing BP messages between them, perform 
	a re-gauging of their common leg to bring it to the Vidal gauge, 
	and calculate the weight of that edge.
	
	After the regauging, the TN looks like:
	
	  newT1      w      newT2
	----O--------o--------O----
	    |                 |
	
	
	and satisfies the Vidal gauge condition:
	
	         newT1
	      +---O----        +----
	     /    |            |
	m01 O     |      =     |
	     \    |            |
	      +---O----        +----
	         newT1*
	         
	         
	Input Parameters:
	------------------
	
	T1, T2 --- The tensors on  which we work
	
	leg1   --- The number of the leg in T1 that connects to T2, 0 being
	           the first leg, etc
	           
	leg2   --- Like leg1, but for T2
	
	m12    --- The T1 => T2 converged BP message
	
	m21    --- The T2 => T1 converged BP message
	
	Output:
	---------
	
	newT1, w, newT2 --- the new T1,T2, together with the Vidal weights.
	
	
	"""
	
	#
	# First calculate the square-root and its inverse for both BP 
	# messages.
	#
	m12_sq, m12inv_sq = sqrt_message(m12)
	m21_sq, m21inv_sq = sqrt_message(m21)
	
	#
	# Create the matrix in the middle
	#
	M = m12_sq.T@m21_sq
	
	U,s,V = svd(M, full_matrices=False)
		
	#
	# Calculate g1, g2 --- the gauge trans we apply to leg1,leg2 in T1, T2
	#
	g1 = m12inv_sq.T@U
	
	g2 = m21inv_sq.T@V.T
		
	#
	# Apply g1, g2 to T1, T2 and obtain newT1, newT2
	#
	
	newT1 = contract_leg(T1, g1, leg1)
	
	newT2 = contract_leg(T2, g2, leg2)
	
	return newT1, s, newT2
	



#
# ---------------------------  BP_gauging  -----------------------------
#

def BP_gauging(T_list, e_dict, m_list):
	"""
	
	Give a TN described by T_list, e_dict, together with a converged set
	of BP messages m_list, move the TN into the Vidal gauge in which 
	at the middle of every edge we place a diagonal weight tensor.
	
	We are following "Gauging tensor networks with belief propagation",  
	Joseph Tindall and Matt Fishman, SciPost Phys. 15, 222 (2023) here.
	
	Input Parameters:
	-----------------
	T_list --- The list of tensors that make up the TN
	
	e_dict --- The edges dictionary. The key is the edge name. The value
	           for an edge e=(i,j) is a 4-taple (i, leg_i, j, leg_j) 
	           
	m_list --- The converged BP messages. For every neighboring vertices
	           i,j, m_list[i][j] is the converged i=>j BP  message.
	
	
	Output:
	---------
	
	T_list --- The updated T_list
	w_dict --- A dictionary holding the weights of the Vidal gauge for
	           ever edge e.
	
	"""
	
	#
	# first, copy T_list to a new list
	#
	
	gauged_T_list = T_list.copy()
	
	
	w_dict = {}
	
	for e in e_dict.keys():
		
		vi, i_leg, vj, j_leg = e_dict[e]
				
		new_Ti, w_e, new_Tj = edge_BP_gauging(gauged_T_list[vi], i_leg, \
			gauged_T_list[vj], j_leg, m_list[vi][vj], m_list[vj][vi])
		
		gauged_T_list[vi] = new_Ti
		gauged_T_list[vj] = new_Tj
		
		# Normalize the weights
		w_dict[e] = w_e/sum(w_e)
		
		
		
	return gauged_T_list, w_dict
	
#
# ---------------------------  merge_SU_weights  -----------------------
#

def merge_SU_weights(T_list, e_dict, w_dict):
	
	"""
	
	Merge the SU weights back into the TN tensors. Each weight is split
	into 2 by taking a square, and then we swallow each part at a 
	neighboring tensor.
	
	"""
	
	merged_T_list = T_list.copy()
	
	for e in e_dict.keys():
		
		w = w_dict[e]
		sqw = sqrt(abs(w))
		sqM = diag(sqw)
		
		i1,leg1, i2,leg2 = e_dict[e]
		
		T1 = contract_leg(merged_T_list[i1], sqM, leg1)
		T2 = contract_leg(merged_T_list[i2], sqM, leg2)
		
		merged_T_list[i1] = T1
		merged_T_list[i2] = T2
		
	return merged_T_list
		
#
# --------------------------   gather_ext_legs   -----------------------
#

def gather_ext_legs(T, leg):
	
	"""
	
	Permute legs into [D0, D1, ..., d, Di] and then fuse the first Ds, 
	so that we get: [Drest, d, Di] ==> [Drest, d*Di]
	
	In addition, return the shape of the tensor before we coarse-grain
	the indices, so that we will know later how to undo this.
	
	
	"""
	
	L = len(T.shape)
	
	perm = list(range(L))
	perm.remove(0)
	perm.remove(leg+1)
	
	perm = perm + [0,leg+1]
	
	
	dD = T.shape[0]*T.shape[leg+1]
	Drest = T.size//dD
	
	M = T.transpose(perm)
	sh = list(M.shape)
	M = M.reshape([Drest, dD])
	
	return M, sh
	
	
#
# ---------------------------  apply_2local_gate -----------------------
#

def apply_2local_gate(T_list, e_list,  e_dict, w_dict, g, e, \
	Dmax=None, eps=None):
		
	r"""
	
	Given a TN in the Vidal gauge, apply a 2-body gate g on the tensors 
	of a given edge and truncate the bond dimension to Dmax using the 
	Simple-Update framework.
	
	A detailed explanation of the algorithm can be found at 
	"Universal tensor-network algorithm for any infinite lattice",
	PRB 99, 195105 (2019) 
	
	Input Parameters:
	------------------
	
	T_list, e_list, e_dict, w_dict --- The description of the TN
	
	g    --- The 2-local gate, given as [i1,j1; i2,j2] where j1,j2 are 
	         the ket legs and i1,i2 are the bra legs.
	      
  e    --- The label of the edge on which the gate is acting
           note that e=(i,j) where i<j
  
  Dmax --- The maximal final bond dimension. If not given, no truncation
           is done.
           
  eps  --- Another truncation criteria. If given, we truncate all 
           singular values starting from k such that
           \sqrt{\sum_{i>= k} s_i^2} \le \eps. If given in conjuncation 
           with Dmax, then the minimal bond dimension is used.
  
          
  Output:
  --------
  
  T_list, w_dict --- Tensors of the updated TN.
  
  truncation_error --- The relative truncation error of the SVD 
                       coefficients. If we truncated all s_i 
                       with i>R then:
                       
                           sqrt[ \sum_{i>R} s_i^2 / \sum_i s^2]
	      
	
	
	"""
	
	
	if Dmax is None:
		Dmax = 1000000
	
	#
	# Locate the vertices of the edge e=(i1,i2) and their tensors T1, T2
	#
	
	
	i1,leg1, i2,leg2 = e_dict[e]
		
	T1 = T_list[i1]
	T2 = T_list[i2]
	w = w_dict[e]
	
	D = T1.shape[leg1+1]  # Original dimension of the common leg
	d1 = T1.shape[0]  # physical leg T1
	d2 = T2.shape[0]  # physical leg T2
	
	# ---------------------------------------------------------------
	# 1. Absorb all the weights of T1, T2 into these tensors (except
	#    for the weight of the common leg
	# ---------------------------------------------------------------
		
	es1 = e_list[i1]
	for leg,f in enumerate(es1):
		
		if f==e:
			continue
		
		w_mat = diag(w_dict[f])
		
		T1 = contract_leg(T1, w_mat, leg)

	es2 = e_list[i2]
	for leg,f in enumerate(es2):
		
		if f==e:
			continue
		
		w_mat = diag(w_dict[f])
		
		T2 = contract_leg(T2, w_mat, leg)
	
	# -----------------------------------------------------------------
	# 2. Reshape T1, T2 into matrices, where one leg is the fusion of 
	#    all non-participating legs, and the second is (d,D), where 
	#    d is the physical leg and D is the common leg
	# -----------------------------------------------------------------
	
	M1, T1_shape = gather_ext_legs(T1, leg1)
	M2, T2_shape = gather_ext_legs(T2, leg2)


	# -----------------------------------------------------------------
	# 3. Perfrom QR on M1, M2 to separate (d,D) legs from the rest.
	# -----------------------------------------------------------------
	
	Q1,R1 = qr(M1)
	Q2,R2 = qr(M2)
		
	# -------------------------------------------------
	# 4. Separate d from the common leg in R1, R2
	# -------------------------------------------------
	
	R1 = R1.reshape([R1.shape[0], d1, D])
	R2 = R2.reshape([R2.shape[0], d2, D])
	
	# -------------------------------------------------
	# 5. Contract: R1 + R2 + w + g
	# -------------------------------------------------
	
	#
	# First, contract R1 with the SU weight
	#
	
	R1 = tensordot(R1, diag(w), axes=([2],[0]))
	
	#
	# Second, contract R1 with the gate g.
	# 
	#  R1 shape: [RestL, d1, D]
	#  g  shape: [i1, j1; i2, j2]
	# 
	#  We contract d1<-->j1
	#
	#

	R1 = tensordot(R1, g, axes=([1], [1]))
	#
	# R1 form: [RestL, D, i1, i2, j2]
	#
	# R2 form: [RestR, d2, D]
	#
	# Now contract with R2 along D<-->D and d2<-->j2
	#
	R12 = tensordot(R1, R2, axes=([1, 4], [2,1]))
	
	# Final R12 form: [RestL, i1, i2, RestR]
	
	#
	# 6. Turn R12 into a matrix, and SVD it
	#
  
	sh = R12.shape
	
	R12 = R12.reshape([sh[0]*sh[1], sh[2]*sh[3]])
		
	U, s, V = svd(R12, full_matrices=False)
	
	D_full = len(s)
	
	# -------------------------------------------------
	# 7. Truncate (if needed)
	# -------------------------------------------------
	if D_full>Dmax or eps is not None:
		
		if eps is not None:
			#
			# If eps is given, then we truncate all singular values from 
			# the index k s.t. (s[k]**2 + s[k+1]**2 + ...)^{1/2} < eps*||s||
			# where ||s|| is the L_2 norm of s.
			#
			# In other words, we truncate such that the L_2 truncation error
			# will be at most eps.
			#
			
			#
			# Calculate psums --- an array of partial sums of s^2, where:
			#
			# psums[k] = s[k]**2 + s[k+1]**2 + ...
			#
			
			s2 = s**2
			psums = np.cumsum(s2[::-1])
			psums = psums[::-1]

			# normalize it by the overall L2 norm
			psums = psums/psums[0]
			
			# Find the place where we need to truncate
			i = np.where(psums<eps**2)[0][0]
			

			if Dmax>i:
				Dmax = i
		
		#
		# Re-define the unitaries U,V and the weights s to contain only
		# the non-truncated values
		# 
		
		#
		# Calculate the relative truncation error
		#
		truncation_error = sqrt( sum(s[Dmax:]**2)/sum(s**2) )
		
		s = s[:Dmax]
		U = U[:,:Dmax]
		V = V[:Dmax, :]
		
		
	else:
		
		truncation_error = 0.0
	
	#
	# Normalize the final SU weights by the L_2 norm
	#
	s = s/sqrt(sum(s**2))
	
		
	# -------------------------------------------------
	# 8. Open up the d legs in U,V
	# -------------------------------------------------
	
	sh = U.shape
	D = sh[1]  # Final bond dimension
	
	U = U.reshape([sh[0]//d1, d1, sh[1]])
	# U shape: [RestL, d, D]
	
	
	sh = V.shape
	V = V.reshape([sh[0], d2, sh[1]//d2])
	# V shape: [D, d,  RestR]
	V = V.transpose([2,1,0])
	# V shape: [RestR, d, D]
	
	
	# -------------------------------------------------
	# 9. Contract U<-->Q1 and V<-->Q2
	# -------------------------------------------------
	
	Q1 = tensordot(Q1, U, axes=([1],[0]))
	Q2 = tensordot(Q2, V, axes=([1],[0]))
	
	# Q1 shape: [RestL, d, D]     Q2: [RestR, d, D]
	
	# -------------------------------------------------
	# 10. Separete the rest of the legs in Q1, Q2
	# -------------------------------------------------
	T1_shape[-1] = D
	T2_shape[-1] = D
	
	T1 = Q1.reshape(T1_shape)
	T2 = Q2.reshape(T2_shape)
	
	# T1 shape: other-legs, d1, D
	# T2 shape: other-legs, d2, D
	
	# -------------------------------------------------
	# 11. Re-arrange the legs of T1, T2
	# -------------------------------------------------
	
	sh = T1.shape
	L = len(T1.shape)
	perm = [L-2] + list(range(leg1)) + [L-1] + list(range(leg1,L-2))
	T1 = T1.transpose(perm)
	
	sh = T2.shape
	L = len(T2.shape)
	perm = [L-2] + list(range(leg2)) + [L-1] + list(range(leg2,L-2))
	T2 = T2.transpose(perm)
	
	# -----------------------------------------------------
	# 12. Remove the SU weights from the rest of the legs 
	# -----------------------------------------------------
	
	es1 = e_list[i1]
	for leg,f in enumerate(es1):
		
		if f==e:
			continue
		
		smax = w_dict[f][0]*PINV_THRESH
		k = w_dict[f].shape[0]
		w_mat = diag(1/(w_dict[f] + smax*ones(k)))
		
		T1 = contract_leg(T1, w_mat, leg)

	es2 = e_list[i2]
	for leg,f in enumerate(es2):
		
		if f==e:
			continue
		
		smax = w_dict[f][0]*PINV_THRESH
		k = w_dict[f].shape[0]
		w_mat = diag(1/(w_dict[f] + smax*ones(k)))
		
		T2 = contract_leg(T2, w_mat, leg)

	
	# -----------------------------------------------------------
	# 13. Update T1, T2, w in the T_list, w_dict list/dictionary
	# -----------------------------------------------------------
		
	T_list[i1] = T1
	T_list[i2] = T2
	w_dict[e] = s

	
	return T_list, w_dict, truncation_error
	
	
#
# -------------------------  local_2RDMs  ------------------------------
#

def local_2RDMs(T_list, e_list,  e_dict, w_dict):
	
	rho_dict={}
	
	for e in e_dict.keys():
		v1,leg1, v2,leg2 = e_dict[e]
		
		T1 = T_list[v1]
		T2 = T_list[v2]
		w = w_dict[e]
	
		D = T1.shape[leg1+1]  # Original dimension of the common leg
		d1 = T1.shape[0]  # physical leg T1
		d2 = T2.shape[0]  # physical leg T2
	
		#
		# Absorb all the weights of T1, T2 into these tensors (except
		# for the weight of the common leg
		# 
			
		es1 = e_list[v1]
		for leg,f in enumerate(es1):
			
			
			w_mat = diag(w_dict[f])
			
			T1 = contract_leg(T1, w_mat, leg)

		#
		# contract T1 with the bra along all legs except for the connecting
		# one
		#
		L = len(T1.shape)
		sh = list(range(L))
		sh.remove(0)
		sh.remove(leg1+1)
		
		T1ketbra = tensordot(T1, conj(T1), axes=(sh, sh))
		# T1ketbra form: d, D, d*, D*
		

		es2 = e_list[v2]
		for leg,f in enumerate(es2):
			
			if f==e:
				continue
			
			w_mat = diag(w_dict[f])
			
			T2 = contract_leg(T2, w_mat, leg)
			
		#
		# contract T2 with the bra along all legs except for the connecting
		# one
		#
		L = len(T2.shape)
		sh = list(range(L))
		sh.remove(0)
		sh.remove(leg2+1)
			
		T2ketbra = tensordot(T2, conj(T2), axes=(sh, sh))
		# T2ketbra form: d, D, d*, D*
		
		
		#
		# get rho12 by contracting T1ketbra with T2ketbra along D,D*
		#
			
		rho12 = tensordot(T1ketbra, T2ketbra, axes=([1,3],[1,3]))
		
		tr = trace(rho12, axis1=0, axis2=1)
		tr = trace(tr, axis1=0, axis2=1)
		
		rho12 = rho12/tr
		
		rho_dict[e] = rho12
		
		
	
	
	return rho_dict
	
	
