# ----------------------------------------------------------------------
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
# 18-Oct-2024: Itai  Add documentation to contract_leg() function
#
#
# 20-Jun-2025: Major changes:
#   - add sqrt_message2() to have a more rubust way of taking the
#     square of a BP message, as used in the function edge_BP_gaugin().
#
#   - Make edge_BP_gauging() more robust by adding a small random
#     perturbation if the SVD does not converge. Also normalize the weights according to L_2 instead of L_1
#
#   - add the local_enviless_truncation(), global_enviless_truncation()
#     functions to perfrom an evnironment-free truncation of a PEPS tensor
#
#   - added apply_2local_gate_notrunc() function to apply 2-gates
#     without any truncaiton
#
#   - Added apply_PEPO_to_PEPS() function, which applies a PEPO to a
#     PEPS (without compression)
#
#   - Added truncate_weights() function that truncates a set of Vidal
#     weights given a Dmax and L2thresh thereshold. It is called from
#     the BP_compress() function
#
#   - Added BP_compress() to compress a PEPS according to a given Dmax,
#     L2thresh threshold. This is done using BP.
#
#   - Added PEPO_to_PEPS(), PEPS_to_PEPO() functions to map between
#     these two TNs (simply by fusing or un-fusing the two physical
#     legs of the PEPO)
#
#   - Added peps_dist() --- A debugging function to calculate the
#     distance between two PEPS.
#
#   - Added the  BP_compress_PEPO() function which compresses a PEPO by
#     first turning it to a PEPS and then running BP_compress().
#
#
# 9-Jul-2025: Normalize the output tensors in BP_gauging according
#             to their L_2 norm (before it was just the weights that
#             were normalized)
#
# 9-Jul-2025: Add Dmax parameter to BP_compress_PEPO
#
# 10-Jul-2025: Add the functions direct_apply_2local_gate,
#              apply_gate_to_PEPS, apply_gate_to_PEPO. Add the normalize
#              flag to BP_compress and BP_compress_PEPO functions. Add
#              Dmax parameter to BP_compress_PEPO function.
#
# 15-Jul-2025: Added the lazy-compress functionality. This includes
#              adding the functions lazy_PEPS_compression,
#              lazy_PEPO_compression, and the functions on which they
#              rely: lazy_edge_truncation, lazy_sqrt_message.
#
# 21-Jul-2025: Fixed small bug in lazy_PEPS_compression
#
# 10-Aug-2025: Added some comments, increased EPS, EPS_TEST in
#              sqrt_message2 since now we use the Vidal gauge more for
#              regularization rather than actually fulfilling the gauge
#              equations.
#
# 22-Sep-2025: Fixed a crucial bug in lazy_edge_truncation: call
#              lazy_sqrt_message with m.T instead of m. In addition
#              small cosmetic changes.
#
# 24-Nov-2025: In merge_SU_weights(), make sure that the merged tensors
#              have the same precision (DP or SP) as the input tensors.
#
# 04-Mar-2026: In lazy_PEPS_compression(), added the calculation of the
#              simulation fidelity as defined in arXiv:2503.20870v2.
#
# 14-Mar-2026: Fixed a small error in lazy_edge_truncation (a line
#              D = T1.shape[leg1+1] was probably left there by accident)
#              and in addition add some documentation and re-arrange 
#              the truncation logic.
#
# 21-Mar-2026: 1. Removed the function peps_dist() and introduced the 
#                 functions fuse_ket_bra_tensors(), fuse_ket_bra_PEPS()
#              2. Added documentation to apply_gate_to_PEPO()
#              3. Added normalize_tensors flag to lazy_PEPS_compression()
#
# ======================================================================



import numpy as np
import scipy

from numpy.linalg import norm, svd, qr

from numpy import zeros, ones, array, tensordot, sqrt, diag, conj, \
	eye, trace, pi, exp, isnan, vdot


from qbp import qbp, get_Bethe_free_energy, adj_vert

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
# ---------------------------- sqrt_message  ---------------------------
#

def sqrt_message2(m):

	"""

	Given a message m (which is a PSD matrix), calculate m^{1/2}, m^{-1/2}.

	Do that in a robust way by first diagonalizing, and then padding
	the smallest values by EPS*(largest e.v.).

	We do not want to use the Penrose inverse because that would mean
	we will not create a fully invertible matrix. But we need the matrix
	to be fully invertible, for otherwise the BP-assited Vidal gauge will
	not be an actual gauge, i.e., it will change the underlying quantum
	state. So there's always a tradeoff between how much you preserve the
	underlying quantum state and how much the resultant TN satisfies the
	canonical equations.

	If the Vidal gauge is used only for sake of numerical stability, then
	it is better to increase EPS --- going for numerical stabilility in
	favour of canonicality.



	"""

	#
	# Inversion tolerance. We change any e.v. with relative size<EPS
	# to EPS, thereby making the matrix invertible.
	#
	EPS = 1e-10


	#
	# Self test the sqrt messages
	#
	TEST_CORRECTNESS = False
	EPS_TEST=1e-8


	#
	# Diagonalize (in a robust way)
	#

	converged = False
	d = m.shape[0]
	m1 = m

	dround = 1
	while not converged:

		converged = True

		try:
			evals, U = np.linalg.eigh(m1)

		except:
			print(f"Warnning: LinAlgError occured in BPSU.sqrt_message while "\
				f"trying to use linalg.eigh. Adding a small "\
				f"random perturbation and trying again (round {dround}).")

			N = np.random.normal(size=(d,d))
			N2 = N@N.T
			N2 = N2/norm(N2)
			m1 = m + EPS*N2*norm(m)
			dround += 1
			converged = False

	#
	# The eigenvalues threshold: all the eigenvalues that are smaller
	# then some threshold, are changed to that threshold. This gurantees
	# that:
	# 1) m is invertible (we need that for m^{-1/2})
	# 2) our approximation is close to the original m
	#

	thresh = evals[-1]*PINV_THRESH
	if evals[0]<thresh:
		i = np.where(evals>thresh)[0][0]
		evals[:i] = evals[-1]*PINV_THRESH

	#
	# Calculate m^{1/2}, and m^{-1/2}
	#
	M_sq = U@diag(sqrt(evals))@conj(U.T)
	Minv_sq = U@diag(1/sqrt(evals))@conj(U.T)


	if TEST_CORRECTNESS:
		eq1 = norm(M_sq@M_sq-m)
		eq2 = norm(M_sq@Minv_sq-eye(m.shape[0]))
		nr_m = norm(m)

		if eq1>EPS_TEST*nr_m:
			print("Warnning: Error in sqrt_message2: " \
				f"norm(M_sq@M_sq-m)/norm(m)={eq1:.6g} > norm(m)*{EPS_TEST} "\
				f"for norm(m)={nr_m:.6g}\n")

		if eq2>EPS_TEST*nr_m:
			print("Warnning: Error in sqrt_message2: " \
				f"norm(M_sq@Minv_sq-I)={eq2:.6g} > norm(m)*{EPS_TEST} "\
				f"for norm(m)={nr_m:.6g}\n")


	return M_sq, Minv_sq



#
# ------------------------ contract_leg  -------------------------------
#

def contract_leg(T, g, leg):
	r"""

	Given a tensor T and a matrix g, return a new tensor T' which is
	the contraction of g along the T's leg (indicated by leg).

	The leg of T is contracted to the *first* leg of g.

	If alpha_i is the index of the i'th leg, then the new tensor
	is given by

	\sum_{\alpha_i} T_{..., alpha_i, ...} g_{alpha_i, \beta}

	The indexing of leg *does not* include the physical leg. So
	if T=T[i0,i1,i2,...] then i0 is the physical leg, and so setting
	leg=0 means we contract via i1.

	Input Parameters:
	-------------------
	T   --- The tensor to be contractged
	g   --- The matrix
	leg --- index of the leg of T that we contract

	Output:
	-------
	The new T (legs are permuted back to their original order)


	"""

	newT = tensordot(T, g, axes=([leg+1],[0]))
	L = len(T.shape)
	perm = list(range(leg+1)) + [L-1] + list(range(leg+1,L-1))
	newT = newT.transpose(perm)

	return newT


#
# ------------------------  lazy_sqrt_message  ----------------------
#
def lazy_sqrt_message(m):
	r"""

	Find the square root of a BP message to be then used in the lazy
	compression.

	We do not need an Hermitian square root. So if m is the BP message,
	which is an SPD, then we diagonalize it

	m = U \lambda U^\dagger

	and then return sqrt(m) := \sqrt(\lambda) U^\dagger



	"""

	ZERO_THRESH = 1e-14

	#
	# Diagonalize
	#
	evals, U = np.linalg.eigh(m)

	#
	# The eigenvalues threshold: ignore the space of eigenvalues smaller
	# than that.
	#

	thresh = evals[-1]*ZERO_THRESH
	i = np.where(evals>thresh)[0][0]
	evals_red = evals[i:]
	U_red = U[:,i:]

	#
	# Calculate m^{1/2}: m = M_sq^\dagger \cdot M_sq
	#

	M_sq = diag(sqrt(evals_red))@conj(U_red.T)
#	M_sq = U_red@diag(sqrt(evals_red))@conj(U_red.T)

	return M_sq




#
# ------------------------  lazy_edge_truncation  ----------------------
#

def lazy_edge_truncation(T1, leg1, T2, leg2, m12, m21, \
	L2thresh=None, Dmax=None):

	r"""

	Perform a "lazy edge truncation", following the method of

	T. Begušić, J. Gray, and G. K.-L. Chan,
	“Fast and converged classical simulations of evidence for the
	utility of quantum computing before fault tolerance,”
	Science Advances, vol. 10, no. 3, p. eadk4321, 2024, arXiv:2308.05077

	It is also explained in more details in 5480/BPtruncation4.pdf
	
	There are two constants which are set in the function:
	
	1) DEFAULT_L2THRESH --- The default L2 truncation thereshold: we remove
	                        the singular value tail at the point where
	                        [(sum_{i>D_2} s^2_i)/s_max]^{1/2} <= L_2 thereshold
	
	2) POS_THRESHOLD    --- Automatically remove singular values smaller
	                        than the maximal singular value times 
	                        POS_THRESHOLD.
	                        
	                        
	The truncation bond is determined by the minima of the following 3 bonds:
	a) The positivity thereshold D_1
	b) The L2 thereshold D_2  (if given)
	c) D_max (if given)
	

	Input Parameters:
	------------------
	T1, leg1 --- The first tensor and the index of the leg connecting
	             to the other tensor.

	             Note: if T_1 shape is [d, D_0, D_1, D_2, ...]
	                   then leg_1=1 will truncate the D_1 leg

	T2, leg2 --- Same but for the second tensor

	m12, m21 --- The T1->T2 BP message and the T2->T1 message

	L2thresh --- A L_2 truncation threshold. This means that we normalize
	             the Vidal weights so that their L_2 norm is 1, and then
	             we truncate at the point where the *accumulated sum*
	             is < L2thresh.

	Dmax     --- Maximal bond dimension.

	If both Dmax and L2thresh are given, we truncate at the smallest
	effective bond (so that both requirements are fulfilled)


	"""

	#
	# See if any truncation is actually needed. This can happen if 
	# L2thresh is not given, while Dmax is given and is larger than 
	# the bond of edge that we want to truncate.
	#

	if Dmax is not None and L2thresh is None:
		if T1.shape[leg1+1] <= Dmax:
			return T1.copy(), T2.copy(), 0

	#
	# Default L_2 truncation threshold
	#
	DEFAULT_L2THRESH = 1e-12

	if L2thresh is None:
		L2thresh = DEFAULT_L2THRESH
	
	#
	# Positivity threshold: discard any singular values smaller than that
	#
	POS_THRESH = 1e-14  

	#
	# Calculate R_1, R_2, the squares of m12, m21
	#
	# NOTE: we use m12.T and m21.T because by definition for matrix
	#       mij[al,bet] that represents the i->j message, al is the
	#       ket leg and bet is the bra leg. Therefore, as
	#       R = lazy_sqrt_message(m) satisfies m = R^\dagger\cdot R,
	#       then if we call lazy_sqrt_message(m.T), we get R s.t.,
	#       m.T = R^\dagger\cdot R
	#       And so the bet index of R[al,bet] is identical to that of
	#       m.T[al,bet], which is identical to m[bet,al] --- So, we needed
	#       we multiply by the ket leg.
	#
	#

	R1 = lazy_sqrt_message(m12.T)
	R2 = lazy_sqrt_message(m21.T)

	#
	# Calculate M = R_1\cdot R_2^T and SVD it: M = UsV
	# We try to do it in a robust way: if numpy SVD does not converge
	# for some reason, then add to it a small random perturbation. 
	# Try this for at most 20 times before giving up.
	#

	M = R1@R2.T

	M1 = M
	converged = False
	dround = 1
	while not converged:

		converged = True

		try:
			U,s_orig,V = svd(M1, full_matrices=False)

		except:
			print(f"Warnning: LinAlgError occured in BPSU.lazy_edge_truncation while "\
				f"trying to perform svd. Adding a small "\
				f"random perturbation and trying again (round {dround}).")

			N = np.random.normal(size=M.shape)
			N = N/norm(N, ord=2)
			M1 = M + EPS*N*norm(M, ord=2)
			dround += 1
			converged = False

		if dround==20:
			print("\n\n")
			print("Error --- SVD  unable to converge in "\
				f"BPSU.lazy_edge_truncation  after 20 tries... quitting\n")
			exit(1)


	#
	# First, discard any singular values that are smaller than
	# then positivity threshold |M|*POS_THRESH
	#

	good_locations = np.where(s_orig>=s_orig[0]*POS_THRESH)[0]

	s = s_orig[:(good_locations[-1]+1)]

	#
	# Now truncate the weights according to L2thresh and Dmax (if given).
	#
	# 1. We start from D that is given by the positivity threshold
	# 2. We the calculate the L_2 truncation point, and see if it lowers
	#    D
	# 3. We then see if the resultant D is larger than D_max (if given), 
	#    in which case, we set it to D_max
	#    
	#

	D = s.shape[0]
	
	s2 = s**2

	
	#
	# Take care of the L_2 truncation:
	# --------------------------------
	#
	# Find Dthresh --- the place where the normalized accumulated sum of 
	# their squares is smaller than L2thresh**2. If Dthresh < D, then 
	# set D:=Dthresh
	#

	psums = np.cumsum(s2[::-1])
	psums = psums[::-1]

	# normalize it by the overall L2 norm
	psums = psums/psums[0]

	# Find the place where we need to truncate
	psums_loc = np.where(psums<L2thresh**2)[0]

	if psums_loc.shape[0]>0:
		Dthresh = psums_loc[0]

		if Dthresh<D:
			D = Dthresh

	#
	# Finally take care of Dmax. If Dmax<D ==> set D:=Dmax
	#
	if Dmax is not None:
		if Dmax<D:
			D = Dmax

	#
	# Now that we have found D, we can simply truncate
	#
	trunc_s = s[:D]


	#
	# truncate U, V to match trunc_s
	#
	U = U[:,:D]
	V = V[:D, :]

	#
	# Calculate the (normalized) L_2 truncation error
	#
	err = sqrt( sum(s2[D:])/sum(s2) )
	
	#
	# Now calculate P_1, P_2
	#
	inv_s_factor = diag(1/sqrt(trunc_s))

	P1 = R2.T@conj(V.T)@inv_s_factor
	P2 = R1.T@conj(U)@inv_s_factor

	#
	# Truncate T_1, T_2 by contracting P_1, P_2 to their common legs
	#

	newT1 = contract_leg(T1, P1, leg1)
	newT2 = contract_leg(T2, P2, leg2)


	return newT1, newT2, err


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
	      +---O---- (ket)  +----  (ket)
	     /    |            |
	m01 O     |      =     |
	     \    |            |
	      +---O---- (bra)  +----  (bra)
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

	EPS = 1e-8


	#
	# First calculate the square-root and its inverse for both BP
	# messages.
	#
	# Note: the qbp routine outputs messages m12[alpha,beta], where
	#       alpha is the ket and beta is the bra. But for being consistent
	#       with the derivation in Tinder et al, we use m12.T so that
	#       the first index is the bra and the second index is the ket.
	#

	m12_sq, m12inv_sq = sqrt_message2(m12.T)
	m21_sq, m21inv_sq = sqrt_message2(m21.T)
#	m12_sq, m12inv_sq = sqrt_message(m12.T)
#	m21_sq, m21inv_sq = sqrt_message(m21.T)


	#
	# Create the matrix in the middle
	#
	M = m12_sq@m21_sq.T

	M1 = M
	converged = False
	dround = 1
	while not converged:

		converged = True

		try:
			U,s,V = svd(M1, full_matrices=False)

		except:
			print(f"Warnning: LinAlgError occured in BPSU.edge_BP_gauging while "\
				f"trying to perform svd. Adding a small "\
				f"random perturbation and trying again (round {dround}).")

			N = np.random.normal(size=M.shape)
			N = N/norm(N, ord=2)
			M1 = M + EPS*N*norm(M, ord=2)
			dround += 1
			converged = False

		if dround==20:
			print("\n\n")
			print("Error --- unable to converge after 20 tries... quitting\n")
			exit(1)


	#
	# Calculate g1, g2 --- the gauge trans we apply to leg1,leg2 in T1, T2
	#
	g1 = m12inv_sq@U

	g2 = m21inv_sq@V.T

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

	Both output tensors & weights are locally normalized by the L_2 norm.

	Input Parameters:
	-----------------
	T_list --- The list of tensors that make up the TN

	e_dict --- The edges dictionary. The key is the edge name. The value
	           for an edge e=(i,j) is a 4-tuple (i, leg_i, j, leg_j)

	m_list --- The converged BP messages. For every neighboring vertices
	           i,j, m_list[i][j] is the converged i=>j BP  message.


	Output:
	---------

	gauged_T_list --- The updated T_list

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


		# Normalize the weights (according to L_2 norm)
		w_e = w_e/norm(w_e)

		gauged_T_list[vi] = new_Ti/norm(new_Ti)
		gauged_T_list[vj] = new_Tj/norm(new_Tj)

		w_dict[e] = w_e



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

		i1,leg1, i2,leg2 = e_dict[e]
		w = w_dict[e]
		D_w = w.shape[0]

		D_tensor = merged_T_list[i1].shape[leg1+1]

		sqw = sqrt(abs(w))
		sqM = zeros([D_tensor, D_w]).astype(merged_T_list[i1].dtype)
		sqM[:D_w,:D_w] = diag(sqw)

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

	Given a tensor T with the legs [d, D_0, D_1, ..., D_k], and a leg
	number leg=0,1,2,...k, permute the legs of T into the order

	[D_0, D_1, ..., D_k, d, D_leg] and then fuse the first Ds, as well
	as the d, D_leg legs

	We therefore get:

	[D_0, D_1, ..., D_k, d, D_leg] ==> [Drest, d, D_leg] ==> [Drest, d*D_leg]

	In addition, return the shape of the tensor before we coarse-grain
	the indices, so that we will know later how to undo this.

	Input Parameters:
	-----------------
	T   --- The ket tensor

	leg --- The index of the leg. Note leg=0 means the first *logical*
	        leg


	Output:
	-------

	M --- The permuted/fused tensor

	sh --- The shape of the intermediate tensor
	       [D_0, D_1, ..., D_k, d, D_leg]  (before fusing the legs)
	       This is useful if we want to un-fuse it.


	"""

	L = len(T.shape)

	#
	# Define the permutation that takes d, D_leg to the end
	#
	perm = list(range(L))
	perm.remove(0)
	perm.remove(leg+1)

	perm = perm + [0,leg+1]

	#
	# Permute, and keep a record of the tensor shape
	#
	M = T.transpose(perm)
	sh = list(M.shape)

	#
	# Fuse the rest of the legs, as well as d, D_leg
	#
	dD = T.shape[0]*T.shape[leg+1]
	Drest = T.size//dD

	M = M.reshape([Drest, dD])

	return M, sh


#
# ------------------   local_enviless_truncation   ---------------------
#

def local_enviless_truncation(T, leg, eps):
	"""

	Perform a simple L_2 truncation of a tensor, which does *not* use
	the environment.

	Specifically, we are given a PEPS tensor with legs [d, D0, D1, ...]
	and a leg index leg. We then turn it into a matrix [D_leg, D_rest],
	and perform SVD. We remove all singular values whose normalized
	*accumulated* L_2 weight is smaller than a fraction of eps.

	If truncation is performed, then also return a corresponding matrix
	to be multiplied by the adjoint tensor.


	This is much faster than regular truncation but
	also much less accurate. It is therefore recommended to use with
	very small eps on tensors where the bond dimension grew artificially.

	Input Parameters:
	------------------
	T   --- The PEPS tensor. Assumed to be of the form [d, D0, D1, ...]
	        where d is the physical leg

	leg --- The index of the leg to be truncated

	eps --- The accumulated, normalized L_2 error

	Output:
	-------
	newT --- The truncated T (along the leg).

	R    --- A matrix to be multiplied on the adjoint tensor using
	         the contract_leg function.

	Note: If not truncation is needed, then newT=T and R=None.

	"""

	sh = T.shape
	L = len(sh)
	D = sh[leg+1]  # The un-compressed, original bond dimension

	#
	# How many physical legs
	#
	leg_shift=1

	#
	# Define the permutation that takes D_leg to the beginning
	#
	perm = list(range(L))
	perm.remove(leg+leg_shift)

	perm = [leg+leg_shift] + perm

	#
	# Turn into a matrix T1, of the shape [D, D_rest]
	#
	T1 = T.transpose(perm)
	sh1 = list(T1.shape)
	T1 = T1.reshape([D, -1])

	#
	# Perform SVD
	#

	U, s, V = svd(T1, full_matrices=False)

	#
	# Calculate the normalized L_2 weights and look for a place to
	# truncate
	#
	try:
		s2 = s**2
	except FloatingPointError:
		print("got s=",s)
		exit(1)
	psums = np.cumsum(s2[::-1])
	psums = psums[::-1]

	# normalize it by the overall L2 norm
	psums = psums/psums[0]


	# Find the place where we need to truncate
	newD = None
	psums_loc = np.where(psums<eps**2)[0]
	if psums_loc.shape[0]>0:
		newD = psums_loc[0]


	if newD is None or newD==D:
		#
		# In such case no truncation is needed
		#
		return T, None

	#
	# If we got up to here, then we need to truncate.
	#

	newT = diag(s[:newD])@V[:newD, :]
	R = U[:, :newD]

	#
	# reshape and transpose newT back to its original legs order
	#

	sh1[0] = newD
	newT = newT.reshape(sh1)

	perm = list(range(1,L))
	perm.insert(leg+leg_shift, 0)

	newT = newT.transpose(perm)

	return newT, R



#
# ------------------   global_enviless_truncation   --------------------
#

def global_enviless_truncation(TN_params, eps, verts_list=None, \
	es_list=None):

	r"""

	Performs an environment-less truncation of a PEPS. That is, we take
	a single ket tensor and one leg of it, and use simple SVD to truncate
	it, *regardless* of its environment. This is OK if we allow ourself
	to truncate really small weights.

	The truncation is defined by an eps parameter. We truncate all
	singular values s_l0, s_{l0+1}, ... such that

	\sqrt(sum_{l>= l0) s^2_l) < eps\sqrt(\sum_l s^2_l)

	This is not an optimal truncation. But is very efficient, and can
	be used to reduce 'spurious' dimension before doing the more exact
	BP-based truncation. In such case, we should take eps<<1, say, 1e-9.


	Input Parameters:
	-----------------
	TN_params --- a dictionary with the parameters of the TN.
	              Specifically, we need T_list, e_list, e_dict

	eps       --- The normalized accumulated L_2 norm that we wish
	              to truncate.

	verts_list --- An optional list of vertices we wish to truncate. If
	               omitted then we consider all vertices

	es_list    --- An optional list of lists of edges for each tensors
	               we truncate. This should correspond to the tensors
	               in verts_list

	Output:
	-------

	newT_list --- The updated truncated tensors list



	"""

	#
	# Extract the TN params
	#
	T_list = TN_params['T_list']
	e_list = TN_params['e_list']
	e_dict = TN_params['e_dict']

	n = len(T_list)


	if verts_list is None:
		verts_list = list(range(n))

	newT_list = T_list.copy()

	#
	# Main loop: go over the tensors we wish to truncate, and for each
	#            tensor, go over the legs we wish to truncate.
	#
	for i,v in enumerate(verts_list):
		T = newT_list[v]

		#
		# Find the list of edges that we wish to truncate. If it is not
		# given, then truncate all legs
		#
		if es_list is None:
			es = e_list[v]
		else:
			es = es_list[i]

		newT = T

		#
		# Inner loop: go over the edges we truncate
		#
		for e in es:
			i,i_leg, j, j_leg = e_dict[e]
			if i==v:
				leg = i_leg
				v1 = j
				leg1 = j_leg
			else:
				leg = j_leg
				v1 = i
				leg1 = i_leg

			newT,R = local_enviless_truncation(newT, leg, eps)

			#
			# If truncation happened then also truncate the corresponding leg
			# of the tensor we are contracted with.
			#
			if R is not None:
				T1 = newT_list[v1]
				newT1 = contract_leg(T1, R, leg1)
				newT_list[v1] = newT1

		newT_list[v] = newT


	return newT_list






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

	EPS = 1e-8

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


	R12a = R12
	converged = False
	dround = 1
	while not converged:

		converged = True

		try:
			U,s,V = svd(R12a, full_matrices=False)

		except:
			print(f"Warnning: LinAlgError occured in BPSU.apply_2local_gate while "\
				f"trying to perform svd. Adding a small "\
				f"random perturbation and trying again (round {dround}).")

			N = np.random.normal(size=R12.shape)
			N = N/norm(N, ord=2)
			R12a = R12 + EPS*N*norm(R12, ord=2)
			dround += 1
			converged = False

		if dround==20:
			print("\n\n")
			print("Error --- unable to converge after 20 tries... quitting\n")
			exit(1)




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
			psums_loc = np.where(psums<eps**2)[0]
			if psums_loc.shape[0]>0:
				i = psums_loc[0]

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
# -------------------   direct_apply_2local_gate   ---------------------
#
def direct_apply_2local_gate(T1, T2, leg1, leg2, g):
	r"""

	Apply a 2-local gate g on two tensors T1, T2 that are connected by
	a common edge. The result is two new tensors newT1, newT2 with
	a larger bond dimension along the common edge.

	Note: no truncation is performed.

	Input Parameters:
	------------------
	T1, T2     --- The input tensors. Each tensor is of the form
	               [d, D0, D1, D2, ...] where d is the physical leg

	leg1, leg2 --- The location of the common edge in T1, T2. The index
								does not count the physical leg. So leg1=0 means that
								D0 is the common leg.

	g          --- The 2-local gate, given as [i1,j1; i2,j2] where
	               j1,j2 are the ket legs (contracted with the physical
	               legs of T1, T2) and i1,i2 are the bra legs.


  Output:
  -------

  newT1, newT2 --- The updated tensors


	"""

	TRUNC_THRESH = 1e-8

	# -----------------------------------------------------------------
	# 1. Find the physical bond dimensions of T1, T2 and the bond
	#    dimension of the edge that connects them
	# -----------------------------------------------------------------


	D = T1.shape[leg1+1]  # Original dimension of the common leg
	d1 = T1.shape[0]      # physical leg T1
	d2 = T2.shape[0]      # physical leg T2


	# -----------------------------------------------------------------
	# 2. Reshape T1, T2 into matrices, where the first leg is the fusion
	#    of all non-participating legs, and the second is (d,D), where
	#    d is the physical leg and D is the common leg
	# -----------------------------------------------------------------

	M1, T1_shape = gather_ext_legs(T1, leg1)
	M2, T2_shape = gather_ext_legs(T2, leg2)

	#
	# Seprate the d, D legs
	#
	M1 = M1.reshape([M1.shape[0], d1, D])  # M1 shape: [D1_rest, d1, D]
	M2 = M2.reshape([M2.shape[0], d1, D])  # M2 shape: [D2_rest, d2, D]

	#
	# Separate g into g1 and g2 using SVD. Truncate very small singular
	# values in order to save in enganglement cost
	#

	sh = g.shape
	gmat = g.reshape([sh[0]*sh[1], sh[2]*sh[3]])

	U,s,V = np.linalg.svd(gmat, full_matrices=False)
	thresh = s[0]*TRUNC_THRESH

	if s[-1]<thresh:
		i = np.where(s<thresh)[0][0]
	else:
		i=s.shape[0]

	s=s[:i]
	g1 = U[:,:i]@diag(sqrt(s))
	g2 = diag(sqrt(s))@V[:i,:]

	Dg = g1.shape[1]

	g1 = g1.reshape([sh[0],sh[1], Dg]) # g1 shape: [i1,j1,Dg]
	g2 = g2.reshape([Dg, sh[2],sh[3]]) # g2 shape: [Dg, i2,j2]

	#
	# Contract M1 + g1:  d1 <--> j1
	#

	M1 = tensordot(M1, g1, axes=([1],[1]))

	#
	# Now M1 shape is: [D1_rest, D, i1, Dg]
	# We move it to [D1_rest, i1, D*Dg]
	#

	M1 = M1.transpose([0,2,1,3])
	sh = M1.shape
	M1 = M1.reshape([sh[0],sh[1], -1])

	#
	# Contract M2 + g2:  d2 <--> j2
	#

	M2 = tensordot(M2, g2, axes=([1],[2]))

	#
	# Now M2 shape is: [D2_rest, D, Dg, i2]
	# We move it to [D2_rest, i2, D*Dg]
	#

	M2 = M2.transpose([0,3,1,2])
	sh = M2.shape
	M2 = M2.reshape([sh[0],sh[1], -1])

	DDg = M2.shape[2]

	T1_shape[-1] = DDg
	T2_shape[-1] = DDg

	newT1 = M1.reshape(T1_shape)
	newT2 = M2.reshape(T2_shape)


	# newT1 shape: other-legs, d1, D
	# newT2 shape: other-legs, d2, D


	# -------------------------------------------------
	# 11. Re-arrange the legs of newT1, newT2
	# -------------------------------------------------

	sh = newT1.shape
	L = len(newT1.shape)
	perm = [L-2] + list(range(leg1)) + [L-1] + list(range(leg1,L-2))
	newT1 = newT1.transpose(perm)

	sh = newT2.shape
	L = len(newT2.shape)
	perm = [L-2] + list(range(leg2)) + [L-1] + list(range(leg2,L-2))
	newT2 = newT2.transpose(perm)

	return newT1, newT2






#
# ---------------------  apply_2local_gate_notrunc   -------------------
#

def apply_2local_gate_notrunc(T_list, e_list,  e_dict, g, e):

	r"""

	Given a TN (not in the Vidal gauge), apply a 2-body gate g on the
	tensors of a given edge *without* truncation. Consequently, the
	bond dimension between the two sites increases.


	Input Parameters:
	------------------

	T_list, e_list, e_dict --- The description of the TN

	g    --- The 2-local gate, given as [i1,j1; i2,j2] where j1,j2 are
	         the ket legs and i1,i2 are the bra legs.

  e    --- The label of the edge on which the gate is acting
           note that e=(i,j) where i<j


  Output:
  --------

  T_list --- Tensors of the updated TN.


	"""

	TRUNC_THRESH = 1e-8


	#
	# Locate the vertices of the edge e=(i1,i2) and their tensors T1, T2
	#

	i1,leg1, i2,leg2 = e_dict[e]

	T1 = T_list[i1]
	T2 = T_list[i2]

	D = T1.shape[leg1+1]  # Original dimension of the common leg
	d1 = T1.shape[0]  # physical leg T1
	d2 = T2.shape[0]  # physical leg T2


	# -----------------------------------------------------------------
	# 2. Reshape T1, T2 into matrices, where the first leg is the fusion
	#    of all non-participating legs, and the second is (d,D), where
	#    d is the physical leg and D is the common leg
	# -----------------------------------------------------------------

	M1, T1_shape = gather_ext_legs(T1, leg1)
	M2, T2_shape = gather_ext_legs(T2, leg2)

	#
	# Seprate the d, D legs
	#
	M1 = M1.reshape([M1.shape[0], d1, D])  # M1 shape: [D1_rest, d1, D]
	M2 = M2.reshape([M2.shape[0], d1, D])  # M2 shape: [D2_rest, d2, D]

	#
	# Separate g into g1 and g2 using SVD. Truncate very small singular
	# values in order to save in enganglement cost
	#

	sh = g.shape
	gmat = g.reshape([sh[0]*sh[1], sh[2]*sh[3]])

	U,s,V = np.linalg.svd(gmat, full_matrices=False)
	thresh = s[0]*TRUNC_THRESH

	if s[-1]<thresh:
		i = np.where(s<thresh)[0][0]
	else:
		i=s.shape[0]

	s=s[:i]
	g1 = U[:,:i]@diag(sqrt(s))
	g2 = diag(sqrt(s))@V[:i,:]

	Dg = g1.shape[1]

	g1 = g1.reshape([sh[0],sh[1], Dg]) # g1 shape: [i1,j1,Dg]
	g2 = g2.reshape([Dg, sh[2],sh[3]]) # g2 shape: [Dg, i2,j2]

	#
	# Contract M1 + g1:  d1 <--> j1
	#

	M1 = tensordot(M1, g1, axes=([1],[1]))

	#
	# Now M1 shape is: [D1_rest, D, i1, Dg]
	# We move it to [D1_rest, i1, D*Dg]
	#

	M1 = M1.transpose([0,2,1,3])
	sh = M1.shape
	M1 = M1.reshape([sh[0],sh[1], -1])

	#
	# Contract M2 + g2:  d2 <--> j2
	#

	M2 = tensordot(M2, g2, axes=([1],[2]))

	#
	# Now M2 shape is: [D2_rest, D, Dg, i2]
	# We move it to [D2_rest, i2, D*Dg]
	#

	M2 = M2.transpose([0,3,1,2])
	sh = M2.shape
	M2 = M2.reshape([sh[0],sh[1], -1])

	DDg = M2.shape[2]

	T1_shape[-1] = DDg
	T2_shape[-1] = DDg

	T1 = M1.reshape(T1_shape)
	T2 = M2.reshape(T2_shape)


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


	# -----------------------------------------------------------
	# 13. Update T1, T2, w in the T_list, w_dict list/dictionary
	# -----------------------------------------------------------

	T_list[i1] = T1
	T_list[i2] = T2


	return T_list



#
# ---------------------- apply_gate_to_PEPS  -----------------------
#

def apply_gate_to_PEPS(T_list, e_list,  e_dict, g, i=None, e=None):
	r"""

	Applies a gate to a PEPS TN *without* any truncation. The gate can
	be either 1-local or 2-local. In the first case, a site location i
	must be given, whereas in the second case and edge index e is given.

	After the gate is applied, and updated T_list is returned.

	Input Parameters:
	-----------------
	T_list, e_list, e_dict --- TN parameters

	g --- The gate to be applied.

	i --- A site location (when g is 1-local)

	e --- An edge index (when g is 2-local)

	Output:
	-------
	T_list --- Updated T_list


	"""

	#
	# First, make sure that either i or e is given.
	#
	if (i is None and e is None) or (i is not None and e is not None):
		print("Error in apply_gate_to_PEPS: either i or e must be given "\
			"(they cannot be both None and they cannot be both not None)\n")
		exit(1)

	if i is not None:
		mode = '1-local'
	else:
		mode = '2-local'


	if mode == '1-local':
		#
		# --------------------  1-local gate  ----------------------------
		#
		T = T_list[i]

		T = tensordot(g, T, axes=([1],[0]))

		T_list[i] = T

	if mode == '2-local':

		#
		# --------------------  2-local gate  ----------------------------
		#

		#
		# Locate the vertices of the edge e=(i1,i2) and their tensors T1, T2
		#

		i1,leg1, i2,leg2 = e_dict[e]

		T1 = T_list[i1]
		T2 = T_list[i2]

		newT1, newT2 = direct_apply_2local_gate(T1, T2, leg1, leg2, g)

		T_list[i1] = newT1
		T_list[i2] = newT2

	return T_list




#
# ---------------------- apply_gate_to_PEPO  -----------------------
#

def apply_gate_to_PEPO(T_list, e_list,  e_dict, gket=None, gbra=None, \
	i=None, e=None):
		
	r"""
	
	Applies a 1-local or 2-local Kraus gate to a PEPO: 
	
	Given two gates gket, gbra and a PEPO rho, 
	
	                rho ===>  gket \cdot \rho \cdot gbra.T
	                
	Recall that a PEPO has the following leg structure:
	
	                 [d_bra, d_ket, DP_0, DP_1, ...]
	                 
	Therefore gket is contracted with d_bra and gbra is contracted with 
	d_ket
	                
	Note: 
	-----
	1) We do *not* complex-conjugate gbra
	2) gket and gbra are optional, they can be both given or just one
	   of them.
	   
	Input Parameters:
	-----------------
	
	T_list, e_list, e_dict --- TN params
	
	gket, gbra --- The ket and the bra gates.
	               Note if g_{ij} is the matrix of a gate, the leg j
	               is the leg that is contracted to the PEPO.
	               
	i, e --- Location of the gate. Only one of them can be given.
	         If it is a single-qubit gate, then i is given.
	         If it is a two-qubits gate, e is given.
	         
	Output:
	-------
	T_list --- The updated tensors list of the PEPO
	
	"""
	

	#
	# Make sure that either i or e is given.
	#
	if (i is None and e is None) or (i is not None and e is not None):
		print("Error in apply_gate_to_PEPO: either i or e must be given "\
			"(they cannot be both None and they cannot be both not None)\n")
		exit(1)

	if i is not None:
		mode = '1-local'
	else:
		mode = '2-local'


	if mode == '1-local':
		T = T_list[i]

		if gket is not None:
			#
			# Contract gket to the d-bra leg of T
			#
			T = tensordot(gket, T, axes=([1],[0]))

		if gbra is not None:
			#
			# Contract gbra to the d-ket leg of T
			#
			T = tensordot(gbra, T, axes=([1],[1]))

			# permute the 0 <-> 1 legs
			perm = list(range(len(T.shape)))
			perm[0] = 1
			perm[1] = 0
			T = T.transpose(perm)

		T_list[i] = T


	if mode == '2-local':
		#
		# Locate the vertices of the edge e=(i1,i2) and their tensors T1, T2
		#

		i1,leg1, i2,leg2 = e_dict[e]

		T1 = T_list[i1]
		T2 = T_list[i2]

		if gket is not None:
			#
			# We invoke the direct_apply_2local_gate as if T1, T2 are PEPS
			# tensors. To account for the extra bra leg we pass
			# leg1 -> leg1+1, leg2 -> leg2+1
			#

			newT1, newT2 = direct_apply_2local_gate(T1, T2, leg1+1, leg2+1, gket)

		else:
			newT1 = T1
			newT2 = T2

		if gbra is not None:
			#
			# In such case we first permute the physical bra and ket legs
			# and then use the same steps as in the ket case.
			#

			perm1 = list(range(len(newT1.shape)))
			perm1[0]=1
			perm1[1]=0

			perm2 = list(range(len(newT2.shape)))
			perm2[0]=1
			perm2[1]=0

			newT1 = newT1.transpose(perm1)
			newT2 = newT2.transpose(perm2)

			newT1, newT2 = direct_apply_2local_gate(newT1, newT2, \
				leg1+1, leg2+1, gbra)

			#
			# Return the ket/bra legs to their original position
			#

			newT1 = newT1.transpose(perm1)
			newT2 = newT2.transpose(perm2)



		T_list[i1] = newT1
		T_list[i2] = newT2

	return T_list






#
# ---------------------  apply_PEPO_to_PEPS   -------------------
#

def apply_PEPO_to_PEPS(T_list, T_PEPO_list):

	"""
	
	Applies an operator A, described by a PEPO T_PEPO_list to a ket state
	psi, described by T_list. The result is a new ket state 
	
	                      |psi'> = A|psi>

	The PEPO tensor has legs [d_bra, d_ket, DP_0, DP_1, ...]

	When acting on a PEPS tensor with legs [d, D_0, D_1, ...]
	we contract d_ket with d
	
	Input Parameters:
	-----------------
	
	T_list      --- The tensors list of the ket state |psi>

	T_PEPO_list --- The tensors list of the PEPO operator A
	
	
	Output:
	-------
	The resultant T_list of |psi'> = A|psi>


	"""

	newT_list = []

	for i,T in enumerate(T_list):

		TP = T_PEPO_list[i]

		sh_T = T.shape
		sh_TP = TP.shape

		k = len(sh_T)-1  # number of virtual legs

		newT = tensordot(TP, T, axes=([1],[0]))

		#
		#                0    1    2            k      k+1
		# newT has legs [d, DP_0, DP_1, ..., DP_{k-1}, D_0, ..., D_{k-1}]
		#
		# we permute it to [d; D_0,DP_0; D_1,DP_1; ...]
		#

		perm = [0]
		sh_fused = [sh_TP[0]]
		for i in range(k):
			perm     = perm + [k+1+i,i+1]
			sh_fused = sh_fused + [ sh_T[i+1]*sh_TP[i+2] ]

		newT = newT.transpose(perm)
		newT = newT.reshape(sh_fused)

		newT_list.append(newT)

	return newT_list







#
# --------------------------  truncate_weights  -----------------------
#
def truncate_weights(w_dict, Dmax=None, L2thresh=None):

	"""

	Given a dictionary of Vidal gauge weights (one at the middle of
	every edge), truncate them according to Dmax or L_2 threshold:

	Looking at the singular values of an edge, we truncate at bond which
	is the min of:
	1) Dmax
	2) The point Dthresh aftter which the accumulated mass of the
	   *square* of the singular values is smaller than L2thresh**2

	Input Parameters:
	------------------

	w_dict --- A dictionary holding the Vidal gauge singualr values of
	           each edge. Here the key is the edge index, and the value
	           is an array of singular values

	Dmax   --- Maximal bond dimension (ignore if None)

	L2thresh --- The L_2 threshold: if given then truncate at the point
	             where the accumulated mass of the squares is < L2thresh**2

	Output:
	-------

	A dictionary with the truncated weights.


	"""

	log = False

	if Dmax is None and L2thresh is None:
		return w_dict.copy(), 0

	trunc_w_dict = {}

	truncation_error=0

	for e in w_dict.keys():

		w = w_dict[e]

		#
		# Find the D where we need to truncate the weights. By default,
		# we start with the maximal value of D. If L2thresh is given
		# and/or Dmax is given --- we take the minimal value of D we can
		# from either of them.
		#

		D = w.shape[0]
		w2 = w**2

		Dthresh=None
		if L2thresh is not None:
			#
			# If L2thresh is given, then we truncate the weights where
			# the normalized accumulated sum of their squares is smaller
			# than L2thresh**2
			#

			psums = np.cumsum(w2[::-1])
			psums = psums[::-1]

			# normalize it by the overall L2 norm
			psums = psums/psums[0]

			# Find the place where we need to truncate
			psums_loc = np.where(psums<L2thresh**2)[0]

			if psums_loc.shape[0]>0:
				Dthresh = psums_loc[0]

				if Dthresh<D:
					D = Dthresh

		if Dmax is not None:
			if Dmax<D:
				D = Dmax

		trunc_w_dict[e] = w[:D]

		if log:
			print(f"Truncating e={e}: w={w}  trunc_w={trunc_w_dict[e]}")

		#
		# Calculate the truncation error
		#
		if D<w.shape[0]:
			truncation_error += sqrt( sum(w2[D:])/sum(w2) )

	return trunc_w_dict, truncation_error



#
# ------------------------   BP_compress   -----------------------------
#

def BP_compress(TN_params, m_list, Dmax=None, L2thresh=None, normalize=True):

	"""

	Given a TN and a set of converged BP messages, compress the TN using
	BPSU framework:

	1) Use the BP fix-point messages to move to the Vidal gauge

	2) Truncate the Vidal weights

	Note that this truncation is optimal when the underlying graph is
	a tree, generalizing the MPS case.

	The truncation is done using to min of two possible criterias:
	Dmax (maximal bond dimension) or L2thresh --- a limit on the accumulated
	mass of the square of the weights. See the doc in the truncate_weights
	function for more details.

	Input Parameters:
	------------------
	TN_params --- A dictionary holding the TN data

	m_list    --- A list of the converged BP messages

	Dmax      --- A possible maximal bond-dim

	L2thresh  --- A possible L_2 thereshold for truncation (truncate
	              at the point where the normalized accumulated sum of the
	              *square* of the weight

	Output:
	--------
	T_list    --- The list of updated truncated tensors
	trunc_err --- The truncation error


	"""

	log = True


	#
	# Extract the TN params and the BP params
	#
	T_list = TN_params['T_list']
	e_dict = TN_params['e_dict']

	#
	# Move to a Vidal gauge
	#

	gT_list, w_dict = BP_gauging(T_list, e_dict, m_list)


	#
	# Truncate the Vidal weights
	#
	trunc_w_dict, trunc_err = truncate_weights(w_dict, Dmax, L2thresh)

	#
	# Merge them back into the TN
	#
	T_list = merge_SU_weights(gT_list, e_dict, trunc_w_dict)

	#
	# Optionally, normalize the TN (using L_2 norm)
	#

	if normalize:
		for i, T in enumerate(T_list):
			T_list[i] = T/norm(T)

	return T_list, trunc_err


#
# ----------------------  PEPO_to_PEPS  --------------------------------
#

def PEPO_to_PEPS(TP_list):
	"""

	Turn a PEPO tensor list into a PEPS tensor list by fusing the ket and
	bra physical legs into one leg

	"""
	TP_ket_list = []
	for i, TP in enumerate(TP_list):
		sh = list(TP.shape)
		sh2 = [sh[0]*sh[1]] + sh[2:]

		TP_ket = TP.reshape(sh2)
		TP_ket_list.append(TP_ket)

	return TP_ket_list


#
# -------------------------  PEPS_to_PEPO  -----------------------------
#

def PEPS_to_PEPO(TP_ket_list):
	"""

	Turn a PEPS tensor list into a PEPO tensor list by un-fusing the
	'physical' into a pair of (ket,bra) legs.

	"""

	TP_list = []
	for i, TP_ket in enumerate(TP_ket_list):
		sh = list(TP_ket.shape)
		D2 = sh[0]
		D = int(sqrt(D2)+1e-7)
		sh2 = [D,D] + sh[1:]

		TP = TP_ket.reshape(sh2)
		TP_list.append(TP)

	return TP_list



#
# ~~~~~~~~~~~~~~~~~~~~~~~~~  fuse_ket_bra_tensors  ~~~~~~~~~~~~~~~~~~~~~
#

def fuse_ket_bra_tensors(Ta, Tb, conjB=False):

		r"""
		
		Take two PEPS tensors with the same dimensions and contract them
		along the physical leg, producing a double-layer PEPS tensor with
		legs bonds that are products of the individual bonds.
		
		Input Parameters:
		------------------
		Ta, Tb --- The two tensors
		
		conjB  --- Whether or not to complex-conjugate Tb
		
		
		Output:
		-------
		The fused double-layer tensor

		"""

		n = len(Ta.shape)

		if conjB:
			T2 = tensordot(Ta, conj(Tb), axes=([0],[0]))
		else:
			T2 = tensordot(Ta, Tb, axes=([0],[0]))

		#
		# Permute the legs:
		# [D1, D2, ..., D1^*, D2^*, ...] ==> [D1, D1^*, D2, D2^*, ...]
		#
		perm = []
		for i in range(n-1):
			perm = perm + [i, i+n-1]

		T2 = T2.transpose(perm)

		#
		# Fuse the ket-bra pairs: [D1, D1^*, D2, D2^*, ...] ==> [D1^2, D2^2, ...]
		#

		dims = [Ta.shape[i]*Tb.shape[i] for i in range(1,n)]

		T2 = T2.reshape(dims)

		return T2


#
# ~~~~~~~~~~~~~~~~~~~~~~~~~  fuse_ket_bra_PEPS  ~~~~~~~~~~~~~~~~~~~~~
#

def fuse_ket_bra_PEPS(T_list_a, T_list_b, conjB=False):
	
	r"""
	
	Take two PEPS psi_a, psi_b, represented by the tensor lists T_list_a, 
	T_list_b, and creates a double-layer PEPS by contracting them along 
	their physical leg.
	
	Input Parameters:
	-----------------
	T_list_a, T_list_b --- The tensor lists of the psi_a, psi_b PEPS
	
	conjB --- Whether or not to conjugate the psi_b tensor
	
	Output:
	-------
	T2_list --- The resultant double-layer PEPS
	
	
	"""
	
	n = len(T_list_a)
	
	T2_list = []
	
	for i in range(n):
		T2 = fuse_ket_bra_tensors(T_list_a[i], T_list_b[i], conjB)
		T2_list.append(T2)

	return T2_list


#
# ----------------------  BP_compress_PEPO  ----------------------------
#

def BP_compress_PEPO(TP_list, e_list, e_dict, Dmax=None, L2thresh=1e-9,
	normalize=True, BP_max_iter=None, BP_delta=None, BP_damping=None):

	r"""

	Uses BP + Vidal gauge to compress a PEPO. When the underlying PEPO
	graph is a tree, the compression is optimal.

	The algorithm essentially fuses the two physical PEPO legs into one
	leg, thereby turning it into a PEPS. Then it uses BP_compress on it.

	Input Parameters:
	-----------------
	TP_list --- List of PEPO tensors. Each tensor is of the form
	            T[d,d, D_0, D_1, ...], where d are the physical legs

	e_list, e_dict --- list + dictionary holding the TN structure

	Dmax     --- The maximal bond dim (

	L2thresh --- A L2 threshold for the compression (the normalized
	             mass of squared singular values we are allowed to throw)

	BP_max_iter, BP_delta, BP_damping --- optional BP parameters



	"""

	log = False

	if log:
		print("\n\n")
		print(f"Entering BP_compress_PEPO with L2thresh={L2thresh}...\n")

	#
	# First, we turn the PEPO tensors to ket tensors, thereby getting
	# a PEPS
	#
	TP_ket_list =  PEPO_to_PEPS(TP_list)

	#
	# Now we run BP on the PEPS
	#
	if BP_max_iter is None:
		BP_max_iter = len(TP_ket_list) + 1

	if BP_delta is None:
		BP_delta = 1e-9

	if BP_damping is None:
		BP_damping = 0

	if log:
		print(f"Running BP...\n")

	m_list, err, iter_no = qbp(TP_ket_list, e_list, e_dict, initial_m='U', \
			max_iter=BP_max_iter, delta=BP_delta, damping=BP_damping)

	if log:
		print(f"BP ended after {iter_no} iterations with BP-err={err:.6g}\n")

	#
	# Compress the PEPS
	#
	TN_params = {}
	TN_params['T_list'] = TP_ket_list
	TN_params['e_list'] = e_list
	TN_params['e_dict'] = e_dict

	TP_ket_list1, trunc_err = BP_compress(TN_params, m_list, Dmax=Dmax,
		L2thresh=L2thresh, normalize=normalize)

	if log:
		print(f"BP compressed the PEPO with err={trunc_err:.6g}\n")

		d = peps_dist(TP_ket_list, TP_ket_list1, e_list, e_dict)
		print("**** COMPRESSION Distance: ", d, "\n")
	#
	# Turn it back into a PEPO
	#
	TP_list = PEPS_to_PEPO(TP_ket_list1)

	return TP_list, trunc_err



#
# ----------------------  lazy_PEPS_compression  ----------------------------
#

def lazy_PEPS_compression(T_list, e_list, e_dict, Dmax=None, L2thresh=1e-9,
	normalize_tensors=True, BP_max_iter=None, BP_delta=None, BP_damping=None):

	r"""

	Uses BP to perform a "lazy PEPS compression" of the entire TN. This
	is explained in:

	T. Begušić, J. Gray, and G. K.-L. Chan,
	“Fast and converged classical simulations of evidence for the
	utility of quantum computing before fault tolerance,”
	Science Advances, vol. 10, no. 3, p. eadk4321, 2024, arXiv:2308.05077

	It is also explained in more details in 5480/BPtruncation4.pdf

	Essentially, we run the BP, and the on each edge we use the two
	opposite converged BP messages to find two "projectors" P_i, P_j
	which truncate the bond. The actual truncation is done in the
	lazy_edge_truncation function.

	Note: The compression is done *in-place* (to save space) --- so the
	      input T_list is updated.


	Input Parameters:
	-----------------
	T_list --- List of PEPS tensors. The update (compression) is done
	           *in-place*

	e_list, e_dict --- list + dictionary holding the TN structure

	Dmax     --- The maximal bond dim

	L2thresh --- A L2 threshold for the compression (the normalized
	             mass of squared singular values we are allowed to throw)

	normalize_tensors --- Whether to normalize the truncated tensors after
	              truncation

	BP_max_iter, BP_delta, BP_damping --- optional BP parameters


	Output:
	-------

	T_list  --- The compressed tensors list (this is actually the same
	            list as the input list, since the compression is done
	            in-place.

	err     --- Total normalized L_2 compression error
	
	f_sim   --- Total simulation fidelity as defined in appendix A.2 in
	            arXiv:2503.20870v2



	"""

	elog = True

	if elog:
		print("\n\n")
		print(f"Entering lazy_PEPS_compression with L2thresh={L2thresh} "\
			f"and Dmax={Dmax}...\n")


	if Dmax is None and L2thresh is None:
		return T_list, 0


	#
	# Run BP on the PEPS and obtain the converged messages
	#
	if BP_max_iter is None:
		BP_max_iter = len(T_list) + 1

	if BP_delta is None:
		BP_delta = 1e-9

	if BP_damping is None:
		BP_damping = 0

	if elog:
		print(f"lazy_PEPS_compression: Running BP...\n")

	m_list, err, iter_no = qbp(T_list, e_list, e_dict, initial_m='U', \
			max_iter=BP_max_iter, delta=BP_delta, damping=BP_damping)

	if elog:
		print(f"lazy_PEPS_compression: BP ended after {iter_no} "\
			f"iterations with BP-err={err:.6g}\n")

	total_err=0  # Sum of the L_2 norms of the truncations in all sites
	
	f_sim = 1    # Accumulated fidelity. If err is the L_2 truncation
	             # *norm* (i.e., (\sum_{i>D} s_i^2 )^0.5 ), 
	             # then f := 1-err^2

	#
	# Main loop: go over all TN edges, and truncate each edge using the
	#            two BP messages on it
	#
	for e in e_dict.keys():

		i, i_leg, j, j_leg = e_dict[e]

		Ti = T_list[i]
		Tj = T_list[j]


		m_ij = m_list[i][j]
		m_ji = m_list[j][i]

		# Truncate the edge, defining two new tensors at sites i,j
		newTi, newTj, err = lazy_edge_truncation(Ti, i_leg, Tj, j_leg,\
			m_ij, m_ji, L2thresh, Dmax)

		total_err += err
		f_sim *= 1 - err**2

		if normalize_tensors:
			newTi = newTi/norm(newTi)
			newTj = newTj/norm(newTj)


		T_list[i] = newTi
		T_list[j] = newTj

	if elog:
		print(f"lazy_PEPS_compression: total L_2 error: {total_err:.6g}, "\
			f"total_simulation_fidelity={f_sim:.6g}")

	return T_list, total_err, f_sim


#
# ----------------------  lazy_PEPO_compression  ----------------------------
#

def lazy_PEPO_compression(TP_list, e_list, e_dict, Dmax=None, L2thresh=1e-9,
	normalize=True, BP_max_iter=None, BP_delta=None, BP_damping=None):

	TP_ket_list = PEPO_to_PEPS(TP_list)

	TP_ket_list, err = lazy_PEPS_compression(TP_ket_list, e_list, e_dict,\
		Dmax=Dmax, L2thresh=L2thresh, normalize=normalize, \
		BP_max_iter=BP_max_iter, BP_delta=BP_delta, BP_damping=BP_damping)

	TP_list = PEPS_to_PEPO(TP_ket_list)

	return TP_list, err



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


