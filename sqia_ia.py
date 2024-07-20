from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT
from qiskit import Aer, IBMQ, execute, transpile
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.transpiler.passes import RemoveBarriers

from math import pi, log, ceil, sqrt, floor
import pandas as pd
from random import randrange

import json
import os, sys

import dataframe_image as dfi
import numpy as np
import matplotlib.pyplot as plt

from utils import *

Num_runs = None
lamb = None

in_q_ct = None
val_q_ct = None
wt_q_ct = None
flag_ct = None

in_qubits = None
val_qubits = None
wt_qubits = None
val_flag = None
wt_flag = None
out_qubit = None
in_cbits = None

Num_I = None
Values = None
Weights = None

def init_gs_params(N_runs, lb):
    global Num_runs
    global lamb
    Num_runs = N_runs
    lamb = lb
    
    
def init_prob_det(N_I, Val, Wgt):
    global Num_I, Values, Weights
    
    Num_I = N_I
    Values = Val
    Weights = Wgt
    
def init_gs_qubits(in_q, val_q, wt_q, flag_c):
    # Declaring qubits 
    global in_qubits, val_qubits, wt_qubits, val_flag, wt_flag, out_qubit, in_cbits
    global in_q_ct, val_q_ct, wt_q_ct, flag_ct
    
    in_q_ct = in_q
    val_q_ct = val_q
    wt_q_ct = wt_q
    flag_ct = flag_c
    
    in_qubits = QuantumRegister(in_q_ct, name='index') 
    val_qubits = QuantumRegister(val_q_ct, name='val')
    wt_qubits = QuantumRegister(wt_q_ct, name='wt')

    val_flag = QuantumRegister(flag_ct, name='val_flag')
    wt_flag = QuantumRegister(flag_ct, name='wt_flag')

    out_qubit = QuantumRegister(flag_ct, name='out')

    # Declaring CBits
    in_cbits = ClassicalRegister(in_q_ct, name='in_cb')

def oracle(Max_W, Test_Val):
    
    
    qc = QuantumCircuit(in_qubits, val_qubits, wt_qubits, val_flag, wt_flag)
    
    # Lets load the values of each combination
    for i in range(in_q_ct):
        cu_val = Values[i]
        theta = ((2 * pi) / (2**val_q_ct)) * cu_val

        for j in range(val_q_ct):
            angle = 2**(val_q_ct - (j + 1)) * theta
            qc.cp(angle, in_qubits[i], val_qubits[j])

        qc.barrier()
        
    # Substract the Test_Val to identify all the elements that have value >= Test Val
    cu_val = -1 * Test_Val
    theta = ((2 * pi) / (2**val_q_ct)) * cu_val

    for j in range(val_q_ct):
        angle = 2**(val_q_ct - (j + 1)) * theta
        qc.p(angle, val_qubits[j])

    qc.barrier()

    qc = qc.compose(QFT(num_qubits=val_q_ct, approximation_degree=0, do_swaps=False, inverse=True, insert_barriers=False, name='qft'), val_qubits)

    # Flag the Value qualification
    # The values that are positive have sign bit 0 are >= Test Val
    
    qc.x(val_qubits[val_q_ct-1])
    qc.cx(val_qubits[val_q_ct-1], val_flag)
    qc.x(val_qubits[val_q_ct-1])
    
    
    qc.barrier()
    # Value Update ends

    # Lets load the weight of each combination
    for i in range(in_q_ct):
        cu_wt = Weights[i]
        theta = ((2 * pi) / (2**wt_q_ct)) * (cu_wt)

        for j in range(wt_q_ct):
            angle = 2**(wt_q_ct - (j + 1)) * theta
            qc.cp(angle, in_qubits[i], wt_qubits[j])

        qc.barrier()

    # Substract the Max Weight
    cu_wt = -1 * Max_W
    theta = ((2 * pi) / (2**wt_q_ct)) * (cu_wt)

    for j in range(wt_q_ct):
        angle = 2**(wt_q_ct - (j + 1)) * theta
        qc.p(angle, wt_qubits[j])

    qc.barrier()
    qc = qc.compose(QFT(num_qubits=wt_q_ct, approximation_degree=0, do_swaps=False, inverse=True, insert_barriers=False, name='qft'), wt_qubits)
    qc.barrier()
    # Weight Update ends
    
    # Flag Weight candidates
    qc.cx(wt_qubits[wt_q_ct-1], wt_flag)
    
    qc.barrier()
    qc.x(wt_qubits[0:wt_q_ct-1])
    qc.mct(wt_qubits[0:wt_q_ct-1],wt_flag)
    qc.x(wt_qubits[0:wt_q_ct-1])
    qc.barrier()
    
    qc = RemoveBarriers()(qc)
    
    U_o = qc.to_gate(label='Oracle')
    
    return U_o



def diffuser(n):
    ''' 
This is an implementation of the grover diffuser operator
Inputs:
n: Number of bits used to represent the tuples ids
'''
    
    qc = QuantumCircuit(n)
    
    for q in range(n):
        qc.h(q)
        
    for q in range(n):
        qc.x(q)
    
    qc.h(n-1)
    qc.mct(list(range(n-1)),n-1)
    qc.h(n-1)
        
    for q in range(n):
        qc.x(q)
    
    for q in range(n):
        qc.h(q)
    
    U_s = qc.to_gate(label='Diffuser')
    return U_s


# To check if the predicted indexes is qualifying
def validate_prediction(pred_ind, Max_W, Test_Val):
    index_set = []
    
    # Calculating Weight
    cu_wt = 0
    cu_in = pred_ind[::-1]
    for j in range(Num_I):
        if cu_in[j] == '1':
            cu_wt += Weights[j]
    
    # Calculating Value
    cu_val = 0
    cu_in = pred_ind[::-1]
    for j in range(Num_I):
        if cu_in[j] == '1':
            cu_val += Values[j]
            index_set.append(j)
            
    qualify = False
    
    if cu_wt <= Max_W and cu_val >= Test_Val:
        qualify = True
        
    return qualify, index_set, cu_val, cu_wt


# Given a test Max Value -- performing Grover's search 
def grover_search(Test_Val, num_iter, backend, shot, Max_W):
    
    qc = QuantumCircuit(in_qubits, val_qubits, wt_qubits, val_flag, wt_flag, out_qubit, in_cbits)

    # Init qubits
    # Initializing all qubits in equal superposition
    qc.h(in_qubits) # Indexes hold all the indexes 
    qc.h(val_qubits)
    qc.h(wt_qubits)

    # Init the Flag qubit in |-> state
    qc.x(out_qubit)
    qc.h(out_qubit)

    # Implement Grover Search Here

    U_o = oracle(Max_W,Test_Val)
    U_o_inv = U_o.inverse()
    U_o_inv.name = 'Oracle_inv'

    q_list = list()
    q_list.extend(in_qubits)
    q_list.extend(val_qubits)
    q_list.extend(wt_qubits)
    q_list.extend(val_flag)
    q_list.extend(wt_flag)

    for i in range(num_iter):
        # Applying Oracle
#         qc.append(U_o, q_list)
        qc = qc.compose(U_o, q_list)

        # Indicating final qualifying tuples
        qc.ccx(val_flag, wt_flag, out_qubit)

        # Undoing Oracle
#         qc.append(U_o_inv, q_list)
        qc = qc.compose(U_o_inv, q_list)

        # Adding Diffuser
        qc = qc.compose(diffuser(Num_I), in_qubits)

    qc.measure(in_qubits, in_cbits)

    t_qc = transpile(qc, backend=backend)
    job = execute(t_qc, backend, shots=shot)
    counts = job.result().get_counts()
    time_taken = job.result().time_taken

    pred_ind = max(counts, key=counts.get)
    
    return pred_ind, t_qc, time_taken



# Unknown Number of execution 
def find_index_set(Test_Val, Q_max, L, curr_best, backend, shot, Max_W):
    global lamb
    global Num_runs
    
    from utils import backend
    
    r = 0
    qualify = False
    pred_ind = ''
    index_set = [] 
    pred_val = 0
    pred_wt = 0
    qc = None
    
    total_iters = 0
    total_time = 0
    
    print('Num_runs', Num_runs)
    print('lamb', lamb)

    optimal_bs = {'pred_val': -100}
    optimal_bs['pred_ind'] = None
    optimal_bs['index_set'] = None
    optimal_bs['pred_wt'] = None
    optimal_bs['qc'] = None
    optimal_bs['qualify'] = False
    
    while r < Num_runs:
        m = lamb
        Q_sum = 0

        # Sampling a non-negative integer j less than m uniformly at random
        j = randrange(1, ceil(m))
        while Q_sum + j <= Q_max and j > 0:
#             print('Current j', j)
            pred_ind, qc, time_taken = grover_search(Test_Val, j, backend, shot, Max_W)
            
            # Iterations Used
            print('Iteration Used: ', Q_sum + j)
            total_iters += j
            
            # Time Taken
            total_time += time_taken

            qualify, index_set, pred_val, pred_wt = validate_prediction(pred_ind, Max_W, Test_Val)
            if qualify:
                if pred_val > optimal_bs['pred_val']:
                    optimal_bs['pred_val'] = pred_val
                    optimal_bs['pred_ind'] = pred_ind
                    optimal_bs['index_set'] = index_set
                    optimal_bs['pred_wt'] = pred_wt
                    optimal_bs['qc'] = qc
                    optimal_bs['qualify'] = qualify
                # return qualify, pred_ind, index_set, pred_val, pred_wt, qc, total_iters, total_time
            else:
                optimal_bs['qc'] = qc
            Q_sum = Q_sum + j 
            m = min(lamb*m, sqrt(L))
            j = randrange(1, ceil(m))

            # Random thought: to use the final remainig oracle calls
            if Q_sum + j > Q_max and Q_sum < Q_max:
                j = Q_max - Q_sum
                
        print('Completed ', r+1, ' runs')
        r = r + 1
    
    
    print('Test_Val', Test_Val)
    print('Total Iteartions for this Test Val:', total_iters)
    print('Total Q Time for this Test Val:', total_time)
    return optimal_bs['qualify'], optimal_bs['pred_ind'], optimal_bs['index_set'], optimal_bs['pred_val'], optimal_bs['pred_wt'], optimal_bs['qc'], total_iters, total_time


def find_max(Q_max, Max_Val, Max_W, L, backend, shot):

    qualify = False
    qualify_indexes = ''
    pred_wt = 0
    qualify_qc = None
    
    curr_best = 0
    upper_limit = Max_Val
        
    curr_wt = 0
    total_o_calls = 0
    total_time = 0
    
    # Starting from mid
    Test_Val = ceil((curr_best + upper_limit)/2)
    
    # Tracking the best indexes identified till now
    opt_idx = ''
    opt_idx_wt = 0
    opt_idx_val = 0
    opt_qc = ''
    convergence = []
    

    while curr_best <= upper_limit :
        print('Current Best:', curr_best, ', Upper Limit:', upper_limit, ', Test Val:', Test_Val)
        qualify, pred_ind, index_set, pred_val, pred_wt, qc, o_calls, time_taken = find_index_set(Test_Val, Q_max, L, curr_best, backend, shot, Max_W)
        conv_str = 'Predicted Index Set:' + str(index_set) + ', With Value:'+ str(pred_val) + ', And Weight:'+ str(pred_wt)
        convergence.append(conv_str)
        print(conv_str)
        print('Iterations done for this test val:', o_calls)
        total_o_calls +=  o_calls
        total_time += time_taken
        
        if qualify:
            if pred_val > opt_idx_val:
                opt_idx = pred_ind
                opt_idx_wt = pred_wt
                opt_idx_val = pred_val
                opt_qc = qc
                
            curr_best = Test_Val + 1
            print('Successful Prediction')
        else:
            upper_limit = Test_Val - 1
            print('Failed Prediction')
            
        Test_Val = curr_best + (upper_limit - curr_best) // 2

        print('Next Test Val:', Test_Val)
            
    print('Total Iterations: ', total_o_calls)
    print('Total Q Time: ', total_time)
    opt_qc = qc
    return opt_idx, opt_idx_wt, opt_idx_val, opt_qc, total_o_calls, total_time, convergence
    
    
def run_gs_exp(repeat, shots, p_list, succ_prob, epsilon, alpha_list):
    sol_list = list()
    wt_list = list()
    val_list = list()
    shot_list = list()
    status_list = list()
    time_list = list()
    qubit_list = list()
    depth_list = list()
    ocall_list = list()

    from utils import backend_name, filepath

    exp_res = []
    for r in range(repeat):
        for shot in shots:
            for alpha in alpha_list:
                for cur_prob in p_list:
    #                 gc.collect()
                    output_dict = dict()
                    prob_instance = problems[cur_prob]
    #                 backend, backend_name, filepath = initBackend(local, is_Noisy)
                    print('Backend:', backend_name)
                    print('Store Location:', filepath)
                    Num_I = len(prob_instance['profits'])
                    Values = prob_instance['profits']
                    Weights = prob_instance['weights']
                    Max_W = prob_instance['max_wgt']
                    Indexes = range(Num_I)
                    L = 2 ** Num_I

                    df_prob = pd.DataFrame(list(zip(Indexes, Values, Weights)),
                               columns =['Index', 'Value', 'Weight'])
                    print(df_prob)

                    print('Problem Index:', cur_prob)

                    output_dict['Problem'] = problems[cur_prob]


                    output_dict['Problem_ID'] = cur_prob

                    init_prob_det(Num_I, Values, Weights)

                    # Lets determine the number of qubits
                    in_q_ct = Num_I
                    val_q_ct = ceil(log(sum(Values),2)) + 1 # Adding 1 to store the sign of the final weights
                    wt_q_ct = ceil(log(sum(Weights),2)) + 1 # Adding 1 to store the sign of the final weights

                    output_dict['num_index_qubits (I)'] = in_q_ct
                    output_dict['num_value_qubits'] = val_q_ct
                    output_dict['num_weight_qubits'] = wt_q_ct

                    # Identifying the Candidate Solutions
                    flag_ct = 1

                    output_dict['num_ancilla_qubits'] = flag_ct * 3
                    output_dict['Qubits'] = in_q_ct + val_q_ct + wt_q_ct + flag_ct * 3
                    print('Total Qubits Needed:', output_dict['Qubits'])

                    init_gs_qubits(in_q_ct, val_q_ct, wt_q_ct, flag_ct)

                    # Find Optimal Configuration
                    # Init Params
                    L = 2 ** Num_I
                    Iter_fudge = 1
                    Q_max = floor(alpha * sqrt(L))  ## Changed along with the BS fix
                    Max_Val = sum(Values) - 1
                    print('Iter_Time_Out (alpha * sqrt(L)):', Q_max)

                    # Find Max
                    qualify_indexes, qualify_idx_wt, qualify_idx_val, qc_res, total_o_calls, total_time, convergence = find_max(Q_max, Max_Val, Max_W, L, backend, shot)
                    print('Exit Find Max with qualify_indexes', qualify_indexes)
                    output_dict['Tot_Oracle_Calls'] = total_o_calls
                    output_dict['Total_Time'] = total_time
                    output_dict['Convergence'] = convergence


                    # Test if we found a valid index configuration
                    Test_Val = 0
                    if qualify_indexes:
                        qualify, index_set, pred_val, pred_wt = validate_prediction(qualify_indexes, Max_W, Test_Val)
                    else:
                        qualify = False
                        index_set = [] 
                        pred_val = -10000
                        pred_wt = -10000

                    # Record Status
                    if qualify:                        
                        output_dict['IS_Valid'] = 'True'
                        status_list.append('SUCCESS')
                    else:
                        print('Failed Execution')
                        output_dict['IS_Valid'] = 'False'
                        status_list.append('INFEASIBLE')

                    output_dict['Opt_indexes'] = index_set
                    output_dict['Opt_Value'] = pred_val
                    output_dict['Opt_Weight'] = pred_wt

                    # Execution Time, Normalize Value, Total oracle calls, Record Intermediate results

                    output_dict['IS_Sub_Optimal'] = 'True'
                    # Record Run Status -- if the execution was really successfull and we did found the optimal value
                    if sorted(prob_instance['opt_sol']['opt_index']) == sorted(index_set):
                        output_dict['Run_Status'] = 'SUCCESS'
                        output_dict['IS_Sub_Optimal'] = 'False'
                    else:
                        output_dict['Run_Status'] = 'FAILURE'

                    output_dict['Success_Prob'] = succ_prob
                    output_dict['Num_Runs'] = Num_runs
                    output_dict['Max_Val'] = Max_Val
                    output_dict['alpha'] = alpha
                    output_dict['Total_Configs(L)'] = L
                    output_dict['Iter_Time_Out (alpha * sqrt(L))'] = Q_max


                    # Save Location

                    filename = 'P_'+ str(cur_prob) + '_alpha_' + str(alpha) + '_rep_' + str(r+1) +'_b_' + backend_name + '_shot_' + str(shot)

                    # Record Qc Details
                    output_dict['backend'] = backend_name
                    output_dict['Shots'] = shot
                    if qc_res:
                        output_dict['Depth'] = qc_res.depth()

                        # Persist the quantum circuit
                        qc_file = os.path.join(filepath, filename + '_qc.qasm')
                        output_dict['qc_file'] = qc_file
                        blockPrint()
                        qc_res.qasm(formatted=True, filename=qc_file)
                        enablePrint()


                    output_dict['IsOptimal'] = sorted(output_dict['Opt_indexes']) == sorted(problems[cur_prob]['opt_sol']['opt_index'])

                    # Persist Json result
                    json_object = json.dumps(output_dict, indent=4,cls=NpEncoder)
                    res_file = os.path.join(filepath, filename + '_result.json')
                    with open(res_file, "w") as outfile:
                        outfile.write(json_object)

                    print(output_dict)
                    exp_res.append(output_dict)
                    print('Execution Done:', filepath + filename)
                    print('***************************************************************************')

                    sol_list.append(index_set)
                    wt_list.append(pred_wt)
                    val_list.append(pred_val)
                    shot_list.append(shot)
                    time_list.append(total_time)
                    ocall_list.append(total_o_calls)
                    qubit_list.append(output_dict['Qubits'])
                    depth_list.append(output_dict['Depth'])

    print('RAN SUCCESSFULLY!!!!!')

    return exp_res

