import sys
from utils import *
from sqia_ia import *
import imgkit
import argparse


def execute_gs(prob_id, noisy, base_path, shot, succ_prob = 0.9, repeat = 10, alpha = 2.0, lamb = 6/5):

    # Read Problem Instance
    inst_obj = get_problem_instance(prob_id)
    populate_prob_instance(inst_obj)

    add_optimal()

    add_greedy()

    problems

    base_path = os.path.join(base_path, str(prob_id))
    base_path = os.path.join(base_path, str(shot))

    # Init Global Parameters
    init_global_param(noisy, base_path)

    # Save problem
    from utils import filepath
    res_file = os.path.join(filepath, 'problem.json')
    persist_json_file(problems, res_file)

    # Calculate Internal Parameters
    raw_alpha, alpha, fudge = get_alpha_and_fudge(problems, succ_prob)    
    p_list = list(problems.keys())
    epsilon = 1 - succ_prob
    alpha_list = [alpha]
    Num_runs = ceil(log(1/epsilon,3)) + fudge
    shots = [shot]

    exp_profile = dict()
    exp_profile['noisy'] = noisy
    exp_profile['base_path'] = base_path
    exp_profile['repeat'] = repeat
    exp_profile['succ_prob'] = succ_prob
    exp_profile['alpha_list'] = alpha_list
    exp_profile['lamb'] = lamb
    exp_profile['Num_runs'] = Num_runs
    exp_profile['fudge'] = fudge
    exp_profile['alpha'] = alpha
    exp_profile['raw_alpha'] = raw_alpha
    print('exp_profile', exp_profile)
    
    from utils import filepath
    res_file = os.path.join(filepath, 'exp_profile.json')
    persist_json_file(exp_profile, res_file)

    init_gs_params(Num_runs, lamb)

    # Invoke SQIA Search
    exp_res = run_gs_exp(repeat, shots, p_list, succ_prob, epsilon, alpha_list)

    # Persist Results
    from utils import filepath
    data = {'Results': exp_res}
    res_file = os.path.join(filepath, 'all_res_repeat_' + str(repeat) +'.json')
    persist_json_file(data, res_file)

    print('Result Json at:', res_file)
    print(getScore_GS_relaxed(exp_res))


def main():
    print("In GS Exepriment")
    parser = argparse.ArgumentParser()
    parser.add_argument('-pid', '--prob_id', type=int)
    parser.add_argument('-bp', '--base_path', type=str)
    parser.add_argument('-shot', '--shot', type=int)
    parser.add_argument('-r', '--repeat', type=int, default=10)
    parser.add_argument('-n', '--noisy', type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('-sp', '--successprob', type=float)
    
    args = parser.parse_args()

    repeat = None
    if args.repeat:
        repeat = args.repeat
        print('repeat is set to:', repeat)
    else:
        print('No repeat, setting to default value of 10')
        repeat = 10

    success_prob = None 
    if args.successprob:
        success_prob = args.successprob
        print('success prob is set to:', success_prob)
    else:
        print('No success_prob, setting to default value of 0.9')
        success_prob = 0.9

#     shots = [1,10,100,1000]
    shots = [1]

    for shot in shots:
        execute_gs(args.prob_id, args.noisy, args.base_path, shot, repeat = repeat, succ_prob=success_prob)

if __name__ == "__main__":
    main()




