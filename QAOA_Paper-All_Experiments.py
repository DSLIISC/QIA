import sys
from utils import *
from qaoa_ia import *
import imgkit
import argparse


def execute_qaoa(prob_id, noisy, base_path, is_inc_shot, repeat = 10, p_max = 1, shots = [100]):
    
    inst_obj = get_problem_instance(prob_id)
    populate_prob_instance(inst_obj)

    add_optimal()

    add_greedy()

    problems

    base_path = os.path.join(base_path, str(prob_id))
    if is_inc_shot:
        base_path = os.path.join(base_path, 'inc_shot')
    else:
        base_path = os.path.join(base_path, 'one')
    init_global_param(noisy, base_path)

    # Save problem
    from utils import filepath
    res_file = os.path.join(filepath, 'problem.json')
    persist_json_file(problems, res_file)


    from utils import filepath


    # Perform Experiments
    results = []
    if is_inc_shot:
#         shots = [1, 10, 100, 1000]
        shots = [100]
        for shot in shots:
            res_p = solve_problem_qaoa(repeat, p_max, [shot])
            results.append(res_p)
    else: 
        p_max = 5
        shot = 1000
        for p in range(1,p_max+1):
            res_p = solve_problem_qaoa(repeat, p, [shot])
            results.append(res_p)


    # Persist Result Json
    from utils import filepath
    data = {'Results': results}
    res_file = os.path.join(filepath, 'all_res_inc_shot_repeat_' + str(repeat) +'.json')
    persist_json_file(data, res_file)


    print('Result Json at:', res_file)
    print(getScore_OQIA(results[0]))


def main():
    print("In QAOA Exepriment")
    parser = argparse.ArgumentParser()
    parser.add_argument('-pid', '--prob_id', type=int)
    parser.add_argument('-bp', '--base_path', type=str)
    parser.add_argument('-n', '--noisy', type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('-inc', '--inc', type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('-r', '--repeat', type=int, default=10)
    parser.add_argument('-p', '--p_max', type=int)
    args = parser.parse_args()

    print (args.prob_id)
    print (args.base_path)
    print(args.noisy, type(args.noisy))
    print (args.inc)

    repeat = None
    if args.repeat:
        repeat = args.repeat
        print('repeat is set to:', repeat)
    else:
        print('No repeat, setting to default value of 10')
        repeat = 10

    p_max = None
    if args.p_max:
        p_max = args.p_max
        print('p is set to:', p_max)
    else:
        print('No p_max, setting to default value of 1')
        p_max = 1

    execute_qaoa(args.prob_id, args.noisy, args.base_path, args.inc, repeat=repeat, p_max=p_max)

if __name__ == "__main__":
    main()

