import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy
import matplotlib.pyplot as plt
import json

import os


# self.results_dict = {'Sequencing Time':[], 'Planner Computation Time':[], 'Planner Execution Time':[], 'Approach Time':[], 'Num Apples':0}
# self.failures_dict = {'Joint Limits':[], 'Low Manip':[], 'Planner': [], 'Grasp Misalignment':[], 'Grasp Obstructed':[]}

eucl_dir = '/home/fred/hydra_ws/src/sawyer_planner/results/penalty_tanh/10_fixed_neq1/euclidean'
fred_dir = '/home/fred/hydra_ws/src/sawyer_planner/results/penalty_tanh/10_fixed_neq1/fredsmp'

eucl_dir = '/home/fred/hydra_ws/src/sawyer_planner/results/penalty_tanh/10_fixed_neq1_joint_diff_tsp/euclidean'
fred_dir = '/home/fred/hydra_ws/src/sawyer_planner/results/penalty_tanh/10_fixed_neq1_joint_diff_tsp/fredsmp'

eucl_dir = '/home/fred/hydra_ws/src/sawyer_planner/results/penalty_tanh/10_fixed/euclidean'
fred_dir = '/home/fred/hydra_ws/src/sawyer_planner/results/penalty_tanh/10_fixed/fredsmp'

eucl_dir = '/home/fred/hydra_ws/src/sawyer_planner/results/penalty_tanh/20/euclidean'
fred_dir = '/home/fred/hydra_ws/src/sawyer_planner/results/penalty_tanh/20/fredsmp'

eucl_dir = '/home/fred/hydra_ws/src/sawyer_planner/results/penalty_tanh/15/euclidean'
fred_dir = '/home/fred/hydra_ws/src/sawyer_planner/results/penalty_tanh/15/fredsmp'

# eucl_dir = '/home/fred/hydra_ws/src/sawyer_planner/results/penalty_tanh/10/euclidean'
# fred_dir = '/home/fred/hydra_ws/src/sawyer_planner/results/penalty_tanh/10/fredsmp'

# eucl_dir = '/home/fred/hydra_ws/src/sawyer_planner/results/penalty_tanh/5/euclidean'
# fred_dir = '/home/fred/hydra_ws/src/sawyer_planner/results/penalty_tanh/5/fredsmp'

# eucl_dir = '/home/fred/hydra_ws/src/sawyer_planner/results/euclidean'
# fred_dir = '/home/fred/hydra_ws/src/sawyer_planner/results/fredsmp'
# hybrid_dir = '/home/fred/hydra_ws/src/sawyer_planner/results/hybrid'
# objects = ('naive', 'fredsmp', 'hybrid')
objects = ('naive', 'fredsmp')
# dirs_array = [eucl_dir, fred_dir, hybrid_dir]
dirs_array = [eucl_dir, fred_dir]


planner_times_arr = []
sequencing_times_arr = []
planner_executions_arr = []
fails_arr_limit = []
fails_arr_manip = []
fails_arr_plan = []
approach_times_arr = []

for dir_name in dirs_array:
    planner_times = []
    sequencing_times = []
    planner_executions = []
    approach_times = []
    fails_limit = []
    fails_manip = []
    fails_plan = []
    print("num files: " + str(len(os.listdir(dir_name))))
    for filename in os.listdir(dir_name):
        fail_count_limit = 0
        fail_count_manip = 0
        fail_count_plan = 0
        if filename.startswith("results_"):
            json_file = os.path.join(dir_name, filename)
            with open(json_file, 'r') as f:
                results_dict = json.load(f)
            sequencing_times.append(results_dict['Sequencing Time'][0])
            planner_times.append(results_dict['Planner Computation Time'])
            planner_executions.append(results_dict['Planner Execution Time'][1:-1])
            approach_times.append(results_dict['Approach Time'])
            # print len(results_dict['Approach Time'])
        if filename.startswith("fails_"):
            json_file = os.path.join(dir_name, filename)
            with open(json_file, 'r') as f:
                fails_dict = json.load(f)
            fail_count_limit += sum(fails_dict['Joint Limits'])
            fail_count_manip += sum(fails_dict['Low Manip'])
            fail_count_plan += sum(fails_dict['Planner'])
            if len(fails_dict['Joint Limits']):
	            fails_limit.append(float(fail_count_limit) / float(len(fails_dict['Joint Limits'])) * 100.0)
            if len(fails_dict['Low Manip']):
    	        fails_manip.append(float(fail_count_manip) / float(len(fails_dict['Low Manip'])) * 100.0)
            if len(fails_dict['Planner']):
        	    fails_plan.append(float(fail_count_plan) / float(len(fails_dict['Planner'])) * 100.0)
    planner_times_arr.append(numpy.asarray(planner_times))
    sequencing_times_arr.append(numpy.asarray(sequencing_times))
    planner_executions_arr.append(numpy.asarray(planner_executions))
    approach_times_arr.append(numpy.asarray(approach_times))
    fails_arr_limit.append(numpy.asarray(fails_limit))
    fails_arr_manip.append(numpy.asarray(fails_manip))
    fails_arr_plan.append(numpy.asarray(fails_plan))


y_pos = numpy.arange(len(objects))

plt.close('all')
f, ax = plt.subplots(3, 3)
# plt.figure()
# print(planner_times_arr)
# planner_times_avg = []
# for mat in planner_times_arr:
#     planner_times_avg.append(numpy.nanmean([numpy.nanmean(row) for row in mat]))
# planner_times_avg = [numpy.nanmean(row) for row in planner_times_arr]
# performance = [10,8,6,4,2,1]
 
# ax[0,0].bar(y_pos, planner_times_avg, align='center', alpha=0.5)
# # ax[0,0].set_xticks(y_pos, objects)
# ax[0,0].set_ylabel('Time (s)')
# ax[0,0].set_title('Average Planner Computation Time')

box_data = []
for mat in planner_times_arr:
	# box_data.append(mat.flatten()[mat.flatten() < 3.0])
	box_data.append(mat.flatten())

ax[0,0].boxplot(box_data)
ax[0,0].set_ylabel('Time (s)')
ax[0,0].set_title('Average Planner Computation Time')

# plt.figure()
# print(planner_times_arr)
sequencing_times_avg = []
for mat in sequencing_times_arr:
    sequencing_times_avg.append(numpy.nanmean([numpy.nanmean(row) for row in mat]))
# performance = [10,8,6,4,2,1]
 
ax[0,1].bar(y_pos, sequencing_times_avg, align='center', alpha=0.5)
# ax[0,1].set_xticks(y_pos, objects)
ax[0,1].set_ylabel('Time (s)')
ax[0,1].set_title('Sequencing Time')

# plt.figure()
# # print(planner_times_arr)
planner_executions_avg = []
for mat in planner_executions_arr:
    planner_executions_avg.append(numpy.nanmean([numpy.nanmean(row) for row in mat]))
# performance = [10,8,6,4,2,1]
 
ax[2,1].bar(y_pos, planner_executions_avg, align='center', alpha=0.5)
ax[2,1].set_xticks(y_pos, objects)
ax[2,1].set_ylabel('Time (s)')
ax[2,1].set_title('Average Planner Execution Time')

# plt.figure()
box_data = []
for mat in planner_executions_arr:
	box_data.append(mat.flatten())
ax[0,2].boxplot(box_data)
ax[0,2].set_ylabel('Time (s)')
ax[0,2].set_title('Average Planner Execution Time')

# plt.figure()
# print(fails_arr)
fails_avg = []
for mat in fails_arr_limit:
    fails_avg.append(numpy.nanmean([numpy.nanmean(row) for row in mat]))
# performance = [10,8,6,4,2,1]
 
ax[1,0].bar(y_pos, fails_avg, align='center', alpha=0.5)
# ax[1,0].set_xticks(y_pos, objects)
ax[1,0].set_ylabel('Percentage (%)')
ax[1,0].set_title('Average Fails limit')

# plt.figure()
# print(fails_arr)
fails_avg = []
for mat in fails_arr_manip:
    fails_avg.append(numpy.nanmean([numpy.nanmean(row) for row in mat]))
# performance = [10,8,6,4,2,1]
 
ax[1,1].bar(y_pos, fails_avg, align='center', alpha=0.5)
# ax[1,1].set_xticks(y_pos, objects)
ax[1,1].set_ylabel('Percentage (%)')
ax[1,1].set_title('Average Fails manip')

# plt.figure()
# print(fails_arr)
fails_avg = []
for mat in fails_arr_plan:
    fails_avg.append(numpy.nanmean([numpy.nanmean(row) for row in mat]))
# performance = [10,8,6,4,2,1]
 
ax[1,2].bar(y_pos, fails_avg, align='center', alpha=0.5)
# ax[1,2].set_xticks(y_pos, objects)
ax[1,2].set_ylabel('Percentage (%)')
ax[1,2].set_title('Average Fails Planner')


# plt.figure()
# print approach_times_arr
approach_times_avg = []
for mat in approach_times_arr:
    approach_times_avg.append(numpy.nanmean([numpy.nanmean(row) for row in mat]))
print approach_times_avg
# performance = [10,8,6,4,2,1]
ax[2,0].bar(y_pos, approach_times_avg, align='center', alpha=0.5)
# ax[2,0].set_xticks(y_pos, objects)
ax[2,0].set_ylabel('Time (s)')
ax[2,0].set_title('Average Approach Time')

plt.setp(ax, xticks=y_pos, xticklabels=objects)

plt.show()