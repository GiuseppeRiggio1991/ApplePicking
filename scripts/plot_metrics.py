import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy
import matplotlib.pyplot as plt
import json

import os


# self.results_dict = {'Sequencing Time':[], 'Planner Computation Time':[], 'Planner Execution Time':[], 'Approach Time':[], 'Num Apples':0}
# self.failures_dict = {'Joint Limits':[], 'Low Manip':[], 'Planner': [], 'Grasp Misalignment':[], 'Grasp Obstructed':[]}

eucl_dir = '/home/planner/hydra_ws/src/sawyer_planner/results/euclidean'
fred_dir = '/home/planner/hydra_ws/src/sawyer_planner/results/fredsmp'

dirs_array = [eucl_dir, fred_dir]
planner_times_arr = []
sequencing_times_arr = []
planner_executions_arr = []
fails_arr = []
approach_times_arr = []

for dir_name in dirs_array:
    planner_times = []
    sequencing_times = []
    planner_executions = []
    approach_times = []
    fails = []
    for filename in os.listdir(dir_name):
        fail_count = 0
        if filename.startswith("results_"):
            json_file = os.path.join(dir_name, filename)
            with open(json_file, 'r') as f:
                results_dict = json.load(f)
            sequencing_times.append(results_dict['Sequencing Time'])
            planner_times.append(results_dict['Planner Computation Time'])
            planner_executions.append(results_dict['Planner Execution Time'])
            approach_times.append(results_dict['Approach Time'])
            print len(results_dict['Approach Time'])
        if filename.startswith("fails_"):
            json_file = os.path.join(dir_name, filename)
            with open(json_file, 'r') as f:
                results_dict = json.load(f)
            fail_count += sum(results_dict['Joint Limits'])
            fail_count += sum(results_dict['Low Manip'])
            fail_count += sum(results_dict['Planner'])
            fails.append(float(fail_count) / float(len(results_dict['Joint Limits'])) * 100.0)
    planner_times_arr.append(numpy.asarray(planner_times))
    sequencing_times_arr.append(numpy.asarray(sequencing_times))
    planner_executions_arr.append(numpy.asarray(planner_executions))
    approach_times_arr.append(numpy.asarray(approach_times))
    fails_arr.append(numpy.asarray(fails))


objects = ('euclidean', 'fredsmp')
y_pos = numpy.arange(len(objects))

plt.figure()
# print(planner_times_arr)
planner_times_avg = []
for mat in planner_times_arr:
    planner_times_avg.append(numpy.nanmean([numpy.nanmean(row) for row in mat]))
# planner_times_avg = [numpy.nanmean(row) for row in planner_times_arr]
# performance = [10,8,6,4,2,1]
 
plt.bar(y_pos, planner_times_avg, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Time (s)')
plt.title('Average Planner Computation Time')

plt.figure()
# print(planner_times_arr)
sequencing_times_avg = []
for mat in sequencing_times_arr:
    sequencing_times_avg.append(numpy.nanmean([numpy.nanmean(row) for row in mat]))
# performance = [10,8,6,4,2,1]
 
plt.bar(y_pos, sequencing_times_avg, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Time (s)')
plt.title('Average Sequencing Time')

plt.figure()
# print(planner_times_arr)
planner_executions_avg = []
for mat in planner_executions_arr:
    planner_executions_avg.append(numpy.nanmean([numpy.nanmean(row) for row in mat]))
# performance = [10,8,6,4,2,1]
 
plt.bar(y_pos, planner_executions_avg, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Time (s)')
plt.title('Average Planner Execution Time')

plt.figure()
print(fails_arr)
fails_avg = []
for mat in fails_arr:
    fails_avg.append(numpy.nanmean([numpy.nanmean(row) for row in mat]))
# performance = [10,8,6,4,2,1]
 
plt.bar(y_pos, fails_avg, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Percentage (%)')
plt.title('Average Fails')


plt.figure()
# print approach_times_arr
approach_times_avg = []
for mat in approach_times_arr:
    approach_times_avg.append(numpy.nanmean([numpy.nanmean(row) for row in mat]))
print approach_times_avg
# performance = [10,8,6,4,2,1]
plt.bar(y_pos, approach_times_avg, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Time (s)')
plt.title('Average Approach Time')

plt.show()