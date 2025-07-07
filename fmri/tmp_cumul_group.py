import numpy as np  
cg = True
group_size = 6
number_of_groups = 4
runs = list(range(number_of_groups*group_size)) 

groups = []

group_accumulator = []
cumulative_groups = []
for grouped_idx in range(number_of_groups):
    runs_in_group = runs[grouped_idx*group_size:(grouped_idx+1)*group_size]
    groups.append(runs_in_group)
    print(groups)
    group_accumulator.extend(runs_in_group)
    print(group_accumulator)
    cumulative_groups.append(group_accumulator.copy())
    print(cumulative_groups)

# runs_in_group = f"runs_{groups[grouped_run_idx*group_size:(grouped_run_idx+1)*group_size]}"

# # groups = np.arange(6,11)

# if (cg):
#     cumulative_groups = []
#     group_accumulator = []
#     grouped_run_idx in range(number_of_groups):
#     for group in groups:
#         print(group)
#         group_accumulator.append(group)
#         print(group_accumulator)
#         cumulative_groups.append(group_accumulator)
#         print(cumulative_groups)
