import sys
from cspProblem import CSP, Constraint
from math import floor
from cspConsistency import Search_with_AC_from_CSP
import searchGeneric
from display import Displayable


workday_domain={"mon":1, "tue":2, "wed":3, "thu":4, "fri":5}
worktime_domain={"9am":0, "10am":1, "11am":2, "12pm":3, "1pm":4, "2pm":5, "3pm":6, "4pm":7, "5pm":8}
tasks_domain=set([10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58])
file_input = sys.argv[1]
input_dict={1:[], 2:[], 3:[], 4:[]}        # for read lines of file


# """
#  change searchGeneric.py: delete path.cost() in AStarSearcher
#  change cspConsistency.py: dom2 = set(dom )- dom1 in partition_domain
#
# """



# """extend CSP in cspProblem
#    increase soft_constraints(task soft constraint time, task cost)
# """

class Fuzzy_Schedule_CSP(CSP):
    def __init__(self,domains,constraints,soft_constraints):
        super().__init__(domains,constraints)
        self.soft_constraints=soft_constraints



# """
#    extend Search_with_AC_from_CSP in cspConsistency
#    overwrite heuristic function for increasing cost
# """



class Search_with_AC_from_Fuzzy_Schedule_CSP(Search_with_AC_from_CSP):
    def __init__(self,fuzzy_schedule_csp):
        super().__init__(fuzzy_schedule_csp)
        self.soft_constraints=fuzzy_schedule_csp.soft_constraints
    def heuristic(self,n):

        """
           find minimun cost of tasks
           minimun end time of task is needed cost
           sum all cost of tasks 
           return:total cost
        
        """
        sum_cost = []
        for i in n.keys():
            if i in self.soft_constraints.keys():
                needed_time = self.soft_constraints[i][0][0]
                if len(n[i])>0:
                    min_val = min(n[i])
                    if min_val[1] <= int(needed_time):
                        min_cost = 0
                    else:
                        cost_d = floor(min_val[1] / 10) - floor(int(needed_time) / 10)
                        cost_t = floor(min_val[1] % 10) - floor(int(needed_time) % 10)
                        min_cost = (cost_d * 24 + cost_t) * int(self.soft_constraints[i][0][1])
                    sum_cost.append(min_cost)

        return sum(sum_cost)



def r_file():
    """
    read input file
    Save the task domain and constraint of the file to the dictionary:
    key =1 means tasks with name and duration
    key =2 means binary constraints
    key =3 means  hard domain constraints
    key =4 means soft deadline constraints
    return dictionary

    """
    file_line = []
    with open(file_input, 'r') as file:
        for line in file:
            file_line.append(line)
    for x in range(0, len(file_line)):
        if "task," in file_line[x]:
            input_dict[1].append(file_line[x])
            continue
        elif "constraint," in file_line[x]:
            input_dict[2].append(file_line[x])
            continue
        elif"domain,"in file_line[x]:
            if "ends-by" in file_line[x]:
                input_dict[4].append(file_line[x])
            else:
                input_dict[3].append(file_line[x])
    return input_dict




domain={}
def find_domain():
    """
    find every task domain
    every task domain's starting time and end time are valid
    end time=start time+duration
    return:valid task domain

    """
    task_list=[]
    for x in range(0, len(input_dict[1])):
        task_list.append(input_dict[1][x].split(",")[1].split())
    for task_num in range(0,len(task_list)):
        domain[task_list[task_num][0]]=[]
        task_duration=task_list[task_num][1]
        for time_s in tasks_domain:
            if int(time_s)+int(task_duration) in tasks_domain and floor(int(time_s) / 10)==floor((int(time_s) + int(task_duration)) / 10):
                domain[task_list[task_num][0]].append((int(time_s),int(time_s)+int(task_duration)))
    return domain

def h_domain(domain_day):
    """ is true if task starts on any time of specific day"""
    day_val=lambda s: domain_day==floor(s[0]/10)
    return day_val
def domain_constraint_time(task_time):
    """ is true if task starts on any day of specific time"""
    constraint_time=lambda t:task_time==floor(t[0]%(floor(t[0]/10)*10))
    return constraint_time
def start_before(start_day,start_time):
    """is true if task starts at or before given day and time"""
    start_before_time=lambda s_time: ((start_day*10)+start_time)>=int(s_time[0])
    return start_before_time
def start_after(start_after_day,start_after_time):
    """is true if task starts at or after given day and time"""
    start_after_time_val=lambda s_a_time: ((start_after_day*10)+start_after_time)<=int(s_a_time[0])
    return start_after_time_val
def end_before(end_day,end_time):
    """is true if task ends at or before given day and time"""
    end_time_val=lambda e_time: ((end_day*10)+end_time)>=int(e_time[1])
    return end_time_val
def end_after(end_after_day,end_after_time):
    """is true if task ends at or after given day and time"""
    end_after_time_val=lambda e_a_time: ((end_after_day*10)+end_after_time)<=int(e_a_time[1])
    return end_after_time_val
def start_before_time(constraint_time):
    """is true if task starts at or before given time on any day"""
    s_b_t_val=lambda task_time: constraint_time>=floor(int(task_time[0])%10)
    return s_b_t_val
def start_after_time(constraint_time):
    """is true if task starts at or after given time on any day"""
    s_a_t_val=lambda task_time: constraint_time<=floor(int(task_time[0])%10)
    return s_a_t_val
def end_before_time(constraint_time):
    """is true if task ends at or before given time on any day"""
    e_b_t_val=lambda task_time: constraint_time>=floor(int(task_time[1])%10)
    return e_b_t_val
def end_after_time(constraint_time):
    """is true if task ends at or after given time on any day"""
    e_a_t_val=lambda task_time: constraint_time<=floor(int(task_time[1])%10)
    return e_a_t_val
def start_in (constraint_time_1,constraint_time_2):
    """ is true if the starting time of task equal or within range"""
    range_time_val=lambda r_time:constraint_time_1<=r_time[0] and constraint_time_2>=r_time[0]
    return  range_time_val
def end_in (constraint_time_1,constraint_time_2):
    """ is true if the end time of task equal or within range"""
    range_time_val=lambda r_time:constraint_time_1<=r_time[1] and constraint_time_2>=r_time[1]
    return range_time_val





def find_hard_constraint():
    """
    find hard constraint including binary constraints and hard domain constraints
    split input_dict[2] and input_dict[3]
    binary constraints: scopes and conditions which satisfies the "before","same-day","after","starts_at" is added to hard_constraints
    hard domain constraints:scopes and conditions which satisfies the "starts-before","starts-after",
    "ends-before","ends-after","starts-in","ends-in"is added to hard_constraints
    return: hard_constraint
    """
    binary_list=[]
    hard_constraints=[]
    hard_list = []
    for num in range(0, len(input_dict[2])):
        binary_list.append(input_dict[2][num].split(",")[1].split())    #split binary constraint to <t1> <command> <t2> \n
        task_scope=(binary_list[num][0],binary_list[num][2])                 #task_scope(first task,second task)
        if binary_list[num][1]=="before":                               #first task end time <=second task start rime
            condition=lambda a,b:a[1]<=b[0]
            hard_constraints.append(Constraint(task_scope,condition))
        if binary_list[num][1]=="same-day":                             #first task start time =second task start rime
            condition=lambda x,y:floor(int(x[0])/10)==floor(int(y[0])/10)
            hard_constraints.append(Constraint(task_scope,condition))
        if binary_list[num][1]=="after":                                #first task end time >=second task start rime
            condition=lambda k,j:k[0]>=j[1]
            hard_constraints.append(Constraint(task_scope,condition))
        if binary_list[num][1]=="starts-at":                            #first task end time =second task start rime
            condition=lambda first_task,second_task:first_task[0]==second_task[1]
            hard_constraints.append(Constraint(task_scope,condition))
    for number in range(0, len(input_dict[3])):                         # hard domain constraint
        hard_list.append(input_dict[3][number].split(",")[1].split()) #split hard domain constraint to <task> <command> <day> <time> \n
        task_scope=(hard_list[number][0],)                                           #task_scope(task,)
        if hard_list[number][1] in workday_domain.keys():                            #task starts on any time
            condition = h_domain(int(workday_domain[hard_list[number][1]]))
            hard_constraints.append(Constraint(task_scope, condition))
        if hard_list[number][1] in worktime_domain.keys():                           #task starts on any day
            condition=domain_constraint_time(int(worktime_domain[hard_list[number][1]]))
            hard_constraints.append(Constraint(task_scope,condition))
        if hard_list[number][1]=="starts-before":
            if len(hard_list[number])==4:                                            #task starts at or before day time
                condition = start_before(int(workday_domain[hard_list[number][2]]),
                                         int(worktime_domain[hard_list[number][3]]))
                hard_constraints.append(Constraint(task_scope, condition))
            elif len(hard_list[number])==3:                                          #task starts at or before time
                condition = start_before_time(int(worktime_domain[hard_list[number][2]]))
                hard_constraints.append(Constraint(task_scope, condition))
        if hard_list[number][1]=="starts-after":
            if len(hard_list[number])==4:                                            #task starts at or after day time
                condition = start_after(int(workday_domain[hard_list[number][2]]),
                                        int(worktime_domain[hard_list[number][3]]))
                hard_constraints.append(Constraint(task_scope, condition))
            elif len(hard_list[number])==3:                                         #task starts at or after time
                condition = start_after_time(int(worktime_domain[hard_list[number][2]]))
                hard_constraints.append(Constraint(task_scope, condition))
        if hard_list[number][1]=="ends-before":
            if len(hard_list[number])==4:                                           #task ends at or before day time
                condition = end_before(int(workday_domain[hard_list[number][2]]),
                                       int(worktime_domain[hard_list[number][3]]))
                hard_constraints.append(Constraint(task_scope, condition))
            elif len(hard_list[number])==3:                                         #task ends at or before time
                condition = end_before_time(int(worktime_domain[hard_list[number][2]]))
                hard_constraints.append(Constraint(task_scope, condition))
        if hard_list[number][1]=="ends-after":
            if len(hard_list[number])==4:                                           #task ends at or after day time
                condition = end_after(int(workday_domain[hard_list[number][2]]),
                                      int(worktime_domain[hard_list[number][3]]))
                hard_constraints.append(Constraint(task_scope, condition))
            elif len(hard_list[number])==3:                                         #task ends at or after time
                condition = end_after_time(int(worktime_domain[hard_list[number][2]]))
                hard_constraints.append(Constraint(task_scope, condition))
        if hard_list[number][1]=="starts-in":                                       #task starts equal or within range
            range_time=[]
            range_time.append(int(workday_domain[hard_list[number][2]]))
            range_time.append(int(worktime_domain[hard_list[number][3].split("-")[0]]))
            range_time.append(int(workday_domain[hard_list[number][3].split("-")[1]]))
            range_time.append(int(worktime_domain[hard_list[number][4]]))
            condition=start_in((range_time[0]*10+range_time[1]),(range_time[2]*10+range_time[3]))
            hard_constraints.append(Constraint(task_scope,condition))
        if hard_list[number][1]=="ends-in":                                        #task ends equal or within range
            range_time=[]
            range_time.append(int(workday_domain[hard_list[number][2]]))
            range_time.append(int(worktime_domain[hard_list[number][3].split("-")[0]]))
            range_time.append(int(workday_domain[hard_list[number][3].split("-")[1]]))
            range_time.append(int(worktime_domain[hard_list[number][4]]))
            condition=end_in((range_time[0]*10+range_time[1]),(range_time[2]*10+range_time[3]))
            hard_constraints.append(Constraint(task_scope,condition))
    return hard_constraints



def find_soft_constraint():

    """
    find tasks soft constraint
    split input_dict[4]
    create a list of (task soft constraint time,task cost)
    soft constraint time = day value * 10 + time
    return:soft constraint list

    """
    soft_list=[]
    soft_constraints={}
    for soft_num in range(0, len(input_dict[4])):
        soft_list.append(input_dict[4][soft_num].split(",")[1].split())
        soft_constraints[soft_list[soft_num][0]]=[]
        soft_t= int(workday_domain[soft_list[soft_num][2]]) * 10 + int(worktime_domain[soft_list[soft_num][3]])
        soft_cost=int(soft_list[soft_num][4])
        soft_constraints[soft_list[soft_num][0]].append((soft_t,soft_cost))
    return soft_constraints






if __name__ == "__main__":
    Displayable.max_display_level=0                                # to modify max_display_level=0 of Displayable in display.py
    r_file()
    fuzzy_schedule_csp=Fuzzy_Schedule_CSP(find_domain(),find_hard_constraint(),find_soft_constraint())
    problem=Search_with_AC_from_Fuzzy_Schedule_CSP(fuzzy_schedule_csp)
    search_test=searchGeneric.AStarSearcher(problem)
    schedule=search_test.search()
    if schedule==None:
        print("No solution")
        exit(1)
    schedule_copy = schedule.end().copy()
    day_dict={}
    time_dict={}
    for k,v in workday_domain.items():
        day_dict[v]=k
    for k_time,v_time in worktime_domain.items():
        time_dict[v_time]=k_time
    for task in schedule_copy.keys():
        for time_start,time_end in schedule_copy[task]:
              day=day_dict[floor(int(time_start)/10)]
              work_time=time_dict[floor(int(time_start)%10)]
        print(f'{task}:{day} {work_time}')
    print(f'cost:{problem.heuristic(schedule_copy)}')














