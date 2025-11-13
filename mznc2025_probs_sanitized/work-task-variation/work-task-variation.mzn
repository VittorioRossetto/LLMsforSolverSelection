include "globals.mzn";
enum Resources;
enum Activities;
enum ActivitiesOrNone = A(Activities) ++ { None };
int: slots;
set of int: Slots = 1..slots;
set of int: SlotsAndZero = 0..slots;
array[Activities, Slots] of 0..card(Resources): requirements;
array[Resources, Slots] of opt ActivitiesOrNone: fixed;
array[Activities, SlotsAndZero] of int: activity_run_cost;
array[Activities, SlotsAndZero] of int: activity_frequency_cost;
array[ActivitiesOrNone, SlotsAndZero] of int: extended_activity_run_cost =
    array2d(ActivitiesOrNone, SlotsAndZero,
        array1d(activity_run_cost) ++
        [0 | r in SlotsAndZero]
);
array[ActivitiesOrNone, Slots] of 0..card(Resources): extended_requirements = array2d(ActivitiesOrNone, Slots,
    array1d(requirements) ++
    [card(Resources) - sum(requirements[.., s]) | s in Slots]
);
array[Resources, Slots] of var ActivitiesOrNone: schedule;
array[Resources, Slots] of var bool: run_end;
array[Resources, Slots] of var SlotsAndZero: run_length;
array[Resources, Slots] of var int: run_cost;
array[Resources, Activities] of var int: frequency_cost;
var int: cost = sum(run_cost) + sum(frequency_cost);
constraint forall (r in Resources) (
    regular(schedule[r, ..], "None* [^None]* None*")
);
constraint forall (s in Slots) (
    global_cardinality(schedule[.., s], ActivitiesOrNone, extended_requirements[.., s])
);
constraint forall (r in Resources, s in Slots where occurs(fixed[r, s])) (
    schedule[r, s] = deopt(fixed[r, s])
);
constraint forall (r in Resources, s in Slots) (
    if s = slots then
        run_end[r, s] = true
    else
        run_end[r, s] = (schedule[r, s] != schedule[r, s+1])
    endif
);
constraint forall (r in Resources, s in Slots) (
    if s = 1 \/ run_end[r, s-1] then
        run_length[r, s] = 1
    else
        run_length[r, s] = run_length[r, s-1] + 1
    endif
);
constraint forall (r in Resources, s in Slots) (
    if run_end[r, s] then
        run_cost[r, s] = extended_activity_run_cost[schedule[r, s], run_length[r, s]]
    else
        run_cost[r, s] = 0
    endif
);
constraint forall (r in Resources, a in Activities) (
    let {
        var int: run_ends = count(s in Slots) (run_end[r, s] /\ schedule[r, s] = A(a))
    } in
    frequency_cost[r, a] = activity_frequency_cost[a, run_ends]
);
var int: objective :: output = cost;
solve :: int_search(schedule, first_fail, indomain_min) minimize objective;
output ["objective = \(cost)\n"];