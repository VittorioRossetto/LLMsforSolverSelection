include "globals.mzn";
int: num_occupants;
set of int: C = 1..num_occupants;
int: num_patients;
set of int: P = 1..num_patients;
int: horizon;
set of int: D = 0..horizon-1;
int: num_rooms;
set of int: R = 1..num_rooms;
int: num_ot;
set of int: O = 1..num_ot;
int: max_incompatible;
set of int: MI = 1..max_incompatible;
int: num_surgeons;
set of int: S = 1..num_surgeons;
int: weight_selection;
int: weight_delay;
array[C] of int: oc_length_of_stay;
array[C] of int: oc_gender;
array[C] of int: oc_room;
array[P] of int: length_of_stay;
array[P] of int: gender;
array[P,MI] of int: incompatible_rooms;
array[P] of bool: mandatory;
array[P] of int: surgery_duration;
array[P] of int: surgeon;
array[P] of int: release_day;
array[P] of int: due_day;
array[S,D] of int: max_surgery;
array[O,D] of int: max_ot;
array[R] of int: capacity;
array[P] of var bool: selection;
array[P] of var D: admission;
array[P] of var R: room;
array[P] of var O: ot;
constraint forall (c in C, p in P where oc_gender[c] != gender[p] /\ selection[p]) (oc_room[c] != room[p] \/ admission[p] >= oc_length_of_stay[c]);
constraint forall (p1 in P, p2 in P where gender[p1] != gender[p2] /\ selection[p1] /\ selection[p2]) (room[p1] != room[p2] \/ admission[p1] >= admission[p2] + length_of_stay[p2] \/ admission[p2] >= admission[p1] + length_of_stay[p1]);
constraint forall (p in P, i in MI where incompatible_rooms[p,i] > 0) (room[p] != incompatible_rooms[p,i]);
constraint forall (s in S, d in D) (sum(constraint forall (p in P, i in MI where incompatible_rooms[p,i] > 0) (room[p] ! = [surgery_duration[p] | p in P where selection[p] /\ admission[p] == d /\ surgeon[... ] // array too long to display, dimensions: (1)
) <= max_surgery[s,d]);
constraint forall (o in O, d in D) (sum(constraint forall (s in S, d in D) (sum([surgery_duration[p] | p in P where selection[p] /\ admission[p] == d /\ surgeon[p] == s]) < = [surgery_duration[p] | p in P where selection[p] /\ admission[p] == d /\ ot[p] ==... ] // array too long to display, dimensions: (1)
) <= max_ot[o,d]);
constraint forall (p in P) (mandatory[p] -> selection[p]);
array[P] of var D: admission_delay = [if selection[p] then admission[p] - release_day[p] else 0 endif | p in P];
constraint forall (p in P) (admission[p] <= due_day[p]);
array[R,P] of var 0..max(length_of_stay)*num_patients: room_admission;
constraint forall (p in P where selection[p]) (room_admission[room[p],p] == admission[p]);
constraint forall (r in R) (cumulative([0 | c in C where oc_room[c] == r] ++ row(room_admission, r), [oc_length_of_stay[c] | c in C where oc_room[c] == r] ++ length_of_stay, [1 | c in C where oc_room[c] == r] ++ [1 | p in P], capacity[r]));
var int: objective = count(selection, false) * weight_selection + sum(admission_delay) * weight_delay;
solve :: seq_search(var int: objective = [
    bool_search(selection, input_order, indomain_min),
    int_search(admission... ] // array too long to display, dimensions: (6)
) minimize objective;