include "count.mzn";
include "table.mzn";
set of int: user_ids;
set of int: activity1_ids;
set of int: activity2_ids;
set of int: cell_ids;
set of int: group_ids;
set of int: time_slot_ids;
set of int: pub_rating_domain = 0..5;
set of int: user_rating_domain = -2..2;
int: min_group_size;
int: usersn;
int: max_wait;
int: startAfter;
int: eta;
array[activity1_ids,1..5] of int: activities1;
array[activity2_ids,1..5] of int: activities2;
array[user_ids,activity1_ids] of user_rating_domain: preferences1;
array[user_ids,activity2_ids] of user_rating_domain: preferences2;
array[activity1_ids] of int: oid1;
array[activity2_ids] of int: oid2;
array[cell_ids,cell_ids] of int: distances;
array[activity1_ids,1..6] of int: activities1_data = array2d(activity1_ids,1..6,
  [ if i=1 then j else activities1[j,i-1] endif | j in activity1_ids, i in 1..6 ]);
array[activity2_ids,1..6] of int: activities2_data = array2d(activity2_ids,1..6,
  [ if i=1 then j else activities2[j,i-1] endif | j in activity2_ids, i in 1..6  ]);
array[1..max(user_ids)*max(activity1_ids),1..3] of int: preferences1_data = array2d(1..max(user_ids)*max(activity1_ids),1..3,
  array[1..max(user_ids)*max(activity1_ids),1..3] of int: preferences1_data = [ if k=1 then i else if k=2 then j else preferences1[i,j] endif endif | i in user... ] // array too long to display, dimensions: (3)
);
array[1..max(user_ids)*max(activity2_ids),1..3] of int: preferences2_data = array2d(1..max(user_ids)*max(activity2_ids),1..3,
  array[1..max(user_ids)*max(activity2_ids),1..3] of int: preferences2_data = [ if k=1 then i else if k=2 then j else preferences2[i,j] endif endif | i in user... ] // array too long to display, dimensions: (3)
);
array[1..max(cell_ids)*max(cell_ids),1..3] of int: distances_data = array2d(1..max(cell_ids)*max(cell_ids),1..3,
  array[1..max(cell_ids)*max(cell_ids),1..3] of int: distances_data = [ if k=1 then i else if k=2 then j else distances[i,j] endif endif | i in cell_id... ] // array too long to display, dimensions: (3)
);
array[user_ids] of var group_ids: user_group_map1;
array[user_ids] of var group_ids: user_group_map2;
array[group_ids] of var activity1_ids: group_act_map1;
array[group_ids] of var activity2_ids: group_act_map2;
array[user_ids] of var activity1_ids: user_act_map1;
array[user_ids] of var activity2_ids: user_act_map2;
array[user_ids] of var time_slot_ids: user_start_time_map1;
array[user_ids] of var time_slot_ids: user_start_time_map2;
array[user_ids] of var time_slot_ids: user_duration_map1;
array[user_ids] of var time_slot_ids: user_duration_map2;
array[user_ids] of var time_slot_ids: activity_available_from1;
array[user_ids] of var time_slot_ids: activity_available_from2;
array[user_ids] of var time_slot_ids: user_end_map1;
array[user_ids] of var time_slot_ids: user_end_map2;
array[user_ids] of var cell_ids: user_cell_map1;
array[user_ids] of var cell_ids: user_cell_map2;
array[user_ids] of var pub_rating_domain: user_pub_rating_map1;
array[user_ids] of var pub_rating_domain: user_pub_rating_map2;
array[user_ids] of var user_rating_domain: user_weight_map1;
array[user_ids] of var user_rating_domain: user_weight_map2;
array[user_ids] of var time_slot_ids: user_distance_map;
constraint forall(i in group_ids) (
  let { var min_group_size..usersn: c} in (
    count(user_group_map1,i,c)) );
constraint forall(i in group_ids) (
  let { var min_group_size..usersn: c} in (
    count(user_group_map2,i,c)) );
constraint forall(i in user_ids) (
   table(   [ if k=1 then i else if k = [ user_act_map1[i], activity_available_from1[i], user_end_map1[i], user_duration_... ], // array too long to display, dimensions: (6)
 activities1_data) );
constraint forall(i in user_ids) (
   table(   [ if k=1 then i else if k = [ user_act_map2[i], activity_available_from2[i], user_end_map2[i], user_duration_... ], // array too long to display, dimensions: (6)
 activities2_data) );
constraint forall(i in user_ids) (
  table( [ i, user_act_map1[i], user_weight_map1[i] ], preferences1_data));
constraint forall(i in user_ids) (
  table( [ i, user_act_map2[i], user_weight_map2[i] ], preferences2_data));
constraint forall(i in user_ids) (
  table( [ user_cell_map1[i], user_cell_map2[i], user_distance_map[i] ], distances_data));
constraint forall(i in user_ids) (
  user_act_map1[i] = group_act_map1[user_group_map1[i]] );
constraint forall(i in user_ids) (
  user_act_map2[i] = group_act_map2[user_group_map2[i]] );
constraint redundant_constraint(
  user_group_map1[min(user_ids)] = min(group_ids) /\
  user_group_map2[min(user_ids)] = min(group_ids));
constraint symmetry_breaking_constraint(
  forall (i in 1..max(group_ids)-min(group_ids)+1) (
  user_group_map1[min(user_ids)+i] in min(group_ids)..min(group_ids)+i /\
  user_group_map2[min(user_ids)+i] in min(group_ids)..min(group_ids)+i
));
constraint forall(i in user_ids) (
  user_start_time_map1[i] >= activity_available_from1[i] /\
  user_start_time_map1[i] <= user_end_map1[i] - user_duration_map1[i]);
constraint forall(i in user_ids) (
  user_start_time_map2[i] >= activity_available_from2[i] /\
  user_start_time_map2[i] <= user_end_map2[i] - user_duration_map2[i]);
constraint forall(i in user_ids) (
  user_start_time_map2[i] >= user_start_time_map1[i] + user_duration_map1[i] +
    user_distance_map[i] /\
  user_start_time_map2[i] <= user_start_time_map1[i] + user_duration_map1[i] + max_wait );
constraint forall(i in user_ids) (
  user_start_time_map1[i] >= startAfter);
int: objub = eta*card(user_ids)*2 + (10-eta)*card(user_ids)*5 + (10-eta)*card(user_ids)*5 + eta*card(user_ids)*2;
int: objlb=eta*card(user_ids)*(-2)+ (10-eta)*card(user_ids)*0 + (10-eta)*card(user_ids)*0 + eta*card(user_ids)*(-2);
var objlb..objub: objective;
constraint objective = (
  eta * sum (user_weight_map1) + (10-eta) * sum (user_pub_rating_map1) + (10-eta) * sum (user_pub_rating_map2) + eta * sum (user_weight_map2) );
solve :: seq_search(constraint objective = [
    int_search(user_group_map1,first_fail, indoma... ] // array too long to display, dimensions: (48)
)
  maximize objective;
output constraint objective = [
    "user_group_map1 = \(user_group_map1);\n",
  ... ]; // array too long to display, dimensions: (13)
