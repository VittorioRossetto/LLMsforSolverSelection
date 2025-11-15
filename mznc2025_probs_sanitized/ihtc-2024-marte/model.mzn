include "all_different.mzn";
include "bin_packing_load.mzn";
include "cumulative.mzn";
include "nvalue_fn.mzn";
int: days;
int: skill_levels;
array[1..3] of string: shift_types;
list of string: age_groups;
list of string: genders;
type Occupant = record(
  string: id,
  int: index,
  string: gender,
  string: age_group,
  int: length_of_stay,
  list of int: workload_produced,
  list of int: skill_level_required,
  string: room_id
);
list of Occupant: occupants;
type Patient = record(
  string: id,
  int: index,
  string: gender,
  string: age_group,
  int: length_of_stay,
  int: surgery_release_day,
  int: surgery_due_day,
  int: surgery_duration,
  list of int: workload_produced,
  list of int: skill_level_required,
  bool: mandatory,
  string: surgeon_id,
  list of string: incompatible_room_ids
);
list of Patient: patients;
type Surgeon = record(
  string: id,
  int: index,
  list of int: max_surgery_time
);
list of Surgeon: surgeons;
type OperatingTheater = record(
  string: id,
  int: index,
  list of int: availability
);
list of OperatingTheater: operating_theaters;
type Room = record(
  string: id,
  int: index,
  int: capacity
);
list of Room: rooms;
type WorkingShift = record(
  int: day,
  string: shift,
  int: max_load
);
type Nurse = record(
  string: id,
  int: index,
  int: skill_level,
  list of WorkingShift: working_shifts
);
list of Nurse: nurses;
type Weights = record(
  int: room_mixed_age,
  int: room_nurse_skill,
  int: continuity_of_care,
  int: nurse_eccessive_workload,
  int: open_operating_theater,
  int: surgeon_transfer,
  int: patient_delay,
  int: unscheduled_optional
);
Weights: weights;
set of int: DAYS = 1..days;
set of int: XDAYS = 1..(days + max(1, days div 3));
set of int: XXDAYS = 1..(max([occupant.length_of_stay | occupant in occupants] ++
                             [max(XDAYS) + patient.length_of_stay - 1 | patient in patients]));
set of int: SHIFTS = 1..length(shift_types);
set of int: AGE_GROUPS = 1..length(age_groups);
set of int: GENDERS = 1..length(genders);
set of int: PATIENTS = 1..length(patients);
set of int: OCCUPANTS = 1..length(occupants);
set of int: ROOMS = 1..length(rooms);
set of int: SURGEONS = 1..length(surgeons);
set of int: OPERATING_THEATERS = 1..length(operating_theaters);
set of int: NURSES = 1..length(nurses);
set of int: SKILL_LEVELS = 0..(skill_levels - 1);
array[PATIENTS] of var XDAYS: admission_days;
array[PATIENTS] of var ROOMS: room_assignments;
array[PATIENTS] of var OPERATING_THEATERS: ot_assignments;
int: NO_NURSE_ASSIGNMENT = min(NURSES) - 1;
array[ROOMS, XXDAYS, SHIFTS] of var NO_NURSE_ASSIGNMENT..max(NURSES): nurse_assignments;
constraint forall(r in ROOMS, d in XXDAYS, s in SHIFTS)(
  if d in DAYS
  then nurse_assignments[r, d, s] != NO_NURSE_ASSIGNMENT
  else nurse_assignments[r, d, s] = NO_NURSE_ASSIGNMENT
  endif
);
function Room: room(Occupant: occupant) =
  [room | room in rooms where occupant.room_id == room.id][1];
function Surgeon: surgeon(Patient: patient) =
  [surgeon | surgeon in surgeons where patient.surgeon_id == surgeon.id][1];
function bool: nurse_is_scheduled(Nurse: nurse, DAYS: d, SHIFTS: s) =
  length(
    function bool: nurse_is_scheduled(Nurse: nurse, DAYS: d, SHIFTS: s) = [working_shift |
     working_shift in nurse.working_shifts
     where working_sh... ] // array too long to display, dimensions: (1)

  ) == 1;
function int: nurse_capacity(Nurse: nurse, DAYS: d, SHIFTS: s) =
  function int: nurse_capacity(Nurse: nurse, DAYS: d, SHIFTS: s) = [working_shift.max_load |
   working_shift in nurse.working_shifts
   where worki... ] // array too long to display, dimensions: (1)
[1];
array[NURSES] of int: max_nurse_capacities =
  array[NURSES] of int: max_nurse_capacities = [max([nurse_capacity(nurse, d, s) | d in DAYS, s in SHIFTS where nurse_is_schedul... ]; // array too long to display, dimensions: (1)

function bool: nurse_is_scheduled_with_reduced_capacity(NURSES: n, DAYS: d, SHIFTS: s) =
  nurse_is_scheduled(nurses[n], d, s) /\ nurse_capacity(nurses[n], d, s) < max_nurse_capacities[n];
function bool: patient_can_have_surgery_on_day(Patient: patient, DAYS: d) =
  patient.surgery_release_day <= d /\
  (not patient.mandatory \/ d <= patient.surgery_due_day);
function bool: patient_can_be_in_hospital_on_day(Patient: patient, DAYS: d) =
  patient.surgery_release_day <= d /\
  (not patient.mandatory \/ d <= patient.surgery_due_day + patient.length_of_stay - 1);
function bool: patient_can_be_in_room_on_day(Patient: patient, Room: room, DAYS: d) =
    patient_can_be_in_hospital_on_day(patient, d) /\
    length([id | id in patient.incompatible_room_ids where room.id == id]) == 0;
int: total_length_of_stays =
  sum(occupant in occupants)(occupant.length_of_stay) +
  sum(patient in patients)(patient.length_of_stay);
predicate cumulatives(
  array[$$E] of var int: s,
  array[$$E] of var int: d,
  array[$$E] of var int: r,
  array[$$E] of var $$M: m,
  array[$$M] of var int: b)
=
  forall(i in index_set(b))(
    let {set of int: J = {j | j in index_set(s) where i in dom(m[j])}}
    in cumulative(
      [s[j] | j in J],
      [d[j] | j in J],
      [if m[j] == i then r[j] else 0 endif | j in J],
      b[i])
  );
constraint forall(p in PATIENTS)(admission_days[p] >= patients[p].surgery_release_day);
constraint forall(p in PATIENTS where patients[p].mandatory)(
  admission_days[p] in DAYS /\ admission_days[p] <= patients[p].surgery_due_day
);
array[ROOMS, XXDAYS, GENDERS] of var 0..total_length_of_stays: gender_counts;
constraint forall(g in GENDERS, gender = genders[g])(
  let {
    list of Occupant: occupant_subset =
      [occupant | occupant in occupants where occupant.gender == gender],
    list of Patient: patient_subset =
      [patient | patient in patients where patient.gender == gender]
  }
  in bin_packing_load(
    [gender_counts[r, d, g] | r in ROOMS, d in XXDAYS],
          [patient | patient in patients where patient.gender = = [(room(occupant).index - 1) * card(XXDAYS) + i |
     occupant in occupant_subset... ] // array too long to display, dimensions: (2)
 ++
          [patient | patient in patients where patient.gender = = [(room_assignments[p] - 1) * card(XXDAYS) + (admission_days[p] - 1) + i |
     pa... ], // array too long to display, dimensions: (3)

    [1 | occupant in occupant_subset, i in 1..occupant.length_of_stay] ++
    [1 | patient in patient_subset, i in 1..patient.length_of_stay]
  )
);
constraint forall(r in ROOMS, d in DAYS)(
  sum(g in GENDERS)(gender_counts[r, d, g] > 0) <= 1
);
constraint forall(p in PATIENTS, room_id in patients[p].incompatible_room_ids, r in ROOMS where rooms[r].id == room_id)(
  room_assignments[p] != r
);
constraint forall(surgeon in surgeons)(
  sum(patient in patients where patient.mandatory /\ patient.surgeon_id == surgeon.id)(patient.surgery_duration) <=
  sum(surgeon.max_surgery_time)
);
constraint forall(p in PATIENTS, patient = patients[p])(
  admission_days[p] in
    ({d | d in DAYS where surgeon(patient).max_surgery_time[d] >= patient.surgery_duration}
    union (XDAYS diff DAYS))
);
constraint forall(surgeon in surgeons)(
  let {
    list of Patient: patient_subset =
      [patient | patient in patients where patient.surgeon_id == surgeon.id],
    int: max_demand = sum(patient in patient_subset)(patient.surgery_duration)
  }
  in cumulative(
    [admission_days[patient.index] | patient in patient_subset] ++
    [d | d in DAYS where surgeon.max_surgery_time[d] < max_demand],
    [1 | patient in patient_subset] ++
    [1 | d in DAYS where surgeon.max_surgery_time[d] < max_demand],
    [patient.surgery_duration | patient in patient_subset] ++
        int: max_demand = [max_demand - surgeon.max_surgery_time[d] |
     d in DAYS
     where surgeon.max... ], // array too long to display, dimensions: (1)

    max_demand
  )
);
constraint
  sum(patient in patients where patient.mandatory)(patient.surgery_duration) <=
  sum(operating_theater in operating_theaters)(sum(operating_theater.availability));
constraint forall(patient in patients)(
  admission_days[patient.index] in
    {d |
     d in DAYS,
     operating_theater in operating_theaters
     where operating_theater.availability[d] >= patient.surgery_duration}
    union (XDAYS diff DAYS)
);
constraint forall(patient in patients)(
  ot_assignments[patient.index] in
    {operating_theater.index |
     operating_theater in operating_theaters
     where exists(d in DAYS)(
       patient_can_have_surgery_on_day(patient, d) /\ operating_theater.availability[d] >= patient.surgery_duration
    )}
);
constraint let {
    int: max_demand = sum(patient in patients)(patient.surgery_duration)
  }
  in cumulatives(
    admission_days ++ [d | ot in OPERATING_THEATERS, d in DAYS],
    [1 | p in PATIENTS] ++ [1 | ot in OPERATING_THEATERS, d in DAYS],
    [patient.surgery_duration | patient in patients] ++
        int: max_demand = [max_demand - operating_theater.availability[d] |
    operating_theater in operat... ], // array too long to display, dimensions: (2)

    ot_assignments ++ [ot | ot in OPERATING_THEATERS, d in DAYS],
    [max_demand | operating_theater in operating_theaters],
  );
constraint let {
    int: max_demand = card(OCCUPANTS) + card(PATIENTS)
  }
  in cumulatives(
    [1 | occupant in occupants] ++ admission_days ++ [1 | room in rooms],
    [occupant.length_of_stay | occupant in occupants] ++
    [patient.length_of_stay | patient in patients] ++
    [card(DAYS) | room in rooms],
    [1 | occupant in occupants] ++ [1 | patient in patients] ++ [max_demand - room.capacity | room in rooms],
    [room(occupant).index | occupant in occupants] ++ room_assignments ++ [room.index | room in rooms],
    [max_demand | room in rooms],
  );
constraint forall(d in DAYS, s in SHIFTS)(
  let {set of int: available_nurses = {nurse.index | nurse in nurses where nurse_is_scheduled(nurse, d, s)}}
  in forall(r in ROOMS)(nurse_assignments[r, d, s] in available_nurses)
);
array[ROOMS, XXDAYS, AGE_GROUPS] of var 0..total_length_of_stays: room_occupations;
constraint forall(a in AGE_GROUPS)(
  let {
    list of Occupant: occupant_subset =
      [occupant | occupant in occupants where occupant.age_group == age_groups[a]],
    list of Patient: patient_subset =
      [patient | patient in patients where patient.age_group == age_groups[a]]
  }
  in bin_packing_load(
    [room_occupations[r, d, a] | r in ROOMS, d in XXDAYS],
          [patient | patient in patients where patient.age_group = = [(room(occupant).index - 1) * card(XXDAYS) + i |
     occupant in occupant_subset... ] // array too long to display, dimensions: (2)
 ++
          [patient | patient in patients where patient.age_group = = [(room_assignments[p] - 1) * card(XXDAYS) + (admission_days[p] - 1) + i |
     pa... ], // array too long to display, dimensions: (3)

    [1 | occupant in occupant_subset, i in 1..occupant.length_of_stay] ++
    [1 | patient in patient_subset, i in 1..patient.length_of_stay]
  )
);
array[ROOMS, DAYS, AGE_GROUPS] of var bool: room_has_age_group =
  array3d(
    ROOMS, DAYS, AGE_GROUPS,
    [room_occupations[r, d, a] > 0 | r in ROOMS, d in DAYS, a in AGE_GROUPS]
  );
function var int: min_age_group(ROOMS: r, DAYS: d, int: n, var int: fallback) =
  let {int: b = min(AGE_GROUPS)}
  in if n == 1
  then (if room_has_age_group[r, d, b] then b else fallback endif)
  elseif n == 2
  then (if room_has_age_group[r, d, b] then b elseif room_has_age_group[r, d, b + 1] then b + 1 else fallback endif)
  elseif n == 3
  then (if room_has_age_group[r, d, b]then b elseif room_has_age_group[r, d, b + 1] then b + 1 elseif room_has_age_group[r, d, b + 2] then b + 2 else fallback endif)
  else min_age_group(r, d, n - 1, if room_has_age_group[r, d, b + n - 1] then b + n - 1 else fallback endif)
  endif;
function var int: max_age_group(ROOMS: r, DAYS: d, int: n, var int: fallback) =
  let {int: b = min(AGE_GROUPS)}
  in if n == 1
  then (if room_has_age_group[r, d, b] then b else fallback endif)
  elseif n == 2
  then (if room_has_age_group[r, d, b + 1] then b + 1 elseif room_has_age_group[r, d, b] then b else fallback endif)
  elseif n == 3
  then (if room_has_age_group[r, d, b + 2] then b + 2 elseif room_has_age_group[r, d, b + 1] then b + 1 elseif room_has_age_group[r, d, b] then b else fallback endif)
  else (if room_has_age_group[r, d, b + n - 1] then b + n - 1 else max_age_group(r, d, n - 1, fallback) endif)
  endif;
var int: total_age_group_differences =
  if card(AGE_GROUPS) == 1
  then 0
  else sum(r in ROOMS, d in DAYS)(
          max(
            0,
            max_age_group(r, d, card(AGE_GROUPS), min(AGE_GROUPS) - 1) -
            min_age_group(r, d, card(AGE_GROUPS), max(AGE_GROUPS) + 1)
          )
       )
  endif;
array[ROOMS, XXDAYS, SHIFTS, 1..max(SKILL_LEVELS)] of var 0..total_length_of_stays: skill_level_requirements;
constraint forall(l in SKILL_LEVELS where l > 0)(
  bin_packing_load(
    [skill_level_requirements[r, d, s, l] | r in ROOMS, d in XXDAYS, s in SHIFTS],
      if card(AGE_GROUPS) = = [(room(occupant).index - 1) * (card(XXDAYS) * card(SHIFTS)) + i |
     occupant i... ] // array too long to display, dimensions: (2)
 ++
         where occupant.skill_level_required[i] = = [(room_assignments[p] - 1) * (card(XXDAYS) * card(SHIFTS)) +
     (admission_days... ], // array too long to display, dimensions: (3)

    [1 | occupant in occupants, r in occupant.skill_level_required where r == l] ++
    [1 | patient in patients, r in patient.skill_level_required where r == l]
  )
);
array [NURSES] of SKILL_LEVELS: nurse_skill_levels = [nurses[n].skill_level | n in NURSES];
var int: total_skill_level_deviation = sum(r in ROOMS, d in DAYS, s in SHIFTS)(
  let {var int: n = nurse_assignments[r, d, s]}
  in sum(l in SKILL_LEVELS where l > 0)(
    max(0, skill_level_requirements[r, d, s, l] * (l - nurse_skill_levels[n]))
  )
);
var int: total_discontinuity_of_care = (
  sum(occupant in occupants)(
    nvalue(
      [NO_NURSE_ASSIGNMENT] ++
      var int: total_discontinuity_of_care = [nurse_assignments[room(occupant).index, min(DAYS) + d - 1, s] |
       d in 1..o... ] // array too long to display, dimensions: (2)

    )
  ) +
  sum(patient in patients, p = patient.index)(
    nvalue(
      [NO_NURSE_ASSIGNMENT] ++
        sum(patient in patients, p = [nurse_assignments[room_assignments[p], admission_days[p] + i - 1, s] |
       i ... ] // array too long to display, dimensions: (2)

    )
  ) -
  (card(OCCUPANTS) + card(PATIENTS)));
set of int: ROOM_WORKLOAD =
  0..(sum([workload | occupant in occupants, workload in occupant.workload_produced]) +
      sum([workload | patient in patients, workload in patient.workload_produced]));
array[ROOMS, XXDAYS, SHIFTS] of var ROOM_WORKLOAD: room_workloads;
constraint bin_packing_load(
  array1d(room_workloads),
  set of int: ROOM_WORKLOAD = [(room(occupant).index - 1) * (card(XXDAYS) * card(SHIFTS)) + i |
   occupant in ... ] // array too long to display, dimensions: (2)
 ++
  set of int: ROOM_WORKLOAD = [(room_assignments[p] - 1) * (card(XXDAYS) * card(SHIFTS)) + (admission_days[p] -... ], // array too long to display, dimensions: (3)

  [workload | occupant in occupants, workload in occupant.workload_produced] ++
  [workload | patient in patients, workload in patient.workload_produced]
);
var int: total_excess_nurse_workload =
  sum(nurse in nurses, d in DAYS, s in SHIFTS where nurse_is_scheduled(nurse, d, s))(
    max(
      0,
      sum(r in ROOMS)(
        if nurse_assignments[r, d, s] == nurse.index
        then room_workloads[r, d, s]
        else 0
        endif
      ) - nurse_capacity(nurse, d, s)
    )
  );
int: NO_OT_ASSIGNMENT = min(OPERATING_THEATERS) - 1;
array[PATIENTS, DAYS] of var NO_OT_ASSIGNMENT..max(OPERATING_THEATERS): ot_assignment_matrix;
constraint forall(patient in patients, p = patient.index, d in DAYS)(
  ot_assignment_matrix[p, d] =
    if patient.surgery_release_day <= d /\
       (not patient.mandatory \/ d <= patient.surgery_due_day) /\
       patient.surgery_duration <= surgeon(patient).max_surgery_time[d]
    then (if admission_days[p] == d then ot_assignments[p] else NO_OT_ASSIGNMENT endif)
    else NO_OT_ASSIGNMENT
    endif
);
var int: total_opened_ots = (
  sum(d in DAYS)(nvalue([NO_OT_ASSIGNMENT] ++ [ot_assignment_matrix[p, d]| p in PATIENTS])) - card(DAYS)
);
var int: surgeon_transfer = sum(surgeon in surgeons, d in DAYS where surgeon.max_surgery_time[d] > 0)(
  max(
    0,
    nvalue(
      [NO_OT_ASSIGNMENT] ++
      var int: surgeon_transfer = [ot_assignment_matrix[p, d]| p in PATIENTS where patients[p].surgeon_id == surgeo... ] // array too long to display, dimensions: (1)

    ) - 2
  )
);
var int: total_admission_delay = sum(p in PATIENTS)(
    if admission_days[p] > max(DAYS) then 0 else admission_days[p] - patients[p].surgery_release_day endif
);
var int: unscheduled_patients = sum(p in PATIENTS)(admission_days[p] > max(DAYS));
var int: objective = sum(var int: objective = [
  weights.room_mixed_age * total_age_group_differences,
  weights.nurse_eccessi... ] // array too long to display, dimensions: (8)
);
solve :: seq_search(var int: objective = [
  int_search(nurse_assignments, first_fail, indomain_min),
  int_search(admissi... ] // array too long to display, dimensions: (6)
) minimize objective;
output var int: objective = [
  showJSON(
    (
      'room-age-mix': total_age... ]; // array too long to display, dimensions: (17)
