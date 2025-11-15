include "globals.mzn";
int: nNewSkillsPerPerson;
int: nTrainingCap = -1;
int: nInterstateCap = -1;
int: nOverseasCap = 5;
int: nMaxJobs;
int: nMinJobs = 0;
bool: allEng = false;
array[int] of string: sSkills;
set of int: SKILLS = index_set(sSkills);
set of int: TRAINING = 1..nNewSkillsPerPerson;
array[int,int] of int: engineer_skills;
array[int] of int: engineer_location;
set of int: ENGS = index_set_1of2(engineer_skills);
array[int,int] of int: jobs;
set of int: JOBS = index_set_1of2(jobs);
array[JOBS] of var ENGS: allocations;
array[ENGS,TRAINING] of var 0..0 union SKILLS: new_skills;
constraint forall(e in ENGS, t in TRAINING)(
let {
  set of int: NEWSKILLS = {s | s in SKILLS where engineer_skills[e,s]==0}
} in
   new_skills[e,t] in 0..0 union NEWSKILLS
);
constraint if allEng then true else forall(i in JOBS)(
let {
  set of int: POSENGS = {e | e in ENGS where engineer_skills[e,jobs[i,1]] == 1},
} in
  (allocations[i] in POSENGS)
  \/ exists(e in ENGS, t in TRAINING)(jobs[i,1] == new_skills[e,t] /\ allocations[i]==e)
) endif;
constraint if nTrainingCap < 0 then true else sum(e in ENGS, t in TRAINING)(
   new_skills[e,t] > 0
)<=nTrainingCap
endif;
constraint
  if nMaxJobs == 0 then true else forall(e in ENGS)(
    sum(i in JOBS)( allocations[i] == e) <= nMaxJobs )
  endif
;
constraint
  if nMinJobs <= 0 then true else forall(e in ENGS)(
    sum(i in JOBS)( allocations[i] == e) >= nMinJobs )
  endif
;
constraint
  if nOverseasCap < 0 then true else forall(e in ENGS)(
    sum(i in JOBS)( (allocations[i] == e) * (jobs[i,5]==1) ) <= nOverseasCap )
  endif
;
constraint
  if nInterstateCap < 0 then true else forall(e in ENGS where engineer_location[e] > 0)(
    sum(i in JOBS)( (allocations[i] == e) * not (ceil(jobs[i,4]/1000) == ceil(engineer_location[e]/1000) ) ) <= nInterstateCap )
  endif;
var int: objective;
constraint objective == sum(e in ENGS, t in TRAINING)(
   new_skills[e,t] > 0
);
solve
::seq_search( constraint objective = = [
    int_search([new_skills[e,t] | t in TRAINING, e in ENGS],input_order, indoma... ] // array too long to display, dimensions: (8)
)
minimize objective;
output constraint objective = = [
	"allocations = array1d(\(JOBS), \(allocations));\n",
	"new_skills = array2d(\(... ]; // array too long to display, dimensions: (6)
