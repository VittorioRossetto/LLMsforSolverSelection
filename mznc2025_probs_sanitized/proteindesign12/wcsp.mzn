include "globals.mzn";
int: num_variables;
int:max_domain;
array[1..num_variables] of int: domains;
int: top;
int: num_constraints1;
int: max_constraints1;
int: max_costs1;
int: num_constraints2;
int: max_constraints2;
int: max_costs2;
array[1..num_constraints1] of int: func1x;
array[1..num_constraints1] of int: num_tuples1;
array[1..num_constraints1] of int: cum_tuples1;
array[1..max_constraints1] of int: costs1;
array[1..num_constraints2] of int: func2x;
array[1..num_constraints2] of int: func2y;
array[1..num_constraints2] of int: num_tuples2;
array[1..num_constraints2] of int: cum_tuples2;
array[1..max_constraints2] of int: costs2;
array[1..num_variables] of var 0..max_domain: p;
array[1..num_constraints1] of var 0..max_costs1: ocosts1;
array[1..num_constraints2] of var 0..max_costs2: ocosts2;
var 0..(top-1): objective;
solve
	:: seq_search(	:: [
		int_search(p, first_fail, indomain_min, complete),
		int_search(ocosts1, firs... ] // array too long to display, dimensions: (16)
)
	minimize objective;
constraint
   	objective = sum(j in 1..num_constraints1)( ocosts1[j] )
   		+ sum(j in 1..num_constraints2) ( ocosts2[j] );
constraint
   forall(j in 1..num_variables) (
       p[j] < domains[j]
   );
constraint
   	forall(j in 1..num_constraints1) (
       	table(
	   		[ocosts1[j], p[func1x[j]]],
			array2d(1..num_tuples1[j],1..2,
				[costs1[u] | u in (2*cum_tuples1[j]+1)..(2*cum_tuples1[j]+num_tuples1[j]*2)]
			)
		)
   	);
constraint
	forall(j in 1..num_constraints2) (
       	table(
			[ocosts2[j], p[func2x[j]], p[func2y[j]]],
			array2d(1..num_tuples2[j],1..3,
				[costs2[u] | u in (3*cum_tuples2[j]+1)..(3*cum_tuples2[j]+num_tuples2[j]*3)]
			)
		)
   	);
output
   	objective = [
    "p = \(p);\n",
    "ocosts1 = \(ocosts1);\n",
    "ocosts2 = \(ocosts2);\n"... ]; // array too long to display, dimensions: (4)
