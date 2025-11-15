include "table.mzn";
include "inverse.mzn";
array[1..416, 1..2] of int: neighbours;
array[1..17, 1..3] of int: layout;
array[1..52] of var 1..52: x;
array[1..52] of var 1..52: y;
constraint x[1] == 1;
constraint
	forall(i in 1..51) (
		 table([x[i], x[i+1]], neighbours)
	);
constraint inverse(x, y) :: domain;
constraint
	forall(i in 1..17, j in 1..2) (
		 y[layout[i,j]] < y[layout[i,j+1]]
	);
solve
	:: int_search(x, input_order, indomain_min, complete)
	satisfy;
output [ "x = ", show(x), ";\n" ];
neighbours = array2d(1..416, 1..2, neighbours = neighbours = array2d(1..416, 1..2, neighbours = [1, 2, 1, 13, 1, 15, 1, 26, 1, 28, 1, 39, 1, 41, 1, 52, 2, 1, 2, 3, 2, 14, 2, 16,... ] // array too long to display, dimensions: (30)
 // array too long to display, dimensions: (832)
);