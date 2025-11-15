include "globals.mzn";
int: W;
int: H;
int: C;
set of int: Colors = 1..C;
set of int: Width  = 1..W;
set of int: Height = 1..H;
array[Width, Height] of var Colors: puzzle;
array[Width, Height] of var {0} union Colors: across = array2d(Width, Height, array[Width, Height] of var {0} union Colors: across = [
    if puzzle[w-1,h] default 0 != puzzle[w,h] then
        puzzle[w, h]
    els... ] // array too long to display, dimensions: (2)
);
array[Width, Height] of var {0} union Colors: down = array2d(Width, Height, array[Width, Height] of var {0} union Colors: down = [
    if puzzle[w,h-1] default 0 != puzzle[w,h] then
        puzzle[w, h]
    els... ] // array too long to display, dimensions: (2)
);
constraint forall(h in Height) (
    global_cardinality(across[.., h],
                       set2array({0} union Colors),
                       [W - C] ++ [1 | c in Colors]
    )
);
constraint forall(w in Width) (
    global_cardinality(down[w, ..],
                       set2array({0} union Colors),
                       [H - C] ++ [1 | c in Colors]
    )
);
array[int] of var int: diff_across = array[int] of var int: diff_across = [
    let {
        array[int] of var int: occurences = global_cardinality(puzzle... ]; // array too long to display, dimensions: (2)

array[int] of var int: diff_down = array[int] of var int: diff_down = [
    let {
        array[int] of var int: occurences = global_cardinality(puzzle... ]; // array too long to display, dimensions: (2)

constraint symmetry_breaking_constraint(value_precede_chain(Colors, array1d(puzzle)));
var int: objective = max(diff_across ++ diff_down);
solve :: int_search(puzzle, first_fail, indomain_min) minimize objective;