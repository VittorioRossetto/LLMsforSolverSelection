include "diffn_nonstrict_k.mzn";
include "increasing.mzn";
include "lex_chain_lesseq.mzn";
include "seq_precede_chain.mzn";
enum Dimension = {Length, Width, Height};
enum Product;
array[Dimension] of int: shelves;
array[Product, Dimension] of int: product_size;
int: nr_shelves;
enum Shelf = S(1..nr_shelves);
array[Product] of int: nr_products;
enum Item = I(1..sum(nr_products));
array[Item] of Product: product = array1d(Item, [
	p | p in Product, _ in 1..nr_products[p]
]);
array[Item, Shelf, Dimension] of var 0..max(product_size): item_shelve_size ::output;
array[Item] of var Shelf: item_shelving ::output;
array[Item, Shelf, Dimension] of var 0..max(shelves): item_shelve_placement ::output;
var Shelf: last_loaded_shelf ::output = max(item_shelving);
constraint forall(s in Shelf, it in Item, p = product[it]) (
	if item_shelving[it] == s then
		item_shelve_size[it, s, ..] = product_size[p,..]
	else
		item_shelve_size[it, s, ..] = [0, 0, 0]
	endif
);
constraint forall(s in Shelf) (
	diffn_nonstrict_k(item_shelve_placement[.., s, ..], item_shelve_size[.., s, ..])
);
constraint forall(s in Shelf, it in Item, d in Dimension) (
	item_shelve_placement[it, s, d] + item_shelve_size[it, s, d] <= shelves[d]
);
constraint symmetry_breaking_constraint(
	seq_precede_chain(item_shelving)
);
constraint symmetry_breaking_constraint(
	forall(p in Product) (
		increasing([item_shelving[it] | it in Item where product[it] == p])
	)
);
constraint symmetry_breaking_constraint(
	forall(s in Shelf, p in Product) (
		let {
			any: coord =
			  			any: coord = [ (d, i): item_shelve_placement[i, s, d]
			  | i in Item where product[i] == p, ... ]; // array too long to display, dimensions: (3)

		} in lex_chain_lesseq(coord)
	)
);
var int: objective :: output = enum2int(last_loaded_shelf);
solve :: seq_search(var int: objective :: output = [
			int_search(item_shelve_size, first_fail, indomain_min, complete),
			int_sea... ] // array too long to display, dimensions: (8)
)
minimize objective;