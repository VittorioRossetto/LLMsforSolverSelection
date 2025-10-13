include "globals.mzn";
enum Run = { BASELINE, BEST, WORST};
Run: run::output;
array[int] of int: c::output;
int: n::output;
int: ub::output;
float: f::output;
array[int] of set of int: s = [ {0, ci} | ci in reverse(sort(c)) ];
int: u::output = sum([ max(si) | si in s]);
int: k::output = round(f * u);
set of int: L = 1..(1+ceil(log2(length(c))));
array[L,index_set(s)] of var set of min(si in s, a in si)(a)..k: x;
constraint x[1,..] = s;
any: y_pairs = [
  let {
    int: pairs = ceil(length(c) / (2^i));
    array[1..pairs,1..2] of var 1..(2*pairs): y;
  } in (pairs: pairs, y: y)
| i in L diff {max(L)}];
any: y_vars :: output = [y_j | z_i in y_pairs, y_j in z_i.y];
constraint forall(i in L diff {max(L)})(
  let {
    int: pairs = y_pairs[i].pairs;
    int: next_pow2 = 2^(pairs-1);
    any: y = y_pairs[i].y;
    constraint alldifferent(y);
    constraint strictly_increasing(j in 1..pairs)(y[j,1]);
    constraint if run = BASELINE then y = array2d(1..pairs, 1..2, [ z | z in 1..(pairs*2) ]) endif ;
  } in
  forall(j in index_set(s))(
  if j <= pairs then
    let {
      var int: ja = y[j,1];
      var int: jb = y[j,2];
      constraint ja < jb;
      var set of int: a = x[i,ja];
      var set of int: b = x[i,jb];
    } in
    (x[i+1,j] = { (ai in a) * (bi in b) * (ai + bi) | ai in ub(a), bi in ub(b) where ai+bi <= k })
  elseif j <= next_pow2 then
    x[i+1,j] = {0}
  else
    x[i+1,j] = {}
  endif
));
var int: obj ::output =  sum(xi in x)(card(xi));
var int: objective = if run = WORST then -obj else obj endif;
solve :: seq_search([
  set_search(x, input_order, indomain_min),
  int_search(y_vars, smallest, indomain_min)
]) minimize objective;