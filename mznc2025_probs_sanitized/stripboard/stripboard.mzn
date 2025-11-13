include "globals.mzn";
int: max_w;
int: max_h;
int: max_links;
enum COMPONENT;
enum PIN;
enum NET;
array [COMPONENT] of 1..max_w: footprint_w;
array [COMPONENT] of 1..max_h: footprint_h;
array [COMPONENT] of set of ORIENTATION: allowed_orientation;
array [PIN] of 1..max_w: pin_dx;
array [PIN] of 1..max_h: pin_dy;
array [PIN] of NET: pin_net;
array [COMPONENT] of set of PIN: pins;
set of int: LINKS = 1..max_links;
array [LINKS] of var 1..full_w: link_x;
array [LINKS] of var -1..max_h: link_y;
array [LINKS] of var 1..max_h: link_length;
enum ORIENTATION = {Upright, Clockwise, UpsideDown, Anticlockwise};
array [COMPONENT] of var 1..max_w: component_x;
array [COMPONENT] of var 1..max_h: component_y;
array [COMPONENT] of var ORIENTATION: component_orientation;
enum PAD = {Dummy} ++ Pin(PIN) ++ LinkStart(LINKS) ++ LinkEnd(LINKS);
enum NET_OR_UNCONNECTED = { Unconnected } ++ Net(NET);
set of PAD: roots = {[Pin(p) | p in PIN where pin_net[p] = n][1] | n in NET} union {LinkEnd(l) | l in LINKS} union {Dummy};
array [PAD] of var PAD: parent;
array [PAD] of var 0..card(PAD) - 1: distance;
array [PAD] of var NET_OR_UNCONNECTED: connection;
array [PAD] of var 0..full_w: pad_x;
array [PAD] of var -1..max_h: pad_y;
int: full_w = max(max_w, max_links);
array [PAD] of var int: pad_xy :: output = [(full_w + 1) * (pad_y[p] + 1) + pad_x[p] | p in PAD];
constraint forall (c in COMPONENT) (component_orientation[c] in allowed_orientation[c]);
constraint let {
  array [int] of var int: x = component_x ++ link_x;
  array [int] of var int: y = component_y ++ link_y;
  array [int] of var bool: u = [c: component_orientation[c] in {Upright, UpsideDown} | c in COMPONENT];
  array [int] of var int: dx = [if u[c] then footprint_w[c] else footprint_h[c] endif | c in COMPONENT] ++ [1 | l in LINKS];
  array [int] of var int: dy = [if u[c] then footprint_h[c] else footprint_w[c] endif | c in COMPONENT] ++ [link_length[l] + 1 | l in LINKS];
  constraint board_w = max (i in index_set(x) where y[i] >= 1) (x[i] + dx[i]) - 1;
  constraint board_h = max (i in index_set(y)) (y[i] + dy[i]) - 1;
} in diffn(x, y, dx, dy);
constraint forall (c in COMPONENT, p in pins[c]) (let {
  array [ORIENTATION] of var int: px = [
    Upright: pin_dx[p] - 1,
    Clockwise: footprint_h[c] - pin_dy[p],
    UpsideDown: footprint_w[c] - pin_dx[p],
    Anticlockwise: pin_dy[p] - 1
  ];
  array [ORIENTATION] of var int: py = [
    Upright: pin_dy[p] - 1,
    Clockwise: pin_dx[p] - 1,
    UpsideDown: footprint_h[c] - pin_dy[p],
    Anticlockwise: footprint_w[c] - pin_dx[p]
  ];
} in pad_x[Pin(p)] = component_x[c] + px[component_orientation[c]] /\
     pad_y[Pin(p)] = component_y[c] + py[component_orientation[c]]);
constraint forall (l in LINKS) (
    pad_x[LinkStart(l)] = link_x[l] /\
    pad_y[LinkStart(l)] = link_y[l] /\
    pad_x[LinkEnd(l)] = link_x[l] /\
    pad_y[LinkEnd(l)] = link_y[l] + link_length[l]
);
constraint pad_x[Dummy] = 0 /\ pad_y[Dummy] = -1;
constraint forall (p, q in PAD where p < q) (
  connection[p] != connection[q] -> (
    pad_xy[p] - pad_xy[q] > 1 \/ pad_xy[p] - pad_xy[q] < 1
  )
);
constraint forall (p in PIN) (
  connection[Pin(p)] = Net(pin_net[p])
);
constraint forall (l in LINKS) (
  connection[LinkStart(l)] = connection[LinkEnd(l)] /\
  (connection[LinkStart(l)] = Unconnected <-> link_y[l] = -1)
);
constraint forall (p in PAD) (
  if p in roots then
    parent[p] = p /\
    distance[p] = 0
  else
    parent[p] != p /\
    distance[p] = distance[parent[p]] + 1
  endif
);
constraint forall (p in PAD) (
  connection[parent[p]] = connection[p] /\
  pad_y[parent[p]] = pad_y[p]
);
constraint disjunctive(
  [min(pad_xy[parent[p]], pad_xy[p]) | p in PAD],
  [abs(pad_xy[parent[p]] - pad_xy[p]) | p in PAD]
);
var 1..max_w: board_w;
var 1..max_h: board_h;
var int: objective = board_w * board_h;
solve :: seq_search(
  [int_search([component_orientation[c], component_x[c], component_y[c]], input_order, indomain_median) | c in reverse(sort_by(COMPONENT, [footprint_w[c] * footprint_h[c] | c in COMPONENT]))] ++
  [int_search([link_x[l], link_y[l], link_length[l]], input_order, indomain_median) | l in LINKS] ++
  [int_search(parent, input_order, indomain_median)]
) minimize objective;
constraint redundant_constraint(all_different(pad_xy));
constraint symmetry_breaking_constraint(
  strictly_decreasing([pad_xy[LinkStart(l)]| l in LINKS])
);
constraint symmetry_breaking_constraint(
  let {
    array [LINKS] of int: unused_x = array1d(LINKS, reverse(1..card(LINKS)));
  } in forall (l in LINKS) (
    connection[LinkStart(l)] = Unconnected ->
    pad_x[LinkStart(l)] = unused_x[l]
  )
);