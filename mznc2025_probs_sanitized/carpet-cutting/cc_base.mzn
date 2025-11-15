include "globals.mzn";
int: roll_wid;
int: max_roll_len;
set of int: Ori = 1..4;
int: n_rm;
set of int: Rooms = 1..n_rm;
array [Rooms] of set of int: rm_rec_ids;
array [Rooms] of set of Ori: rm_ori;
array [Rooms] of int:        rm_max_len;
array [Rooms] of int:        rm_max_wid;
int: n_rm_rec;
set of int: RmRecs = 1..n_rm_rec;
array [RmRecs] of int:      rm_rec_len;
array [RmRecs] of int:      rm_rec_wid;
array [RmRecs, Ori] of int: rm_rec_os_x;
array [RmRecs, Ori] of int: rm_rec_os_y;
array [Rooms] of int: rm_area =
    [ sum(j in rm_rec_ids[i])(rm_rec_len[j] * rm_rec_wid[j]) | i in Rooms ];
array [Rooms] of int: rm_sort =
    array [Rooms] of int: rm_sort = [ 1 + sum(j in Rooms where j != i /\ rm_area[j] > rm_area[i])(1)
        + sum(j ... ]; // array too long to display, dimensions: (1)

int: n_st;
set of int: Stairs = 1..n_st;
array [Stairs] of int: st_len;
array [Stairs] of int: st_wid;
array [Stairs] of int: st_no_steps;
array [Stairs] of int: st_min_steps;
array [Stairs] of int: st_max_breaks;
array [Stairs] of int: st_rec_id_offset =
    [ sum(j in Stairs where j < i)(st_no_steps[j]) | i in Stairs ];
array [Stairs] of set of int: st_rec_ids =
    array [Stairs] of set of int: st_rec_ids = [(st_rec_id_offset[i] + 1)..(st_rec_id_offset[i] + st_no_steps[i])
    | i in Sta... ]; // array too long to display, dimensions: (1)

int: n_st_rec = sum(i in Stairs)(st_no_steps[i]);
set of int: StRecs = 1..n_st_rec;
array [StRecs] of int: st_rec_len =
    [ st_len[i] div st_no_steps[i] | i in Stairs, j in st_rec_ids[i] ];
array [StRecs] of int: st_rec_wid =
    [ st_wid[i] | i in Stairs, j in st_rec_ids[i] ];
int: TotalArea = sum(i in RmRecs)( rm_rec_len[i] * rm_rec_wid[i] )
                + sum(i in Stairs)( st_len[i] * st_wid[i] );
int: MinRollLen = (TotalArea div roll_wid)
        + if TotalArea mod roll_wid > 0 then 1 else 0 endif;
int: TotalLen = sum(i in Rooms)(max(rm_max_len[i], rm_max_wid[i]))
                + sum(i in Stairs)(st_len[i]);
int: MaxRollLen = min(max_roll_len, TotalLen);
set of int: DomX = 0..MaxRollLen;
set of int: DomY = 0..roll_wid;
int: MinRmRecSize = min(i in RmRecs)(min(rm_rec_len[i], rm_rec_wid[i]));
int: MaxRmRecSize = max(i in RmRecs)(max(rm_rec_len[i], rm_rec_wid[i]));
var MinRollLen..MaxRollLen: objective;
array [Rooms] of var DomX: rm_x;
array [Rooms] of var DomY: rm_y;
array [Rooms] of var Ori:  rm_vori;
array [Rooms] of var bool: rm_ori_0_or_180;
array [Rooms] of var bool: rm_ori_0_or_90;
array [RmRecs] of var DomX: rm_rec_x;
array [RmRecs] of var DomY: rm_rec_y;
array [RmRecs] of var MinRmRecSize..MaxRmRecSize: rm_rec_vlen =
    array [RmRecs] of var MinRmRecSize..MaxRmRecSize: rm_rec_vlen = [ rm_rec_wid[i] + (rm_rec_len[i] - rm_rec_wid[i]) * bool2int(rm_ori_0_or_180[j])
... ]; // array too long to display, dimensions: (2)

array [RmRecs] of var MinRmRecSize..MaxRmRecSize: rm_rec_vwid =
    array [RmRecs] of var MinRmRecSize..MaxRmRecSize: rm_rec_vwid = [ rm_rec_len[i] + (rm_rec_wid[i] - rm_rec_len[i]) * bool2int(rm_ori_0_or_180[j])
... ]; // array too long to display, dimensions: (2)

array [StRecs] of var DomX: st_rec_x;
array [StRecs] of var DomY: st_rec_y;
constraint
    forall(i in Rooms)(
        rm_x[i] + rm_max_wid[i] +
            (rm_max_len[i] - rm_max_wid[i]) * bool2int(rm_ori_0_or_180[i])
            <= objective
    /\  rm_y[i] + rm_max_len[i] +
            (rm_max_wid[i] - rm_max_len[i]) * bool2int(rm_ori_0_or_180[i])
            <= roll_wid
    /\  forall(j in rm_rec_ids[i])(
            rm_rec_x[j] + rm_rec_vlen[j] <= objective
        /\  rm_rec_y[j] + rm_rec_vwid[j] <= roll_wid
        )
    );
constraint
    forall(i in Rooms)(
        rm_vori[i] in rm_ori[i]
    /\  (rm_ori_0_or_90[i]  <-> (rm_vori[i] in {1, 2}))
    /\  (rm_ori_0_or_180[i] <-> (rm_vori[i] in {1, 3}))
    );
constraint
    forall(i in Rooms, j in rm_rec_ids[i])(
        (rm_rec_x[j] = rm_x[i] + rm_rec_os_x[j, rm_vori[i]])
    /\  (rm_rec_y[j] = rm_y[i] + rm_rec_os_y[j, rm_vori[i]])
    );
constraint
    forall(i in Stairs, j in st_rec_ids[i])(
        st_rec_x[j] + st_rec_len[j] <= objective
    /\  st_rec_y[j] + st_rec_wid[j] <= roll_wid
    );
constraint symmetry_breaking_constraint(
    forall(i in Stairs, j1 in st_rec_ids[i] where (j1 + 1) in st_rec_ids[i])(
        let {
            int: j2 = j1 + 1
        } in (
            st_rec_y[j1] <= st_rec_y[j2]
        /\  (   st_rec_y[j1] >= st_rec_y[j2]
            ->  st_rec_x[j1] + st_rec_len[j1] <= st_rec_x[j2]
            )
        )
    )
);
constraint
    forall(i in Stairs)(
        let {
            int: os = st_rec_id_offset[i],
            set of int: J = 1..st_no_steps[i],
            array [J] of var bool: last_part
        } in (
            last_part[st_no_steps[i]] = true
        /\  forall(j in J where j != st_no_steps[i])(
                last_part[j] <-> (
                        st_rec_y[j + os] < st_rec_y[j + os + 1]
                    \/  st_rec_x[j + os] + st_rec_len[j] < st_rec_x[j + os + 1]
                )
            )
        /\  forall(j in J)(
                (st_rec_x[j + os] + 2 * st_rec_len[j + os] > objective) -> last_part[j]
            )
        /\  sum(j in J)(bool2int(last_part[j])) <= st_max_breaks[i] + 1
        /\  if st_min_steps[i] > 1 then
                forall(j in 1..(st_min_steps[i]-1))(
                    last_part[j] = false
                )
            /\  forall(j in J where j >= st_min_steps[i])(
                    forall(k in (j - st_min_steps[i] + 1)..(j - 1))(
                        last_part[j] -> not(last_part[k])
                    )
                )
            else
                true
            endif
        )
    );
constraint
    cumulative(
        rm_rec_x ++ st_rec_x,
        rm_rec_vlen ++ st_rec_len,
        rm_rec_vwid ++ st_rec_wid,
        roll_wid
    );
constraint
    cumulative(
        rm_rec_y ++ st_rec_y,
        rm_rec_vwid ++ st_rec_wid,
        rm_rec_vlen ++ st_rec_len,
        objective
    );
constraint
    diffn(
        rm_rec_x ++ st_rec_x,
        rm_rec_y ++ st_rec_y,
        rm_rec_vlen ++ st_rec_len,
        rm_rec_vwid ++ st_rec_wid
    );
solve
    :: search1
    minimize objective;
ann: fix_ori_0_or_180 =
    bool_search(rm_ori_0_or_180, input_order, indomain_max,  complete);
ann: fix_ori_0_or_90 =
    bool_search(rm_ori_0_or_90, input_order, indomain_max,  complete);
ann: fix_yx = int_search(
        [ [rm_y[rm_sort[i]], rm_x[rm_sort[i]]][p] | i in Rooms, p in 1..2]
        ++ [ [st_rec_x[i], st_rec_y[i]][p] | i in StRecs, p in 1..2],
        input_order, indomain_min, complete );
ann: fix_rl = int_search([objective], input_order, indomain_min, complete);
ann: search1 = seq_search(ann: search1 = [
        fix_ori_0_or_180,
        fix_ori_0_or_90,
        fix_yx,
        fix_... ] // array too long to display, dimensions: (4)
);
output ann: search1 = [
    "rm_vori   = ", show(rm_vori), ";\n",
    "rm_x      = ", show(rm_x), ";\n"... ]; // array too long to display, dimensions: (24)
