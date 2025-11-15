int: numOperationsInFunction;
int: numDataInFunction;
int: numBlocksInFunction;
int: entryBlockOfFunction;
array[allBlocksInFunction] of set of int: domSetOfBlockInFunction;
array[allBlocksInFunction] of set of int: defEdgesForBlockInFunction;
array[int] of allDataInFunction: statesInFunction;
array[allBlocksInFunction] of int: execFrequencyOfBlockInFunction;
int: numLocations;
int: numMatches;
array[allMatches] of set of int: operationsCoveredByMatch;
array[allMatches] of set of int: dataDefinedByMatch;
array[allMatches] of set of int: dataUsedByMatch;
array[allMatches] of set of int: entryBlockOfMatch;
array[allMatches] of set of int: spannedBlocksInMatch;
array[allMatches] of set of int: consumedBlocksInMatch;
array[allMatches] of int: codeSizeOfMatch;
array[allMatches] of int: latencyOfMatch;
array[allMatches] of bool: applyDefDomUseConstraintForMatch;
set of allMatches: nonCopyMatches;
array[int,int] of int: sameLoc;
array[int,int] of int: inBlockSucc;
array[int,int] of int: inBlock;
array[int,int] of int: locDomain;
array[int,int] of int: funLocDomain;
set of allMatches: Dominated;
int: numLocValues = numLocations + 1;
int: numBlockValues = numBlocksInFunction + 1;
int: blockValueForNull = numBlockValues - 1;
int: locValueForNull = numLocValues - 1;
array[allBlocksInFunctionPlusNull] of set of int:
  domSetOfBlockInFunctionPlusNull =
    array1d( allBlocksInFunctionPlusNull
           , domSetOfBlockInFunction
             ++
             array1d( blockValueForNull..blockValueForNull,
                      [allBlocksInFunction]
                    )
           );
array[allBlocksInFunctionPlusNull] of int: execFrequencyOfBlocksPlusNull =
  array1d( allBlocksInFunctionPlusNull
         , execFrequencyOfBlockInFunction
           ++
           array1d(blockValueForNull..blockValueForNull, [0])
         );
set of int: allOperationsInFunction = 0..numOperationsInFunction-1;
set of int: allDataInFunction = 0..numDataInFunction-1;
set of int: allBlocksInFunction = 0..numBlocksInFunction-1;
set of int: allBlocksInFunctionPlusNull = 0..numBlockValues-1;
set of int: allMatches = 0..numMatches-1;
set of int: allLocValues = 0..numLocValues-1;
array[allDataInFunction] of var allBlocksInFunction: def;
array[allDataInFunction] of var allLocValues: loc;
array[allMatches] of var bool: sel;
array[allMatches] of var allBlocksInFunctionPlusNull: place;
array[allBlocksInFunctionPlusNull] of var allBlocksInFunctionPlusNull: succ;
var 0..sum (m in allMatches diff Dominated)( latencyOfMatch[m]* max ([execFrequencyOfBlocksPlusNull[x] | x in allBlocksInFunction])): objective;
include "circuit.mzn";
include "table.mzn";
constraint
  forall (o in allOperationsInFunction)
  (
    let {
      set of int: mset = { m | m in allMatches diff Dominated
                                    where o in operationsCoveredByMatch[m]
                         }
    }
    in if card(mset) = 1
       then sel[min(mset)]
       else if card(mset) = 2
            then sel[min(mset)] xor sel[max(mset)]
            else sum (m in mset)
                 (
                   bool2int(sel[m])
                 ) = 1
            endif
       endif
  );
constraint
  forall (e in allDataInFunction)
  (
    let {
      set of int: mset = { m | m in allMatches diff Dominated
                                    where e in dataDefinedByMatch[m]
                         }
    }
    in if card(mset) = 1
       then sel[min(mset)]
       else if card(mset) = 2
            then sel[min(mset)] xor sel[max(mset)]
            else sum (m in mset)
                 (
                   bool2int(sel[m])
                 ) = 1
            endif
       endif
  );
constraint redundant_constraint (
  sum (m in allMatches diff Dominated)
  (
    card(dataDefinedByMatch[m]) * bool2int(sel[m])
  ) = numDataInFunction
);
constraint
  forall (m in allMatches)
  (
    sel[m] <-> place[m] != blockValueForNull
  );
constraint
  forall (m in allMatches diff Dominated)
  (
    forall (r in entryBlockOfMatch[m])
    (
      place[m] in {r, blockValueForNull}
    )
  );
constraint
  forall (m in allMatches diff Dominated)
  (
    forall (e in dataDefinedByMatch[m])
    (
      if (card(spannedBlocksInMatch[m]) > 0)
      then
        sel[m] -> def[e] in spannedBlocksInMatch[m]
      else
        sel[m] -> def[e] = place[m]
      endif
    )
  );
constraint
  forall (m, mm in allMatches, b in consumedBlocksInMatch[m])
  (
    sel[m] -> place[mm] != b
  );
constraint
  forall (e in allDataInFunction, l in allBlocksInFunction)
  (
    def[e] = l
    ->
    exists (m in allMatches diff Dominated)
    (
      place[m] = l \/ (sel[m] /\ l in spannedBlocksInMatch[m])
    )
  );
constraint
  forall (l in allBlocksInFunction)
  (
    forall (e in defEdgesForBlockInFunction[l])
    (
      def[e] = l
    )
  );
int: DomRelSize =
  sum (l in allBlocksInFunction)
  (
    card(domSetOfBlockInFunction[l])
  ) + numBlocksInFunction;
array[1..DomRelSize, 1..2] of allBlocksInFunctionPlusNull: DomRel =
  array2d(1..DomRelSize, 1..2,
    array[1..DomRelSize, 1..2] of allBlocksInFunctionPlusNull: DomRel = [ if k=1 then i else j endif | i in allBlocksInFunctionPlusNull,
                ... ] // array too long to display, dimensions: (3)
);
constraint
  forall ( m in allMatches diff Dominated, e in dataUsedByMatch[m]
                where applyDefDomUseConstraintForMatch[m]
         )
  (
    table([place[m], def[e]], DomRel)
  );
constraint
  circuit(succ) :: domain;
constraint
  succ[blockValueForNull] = entryBlockOfFunction;
constraint
  forall (e in statesInFunction)
  (
    loc[e] = locValueForNull
  );
constraint
  objective = sum (m in allMatches diff Dominated)
         ( latencyOfMatch[m]
         * execFrequencyOfBlocksPlusNull[place[m]]
         );
constraint
  forall(i in index_set_1of2(sameLoc))(
    let {int: m = sameLoc[i,1],
         int: p = sameLoc[i,2],
         int: q = sameLoc[i,3]} in
	sel[m] -> loc[p] = loc[q]
  );
constraint
  forall(i in index_set_1of2(inBlockSucc))(
    let {int: m = inBlockSucc[i,1],
         int: p = inBlockSucc[i,2],
         int: q = inBlockSucc[i,3]} in
	place[m] in {p,blockValueForNull} /\
	(sel[m] -> succ[p] = q)
  );
constraint
  forall(i in index_set_1of2(inBlock))(
    let {int: m = inBlock[i,1],
         int: p = inBlock[i,2]} in
	place[m] in {p,blockValueForNull}
  );
constraint
  forall(i in index_set_1of2(locDomain))(
    let {int: m = locDomain[i,1],
         int: l = locDomain[i,2]} in
	sel[m] -> loc[l] in locDomain[i,3]..locDomain[i,4]
  );
constraint
  forall(i in index_set_1of2(funLocDomain))(
    let {int: l = funLocDomain[i,1]} in
	loc[l] in funLocDomain[i,2]..funLocDomain[i,3]
  );
constraint redundant_constraint(
  forall(i in index_set_1of2(inBlockSucc), j in index_set_1of2(inBlockSucc) where i<j)(
    let {int: mi = inBlockSucc[i,1],
         int: pi = inBlockSucc[i,2],
         int: qi = inBlockSucc[i,3],
	 int: mj = inBlockSucc[j,1],
         int: pj = inBlockSucc[j,2],
         int: qj = inBlockSucc[j,3]} in
	(pi=pj xor qi=qj) -> (not sel[mi] \/ not sel[mj])
  )
);
constraint redundant_constraint(
  forall(i in index_set_1of2(sameLoc))(
    let {int: m1 = sameLoc[i,1],
         int: p = sameLoc[i,2],
         int: q = sameLoc[i,3]} in
    forall(j1 in index_set_1of2(locDomain) where locDomain[j1,2]=p)(
      forall(j2 in index_set_1of2(locDomain) where locDomain[j2,1]=locDomain[j1,1] /\
                                                   locDomain[j2,2]=q)(
        let {int: m2 = locDomain[j1,1]} in
	    card((locDomain[j1,3]..locDomain[j1,4]) intersect (locDomain[j2,3]..locDomain[j2,4]))=0 ->
            not sel[m1] \/ not sel[m2]
      )
    )
  )
);
constraint redundant_constraint (
  forall(i in index_set_1of2(funLocDomain))(
    let {int: lo = funLocDomain[i,2],
         int: hi = funLocDomain[i,3]} in
	lo<=31 -> hi>=31 \/ hi=0
  ) /\
  forall(i in index_set_1of2(locDomain))(
    let {int: lo = locDomain[i,3],
         int: hi = locDomain[i,4]} in
	lo<=31 -> hi>=31 \/ hi=0
  ) ->
  forall(i in allDataInFunction)(loc[i] in {0} union (31..numLocValues-1))
);
test is_dominated(int: m2) =
  let {set of allMatches: exclude = {sameLoc[i,1] | i in index_set_1of2(sameLoc)}
                              union {inBlock[i,1] | i in index_set_1of2(inBlock)}
			      union {inBlockSucc[i,1] | i in index_set_1of2(inBlockSucc)}} in
  exists(m1 in allMatches where not (m1=m2 \/ m1 in exclude \/ m2 in exclude))(
    ((latencyOfMatch[m1] < latencyOfMatch[m2]) \/
     (latencyOfMatch[m1] = latencyOfMatch[m2] /\ m1 < m2)) /\
    operationsCoveredByMatch[m1] = operationsCoveredByMatch[m2] /\
    dataDefinedByMatch[m1] = dataDefinedByMatch[m2] /\
    dataUsedByMatch[m1] = dataUsedByMatch[m2] /\
    entryBlockOfMatch[m1] = entryBlockOfMatch[m2] /\
    spannedBlocksInMatch[m1] = spannedBlocksInMatch[m2] /\
    applyDefDomUseConstraintForMatch[m1] = applyDefDomUseConstraintForMatch[m2] /\
    forall(i in index_set_1of2(locDomain) where locDomain[i,1] = m1)(
      exists(j in index_set_1of2(locDomain)
             where locDomain[j,1] = m2
                /\ locDomain[j,2] = locDomain[i,2])(
	(locDomain[j,3]..locDomain[j,4]) subset (locDomain[i,3]..locDomain[i,4])
      )
    )
  );
Dominated = {m | m in allMatches where is_dominated(m)};
constraint
  forall(d in Dominated)(not sel[d]);
solve ::
  seq_search(
    Dominated = [ bool_search([ sel[m] | m in nonCopyMatches diff D... ] // array too long to display, dimensions: (28)

  )
minimize objective;
output [ "sel = array1d(\(allMatches), \(sel));\n"
       , "def = array1d(\(allDataInFunction), \(def));\n"
       , "loc = array1d(\(allDataInFunction), \(loc));\n"
       , "place = array1d(\(allMatches), \(place));\n"
       , "succ = array1d(\(allBlocksInFunctionPlusNull), \(succ));\n"
       , "objective = \(objective);\n"
       ];