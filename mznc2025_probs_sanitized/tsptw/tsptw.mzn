include "globals.mzn";
int: numLocations;
set of int: Locations = 1..numLocations;
array[Locations] of int: early;
array[Locations] of int: late;
array[Locations, Locations] of int: duration;
int: depot = 1;
array[Locations] of var Locations: pred;
array[Locations] of var 0..max(duration): durToPred = [
    duration[l, pred[l]] | l in Locations];
array[Locations] of var 0..max(late): arrival;
array[Locations] of var min(early)..max(late): departure = [0] ++ [
    max(arrival[l], early[l]) | l in 2..numLocations];
array[Locations] of var 0..max(late): departurePred = [
    departure[pred[l]] | l in Locations];
var int: objective = arrival[depot];
constraint forall (l in Locations) (
    arrival[l] = departurePred[l] + durToPred[l]);
constraint circuit(pred) :: domain;
constraint forall (l in Locations) (departure[l] <= late[l]);
solve :: seq_search(constraint forall (l in Locations) (departure[l] < = [int_search(pred, first_fail, indomain_min),
                     int_search([arr... ] // array too long to display, dimensions: (6)
)
         minimize objective;