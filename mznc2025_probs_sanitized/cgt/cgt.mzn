include "globals.mzn";
enum TRADE = { Buy, Sell };
enum STOCK;
array [int] of record(
  int: date,
  TRADE: trade,
  STOCK: stock,
  int: units,
  int: price
): trades;
int: UNIT_BUY_COUNT = count (x in trades, i in 1..x.units) (x.trade = Buy);
int: UNIT_SELL_COUNT = count (x in trades, i in 1..x.units) (x.trade = Sell);
enum UNIT_BUY = UnitBuy(1..UNIT_BUY_COUNT);
enum UNIT_SELL = UnitSell(1..UNIT_SELL_COUNT);
any: unit_buys = array1d(UNIT_BUY, any: unit_buys = [(date: x.date, stock: x.stock, price: x.price) | x in trades where x.trade = Buy... ] // array too long to display, dimensions: (4)
);
any: unit_sells = array1d(UNIT_SELL, any: unit_sells = [(date: x.date, stock: x.stock, price: x.price) | x in trades where x.trade = Sel... ] // array too long to display, dimensions: (4)
);
array [UNIT_SELL] of var UNIT_BUY: origin;
constraint all_different(origin);
constraint forall (u in UNIT_SELL) (unit_buys[origin[u]].date < unit_sells[u].date);
constraint forall (t in trades where t.trade = Sell) (
  symmetry_breaking_constraint(
    strictly_increasing (u in UNIT_SELL, x = unit_sells[u] where t.date = x.date /\ t.stock = x.stock /\ t.price = x.price) (origin[u])
  )
);
any: discountable_gains = sum (u in UNIT_SELL, sale = unit_sells[u], bought = unit_buys[origin[u]] where sale.date - bought.date >= 12 /\ sale.price > bought.price) (
    sale.price - bought.price
);
any: undiscountable_gains = sum (u in UNIT_SELL, sale = unit_sells[u], bought = unit_buys[origin[u]] where sale.date - bought.date < 12 /\ sale.price > bought.price) (
    sale.price - bought.price
);
any: losses = sum (u in UNIT_SELL, sale = unit_sells[u], bought = unit_buys[origin[u]] where sale.price < bought.price) (
    bought.price - sale.price
);
any: double_capital_gains =
  if undiscountable_gains > losses then
    discountable_gains + 2 * undiscountable_gains - 2 * losses
  elseif losses - undiscountable_gains < discountable_gains then
    discountable_gains - (losses - undiscountable_gains)
  else
    2 * (discountable_gains + undiscountable_gains - losses)
  endif;
var int: objective :: output = double_capital_gains;
solve :: int_search(origin, input_order, indomain_min) minimize objective;