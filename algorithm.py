import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pytest

import warnings
# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

def generate_price_simulation(
    first_day_price:float,
    number_of_days_to_simulate:int,
    
    upper_price_bound: float,
    lower_price_bound: float
):
    """
    Generate a simulated price sequence, within the range of
    'upper_price_bound' and 'lower_price_bound'.

    Args:
    - first_day_price (float): The initial price on the first day.
    - number_of_days_to_simulate (int): The number of days in simulation.
    - upper_price_bound (float): The upper price bound for simulation.
    - lower_price_bound (float): The lower price bound for simulation.

    Returns:
    - List[float]: A list of simulated prices for each day.
    """
    prices_in_set_period = [first_day_price]
    for day in range(number_of_days_to_simulate):
        big_step_percent = np.random.uniform(0,0.2)
        small_step_percent = np.random.uniform(0,0.03)
        percent = np.random.choice(
            [
                big_step_percent, big_step_percent, big_step_percent, 
                small_step_percent, small_step_percent
            ]
        )
        choice = np.random.choice([True,False])
        if (choice and prices_in_set_period[day] <= upper_price_bound) or (prices_in_set_period[day] <= lower_price_bound):
            next_day_price = prices_in_set_period[day]*(1+percent)
        else:
            next_day_price = prices_in_set_period[day]*(1-percent)
        prices_in_set_period.append(next_day_price)    
    return prices_in_set_period

def create_price_for_orders_in_a_group (
    price_to_descend_from: float,
    group_x_count: int,
    group_x_slot_price_descending: float
) -> list:
    """
    Create a list of slot prices for orders in a specific group. 
    It starts with the 'price_to_descend_from' and calculates descending prices for
    each slot based on the 'group_x_slot_price_descending' rate. The number of slots
    in the group is determined by 'group_x_count', and the resulting slot prices are
    returned as a list.

    Args:
    - price_to_descend_from (float): The initial price for which slot prices will descend from.
    - group_x_count (int): The number of slots in the current group.
    - group_x_slot_price_descending (float): The rate of price descending for the group.

    Returns:
    - List[float]: A list of slot prices for the group.
    
    """
    
    slot_prices_for_group_x = []
    price_for_slot = price_to_descend_from-price_to_descend_from*group_x_slot_price_descending
    
    for group_entity in range(group_x_count):
        slot_prices_for_group_x.append(price_for_slot)
        price_for_slot = price_for_slot-price_for_slot*group_x_slot_price_descending
    
    return slot_prices_for_group_x


def create_an_order (
    db_with_all_share_orders: pd.DataFrame,
    price_for_slot:float,
    initial_price: float,
    group_number: int,
    action: str = "buy",
) -> None:    
    """
    Create a trading order and add it to the DataFrame of all share orders. The order's details, including
    the slot price, initial price, group number, and action, are recorded in the DataFrame.

    Args:
    - db_with_all_share_orders (pd.DataFrame): DataFrame containing all share orders.
    - price_for_slot (float): The price for the trading slot.
    - initial_price (float): The initial price for the order.
    - group_number (int): The group number for the order.
    - action (str, optional): The action for the order, either "buy" or "sell". Defaults to "buy".
    """
    db_with_all_share_orders.loc[
                len(db_with_all_share_orders.index)
            ]=[
                price_for_slot,
                action,
                "on_the_market",
                initial_price,
                0,
                group_number
            ]

def closing_listed_out_order (
    order_index: int|list,
    db_with_all_share_orders:pd.DataFrame,
    end_trade_price:float,
    status:str = "complited"
) -> None:
    """
    Close a listed-out trading order and update its status and end_trade price.

    Args:
    - order_index (int|list): The index of the order to close.
    - db_with_all_share_orders (pd.DataFrame): DataFrame containing all share orders.
    - end_trade_price (float): The price in which order was traded.
    - status (str, optional): The new status for the order, either "completed" or "aborted". Defaults to "completed".
    """
    db_with_all_share_orders.loc[
        order_index,
        "status"
    ] = status
    end_trade_price = float(end_trade_price)
    db_with_all_share_orders.loc[
        order_index,
        "end_trade_price"
    ] = end_trade_price
            
def change_trade_orders(
    db_with_all_share_orders: pd.DataFrame,
    SELL_PROFIT_MARGIN: float,
    price_in_day: float,
    list_of_orders: list,
    action: str
) -> None:
    """
    Adjust trading orders based on the current price and action.

    This function simulates trading actions for a list of orders based on the
    provided action (either "buy" or "sell") and the current price for the day.

    Args:
    - db_with_all_share_orders (pd.DataFrame): DataFrame containing all share orders.
    - SELL_PROFIT_MARGIN (float): The profit margin for selling.
    - price_in_day (float): The price on the current day.
    - list_of_orders (list): List of order indices to change.
    - action (str): The trading action, either "buy" or "sell".
    """

    for order_index in list_of_orders:
        price = db_with_all_share_orders.loc[order_index, "price"]
        if (
            (
                action == "buy" 
                and 
                price >= price_in_day) 
            or 
            (
                action == "sell" 
                and 
                price <= price_in_day
            )
        ):
            closing_listed_out_order(
                order_index=order_index,
                db_with_all_share_orders=db_with_all_share_orders,
                end_trade_price=price
            )
            if action == "buy":
                price_to_trade = price * (1 + SELL_PROFIT_MARGIN)
                opposite_trade_action = "sell"
            elif action == "sell":
                price_to_trade = price / (1 + SELL_PROFIT_MARGIN)
                opposite_trade_action = "buy"
                
                # Can create an order after selling too,
                # however it have worse controle for the shares 
                return None
            
            create_an_order(
                db_with_all_share_orders=db_with_all_share_orders,
                price_for_slot=price_to_trade,
                initial_price=db_with_all_share_orders.loc[order_index, "net_anchor_price"],
                group_number=db_with_all_share_orders.loc[order_index, "group_number"],
                action=opposite_trade_action
            )
                
def execute_day_of_trading_simulation (
    db_with_all_share_orders: pd.DataFrame,
    SELL_PROFIT_MARGIN: float,
    price_in_day: float
):
    """
    Simulate a day of trading actions based on the current price.

    This function simulates trading actions for buy and sell orders based on the
    provided current price for the day. It adjusts orders according to the given
    selling profit margin and updates the status of completed orders.

    Args:
    - db_with_all_share_orders (pd.DataFrame): DataFrame containing all share orders.
    - SELL_PROFIT_MARGIN (float): The profit margin for selling.
    - price_in_day (float): The price on the current day.
    """
    db_grouped = db_with_all_share_orders.groupby(["action", "status"]).groups
    
    if ('buy', 'on_the_market') in db_grouped.keys():
        # Simulate process, when shares being bought out
        change_trade_orders (
            db_with_all_share_orders= db_with_all_share_orders,
            SELL_PROFIT_MARGIN= SELL_PROFIT_MARGIN,
            price_in_day= price_in_day,
            list_of_orders= db_grouped[('buy', 'on_the_market')],
            action="buy"
        )
        
    if ('sell', 'on_the_market') in db_grouped.keys():
        # Simulate process, when shares being sold out        
        change_trade_orders (
            db_with_all_share_orders= db_with_all_share_orders,
            SELL_PROFIT_MARGIN= SELL_PROFIT_MARGIN,
            price_in_day= price_in_day,
            list_of_orders= db_grouped[('sell', 'on_the_market')],
            action="sell"
        )    
    

def abort_buy_orders(
    net_anchor_price:float, 
    price_in_day:float,
    db_with_all_share_orders:pd.DataFrame, 
):
    """
    Abort and update buy orders based on market conditions.

    This function checks for active buy orders in the DataFrame and aborts them.
    It updates the status of aborted orders and adjusts the `net_anchor_price`.

    Args:
    - net_anchor_price (float): The anchor price used for trading strategy.
    - price_in_day (float): The price on the current day.
    - db_with_all_share_orders (pd.DataFrame): DataFrame containing all share orders.

    Returns:
    - float: The updated `net_anchor_price` after aborting buy orders.
    """
    shares_grouped_to_abort = db_with_all_share_orders.groupby(["action", "status"]).groups
    if ('buy', 'on_the_market') in shares_grouped_to_abort:
        closing_listed_out_order(
            order_index=shares_grouped_to_abort[('buy', 'on_the_market')],
            db_with_all_share_orders=db_with_all_share_orders,
            end_trade_price=0,
            status="aborted"
        )
    net_anchor_price = np.average([net_anchor_price, price_in_day])
    return net_anchor_price

def abort_sell_orders(
    net_anchor_price:float, 
    price_in_day:float,
    db_with_all_share_orders:pd.DataFrame, 
):
    """
    Abort and update sell orders based on market conditions.

    This function checks for active buy orders in the DataFrame and aborts them.
    It updates the status of aborted orders and adjusts the `net_anchor_price`.

    Args:
    - net_anchor_price (float): The anchor price used for trading strategy.
    - price_in_day (float): The price on the current day.
    - db_with_all_share_orders (pd.DataFrame): DataFrame containing all share orders.

    Returns:
    - float: The updated `net_anchor_price` after aborting sell orders.
    """
    shares_grouped_to_abort = db_with_all_share_orders.groupby(["action", "status"]).groups
    if ('sell', 'on_the_market') in shares_grouped_to_abort:
        closing_listed_out_order(
            order_index=shares_grouped_to_abort[('sell', 'on_the_market')],
            db_with_all_share_orders=db_with_all_share_orders,
            end_trade_price=price_in_day,
            status="aborted"
        )
    net_anchor_price = np.average([net_anchor_price, price_in_day])
    return net_anchor_price

def manage_strategy_due_to_thresholds (
    fall_percent:float,
    net_anchor_price:float,
    price_in_day:float,
    db_with_all_share_orders:pd.DataFrame,
    big_fall_count:int,
    
    STRATEGY_UPPER_THRESHOLD:float,
    STRATEGY_LOWER_THRESHOLD:float,  
):
    """
    Manage the trading strategy based on specified thresholds and market conditions.

    This function assesses the current market conditions, including the percentage
    fall in prices (`fall_percent`), and manages the trading strategy accordingly.
    It may abort buy orders or sell orders if the thresholds are breached and
    update the `net_anchor_price` accordingly.

    Args:
    - fall_percent (float): The percentage fall in prices.
    - net_anchor_price (float): The anchor price used for trading strategy.
    - price_in_day (float): The price on the current day.
    - db_with_all_share_orders (pd.DataFrame): DataFrame containing all share orders.
    - big_fall_count (int): A count of significant price falls observed.
    - STRATEGY_UPPER_THRESHOLD (float): The upper threshold for the trading strategy.
    - STRATEGY_LOWER_THRESHOLD (float): The lower threshold for the trading strategy.

    Returns:
    - Tuple[float, int]: A tuple containing the updated `net_anchor_price` and
      the updated `big_fall_count` after managing the strategy.
    """
    if (
        fall_percent < STRATEGY_UPPER_THRESHOLD
        or
        fall_percent > STRATEGY_LOWER_THRESHOLD
    ):
        if fall_percent < STRATEGY_UPPER_THRESHOLD:
            net_anchor_price = abort_buy_orders(
                net_anchor_price=net_anchor_price,
                price_in_day=price_in_day,
                db_with_all_share_orders=db_with_all_share_orders
            )
        elif fall_percent > STRATEGY_LOWER_THRESHOLD:
            print("BigFall!", fall_percent)
            big_fall_count += 1
            net_anchor_price = abort_sell_orders(
                net_anchor_price=net_anchor_price,
                price_in_day=price_in_day,
                db_with_all_share_orders=db_with_all_share_orders
            )
    return net_anchor_price, big_fall_count

def create_group_dict(
    *args:list,
):
    """
    Create a dictionary representing groups with specified parameters.

    This function takes a variable number of arguments, where each argument is a list
    containing information about a group. Each list should contain next information in order:
    - 0: count;
    - 1: slot_price_descending_constant;
    - 2: fall_percent_start;
    - 3: fall_percent_end.
    
    After that, function generates a dictionary for each list and put them into one dictionary
    
    Args:
        - *args (list): Variable number of lists, each containing group information: [
            - 0: count (int): Number of instances in group
            - 1: slot_price_descending_constant (float): Constant value that used in formula to determine price descending for group
            - 2: fall_percent_start (float): Percent value, that indicates start of the group percent range
            - 3: fall_percent_end (float): Percent value, that indicates end of the group percent range
        ]

    Returns:
        dict: A dictionary representing groups with their parameters.
    """
    group_dict = {}
    for group_number, group in enumerate(args):
        group_dict.update(
            {
                f"group{group_number}":{
                    "group_number": group_number,
                    "count": group[0],
                    "slot_price_descending_constant": group[1],
                    "fall_percent_start": group[2],
                    "fall_percent_end": group[3]
                }
            }
        )
       
    return group_dict



def build_net_of_groups (
    fall_percent:float,
    net_anchor_price:float,
    price_in_day:float,
    db_with_all_share_orders:pd.DataFrame,
    
    group_dictionary:dict[dict],
    
    USE_PRICE_ADJUSTMENT:bool = True
) -> None:
    """
    Build a network of trading groups based on specified criteria.

    This function calculates and executes trading strategies for different groups based on
    the provided parameters and group information. It updates the `db_with_all_share_orders`
    DataFrame with trading actions for each group.

    Args:
        - fall_percent (float): The percentage fall in price for the current day.
        - net_anchor_price (float): The anchor price for the trading strategy.
        - price_in_day (float): The current day's price.
        - db_with_all_share_orders (pd.DataFrame): A DataFrame containing all share orders. Is used to store new trading actions.
        - group_dictionary (dict): A dictionary containing information about different groups. Keys are group names, and values are dictionaries with group details: {
            - count (int): Number of instances in group
            - slot_price_descending_constant (float): Constant value that used in formula to determine price descending for group
            - fall_percent_start (float): Percent value, that indicates start of the group percent range
            - fall_percent_end (float): Percent value, that indicates end of the group percent range
        }
        - USE_PRICE_ADJUSTMENT (bool, optional): Whether to use price adjustment. It activates if group_dictionary containce more then 1 group and fall_percent bigger then 2nd group fall_percent_start. Defaults to True.

    Returns:
       - None
    """
    for group in group_dictionary:
        if (
            fall_percent > group_dictionary[group]["fall_percent_start"] 
            and 
            group_dictionary[group]["group_number"] == 1
            and
            USE_PRICE_ADJUSTMENT
            ):
            adjusted_price = np.average(
                [net_anchor_price, price_in_day]
            )
        else:
            adjusted_price = net_anchor_price
    
    present_share_orders_of_all_groups = db_with_all_share_orders.groupby(
            ["status", "group_number"]
        ).groups
    for group in group_dictionary: 
        if group_dictionary[group]["group_number"] == 0:
            slot_prices_for_group = create_price_for_orders_in_a_group(
                price_to_descend_from=adjusted_price,
                group_x_count=group_dictionary[group]["count"],
                group_x_slot_price_descending=group_dictionary[group]["slot_price_descending_constant"]
            )
        else:
            if fall_percent<0:
                sqrt_of_fall_percent = -np.sqrt(-fall_percent*100)
            else:
                sqrt_of_fall_percent = np.sqrt(fall_percent*100)
                
            constant_from_parts_into_percent = group_dictionary[group]["slot_price_descending_constant"]*100
            group_slot_price_descending = (
                constant_from_parts_into_percent
                +
                sqrt_of_fall_percent/2
                )/100
            slot_prices_for_group = create_price_for_orders_in_a_group(
                price_to_descend_from=slot_prices_for_group[-1],
                group_x_count=group_dictionary[group]["count"],
                group_x_slot_price_descending=group_slot_price_descending
            )
            
        if (
            fall_percent >= group_dictionary[group]["fall_percent_start"] 
            and
            fall_percent <= group_dictionary[group]["fall_percent_end"] 
            and not(
                ('on_the_market', group_dictionary[group]["group_number"]) 
                in present_share_orders_of_all_groups
            ) 
        ):
            for slot_price in slot_prices_for_group:
                create_an_order(
                    db_with_all_share_orders=db_with_all_share_orders,
                    price_for_slot=slot_price,
                    initial_price=adjusted_price,
                    group_number=group_dictionary[group]["group_number"],  
                )
            
def execute_strategy_simulation_process (
    net_anchor_price: float,
    prices_per_day: list,
    
    STRATEGY_UPPER_THRESHOLD:float,
    STRATEGY_LOWER_THRESHOLD:float,
    
    SELL_PROFIT_MARGIN:float,
    
    dictionary_with_all_groups_info:dict[str, dict],
    
    USE_PRICE_ADJUSTMENT:bool = True
) -> tuple[pd.DataFrame, int]:  
    """
    Execute the simulation of a trading strategy over a series of days.

    This function simulates the execution of a trading strategy over a period of days
    based on the provided parameters and group information. It updates the
    `db_with_all_share_orders` DataFrame with trading actions.

    Args:
        - net_anchor_price (float): The initial anchor price for the trading strategy.
        - prices_per_day (List[float]): A list of daily share prices for the simulation period.
        - STRATEGY_UPPER_THRESHOLD (float): The upper threshold for strategy adjustment.
        - STRATEGY_LOWER_THRESHOLD (float): The lower threshold for strategy adjustment.
        - SELL_PROFIT_MARGIN (float): The profit margin for selling shares.
        - dictionary_with_all_groups_info (dict[str, dict]): A dictionary containing information about different trading groups. See build_net_of_groups and create_group_dict for more details.
        - USE_PRICE_ADJUSTMENT (bool, optional): Whether to use price adjustment. See build_net_of_groups for more details. Defaults to True.

    Returns:
        - Tuple[pd.DataFrame, int]:
            - A DataFrame (`db_with_all_share_orders`) containing trading actions. Each action contains next set of columns:
                - price: price for the share that we listed in order;
                - action: buy or sell action for the share;
                - status: complited, aborted or on_the_market;
                - net_anchor_price: anchor price that we used for trading strategy in the moment of posting order;
                - end_trade_price: price that order was closed with;
                - group_number: group_number from which group we made the share order.
            - An integer representing the count of big falls that triggered strategy adjustments.
    """
    
    db_with_all_share_orders = pd.DataFrame(
        {
            "price": pd.Series(dtype=float),
            "action": pd.Series(dtype=str),
            "status": pd.Series(dtype=str),
            "net_anchor_price": pd.Series(dtype=float),
            "end_trade_price": pd.Series(dtype=float),
            "group_number": pd.Series(dtype=int)
        }
    )

    big_fall_count = 0
    
    for price_in_day in prices_per_day:
        
        execute_day_of_trading_simulation (
            db_with_all_share_orders=db_with_all_share_orders,
            SELL_PROFIT_MARGIN=SELL_PROFIT_MARGIN,
            price_in_day=price_in_day
        )
        
        
        fall_percent = (net_anchor_price-price_in_day)/net_anchor_price
        
        net_anchor_price, big_fall_count = manage_strategy_due_to_thresholds(
            fall_percent=fall_percent,
            net_anchor_price=net_anchor_price,
            price_in_day = price_in_day,
            db_with_all_share_orders=db_with_all_share_orders,
            big_fall_count=big_fall_count,
            
            STRATEGY_UPPER_THRESHOLD=STRATEGY_UPPER_THRESHOLD,
            STRATEGY_LOWER_THRESHOLD=STRATEGY_LOWER_THRESHOLD
        )
        
        build_net_of_groups(
            fall_percent=fall_percent,
            price_in_day=price_in_day,
            net_anchor_price=net_anchor_price,
            db_with_all_share_orders=db_with_all_share_orders,
            group_dictionary=dictionary_with_all_groups_info,
            USE_PRICE_ADJUSTMENT=USE_PRICE_ADJUSTMENT
        )
        
    return db_with_all_share_orders, big_fall_count

        
def create_statistic_df (
    db_with_all_share_orders:pd.DataFrame,  
) -> pd.DataFrame:
    """
    Create a DataFrame containing statistics of trading actions.

    This function takes a DataFrame `db_with_all_share_orders` that contains information
    about trading actions and groups them by action, status, and group_number to calculate
    statistics.

    Args:
        - db_with_all_share_orders (pd.DataFrame): A DataFrame containing all trading actions, that were made during execute_strategy_simulation_process.

    Returns:
        - pd.DataFrame: A DataFrame containing statistics of trading actions grouped by action, status, and group_number, that contains next columns:
            - action: buy or sell action for the share
            - status: complited, aborted or on_the_market
            - group: group_number from which group we made the share order
            - price: sum of prices for the share that we listed in order
            - net_anchor_price: sum of anchor prices used for trading strategy
            - end_trade_price: sum of prices, that orders were closed with
            - count: count of orders of each action status group combination.
        
    """
    group_actions_to_make_statistic =db_with_all_share_orders.groupby(
        ["action","status", "group_number"]
    ).groups
    
    df_statistic = pd.DataFrame(
        {
            "action" : pd.Series(),
            "status" : pd.Series(),
            "group" : pd.Series(),
            "price" : pd.Series(),
            "net_anchor_price" : pd.Series(),
            "end_trade_price" : pd.Series(),
            "count" : pd.Series(),
        }
    )
    
    for action,status,group in group_actions_to_make_statistic:
        key = (action,status,group)
        
        
        total_price = db_with_all_share_orders.loc[
            group_actions_to_make_statistic[key],
            :
        ].sum()
        
        total_price.pop("action")
        total_price.pop("status")
        total_price.pop("group_number")

        count = db_with_all_share_orders.loc[
            group_actions_to_make_statistic[key],
            :
        ].count()
        
        df_statistic.loc[len(df_statistic.index),:]=[
            action,
            status,
            group,
            total_price.loc["price"],
            total_price["net_anchor_price"],
            total_price["end_trade_price"],
            count.iloc[0]
        ]
        
    return df_statistic
    

def statistic_in_the_end (
    df_with_statistic:pd.DataFrame, 
) -> dict:
    """
    Calculate final trading statistics based on the provided DataFrame.

    This function calculates various trading statistics based on DataFrame from create_statistic_df, 
    which contains statistics of trading actions.

    Args:
        - df_with_statistic (pd.DataFrame): A DataFrame containing statistics of trading actions. It should contain next columns: 
            - action
            - status
            - group
            - price
            - net_anchor_price
            - end_trade_price
            - count

    Returns:
        - dict: A dictionary containing the following trading statistics:
            - 'buy_count': Total count of buy actions.
            - 'completed_sell_count': Total count of completed sell actions.
            - 'aborted_sell_count': Total count of aborted sell actions.
            - 'on_the_market_count': Total count of sell orders remaining on the market.
            - 'overall_spendings': Total amount spent on buying shares.
            - 'completed_sells_gains': Total gains from completed sell actions.
            - 'probable_abort_gains': Total gains from probable aborted sell actions.
            - 'actual_abort_gains': Total gains from actual aborted sell actions.
            - 'abort_losses': Losses due to aborted sell actions (probable - actual).
            - 'didnt_cash_out': Total value of unsold shares on the market.
            - 'profit': Total profit (completed + actual aborted).
            
    """
    buy_mask = (df_with_statistic['action'] == 'buy') & (df_with_statistic['status'] == 'complited')
    sell_mask = (df_with_statistic['action'] == 'sell') & (df_with_statistic['status'] == 'complited')
    aborted_sell_mask = (df_with_statistic['action'] == 'sell') & (df_with_statistic['status'] == 'aborted')
    on_the_market_sell_mask = (df_with_statistic['action'] == 'sell') & (df_with_statistic['status'] == 'on_the_market')

    buy_count = df_with_statistic[buy_mask]['count'].sum()
    completed_sell_count = df_with_statistic[sell_mask]['count'].sum()
    aborted_sell_count = df_with_statistic[aborted_sell_mask]['count'].sum()
    on_the_market_count = df_with_statistic[on_the_market_sell_mask]['count'].sum()

    overall_spendings = df_with_statistic[buy_mask]['end_trade_price'].sum()
    completed_sells_gains = df_with_statistic[sell_mask]['end_trade_price'].sum()
    
    probable_abort_gains = df_with_statistic[aborted_sell_mask]['price'].sum()
    actual_abort_gains = df_with_statistic[aborted_sell_mask]['end_trade_price'].sum()
    abort_losses = probable_abort_gains - actual_abort_gains
    
    didnt_cash_out = df_with_statistic[on_the_market_sell_mask]['price'].sum()

    profit = (completed_sells_gains+actual_abort_gains) - overall_spendings
    
    statistics = {
        'buy_count': buy_count,
        'completed_sell_count': completed_sell_count,
        'aborted_sell_count': aborted_sell_count,
        'on_the_market_count': on_the_market_count,
        'overall_spendings': overall_spendings,
        'completed_sells_gains': completed_sells_gains,
        'probable_abort_gains': probable_abort_gains,
        'actual_abort_gains': actual_abort_gains,
        'abort_losses': abort_losses,
        'didnt_cash_out': didnt_cash_out,
        'profit': profit
    }
    
    return statistics

def generate_speech(
    df_with_statistic:pd.DataFrame, 
    STRATEGY_LOWER_THRESHOLD:float,
    last_price:float,
    big_fall_count:int
):
    shares_statistics = statistic_in_the_end(
        df_with_statistic=df_with_statistic
    )

    speech = f"""Trading Summary:
    ----------------

    Total Bought Shares: {shares_statistics["buy_count"]}
    Total Spent: {shares_statistics["overall_spendings"]}

    Total Compliteed Sales: {shares_statistics["completed_sell_count"]}
    Gains from Complited Sales: {shares_statistics["completed_sells_gains"]}
    
    Total Aborted Sales: {shares_statistics["aborted_sell_count"]}
    Abort Threshold: {STRATEGY_LOWER_THRESHOLD}
    Threshold Breach: {big_fall_count}
    Actual Gains from Aborted Sales: {shares_statistics["actual_abort_gains"]}
    Probable Gains from Aborted Sales: {shares_statistics["probable_abort_gains"]}
    Losses due to Aborted Sales: {shares_statistics["abort_losses"]}

    Total Sold Shares: {shares_statistics["completed_sell_count"]+shares_statistics["actual_abort_gains"]}
    Total Gains from Sales: {shares_statistics["completed_sells_gains"]+shares_statistics["actual_abort_gains"]}

    Shares Remaining on the Market: {shares_statistics["on_the_market_count"]}
    Value of Unsold Shares: {shares_statistics["didnt_cash_out"]}
    Last Price: {last_price}
    Value of Unsold Shares With Last Price: {shares_statistics["on_the_market_count"]*last_price}

    Profit (without unsold shares): {shares_statistics["profit"]}
    Profit (with unsold shares): {shares_statistics["profit"] + shares_statistics["didnt_cash_out"]}
    Profit (if unsold shares are sold at last day price): {shares_statistics["profit"] + last_price * shares_statistics["on_the_market_count"]}

    """
    
    return speech

def main ():
    """
    Main function for executing a trading strategy simulation.

    This function serves as the entry point for executing a trading strategy simulation. It sets 
    up various parameters for the simulation, including initial prices, strategy thresholds, group 
    information, and more. It then performs the simulation using the `execute_strategy_simulation_process` 
    function, generates trading statistics, and prints a summary report. Finally, it saves the 
    results to CSV files and plots the price simulation.
    
    """
    # Set up parameters for the trading strategy simulation.
    # For generate_price_simulation:
    FIRST_DAY_PRICE = 19176
    DAYS_TO_SIMULATE = 365
    UPPER_PRICE_BOUND_FOR_PRICE_SIMULATION = 40000
    LOWER_PRICE_BOUND_FOR_PRICE_SIMULATION = 12000
    
    STRATEGY_ANCHOR_POINT = FIRST_DAY_PRICE
    
    # strategy settings for execute_strategy_simulation_process: 
    STRATEGY_UPPER_THRESHOLD = -0.12
    STRATEGY_LOWER_THRESHOLD = 0.7

    SELL_PROFIT_MARGIN = 0.04

    # Groups for net creation:
    GROUP_0_COUNT = 6
    GROUP_0_SLOT_PRICE_DESCENDING_CONSTANT = 0.0366
    GROUP_0_FALL_PERCENT_START = -0.05
    GROUP_0_FALL_PERCENT_END = 0.20

    GROUP_1_COUNT = 6
    GROUP_1_SLOT_PRICE_DESCENDING_CONSTANT = 0.0195
    GROUP_1_FALL_PERCENT_START = 0.1
    GROUP_1_FALL_PERCENT_END = 0.3
    
    GROUP_2_COUNT = 4
    GROUP_2_SLOT_PRICE_DESCENDING_CONSTANT = 0.03
    GROUP_2_FALL_PERCENT_START = 0.25
    GROUP_2_FALL_PERCENT_END = 0.45
    
    # From every group create list with:
    # [count, slot_price_descending_constant, fall_percent_start, fall_percent_end];
    # and add it to the create_group_dict args ->
    group_info_dictionary = create_group_dict(
        [GROUP_0_COUNT, GROUP_0_SLOT_PRICE_DESCENDING_CONSTANT, GROUP_0_FALL_PERCENT_START, GROUP_0_FALL_PERCENT_END],
        [GROUP_1_COUNT, GROUP_1_SLOT_PRICE_DESCENDING_CONSTANT, GROUP_1_FALL_PERCENT_START, GROUP_1_FALL_PERCENT_END],
        [GROUP_2_COUNT, GROUP_2_SLOT_PRICE_DESCENDING_CONSTANT, GROUP_2_FALL_PERCENT_START, GROUP_2_FALL_PERCENT_END],
    )
    
    # Parametr, that determines we should use price adjusment,
    # outside of strategy threshold violation:
    USE_PRICE_ADJUSTMENT = True

    
    # Generate prices by given parameters:
    prices_in_set_period=generate_price_simulation(
        first_day_price=FIRST_DAY_PRICE,
        number_of_days_to_simulate=DAYS_TO_SIMULATE,
        upper_price_bound = UPPER_PRICE_BOUND_FOR_PRICE_SIMULATION,
        lower_price_bound=LOWER_PRICE_BOUND_FOR_PRICE_SIMULATION
    )
    
    
    # Call the 'execute_strategy_simulation_process' function to perform the simulation.
    strategy_actions_df,  big_fall_count = execute_strategy_simulation_process (
        net_anchor_price = STRATEGY_ANCHOR_POINT,
        prices_per_day=prices_in_set_period,
        STRATEGY_UPPER_THRESHOLD = STRATEGY_UPPER_THRESHOLD,
        STRATEGY_LOWER_THRESHOLD = STRATEGY_LOWER_THRESHOLD,
        
        SELL_PROFIT_MARGIN = SELL_PROFIT_MARGIN,
        
        dictionary_with_all_groups_info=group_info_dictionary,
        
        USE_PRICE_ADJUSTMENT = USE_PRICE_ADJUSTMENT
    )
    
    # Create statistic metrics and wiev them for executeed simulation:
    df_with_statistic = create_statistic_df(
        db_with_all_share_orders=strategy_actions_df
    )
    
    speech = generate_speech(
        df_with_statistic=df_with_statistic,
        STRATEGY_LOWER_THRESHOLD=STRATEGY_LOWER_THRESHOLD,
        last_price=prices_in_set_period[-1],
        big_fall_count = big_fall_count
    )
    
    # Create files with info about this strategy execution: 
    with open("prices_in_period.txt", "w") as file:
        file.write(str(prices_in_set_period))    
    strategy_actions_df.to_csv("actions.csv")
    df_with_statistic.to_csv("statistic.csv")
    print(speech)
    with open("speech.txt", "w") as file:
        file.write(speech)
    
    # Create chart of generated price simulation:
    range_of_days_in_price_simulation = np.arange(
        0, 
        DAYS_TO_SIMULATE+1
    )
    
    plt.plot(range_of_days_in_price_simulation,prices_in_set_period)
    plt.show()
    
def test_strategy_execution():
    """
    Acts as main and is used to executing a trading strategy simulation multiple time and 
    asserts that the strategy has resulted in a positive or zero profit.
    pytest --report-log="tests.log" --count=1000 algorithm.py - to test algorithm 1000 times

    """
    # Set up parameters for the trading strategy simulation.
    # For generate_price_simulation:
    FIRST_DAY_PRICE = 19176
    DAYS_TO_SIMULATE = 365
    UPPER_PRICE_BOUND_FOR_PRICE_SIMULATION = 39000
    LOWER_PRICE_BOUND_FOR_PRICE_SIMULATION = 15000
    
    STRATEGY_ANCHOR_POINT = FIRST_DAY_PRICE
    
    # strategy settings for execute_strategy_simulation_process: 
    STRATEGY_UPPER_THRESHOLD = -0.12
    STRATEGY_LOWER_THRESHOLD = 0.7

    SELL_PROFIT_MARGIN = 0.04

    # Groups for net creation:
    GROUP_0_COUNT = 6
    GROUP_0_SLOT_PRICE_DESCENDING_CONSTANT = 0.0366
    GROUP_0_FALL_PERCENT_START = -0.05
    GROUP_0_FALL_PERCENT_END = 0.20

    GROUP_1_COUNT = 6
    GROUP_1_SLOT_PRICE_DESCENDING_CONSTANT = 1.95
    GROUP_1_FALL_PERCENT_START = 0.1
    GROUP_1_FALL_PERCENT_END = 0.3
    
    GROUP_2_COUNT = 4
    GROUP_2_SLOT_PRICE_DESCENDING_CONSTANT = 3
    GROUP_2_FALL_PERCENT_START = 0.25
    GROUP_2_FALL_PERCENT_END = 0.45
    
    # From every group create list with:
    # [count, slot_price_descending_constant, fall_percent_start, fall_percent_end];
    # and add it to the create_group_dict args ->
    group_info_dictionary = create_group_dict(
        [GROUP_0_COUNT, GROUP_0_SLOT_PRICE_DESCENDING_CONSTANT, GROUP_0_FALL_PERCENT_START, GROUP_0_FALL_PERCENT_END],
        [GROUP_1_COUNT, GROUP_1_SLOT_PRICE_DESCENDING_CONSTANT, GROUP_1_FALL_PERCENT_START, GROUP_1_FALL_PERCENT_END],
        [GROUP_2_COUNT, GROUP_2_SLOT_PRICE_DESCENDING_CONSTANT, GROUP_2_FALL_PERCENT_START, GROUP_2_FALL_PERCENT_END],
    )
    
    # Parametr, that determines we should use price adjusment,
    # outside of strategy threshold violation:
    USE_PRICE_ADJUSTMENT = True

    
    # Generate prices by given parameters:
    prices_in_set_period=generate_price_simulation(
        first_day_price=FIRST_DAY_PRICE,
        number_of_days_to_simulate=DAYS_TO_SIMULATE,
        upper_price_bound = UPPER_PRICE_BOUND_FOR_PRICE_SIMULATION,
        lower_price_bound=LOWER_PRICE_BOUND_FOR_PRICE_SIMULATION
    )
    
    
    # Call the 'execute_strategy_simulation_process' function to perform the simulation.
    strategy_actions_df,  big_fall_count = execute_strategy_simulation_process (
        net_anchor_price = STRATEGY_ANCHOR_POINT,
        prices_per_day=prices_in_set_period,
        STRATEGY_UPPER_THRESHOLD = STRATEGY_UPPER_THRESHOLD,
        STRATEGY_LOWER_THRESHOLD = STRATEGY_LOWER_THRESHOLD,
        
        SELL_PROFIT_MARGIN = SELL_PROFIT_MARGIN,
        
        dictionary_with_all_groups_info=group_info_dictionary,
        
        USE_PRICE_ADJUSTMENT = USE_PRICE_ADJUSTMENT
    )
    
    # Create statistic metrics and wiev them for executeed simulation:
    df_with_statistic = create_statistic_df(
        db_with_all_share_orders=strategy_actions_df
    )
    
    statiscs = statistic_in_the_end(
        df_with_statistic=df_with_statistic
    )
    
    profit_without_selling = statiscs["profit"]
    profit_witt_selling = statiscs["profit"] + statiscs["didnt_cash_out"]
    profit_witt_selling_now = statiscs["profit"] + prices_in_set_period[-1] * statiscs["on_the_market_count"]
    assert (
            profit_without_selling >= 0 or
            profit_witt_selling >= 0 or
            profit_witt_selling_now >= 0 or
            big_fall_count < -1
    ) 
    
if __name__ == "__main__":
    main()