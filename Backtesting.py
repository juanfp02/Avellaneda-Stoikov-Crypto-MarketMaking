import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from AvellanedaStoikovBot import AvellanedaStoikov, Backtest_order
from Symmetric_MMBot import Symmetric_Bot
import math



"""

Backtesting.py

Comparing the performance of the Avellaneda-Stoikov against a Symmetric Spread Strategy on real market data.

Juan Francisco Perez

"""

# Read the data
df = pd.read_csv('BTCUSDT.csv')

# Normalize the data to base 100
initial_mid_price = df['Mid-Price'].iloc[0]
df['Mid-Price'] = df['Mid-Price'] / initial_mid_price * 100
df['OrderBook_Bid'] = df['OrderBook_Bid'] / initial_mid_price * 100
df['OrderBook_Ask'] = df['OrderBook_Ask'] / initial_mid_price * 100

S0 = df['Mid-Price'].iloc[0]  # Initial price

def main():
    # Initiate instance of the algorithm
    gamma = 0.2
    k = 0.5
    A = 50

    #Initialize instances of both Market making strategies.
    AS = AvellanedaStoikov(gamma=gamma, k=k)
    Sym = Symmetric_Bot(spread=0.025*S0)


    sigma = 5  # Market Volatility
    T = 1      #Time horizon 
    M = len(df) - 1  # Number of time steps from the data
    dt = T / M 

    #Arrays to story results
    S = np.zeros(M + 1)
    Bids_AS = np.zeros(M + 1)
    Asks_AS = np.zeros(M + 1)
    Bids_Sym = np.zeros(M + 1)
    Asks_Sym = np.zeros(M + 1)
    Reservation_prices = np.zeros(M + 1)
    spreads = np.zeros(M + 1)

    q_AS = np.zeros(M + 1)
    q_Sym = np.zeros(M + 1)
    cash_AS = np.zeros(M + 1)
    cash_Sym = np.zeros(M + 1)
    profit_AS = np.zeros(M + 1)
    profit_Sym = np.zeros(M + 1)

    S[0] = S0
    Reservation_prices[0] = S0
    Bids_AS[0] = S0
    Asks_AS[0] = S0
    Bids_Sym[0] = S0
    Asks_Sym[0] = S0

    q_AS[0] = 0  # position
    q_Sym[0] = 0
    cash_AS[0] = 0  # cash
    cash_Sym[0] = 0
    profit_AS[0] = 0
    profit_Sym[0] = 0


    for t in range(1, M + 1):
        # Asset spot price from DataFrame
        S[t] = df['Mid-Price'].iloc[t]

        Reservation_prices[t] = AS.reservation_price(S[t], sigma, q_AS[t-1], T, t, M)
        spreads[t] = AS.spread(sigma, T, t, M)

        Bids_AS[t], Asks_AS[t] = AS.optimal_bid_ask(Reservation_prices[t], spreads[t])
        Bids_Sym[t], Asks_Sym[t] = Sym.bid_ask(S[t])


        # Result of Avellaneda-Stoikov

        executed_buy, executed_sell = Backtest_order(df['OrderBook_Bid'].iloc[t], df['OrderBook_Ask'].iloc[t], Bids_AS[t], Asks_AS[t], k,A, dt)

        # According to the results of the simulated trading, we model the inventory
        # Market Maker Long
        if executed_buy and not executed_sell:
            q_AS[t] = q_AS[t - 1] + 1  # position
            cash_AS[t] = cash_AS[t - 1] - Bids_AS[t]  # cash

        # Market Maker Short
        if not executed_buy and executed_sell:
            q_AS[t] = q_AS[t - 1] - 1  # position
            cash_AS[t] = cash_AS[t - 1] + Asks_AS[t]  # cash

        # No position taken
        if not executed_buy and not executed_sell:
            q_AS[t] = q_AS[t - 1]  # position
            cash_AS[t] = cash_AS[t - 1]  # cash

        # Both legs taken - Market Maker Neutral
        if executed_buy and executed_sell:
            q_AS[t] = q_AS[t - 1]  # position
            cash_AS[t] = cash_AS[t - 1] - Bids_AS[t] + Asks_AS[t]  # cash

        profit_AS[t] = cash_AS[t] + q_AS[t] * S[t]  # Update profit
        

        # Result of Symmetrical Spread


        executed_buy, executed_sell = Backtest_order(df['OrderBook_Bid'].iloc[t], df['OrderBook_Ask'].iloc[t], Bids_Sym[t], Bids_Sym[t], k,A, dt)

        # According to the results of the simulated trading, we model the inventory
        # Market Maker Long
        if executed_buy and not executed_sell:
            q_Sym[t] = q_Sym[t - 1] + 1  # position
            cash_Sym[t] = cash_Sym[t - 1] - Bids_Sym[t]  # cash

        # Market Maker Short
        if not executed_buy and executed_sell:
            q_Sym[t] = q_Sym[t - 1] - 1  # position
            cash_Sym[t] = cash_Sym[t - 1] + Asks_Sym[t]  # cash

        # No position taken
        if not executed_buy and not executed_sell:
            q_Sym[t] = q_Sym[t - 1]  # position
            cash_Sym[t] = cash_Sym[t - 1]  # cash

        # Both legs taken - Market Maker Neutral
        if executed_buy and executed_sell:
            q_Sym[t] = q_Sym[t - 1]  # position
            cash_Sym[t] = cash_Sym[t - 1] - Bids_Sym[t] + Asks_Sym[t]  # cash

        profit_Sym[t] = cash_Sym[t] + q_Sym[t] * S[t]  # Update profit

    # Plotting the graphs
    plt.figure(figsize=(15, 20))

    # Plot S, Bids, and Asks for Avellaneda-Stoikov
    plt.subplot(4, 1, 1)
    plt.plot(S, label='Spot Price')
    plt.plot(Bids_AS, label='Bid Price')
    plt.plot(Asks_AS, label='Ask Price')
    plt.title('Avellaneda-Stoikov')
    plt.legend()

    # Plot S, Bids, and Asks for Symmetric spreads
    plt.subplot(4, 1, 2)
    plt.plot(S, label='Spot Price')
    plt.plot(Bids_Sym, label='Bid Price')
    plt.plot(Asks_Sym, label='Ask Price')
    plt.title('Symmetric Spreads')
    plt.legend()

    # Plot q for both strategies
    plt.subplot(4, 1, 3)
    plt.plot(q_AS, label='Inventory Avellaneda-Stoikov', color='orange')
    plt.plot(q_Sym, label='Inventory Symmetric Spreads', color='green')
    plt.title('Inventory Position')
    plt.legend()

    # Plot profit for both strategies
    plt.subplot(4, 1, 4)
    plt.plot(profit_AS, label='Profit Avellaneda-Stoikov', color='blue')
    plt.plot(profit_Sym, label='Profit Symmetric Spreads', color='red')
    plt.title('Profit')
    plt.legend()

    plt.tight_layout()
    plt.show()

        
    print('---------STATS--AS---------')


    spreads_series = pd.Series(spreads)
    q_series = pd.Series(q_AS)

    print(f'Average Spread: {np.mean(spreads_series):.2f}')
    print(f'Max Spread: {max(spreads_series):.2f}')
    print("")
    print("")
    print(f'Average Inventory held: {np.mean(q_series):.2f}')
    print(f'Maximum Inventory held (Short or Long): {max(abs(q_series)):.2f}')
    print("")
    print("")
    print(f'Generated Profit: {profit_AS[-1]:.2f}')
    print(f'Profit StDev: {profit_AS.std():.2f}')


    print('---------STATS--Sym--------')


    q_series = pd.Series(q_Sym)

    print(f'Average Inventory held: {np.mean(q_series):.2f}')
    print(f'Maximum Inventory held (Short or Long): {max(abs(q_series)):.2f}')
    print("")
    print("")
    print(f'Generated Profit: {profit_Sym[-1]:.2f}')
    print(f'Profit StDev: {profit_Sym.std():.2f}')

if __name__ == '__main__':
    main()









"""

Backtesting.py

Comparing the performance of the Avellaneda-Stoikov against a Symmetric Spread Strategy on real market data.

Juan Francisco Perez

"""