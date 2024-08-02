import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from AvellanedaStoikovBot import AvellanedaStoikov, Simulate_Trading


"""

Numerical_Simulation.py

Numerical simulation with parameters from the original Avellaneda-Stoikov paper.

Juan Francisco Perez

"""


def main():
    # Initiate instance of the algorithm
    gamma = 0.1 #Inventory Risk Aversion
    k = 1.5 #Order Book Liquidity Parameter
    AS = AvellanedaStoikov(gamma=gamma, k=k)

    # Parameters of the model
    S0 = 100  # Initial price
    sigma = 2  # Market Volatility
    T = 1      #Time horizon 
    M = 20000  #Time steps
    dt = T / M 


    #Arrays to story results
    S = np.zeros(M + 1)  #Mid-Price
    Bids = np.zeros(M + 1)
    Asks = np.zeros(M + 1)
    Reservation_prices = np.zeros(M + 1)
    spreads = np.zeros(M + 1)
    q = np.zeros(M + 1) #Inventory
    cash = np.zeros(M + 1)
    profit = np.zeros(M + 1)

    S[0] = S0
    Reservation_prices[0] = S0
    Bids[0] = S0
    Asks[0] = S0
    q[0] = 0  # position
    cash[0] = 0  # cash
    profit[0] = 0

    for t in range(1, M + 1):
        # Asset spot price random walk
        z = np.random.standard_normal(1)  # dW
        S[t] = S[t - 1] + sigma * math.sqrt(dt) * z


        Reservation_prices[t] = AS.reservation_price(S[t], sigma, q[t-1], T, t, M)
        spreads[t] = AS.spread(sigma, T, t, M)

        Bids[t], Asks[t] = AS.optimal_bid_ask(Reservation_prices[t], spreads[t])

        executed_buy, executed_sell = Simulate_Trading(S[t], Bids[t], Asks[t], k, 140, dt)

        # According to the results of the simulated trading, we model the inventory

        # Market Maker Long
        if executed_buy and not executed_sell:
            q[t] = q[t - 1] + 1  # position
            cash[t] = cash[t - 1] - Bids[t]  # cash

        # Market Maker Short
        if not executed_buy and executed_sell:
            q[t] = q[t - 1] - 1  # position
            cash[t] = cash[t - 1] + Asks[t]  # cash

        # No position taken
        if not executed_buy and not executed_sell:
            q[t] = q[t - 1]  # position
            cash[t] = cash[t - 1]  # cash

        # Both legs taken - Market Maker Neutral
        if executed_buy and executed_sell:
            q[t] = q[t - 1]  # position
            cash[t] = cash[t - 1] - Bids[t] + Asks[t]  # cash

        profit[t] = cash[t] + q[t] * S[t]  # Update profit

    # Plotting the results
    plt.figure(figsize=(15, 10))

    # Plot S, Bids, and Asks
    plt.subplot(3, 1, 1)
    plt.plot(S, label='Spot Price')
    plt.plot(Bids, label='Bid Price')
    plt.plot(Asks, label='Ask Price')
    plt.title('Spot Price, Bid Price, and Ask Price')
    plt.legend()

    # Plot inventory
    plt.subplot(3, 1, 2)
    plt.plot(q, label='Inventory Position', color='orange')
    plt.title('Inventory Position')
    plt.legend()

    # Plot profit
    plt.subplot(3, 1, 3)
    plt.plot(profit, label='Profit', color='blue')
    plt.title('Profit')
    plt.legend()

    plt.tight_layout()
    plt.show()

    
    print('---------STATS--------')


    spreads_series = pd.Series(spreads)
    q_series = pd.Series(q)

    print(f'Average Spread: {np.mean(spreads_series):.2f}')
    print(f'Max Spread: {max(spreads_series):.2f}')
    print("")
    print("")
    print(f'Average Inventory held: {np.mean(q_series):.2f}')
    print(f'Maximum Inventory held (Short or Long): {max(abs(q_series)):.2f}')
    print("")
    print("")
    print(f'Generated Profit: {profit[-1]:.2f}')
    print(f'Profit StDev: {profit.std():.2f}')


if __name__ == '__main__':
    main()











"""

Numerical_Simulation.py

Numerical simulation with parameters from the original Avellaneda-Stoikov paper.

Juan Francisco Perez

"""