import math
import numpy as np
import random

"""

AvellanedaStoikovBot.py

Implementation of the reservation price and spread calculator for the Avellaneda-Stoikov strategy

Juan Francisco Perez

"""



class AvellanedaStoikov(object):

    def __init__(self, gamma, k):
        self._gamma = gamma
        self._k = k

    def reservation_price(self, S, sigma, q, T, t, M):
        reservation_price = S - q * self._gamma * (sigma ** 2) * (T - t/M)
        return reservation_price

    def spread(self, sigma, T, t, M):
        spread = self._gamma * (sigma ** 2) * (T - t/M) + (2/self._gamma) * math.log(1 + (self._gamma/self._k))
        return spread

    def optimal_bid_ask(self, reservation_price, spread):
        bid = reservation_price - spread/2
        ask = reservation_price + spread/2
        return bid, ask

def Simulate_Trading(S, Bid, Ask, k, A, dt):

    Delta_Bid = S - Bid
    Delta_Ask = Ask - S

    lambda_Bid = A * np.exp(-k * Delta_Bid)
    Prob_Bid = 1 - np.exp(-lambda_Bid * dt)
    fb = random.random()

    lambda_Ask = A * np.exp(-k * Delta_Ask)
    Prob_Ask = 1 - np.exp(-lambda_Ask * dt)
    fa = random.random()

    executed_buy = Prob_Bid > fb
    executed_sell = Prob_Ask > fa

    return executed_buy, executed_sell

def Backtest_order(OB_Bid,OB_Ask, Bid, Ask, k,A, dt):
    
    Delta_Bid = OB_Bid - Bid
    Delta_Ask = Ask - OB_Ask

    lambda_Bid = A * np.exp(-k * Delta_Bid)
    Prob_Bid = 1 - np.exp(-lambda_Bid * dt)
    fb = random.random()

    lambda_Ask = A * np.exp(-k * Delta_Ask)
    Prob_Ask = 1 - np.exp(-lambda_Ask * dt)
    fa = random.random()

    executed_buy = Prob_Bid > fb
    executed_sell = Prob_Ask > fa
    return executed_buy, executed_sell


"""

AvellanedaStoikovBot.py

Implementation of the reservation price and spread calculator for the Avellaneda-Stoikov strategy

Juan Francisco Perez

"""