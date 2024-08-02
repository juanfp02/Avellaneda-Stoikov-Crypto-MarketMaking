import random
import numpy as np



"""

Symmetric_Bot.py

Juan Francisco Perez

"""

class Symmetric_Bot(object):

    def __init__(self, spread):
        self._spread = spread

    def bid_ask(self, Mid_price):
        bid = Mid_price - self._spread/2
        ask = Mid_price + self._spread/2
        return bid, ask
    
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

Symmetric_Bot.py

Juan Francisco Perez

"""
