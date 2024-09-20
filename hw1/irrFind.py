import scipy.optimize as opt

def irrFind(cashFlowVec, cashFlowPeriod, compoundPeriod):
    def equation(r):
        npv = cashFlowVec[0]
        for i in range(1, len(cashFlowVec)):
            npv += cashFlowVec[i] / (((1 + (r / (12 / compoundPeriod))) ** (i * cashFlowPeriod / compoundPeriod)))
        return npv
    return opt.fsolve(equation, 0)[0]