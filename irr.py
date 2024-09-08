def irr_cal(year, target):
    cost = 1.0 + target
    irr = cost ** (1 / year) - 1
    irr_year = (cost ** (1 / (year * 12)) - 1) * 12
    return irr, irr_year

print(irr_cal(2, 0.2))
print(irr_cal(5, 0.5))
print(irr_cal(10, 1))
