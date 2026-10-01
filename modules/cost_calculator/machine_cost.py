class MachineCostCalculator:
    def __init__(self, fixed_cost=100, var_cost=0.5, fixed_time=10.0, var_time=5):
        self._machine_time = 0
        self._machine_cost = 0

    def machine_time(self, coeff, e):
        """Calculate machine time: time = k * e + b"""
        k = coeff[0]
        b = coeff[1]
        self._machine_time = b + k * e
        return self._machine_time

    def machine_cost(self, coeff):
        """Calculate machine cost: cost = fixed_cost + time * var_cost"""
        fixed_cost = coeff[0]
        var_cost = coeff[1]
        self._machine_cost = fixed_cost + self._machine_time * var_cost
        return self._machine_cost
