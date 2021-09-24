


class PID(object):
    """A proportional-integral-derivative controller.

    Args:
        kp (float): Proportional coefficient.
        ki (float): Integral coefficient.
        kd (float): Derivative coefficient.
        out_min (float): Lower output limit.
        out_max (float): Upper output limit.
    """

    def __init__(self, kp, ki, kd, out_min=float('-inf'),out_max=float('inf')):
        self._Kp = kp
        self._Ki = ki 
        self._Kd = kd 
        self._out_min = out_min
        self._out_max = out_max
        self._integral = 0
        self._last_input = 0
        self._last_output = 0


    def calc(self, input_val, setpoint):
        # Compute all the working error variables
        error = setpoint - input_val
        input_diff = input_val - self._last_input

        # In order to prevent windup, only integrate if the process is not saturated
        if self._last_output < self._out_max and self._last_output > self._out_min:
            self._integral += self._Ki * error
            self._integral = min(self._integral, self._out_max)
            self._integral = max(self._integral, self._out_min)

        p = self._Kp * error
        i = self._integral
        d = -(self._Kd * input_diff)

        # Compute PID Output
        self._last_output = p + i + d
        self._last_output = min(self._last_output, self._out_max)
        self._last_output = max(self._last_output, self._out_min)

        # Remember some variables for next time
        self._last_input = input_val
        return self._last_output



        