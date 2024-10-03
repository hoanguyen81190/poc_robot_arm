class ExponentialSmoother:
    def __init__(self, alpha):
        self.alpha = alpha
        self.smoothed_value = None  # No initial value yet

    def update(self, new_value):
        # If smoothed_value is None (first point), initialize it with the first point
        if self.smoothed_value is None:
            self.smoothed_value = new_value
        else:
            # Update the smoothed value using the exponential smoothing formula
            self.smoothed_value = self.alpha * new_value + (1 - self.alpha) * self.smoothed_value
        return self.smoothed_value