import numpy as np

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp  # Proportional gain
        self.Ki = Ki  # Integral gain
        self.Kd = Kd  # Derivative gain
        self.prev_error = np.zeros(3)  # Error for both position and orientation (x, y, z, qx, qy, qz, qw)
        self.integral = np.zeros(3)

    def update(self, target_pose, actual_pose, dt):
        # Calculate error for position and orientation
        error = target_pose - actual_pose

        # Integral term
        self.integral += error * dt

        # Derivative term
        derivative = (error - self.prev_error) / dt

        # Compute control output
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        # Update previous error
        self.prev_error = error

        return output