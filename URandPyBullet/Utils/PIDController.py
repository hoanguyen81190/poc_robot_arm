import numpy as np

# TODO nb the below commented code is upposed to go in the script using the PID controller. I ripped it out as it was cluttering it when not used
### PID controller for end-effector control
#bUsePID = False
#pid_controller = PIDController.PIDController(Kp=0.15, Ki=0.01, Kd=0.001)
#prev_time = time.time()

# Only used for PID control
#dt = time.time()-prev_time

# This requires the current location and orientation of the end effector
#if bUsePID:
#     control_signal = pid_controller.update(target_position_in, end_effector_pose[0:3], dt)
#     new_target_position = control_signal[:3] #target_position_in + control_signal[:3]
#else:
#    new_target_position = target_position_in 
#    new_target_orientation = target_orientation + control_signal[3:]

#prev_time = time.time()

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