from mp2_simulator.simulator import Observation

class Controller:
    def __init__(self, target_speed: float, distance_threshold: float):
        self.target_speed = target_speed
        self.distance_threshold = distance_threshold

        self.kp = 1.5
        self.kd = 0.3

        self.previous_speed_error = 0.0
        self.previous_dist_to_lead = None

        self.min_velocity = 0.5

    def run_step(self, obs: Observation, estimate_dist) -> float:
        ego_velocity = obs.ego_velocity
        dist_to_lead = estimate_dist
        max_acceleration = 3.0
        max_deceleration = -10.0

        if dist_to_lead is None or dist_to_lead <= 0:
            return max_deceleration  

        critical_distance = 8.0  
        safe_distance = 18.0  

        speed_error = self.target_speed - ego_velocity
        speed_error_factor = speed_error - self.previous_speed_error
        self.previous_speed_error = speed_error

        if self.previous_dist_to_lead is not None:
            closing_rate = self.previous_dist_to_lead - dist_to_lead
        else:
            closing_rate = 0.0
        self.previous_dist_to_lead = dist_to_lead

        acceleration = 0.0

        if closing_rate == 0 and ego_velocity > 0:  
            if dist_to_lead < safe_distance:
                
                acceleration = max_deceleration * ((dist_to_lead - critical_distance) / safe_distance)
            else:
                acceleration = 0.0  
        
        elif dist_to_lead < critical_distance:
            acceleration = max_deceleration
        elif dist_to_lead < safe_distance:
            distance_factor = (dist_to_lead - critical_distance) / (safe_distance - critical_distance)
            acceleration = max_deceleration * (1 - distance_factor)
            acceleration += 1.5 * closing_rate
        else:
            if ego_velocity < self.target_speed:
                acceleration = self.kp * speed_error + self.kd * speed_error_factor
                if ego_velocity + acceleration < self.target_speed:
                    acceleration = min(acceleration, max_acceleration)
            elif ego_velocity >= self.target_speed:
                acceleration = 0.0
                if ego_velocity > self.target_speed + 0.1:
                    acceleration = max_deceleration  

        acceleration = max(min(acceleration, max_acceleration), max_deceleration)

        return acceleration
