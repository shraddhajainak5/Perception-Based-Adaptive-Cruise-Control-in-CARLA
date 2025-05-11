import random
from collections import deque
from typing import NamedTuple, Optional, Tuple


import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


from scipy import signal
from skimage.transform import resize

from mp2_simulator.misc import *
from mp2_simulator.render import BirdeyeRender


CONFIG = {
    "display_size": 400,  # screen size of bird-eye render
    "max_past_step": 1,  # the number of past steps to draw
    "dt": 0.1,  # time interval between two frames
    "max_timesteps": 200,  # maximum timesteps per episode
    "ego_vehicle_filter": "vehicle.lincoln.*",  # filter for defining ego vehicle
    "ado_vehicle_filter": "vehicle.toyota.prius",
    "port": 2000,  # connection port
    "town": "Town06",  # which town to simulate
    "obs_range": 50,  # observation range (meter)
    "d_behind": 12,  # distance behind the ego vehicle (meter)
    "desired_speed": 10,  # desired speed (m/s)
    "distance_threshold": 4,  # distance threshold (meters)
    "ado_sawtooth_width": 0.5,
    "ado_sawtooth_period": 10,  # seconds
    "render": True,
}

CAMERA_POSITION = {
    "x": 0.8,
    "z": 1.7,
}


class Observation(NamedTuple):

    ego_velocity: float
    desired_speed: float
    distance_to_lead: float


def distance_between_transforms(p1: carla.Transform, p2: carla.Transform) -> float:
    from numpy.linalg import norm

    p1_pos = np.array([p1.location.x, p1.location.y], dtype=np.float64)

    p2_pos = np.array([p2.location.x, p2.location.y], dtype=np.float64)

    return norm(p2_pos - p1_pos)


def create_ado_sawtooth(
    max_timesteps: int,
    dt: float,
    *,
    time_period: float = 10.0,  # seconds
    rise_width: float = 0.5,  # [0, 1]
) -> np.ndarray:
    t = np.linspace(0, max_timesteps * dt, max_timesteps)
    # Frequency
    freq = 1 / time_period
    # Width of the rise window
    width = rise_width
    # We want to offset so that we start at 0
    offset = (time_period / 2) * width
    f = signal.sawtooth(freq * 2 * np.pi * (t + offset), width)
    return f


class Simulator:
    """
    Class responsible of handling all the different CARLA functionalities, such as server-client connecting,
    actor spawning and getting the sensors data.
    """

    def __init__(self, **config):
        self.config = CONFIG
        self.config.update(config)

        self.server_port = self.config["port"]
        self.dt = self.config["dt"]
        self.display_size = self.config["display_size"]

        print("connecting to Carla server...")
        self.client = carla.Client("127.0.0.1", self.server_port)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(self.config["town"])
        print("Carla server connected!")

        # Set weather
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

        # Get spawn points
        self.ego_vehicle_spawn_point = carla.Transform(
            carla.Location(x=584.882446, y=-20.577578, z=0.300000),
            carla.Rotation(pitch=0.000000, yaw=179.857498, roll=0.000000),
        )
        self.ado_vehicle_spawn_point = carla.Transform(
            carla.Location(x=584.882446 - 20, y=-20.577578, z=0.300000),  # was - 100
            carla.Rotation(pitch=0.000000, yaw=179.857498, roll=0.000000),
        )

        # Create some behavior for the ado
        # self.ado_acc_control_signal = create_ado_sawtooth(
        #     self.config["max_timesteps"],
        #     self.dt,
        #     rise_width=self.config["ado_sawtooth_width"],
        #     time_period=self.config["ado_sawtooth_period"],
        # )

        # Create the ego vehicle blueprint
        self.ego_bp = self._create_vehicle_blueprint(
            self.config["ego_vehicle_filter"], color="49,8,8")
        self.ego: Optional[carla.Vehicle] = None
        self.ado_bp = self._create_vehicle_blueprint(
            self.config["ado_vehicle_filter"], color="49,8,8")
        self.ado: Optional[carla.Vehicle] = None

        # Collision sensor
        self.collision_hist = deque(maxlen=1)  # type: deque[bool]
        self.collided_event = False
        self.collision_bp = self.world.get_blueprint_library().find(
            "sensor.other.collision"
        )
        self.collision_sensor = None

        # Camera sensor
        self.obs_size = 600
        self.camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
        self.camera_trans = carla.Transform(
            carla.Location(x=CAMERA_POSITION["x"], z=CAMERA_POSITION["z"])
        )
        self.camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute("image_size_x", str(self.obs_size))
        self.camera_bp.set_attribute("image_size_y", str(self.obs_size))
        self.camera_bp.set_attribute("fov", "110")
        # Set the time in seconds between sensor captures
        self.camera_bp.set_attribute("sensor_tick", "0.02")
        self.camera_sensor = None

        self.settings = self.world.get_settings()

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0
        self.time_step = 0

        self.render = config["render"]

        # Initialize the renderer
        self._init_renderer()

    def _create_vehicle_blueprint(self, actor_filter, color=None):
        """Create the blueprint for a specific actor type.
        Args:
          actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.
        Returns:
          bp: the blueprint object of carla.
        """
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = [
            x for x in blueprints if int(x.get_attribute("number_of_wheels")) == 4
        ]
        bp = random.choice(blueprint_library)
        if bp.has_attribute("color"):
            if not color:
                color = random.choice(
                    bp.get_attribute("color").recommended_values)
            bp.set_attribute("color", color)
        return bp

    def _clear_all_actors(self):
        """Clear specific actors."""
        # [
        #     "sensor.other.collision",
        #     "sensor.camera.rgb",
        #     "vehicle.*",
        #     "controller.ai.walker",
        #     "walker.*",
        # ]

        if self.camera_sensor is not None:
            self.camera_sensor.stop()
            self.camera_sensor.destroy()
            self.camera_sensor = None

        # CLear the ego and ado cars also
        if self.ego is not None:
            self.ego.destroy()
            self.ego = None
        if self.ado is not None:
            self.ado.destroy()
            self.ado = None

    def _set_synchronous_mode(self, synchronous=True):
        """Set whether to use the synchronous mode."""
        self.settings.fixed_delta_seconds = self.dt
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)

    def _init_renderer(self):
        """Initialize the birdeye view renderer."""
        if self.render:
            pygame.init()
            self.display = pygame.display.set_mode(
                (self.display_size * 3, self.display_size),
                pygame.HWSURFACE | pygame.DOUBLEBUF,
            )

            pixels_per_meter = self.display_size / self.config["obs_range"]
            pixels_ahead_vehicle = (
                self.config["obs_range"] / 2 - self.config["d_behind"]
            ) * pixels_per_meter
            birdeye_params = {
                "screen_size": [self.display_size, self.display_size],
                "pixels_per_meter": pixels_per_meter,
                "pixels_ahead_vehicle": pixels_ahead_vehicle,
            }
            self.birdeye_render = BirdeyeRender(self.world, birdeye_params)
        else:
            self.display = None
            self.birdeye_render = None

    def reset(self) -> Tuple[Observation, np.ndarray]:
        # Delete sensors, vehicles and walkers
        self._clear_all_actors()

        # Disable sync mode
        self._set_synchronous_mode(False)

        # Spawn ado car
        self._try_spawn_ado_vehicle_at(self.ado_vehicle_spawn_point)

        # Get actors polygon list
        self.vehicle_polygons = []
        vehicle_poly_dict = self._get_actor_polygons("vehicle.*")
        self.vehicle_polygons.append(vehicle_poly_dict)

        # Spawn ego car
        self._try_spawn_ego_vehicle_at(self.ego_vehicle_spawn_point)
        assert self.ego is not None

        # Add collision sensor
        self.collision_sensor = self.world.spawn_actor(
            self.collision_bp, carla.Transform(), attach_to=self.ego
        )
        self.collision_sensor.listen(
            lambda event: self.handle_collision(event))
        self.collided_event = False
        self.collision_hist = deque(maxlen=1)  # type: deque[bool]

        # Add camera sensor
        self.camera_sensor = self.world.spawn_actor(
            self.camera_bp, self.camera_trans, attach_to=self.ego
        )
        self.camera_sensor.listen(lambda data: get_camera_img(data))

        def get_camera_img(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.camera_img = array

        # Update timesteps
        self.time_step = 0
        self.reset_step += 1

        # Enable sync mode
        self._set_synchronous_mode(True)

        # Set ego information for render
        if self.birdeye_render is not None:
            self.birdeye_render.set_hero(self.ego, self.ego.id)

        return self._get_obs()

    def handle_collision(self, event):
        self.collision_hist.append(True)
        self.collided_event = True
        print("Vehicle collision detected!!!")

    def _try_spawn_ego_vehicle_at(self, transform):
        """Try to spawn the ego vehicle at specific transform.
        Args:
          transform: the carla transform object.
        Returns:
          Bool indicating whether the spawn is successful.
        """
        vehicle = None
        if self.ego is not None:
            vehicle = self.ego if self.ego.is_alive else None
        else:
            vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

        if vehicle is not None:
            vehicle.set_transform(transform)
            self.ego = vehicle
            print("Using Ego Vehicle ID: ", self.ego.id)
            return True

        return False

    def _try_spawn_ado_vehicle_at(self, transform):
        """Try to spawn the ado vehicle at specific transform.
        Args:
          transform: the carla transform object.
        Returns:
          Bool indicating whether the spawn is successful.
        """
        vehicle = None
        if self.ado is not None:
            vehicle = self.ado if self.ado.is_alive else None
        else:
            vehicle = self.world.try_spawn_actor(self.ado_bp, transform)

        if vehicle is not None:
            vehicle.set_transform(transform)
            self.ado = vehicle
            print("Using Ado Vehicle ID: ", self.ado.id)
            return True

        return False

    def _get_control(self, acc) -> carla.VehicleControl:
        # Convert acceleration to throttle and brake
        if acc > 0:
            throttle = np.clip(acc / 10, 0, 1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acc / 20, 0, 1)

        # Apply control
        act = carla.VehicleControl(
            throttle=float(throttle), brake=float(brake))
        return act

    def _get_sawtooth_control(self) -> carla.VehicleControl:
        acc = self.ado_acc_control_signal[self.time_step] * 10

        return self._get_control(acc)

    def _set_sawtooth_control(self, ado_sawtooth_period, ado_sawtooth_width):
        self.ado_acc_control_signal = create_ado_sawtooth(
            self.config["max_timesteps"],
            self.dt,
            rise_width=ado_sawtooth_width,
            time_period=ado_sawtooth_period,
        )

    @property
    def completed(self) -> bool:
        return self.time_step >= self.config["max_timesteps"] or self.collided_event

    def step(self, acc: float) -> Tuple[Observation, np.ndarray]:
        assert self.ego is not None
        assert self.ado is not None

        acc = np.clip(acc, -10.0, 10.0)
        act = self._get_control(acc)
        self.ego.apply_control(act)

        # ado actions
        self.ado.apply_control(self._get_sawtooth_control())

        self.world.tick()

        # Update timesteps
        self.time_step += 1
        self.total_step += 1

        return self._get_obs()

    def _get_obs(self) -> Tuple[Observation, np.ndarray]:
        """Get the observations."""
        assert self.ego is not None
        assert self.ado is not None

        ## Birdeye rendering
        if self.render:
            assert self.birdeye_render is not None
            assert self.display is not None
            self.birdeye_render.vehicle_polygons = self.vehicle_polygons

            # birdeye view with roadmap and actors
            birdeye_render_types = ["roadmap", "actors"]
            self.birdeye_render.render(self.display, birdeye_render_types)
            birdeye = pygame.surfarray.array3d(self.display)
            birdeye = birdeye[0: self.display_size, :, :]
            birdeye = display_to_rgb(birdeye, self.obs_size)

            # Display birdeye image
            birdeye_surface = rgb_to_display_surface(
                birdeye, self.display_size)
            self.display.blit(birdeye_surface, (0, 0))

            # Display camera image
            camera = resize(self.camera_img,
                            (self.obs_size, self.obs_size)) * 255
            camera_surface = rgb_to_display_surface(camera, self.display_size)
            self.display.blit(camera_surface, (self.display_size * 2, 0))

            # Display on pygame
            pygame.display.flip()

        # Get state observation
        ego_trans = self.ego.get_transform()
        ado_trans = self.ado.get_transform()

        distance_to_ado = distance_between_transforms(ego_trans, ado_trans)
        if self.collided_event:
            distance_to_ado = 0
        speed = self._get_ego_velocity()

        return Observation(
                ego_velocity=speed,
                distance_to_lead=distance_to_ado,
                desired_speed=self.config["desired_speed"],
            ), self.camera_img


    def _get_actor_polygons(self, filt):
        """Get the bounding box polygon of actors.
        Args:
          filt: the filter indicating what type of actors we'll look at.
        Returns:
          actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
        """
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt):
            # Get x, y and yaw of the actor
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi  # type: ignore
            # Get length and width
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            # Get bounding box polygon in the actor's local coordinate
            poly_local = np.array(
                [[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
            # Get rotation matrix to transform to global coordinate
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            # Get global bounding box polygon
            poly = np.matmul(R, poly_local).transpose() + \
                np.repeat([[x, y]], 4, axis=0)
            actor_poly_dict[actor.id] = poly
        return actor_poly_dict

    def _get_ado_velocity(self) -> float:
        assert self.ado is not None

        velocity = self.ado.get_velocity()
        magnitude = np.sqrt(velocity.x ** 2 + velocity.y ** 2)  # type: ignore

        return magnitude

    def _get_ego_velocity(self) -> float:
        assert self.ego is not None

        velocity = self.ego.get_velocity()
        magnitude = np.sqrt(velocity.x ** 2 + velocity.y ** 2)  # type: ignore

        return magnitude
