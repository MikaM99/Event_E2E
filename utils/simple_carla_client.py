# HOW TO
#
# in Terminal go to "/home/carlas/Simulators/CARLA_0.9.16" 
#   -> launch CARLA by running ./CarlaUE4.sh 
#   -> CARLA opens
# Split Terminal
# in second Terminal go to "/home/carlas/Experiments/event-e2e/utils"
#   -> activate cond env E2E
#   -> run this file "python3 simple_carla_client.py"
#   -> things should start moving in CARLA window

# More Information: https://carla.readthedocs.io/en/0.9.16/python_api/#

import carla
import random
import logging
import time
import numpy as np

def camera_callback(image):
    bgra_img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    # continue... save as image or save data in array


if __name__ == "__main__":

    log_level = logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    
    try:
        # Connect to CARLA
        client = carla.Client("localhost", 2000)
        logging.info(f"Client: {client.get_client_version()}, Server: {client.get_server_version()}")

        # Define World
        world = client.get_world()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)

        # Weather
        world.set_weather(carla.WeatherParameters.ClearSunset)

        # Traffic and View
        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)
        spectator = world.get_spectator()

        # Destroy all cars and sensors that might still be in this world
        for v in world.get_actors().filter("*vehicle*"):
            v.destroy()
        for s in world.get_actors().filter("*sensor*"):
            s.destroy()

        # Spawn NPCs
        #npc_spawnpoints = world.get_map().get_spawn_points()
        #npc_vehicles = world.get_blueprint_library().filter("vehicle.carlamotors.carlacola")[0]
        #for npc in range(0, 10):
        #    world.try_spawn_actor(npc_vehicles, random.choice(npc_spawnpoints))

        logging.info("World is ready")

        # Spawn ego vehicle
        vehicle_bp = world.get_blueprint_library().filter("*vehicle.dodge.charger_2020*")[0]
        vehicle_bp.set_attribute("color", "255, 0, 0")
        vehicle_bp.set_attribute("role_name", "ego")
        vehicle_spawnpoint = world.get_map().get_spawn_points()[0]
        ego_vehicle = world.spawn_actor(vehicle_bp, vehicle_spawnpoint)
        ego_vehicle.set_autopilot(True)

        # Attach spectator view to behind the ego vehicle
        spectator.set_transform(
            carla.Transform(
                ego_vehicle.get_transform().transform(carla.Location(x=-4, z=2.5)),
                ego_vehicle.get_transform().rotation
            )
        )

        time.sleep(1)
        logging.info("Car is ready")

        # Spawn camera
        camera_position = carla.Transform(carla.Location(z=1.5))
        camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        camera = world.spawn_actor(camera_bp, camera_position, attach_to=ego_vehicle)

        time.sleep(1)
        logging.info("Camera is ready")

        # Process sensor data
        camera.listen(lambda stream: camera_callback(stream))

        # Run Simulation
        while True: # ends as soon as process is killed with strg+c
            _ = world.tick()

            spectator.set_transform(
            carla.Transform(
                ego_vehicle.get_transform().transform(carla.Location(x=-4, z=2.5)),
                ego_vehicle.get_transform().rotation
                )
            )

    except KeyboardInterrupt:
        logging.info("Cancelled by user!")

