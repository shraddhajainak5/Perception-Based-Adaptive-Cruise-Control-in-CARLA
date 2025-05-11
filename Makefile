
DOCKER_ARGS = -d \
							--gpus all \
							--net=host \
							--rm \
							--name="carla_sim"

.PHONY: build
build:
	@echo "===== Building Docker image ====="
	sudo docker build --no-cache --pull . -f Carla.Dockerfile -t csci513/mp2

.PHONY: run-carla
run-carla:
	@echo "===== Running Carla in background  ====="
	@echo "===== Use 'make stop' to stop      ====="
	sudo docker run $(DOCKER_ARGS) --privileged  -v /tmp/.X11-unix:/tmp/.X11-unix:rw csci513/mp2

.PHONY: run-headless
run-headless:
	@echo "===== Running Carla in background ====="
	sudo docker run $(DOCKER_ARGS) csci513/mp2

.PHONY: stop
stop:
	@echo "===== Stopping Carla container ===="
	sudo docker stop carla_sim
