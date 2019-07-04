#!/bin/sh

DOCKER_FILE=""
DOCKER_HOME="$HOME/docker-home"
MAPSOCKETS=y
while test -n "$1" ; do
  case "$1" in
    -h|--help)
      echo "Usage: $0 [--map-sockets]" >&2
      exit 1
      ;;
    --no-map-sockets)
      MAPSOCKETS=n
      ;;
    *)
      DOCKER_FILE="$1"
      ;;
  esac
  shift
done

if [[ -z "$DOCKER_FILE" ]]; then
  echo "Please provide the docker file name"
fi

CWD=$(cd `dirname "$DOCKER_FILE"`; pwd;)
TAG="$(basename "$DOCKER_FILE")"

# Remap detach to Ctrl+e,e
DOCKER_CFG="/tmp/docker-mrc-nlp-$UID"
mkdir "$DOCKER_CFG" 2>/dev/null || true
cat >$DOCKER_CFG/config.json <<EOF
{ "detachKeys": "ctrl-e,e" }
EOF

set -x

docker build \
  --build-arg=http_proxy=$https_proxy \
  --build-arg=https_proxy=$https_proxy \
  --build-arg=ftp_proxy=$https_proxy \
  -t "$TAG" \
  -f "$DOCKER_FILE" "$CWD"

set +x

echo "Ran docker build"
echo "for file $DOCKER_FILE"
echo "with tag $TAG"
echo "with path $CWD"

if test "$MAPSOCKETS" = "y"; then
  PORT_TENSORBOARD=`expr 6000 + $UID - 1000`
  PORT_JUPYTER=`expr 8000 + $UID - 1000`
  DOCKER_PORT_ARGS="-p 0.0.0.0:$PORT_TENSORBOARD:6006 -p 0.0.0.0:$PORT_JUPYTER:8888"
  (
  echo
  echo "***************************"
  echo "Host Jupyter port:     ${PORT_JUPYTER}"
  echo "Host Tensorboard port: ${PORT_TENSORBOARD}"
  echo "***************************"
  )
fi

# To allow X11 connections from docker
xhost +local: || true
cp "$HOME/.Xauthority" "$DOCKER_HOME/.Xauthority" || true

set -x

nvidia-docker --config "$DOCKER_CFG" \
    run -it \
    -v "$DOCKER_HOME:/workspace" \
    -v "$HOME/proj:/workspace/proj" \
    --workdir /workspace \
    -e HOST_PERMS="$(id -u):$(id -g)" \
    -e "CI_BUILD_HOME=/workspace" \
    -e "CI_BUILD_USER=$(id -u -n)" \
    -e "CI_BUILD_UID=$(id -u)" \
    -e "CI_BUILD_GROUP=$(id -g -n)" \
    -e "CI_BUILD_GID=$(id -g)" \
    -e "DISPLAY=$DISPLAY" \
    -e "TERM=$TERM" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    ${DOCKER_PORT_ARGS} \
    "$TAG" \
    bash --login /install/with_the_same_user.sh bash


