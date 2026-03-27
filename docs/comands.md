# Comandos útiles

## CARLA

```bash
cd ~/Downloads/CARLA/CARLA_0.9.15/

__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./CarlaUE4.sh -prefernvidia -vulkan
```

### Generar tráfico

```bash
cd ~/Downloads/CARLA/CARLA_0.9.15/PythonAPI/examples
python3.7 generate_traffic.py
```

### Ejecutar CARLA en calidad baja

```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./CarlaUE4.sh -quality-level=Low
```

### Monitoreo de GPU (en otra terminal)

```bash
watch -n 1 nvidia-smi
```

---

## 🔑 Variables de entorno

```bash
export WANDB_API_KEY=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
```

---

## Entorno virtual y scripts

```bash
source ~/carla_env/bin/activate

# Ir a carpeta del script
python primer_vehiculo.py
```

### Monitoreo de sensores

```bash
watch -n 1 sensors
```

---

## CoppeliaSim

```bash
cd ~/Documents/Coppelia/CoppeliaSim_Edu_V4_9_0_rev2_Ubuntu22_04
./coppeliaSim.sh
```

---

## Docker + GPU + Coppelia

```bash
xhost +local:docker

docker run --rm -it \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e __NV_PRIME_RENDER_OFFLOAD=1 \
    -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/.Xauthority:/root/.Xauthority:ro \
    -v ~/.CoppeliaSim:/root/.CoppeliaSim \
    --net=host \
    coppelia-gpu
```

### Script alternativo

```bash
~/coppelia-docker.sh
```

---

## TensorBoard

```bash
tensorboard --logdir=./tensorboard/
```
