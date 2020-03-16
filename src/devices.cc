#include "ctranslate2/devices.h"

#ifdef WITH_CUDA
#  include "./cuda/utils.h"
#endif

#include "ctranslate2/primitives/primitives.h"
#include "./device_dispatch.h"

namespace ctranslate2 {

  Device str_to_device(const std::string& device) {
#ifdef WITH_CUDA
    if (device == "cuda" || device == "CUDA")
      return Device::CUDA;
#endif
    if (device == "cpu" || device == "CPU")
      return Device::CPU;
    if (device == "auto" || device == "AUTO")
#ifdef WITH_CUDA
      return cuda::has_gpu() ? Device::CUDA : Device::CPU;
#else
      return Device::CPU;
#endif
    throw std::invalid_argument("unsupported device " + device);
  }

  std::string device_to_str(Device device) {
    switch (device) {
    case Device::CUDA:
      return "CUDA";
    case Device::CPU:
      return "CPU";
    }
    return "";
  }

  ScopedDeviceSetter::ScopedDeviceSetter(Device device, int index)
    : _device(device) {
    DEVICE_DISPATCH(_device, _prev_index = primitives<D>::get_device());
    DEVICE_DISPATCH(_device, primitives<D>::set_device(index));
  }

  ScopedDeviceSetter::~ScopedDeviceSetter() {
    DEVICE_DISPATCH(_device, primitives<D>::set_device(_prev_index));
  }

}
