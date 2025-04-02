import roop.globals
import numpy as np
import onnx
import onnxruntime
import torch

from roop.typing import Face, Frame
from roop.utilities import resolve_relative_path, check_cuda_system_compatibility


class FaceSwapInsightFace:
    plugin_options: dict = None
    model_swap_insightface = None

    processorname = "faceswap"
    type = "swap"

    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if (
                self.plugin_options["devicename"] != plugin_options["devicename"]
                or self.plugin_options["modelname"] != plugin_options["modelname"]
            ):
                self.Release()

        self.plugin_options = plugin_options
        if self.model_swap_insightface is None:
            model_path = resolve_relative_path(
                "../models/" + self.plugin_options["modelname"]
            )
            graph = onnx.load(model_path).graph
            self.emap = onnx.numpy_helper.to_array(graph.initializer[-1])

            # Check if we should force CPU mode due to missing CUDA dependencies
            force_cpu = False
            if "cuda" in self.plugin_options["devicename"].lower():
                if not check_cuda_system_compatibility():
                    print(
                        "Warning: CUDA requested but missing dependencies. Forcing CPU mode."
                    )
                    self.devicename = "cpu"
                    force_cpu = True
                else:
                    self.devicename = self.plugin_options["devicename"]
            else:
                self.devicename = self.plugin_options["devicename"].replace(
                    "mps", "cpu"
                )

            self.input_mean = 0.0
            self.input_std = 255.0

            sess_options = onnxruntime.SessionOptions()
            sess_options.enable_cpu_mem_arena = False

            providers = roop.globals.execution_providers
            if force_cpu:
                providers = ["CPUExecutionProvider"]

            self.model_swap_insightface = onnxruntime.InferenceSession(
                model_path, sess_options, providers=providers
            )

    def Run(self, source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
        latent = source_face.normed_embedding.reshape((1, -1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)

        use_gpu = False
        try:
            use_gpu = (
                "CUDAExecutionProvider" in roop.globals.execution_providers
                and torch.cuda.is_available()
                and check_cuda_system_compatibility()
                and not self.devicename.startswith("cpu")
            )
        except Exception as e:
            print(f"Error checking GPU availability: {e}")

        io_binding = self.model_swap_insightface.io_binding()
        io_binding.bind_cpu_input("target", temp_frame)
        io_binding.bind_cpu_input("source", latent)

        if use_gpu:
            try:
                io_binding.bind_output("output", "cuda")
            except Exception as e:
                print(f"Error binding output to CUDA: {e}")
                io_binding.bind_output("output", "cpu")
                use_gpu = False
        else:
            io_binding.bind_output("output", self.devicename)

        self.model_swap_insightface.run_with_iobinding(io_binding)
        ort_outs = io_binding.copy_outputs_to_cpu()[0]
        return ort_outs[0]

    def Release(self):
        del self.model_swap_insightface
        self.model_swap_insightface = None
