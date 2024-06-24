#!/usr/bin/env python

# base libs
import os
import base64
import io
from typing import Dict, Union

# import libraries
try:
    import torch
    from diffusers import DiffusionPipeline
    from kserve import Model, InferRequest, InferResponse
    from kserve.errors import InvalidInput
    from .tools import get_accelerator_device
except Exception as e:
    print(f"Caught Exception during library loading: {e}")
    raise e

# stable diffusion class
# instantiate this to perform image generation
class DiffusersModel(Model):
    # initialize class
    def __init__(self, name: str):
        super().__init__(name)
        self.model_id = os.environ.get("MODEL_ID", default="/mnt/models")
        # stable diffusion pipeline
        self.pipeline = None
        # refiner model
        self.refiner = None
        # health check
        self.ready = False
        # load model
        self.load()

    # load weights and instantiate pipeline
    def load(self):
        pipeline = DiffusionPipeline.from_pretrained(self.model_id)
        device = get_accelerator_device()
        pipeline.to(device)
        self.pipeline = pipeline
        # The ready flag is used by model ready endpoint for readiness probes,
        # set to True when model is loaded successfully without exceptions.
        self.ready = True

    # process incoming request payload.
    # example payload:
    # {
    #   "instances": [
    #     {
    #       "prompt": "photo of the beach",
    #       "negative_prompt": "ugly, deformed, bad anatomy",
    #       "num_inference_steps": 60
    #     }
    #   ]
    # }
    #
    # validate input request: v2 payloads not yet supported
    def preprocess(self, payload: Union[Dict, InferRequest], headers: Dict[str, str] = None) -> Dict:
        if isinstance(payload, Dict) and "instances" in payload:
            headers["request-type"] = "v1"
        # KServe InferRequest not yet supported
        elif isinstance(payload, InferRequest):
            raise InvalidInput("v2 protocol not implemented")
        else:
            # malformed or missing input payload
            raise InvalidInput("invalid payload")

        # return generation data
        return payload["instances"][0]

    # perform a forward pass (inference) and return generated data
    def predict(self, payload: Union[Dict, InferRequest], headers: Dict[str, str] = None) -> Union[Dict, InferResponse]:
        # generate images
        image = self.pipeline(**payload).images[0]

        # convert images to PNG and encode in base64
        # for easy sending via response payload
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes.seek(0)
        # base64 encoding
        im_b64 = base64.b64encode(image_bytes.read())

        # return payload
        return {
            "predictions": [
                {
                    "model_name": self.model_id,
                    "prompt": payload["prompt"],
                    "image": {
                        "format": "PNG",
                        "b64": im_b64
                    }
                }
            ]}


