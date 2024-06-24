# KServe Compatible Demo Model Server

This is a demo model server that can run inside Openshift AI as a KServe model server.
It exposes a way to perform inference with a Stable Diffusion compatible model in the backend.

## Parameters

The server expects a JSON-encoded payload to start inference:

```json
 // example payload:
 {
   "instances": [
     {
       "prompt": "photo of the beach",
       "negative_prompt": "ugly, deformed, bad anatomy",
       "num_inference_steps": 60
     }
   ]
 }
```

## Known Parameters:

- "prompt": the stable diffusion positive prompt
- "negative_prompt": put here all negative embeddings
- "num_inference_steps": number of generation steps to run during inference
- "width" and "height": size of the generated image
- "guidance_scale": the guidance scale value to feed to the neural network
- "seed": if specified, use this value as the generation seed.

