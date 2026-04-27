import numpy as np, tensorflow as tf

interp = tf.lite.Interpreter(model_path="kws_results/models/ds_cnn_qat_int8.tflite")
interp.allocate_tensors()

inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]

print(f"input:  shape={inp['shape']}  dtype={inp['dtype']}  "
      f"scale={inp['quantization'][0]:.6f}  zp={inp['quantization'][1]}")
print(f"output: shape={out['shape']}  dtype={out['dtype']}  "
      f"scale={out['quantization'][0]:.6f}  zp={out['quantization'][1]}")

# Expected:
#   input:  shape=[1, 49, 10, 1]  dtype=<class 'numpy.int8'>
#   output: shape=[1, 12]         dtype=<class 'numpy.int8'>
