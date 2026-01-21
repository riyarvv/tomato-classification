import tflite_runtime.interpreter as tflite

# Path to your model
MODEL_PATH = "tomato_model_pi.itflite"

# Load the model and get details
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get the list of all operations in the model
# Note: Some versions of tflite-runtime may not expose this easily, 
# but we can try to get the details of the input/output layers
print(f"Input details: {interpreter.get_input_details()}")
print(f"Output details: {interpreter.get_output_details()}")
