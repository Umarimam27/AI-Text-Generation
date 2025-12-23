import h5py
import json
import tensorflow as tf
from tensorflow.keras.mixed_precision import Policy

OLD_MODEL = "next_word.h5"
NEW_MODEL = "next_word_fixed.h5"

# -------------------------------
# 1️⃣ Fix InputLayer batch_shape
# -------------------------------
with h5py.File(OLD_MODEL, "r+") as f:
    model_config = json.loads(f.attrs["model_config"])

    for layer in model_config["config"]["layers"]:
        if layer["class_name"] == "InputLayer":
            cfg = layer["config"]
            if "batch_shape" in cfg:
                cfg["input_shape"] = cfg["batch_shape"][1:]
                del cfg["batch_shape"]

    f.attrs["model_config"] = json.dumps(model_config)

print("✔ Removed batch_shape from model config")

# -------------------------------
# 2️⃣ Load model with custom dtype
# -------------------------------
custom_objects = {
    "DTypePolicy": Policy,
}

model = tf.keras.models.load_model(
    OLD_MODEL,
    compile=False,
    custom_objects=custom_objects
)

# -------------------------------
# 3️⃣ Save clean model
# -------------------------------
model.save(NEW_MODEL)
print("✅ Model fixed and saved as next_word_fixed.h5")
