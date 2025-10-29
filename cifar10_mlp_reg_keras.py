# import modules
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import cifar100
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import regularizers as regularizer
import matplotlib.pyplot as plt

# ---------------------- Load CIFAR-100 ----------------------
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

# One-hot encode labels (100 classes)
y_train = to_categorical(y_train, 100)
y_test  = to_categorical(y_test, 100)

# ============================================================
# Model 1: Base model (no regularization)
# ============================================================
model_base = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(1024, activation='relu'),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(100, activation='softmax')
])

model_base.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🚀 Training Base Model...")
history_base = model_base.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(x_test, y_test),
    verbose=2
)

# ============================================================
# Model 2: L2 Regularized model
# ============================================================
model_l2 = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(1024, activation='relu', kernel_regularizer=regularizer.l2(1e-4)),
    Dense(512, activation='relu', kernel_regularizer=regularizer.l2(1e-4)),
    Dense(256, activation='relu', kernel_regularizer=regularizer.l2(1e-4)),
    Dense(128, activation='relu', kernel_regularizer=regularizer.l2(1e-4)),
    Dense(100, activation='softmax', kernel_regularizer=regularizer.l2(1e-4))
])

model_l2.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🚀 Training L2 Regularized Model...")
history_l2 = model_l2.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(x_test, y_test),
    verbose=2
)

# ============================================================
# Model 3: L4 model — deeper + L2 regularization
# ============================================================
model_l4 = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(1024, activation='relu', kernel_regularizer=regularizer.l2(1e-4)),
    Dense(768, activation='relu', kernel_regularizer=regularizer.l2(1e-4)),  # extra layer
    Dense(512, activation='relu', kernel_regularizer=regularizer.l2(1e-4)),
    Dense(256, activation='relu', kernel_regularizer=regularizer.l2(1e-4)),
    Dense(128, activation='relu', kernel_regularizer=regularizer.l2(1e-4)),
    Dense(100, activation='softmax', kernel_regularizer=regularizer.l2(1e-4))
])

model_l4.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🚀 Training L4 Deeper Model...")
history_l4 = model_l4.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(x_test, y_test),
    verbose=2
)

# ============================================================
# Evaluate all models
# ============================================================
loss_base, acc_base = model_base.evaluate(x_test, y_test, verbose=0)
loss_l2, acc_l2 = model_l2.evaluate(x_test, y_test, verbose=0)
loss_l4, acc_l4 = model_l4.evaluate(x_test, y_test, verbose=0)

print("\n📊 Accuracy comparison:")
print(f"Base Model Accuracy: {acc_base*100:.2f}%")
print(f"L2 Regularized Model Accuracy: {acc_l2*100:.2f}%")
print(f"L4 Deeper Model Accuracy: {acc_l4*100:.2f}%")

# ============================================================
# 📈 Plot training and validation accuracy
# ============================================================
plt.figure(figsize=(10, 6))
plt.plot(history_base.history['accuracy'], label='Base Train Acc', linestyle='-')
plt.plot(history_base.history['val_accuracy'], label='Base Val Acc', linestyle='--')
plt.plot(history_l2.history['accuracy'], label='L2 Train Acc', linestyle='-')
plt.plot(history_l2.history['val_accuracy'], label='L2 Val Acc', linestyle='--')
plt.plot(history_l4.history['accuracy'], label='L4 Train Acc', linestyle='-')
plt.plot(history_l4.history['val_accuracy'], label='L4 Val Acc', linestyle='--')

plt.title('Training vs Validation Accuracy (Base vs L2 vs L4)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
