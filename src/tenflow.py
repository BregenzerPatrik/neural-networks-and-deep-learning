import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import sys
import numpy as np

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize the pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
# Define the number of steps and the step size for changing the image
num_steps = 100
step_size = 0.01
line = 0
collumn = 0
nr_of_hidden_layers = 20
nr_of_neurons_per_hidden_layer = 218
# Define the model architecture
# keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
# keras.layers.MaxPooling2D((2, 2)),
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),
    keras.layers.Dense(nr_of_neurons_per_hidden_layer, activation='relu'),
    keras.layers.Dense(nr_of_neurons_per_hidden_layer, activation='relu'),
    keras.layers.Dense(nr_of_neurons_per_hidden_layer, activation='relu'),
    keras.layers.Dense(nr_of_neurons_per_hidden_layer, activation='relu'),
    keras.layers.Dense(nr_of_neurons_per_hidden_layer, activation='relu'),
    keras.layers.Dense(nr_of_neurons_per_hidden_layer, activation='relu'),
    keras.layers.Dense(nr_of_neurons_per_hidden_layer, activation='relu'),
    keras.layers.Dense(nr_of_neurons_per_hidden_layer, activation='relu'),
    keras.layers.Dense(nr_of_neurons_per_hidden_layer, activation='relu'),
    keras.layers.Dense(nr_of_neurons_per_hidden_layer, activation='relu'),
    keras.layers.Dense(nr_of_neurons_per_hidden_layer, activation='relu'),
    keras.layers.Dense(nr_of_neurons_per_hidden_layer, activation='relu'),
    keras.layers.Dense(nr_of_neurons_per_hidden_layer, activation='relu'),
    keras.layers.Dense(nr_of_neurons_per_hidden_layer, activation='relu'),
    keras.layers.Dense(nr_of_neurons_per_hidden_layer, activation='relu'),
    keras.layers.Dense(nr_of_neurons_per_hidden_layer, activation='relu'),
    keras.layers.Dense(nr_of_neurons_per_hidden_layer, activation='relu'),
    keras.layers.Dense(nr_of_neurons_per_hidden_layer, activation='relu'),
    keras.layers.Dense(nr_of_neurons_per_hidden_layer, activation='relu'),
    keras.layers.Dense(nr_of_neurons_per_hidden_layer, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('Test accuracy:', test_acc)
model.save('models/MINST')
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
grad_norms = []
for i in range(num_steps):
    # get batch of images and labels
    x_batch, y_batch = x_test, y_test
    y_batch = tf.one_hot(y_batch, depth=10)
    # compute gradient and apply updates to weights
    with tf.GradientTape() as tape:
        logits = model(x_batch)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_batch, logits=logits))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    grads_flat = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
    grad_norm = tf.norm(grads_flat)
    grad_norms.append(grad_norm)
    # calculate gradient norm for each sample in batch
    # batch_grad_norms = []
    # for j in range(x_batch.shape[0]):
    # sample_grad_norm = tf.norm([grad[j] for grad in grads])
    # batch_grad_norms.append(sample_grad_norm)
# grad_norms.append(batch_grad_norms)

# plot gradient norm distribution every n iterations

plt.plot(grad_norms)
plt.title("Gradient over Time")
plt.xlabel("Epoche")
plt.ylabel("gradient")
plt.show()

sys.exit(0)
# for epoch in range(5):
#    print(f"Epoch {epoch + 1}")
#    with tf.GradientTape() as tape:
#        # Forward pass
#        y_pred = model(x_train, training=True)
#        # Compute the loss
#        loss = keras.losses.sparse_categorical_crossentropy(y_train, y_pred)
#    # Compute the gradients with respect to the trainable variables
#    grads = tape.gradient(loss, model.trainable_variables)
#    print(f"Gradients: {grads}")
#    # Update the trainable variables
#    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('Test accuracy:', test_acc)

print("gradient für einzelnes Bild:")
# Select a random image from the test set
image = x_test[0]
label = y_test[0]
# Display the image using pyplot.imshow
plt.imshow(image, cmap='gray')
# plt.show()
print("true label", label)
plt.clf()
# Compute the gradient for the selected image
with tf.GradientTape() as tape:
    # Forward pass
    y_pred = model(tf.expand_dims(image, 0), training=False)
    print("predicted label", y_pred)
    # Compute the loss for the selected image and label
    loss = keras.losses.sparse_categorical_crossentropy(label, y_pred)
# Compute the gradient of the loss with respect to the model's trainable variables
grads = tape.gradient(loss, model.trainable_variables)

# Pack the gradients into a single tensor
grads_flat = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)

# Compute the norm of the flattened gradient tensor
grad_norm = tf.norm(grads_flat)

print(f"Gradient norm: {grad_norm}")

# for line in range(28):
# for collumn in range(28):
# Initialize the image to the original image
image_perturbed = image.copy()
# Create an empty list to store the gradient values
gradsArray = []
# Loop through the steps and compute the gradients and update the image
for i in range(num_steps):
    # Compute the gradient for the selected image
    with tf.GradientTape() as tape:
        # Forward pass
        y_pred = model(tf.expand_dims(image_perturbed, 0), training=False)
        # Compute the loss for the selected image and label
        loss = keras.losses.sparse_categorical_crossentropy(label, y_pred)
        # Compute the gradient of the loss with respect to the model's trainable variables
    grads = tape.gradient(loss, model.trainable_variables)

    # Pack the gradients into a single tensor
    grads_flat = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)

    # Compute the norm of the flattened gradient tensor
    grad_norm = tf.norm(grads_flat)
    gradsArray.append(grad_norm)
    # image_perturbed[line,collumn] += step_size
    image_perturbed += step_size
    # for pixel in image_perturbed:
    #    pixel += step_size
    # Display the final perturbed image
plt.imshow(image_perturbed, cmap='gray')
plt.savefig(
    f'..\images8\perturbated_image_line_all_collumn_all_hidden_layers{nr_of_hidden_layers}_nr_of_neurons{nr_of_neurons_per_hidden_layer}_sigmoid_acc_{test_acc}.png')
plt.clf()
# Plot the gradient values over time
plt.plot(range(num_steps), gradsArray)
plt.xlabel('Step')
plt.ylabel('Gradient')
plt.savefig(
    f'..\images8\gradient_line_all_collumn_all_hidden_layers{nr_of_hidden_layers}_nr_of_neurons{nr_of_neurons_per_hidden_layer}_sigmoid_acc_{test_acc}.png')
plt.clf()
