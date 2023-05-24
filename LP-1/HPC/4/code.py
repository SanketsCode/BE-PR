import tensorflow as tf
model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
tf.keras.layers.MaxPooling2D((2, 2)),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(10, activation='softmax')
])
#  New Tab 
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#new tab

!pip install mpi4py
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def train(model, x_train, y_train, rank, size):
# Split the data across the nodes 
  n = len(x_train)
  chunk_size = n / size 
  start = rank * chunk_size 
  print(start)
  end = (rank + 1) * chunk_size
  print(end)
  if rank == size - 1:
    end = n
  x_train_chunk = x_train[int(start):int(end)]
  y_train_chunk = y_train[int(start):int(end)]
  # Compile the model
  model.compile(optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])
  #Train the model
  model.fit(x_train_chunk, y_train_chunk, epochs=1, batch_size=32)
  # Compute the accuracy on the training data
  train_loss, train_acc = model.evaluate(x_train_chunk, y_train_chunk, verbose=2)
  # Reduce the accuracy across all nodes
  train_acc = comm.allreduce(train_acc, op=MPI.SUM)
  return train_acc / size
  
epochs = 5
for epoch in range(epochs):
  # Train the model
  train_acc = train(model, x_train, y_train, rank, size)
  # Compute the accuracy on the test data
  test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
  # Reduce the accuracy across all nodes
  test_acc = comm.allreduce(test_acc, op=MPI.SUM)
  # Print the results 
  if rank == 0:
    print(f"Epoch {epoch + 1}: Train accuracy = {train_acc:.4f}, Test accuracy = {test_acc / size:.4f}")