from matplotlib import pyplot as plt

def plot_training_history(history):
  """
  Plots the training and validation accuracy and loss.

  Parameters:
   - history: A Keras History object. Contains the logs from the training process.

   Returns:
    - None. Displays the matplotlib plots for training/validation accuracy and loss.
    """
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(len(acc))

  plt.figure(figsize = (20, 5))

  # Plot training and validation accuracy
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label = 'Training Accuracy')
  plt.plot(epochs_range, val_acc, label = 'Validation Accuracy')
  plt.legend(loc = 'lower right')
  plt.title('Training and Validation Accuracy')

  # Plot training and Validation Loss
  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label = 'Training Loss')
  plt.plot(epochs_range, val_loss, label = 'Validation Loss')
  plt.legend(loc = 'upper right')
  plt.title('Training and Validation Loss')

  plt.show()




