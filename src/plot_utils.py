from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_training(acc, val_acc, loss, val_loss):
    plt.subplots(figsize=(12, 3))

    plt.subplot(1, 2, 1)
    plt.plot(
        acc,
        label='Training Accuracy'
    )
    plt.plot(
        val_acc,
        label='Validation Accuracy'
    )
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(
        loss,
        label='Training Loss',
        color='teal'
    )
    plt.plot(
        val_loss,
        label='Validation Loss',
        color='brown'
    )
    plt.xlabel('Epoch')
    plt.legend()

    plt.show()


def show_confusion_matrix(y_true, y_pred, classes):
    matrix = confusion_matrix(y_true, y_pred)
    matrix_display = ConfusionMatrixDisplay(matrix, display_labels=classes)
    matrix_display.plot(cmap='Blues', xticks_rotation='45')
