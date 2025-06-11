from neuralNet import NeuralNet
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

# ======= Setup neural net (simple example) =======
# sampleNN = NeuralNet([2, 3, 1])
# inputLayer = np.array([[0.9], [0.1]])
# expectedOutput = np.array([[1]])

# ======= More complex example (try to break it) =======
# sampleNN = NeuralNet([5, 10, 20, 10, 1])
# inputLayer = np.random.rand(5, 1)
# expectedOutput = np.array([[0.75]])

# ======= XOR example =======
sampleNN = NeuralNet([2, 4, 1], optimizer= "stochastic")
xor_data = [
    (np.array([[0], [0]]), np.array([[0]])),
    (np.array([[0], [1]]), np.array([[1]])),
    (np.array([[1], [0]]), np.array([[1]])),
    (np.array([[1], [1]]), np.array([[0]])),
]
xor_index = 0  # to cycle through the XOR dataset

# ======= Tracking data =======
losses = []
epoch = 0

# ======= Setup interactive plot =======
plt.ion()
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

line_loss, = ax.plot([], [], label="MSE Loss", color='blue')
current_point = ax.scatter([], [], color='red', s=50, label="Current Loss")
ax.set_xlim(0, 10)
ax.set_ylim(0, 1)
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.set_title("Interactive Backpropagation Visualizer")
ax.legend()
ax.grid(True)

# ======= Controls =======
ax_button = plt.axes([0.7, 0.05, 0.1, 0.075])
btn_step = Button(ax_button, 'Step')

ax_slider = plt.axes([0.15, 0.1, 0.4, 0.03])
slider_lr = Slider(ax_slider, 'Learning Rate', 0.001, 1.0, valinit=0.1)

# ======= Step function to advance one epoch =======
def step(event):
    global epoch, xor_index

    # Cycle through XOR data
    inputLayer, expectedOutput = xor_data[xor_index]
    xor_index = (xor_index + 1) % len(xor_data)

    # Train one step
    output, _ = sampleNN.feedForward(inputLayer)
    loss = sampleNN.mse(expectedOutput, output)
    sampleNN.updateWeightAndBias(inputLayer, expectedOutput, learningRate=slider_lr.val)
    losses.append(loss)

    # Update plot
    line_loss.set_data(range(len(losses)), losses)
    current_point.set_offsets([[epoch, loss]])
    ax.set_xlim(0, max(10, epoch + 1))
    ax.set_ylim(0, max(losses) * 1.1 if losses else 1)
    epoch += 1
    fig.canvas.draw_idle()

# ======= Connect button =======
btn_step.on_clicked(step)

plt.show(block=True)