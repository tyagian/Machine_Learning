import tensorflow.keras as keras

# Our model is linear with 1 input and 1 output: hard to be simpler than that.
input_dim = 1
output_dim = 1

# The Keras class that implements "outputs as a linear combination of inputs" 
# is called "keras.layers.Dense".
# It is a "layer" because many types and instances of layers can be grouped together 
# to build more complicated models.
#
# The layers (here only one) must be packaged into a "Model".
# The simplest model is "Sequential", one where you can list your layers sequentially without needing
# to specify additional details in between
model = keras.models.Sequential([
  keras.layers.Dense(units=output_dim, input_dim=input_dim)
])

# After creation, a Keras model must always be compiled

learning_rate = 0.01 # <= this is the main topic of the next few cells

sgd = keras.optimizers.SGD(lr=learning_rate)
model.compile(optimizer=sgd, # Stochastic Gradient Descent
              loss='mse') # Mean Square Error

# Quick reminder from the course:
# At this stage the created model is full of randomly initialized weights (aka, parameters).
# We train a model by fitting it to a data set (called the training set).
# Each record in the training set tells us:
#     - given this input x
#     - you should predict this output y
# The Gradient Descent algorithm, by looking at the difference between actual and predicted values
# slowly adjusts the weights of the model such that the next prediction becomes slightly better.



x = [xy[0] for xy in area2price] # input
y = [xy[1] for xy in area2price] # correct output

# While not strictly necessary, it is useful to know that Keras likes float32 Numpy arrays
x = np.array(x, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Fit the model parameters to the training data
#     which here means: little by little change the values of w and b such that the line
#            h(x) = w*x + b
#     is very close to the points (x, y) of the training data set 
model.fit(x, y, 
          epochs=10, # One "epoch" trains once over the whole training data set
          verbose=2) # verbose=2 is perfect for notebook environment (0 = nothing, 1 = too much)

print('Model training done')

# IF YOU GET INFINITIES READ THE NEXT CELL

