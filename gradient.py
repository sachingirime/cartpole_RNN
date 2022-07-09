import tensorflow as tf
@tf.custom_gradient
def grad_clip(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return tf.clip_by_value(dy, clip_value_min=0, clip_value_max=0.5)
    return y, custom_grad

class Clip(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        
    def call(self, x):
        return grad_clip(x)

@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)
    
    def custom_grad(dy):
        return -dy   
    
    return y, custom_grad

class GradReverse(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        
    def call(self, x):
        return grad_reverse(x)


def step(real_x, real_y):
    with tf.GradientTape() as tape:
        # Make prediction
        pred_y = model(real_x.reshape((-1, 28, 28, 1)))
        # Calculate loss
        model_loss = tf.keras.losses.categorical_crossentropy(real_y, pred_y)
    
    # Calculate gradients
    model_gradients = tape.gradient(model_loss, model.trainable_variables)
    
    # Update model
    optimizer.apply_gradients(zip(model_gradients, model.trainable_variables))
    