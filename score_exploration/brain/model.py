
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D

def build_model(X,y):
    input_tensor = Input(shape=X.shape[1:])

    # create the base pre-trained model
    base_model = tf.keras.applications.ResNet50(
                include_top=False,
                weights=None,
                input_tensor=input_tensor,
                input_shape=None,
                pooling=None
            )

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # let's add a fully-connected layer
    x = Dense(1024, activation='elu')(x)

    # and a logistic layer
    predictions = Dense(y.shape[-1], activation='linear')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)


    # model.summary()

    optimizer = tfk.optimizers.Adam()
    loss = tfk.losses.CategoricalCrossentropy(from_logits=True)
    metrics = [tfk.metrics.CategoricalAccuracy('accuracy', dtype=tf.float32)]
    model.compile(optimizer=optimizer, 
                loss=loss, 
                metrics=metrics)
    
    return model
