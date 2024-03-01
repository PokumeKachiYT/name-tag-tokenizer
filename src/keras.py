import keras.models as models
import keras.layers as layers
import keras.losses as losses
import keras.utils as utils
import matplotlib.pyplot as plt

train_data = utils.image_dataset_from_directory(
    '.',
    seed=123,
    image_size=(320,180),
    batch_size=2
)

model = models.Sequential()

model.add(layers.Rescaling(1./255))

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(320, 180, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(12))

model.compile(optimizer='adam',
              loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print(train_data)

#history = model.fit(train_data,epochs=10)

