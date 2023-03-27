import discriminator
import generator
import tensorflow as tf

# class DGMR(snt.Module):
#     def __init__(self, name=None):
#         super().__init__(name=name)
#         self.generator = generator.Generator(lead_time=90, time_delta=5)
#         self.discriminator = discriminator.Discriminator()

#     def generate(self, inputs, is_training=True):
#         return self.generator(inputs, is_training=is_training)

#     def discriminate(self, inputs, is_training=True):
#         return self.discriminator(inputs, is_training=is_training)


class DGMR(tf.keras.models.Model):
    def __init__(self, name=None):
        super(DGMR, self).__init__(name=name)
      
    def build(self, input_shape):
        self.generator = generator.Generator(lead_time=90, time_delta=5)
        self.discriminator = discriminator.Discriminator()
        
    def call(self, inputs):
        return self.discriminator(inputs, is_training=is_training)

