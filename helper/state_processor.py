
import tensorflow as tf

class StateProcessor():
    """
    Processes a raw Atari iamges. Resizes it and converts it to grayscale.
    # D. Britz Implementation
    """




    """
    NOTES 

    Lunar lander needs two things to work

    1. It has an observation space: That is, it sees the environment through sensors. 
    Each sensor sees in one direction.
    You get that input in an array, after you call the env.step() function
    see main.py Line# 211

    The input to the state is NOT AN IMAGE 
    IT IS AN ARRAY of Values

    
    2. It has action space: a list of actions that it can perform.

    therefore;
    for observation
    self.input_state = tf.placeholder ( shape=[8])

    
    
    
    """

    def __init__(self):
        # Build the Tensorflow graph
        with tf.variable_scope("state_processor"):
            # self.input_state = tf.placeholder(
            #     shape=[210, 160, 3], dtype=tf.uint8)
            # self.output = tf.image.rgb_to_grayscale(self.input_state)
            # self.output = tf.image.crop_to_bounding_box(
            #     self.output, 34, 0, 160, 160)
            # self.output = tf.image.resize_images(
            #     self.output, (84, 84),
            #     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # self.output = tf.squeeze(self.output)


            self.input_state = tf.placeholder(
                shape=[4], dtype=tf.float16)
            print("INPUT STATE ******************** ",self.input_state)
            self.output = self.input_state

            # self.output = tf.image.rgb_to_grayscale(self.input_state)
            # self.output = tf.image.crop_to_bounding_box(
            #     self.output, 34, 0, 160, 160)
            # self.output = tf.image.resize_images(
            #     self.output, (84, 84),
            #     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [210, 160, 3] Atari RGB State
        Returns:
            A processed [84, 84, 1] state representing grayscale values.
        """
        return sess.run(self.output, {self.input_state: state})