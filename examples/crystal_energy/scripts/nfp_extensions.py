import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from nfp.preprocessing import zero, SmilesPreprocessor
from nfp.preprocessing.features import Tokenizer

class CifPreprocessor(SmilesPreprocessor):
    def __init__(self, radius=None, num_neighbors=12):
        self.site_tokenizer = Tokenizer()
        self.radius = radius
        self.num_neighbors = num_neighbors
        self.max_sites = 0
        self.max_distance = 0
        
    @property
    def site_classes(self):
        return self.site_tokenizer.num_classes + 1
    
    @staticmethod
    def site_features(site):
        species = site.as_dict()['species']
        assert len(species) == 1
        return species[0]['element']
    
    def construct_feature_matrices(self, crystal, train=False):
        
        self.site_tokenizer.train = train        
        site_features = np.array([self.site_tokenizer(self.site_features(site))
                                  for site in crystal.sites])
        
        # Record for array sizing later
        if train & (crystal.num_sites > self.max_sites):
            self.max_sites = crystal.num_sites
        
        connectivity = []
        distances = []
        
        if self.radius is None:
            # Get the expected number of sites / volume, then find a radius 
            # expected to yield 2x the desired number of neighbors
            desired_vol = (crystal.volume / crystal.num_sites) * self.num_neighbors
            radius = 2 * (desired_vol / (4 * np.pi / 3))**(1/3)
        else:
            radius = self.radius
        
        # Iterate over each site's neighbors
        for i, neighbors in enumerate(crystal.get_all_neighbors(radius)):
            assert len(neighbors) >= self.num_neighbors, \
                f"Only {len(neighbors)} neighbors for site {i}"
                        
            sorted_neighbors = sorted(neighbors, key=lambda x: x[1])[:self.num_neighbors]
            for _, distance, j, _ in sorted_neighbors:
                connectivity += [(i, j)]
                distances += [distance]
                
                if train & (distance > self.max_distance):
                    self.max_distance = distance
                
        connectivity = np.array(connectivity, dtype='int')
        distances = np.array(distances, dtype='float')
        
        return {'site_features': site_features,
                'distance': distances,
                'connectivity': connectivity}
    
    tfrecord_features = {
        'site_features': tf.io.FixedLenFeature([], dtype=tf.string),
        'distance': tf.io.FixedLenFeature([], dtype=tf.string),
        'connectivity': tf.io.FixedLenFeature([], dtype=tf.string)
    }    
    
    output_types = {'site_features': tf.int64,
                    'distance': tf.float64,
                    'connectivity': tf.int64}

    output_shapes = {'site_features': tf.TensorShape([None]),
                     'distance': tf.TensorShape([None]),
                     'connectivity': tf.TensorShape([None, 2])}
    
    @staticmethod
    def padded_shapes(max_sites=-1, max_bonds=-1):
        
        return {
            'site_features': [max_sites],
            'distance': [max_bonds],
            'connectivity': [max_bonds, 2]
        }
    
    padding_values = {
        'site_features': zero,
        'distance': tf.constant(np.nan, dtype=tf.float64),
        'connectivity': zero}
    
    
class RBFExpansion(layers.Layer):
    def __init__(self, dimension=128, init_gap=10, init_max_distance=7, trainable=False):
        super(RBFExpansion, self).__init__()
        self.init_gap = init_gap
        self.init_max_distance = init_max_distance
        self.dimension = dimension
        self.trainable = trainable
    
    def build(self, input_shape):
        self.centers = tf.Variable(
            name='centers',
            initial_value=tf.range(0, self.init_max_distance,
                                   delta=self.init_max_distance/self.dimension),
            trainable=self.trainable,
            dtype=tf.float32)
        
        self.gap = tf.Variable(
            name='gap',
            initial_value=tf.constant(self.init_gap, dtype=tf.float32),
            trainable=self.trainable,
            dtype=tf.float32)
        
    def call(self, inputs):
        distances = tf.where(
            tf.math.is_nan(inputs), tf.zeros_like(inputs, dtype=inputs.dtype), inputs)
        offset = tf.expand_dims(distances, -1) - tf.cast(self.centers, inputs.dtype)
        logits = -self.gap * (offset)**2
        return tf.exp(logits)
    
    def compute_mask(self, inputs, mask=None):
        return tf.logical_not(tf.math.is_nan(inputs))
    
    def get_config(self):
        return {'init_gap': self.init_gap,
                'init_max_distance': self.init_max_distance,
                'dimension': self.dimension,
                'trainable': self.trainable}
