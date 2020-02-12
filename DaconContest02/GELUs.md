## Gaussian Error Linear Units (GELUs)

Authors propose the Gaussian Error Linear Unit (GELU), a high-performing neural network activation function.

$GELU(x)=xP(x)(X \leq x) = x\Phi(x)=0.5x(1 + tanh[\sqrt{2/π}(x + 0.044715x^3 )])$



```python
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
import numpy as np
import tensorflow as tf

class Gelu(Activation):
    def __init__(self, activation, **kwargs):
        super(Gelu, self).__init__(activation, **kwargs)
        self.__name__='gelu'
        
def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * np.pow(x, 3))))

get_custom_objects().update({'gelu': Gelu(gelu)})
```



<출처>

GAUSSIAN ERROR LINEAR UNITS (GELUS) - Dan Hendrycks and Kevin Gimpel (2018)