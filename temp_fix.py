import sys
import os

# Directly add the ttt-lm-kernels to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "ttt-lm-kernels"))

# Continue with the imports
import ttt.modeling_ttt
import ttt.configuration_ttt
from ttt.modeling_ttt import TttForCausalLM
from ttt.configuration_ttt import TttConfig

# Now patch the transformers library to use our modules
import transformers
transformers.models.ttt = sys.modules[__name__]
transformers.models.ttt.modeling_ttt = ttt.modeling_ttt
transformers.models.ttt.configuration_ttt = ttt.configuration_ttt
