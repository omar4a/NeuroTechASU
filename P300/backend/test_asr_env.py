import sys
sys.path.append(r'C:\Users\omari\AppData\Roaming\Python\Python310\site-packages')
import asrpy
import pickle
import numpy as np

asr = pickle.load(open(r'c:\Omar\Education\NeuroTech_ASU\P300\backend\training_data\asr_state.pkl', 'rb'))
out = asr.transform(np.random.randn(8, 100))
print(type(out), out.shape)
