#!/usr/bin/env python
# -*- coding: utf-8 -*-
#                     GNU GENERAL PUBLIC LICENSE
#                        Version 3, 29 June 2007
#
#  Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
#  Everyone is permitted to copy and distribute verbatim copies
#  of this license document, but changing it is not allowed.
#

from gnuradio import gr

import numpy as np
import onnxruntime
import onnxruntime.backend as backend

import time

class dnn_onnx_sync(gr.sync_block):
    """
    docstring for block dnn_onnx_sync
    """
    def __init__(self, onnx_model_file, onnx_batch_size, onnx_runtime_device):

        self.batch_size = onnx_batch_size

        self.session = onnxruntime.InferenceSession(onnx_model_file)
        # Device selecction is still not working on python and depends on the package installed
        self.backend = backend.prepare(self.session, device=onnx_runtime_device)     

        print("Model inputs:")
        for sess_input in  self.session.get_inputs():
            print("  -", sess_input.name, sess_input.shape)

        print("Model outputs:")
        for sess_output in  self.session.get_outputs():
            print("  -", sess_output.name, sess_output.shape)

        # Get port shapes without batch "dimension"
        self.model_inputs_shapes  =  [model_input.shape[1:] for model_input in self.session.get_inputs()]
        self.model_outputs_shapes =  [model_output.shape[1:] for model_output in self.session.get_outputs()]

        gr.sync_block.__init__(self,
            name="dnn_onnx_sync",
            in_sig= [(np.float32, np.prod(model_input_shape)) for model_input_shape in self.model_inputs_shapes],
            out_sig=[(np.float32, np.prod(model_output_shape)) for model_output_shape in self.model_outputs_shapes])

        self.set_output_multiple(self.batch_size)

      
    def work(self, input_items, output_items):
        input_data = [np.array(input_item).reshape(tuple([ self.batch_size]+ self.model_inputs_shapes[0])) for input_idx, input_item in enumerate(input_items)]
        output = self.backend.run(input_data)
        output_items[0][:] = output[0]
        return len(output_items[0])
    
    def start(self):
        self.start_time = time.time()   
        return True

    def stop(self):      
        time_spent = (time.time() - self.start_time)    
        #TODO: flex for multiple input (even item size) and output 
        total_samples =  self.nitems_read(0)
        print("Total samples: {:.0f} ({:.2f} samples/second)".format(total_samples, total_samples/time_spent))
        print()
        return True
