#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 "HEIG-VD, REDS Institute"
# Author: Oscar RODRIGUEZ <oscar.rodriguezzalona@heig-vd.ch>.
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
#

from gnuradio import gr

import numpy as np
import onnxruntime
import onnxruntime.backend as backend

import time

class dnn_onnx_sync(gr.sync_block):
    """
    """

    ONNX_TYPES = {
        'tensor(float)': np.float32,
        'tensor(int)': np.int32
    }

    def __init__(self, onnx_model_file, onnx_batch_size, onnx_runtime_device):

        self.batch_size = onnx_batch_size

        self.session = onnxruntime.InferenceSession(onnx_model_file)
        # Device selecction is still not working on python and depends on the package installed (onnxruntime[gpu] 0.5.0)
        self.backend = backend.prepare(self.session, device=onnx_runtime_device)     

        print("Model inputs:")
        for sess_input in  self.session.get_inputs():
            print("  -", sess_input.name, sess_input.shape)

        print("Model outputs:")
        for sess_output in  self.session.get_outputs():
            print("  -", sess_output.name, sess_output.shape)

        gr.sync_block.__init__(self,
            name="dnn_onnx_sync",
            in_sig= [(self.ONNX_TYPES[model_input.type], np.prod(model_input.shape[1:]))  for model_input in self.session.get_inputs()],
            out_sig=[(self.ONNX_TYPES[model_output.type], np.prod(model_output.shape[1:])) for model_output in self.session.get_outputs()])

        self.set_output_multiple(self.batch_size)
        
        # Get model inputs shape with batch "dimension"
        self.model_inputs_shapes  =  [tuple([self.batch_size] + model_input.shape[1:]) for model_input in self.session.get_inputs()]        
        

    def work(self, input_items, output_items):
        # Input items have to be same as model inputs
        assert len(input_items) == len(self.model_inputs_shapes)

        input_data = [np.array(input_item[:np.prod(self.model_inputs_shapes[input_idx])]).reshape(self.model_inputs_shapes[input_idx]) for input_idx, input_item in enumerate(input_items)]
        outputs = self.backend.run(input_data)
        # TODO: Use all outputs of the model
        output_items[0][:] = outputs[0]
        return self.batch_size
    

    def start(self):
        self.start_time = time.time()   
        return True


    def stop(self):
        time_spent = (time.time() - self.start_time)    
        print("Total time: {:.0f} seconds".format(time_spent))
        for input_idx, model_input_shape in enumerate(self.model_inputs_shapes):
            input_samples =  self.nitems_read(input_idx)
            input_size_mb = (input_samples * np.prod(model_input_shape[1:]) * 4) / (1024 * 1024)
            print("- Input {:d} samples consumed: {:.0f} ({:.2f} MB/s)".format(input_idx, input_samples, input_size_mb/time_spent))        
        return True
