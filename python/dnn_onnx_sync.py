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
    This block use ONNX Python's backend in order to run inference on the selected model.
    The input/output sizes must match with the input/output model sizes.
    """

    # Using ONNX types found in https://github.com/microsoft/onnxruntime/blob/b7cc611563cfd1bafdff14e38fb50ec9c48c3d68/onnxruntime/python/tools/onnxruntime_test.py
    ONNX_TYPES = {
        'tensor(bool)':     np.bool,
        'tensor(float16)':  np.float16,
        'tensor(float)':    np.float32,   
        'tensor(double)':   np.float64,
        'tensor(int)':      np.int32,
        'tensor(int32)':    np.int32,
        'tensor(int8)':     np.int8,
        'tensor(uint8)':    np.uint8,
        'tensor(int16)':    np.int16,
        'tensor(uint16)':   np.uint16,
        'tensor(int64)':    np.int64,
        'tensor(uint64)':   np.uint64
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

        input_items_data = np.asarray(input_items)

        input_data = [input_items_data[input_idx,:self.batch_size].reshape(input_shape) for input_idx, input_shape in enumerate(self.model_inputs_shapes)]
        # input_data_norm = [data/np.linalg.norm(data, ord=1, axis=1, keepdims=True) for data in input_data]

        outputs = self.backend.run(input_data)

        for output_idx, output in enumerate(outputs):
            output_items[output_idx][:self.batch_size] = output

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
