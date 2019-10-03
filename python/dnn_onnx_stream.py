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


class dnn_onnx_stream(gr.basic_block):
    """
    TODO docstring for block onnx_stream
    """
    def __init__(self, onnx_model_file, onnx_batch_size, onnx_runtime_device):
        
        gr.basic_block.__init__(self,
            name="dnn_onnx_stream",
            in_sig = [np.float32],
            out_sig= [np.float32])

        self.batch_size = onnx_batch_size

        # Device selecction is still not working on python and depends on the package installed
        self.session = onnxruntime.InferenceSession(onnx_model_file)
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

        #self.set_relative_rate(max(self.session_input_sizes)/max(self.session_output_sizes))
        #print(self.model_inputs_shapes, self.model_outputs_shapes)
 

    def forecast(self, noutput_items, ninput_items_required):
        # Calculate max oputput size (in number of items)
        max_output_size = max([np.prod(model_output_shape) for model_output_shape in self.model_outputs_shapes])
        # Calcualte input size in items given the number of output items
        ninput_items_required = [noutput_items * np.prod(model_input_shape) // max_output_size for model_input_shape in self.model_inputs_shapes]


    def general_work(self, input_items, output_items):

        assert len(input_items) == len(self.model_inputs_shapes)
        assert len(output_items)== len(output_items)

        # Data available per input
        input_size_available = [len(input_item) for input_item in input_items]    
        # Number of batches available per input
        input_batches_available = [input_size//np.prod(model_input_shape) for input_size, model_input_shape in zip(input_size_available, self.model_inputs_shapes) ]
        # Final number of batches to use        
        current_batch_size = min(self.batch_size, min(input_batches_available))

        # Final input size
        current_input_size = [current_batch_size * np.prod(model_input_shape) for model_input_shape in self.model_inputs_shapes]
        # Final output size
        current_output_size = [current_batch_size * np.prod(model_output_shape) for model_output_shape in self.model_outputs_shapes]

        #print(input_size_available, input_batches_available, current_batch_size, current_input_size, current_output_size)
        #time.sleep(0.2)

        if current_batch_size > 0:
            #TODO: flex for multiple input and output
            input_data = np.asarray(input_items[0][:current_input_size[0]], dtype=np.single)
            input_data.shape = tuple([current_batch_size]+ self.model_inputs_shapes[0])
            output = self.backend.run(input_data)
            output_items[0][0:current_output_size[0]] = np.ravel(output)
        else:
            # Not enough data for one batch            
            pass
        
        # Indicate scheduler data consumed per input
        for input_idx, consumed_data in enumerate(current_input_size):
            self.consume(input_idx, consumed_data)   

        return current_output_size[0]
    

    def start(self):
        self.start_time = time.time()   
        return True


    def stop(self):      
        time_spent = (time.time() - self.start_time)    
        #TODO: flex for multiple input (even item size) and output 
        total_items_read = self.nitems_read(0)
        total_item_size  = total_items_read * 4
        total_samples = total_items_read/np.prod(self.model_inputs_shapes[0])
        print("Total samples: {:.0f} ({:.2f} samples/second)".format(total_samples, total_samples/time_spent))
        print("Total items read:    {:.0f} ({:.2f} MB/s)".format(total_items_read, total_item_size/(time_spent * 1024 * 1024)))
        print("Total items written: ",self.nitems_written(0))
        return True
