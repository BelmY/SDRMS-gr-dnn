#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 "HEIG-VD, REDS Institute"
# Author: Kevin JOLY <kevin.joly@heig-vd.ch>.
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

import numpy
from gnuradio import gr
import onnx
import onnxruntime.backend as backend
import pmt
import cProfile
import time
import pstats
import io

class dnn_onnx_python(gr.basic_block):
    """
    This block use ONNX Python's backend in order to make the inference of the
    selected model. The input/output sizes must match with the input/output
    model sizes. Please specify a .onnx file format for the model.
    """
    def __init__(self, onnxModelFilePath, enableProfiling, profilingPrintInterval):

        self.onnxModelFilePath = onnxModelFilePath
        self.enableProfiling = enableProfiling
        self.profilingPrintInterval = profilingPrintInterval

        gr.basic_block.__init__(self,
            name="dnn_onnx_python",
            in_sig=None,
            out_sig=None)

        self.message_port_register_out(pmt.intern('Output'))

        self.message_port_register_in(pmt.intern('Input'))
        self.set_msg_handler(pmt.intern('Input'), self.handle_msg)

        self.model = onnx.load(self.onnxModelFilePath)
        self.rep = backend.prepare(self.model, device="CPU")
        self.counter = 0

        if self.enableProfiling:
            self.pr = cProfile.Profile(subcalls=False, builtins=False)
            self.lastPrintTime = time.time()

    def handle_msg(self, msg):

        if self.enableProfiling:
            self.pr.enable()

        if pmt.is_f32vector(pmt.cdr(msg)):
            netInput = numpy.array(pmt.f32vector_elements(pmt.cdr(msg)), dtype=numpy.float32)
        else:
            raise ValueError('Unhandled type')

        netInput.shape = tuple((d.dim_value) for d in self.model.graph.input[0].type.tensor_type.shape.dim)
        netOutput = self.rep.run(netInput)

        outputPmt = pmt.cons(pmt.make_dict(), pmt.to_pmt(numpy.array(netOutput, dtype=numpy.float32)))
        self.message_port_pub(pmt.intern('Output'), outputPmt)

        if self.enableProfiling:
            self.pr.disable()
            if time.time() > self.lastPrintTime + self.profilingPrintInterval:
                s = io.StringIO()
                p = pstats.Stats(self.pr, stream=s)
                p.sort_stats('cumulative')
                p.print_stats()
                print(s.getvalue())
                self.lastPrintTime = time.time()
